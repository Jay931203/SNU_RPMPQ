"""
Lemma 1 Empirical Validation — Two-Panel Figure
  (a) SSM State Error Saturation  (contractive recursion bounds error)
  (b) Architecture Comparison     (encoder weight quantization robustness)

Run on Colab (GPU) after setup_colab.py / Cell 1 of MambaTransNet.ipynb

Output:
  results/plots/lemma1_combined.png
  ../figures/lemma1_combined.pdf
"""

import os, sys, types
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Path Setup ──────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

RESULTS_PLOT = os.path.join(PROJECT, "results", "plots")
FIGURES_DIR  = os.path.join(PROJECT, "..", "figures")
os.makedirs(RESULTS_PLOT, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Bypass models/__init__.py (avoids compressai/torch_geometric crash) ─
for _k in list(sys.modules.keys()):
    if _k == 'models' or _k.startswith('models.'):
        del sys.modules[_k]
for _k in ['ModularModels', 'MambaAE', 'VSS_module', 'csm_triton']:
    sys.modules.pop(_k, None)

_models_dir = os.path.join(PROJECT, 'models')
_stub = types.ModuleType('models')
_stub.__path__ = [_models_dir]
_stub.__package__ = 'models'
sys.modules['models'] = _stub

_orig_path = sys.path[:]
sys.path.insert(0, _models_dir)
try:
    import VSS_module
    sys.modules['models.VSS_module'] = VSS_module
    _stub.VSS_module = VSS_module
finally:
    sys.path[:] = _orig_path

import MambaAE
sys.modules['models.MambaAE'] = MambaAE
from ModularModels import ModularAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[OK] Device: {device}")

# ── Plot Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10,
})

# ════════════════════════════════════════════════════════════════════════
#  Data Loading  (identical to train_ae.py CsiDataset)
# ════════════════════════════════════════════════════════════════════════
class CsiDataset(Dataset):
    def __init__(self, path, key='HT', norm_params=None):
        data = sio.loadmat(path)[key].astype(np.float32)
        if data.ndim == 2:
            data = data.reshape(-1, 2, 32, 32)
        if norm_params is None:
            self.min_val = float(data.min())
            self.range_val = float(data.max() - data.min()) + 1e-9
        else:
            self.min_val, self.range_val = norm_params
        self.data = (data - self.min_val) / self.range_val
        self.norm_params = (self.min_val, self.range_val)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i])


# ════════════════════════════════════════════════════════════════════════
#  Quantization helpers  (identical to train_ae.py)
# ════════════════════════════════════════════════════════════════════════
def quantize_int_asym(w, bits):
    q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
    w_min, w_max = w.min(), w.max()
    if w_max == w_min:
        return w
    scale = (w_max - w_min) / (q_max - q_min)
    zp = torch.round(q_min - w_min / scale)
    w_q = torch.clamp(torch.round(w / scale + zp), q_min, q_max)
    return (w_q - zp) * scale


def quantize_feedback_torch(y, bits):
    if bits <= 0:
        return y
    min_val = y.min(dim=1, keepdim=True)[0]
    max_val = y.max(dim=1, keepdim=True)[0]
    r = max_val - min_val + 1e-9
    levels = 2**bits - 1
    y_norm = (y - min_val) / r
    y_q = torch.round(y_norm * levels) / levels
    return y_q * r + min_val


def calculate_nmse_db(orig, rec, params):
    min_val, range_val = params
    orig = (orig * range_val) + min_val
    rec  = (rec  * range_val) + min_val
    orig_c, rec_c = orig - 0.5, rec - 0.5
    pow_orig = torch.sum(orig_c**2, dim=[1, 2, 3])
    mse      = torch.sum((orig_c - rec_c)**2, dim=[1, 2, 3])
    valid = pow_orig > 1e-8
    if valid.sum() == 0:
        return -100.0
    nmse = torch.mean(mse[valid] / pow_orig[valid])
    return 10 * torch.log10(nmse).item() if nmse > 1e-10 else -100.0


def remap_transnet_state_dict(state, num_enc_layers=2, num_dec_layers=2):
    """Remap TransNet checkpoint to match current ModularModels structure.

    Two differences between checkpoint (old) and current ModularModels:
    1. Attention: separate q/k/v projections  ->  combined in_proj_weight
    2. Encoder/Decoder: shared `layer` (singular)  ->  cloned `layers.{i}` (plural)
    """
    # Step 1: Combine q/k/v projections and drop thop buffers
    step1 = {}
    qkv_groups = {}
    skip_suffixes = ('total_ops', 'total_params')

    for k, v in state.items():
        if k.endswith(skip_suffixes):
            continue
        for proj in ('q_proj_weight', 'k_proj_weight', 'v_proj_weight'):
            if k.endswith(proj):
                prefix = k[:-len(proj)]
                qkv_groups.setdefault(prefix, {})[proj[0]] = v
                break
        else:
            step1[k] = v

    for prefix, qkv in qkv_groups.items():
        if len(qkv) == 3:
            combined = torch.cat([qkv['q'], qkv['k'], qkv['v']], dim=0)
            step1[f"{prefix}in_proj_weight"] = combined

    # Step 2: Expand shared `layer.` to cloned `layers.{i}.`
    # Patterns: encoder.encoder.layer.X -> encoder.encoder.layers.{i}.X
    #           decoder.decoder.layer.X -> decoder.decoder.layers.{i}.X
    final = {}
    for k, v in step1.items():
        expanded = False
        if '.encoder.layer.' in k:
            base = k.replace('.encoder.layer.', '.encoder.layers.{i}.')
            for i in range(num_enc_layers):
                final[base.format(i=i)] = v.clone()
            expanded = True
        if '.decoder.layer.' in k:
            base = k.replace('.decoder.layer.', '.decoder.layers.{i}.')
            for i in range(num_dec_layers):
                final[base.format(i=i)] = v.clone()
            expanded = True
        if not expanded:
            final[k] = v

    return final


def apply_encoder_weight_quant(model, bits):
    """Encoder-only weight quantization (same as train_ae.py apply_weight_quantization)."""
    real = model.module if isinstance(model, nn.DataParallel) else model
    count = 0
    with torch.no_grad():
        for name, m in real.encoder.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                m.weight.data = quantize_int_asym(m.weight.data, bits)
                count += 1
    return count


def test_nmse(model, loader, norm_params, aq_bits=8):
    """End-to-end NMSE_dB with encoder-only quant already applied."""
    model.eval()
    real = model.module if isinstance(model, nn.DataParallel) else model
    nmse_sum, count = 0.0, 0
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            z = real.encoder(d)
            if aq_bits > 0:
                z = quantize_feedback_torch(z, aq_bits)
            x_hat = real.decoder(z)
            nmse_sum += calculate_nmse_db(d, x_hat, norm_params)
            count += 1
    return nmse_sum / count


# ════════════════════════════════════════════════════════════════════════
#  PART A — SSM State Error Saturation
# ════════════════════════════════════════════════════════════════════════
def quantize_sym(x, bits):
    if bits >= 32:
        return x.clone()
    abs_max = x.abs().max()
    if abs_max == 0:
        return x.clone()
    qmax = 2**(bits - 1) - 1
    scale = abs_max / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def run_part_a():
    """SSM state/output error vs token position under weight quantization."""
    print("\n" + "="*60)
    print("  PART A: SSM State Error Saturation")
    print("="*60)

    from einops import rearrange
    from models.VSS_module import CrossScan

    # Load model
    net = ModularAE(encoder_type='mamba', decoder_type='transnet',
                    encoded_dim=512, encoder_layers=2, decoder_layers=2).to(device)
    ckpt = torch.load('saved_models/mamba_transnet_L2_dim512_baseline/best.pth',
                       map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    net.load_state_dict(state, strict=False)
    net.eval()

    # Random input (contractivity is a weight property, not data-dependent)
    torch.manual_seed(42)
    x_test = torch.rand(10, 2, 32, 32, device=device)

    # Forward through encoder stem -> first SS2D block
    mamba_blk = net.encoder.layers[0]
    ss2d = mamba_blk.vss[1]

    with torch.no_grad():
        x_stem = net.encoder.stem(x_test)
        x_norm = mamba_blk.norm(x_stem)
        x_act  = mamba_blk.act(x_norm)
        cs = mamba_blk.chunk_size
        x_chunked = rearrange(x_act,
            'b c (h cs_h) (w cs_w) -> (b h w) c cs_h cs_w',
            cs_h=cs, cs_w=cs)
        x_bhwc = x_chunked.permute(0, 2, 3, 1).contiguous()
        xz = ss2d.in_proj(x_bhwc)
        d_inner = xz.shape[-1] // 2
        x_proj = xz[..., :d_inner]
        x_conv = x_proj.permute(0, 3, 1, 2).contiguous()
        x_conv = ss2d.act(ss2d.conv2d(x_conv))

    B_sz, D_dim, H, W = x_conv.shape
    L = H * W
    K = 4
    N_s = ss2d.A_logs.shape[1]
    R = ss2d.dt_projs_weight.shape[2]
    print(f"  SS2D: B={B_sz}, D={D_dim}, H={H}, W={W}, L={L}, K={K}")

    # Compute SSM tensors
    def compute_ssm_tensors(ss2d, x_conv, w_bits=32):
        B, D, H, W = x_conv.shape
        L = H * W
        xpw  = ss2d.x_proj_weight.data.float()
        dtpw = ss2d.dt_projs_weight.data.float()
        dtpb = ss2d.dt_projs_bias.data.float()
        alg  = ss2d.A_logs.data.float()
        ds   = ss2d.Ds.data.float()
        if w_bits < 32:
            xpw  = quantize_sym(xpw, w_bits)
            dtpw = quantize_sym(dtpw, w_bits)
            alg  = quantize_sym(alg, w_bits)
        xs = CrossScan.apply(x_conv).view(B, K, D, L).float()
        x_dbl = torch.einsum("bkdl, kcd -> bkcl", xs, xpw)
        dts_r, Bs, Cs = torch.split(x_dbl, [R, N_s, N_s], dim=2)
        dts = torch.einsum("bkrl, kdr -> bkdl", dts_r, dtpw)
        As = -torch.exp(alg)
        db = dtpb.view(-1)
        return (xs.reshape(B, K*D, L), dts.reshape(B, K*D, L),
                As, Bs.contiguous(), Cs.contiguous(), ds, db)

    # Manual selective scan
    def manual_ssm_dir(u, delta, A, B, C, D_skip, db, k, D):
        Bs_, _, L_ = u.shape
        N = A.shape[1]
        sl = slice(k*D, (k+1)*D)
        u_k, dt_k = u[:, sl], delta[:, sl]
        A_k, B_k, C_k, D_k, db_k = A[sl], B[:, k], C[:, k], D_skip[sl], db[sl]
        states = torch.zeros(Bs_, D, L_, N, device=u.device)
        y      = torch.zeros(Bs_, D, L_,    device=u.device)
        s      = torch.zeros(Bs_, D, N,     device=u.device)
        for t in range(L_):
            dt = F.softplus(dt_k[:, :, t] + db_k)
            dA = torch.exp(dt.unsqueeze(-1) * A_k)
            dBu = dt.unsqueeze(-1) * B_k[:, :, t].unsqueeze(1) * u_k[:, :, t].unsqueeze(-1)
            s = dA * s + dBu
            states[:, :, t, :] = s
            y[:, :, t] = (C_k[:, :, t].unsqueeze(1) * s).sum(-1) + D_k * u_k[:, :, t]
        return y, states

    # FP32 baseline
    print("  Running FP32 baseline SSM loop...")
    with torch.no_grad():
        xs_fp, dts_fp, As_fp, Bs_fp, Cs_fp, Ds_fp, db_fp = \
            compute_ssm_tensors(ss2d, x_conv, w_bits=32)
    fp32_states, fp32_y = {}, {}
    for k in range(K):
        y_k, s_k = manual_ssm_dir(xs_fp, dts_fp, As_fp, Bs_fp, Cs_fp, Ds_fp, db_fp, k, D_dim)
        fp32_states[k] = s_k
        fp32_y[k] = y_k

    # Quantized versions
    bit_configs_a = [16, 8, 4, 2]
    results_a = {}
    for bits in bit_configs_a:
        print(f"  W{bits}...", end="", flush=True)
        with torch.no_grad():
            xs_q, dts_q, As_q, Bs_q, Cs_q, Ds_q, db_q = \
                compute_ssm_tensors(ss2d, x_conv, w_bits=bits)
        state_errs, output_errs = [], []
        for k in range(K):
            y_q, s_q = manual_ssm_dir(xs_q, dts_q, As_q, Bs_q, Cs_q, Ds_q, db_q, k, D_dim)
            se = (fp32_states[k] - s_q).norm(dim=-1).mean(dim=(0, 1))
            oe = (fp32_y[k] - y_q).abs().mean(dim=(0, 1))
            state_errs.append(se)
            output_errs.append(oe)
        results_a[bits] = {
            'state_err':  torch.stack(state_errs).mean(0).cpu().numpy(),
            'output_err': torch.stack(output_errs).mean(0).cpu().numpy(),
        }
        sp = results_a[bits]['state_err'][-5:].mean()
        op = results_a[bits]['output_err'][-5:].mean()
        print(f"  state_plateau={sp:.4f}  out_plateau={op:.5f}")

    # Contractivity check
    with torch.no_grad():
        dt_vals = F.softplus(dts_fp[:, :D_dim, :] + db_fp[:D_dim][None, :, None])
        dA_vals = torch.exp(dt_vals.unsqueeze(-1) * As_fp[:D_dim][None, :, None, :])
        rho_max  = dA_vals.max().item()
        rho_mean = dA_vals.mean().item()
    print(f"  rho_max = {rho_max:.4f} (<1 OK), rho_mean = {rho_mean:.4f}")

    return results_a, L, rho_max, rho_mean


# ════════════════════════════════════════════════════════════════════════
#  PART B — Architecture Comparison (Encoder Weight Quantization)
# ════════════════════════════════════════════════════════════════════════
def run_part_b():
    """Compare CsiNet, TransNet, Mamba encoder quantization robustness."""
    print("\n" + "="*60)
    print("  PART B: Architecture Comparison under Encoder Weight Quantization")
    print("="*60)

    # Load data (same as train_ae.py)
    train_set = CsiDataset('data/DATA_Htrainout.mat')
    test_set  = CsiDataset('data/DATA_Htestout.mat', norm_params=train_set.norm_params)
    test_loader = DataLoader(test_set, batch_size=200, shuffle=False,
                             num_workers=2, pin_memory=(device.type == 'cuda'))
    norm_params = train_set.norm_params
    print(f"  Data: train={len(train_set)}, test={len(test_set)}")
    print(f"  Normalization: min={norm_params[0]:.4f}, range={norm_params[1]:.4f}")

    # Model configs — using EXACT same checkpoints as train_ae.py
    models_cfg = [
        {
            'name': 'CsiNet',
            'encoder': 'csinet', 'decoder': 'csinet',
            'checkpoint': 'saved_models/csinet_csinet_dim512/best.pth',
            'kwargs': dict(encoded_dim=512, encoder_layers=2, decoder_layers=2),
        },
        {
            'name': 'TransNet',
            'encoder': 'transnet', 'decoder': 'transnet',
            'checkpoint': 'saved_models/transnet_transnet_L2_dim512/best.pth',
            'kwargs': dict(encoded_dim=512, encoder_layers=2, decoder_layers=2),
        },
        {
            'name': 'MT-AE',
            'encoder': 'mamba', 'decoder': 'transnet',
            'checkpoint': 'saved_models/mamba_transnet_L2_dim512_baseline/best.pth',
            'kwargs': dict(encoded_dim=512, encoder_layers=2, decoder_layers=2),
        },
    ]

    bit_configs_b = [32, 16, 8, 4]   # FP32, INT16, INT8, INT4
    results_b = {}

    for cfg in models_cfg:
        name = cfg['name']
        print(f"\n  -- {name}: {cfg['encoder']}+{cfg['decoder']} --")

        # Build model (same as train_ae.py)
        net = ModularAE(
            encoder_type=cfg['encoder'],
            decoder_type=cfg['decoder'],
            **cfg['kwargs']
        ).to(device)

        # Load checkpoint (same as train_ae.py: strict=False)
        ckpt_path = cfg['checkpoint']
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

            # TransNet checkpoint has separate q/k/v projections — remap
            if cfg['encoder'] == 'transnet':
                state = remap_transnet_state_dict(state)

            missing, unexpected = net.load_state_dict(state, strict=False)
            # Filter out thop profiling keys from unexpected
            unexpected = [k for k in unexpected if not k.endswith(('total_ops', 'total_params'))]
            print(f"  Loaded: {ckpt_path}")
            print(f"  Keys -- missing: {len(missing)}, unexpected: {len(unexpected)}")
            if len(missing) > 0:
                print(f"  [WARN] Missing keys (first 5): {missing[:5]}")
            if len(unexpected) > 0:
                print(f"  [WARN] Unexpected keys (first 5): {unexpected[:5]}")
        else:
            print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
            continue

        net.eval()

        # Save FP32 weights for restoration between quantization levels
        fp32_state = {k: v.clone() for k, v in net.state_dict().items()}

        results_b[name] = {}
        for bits in bit_configs_b:
            # Restore FP32 weights
            net.load_state_dict(fp32_state)

            # Apply encoder-only weight quantization (identical to train_ae.py)
            if bits < 32:
                n_layers = apply_encoder_weight_quant(net, bits)
            else:
                n_layers = 0

            # Measure NMSE_dB (encoder quant + 8-bit feedback, same as paper)
            nmse = test_nmse(net, test_loader, norm_params, aq_bits=8)
            results_b[name][bits] = nmse

            label = 'FP32' if bits == 32 else f'INT{bits}'
            print(f"  {label}: NMSE = {nmse:.2f} dB" +
                  (f"  ({n_layers} layers quantized)" if bits < 32 else ""))

    return results_b


# ════════════════════════════════════════════════════════════════════════
#  COMBINED FIGURE
# ════════════════════════════════════════════════════════════════════════
def plot_combined(results_a, L, rho_max, rho_mean, results_b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- Panel (a): SSM State Error --
    colors_a = {16: '#2196F3', 8: '#FF9800', 4: '#E91E63', 2: '#9C27B0'}
    tokens = np.arange(L)

    for bits in [16, 8, 4, 2]:
        ax1.plot(tokens, results_a[bits]['state_err'], color=colors_a[bits],
                 linewidth=1.8, label=f'W{bits}', alpha=0.85)

    ax1.set_xlabel('Token position $t$')
    ax1.set_ylabel(r'$\Vert \mathbf{e}_t \Vert_2$')
    ax1.set_title('(a) SSM state error saturation')
    ax1.legend(fontsize=10)
    ax1.grid(True, ls='--', alpha=0.3)
    ax1.text(0.97, 0.05,
             f'$\\rho_{{\\max}}={rho_max:.4f}$\n$\\bar{{\\rho}}={rho_mean:.4f}$',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    # -- Panel (b): Architecture Comparison --
    colors_b = {'CsiNet': '#D84315', 'TransNet': '#1565C0', 'MT-AE': '#2E7D32'}
    markers_b = {'CsiNet': 's', 'TransNet': '^', 'MT-AE': 'o'}

    bit_labels = {32: 'FP32', 16: 'INT16', 8: 'INT8', 4: 'INT4'}
    bit_positions = [32, 16, 8, 4]

    for name in results_b:
        xs = [b for b in bit_positions if b in results_b[name]]
        ys = [results_b[name][b] for b in xs]
        ax2.plot(xs, ys, marker=markers_b[name], color=colors_b[name],
                 linewidth=2.2, markersize=9, label=name, alpha=0.9)

    ax2.set_xlabel('Weight precision (bits)')
    ax2.set_ylabel('NMSE (dB)')
    ax2.set_title('(b) Encoder quantization robustness')
    ax2.set_xticks(bit_positions)
    ax2.set_xticklabels([bit_labels[b] for b in bit_positions])
    ax2.legend(fontsize=10)
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.invert_xaxis()

    fig.suptitle('Lemma 1: Contractive SSM bounds quantization error',
                 fontsize=13, y=1.02, fontweight='bold')
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        p_plot = os.path.join(RESULTS_PLOT, f'lemma1_combined.{ext}')
        p_fig  = os.path.join(FIGURES_DIR,  f'lemma1_combined.{ext}')
        fig.savefig(p_plot, dpi=300, bbox_inches='tight')
        fig.savefig(p_fig,  dpi=300, bbox_inches='tight')
        print(f"Saved: {p_plot}")
        print(f"Saved: {p_fig}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    results_a, L, rho_max, rho_mean = run_part_a()
    results_b = run_part_b()
    plot_combined(results_a, L, rho_max, rho_mean, results_b)

    # Summary table
    print("\n" + "="*60)
    print("  Summary: NMSE (dB) under encoder-only weight quantization")
    print("  Scenario: Outdoor, CR=1/4, Feedback=8-bit")
    print("-"*60)
    header = f"{'Model':<12}" + "".join(f"{'FP32' if b==32 else f'INT{b}':>10}" for b in [32, 16, 8, 4])
    print(header)
    print("-"*60)
    for name in results_b:
        row = f"{name:<12}" + "".join(f"{results_b[name].get(b, float('nan')):>10.2f}" for b in [32, 16, 8, 4])
        print(row)
    print("="*60)
