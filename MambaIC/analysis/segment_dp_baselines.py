"""
Segment DP evaluation on baseline models (CsiNet, CRNet, CLNet).
Runs the SAME segment DP algorithm on different architectures.

For each baseline:
1. Load pretrained model + test data (CsiDataset with normalization)
2. Identify encoder FC layer, split into 32 chunks
3. Identify non-FC encoder blocks (Conv2d with dim>=2)
4. Collect segment-level perturbation omegas for ALL encoder layers
5. Run joint DP optimization (FC segments + non-FC layers) at fine saving levels
6. Evaluate NMSE and outage

v2: All encoder layers (FC chunks + non-FC conv/fc) are jointly optimized.
    Non-FC layers are included as length-1 blocks appended after FC chunks.
    C_steps increased to 5000 for finer budget resolution.
    Savings step reduced to 0.1% for smoother Pareto front.

Usage: !python analysis/segment_dp_baselines.py
"""
import os, sys, re, types, logging, argparse, math
import numpy as np
import pandas as pd
from collections import namedtuple, OrderedDict
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "baselines")
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_ae import quantize_int_asym, CsiDataset, calculate_su_miso_rate_mrt
from rpmpq_v2 import RESULTS_CSV

os.makedirs(RESULTS_CSV, exist_ok=True)


# =====================================================================
# Local DP solver (extended for joint FC + non-FC optimization)
# =====================================================================
def enumerate_segments_joint(M_fc, M_nonfc, L_max):
    """Enumerate segments for joint FC + non-FC optimization.

    FC region [0, M_fc): contiguous segments up to L_max.
    Non-FC region [M_fc, M_fc+M_nonfc): length-1 only (independent layers).

    Returns list of (l, r) tuples (python-style half-open intervals).
    """
    segments = []
    # FC region: contiguous segments
    for l in range(M_fc):
        for r in range(l + 1, min(l + L_max + 1, M_fc + 1)):
            segments.append((l, r))
    # Non-FC region: each layer is an independent block
    for i in range(M_nonfc):
        segments.append((M_fc + i, M_fc + i + 1))
    return segments


def solve_dp(M, segments, omega, kappa, budget, bit_options, anchor_bits,
             C_steps=3000):
    """DP to find optimal segmentation under budget constraint.

    F(m, c) = min distortion using blocks [0..m) with cost <= c

    Args:
        M: total number of blocks (FC chunks + non-FC layers)
        segments: list of (l, r) tuples
        omega: dict[(l,r,b)] -> distortion
        kappa: dict[(l,r,b)] -> cost (fraction of FP32 BOPs)
        budget: max total cost (1 - target_saving/100)
        bit_options: [16, 8, 4, 2]
        anchor_bits: 16
        C_steps: budget discretization granularity (default 5000)

    Returns:
        best_distortion, segmentation list [(l, r, b), ...]
    """
    c_step = budget / C_steps if budget > 0 else 1e-9

    INF = float('inf')
    F = [[INF] * (C_steps + 1) for _ in range(M + 1)]
    back = [[None] * (C_steps + 1) for _ in range(M + 1)]

    for c in range(C_steps + 1):
        F[0][c] = 0.0

    segs_ending_at = {}
    for (l, r) in segments:
        segs_ending_at.setdefault(r, []).append(l)

    for m in range(1, M + 1):
        if m in segs_ending_at:
            for l in segs_ending_at[m]:
                for b in bit_options:
                    sc = kappa.get((l, m, b), 0)
                    sci = math.ceil(sc / c_step) if c_step > 0 else 0
                    sd = omega.get((l, m, b), 0)
                    if sci > C_steps:
                        continue
                    for c in range(sci, C_steps + 1):
                        cand = F[l][c - sci] + sd
                        if cand < F[m][c]:
                            F[m][c] = cand
                            back[m][c] = (l, b)

    best_c, best_val = 0, INF
    for c in range(C_steps + 1):
        if F[M][c] < best_val:
            best_val = F[M][c]
            best_c = c

    seg = []
    m, c = M, best_c
    while m > 0 and back[m][c] is not None:
        l, b = back[m][c]
        seg.append((l, m, b))
        sci = math.ceil(kappa.get((l, m, b), 0) / c_step) if c_step > 0 else 0
        c -= sci
        m = l
    if m > 0:
        seg.append((0, m, anchor_bits))
    seg.reverse()
    return best_val, seg


def segmentation_to_policy(segmentation, block_names):
    """Convert segmentation [(l, r, b), ...] to {block_name: bits} policy.

    Works with both FC-only (32 blocks) and joint (40 blocks) indexing.
    block_names[i] maps index i to the actual block name.
    """
    policy = {}
    for (l, r, b) in segmentation:
        for i in range(l, r):
            if i < len(block_names):
                policy[block_names[i]] = b
    return policy


# Legacy-compatible wrapper: enumerate_segments = FC-only version
def enumerate_segments(M, L_max):
    """Enumerate FC-only segments (legacy compat). Use enumerate_segments_joint for v2."""
    return enumerate_segments_joint(M, 0, L_max)


# =====================================================================
# Utility: install fake 'utils' modules so CLNet/CRNet can import
# =====================================================================
def _install_fake_utils():
    """Install stub modules for utils.logger / utils.solver expected by CLNet/CRNet."""
    if 'utils' in sys.modules and hasattr(sys.modules['utils'], '_is_stub'):
        return  # already installed

    _logger = logging.getLogger('BaselineStub')
    _logger.setLevel(logging.WARNING)
    if not _logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter('%(message)s'))
        _logger.addHandler(_handler)

    _utils_mod = types.ModuleType('utils')
    _utils_mod._is_stub = True
    _utils_mod.logger = _logger
    _utils_mod.line_seg = '-' * 40

    _logger_mod = types.ModuleType('utils.logger')
    _logger_mod.info = _logger.info
    _logger_mod.warning = _logger.warning
    _logger_mod.error = _logger.error
    _logger_mod.line_seg = '-' * 40

    _Result = namedtuple('Result', ('nmse', 'rho', 'epoch'), defaults=(None,) * 3)
    _solver_mod = types.ModuleType('utils.solver')
    _solver_mod.Result = _Result

    sys.modules['utils'] = _utils_mod
    sys.modules['utils.logger'] = _logger_mod
    sys.modules['utils.solver'] = _solver_mod


# =====================================================================
# Data loading
# =====================================================================
def load_test_data():
    """Load outdoor test data with normalization from training set."""
    train_path = os.path.join(PROJECT_ROOT, "data", "DATA_Htrainout.mat")
    try:
        train_set = CsiDataset(train_path, "HT")
        norm_params = train_set.normalization_params
        # Guard: dummy data gives wrong normalization
        if len(train_set) <= 200:
            raise MemoryError("Training set too small (likely dummy fallback)")
    except (MemoryError, OSError):
        # Pre-computed from outdoor training set (100k samples, float32):
        #   min = 5.2025792562915285e-09, range = 1.0
        norm_params = (5.2025792562915285e-09, 1.0)
        print(f"  [INFO] Using pre-computed norm params (train data too large for RAM)")

    test_set = CsiDataset(os.path.join(PROJECT_ROOT, "data", "DATA_Htestout.mat"), "HT",
                           normalization_params=norm_params)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    return test_set, test_loader, norm_params


# =====================================================================
# Encoder analysis helpers
# =====================================================================
def get_encoder_modules(model, model_name):
    """Get all quantizable encoder modules (weight, dim>=2).

    Returns:
        modules: list of (name, module)
        fc_name: name of the bottleneck FC/Conv1d layer (largest params)
        fc_module: the FC module itself
    """
    modules = []

    if model_name == "CLNet":
        # CLNet has model.encoder (Encoder class)
        for name, module in model.encoder.named_modules():
            if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() >= 2:
                modules.append((f"encoder.{name}", module))

    elif model_name == "CRNet":
        # CRNet: encoder parts are encoder1, encoder2, encoder_conv, encoder_fc
        encoder_prefixes = ['encoder1', 'encoder2', 'encoder_conv', 'encoder_fc']
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() >= 2:
                is_encoder = any(name.startswith(p) or name == p for p in encoder_prefixes)
                if is_encoder:
                    modules.append((name, module))

    elif model_name == "CsiNet":
        # CsiNet (ModularAE): model.encoder
        for name, module in model.encoder.named_modules():
            if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() >= 2:
                modules.append((f"encoder.{name}", module))

    elif model_name == "MT-AE":
        # MT-AE (ModularAE mamba+transnet): model.encoder
        for name, module in model.encoder.named_modules():
            if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() >= 2:
                modules.append((f"encoder.{name}", module))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Find the FC bottleneck (largest by param count, must be Linear or Conv1d)
    fc_name, fc_module = None, None
    max_params = 0
    for name, module in modules:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            n_params = module.weight.numel()
            if n_params > max_params:
                max_params = n_params
                fc_name, fc_module = name, module

    return modules, fc_name, fc_module


def run_inference(model, loader, norm_params, device, snr=20):
    """Run full model inference, return per-sample NMSE and rates.

    Works for any model where model(x) returns x_hat.
    """
    model.eval()
    min_val, range_val = norm_params
    nmse_all, rate_all = [], []

    with torch.no_grad():
        for batch in loader:
            d = batch.to(device)
            x_hat = model(d)

            # Denormalize
            h_true = (d * range_val) + min_val - 0.5
            h_hat = (x_hat * range_val) + min_val - 0.5

            error = torch.sum((h_true - h_hat) ** 2, dim=[1, 2, 3])
            power = torch.sum(h_true ** 2, dim=[1, 2, 3])
            nmse_all.extend((error / (power + 1e-9)).cpu().numpy().tolist())

            r = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
            rate_all.extend(r.cpu().numpy().tolist())

    return np.array(nmse_all), np.array(rate_all)


def apply_quantization_to_modules(modules, bits_map, fc_name, fc_module,
                                   fc_original, fc_chunks, chunk_assignments=None):
    """Apply quantization to encoder modules.

    Args:
        modules: list of (name, module)
        bits_map: dict name -> bits for non-FC modules
        fc_name: name of FC module
        fc_module: the FC module
        fc_original: original FC weight tensor
        fc_chunks: number of FC chunks
        chunk_assignments: dict chunk_idx -> bits (for FC chunking)
                           If None, all chunks use bits_map.get(fc_name, 16)
    """
    # Non-FC modules
    for name, module in modules:
        if name == fc_name:
            continue
        b = bits_map.get(name, 16)
        if b < 32:
            module.weight.data = quantize_int_asym(module.weight.data, b)

    # FC module with chunking
    w = fc_original.clone()
    out_dim = w.shape[0]
    chunk_size = out_dim // fc_chunks

    if chunk_assignments is None:
        # Uniform quantization of entire FC
        b = bits_map.get(fc_name, 16)
        if b < 32:
            fc_module.weight.data = quantize_int_asym(w, b)
        else:
            fc_module.weight.data = w
    else:
        # Per-chunk quantization
        for ci in range(fc_chunks):
            cs = ci * chunk_size
            ce = min(cs + chunk_size, out_dim)
            b = chunk_assignments.get(ci, 16)
            if b < 32:
                w[cs:ce] = quantize_int_asym(fc_original[cs:ce].clone(), b)
            else:
                w[cs:ce] = fc_original[cs:ce].clone()
        fc_module.weight.data = w


# =====================================================================
# Model loaders
# =====================================================================
def _import_baseline_model(baseline_dir, module_path, attr_name):
    """Import a model class from a baseline codebase without polluting sys.modules.

    Uses importlib to load from a specific directory, avoiding conflicts
    when multiple baselines have 'models' packages.
    """
    import importlib.util

    full_path = os.path.join(baseline_dir, module_path.replace(".", os.sep) + ".py")
    spec = importlib.util.spec_from_file_location(
        f"_baseline_{attr_name}", full_path,
        submodule_search_locations=[os.path.join(baseline_dir, "models")])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr_name)


def load_clnet(device):
    """Load CLNet model (CR=1/4, outdoor)."""
    _install_fake_utils()

    clnet_dir = os.path.join(BASELINES_ROOT, "CLNet-master")
    clnet_fn = _import_baseline_model(clnet_dir, "models.clnet", "clnet")

    model = clnet_fn(reduction=4)
    model.to(device)

    ckpt = os.path.join(clnet_dir, "checkpoints", "out4.pth")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        sd = state.get('state_dict', state)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        print(f"  [WARN] Checkpoint not found: {ckpt}")

    model.eval()
    return model, "CLNet"


def load_crnet(device):
    """Load CRNet model (CR=1/4, outdoor)."""
    _install_fake_utils()

    crnet_dir = os.path.join(BASELINES_ROOT, "CRNet-master")
    crnet_fn = _import_baseline_model(crnet_dir, "models.crnet", "crnet")

    model = crnet_fn(reduction=4)
    model.to(device)

    ckpt = os.path.join(crnet_dir, "checkpoints", "out_04.pth")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        sd = state.get('state_dict', state)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        print(f"  [WARN] Checkpoint not found: {ckpt}")

    model.eval()
    return model, "CRNet"


def load_csinet(device):
    """Load CsiNet model via original Keras (channels_first, CR=1/4, outdoor).

    Returns a Keras model wrapped in a thin object so segment_dp_for_model
    can detect it as Keras and delegate to segment_dp_csinet_keras().
    """
    return {"_keras_csinet": True}, "CsiNet"


def load_mtae(device):
    """Load MT-AE (Mamba encoder + TransNet decoder, CR=1/4, outdoor)."""
    from ModularModels import ModularAE
    model = ModularAE(encoder_type='mamba', decoder_type='transnet',
                       encoded_dim=512, M=32,
                       encoder_layers=2, decoder_layers=2)
    model.to(device)

    ckpt = os.path.join(PROJECT_ROOT, "saved_models",
                        "mamba_transnet_L2_dim512_baseline", "best.pth")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        sd = state.get('state_dict', state)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        print(f"  [WARN] Checkpoint not found: {ckpt}")

    model.eval()
    return model, "MT-AE"


# =====================================================================
# CsiNet Keras: Joint Segment DP using original Keras model
# =====================================================================
def segment_dp_csinet_keras(test_set, norm_params, snr=20):
    """Run Joint Segment DP on CsiNet using original Keras model directly.

    Uses csinet_onlytest.py's build_model + channels_first H5 weights.
    Matches the exact NMSE from the original CsiNet evaluation (-8.75 dB).
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Dense, BatchNormalization,
                                          Reshape, Conv2D, Add, LeakyReLU)
    from tensorflow.keras.models import Model as KerasModel

    print(f"\n{'=' * 60}")
    print(f"  Segment DP v2 (joint, Keras): CsiNet")
    print(f"{'=' * 60}")

    # ---- Config ----
    encoded_dim = 512
    fc_chunks = 32
    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    L_max = 6
    C_STEPS = 3000

    # ---- Build Keras model (channels_first, matching csinet_onlytest.py) ----
    df_fmt, bn_axis = 'channels_first', 3  # axis=3 matches original CsiNet (BN over width)
    inp = Input((2, 32, 32), name='input_1')
    ec = Conv2D(2, (3,3), padding='same', data_format=df_fmt, name='conv2d_1')
    eb = BatchNormalization(axis=bn_axis, name='batch_normalization_1')
    el = LeakyReLU(alpha=0.3, name='leaky_re_lu_1')
    er = Reshape((2*32*32,), name='reshape_1')
    ed = Dense(encoded_dim, activation='linear', name='dense_1')

    x = ec(inp); x = eb(x); x = el(x); x = er(x); enc = ed(x)

    dd = Dense(2*32*32, activation='linear', name='dense_2')
    dr = Reshape((2, 32, 32), name='reshape_2')
    res_layers = []
    for i in range(2):
        idx1, idx2, idx3 = 2+i*3, 3+i*3, 4+i*3
        res_layers.append({
            'c1': Conv2D(8, (3,3), padding='same', data_format=df_fmt, name=f'conv2d_{idx1}'),
            'b1': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx1}'),
            'l1': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx1}'),
            'c2': Conv2D(16, (3,3), padding='same', data_format=df_fmt, name=f'conv2d_{idx2}'),
            'b2': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx2}'),
            'l2': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx2}'),
            'c3': Conv2D(2, (3,3), padding='same', data_format=df_fmt, name=f'conv2d_{idx3}'),
            'b3': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx3}'),
            'add': Add(name=f'add_{i+1}'),
            'lout': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx3}_out'),
        })
    dfinal = Conv2D(2, (3,3), padding='same', activation='sigmoid',
                     data_format=df_fmt, name='conv2d_8')

    x = dd(enc); x = dr(x)
    for b in res_layers:
        s = x
        x = b['c1'](x); x = b['b1'](x); x = b['l1'](x)
        x = b['c2'](x); x = b['b2'](x); x = b['l2'](x)
        x = b['c3'](x); x = b['b3'](x)
        x = b['add']([s, x]); x = b['lout'](x)
    out = dfinal(x)

    ae = KerasModel(inp, out, name='AE')
    encoder = KerasModel(inp, enc, name='Encoder')

    # Decoder
    dec_in = Input((encoded_dim,))
    x = dd(dec_in); x = dr(x)
    for b in res_layers:
        s = x
        x = b['c1'](x); x = b['b1'](x); x = b['l1'](x)
        x = b['c2'](x); x = b['b2'](x); x = b['l2'](x)
        x = b['c3'](x); x = b['b3'](x)
        x = b['add']([s, x]); x = b['lout'](x)
    dec_out = dfinal(x)
    decoder = KerasModel(dec_in, dec_out, name='Decoder')

    # ---- Load weights ----
    w_dir = os.path.join(BASELINES_ROOT, "Python_CsiNet-master", "saved_model")
    w_path = os.path.join(w_dir, "model_CsiNet_outdoor_dim512.h5")
    ae.load_weights(w_path, by_name=True)
    print(f"  Loaded Keras weights: {w_path}")

    # ---- Load test data as numpy ----
    data_np = np.array([test_set[i].numpy() for i in range(len(test_set))])
    N_test = len(data_np)

    # Perfect rates
    perf_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    r_ref = pd.read_csv(perf_csv)[f"r_perf_{snr}"].values if os.path.exists(perf_csv) else None

    # ---- Quantize helper (numpy, same as csinet_onlytest.py) ----
    def quantize_np(w, bits):
        if bits >= 32 or w.size == 0:
            return w
        q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
        w_min, w_max = np.min(w), np.max(w)
        w_min, w_max = min(w_min, 0.0), max(w_max, 0.0)
        if w_max == w_min:
            return w
        scale = (w_max - w_min) / (q_max - q_min)
        zp = np.clip(np.round(q_min - w_min / scale), q_min, q_max)
        return (np.clip(np.round(w / scale + zp), q_min, q_max) - zp) * scale

    # ---- NMSE helper (complex, same as csinet_onlytest.py line 461-465) ----
    def compute_nmse_keras(x_true, x_hat):
        x_true_c = x_true[:, 0] - 0.5 + 1j * (x_true[:, 1] - 0.5)
        x_hat_c = x_hat[:, 0] - 0.5 + 1j * (x_hat[:, 1] - 0.5)
        mse_per = np.sum(np.abs(x_true_c - x_hat_c)**2, axis=(1, 2))
        pwr_per = np.sum(np.abs(x_true_c)**2, axis=(1, 2))
        return mse_per, pwr_per

    # ---- Identify encoder layers ----
    orig_weights = {l.name: l.get_weights() for l in encoder.layers if l.weights}
    enc_layer_names = [l.name for l in encoder.layers if l.weights]

    # FC layer = 'dense_1', non-FC = 'conv2d_1', 'batch_normalization_1'
    fc_layer_name = 'dense_1'
    fc_layer = encoder.get_layer(fc_layer_name)
    fc_weight_orig = fc_layer.get_weights()[0].copy()  # (2048, 512)
    fc_out_dim = fc_weight_orig.shape[0]  # 2048
    chunk_size = fc_out_dim // fc_chunks

    non_fc_layers = [(l.name, l) for l in encoder.layers
                     if l.weights and l.name != fc_layer_name
                     and hasattr(l, 'kernel')]
    non_fc_originals = {name: l.get_weights() for name, l in non_fc_layers}

    M_fc = fc_chunks
    M_nonfc = len(non_fc_layers)
    M_total = M_fc + M_nonfc

    print(f"  FC layer: {fc_layer_name}  shape={fc_weight_orig.shape}")
    print(f"  Non-FC encoder layers: {M_nonfc}")
    for name, l in non_fc_layers:
        print(f"    {name}: {l.get_weights()[0].shape}  "
              f"params={l.get_weights()[0].size}")

    print(f"\n  Joint DP: {M_fc} FC chunks + {M_nonfc} non-FC = {M_total} total blocks")

    # ---- BOPs calculation ----
    total_enc_params = sum(l.get_weights()[0].size for name, l in non_fc_layers)
    total_enc_params += fc_weight_orig.size
    total_fp32_bops = total_enc_params * 32 * 32
    fc_chunk_params = fc_weight_orig.size // fc_chunks
    non_fc_params_list = [(name, l.get_weights()[0].size) for name, l in non_fc_layers]

    print(f"  Total encoder params: {total_enc_params:,}")
    print(f"  FC params: {fc_weight_orig.size:,} (per chunk: {fc_chunk_params:,})")

    # ---- Segments + Kappa ----
    segments = enumerate_segments_joint(M_fc, M_nonfc, L_max)
    kappa_seg = {}
    for (l, r) in segments:
        if l < M_fc:
            for b in bit_options:
                kappa_seg[(l, r, b)] = ((r-l) * fc_chunk_params * b * 16) / total_fp32_bops
        else:
            nfc_idx = l - M_fc
            nfc_params = non_fc_params_list[nfc_idx][1]
            for b in bit_options:
                kappa_seg[(l, r, b)] = (nfc_params * b * 16) / total_fp32_bops

    # ---- Helper: restore all weights ----
    def restore_all():
        for l in encoder.layers:
            if l.name in orig_weights:
                l.set_weights(orig_weights[l.name])

    # ---- Helper: run quantized inference ----
    def run_eval(fc_chunk_bits, nonfc_bit_map):
        """Apply quantization and run full AE inference, return NMSE."""
        restore_all()
        # FC chunks
        w = fc_weight_orig.copy()
        for ci in range(fc_chunks):
            b = fc_chunk_bits.get(ci, anchor_bits)
            if b < 32:
                cs = ci * chunk_size
                ce = min(cs + chunk_size, fc_out_dim)
                w[cs:ce] = quantize_np(fc_weight_orig[cs:ce], b)
        fc_layer.set_weights([w] + list(orig_weights[fc_layer_name][1:]))

        # Non-FC layers
        for idx, (name, layer) in enumerate(non_fc_layers):
            b = nonfc_bit_map.get(idx, anchor_bits)
            ws = [quantize_np(orig_weights[name][0], b)] + list(orig_weights[name][1:])
            layer.set_weights(ws)

        # Inference
        x_hat = ae.predict(data_np, verbose=0, batch_size=512)
        mse_per, pwr_per = compute_nmse_keras(data_np, x_hat)
        nmse_db = float(10 * np.log10(np.mean(mse_per / pwr_per) + 1e-15))

        # Rate + outage (if r_ref available)
        outage_99, outage_98, outage_95 = 0.0, 0.0, 0.0
        rate_mean = 0.0
        if r_ref is not None:
            # Use PyTorch for rate calculation
            import torch as _torch
            h_true = _torch.from_numpy(data_np).float()
            h_hat = _torch.from_numpy(x_hat).float()
            h_t = h_true - 0.5
            h_h = h_hat - 0.5
            _dev = "cuda" if _torch.cuda.is_available() else "cpu"
            r = calculate_su_miso_rate_mrt(
                h_t.to(_dev), h_h.to(_dev), snr, _dev).cpu().numpy()
            rate_mean = float(np.mean(r))
            N = min(len(r), len(r_ref))
            outage_99 = float(np.mean(r[:N] < 0.99 * r_ref[:N]))
            outage_98 = float(np.mean(r[:N] < 0.98 * r_ref[:N]))
            outage_95 = float(np.mean(r[:N] < 0.95 * r_ref[:N]))

        return nmse_db, float(np.mean(mse_per / pwr_per)), rate_mean, outage_99, outage_98, outage_95

    # ---- [1] Anchor ----
    print(f"\n  [1] Anchor (all INT{anchor_bits})...")
    anc_nmse_db, _, _, _, _, _ = run_eval(
        {ci: anchor_bits for ci in range(fc_chunks)},
        {idx: anchor_bits for idx in range(M_nonfc)})
    print(f"    NMSE: {anc_nmse_db:.2f} dB")

    # ---- [2] Collect omegas ----
    cache_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas_v2_csinet.csv")
    if os.path.exists(cache_csv):
        print(f"\n  [2] Loading cached v2 omegas from {cache_csv}")
        df_c = pd.read_csv(cache_csv)
        omega_nmse = {}
        for _, row in df_c.iterrows():
            omega_nmse[(int(row["l"]), int(row["r"]), int(row["b"]))] = row["omega_nmse"]
        for (l, r) in segments:
            omega_nmse[(l, r, anchor_bits)] = 0.0
    else:
        print(f"\n  [2] Collecting omegas for all segments...")
        omega_nmse = {}
        cache_rows = []

        # Anchor NMSE per-sample for delta computation
        restore_all()
        for l in encoder.layers:
            if l.name in orig_weights and hasattr(l, 'kernel'):
                ws = [quantize_np(orig_weights[l.name][0], anchor_bits)] + list(orig_weights[l.name][1:])
                l.set_weights(ws)
        w_anc = quantize_np(fc_weight_orig, anchor_bits)
        fc_layer.set_weights([w_anc] + list(orig_weights[fc_layer_name][1:]))
        x_hat_anc = ae.predict(data_np, verbose=0, batch_size=512)
        mse_anc, pwr_anc = compute_nmse_keras(data_np, x_hat_anc)
        nmse_anc_per = mse_anc / (pwr_anc + 1e-15)

        # FC segment omegas
        fc_segments = [(l, r) for (l, r) in segments if l < M_fc]
        for seg_i, (sl, sr) in enumerate(tqdm(fc_segments, desc="CsiNet FC")):
            for b in bit_options:
                if b == anchor_bits:
                    omega_nmse[(sl, sr, b)] = 0.0
                    continue

                restore_all()
                # All at anchor
                for ll in encoder.layers:
                    if ll.name in orig_weights and hasattr(ll, 'kernel'):
                        ws = [quantize_np(orig_weights[ll.name][0], anchor_bits)] + list(orig_weights[ll.name][1:])
                        ll.set_weights(ws)
                # FC: anchor, then override segment
                w = quantize_np(fc_weight_orig, anchor_bits)
                for ci in range(sl, sr):
                    cs = ci * chunk_size
                    ce = min(cs + chunk_size, fc_out_dim)
                    w[cs:ce] = quantize_np(fc_weight_orig[cs:ce], b)
                fc_layer.set_weights([w] + list(orig_weights[fc_layer_name][1:]))

                x_hat_p = ae.predict(data_np, verbose=0, batch_size=512)
                mse_p, pwr_p = compute_nmse_keras(data_np, x_hat_p)
                nmse_p_per = mse_p / (pwr_p + 1e-15)
                omega_val = float(np.mean(nmse_p_per - nmse_anc_per))
                omega_nmse[(sl, sr, b)] = omega_val
                cache_rows.append({"l": sl, "r": sr, "b": b, "omega_nmse": omega_val})

        # Non-FC omegas
        print(f"    Collecting {M_nonfc} non-FC omegas...")
        for nfc_idx, (name, layer) in enumerate(
                tqdm(non_fc_layers, desc="CsiNet non-FC")):
            bl = M_fc + nfc_idx
            for b in bit_options:
                if b == anchor_bits:
                    omega_nmse[(bl, bl+1, b)] = 0.0
                    continue

                restore_all()
                for ll in encoder.layers:
                    if ll.name in orig_weights and hasattr(ll, 'kernel'):
                        ws = [quantize_np(orig_weights[ll.name][0], anchor_bits)] + list(orig_weights[ll.name][1:])
                        ll.set_weights(ws)
                w_fc = quantize_np(fc_weight_orig, anchor_bits)
                fc_layer.set_weights([w_fc] + list(orig_weights[fc_layer_name][1:]))
                # Override this non-FC layer
                ws = [quantize_np(orig_weights[name][0], b)] + list(orig_weights[name][1:])
                layer.set_weights(ws)

                x_hat_p = ae.predict(data_np, verbose=0, batch_size=512)
                mse_p, pwr_p = compute_nmse_keras(data_np, x_hat_p)
                nmse_p_per = mse_p / (pwr_p + 1e-15)
                omega_val = float(np.mean(nmse_p_per - nmse_anc_per))
                omega_nmse[(bl, bl+1, b)] = omega_val
                cache_rows.append({"l": bl, "r": bl+1, "b": b, "omega_nmse": omega_val})
                print(f"      {name} INT{b}: omega={omega_val:.6f}")

        pd.DataFrame(cache_rows).to_csv(cache_csv, index=False)
        print(f"    Cached -> {cache_csv}")

    # ---- [3] Joint DP sweep ----
    savings = np.arange(85, 97.01, 0.1).tolist()
    print(f"\n  [3] Joint DP sweep: {len(savings)} targets (C_steps={C_STEPS})...")

    results = []
    policy_cache = {}
    omega_dict = {(l, r, b): omega_nmse.get((l, r, b), 0)
                  for (l, r) in segments for b in bit_options}

    for target_saving in tqdm(savings, desc="CsiNet DP sweep"):
        total_budget = 1.0 - target_saving / 100.0
        _, seg = solve_dp(M_total, segments, omega_dict, kappa_seg,
                          total_budget, bit_options, anchor_bits, C_STEPS)

        # Build policy
        fc_chunk_bits = {}
        nonfc_bit_map = {}
        fc_parts = []
        nonfc_parts = []
        for (sl, sr, sb) in seg:
            if sl < M_fc:
                fc_parts.append(f"[{sl}:{sr}]INT{sb}")
                for ci in range(sl, sr):
                    fc_chunk_bits[ci] = sb
            else:
                nfc_idx = sl - M_fc
                nonfc_bit_map[nfc_idx] = sb
                nfc_name = non_fc_layers[nfc_idx][0]
                nonfc_parts.append(f"{nfc_name}=INT{sb}")

        seg_str = " ".join(fc_parts)
        if nonfc_parts:
            seg_str += " | " + " ".join(nonfc_parts)

        # Actual saving
        total_bops = 0
        for (sl, sr, sb) in seg:
            if sl < M_fc:
                total_bops += (sr-sl) * fc_chunk_params * sb * 16
            else:
                nfc_idx = sl - M_fc
                total_bops += non_fc_params_list[nfc_idx][1] * sb * 16
        actual_saving = (1.0 - total_bops / total_fp32_bops) * 100

        if seg_str not in policy_cache:
            nmse_db, nmse_lin, rate_mean, out99, out98, out95 = run_eval(
                fc_chunk_bits, nonfc_bit_map)
            policy_cache[seg_str] = (nmse_db, nmse_lin, rate_mean, out99, out98, out95)

        nmse_db, nmse_lin, rate_mean, out99, out98, out95 = policy_cache[seg_str]
        results.append({
            "model": "CsiNet", "method": "segment-dp",
            "target_saving": target_saving, "actual_saving": actual_saving,
            "nmse_db": nmse_db, "nmse_linear": nmse_lin,
            "rate_mean": rate_mean, "outage_99": out99, "outage_98": out98, "outage_95": out95,
            "segmentation": seg_str,
        })

    print(f"    {len(savings)} targets -> {len(policy_cache)} unique policies")

    # ---- [4] Uniform baselines ----
    print(f"\n  [4] Uniform baselines...")
    for ub in [16, 8, 4]:
        fc_bits = {ci: ub for ci in range(fc_chunks)}
        nfc_bits = {idx: ub for idx in range(M_nonfc)}
        nmse_db, nmse_lin, rate_mean, out99, out98, out95 = run_eval(fc_bits, nfc_bits)

        enc_bops = total_enc_params * ub * 16
        saving = (1.0 - enc_bops / total_fp32_bops) * 100
        print(f"    INT{ub:2d}: NMSE={nmse_db:.2f}dB  saving={saving:.1f}%")

        results.append({
            "model": "CsiNet", "method": f"uniform-INT{ub}",
            "target_saving": saving, "actual_saving": saving,
            "nmse_db": nmse_db, "nmse_linear": nmse_lin,
            "rate_mean": rate_mean, "outage_99": out99, "outage_98": out98, "outage_95": out95,
            "segmentation": f"all-INT{ub}",
        })

    restore_all()
    print("  CsiNet Keras done.")
    return results


# =====================================================================
# Segment DP pipeline for a single model (PyTorch)
# =====================================================================
def segment_dp_for_model(model, model_name, test_set, test_loader,
                          norm_params, device):
    """Run full Segment DP pipeline on a given model.

    v2: jointly optimizes ALL encoder layers (FC chunks + non-FC conv/fc).
    Non-FC layers are appended as length-1 blocks after FC chunks.
    """
    print(f"\n{'=' * 60}")
    print(f"  Segment DP v2 (joint): {model_name}")
    print(f"{'=' * 60}")

    # ---- Encoder analysis ----
    modules, fc_name, fc_module = get_encoder_modules(model, model_name)

    if fc_module is None:
        print(f"  [WARN] No FC bottleneck found in {model_name}, skipping")
        return None

    fc_out_dim = fc_module.weight.shape[0]
    fc_in_dim = fc_module.weight.shape[1]
    fc_chunks = 32
    actual_chunks = min(fc_chunks, fc_out_dim)
    chunk_size = fc_out_dim // actual_chunks

    non_fc_modules = [(n, m) for n, m in modules if n != fc_name]

    print(f"  FC layer: {fc_name}  (type={type(fc_module).__name__})")
    print(f"    shape={list(fc_module.weight.shape)}, "
          f"params={fc_module.weight.numel():,}")
    print(f"    Splitting into {actual_chunks} chunks "
          f"(chunk_size={chunk_size})")
    print(f"  Non-FC encoder blocks: {len(non_fc_modules)}")
    for n, m in non_fc_modules:
        print(f"    {n}: {type(m).__name__}  {list(m.weight.shape)}  "
              f"params={m.weight.numel():,}")

    # ---- Config ----
    M_fc = actual_chunks
    M_nonfc = len(non_fc_modules)
    M_total = M_fc + M_nonfc
    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    snr = 20
    L_max = 6
    C_STEPS = 3000  # budget discretization (3000 = ~1s/call, good resolution)

    print(f"\n  Joint DP: {M_fc} FC chunks + {M_nonfc} non-FC layers = "
          f"{M_total} total blocks")

    # ---- Perfect rates (from MambaEncoder reference) ----
    perf_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    if not os.path.exists(perf_csv):
        print(f"  [ERROR] Perfect rates CSV not found: {perf_csv}")
        print(f"  Run rpmpq_v2.py --step collect first.")
        return None
    r_ref = pd.read_csv(perf_csv)[f"r_perf_{snr}"].values
    N_test = len(test_set)

    # ---- Save original state ----
    original_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    fc_original = fc_module.weight.data.clone()
    # Save non-FC originals too
    nonfc_originals = {}
    for nfc_name, nfc_mod in non_fc_modules:
        nonfc_originals[nfc_name] = nfc_mod.weight.data.clone()

    # ---- BOPs calculation ----
    total_enc_params = sum(m.weight.numel() for _, m in modules)
    total_fp32_bops = total_enc_params * 32 * 32
    non_fc_params = sum(m.weight.numel() for _, m in non_fc_modules)
    fc_chunk_params = (fc_module.weight.numel()) // actual_chunks

    print(f"\n  Total encoder params: {total_enc_params:,}")
    print(f"  Non-FC params: {non_fc_params:,}")
    print(f"  FC params: {fc_module.weight.numel():,}  "
          f"(per chunk: {fc_chunk_params:,})")

    # ---- Segments: FC (contiguous, up to L_max) + non-FC (length 1) ----
    segments = enumerate_segments_joint(M_fc, M_nonfc, L_max)
    n_fc_segs = sum(1 for (l, r) in segments if l < M_fc)
    print(f"\n  Segments: {len(segments)} total "
          f"({n_fc_segs} FC + {M_nonfc} non-FC)")

    # ---- Kappa for all segments ----
    kappa_seg = {}
    for (l, r) in segments:
        if l < M_fc:
            # FC segment
            n_chunks = r - l
            for b in bit_options:
                kappa_seg[(l, r, b)] = (
                    n_chunks * fc_chunk_params * b * 16) / total_fp32_bops
        else:
            # Non-FC layer
            nonfc_idx = l - M_fc
            nfc_name, nfc_mod = non_fc_modules[nonfc_idx]
            nfc_params = nfc_mod.weight.numel()
            for b in bit_options:
                kappa_seg[(l, r, b)] = (nfc_params * b * 16) / total_fp32_bops

    # ---- [1] Anchor inference (all INT16) ----
    print(f"\n  [1] Anchor (all INT{anchor_bits})...")
    model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
    for name, module in modules:
        if module.weight.dim() >= 2:
            module.weight.data = quantize_int_asym(module.weight.data, anchor_bits)

    nmse_anc, rate_anc = run_inference(model, test_loader, norm_params, device, snr)
    nmse_anc_db = 10 * np.log10(np.mean(nmse_anc) + 1e-15)
    print(f"    NMSE: {nmse_anc_db:.2f} dB")

    if nmse_anc_db > -1.0:
        print(f"    [SKIP] Anchor NMSE too poor ({nmse_anc_db:.2f} dB > -1 dB).")
        model.load_state_dict(
            {k: v.to(device) for k, v in original_state.items()})
        return None

    # ---- [2] Collect omegas for ALL segments (FC + non-FC) ----
    cache_csv = os.path.join(
        RESULTS_CSV, f"segment_dp_omegas_v2_{model_name.lower()}.csv")

    if os.path.exists(cache_csv):
        print(f"\n  [2] Loading cached v2 omegas from {cache_csv}")
        df_c = pd.read_csv(cache_csv)
        omega_nmse = {}
        for _, row in df_c.iterrows():
            key = (int(row["l"]), int(row["r"]), int(row["b"]))
            omega_nmse[key] = row["omega_nmse"]
        # Fill anchor entries
        for (l, r) in segments:
            omega_nmse[(l, r, anchor_bits)] = 0.0
        print(f"    Loaded {len(df_c)} entries")
    else:
        # Check if old FC-only cache exists (reuse FC omegas)
        # MT-AE's old cache is "segment_dp_omegas.csv" (no model suffix)
        old_cache_name = ("segment_dp_omegas.csv" if model_name == "MT-AE"
                          else f"segment_dp_omegas_{model_name.lower()}.csv")
        old_cache = os.path.join(RESULTS_CSV, old_cache_name)
        fc_omega_cache = {}
        if os.path.exists(old_cache):
            df_old = pd.read_csv(old_cache)
            if "j" in df_old.columns:
                # MT-AE cache has per-bin (j) omegas — average across bins
                df_avg = df_old.groupby(["l", "r", "b"])["omega_nmse"].mean()
                for (l, r, b), val in df_avg.items():
                    fc_omega_cache[(l, r, b)] = val
            else:
                for _, row in df_old.iterrows():
                    key = (int(row["l"]), int(row["r"]), int(row["b"]))
                    fc_omega_cache[key] = row["omega_nmse"]
            print(f"  [2] Reusing {len(fc_omega_cache)} FC omegas from old cache")

        print(f"\n  [2] Collecting omegas for all segments...")
        omega_nmse = {}
        cache_rows = []
        fc_segments = [(l, r) for (l, r) in segments if l < M_fc]
        nonfc_segments = [(l, r) for (l, r) in segments if l >= M_fc]

        # --- FC segment omegas ---
        n_fc_new = 0
        for (l, r) in tqdm(fc_segments, desc=f"{model_name} FC segments"):
            for b in bit_options:
                if b == anchor_bits:
                    omega_nmse[(l, r, b)] = 0.0
                    continue

                # Check old cache first
                if (l, r, b) in fc_omega_cache:
                    omega_val = fc_omega_cache[(l, r, b)]
                    omega_nmse[(l, r, b)] = omega_val
                    cache_rows.append({"l": l, "r": r, "b": b,
                                       "omega_nmse": omega_val})
                    continue

                n_fc_new += 1
                model.load_state_dict(
                    {k: v.to(device) for k, v in original_state.items()})
                for name, module in modules:
                    if name != fc_name and module.weight.dim() >= 2:
                        module.weight.data = quantize_int_asym(
                            module.weight.data, anchor_bits)
                w = quantize_int_asym(fc_original.clone(), anchor_bits)
                for ci in range(l, r):
                    cs = ci * chunk_size
                    ce = min(cs + chunk_size, fc_out_dim)
                    w[cs:ce] = quantize_int_asym(fc_original[cs:ce].clone(), b)
                fc_module.weight.data = w

                nmse_p, _ = run_inference(
                    model, test_loader, norm_params, device, snr)
                omega_val = float(np.mean(nmse_p - nmse_anc))
                omega_nmse[(l, r, b)] = omega_val
                cache_rows.append({"l": l, "r": r, "b": b,
                                   "omega_nmse": omega_val})

        if n_fc_new > 0:
            print(f"    {n_fc_new} new FC omega runs (rest from cache)")

        # --- Non-FC layer omegas ---
        print(f"    Collecting {M_nonfc} non-FC layer omegas "
              f"({M_nonfc * (len(bit_options)-1)} runs)...")
        for nonfc_idx, (nfc_name, nfc_mod) in enumerate(
                tqdm(non_fc_modules, desc=f"{model_name} non-FC")):
            block_l = M_fc + nonfc_idx
            block_r = block_l + 1

            for b in bit_options:
                if b == anchor_bits:
                    omega_nmse[(block_l, block_r, b)] = 0.0
                    continue

                # Restore all to original, then quantize all to anchor
                model.load_state_dict(
                    {k: v.to(device) for k, v in original_state.items()})
                for name, module in modules:
                    if module.weight.dim() >= 2:
                        module.weight.data = quantize_int_asym(
                            module.weight.data, anchor_bits)

                # Override this one non-FC layer to bit b
                nfc_mod.weight.data = quantize_int_asym(
                    nonfc_originals[nfc_name].clone().to(device), b)

                nmse_p, _ = run_inference(
                    model, test_loader, norm_params, device, snr)
                omega_val = float(np.mean(nmse_p - nmse_anc))
                omega_nmse[(block_l, block_r, b)] = omega_val
                cache_rows.append({"l": block_l, "r": block_r, "b": b,
                                   "omega_nmse": omega_val})
                print(f"      {nfc_name} INT{b}: omega={omega_val:.6f}")

        pd.DataFrame(cache_rows).to_csv(cache_csv, index=False)
        print(f"    Cached -> {cache_csv}")

    # ---- [3] Joint DP optimization (0.1% step sweep) ----
    savings = np.arange(85, 97.01, 0.1).tolist()
    print(f"\n  [3] Joint DP sweep: {len(savings)} saving levels "
          f"(85-97%, 0.1% step, C_steps={C_STEPS})...")

    results = []
    policy_cache = {}

    omega_dict = {(l, r, b): omega_nmse.get((l, r, b), 0)
                  for (l, r) in segments for b in bit_options}

    for target_saving in tqdm(savings, desc=f"{model_name} DP sweep"):
        total_budget = 1.0 - target_saving / 100.0

        best_dist, seg = solve_dp(
            M_total, segments, omega_dict, kappa_seg,
            total_budget, bit_options, anchor_bits, C_STEPS)

        # Build policy string (separate FC and non-FC for clarity)
        fc_parts = []
        nonfc_parts = []
        for (l, r, b) in seg:
            if l < M_fc:
                fc_parts.append(f"[{l}:{r}]INT{b}")
            else:
                nonfc_idx = l - M_fc
                nfc_name = non_fc_modules[nonfc_idx][0]
                # Shorten name for readability
                short_name = nfc_name.split(".")[-1] if "." in nfc_name else nfc_name
                nonfc_parts.append(f"{short_name}=INT{b}")

        seg_str = " ".join(fc_parts)
        if nonfc_parts:
            seg_str += " | " + " ".join(nonfc_parts)

        # Compute actual saving from segmentation
        total_bops = 0
        for (sl, sr, sb) in seg:
            if sl < M_fc:
                n_ch = sr - sl
                total_bops += n_ch * fc_chunk_params * sb * 16
            else:
                nonfc_idx = sl - M_fc
                nfc_params = non_fc_modules[nonfc_idx][1].weight.numel()
                total_bops += nfc_params * sb * 16
        actual_saving = (1.0 - total_bops / total_fp32_bops) * 100

        if seg_str not in policy_cache:
            # GPU inference for unique policies
            model.load_state_dict(
                {k: v.to(device) for k, v in original_state.items()})

            # Apply FC quantization
            w = quantize_int_asym(fc_original.clone(), anchor_bits)
            for (sl, sr, sb) in seg:
                if sl < M_fc:
                    for ci in range(sl, sr):
                        cs = ci * chunk_size
                        ce = min(cs + chunk_size, fc_out_dim)
                        w[cs:ce] = quantize_int_asym(
                            fc_original[cs:ce].clone(), sb)
            fc_module.weight.data = w

            # Apply non-FC quantization
            nonfc_bit_map = {}
            for (sl, sr, sb) in seg:
                if sl >= M_fc:
                    nonfc_idx = sl - M_fc
                    nonfc_bit_map[nonfc_idx] = sb

            for idx, (nfc_name, nfc_mod) in enumerate(non_fc_modules):
                b = nonfc_bit_map.get(idx, anchor_bits)
                nfc_mod.weight.data = quantize_int_asym(
                    nonfc_originals[nfc_name].clone().to(device), b)

            nmse_eval, rate_eval = run_inference(
                model, test_loader, norm_params, device, snr)
            nmse_db = 10 * np.log10(np.mean(nmse_eval) + 1e-15)

            N = min(len(rate_eval), len(r_ref))
            outage_99 = float(np.mean(rate_eval[:N] < 0.99 * r_ref[:N]))
            outage_98 = float(np.mean(rate_eval[:N] < 0.98 * r_ref[:N]))
            outage_95 = float(np.mean(rate_eval[:N] < 0.95 * r_ref[:N]))

            policy_cache[seg_str] = (
                nmse_db, float(np.mean(nmse_eval)),
                float(np.mean(rate_eval)), outage_99, outage_98, outage_95)

        nmse_db, nmse_lin, rate_mean, outage_99, outage_98, outage_95 = policy_cache[seg_str]

        results.append({
            "model": model_name,
            "method": "segment-dp",
            "target_saving": target_saving,
            "actual_saving": actual_saving,
            "nmse_db": nmse_db,
            "nmse_linear": nmse_lin,
            "rate_mean": rate_mean,
            "outage_99": outage_99,
            "outage_98": outage_98,
            "outage_95": outage_95,
            "segmentation": seg_str,
        })

    n_unique = len(policy_cache)
    print(f"    {len(savings)} targets -> {n_unique} unique policies evaluated")

    # Also add uniform baselines for comparison
    print(f"\n  [4] Uniform quantization baselines...")
    for ub in [16, 8, 4]:
        model.load_state_dict(
            {k: v.to(device) for k, v in original_state.items()})
        for name, module in modules:
            if module.weight.dim() >= 2:
                module.weight.data = quantize_int_asym(module.weight.data, ub)

        nmse_eval, rate_eval = run_inference(
            model, test_loader, norm_params, device, snr)
        nmse_db = 10 * np.log10(np.mean(nmse_eval) + 1e-15)

        N = min(len(rate_eval), len(r_ref))
        outage_99 = float(np.mean(rate_eval[:N] < 0.99 * r_ref[:N]))
        outage_98 = float(np.mean(rate_eval[:N] < 0.98 * r_ref[:N]))
        outage_95 = float(np.mean(rate_eval[:N] < 0.95 * r_ref[:N]))

        enc_bops = total_enc_params * ub * 16
        saving = (1.0 - enc_bops / total_fp32_bops) * 100

        print(f"    INT{ub:2d}: NMSE={nmse_db:.2f}dB  "
              f"out99={outage_99:.4f}  saving={saving:.1f}%")

        results.append({
            "model": model_name,
            "method": f"uniform-INT{ub}",
            "target_saving": saving,
            "actual_saving": saving,
            "nmse_db": nmse_db,
            "nmse_linear": float(np.mean(nmse_eval)),
            "rate_mean": float(np.mean(rate_eval)),
            "outage_99": outage_99,
            "outage_98": outage_98,
            "outage_95": outage_95,
            "segmentation": f"all-INT{ub}",
        })

    # Restore
    model.load_state_dict(
        {k: v.to(device) for k, v in original_state.items()})
    return results


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run (e.g. --models CsiNet). "
                             "Others loaded from cached CSV. Default: all.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    # Disable cuDNN to avoid CUDNN_STATUS_INTERNAL_ERROR on some GPU/driver combos
    torch.backends.cudnn.enabled = False

    print("=" * 60)
    print("  SEGMENT DP ON BASELINE MODELS")
    print("=" * 60)

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize CUDA context BEFORE loading large data files.
    # On memory-constrained systems, CUDA context creation can fail if
    # too much system RAM is consumed by data loading.
    if device == "cuda":
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")
    print(f"Device: {device.upper()}")

    test_set, test_loader, norm_params = load_test_data()
    print(f"Test samples: {len(test_set)}")

    all_results = []
    out_csv = os.path.join(RESULTS_CSV, "segment_dp_baselines.csv")

    model_loaders = [
        (load_csinet, "CsiNet"),
        (load_clnet, "CLNet"),
        (load_crnet, "CRNet"),
        (load_mtae, "MT-AE"),
    ]

    # Determine which models to run vs load from cache
    run_models = [n for _, n in model_loaders] if args.models is None else args.models
    cached_df = pd.read_csv(out_csv) if os.path.exists(out_csv) else None

    # Re-read cached CSV each iteration (supports incremental saves)
    for loader_fn, expected_name in model_loaders:
        # Reload cache to pick up results saved by previous iterations
        cached_df = pd.read_csv(out_csv) if os.path.exists(out_csv) else None

        if expected_name not in run_models:
            if cached_df is not None and expected_name in cached_df["model"].values:
                cached = cached_df[cached_df["model"] == expected_name]
                all_results.extend(cached.to_dict("records"))
                print(f"\n  [{expected_name}] Loaded {len(cached)} cached results")
            else:
                print(f"\n  [{expected_name}] Skipped (no cache)")
            continue

        # Check if already in cache WITH outage_98 (from previous run)
        if cached_df is not None and expected_name in cached_df["model"].values:
            cached = cached_df[cached_df["model"] == expected_name]
            if "outage_98" in cached.columns and cached["outage_98"].notna().all():
                all_results.extend(cached.to_dict("records"))
                print(f"\n  [{expected_name}] Loaded {len(cached)} cached results (already done)")
                continue
            else:
                print(f"\n  [{expected_name}] Cache missing outage_98, re-evaluating...")

        try:
            model, name = loader_fn(device)

            # CsiNet uses Keras directly (not PyTorch conversion)
            if isinstance(model, dict) and model.get("_keras_csinet"):
                results = segment_dp_csinet_keras(
                    test_set, norm_params, snr=20)
            else:
                results = segment_dp_for_model(
                    model, name, test_set, test_loader, norm_params, device)

            if results:
                all_results.extend(results)
                # Incremental save after each model
                df_save = pd.DataFrame(all_results)
                df_save.to_csv(out_csv, index=False)
                print(f"  [SAVED] {out_csv} ({len(df_save)} rows)")
        except Exception as e:
            print(f"\n  [ERROR] {expected_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        print(df.to_string(index=False))
    else:
        print("\n  No results collected.")

    print("\nDone.")


if __name__ == "__main__":
    main()
