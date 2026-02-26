#!/usr/bin/env python3
"""
rpmpq_baselines.py
RP-MPQ (Rate-Adaptive Mixed-Precision Quantization) applied to
CRNet, CLNet, CsiNet at CR=1/4, outdoor COST2100.

Adapted from MambaIC/train_ae.py HAWQ+ILP+KL pipeline.
targets = np.arange(85.0, 95.1, 0.1)  – matches Mamba convention.

Usage:
    python rpmpq_baselines.py                  # CRNet + CLNet + CsiNet
    python rpmpq_baselines.py --plot-only      # Reload CSVs and regenerate plot

Outputs:
    MambaIC/results/csv/mp_policy_lut_crnet_cr4_out.csv
    MambaIC/results/csv/mp_policy_lut_clnet_cr4_out.csv
    MambaIC/results/csv/mp_policy_lut_csinet_cr4_out.csv
    figures/fig_rpmpq_baselines.pdf / .png
"""

import sys, os, types, re, math, copy, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.io as sio
from collections import OrderedDict, namedtuple
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pulp

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "MambaIC", "data")
RESULTS_CSV = os.path.join(ROOT, "MambaIC", "results", "csv")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(RESULTS_CSV, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CRNET_CKPT  = os.path.join(ROOT, "CRNet-master",  "checkpoints", "out_04.pth")
CLNET_CKPT  = os.path.join(ROOT, "CLNet-master",  "checkpoints", "out4.pth")
# CsiNet is NOT run here — use Python_CsiNet-master/csinet_onlytest.py (Keras model)
# CSINET_CKPT intentionally removed

REDUCTION   = 4        # CR = 1/4
ACT_BITS    = 16       # Fixed activation bits for BOPs counting
BATCH_SIZE  = 64
EVAL_N      = 500      # Samples for NMSE eval during search (use 2000 on GPU)

# ─── Legacy utils stub ───────────────────────────────────────────────────────
def _stub_utils():
    pkg       = types.ModuleType("utils")
    solver    = types.ModuleType("utils.solver")
    statics   = types.ModuleType("utils.statics")
    logger_m  = types.ModuleType("utils.logger")
    init_m    = types.ModuleType("utils.init")
    parser_m  = types.ModuleType("utils.parser")
    sched_m   = types.ModuleType("utils.scheduler")

    class _L:
        def info(self, *a, **k): pass
        def debug(self, *a, **k): pass
        def warning(self, *a, **k): pass
    _logger = _L()
    pkg.logger      = _logger
    logger_m.logger = _logger
    pkg.line_seg    = "=" * 60
    solver.Result   = namedtuple("Result", ("nmse", "rho", "epoch"), defaults=(None,)*3)

    for name, mod in [("utils", pkg), ("utils.solver", solver), ("utils.statics", statics),
                      ("utils.logger", logger_m), ("utils.init", init_m),
                      ("utils.parser", parser_m), ("utils.scheduler", sched_m)]:
        sys.modules[name] = mod

_stub_utils()

# ─── Model imports via importlib (avoids `models` namespace collision) ────────
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ─── CRNet ───────────────────────────────────────────────────────────────────
_crnet_mod = _load_module("crnet_models_crnet",
    os.path.join(ROOT, "CRNet-master", "models", "crnet.py"))
_CRNetBase = _crnet_mod.CRNet

class CRNetEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder1     = m.encoder1
        self.encoder2     = m.encoder2
        self.encoder_conv = m.encoder_conv
        self.encoder_fc   = m.encoder_fc
    def forward(self, x):
        n  = x.shape[0]
        e1 = self.encoder1(x)
        e2 = self.encoder2(x)
        out = torch.cat((e1, e2), dim=1)
        out = self.encoder_conv(out)
        return self.encoder_fc(out.view(n, -1))

class CRNetDecoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.decoder_fc      = m.decoder_fc
        self.decoder_feature = m.decoder_feature
        self.sigmoid         = m.sigmoid
    def forward(self, z):
        n   = z.shape[0]
        out = self.decoder_fc(z).view(n, 2, 32, 32)
        out = self.decoder_feature(out)
        return self.sigmoid(out)

class CRNetWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.encoder = CRNetEncoder(base)
        self.decoder = CRNetDecoder(base)
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ─── CLNet ───────────────────────────────────────────────────────────────────
_clnet_mod = _load_module("clnet_models_clnet",
    os.path.join(ROOT, "CLNet-master", "models", "clnet.py"))
_CLNetBase = _clnet_mod.CLNet
# CLNet already has self.encoder (Encoder) and self.decoder (Decoder) — no wrapper needed.

# CsiNet model classes removed — CsiNet RP-MPQ runs via csinet_onlytest.py only.

# ─── Data Loading ────────────────────────────────────────────────────────────
def load_cost2100(split="test", env="outdoor"):
    tag  = "in" if env == "indoor" else "out"
    path = os.path.join(DATA_DIR, f"DATA_H{split}{tag}.mat")
    assert os.path.exists(path), f"Data not found: {path}"
    mat  = sio.loadmat(path)
    data = mat["HT"].astype("float32")
    n    = data.shape[0]
    return torch.FloatTensor(data.reshape(n, 2, 32, 32))

def make_loader(data, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=shuffle)

# ─── NMSE Evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def eval_nmse_dB(model, loader, device):
    """NMSE in dB (de-centered: input - 0.5)"""
    model.eval()
    total_mse, total_pwr = 0.0, 0.0
    for (d,) in loader:
        d    = d.to(device)
        xhat = model(d)
        dc   = d    - 0.5
        rc   = xhat - 0.5
        diff = dc - rc
        total_mse += (diff ** 2).sum().item()
        total_pwr += (dc   ** 2).sum().item()
    return 10.0 * math.log10(total_mse / (total_pwr + 1e-30))

# ─── Quantization Helpers ────────────────────────────────────────────────────
def quantize_int_asym(w, bits):
    q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
    w_min, w_max = w.min(), w.max()
    if w_max == w_min: return w
    scale = (w_max - w_min) / (q_max - q_min)
    zp    = torch.round(q_min - w_min / scale)
    w_q   = torch.clamp(torch.round(w / scale + zp), q_min, q_max)
    return (w_q - zp) * scale

def quantize_tensor(w, bits):
    return quantize_int_asym(w, bits)

# ─── Apply / Restore Precision Policy ────────────────────────────────────────
def apply_precision_policy(model, policy, device=None):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    model_map  = {}
    for name, module in real_model.encoder.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            clean = name.replace("encoder.", "").replace("module.", "")
            model_map[clean] = module
            model_map[name]  = module

    # Group policy entries (handle _partN split FC layers)
    policy_groups = {}
    for p_key, bits in policy.items():
        clean = p_key.replace("encoder.", "").replace("module.", "").replace(".weight", "")
        m = re.search(r"(.+)_part(\d+)$", clean)
        if m:
            base, idx = m.group(1), int(m.group(2))
            if base not in policy_groups: policy_groups[base] = {}
            policy_groups[base][idx] = bits
        else:
            policy_groups[clean] = bits

    for base_name, bits_info in policy_groups.items():
        tgt = model_map.get(base_name)
        if tgt is None:
            for mname, mod in model_map.items():
                if mname.endswith(base_name): tgt = mod; break
        if tgt is None: continue
        w = tgt.weight.data

        if isinstance(bits_info, dict) and "fc" in base_name:
            n_chunks  = max(bits_info.keys()) + 1
            w_chunks  = torch.chunk(w, n_chunks, dim=0)
            q_chunks  = []
            for i, chunk in enumerate(w_chunks):
                b = bits_info.get(i, 32)
                q_chunks.append(quantize_int_asym(chunk, b) if b < 32 else chunk)
            tgt.weight.data = torch.cat(q_chunks, dim=0)
        else:
            bit = bits_info
            if bit < 32:
                tgt.weight.data = quantize_int_asym(w, bit)

def restore_fp32_weights(model, original_weights):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    with torch.no_grad():
        for name, param in real_model.encoder.named_parameters():
            if name in original_weights:
                param.data = original_weights[name].to(param.device)

# ─── ILP Candidate Generator ─────────────────────────────────────────────────
class ILPCoarseCandidateGenerator:
    def __init__(self, hawq_df, layer_params, act_bits=ACT_BITS):
        self.df          = hawq_df
        self.layer_params = layer_params
        self.act_bits    = act_bits
        self.bit_options = [16, 8, 4, 2]
        self.bops_fp32   = sum(p * 32 * 32 for p in layer_params.values())

    def solve_top_k(self, target_savings_pct, top_k=20):
        prob   = pulp.LpProblem("MP_TopK", pulp.LpMinimize)
        layers = self.df["Layer"].tolist()
        x      = pulp.LpVariable.dicts("x", (layers, self.bit_options), cat=pulp.LpBinary)

        # Objective: minimize total Omega
        prob += pulp.lpSum(x[n][b] * self.df[self.df["Layer"]==n].iloc[0][f"Omg_INT{b}"]
                           for n in layers for b in self.bit_options)
        # C1: unique bit per layer
        for n in layers:
            prob += pulp.lpSum(x[n][b] for b in self.bit_options) == 1
        # C2: BOPs budget
        limit = self.bops_fp32 * (1 - target_savings_pct / 100.0)
        prob += pulp.lpSum(x[n][b] * (self.layer_params.get(n, 0) * b * self.act_bits)
                           for n in layers for b in self.bit_options) <= limit

        solutions = []
        for _ in range(top_k):
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] != "Optimal": break
            policy, sel_vars, enc_bops = {}, [], 0
            for n in layers:
                for b in self.bit_options:
                    if pulp.value(x[n][b]) == 1:
                        policy[n] = b
                        sel_vars.append(x[n][b])
                        enc_bops += self.layer_params[n] * b * self.act_bits
            actual_saving = (1 - enc_bops / self.bops_fp32) * 100
            solutions.append({"Policy": policy,
                               "Actual_Saving": actual_saving,
                               "Total_Omega": pulp.value(prob.objective)})
            prob += pulp.lpSum(sel_vars) <= len(layers) - 1
        return solutions

# ─── KL Distributional Refinement ────────────────────────────────────────────
def kl_distributional_refinement(model, candidates, loader, device,
                                  eval_loader, original_weights):
    real_model   = model.module if isinstance(model, nn.DataParallel) else model
    kl_criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)

    # Cache FP32 latents (first 5 batches)
    fp32_zs, inputs_cache = [], []
    real_model.eval()
    with torch.no_grad():
        for i, (d,) in enumerate(loader):
            if i >= 5: break
            d = d.to(device)
            fp32_zs.append(real_model.encoder(d))
            inputs_cache.append(d)

    best_kl, best_idx = float("inf"), 0
    for idx, cand in enumerate(candidates):
        apply_precision_policy(model, cand["Policy"], device)
        kl_val = 0.0
        with torch.no_grad():
            for z_fp, inp in zip(fp32_zs, inputs_cache):
                z_q   = real_model.encoder(inp)
                p_tgt = F.softmax(z_fp.view(z_fp.size(0), -1), dim=1)
                q_in  = F.log_softmax(z_q.view(z_q.size(0), -1), dim=1)
                kl_val += kl_criterion(q_in, p_tgt).item()
        kl_val /= len(fp32_zs)
        if kl_val < best_kl:
            best_kl, best_idx = kl_val, idx
        restore_fp32_weights(model, original_weights)

    # NMSE check: ILP-best vs KL-best
    def _get_nmse(pol):
        apply_precision_policy(model, pol, device)
        val = eval_nmse_dB(model, eval_loader, device)
        restore_fp32_weights(model, original_weights)
        return val

    best = candidates[best_idx]
    best["NMSE_KL"]  = _get_nmse(best["Policy"])
    best["NMSE_ILP"] = _get_nmse(candidates[0]["Policy"])
    return best

# ─── Hessian Sensitivity Analyzer ────────────────────────────────────────────
class HessianSensitivityAnalyzer:
    def __init__(self, model, loader, device):
        self.model  = model
        self.loader = loader
        self.device = device

    def _get_gradients(self):
        self.model.zero_grad()
        try: (inputs,) = next(iter(self.loader))
        except: return None, None, None
        inputs = inputs.to(self.device)
        output = self.model(inputs)
        loss   = F.mse_loss(output, inputs)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads  = torch.autograd.grad(loss, params, create_graph=True)
        return params, grads, inputs

    def compute_importance(self, bit_widths=[16, 8, 4, 2],
                           split_threshold=20000, num_chunks=32):
        print("\n" + "="*80)
        print("[HAWQ] Computing Hessian-weighted importance (Omega) per layer")
        print("="*80)
        real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        real_model.eval()

        params, grads, inputs = self._get_gradients()
        if params is None: return pd.DataFrame()
        param_grad_map = {id(p): g for p, g in zip(params, grads)}

        results = []
        target_types = (nn.Conv2d, nn.Linear, nn.Conv1d)

        for name, module in tqdm(real_model.encoder.named_modules(), desc="Layers"):
            if not isinstance(module, target_types): continue
            w = module.weight
            if id(w) not in param_grad_map: continue
            grad_w = param_grad_map[id(w)]

            is_large = (w.numel() > split_threshold)
            if is_large:
                w_chunks = torch.chunk(w,      num_chunks, dim=0)
                g_chunks = torch.chunk(grad_w, num_chunks, dim=0)
                sub_names = [f"{name}_part{i}" for i in range(len(w_chunks))]
            else:
                w_chunks, g_chunks, sub_names = [w], [grad_w], [name]

            for ci, (sub_w, sub_g, sub_name) in enumerate(zip(w_chunks, g_chunks, sub_names)):
                v         = torch.randint_like(sub_w, high=2, device=self.device) * 2 - 1.0
                grad_v    = torch.sum(sub_g * v)
                if is_large:
                    hv_full = torch.autograd.grad(grad_v, w, retain_graph=True)[0]
                    hv      = torch.chunk(hv_full, num_chunks, dim=0)[ci]
                else:
                    hv = torch.autograd.grad(grad_v, w, retain_graph=True)[0]
                trace_val = abs(torch.sum(v * hv).item())

                row = {"Layer": sub_name, "Params": sub_w.numel(), "Trace(H)": trace_val}
                for b in bit_widths:
                    w_q        = quantize_tensor(sub_w, b)
                    l2_err     = torch.norm(sub_w - w_q, p=2).item() ** 2
                    row[f"Omg_INT{b}"] = trace_val * l2_err
                results.append(row)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("Omg_INT2", ascending=False).reset_index(drop=True)
        return df

# ─── Main RP-MPQ Pipeline (per model) ────────────────────────────────────────
def run_rpmpq(model, model_name, train_loader, eval_loader, device, extend=False, wide_step=None):
    """
    Full offline RP-MPQ: HAWQ → ILP → KL → NMSE eval.
    targets = np.arange(85.0, 95.1, 0.1)  (matches Mamba convention)
    extend=True: run 95.1-97.0% only and append to existing CSV.
    """
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    # ── Stage 0: Backup FP32 encoder weights ─────────────────────────────────
    original_enc = {name: p.clone().detach().cpu()
                    for name, p in real_model.encoder.named_parameters()}

    # ── Stage 1: HAWQ ────────────────────────────────────────────────────────
    analyzer = HessianSensitivityAnalyzer(model, train_loader, device)
    hawq_df  = analyzer.compute_importance()
    if hawq_df.empty:
        print(f"[ERROR] HAWQ returned no layers for {model_name}"); return

    layer_params = dict(zip(hawq_df["Layer"], hawq_df["Params"]))
    csv_hawq = os.path.join(RESULTS_CSV, f"hawq_{model_name}_cr4_out.csv")
    hawq_df.to_csv(csv_hawq, index=False)
    print(f"[INFO] HAWQ saved → {csv_hawq}")
    print(hawq_df.head(10).to_string(index=False))

    # ── Stage 2: ILP + KL refinement loop ────────────────────────────────────
    solver  = ILPCoarseCandidateGenerator(hawq_df, layer_params)
    if extend:
        targets = np.arange(95.1, 97.1, 0.1)
        print(f"\n[INFO] Extend mode: 95.1-97.0% | step 0.1% | {len(targets)} points")
    elif wide_step is not None:
        targets = np.arange(85.0, 98.0 + wide_step * 0.1, wide_step)
        print(f"\n[INFO] Wide Search: 85.0-98.0% | step {wide_step}% | {len(targets)} points")
    else:
        targets = np.arange(85.0, 95.1, 0.1)
        print(f"\n[INFO] Offline Search: 85.0-95.0% | step 0.1% | {len(targets)} points")
    targets = [round(float(t), 2) for t in targets]

    lut_rows = []

    for tgt in tqdm(targets, desc=f"RP-MPQ [{model_name}]"):
        # ILP top-k candidates
        candidates = solver.solve_top_k(tgt, top_k=10)
        if not candidates: continue

        # KL refinement
        real_model.load_state_dict(original_state)
        best = kl_distributional_refinement(
            model, candidates, train_loader, device, eval_loader, original_enc)

        real_model.load_state_dict(original_state)

        # Flatten policy into row
        row = {k: v for k, v in best["Policy"].items()}
        row["Target_Saving"] = tgt
        row["Actual_Saving"] = best["Actual_Saving"]
        row["Total_Omega"]   = best["Total_Omega"]
        row["NMSE_dB"]       = best["NMSE_KL"]
        row["NMSE_dB_ILP"]   = best["NMSE_ILP"]
        lut_rows.append(row)

    if not lut_rows:
        print(f"[WARN] No valid policies found for {model_name}"); return

    df_new = pd.DataFrame(lut_rows).sort_values("Actual_Saving")

    # ── Combine with existing CSV if extending ────────────────────────────────
    csv_out = os.path.join(RESULTS_CSV, f"mp_policy_lut_{model_name}_cr4_out.csv")
    if extend and os.path.exists(csv_out):
        df_exist = pd.read_csv(csv_out)
        df = pd.concat([df_exist, df_new], ignore_index=True).sort_values("Actual_Saving")
    else:
        df = df_new

    # ── Monotonic smoothing ──────────────────────────────────────────────────
    vals, best_so_far, smoothed = df["NMSE_dB"].tolist(), float("inf"), []
    for v in reversed(vals):
        if v < best_so_far: best_so_far = v
        smoothed.append(best_so_far)
    df["NMSE_dB"] = smoothed[::-1]

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(csv_out, index=False)
    print(f"\n[INFO] LUT saved → {csv_out}  ({len(df)} rows)")
    print(df[["Target_Saving","Actual_Saving","NMSE_dB"]].head(10).to_string(index=False))
    return df

# ─── CsiNet Training ─────────────────────────────────────────────────────────
def train_csinet(n_epochs, device):
    """Train ModularCsiNet (csinet+csinet, dim=512) and save to CSINET_CKPT."""
    print(f"\n[TRAIN] ModularCsiNet outdoor dim=512 for {n_epochs} epochs ...")
    x_train = load_cost2100("train", "outdoor")
    x_val   = load_cost2100("val",   "outdoor")

    model = ModularCsiNet(encoded_dim=2048 // REDUCTION).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    bs    = 200

    # DataLoader for efficiency
    from torch.utils.data import TensorDataset, DataLoader
    tr_loader = DataLoader(TensorDataset(x_train), batch_size=bs, shuffle=True)
    va_loader = DataLoader(TensorDataset(x_val),   batch_size=bs, shuffle=False)

    best_val, best_sd = float("inf"), None
    for ep in range(1, n_epochs + 1):
        model.train()
        for (xb,) in tr_loader:
            xb   = xb.to(device)
            loss = F.mse_loss(model(xb), xb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(F.mse_loss(model(xb.to(device)), xb.to(device)).item()
                          for (xb,) in va_loader) / len(va_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0 or ep == 1:
            print(f"  Epoch {ep:4d}/{n_epochs} | val_loss={val_loss:.6f}")

    model.load_state_dict(best_sd)
    torch.save({"state_dict": model.state_dict(), "epoch": n_epochs}, CSINET_CKPT)
    print(f"[TRAIN] Saved -> {CSINET_CKPT}  (best val_loss={best_val:.6f})")
    return model

# ─── Plotting ────────────────────────────────────────────────────────────────
# Uniform quant BOPs savings per model (CR=1/4, based on encoder params)
# CRNet enc params: encoder1+encoder2+encoder_conv+encoder_fc
# Savings formula: (1 - bits/32) * 100  (per-layer all same bits)
UNIFORM_DATA = {
    # model : {INT: (BOPs_saving%, NMSE_dB)}  – CR=1/4 outdoor, ACT=INT16
    "CRNet":  {16: (75.00, -12.71), 8: (87.50,  -3.57), 4: (93.75, 10.36), 2: (96.88,  0.09)},
    "CLNet":  {16: (75.00, -12.82), 8: (87.50,   0.15), 4: (93.75, 23.36), 2: (96.88, 25.75)},
    "CsiNet": {16: (75.00,  -8.74), 8: (87.50,   1.46), 4: (93.75, 19.40), 2: (96.88, 22.32)},
}

MODEL_STYLES = {
    "CRNet":  ("s", "#d62728"),   # red square
    "CLNet":  ("^", "#1f77b4"),   # blue triangle
    "CsiNet": ("o", "#2ca02c"),   # green circle
}

def plot_comparison(csv_paths):
    """
    Generate unified RP-MPQ vs Uniform Quantization comparison plot.
    csv_paths: list of (model_name, csv_file)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing {csv_path}, skipping {model_name}")
            continue

        df = pd.read_csv(csv_path)
        marker, color = MODEL_STYLES.get(model_name, ("D", "gray"))

        # RP-MPQ Pareto curve
        ax.plot(df["Actual_Saving"], df["NMSE_dB"],
                color=color, lw=2, label=f"{model_name} RP-MPQ",
                marker=marker, markersize=3, markevery=5)

        # Uniform quant points
        uni = UNIFORM_DATA.get(model_name, {})
        for bits in [16, 8, 4]:
            sv, nmse = uni.get(bits, (None, None))
            if sv is not None and nmse is not None:
                ax.scatter(sv, nmse, color=color, marker="x", s=60, zorder=5,
                           label=f"{model_name} INT{bits}" if bits == 16 else "")

    ax.set_xlabel("BOPs Saving (%)", fontsize=12)
    ax.set_ylabel("NMSE (dB)", fontsize=12)
    ax.set_title("RP-MPQ vs Uniform Quantization (CR=1/4, Outdoor)", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlim(70, 98)

    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"fig_rpmpq_baselines.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved → {out}")
    plt.close(fig)

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip RP-MPQ, reload existing CSVs and regenerate plot")
    parser.add_argument("--extend", action="store_true",
                        help="Extend existing CSV: run 95.1-97.0%% and append")
    parser.add_argument("--wide_step", type=float, default=None,
                        help="Wide sweep step size (e.g. 0.05). Overrides default 0.1 step, range 85-98%%")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    if args.plot_only:
        csv_paths = [
            ("CRNet",  os.path.join(RESULTS_CSV, "mp_policy_lut_crnet_cr4_out.csv")),
            ("CLNet",  os.path.join(RESULTS_CSV, "mp_policy_lut_clnet_cr4_out.csv")),
            ("CsiNet", os.path.join(RESULTS_CSV, "mp_policy_lut_csinet_cr4_out.csv")),
        ]
        plot_comparison(csv_paths)
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[INFO] Loading COST2100 outdoor test data …")
    test_data  = load_cost2100("test",  "outdoor")
    train_data = load_cost2100("train", "outdoor")

    # Eval loader (subset of EVAL_N for speed)
    eval_idx    = np.linspace(0, len(test_data)-1, EVAL_N, dtype=int)
    eval_loader = DataLoader(TensorDataset(test_data[eval_idx]),
                             batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(TensorDataset(train_data[:2000]),
                              batch_size=BATCH_SIZE, shuffle=True)

    csv_paths = []

    # ── CRNet ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  CRNet  (reduction=4, outdoor)")
    print("="*60)
    base_cr = _CRNetBase(reduction=REDUCTION)
    sd_cr   = torch.load(CRNET_CKPT, map_location="cpu", weights_only=True)
    base_cr.load_state_dict(sd_cr["state_dict"])
    model_cr = CRNetWrapper(base_cr).to(device).eval()

    fp32_nmse = eval_nmse_dB(model_cr, eval_loader, device)
    print(f"  FP32 NMSE: {fp32_nmse:.2f} dB")

    df_cr = run_rpmpq(model_cr, "crnet", train_loader, eval_loader, device, extend=args.extend, wide_step=args.wide_step)
    if df_cr is not None:
        csv_paths.append(("CRNet", os.path.join(RESULTS_CSV, "mp_policy_lut_crnet_cr4_out.csv")))

    # ── CLNet ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  CLNet  (reduction=4, outdoor)")
    print("="*60)
    base_cl = _CLNetBase(reduction=REDUCTION)

    # CLNet checkpoint requires utils stubs (already applied)
    sd_cl = torch.load(CLNET_CKPT, map_location="cpu", weights_only=False)
    base_cl.load_state_dict(sd_cl["state_dict"])
    model_cl = base_cl.to(device).eval()   # already has .encoder / .decoder

    fp32_nmse = eval_nmse_dB(model_cl, eval_loader, device)
    print(f"  FP32 NMSE: {fp32_nmse:.2f} dB")

    df_cl = run_rpmpq(model_cl, "clnet", train_loader, eval_loader, device, extend=args.extend, wide_step=args.wide_step)
    if df_cl is not None:
        csv_paths.append(("CLNet", os.path.join(RESULTS_CSV, "mp_policy_lut_clnet_cr4_out.csv")))

    # ── CsiNet: handled by Python_CsiNet-master/csinet_onlytest.py (Keras) ─────
    # DO NOT run CsiNet RP-MPQ here — the Keras checkpoint is in csinet_onlytest.py.
    # Results are pre-saved in mp_policy_lut_csinet_cr4_out.csv.
    print("\n[SKIP] CsiNet: use Python_CsiNet-master/csinet_onlytest.py instead.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if csv_paths:
        plot_comparison(csv_paths)

    print("\n[DONE] All RP-MPQ results saved.")

if __name__ == "__main__":
    main()
