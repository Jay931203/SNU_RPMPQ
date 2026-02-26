#!/usr/bin/env python3
"""
rpmpq_act8.py  —  RP-MPQ with ACT_BITS=8  (vs. original ACT_BITS=16)

Reuses existing HAWQ CSVs; only re-runs ILP + KL + NMSE with act_bits=8.
Mamba requires GPU — run separately and add results manually.

Outputs:
    MambaIC/results/csv/mp_policy_lut_{crnet,clnet,csinet}_cr4_out_a8.csv
    figures/fig_rpmpq_act8.pdf / .png

Usage:
    python rpmpq_act8.py
    python rpmpq_act8.py --plot-only
"""

import sys, os, types, re, math, copy, argparse
from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pulp

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "MambaIC", "data")
RESULTS_CSV = os.path.join(ROOT, "MambaIC", "results", "csv")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(RESULTS_CSV, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CRNET_CKPT  = os.path.join(ROOT, "CRNet-master",  "checkpoints", "out_04.pth")
CLNET_CKPT  = os.path.join(ROOT, "CLNet-master",  "checkpoints", "out4.pth")
# CsiNet: handled by Python_CsiNet-master/csinet_onlytest.py only

REDUCTION  = 4
ACT_BITS   = 8        # ← KEY CHANGE: 16 → 8
BATCH_SIZE = 64
EVAL_N     = 500

# min feasible saving with act8: all-INT16 → 1 - 16*8/(32*32) = 87.5%
# max: all-INT2 → 1 - 2*8/(32*32) = 98.4%
TARGETS = [round(float(t), 1) for t in np.arange(88.0, 97.1, 0.1)]

# ─── Utils stub ───────────────────────────────────────────────────────────────
def _stub_utils():
    pkg = types.ModuleType("utils"); solver = types.ModuleType("utils.solver")
    statics = types.ModuleType("utils.statics"); logger_m = types.ModuleType("utils.logger")
    init_m  = types.ModuleType("utils.init");   parser_m = types.ModuleType("utils.parser")
    sched_m = types.ModuleType("utils.scheduler")
    class _L:
        def info(self, *a, **k): pass
        def debug(self, *a, **k): pass
        def warning(self, *a, **k): pass
    pkg.logger = logger_m.logger = _L()
    pkg.line_seg = "=" * 60
    solver.Result = namedtuple("Result", ("nmse","rho","epoch"), defaults=(None,)*3)
    for name, mod in [("utils",pkg),("utils.solver",solver),("utils.statics",statics),
                      ("utils.logger",logger_m),("utils.init",init_m),
                      ("utils.parser",parser_m),("utils.scheduler",sched_m)]:
        sys.modules[name] = mod

_stub_utils()

# ─── Model imports ────────────────────────────────────────────────────────────
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# CRNet
_crnet_mod = _load_module("crnet_models_a8",
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
        n = x.shape[0]
        e1 = self.encoder1(x); e2 = self.encoder2(x)
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
        n = z.shape[0]
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

# CLNet — already has .encoder / .decoder
_clnet_mod = _load_module("clnet_models_a8",
    os.path.join(ROOT, "CLNet-master", "models", "clnet.py"))
_CLNetBase = _clnet_mod.CLNet

# CsiNet model classes removed — use Python_CsiNet-master/csinet_onlytest.py only.

# ─── Data ─────────────────────────────────────────────────────────────────────
def load_cost2100(split="test", env="outdoor"):
    tag  = "in" if env == "indoor" else "out"
    path = os.path.join(DATA_DIR, f"DATA_H{split}{tag}.mat")
    assert os.path.exists(path), f"Data not found: {path}"
    mat  = sio.loadmat(path)
    data = mat["HT"].astype("float32")
    n    = data.shape[0]
    return torch.FloatTensor(data.reshape(n, 2, 32, 32))

# ─── NMSE eval (identical to rpmpq_baselines.py) ─────────────────────────────
@torch.no_grad()
def eval_nmse_dB(model, loader, device):
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

# ─── Quantization (identical to rpmpq_baselines.py) ──────────────────────────
def quantize_int_asym(w, bits):
    q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
    w_min, w_max = w.min(), w.max()
    if w_max == w_min: return w
    scale = (w_max - w_min) / (q_max - q_min)
    zp    = torch.round(q_min - w_min / scale)
    w_q   = torch.clamp(torch.round(w / scale + zp), q_min, q_max)
    return (w_q - zp) * scale

def apply_precision_policy(model, policy, device=None):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    model_map  = {}
    for name, module in real_model.encoder.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            clean = name.replace("encoder.", "").replace("module.", "")
            model_map[clean] = module
            model_map[name]  = module

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
            n_chunks = max(bits_info.keys()) + 1
            w_chunks = torch.chunk(w, n_chunks, dim=0)
            q_chunks = []
            for i, chunk in enumerate(w_chunks):
                b = bits_info.get(i, 32)
                q_chunks.append(quantize_int_asym(chunk, b) if b < 32 else chunk)
            tgt.weight.data = torch.cat(q_chunks, dim=0)
        else:
            if bits_info < 32:
                tgt.weight.data = quantize_int_asym(w, bits_info)

def restore_fp32_weights(model, original_weights):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    with torch.no_grad():
        for name, param in real_model.encoder.named_parameters():
            if name in original_weights:
                param.data = original_weights[name].to(param.device)

# ─── ILP with ACT_BITS=8 ──────────────────────────────────────────────────────
class ILPSolverAct8:
    def __init__(self, hawq_df, layer_params):
        self.df           = hawq_df
        self.layer_params = layer_params
        self.act_bits     = ACT_BITS          # 8
        self.bit_options  = [16, 8, 4, 2]
        self.bops_fp32    = sum(p * 32 * 32 for p in layer_params.values())

    def solve_top_k(self, target_savings_pct, top_k=20):
        prob   = pulp.LpProblem("MP_Act8", pulp.LpMinimize)
        layers = self.df["Layer"].tolist()
        x      = pulp.LpVariable.dicts("x", (layers, self.bit_options), cat=pulp.LpBinary)
        prob  += pulp.lpSum(x[n][b] * self.df[self.df["Layer"]==n].iloc[0][f"Omg_INT{b}"]
                            for n in layers for b in self.bit_options)
        for n in layers:
            prob += pulp.lpSum(x[n][b] for b in self.bit_options) == 1
        limit = self.bops_fp32 * (1 - target_savings_pct / 100.0)
        prob  += pulp.lpSum(x[n][b] * (self.layer_params.get(n, 0) * b * self.act_bits)
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

# ─── KL refinement (identical to rpmpq_baselines.py) ─────────────────────────
def kl_distributional_refinement(model, candidates, loader, device,
                                  eval_loader, original_weights):
    real_model   = model.module if isinstance(model, nn.DataParallel) else model
    kl_criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
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

    def _get_nmse(pol):
        apply_precision_policy(model, pol, device)
        val = eval_nmse_dB(model, eval_loader, device)
        restore_fp32_weights(model, original_weights)
        return val

    best = candidates[best_idx]
    best["NMSE_KL"]  = _get_nmse(best["Policy"])
    best["NMSE_ILP"] = _get_nmse(candidates[0]["Policy"])
    return best

# ─── Main RP-MPQ (load HAWQ CSV, skip Hessian) ───────────────────────────────
def run_rpmpq_act8(model, model_name, train_loader, eval_loader, device):
    real_model     = model.module if isinstance(model, nn.DataParallel) else model
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    original_enc   = {n: p.clone().detach().cpu()
                      for n, p in real_model.encoder.named_parameters()}

    # Load existing HAWQ CSV (no Hessian recomputation)
    hawq_csv = os.path.join(RESULTS_CSV, f"hawq_{model_name}_cr4_out.csv")
    if not os.path.exists(hawq_csv):
        print(f"[ERROR] HAWQ CSV not found: {hawq_csv}")
        print(f"        Run rpmpq_baselines.py first.")
        return None
    hawq_df = pd.read_csv(hawq_csv)
    print(f"[INFO] Loaded HAWQ: {hawq_csv}  ({len(hawq_df)} layers)")

    layer_params = dict(zip(hawq_df["Layer"], hawq_df["Params"]))
    solver       = ILPSolverAct8(hawq_df, layer_params)

    lut_rows = []
    print(f"[INFO] Search: {TARGETS[0]}-{TARGETS[-1]}% | {len(TARGETS)} pts | ACT_BITS={ACT_BITS}")

    for tgt in tqdm(TARGETS, desc=f"RP-MPQ-A8 [{model_name}]"):
        candidates = solver.solve_top_k(tgt, top_k=10)
        if not candidates: continue
        real_model.load_state_dict(original_state)
        best = kl_distributional_refinement(
            model, candidates, train_loader, device, eval_loader, original_enc)
        real_model.load_state_dict(original_state)

        row = {k: v for k, v in best["Policy"].items()}
        row["Target_Saving"] = tgt
        row["Actual_Saving"] = best["Actual_Saving"]
        row["Total_Omega"]   = best["Total_Omega"]
        row["NMSE_dB"]       = best["NMSE_KL"]
        row["NMSE_dB_ILP"]   = best["NMSE_ILP"]
        lut_rows.append(row)

    if not lut_rows:
        print(f"[WARN] No valid policies for {model_name}"); return None

    df = pd.DataFrame(lut_rows).sort_values("Actual_Saving")
    vals, best_so_far, smoothed = df["NMSE_dB"].tolist(), float("inf"), []
    for v in reversed(vals):
        if v < best_so_far: best_so_far = v
        smoothed.append(best_so_far)
    df["NMSE_dB"] = smoothed[::-1]

    csv_out = os.path.join(RESULTS_CSV, f"mp_policy_lut_{model_name}_cr4_out_a8.csv")
    df.to_csv(csv_out, index=False)
    print(f"[INFO] Saved → {csv_out}  ({len(df)} rows)")
    print(df[["Target_Saving","Actual_Saving","NMSE_dB"]].head(10).to_string(index=False))
    return df

# ─── Plotting ─────────────────────────────────────────────────────────────────
# Uniform quant points re-calculated with ACT_BITS=8:
#   saving = 1 - w*8/(32*32): INT16→87.5%, INT8→93.75%, INT4→96.875%
UNIFORM_A8 = {
    "CRNet":  {16: (87.50, -12.71), 8: (93.75, -3.57),  4: (96.88, 10.36)},
    "CLNet":  {16: (87.50, -12.82), 8: (93.75,  0.15),  4: (96.88, 23.36)},
    "CsiNet": {16: (87.50,  -8.74), 8: (93.75,  1.46),  4: (96.88, 19.40)},
}
MODEL_STYLES = {
    "CRNet":  ("s", "#d62728"),
    "CLNet":  ("^", "#1f77b4"),
    "CsiNet": ("o", "#2ca02c"),
}

def plot_results(csv_paths):
    matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for model_name, csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing {csv_path}"); continue
        df = pd.read_csv(csv_path).sort_values("Actual_Saving").reset_index(drop=True)
        marker, color = MODEL_STYLES.get(model_name, ("D", "gray"))

        # CsiNet CSV (from csinet_onlytest.py) uses MSE_Smoothed; others use NMSE_dB
        if "MSE_Smoothed" in df.columns:
            nmse_vals = 10 * np.log10(df["MSE_Smoothed"].values + 1e-15)
        else:
            nmse_vals = df["NMSE_dB"].values

        ax.plot(df["Actual_Saving"], nmse_vals,
                color=color, lw=2, marker=marker, markersize=3, markevery=5,
                label=f"{model_name} RP-MPQ (act8)")

        for bits, (sv, nmse) in UNIFORM_A8.get(model_name, {}).items():
            ax.scatter(sv, nmse, color=color, marker="x", s=60, zorder=5)
            ax.annotate(f"INT{bits}", xy=(sv, nmse), xytext=(3, 4),
                        textcoords="offset points", fontsize=8, color=color)

    ax.set_xlabel("BOPs Saving vs. FP32 (%)")
    ax.set_ylabel("NMSE (dB)")
    ax.legend(fontsize=9, ncol=1)
    ax.grid(True, ls="--", alpha=0.45)
    ax.set_xlim(85, 99)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"fig_rpmpq_act8.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved → {out}")
    plt.close(fig)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args   = parser.parse_args()
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}  |  ACT_BITS={ACT_BITS}")

    if args.plot_only:
        plot_results([
            ("CRNet",  os.path.join(RESULTS_CSV, "mp_policy_lut_crnet_cr4_out_a8.csv")),
            ("CLNet",  os.path.join(RESULTS_CSV, "mp_policy_lut_clnet_cr4_out_a8.csv")),
            ("CsiNet", os.path.join(RESULTS_CSV, "mp_policy_lut_csinet_cr4_out_a8.csv")),
        ])
        return

    test_data  = load_cost2100("test",  "outdoor")
    train_data = load_cost2100("train", "outdoor")
    eval_idx   = np.linspace(0, len(test_data)-1, EVAL_N, dtype=int)
    eval_loader  = DataLoader(TensorDataset(test_data[eval_idx]),  batch_size=BATCH_SIZE)
    train_loader = DataLoader(TensorDataset(train_data[:2000]),    batch_size=BATCH_SIZE, shuffle=True)

    csv_paths = []

    # CRNet
    print("\n" + "="*60 + "\n  CRNet\n" + "="*60)
    base_cr = _CRNetBase(reduction=REDUCTION)
    sd_cr   = torch.load(CRNET_CKPT, map_location="cpu", weights_only=True)
    base_cr.load_state_dict(sd_cr["state_dict"])
    model_cr = CRNetWrapper(base_cr).to(device).eval()
    print(f"  FP32 NMSE: {eval_nmse_dB(model_cr, eval_loader, device):.2f} dB")
    df_cr = run_rpmpq_act8(model_cr, "crnet", train_loader, eval_loader, device)
    if df_cr is not None:
        csv_paths.append(("CRNet", os.path.join(RESULTS_CSV, "mp_policy_lut_crnet_cr4_out_a8.csv")))

    # CLNet
    print("\n" + "="*60 + "\n  CLNet\n" + "="*60)
    base_cl = _CLNetBase(reduction=REDUCTION)
    sd_cl   = torch.load(CLNET_CKPT, map_location="cpu", weights_only=False)
    base_cl.load_state_dict(sd_cl["state_dict"])
    model_cl = base_cl.to(device).eval()
    print(f"  FP32 NMSE: {eval_nmse_dB(model_cl, eval_loader, device):.2f} dB")
    df_cl = run_rpmpq_act8(model_cl, "clnet", train_loader, eval_loader, device)
    if df_cl is not None:
        csv_paths.append(("CLNet", os.path.join(RESULTS_CSV, "mp_policy_lut_clnet_cr4_out_a8.csv")))

    # CsiNet: handled by Python_CsiNet-master/csinet_onlytest.py (Keras)
    # DO NOT run here — wrong model.
    print("\n[SKIP] CsiNet: use Python_CsiNet-master/csinet_onlytest.py instead.")

    if csv_paths:
        plot_results(csv_paths)

    print("\n[DONE]")
    print("  CSVs  → MambaIC/results/csv/*_a8.csv")
    print("  Figure → figures/fig_rpmpq_act8.pdf / .png")

if __name__ == "__main__":
    main()
