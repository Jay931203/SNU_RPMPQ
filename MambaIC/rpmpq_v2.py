#!/usr/bin/env python3
"""
rpmpq_v2.py  --  RP-MPQ v2: Rate-adaptive Precision Mixed-Precision Quantization
                  with zeta-proxy state descriptor and SNR as second state dimension.

Single-file implementation.  Run steps independently or together:

    python rpmpq_v2.py --step collect   # GPU: per-sample data collection
    python rpmpq_v2.py --step build     # CPU: importance surface + policy map
    python rpmpq_v2.py --step eval      # GPU: evaluate all schemes
    python rpmpq_v2.py --step all       # everything in sequence

Author : Hyunjae Park (SNU ECE)
Date   : 2026-03
"""

from __future__ import annotations

import os, sys, ast, re, copy, argparse, warnings, itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_ae import (
    apply_precision_policy,
    quantize_feedback_torch,
    calculate_su_miso_rate_mrt,
    NMSELoss,
    CsiDataset,
    compute_hoyer_sparsity_bins,
    restore_fp32_weights,
    calculate_nmse_db,
    quantize_int_asym,
)
from ModularModels import ModularAE

# ── output directories ───────────────────────────────────────────────────────
RESULTS_CSV  = os.path.join(PROJECT_ROOT, "results", "csv")
RESULTS_PLOT = os.path.join(PROJECT_ROOT, "results", "plots")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "..", "figures")
for d in (RESULTS_CSV, RESULTS_PLOT, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

# ── plotting style ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PAPER_FIG_STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
}


# =============================================================================
# SECTION 1 : State Descriptor  (zeta computation)
# =============================================================================

def build_kernel(n: int, alpha: float = 1.0) -> np.ndarray:
    """Build soft-locality kernel K of size n x n (row-normalised).

    K[i,i'] = (1 + |i - i'|)^{-alpha}  /  sum_k (1 + |i - k|)^{-alpha}

    Power-law decay matching Dirichlet kernel analysis (paper Eq. 9).
    Each row sums to 1 (row-stochastic).
    """
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    raw = (1.0 + diff) ** (-alpha)
    return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float64)


def _energy_map(x_np: np.ndarray) -> np.ndarray:
    """Convert (2, 32, 32) normalised [0,1] CSI to energy map (32, 32).

    Centre by subtracting 0.5 (same convention as NMSE computation),
    then compute per-element power  E[i,j] = Re^2 + Im^2.
    """
    xc = x_np - 0.5                       # (2, 32, 32)
    return xc[0] ** 2 + xc[1] ** 2         # (32, 32)


def compute_zeta_full(x_np: np.ndarray,
                      K_d: np.ndarray,
                      K_a: np.ndarray) -> float:
    """Full zeta: 1 - Tr(P^T K_d P K_a)  (paper Eq. 10).

    P = normalised energy map (sums to 1).
    Small zeta = compact (encoder-friendly), large zeta = diffuse.
    """
    E = _energy_map(x_np)                  # (32, 32)
    total = E.sum() + 1e-12
    P = E / total                          # normalised energy map
    return float(1.0 - np.trace(P.T @ K_d @ P @ K_a))


def compute_zeta_proxy(x_np: np.ndarray,
                       K_d: np.ndarray,
                       K_a: np.ndarray,
                       lambda_d: float = 0.5,
                       lambda_a: float = 0.5) -> float:
    """Proxy zeta: 1 - lambda_d * c_d - lambda_a * c_a  (paper Eq. 13).

    c_d = p_d^T K_d p_d   (delay-axis compactness)
    c_a = p_a^T K_a p_a   (angular-axis compactness)
    p_d, p_a are marginals of the normalised energy map P.

    Small zeta = compact (encoder-friendly), large zeta = diffuse.
    O(N^2) complexity — 16x cheaper than full zeta.
    """
    E = _energy_map(x_np)                  # (N_d, N_a) = (32, 32)
    total = E.sum() + 1e-12
    P = E / total                          # normalised energy map

    # Per-axis marginals (probability vectors, sum to 1)
    p_d = P.sum(axis=1)                    # (N_d,) — delay marginal
    p_a = P.sum(axis=0)                    # (N_a,) — angular marginal

    # Per-axis compactness scores
    c_d = p_d @ K_d @ p_d                  # scalar
    c_a = p_a @ K_a @ p_a                  # scalar

    return float(1.0 - lambda_d * c_d - lambda_a * c_a)


def compute_all_zeta(dataset,
                     K_d: np.ndarray,
                     K_a: np.ndarray,
                     use_proxy: bool = True,
                     lambda_d: float = 0.5,
                     lambda_a: float = 0.5) -> np.ndarray:
    """Compute zeta values for every sample in *dataset*.

    Returns 1-D array of length len(dataset).
    """
    fn = compute_zeta_proxy if use_proxy else compute_zeta_full
    N = len(dataset)
    zetas = np.empty(N, dtype=np.float64)
    for i in tqdm(range(N), desc="Computing zeta", leave=False):
        x = dataset[i].numpy()             # (2, 32, 32)
        if use_proxy:
            zetas[i] = fn(x, K_d, K_a, lambda_d, lambda_a)
        else:
            zetas[i] = fn(x, K_d, K_a)
    return zetas


# =============================================================================
# SECTION 2 : Offline Stage  --  Data Collection  (needs GPU)
# =============================================================================

def _compute_per_sample_nmse_rate(model, loader, device, norm_params,
                                  snr_list, aq_bits=0):
    """Run model inference and return per-sample NMSE (linear) and rates.

    Returns:
        nmse_linear : list[float]   -- linear-scale NMSE for each sample
        rates       : dict[snr -> list[float]]
    """
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params

    nmse_linear_all: list[float] = []
    rates_all: dict[int, list] = {snr: [] for snr in snr_list}

    with torch.no_grad():
        for batch in loader:
            d = batch.to(device)
            z = real_model.encoder(d)
            if aq_bits > 0:
                z = quantize_feedback_torch(z, aq_bits)
            x_hat = real_model.decoder(z)

            h_true = (d * range_val) + min_val - 0.5
            h_hat  = (x_hat * range_val) + min_val - 0.5

            error = torch.sum((h_true - h_hat) ** 2, dim=[1, 2, 3])
            power = torch.sum(h_true ** 2, dim=[1, 2, 3])
            nmse_l = (error / (power + 1e-9)).cpu().numpy()
            nmse_linear_all.extend(nmse_l.tolist())

            for snr in snr_list:
                r = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
                rates_all[snr].extend(r.cpu().numpy().tolist())

    return nmse_linear_all, rates_all


def _compute_perfect_rates(loader, device, norm_params, snr_list):
    """Perfect-CSI rates (upper bound)."""
    min_val, range_val = norm_params
    rates = {snr: [] for snr in snr_list}
    with torch.no_grad():
        for batch in loader:
            d = batch.to(device)
            h_true = (d * range_val) + min_val - 0.5
            for snr in snr_list:
                r = calculate_su_miso_rate_mrt(h_true, h_true, snr, device)
                rates[snr].extend(r.cpu().numpy().tolist())
    return rates


def collect_per_sample_data(model, test_loader, policy_lut_path, device,
                            norm_params, snr_list, aq_bits=0,
                            output_csv=None, zeta_csv=None,
                            K_d=None, K_a=None,
                            lambda_d=0.5, lambda_a=0.5):
    """SECTION 2 main entry: collect per-sample performance under each policy.

    For every policy in the LUT and every test sample, record:
        [sample_idx, policy_id, saving, nmse_linear, rate_10, rate_20, rate_30]

    Also compute and save zeta_proxy for each sample.
    """
    if output_csv is None:
        output_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_collected.csv")
    if zeta_csv is None:
        zeta_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_zeta.csv")

    # ---- load LUT ----
    df_lut = pd.read_csv(policy_lut_path)
    if isinstance(df_lut["Policy"].iloc[0], str):
        df_lut["Policy"] = df_lut["Policy"].apply(ast.literal_eval)

    real_model = model.module if isinstance(model, nn.DataParallel) else model
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    N_samples = len(test_loader.dataset)

    # ---- 1. zeta computation ----
    if os.path.exists(zeta_csv):
        print(f"[INFO] Loading cached zeta values from {zeta_csv}")
        zeta_df = pd.read_csv(zeta_csv)
        zeta_vals = zeta_df["zeta_proxy"].values
    else:
        print("[INFO] Computing zeta_proxy for all test samples ...")
        if K_d is None:
            K_d = build_kernel(32, alpha=1.0)
        if K_a is None:
            K_a = build_kernel(32, alpha=1.0)
        zeta_vals = compute_all_zeta(test_loader.dataset, K_d, K_a,
                                     use_proxy=True,
                                     lambda_d=lambda_d, lambda_a=lambda_a)
        zeta_df = pd.DataFrame({
            "sample_idx": np.arange(N_samples),
            "zeta_proxy": zeta_vals,
        })
        zeta_df.to_csv(zeta_csv, index=False)
        print(f"[INFO] Saved zeta values -> {zeta_csv}")

    # ---- 2. Hoyer sparsity (for baseline comparison) ----
    _, hoyer_vals = compute_hoyer_sparsity_bins(test_loader.dataset, device)

    # ---- 3. Perfect-CSI rates ----
    print("[INFO] Computing perfect-CSI rates ...")
    perfect_rates = _compute_perfect_rates(test_loader, device, norm_params, snr_list)
    perf_df = pd.DataFrame({
        "sample_idx": np.arange(N_samples),
        **{f"r_perf_{snr}": perfect_rates[snr] for snr in snr_list},
    })
    perf_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    perf_df.to_csv(perf_csv, index=False)

    # ---- 4. FP32 reference ----
    print("[INFO] Computing FP32 reference ...")
    real_model.load_state_dict(original_state)
    fp32_nmse, fp32_rates = _compute_per_sample_nmse_rate(
        model, test_loader, device, norm_params, snr_list, aq_bits)

    fp32_df = pd.DataFrame({
        "sample_idx": np.arange(N_samples),
        "nmse_linear_fp32": fp32_nmse,
        "hoyer": hoyer_vals,
        "zeta_proxy": zeta_vals,
        **{f"rate_fp32_{snr}": fp32_rates[snr] for snr in snr_list},
    })
    fp32_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_fp32_ref.csv")
    fp32_df.to_csv(fp32_csv, index=False)

    # ---- 5. Per-policy sweep ----
    records: list[dict] = []
    n_policies = len(df_lut)
    print(f"[INFO] Sweeping {n_policies} policies x {N_samples} samples ...")

    for p_idx, row in tqdm(df_lut.iterrows(), total=n_policies, desc="Policy sweep"):
        policy = row["Policy"]
        saving = row["Actual_Saving"]

        # Reset weights then apply this policy
        real_model.load_state_dict(original_state)
        apply_precision_policy(model, policy, device)

        nmse_l, rates = _compute_per_sample_nmse_rate(
            model, test_loader, device, norm_params, snr_list, aq_bits)

        for i in range(N_samples):
            entry = {
                "sample_idx": i,
                "policy_id": p_idx,
                "saving": saving,
                "nmse_linear": nmse_l[i],
            }
            for snr in snr_list:
                entry[f"rate_{snr}"] = rates[snr][i]
            records.append(entry)

    # Restore FP32
    real_model.load_state_dict(original_state)

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"[INFO] Collected data saved -> {output_csv}  ({len(df_out)} rows)")
    return df_out


# =============================================================================
# SECTION 3 : Offline Stage  --  Importance Surface Construction  (no GPU)
# =============================================================================

def compute_shortfall(r_policy: np.ndarray,
                      r_ref: np.ndarray,
                      gamma: float,
                      eta_r: float = 1e-6) -> np.ndarray:
    """Shortfall (violation cost) V_i(pi; gamma).

    V = [gamma * r_ref - r_policy]_+ / (gamma * r_ref + eta_r)
    """
    gap = gamma * r_ref - r_policy
    return np.maximum(gap, 0.0) / (gamma * r_ref + eta_r)


def _compute_bops_saving(policy: dict, layer_params: dict,
                         act_bits: int = 16) -> float:
    """Compute BOPs saving percentage for a given policy."""
    bops_fp32 = sum(p * 32 * 32 for p in layer_params.values())
    if bops_fp32 == 0:
        return 0.0

    # Rebuild policy_groups to handle fc_part splits
    policy_groups: dict = {}
    for p_key, bits in policy.items():
        clean = p_key.replace("encoder.", "").replace("module.", "").replace(".weight", "")
        match = re.search(r"(.+)_part(\d+)$", clean)
        if match:
            base = match.group(1)
            idx = int(match.group(2))
            if base not in policy_groups:
                policy_groups[base] = {}
            policy_groups[base][idx] = bits
        else:
            policy_groups[clean] = bits

    bops = 0
    for name, bits_info in policy_groups.items():
        if isinstance(bits_info, dict):
            # split FC — each part has its own param count
            for part_idx, b in bits_info.items():
                part_key = f"{name}_part{part_idx}"
                params = layer_params.get(part_key, 0)
                bops += params * b * act_bits
        else:
            params = layer_params.get(name, 0)
            bops += params * bits_info * act_bits

    return (1.0 - bops / bops_fp32) * 100.0


def _get_block_names_from_lut(policy_lut_path: str) -> list[str]:
    """Extract ordered list of block (layer) names from the first policy."""
    df = pd.read_csv(policy_lut_path)
    pol = df["Policy"].iloc[0]
    if isinstance(pol, str):
        pol = ast.literal_eval(pol)
    return list(pol.keys())


def _compute_kappa_table(block_names: list[str],
                         layer_params: dict,
                         bit_options: list[int],
                         act_bits: int = 16) -> dict:
    """Compute kappa[m][b] = normalised BOPs cost for block m at bit-width b.

    kappa = params_m * b * act_bits  /  total_fp32_bops
    """
    bops_fp32 = sum(p * 32 * 32 for p in layer_params.values())
    if bops_fp32 == 0:
        bops_fp32 = 1.0

    kappa: dict = {}
    for m in block_names:
        clean = m.replace("encoder.", "").replace("module.", "").replace(".weight", "")
        params = layer_params.get(clean, 0)
        kappa[m] = {}
        for b in bit_options:
            kappa[m][b] = (params * b * act_bits) / bops_fp32
    return kappa


def build_importance_surface(collected_csv: str,
                             fp32_csv: str,
                             perf_csv: str,
                             policy_lut_path: str,
                             snr_list: list[int],
                             gamma_list: list[float],
                             J_bins: int = 3,
                             K_bins: int = 10,
                             tau_shr: float = 10.0,
                             anchor_policy_id: int = 0,
                             output_dir: str | None = None):
    """Build importance surface Omega[m, b, j, k] from collected data.

    For each block m and candidate bit-width b:
      1. Perturb anchor policy: set block m -> b, keep others same
      2. Approximate perturbed performance from collected per-sample data
      3. Compute marginal shortfall DeltaV = V(perturbed) - V(anchor)
      4. Bin by (SNR bin j, zeta bin k), average within each bin
      5. Shrinkage towards global mean
      6. Monotone calibration

    Returns dict with keys 'Omega', 'snr_edges', 'zeta_edges',
    'block_names', 'bit_options', 'kappa'.
    """
    if output_dir is None:
        output_dir = RESULTS_CSV

    # ---- load data ----
    df_coll = pd.read_csv(collected_csv)
    df_fp32 = pd.read_csv(fp32_csv)
    df_perf = pd.read_csv(perf_csv)
    df_lut  = pd.read_csv(policy_lut_path)
    if isinstance(df_lut["Policy"].iloc[0], str):
        df_lut["Policy"] = df_lut["Policy"].apply(ast.literal_eval)

    block_names = list(df_lut["Policy"].iloc[0].keys())
    bit_options = [16, 8, 4, 2]
    M = len(block_names)
    N_samples = len(df_fp32)

    zeta_vals = df_fp32["zeta_proxy"].values
    snr_arr   = np.array(snr_list, dtype=float)

    # ---- bin edges ----
    # SNR bins: one per SNR level by default (J_bins == len(snr_list))
    if J_bins == len(snr_list):
        snr_edges = np.array([snr_list[0] - 5] +
                             [(snr_list[i] + snr_list[i+1]) / 2
                              for i in range(len(snr_list) - 1)] +
                             [snr_list[-1] + 5])
    else:
        snr_edges = np.linspace(snr_arr.min() - 1, snr_arr.max() + 1, J_bins + 1)

    zeta_edges = np.quantile(zeta_vals, np.linspace(0, 1, K_bins + 1))
    zeta_edges[0]  -= 1e-6
    zeta_edges[-1] += 1e-6

    # ---- anchor policy performance ----
    anchor_mask = df_coll["policy_id"] == anchor_policy_id
    df_anchor = df_coll[anchor_mask].sort_values("sample_idx").reset_index(drop=True)

    # ---- build Omega ----
    # Omega[m_idx, b_idx, j, k] — importance of block m at bit b in state (j,k)
    n_bits = len(bit_options)
    Omega_raw   = np.zeros((M, n_bits, J_bins, K_bins))
    Omega_count = np.zeros((M, n_bits, J_bins, K_bins))

    # For each policy in LUT, determine which blocks differ from anchor
    anchor_policy = df_lut["Policy"].iloc[anchor_policy_id]

    print(f"[INFO] Building importance surface: M={M} blocks, "
          f"B={n_bits} bits, J={J_bins} SNR bins, K={K_bins} zeta bins")

    for p_idx, row in tqdm(df_lut.iterrows(), total=len(df_lut),
                           desc="Importance surface"):
        policy = row["Policy"]
        # Find which blocks differ from anchor
        diff_blocks = []
        for bname in block_names:
            if policy.get(bname, 32) != anchor_policy.get(bname, 32):
                diff_blocks.append(bname)

        if len(diff_blocks) == 0:
            continue  # skip anchor itself

        # Paper Eq.15 requires single-block perturbation pi^{m->b}.
        # For multi-block policies, approximate by dividing delta_V equally
        # among changed blocks (additive surrogate assumption).
        n_diff = len(diff_blocks)

        # Get per-sample data for this policy
        p_mask = df_coll["policy_id"] == p_idx
        df_p = df_coll[p_mask].sort_values("sample_idx").reset_index(drop=True)
        if len(df_p) != N_samples:
            continue

        # For each gamma and SNR, compute marginal shortfall
        for gamma in gamma_list:
            for snr in snr_list:
                # Perfect-CSI rate as reference
                r_ref = df_perf[f"r_perf_{snr}"].values

                # Anchor shortfall
                r_anchor = df_anchor[f"rate_{snr}"].values
                V_anchor = compute_shortfall(r_anchor, r_ref, gamma)

                # Perturbed shortfall
                r_pert = df_p[f"rate_{snr}"].values
                V_pert = compute_shortfall(r_pert, r_ref, gamma)

                # Marginal shortfall
                delta_V = V_pert - V_anchor  # positive = worse

                # SNR bin index
                j = int(np.clip(np.digitize(snr, snr_edges) - 1, 0, J_bins - 1))

                # Per-sample binning by zeta
                k_indices = np.clip(
                    np.digitize(zeta_vals, zeta_edges) - 1, 0, K_bins - 1
                )

                for bname in diff_blocks:
                    m_idx = block_names.index(bname)
                    b_val = policy[bname]
                    if b_val not in bit_options:
                        continue
                    b_idx = bit_options.index(b_val)

                    for k in range(K_bins):
                        mask_k = k_indices == k
                        n_k = np.sum(mask_k)
                        if n_k == 0:
                            continue
                        # Divide by n_diff for additive attribution
                        avg_dv = np.mean(delta_V[mask_k]) / n_diff
                        Omega_raw[m_idx, b_idx, j, k] += avg_dv
                        Omega_count[m_idx, b_idx, j, k] += 1

    # Average
    valid = Omega_count > 0
    Omega_avg = np.zeros_like(Omega_raw)
    Omega_avg[valid] = Omega_raw[valid] / Omega_count[valid]

    # ---- Shrinkage towards global mean ----
    Omega_global = np.zeros((M, n_bits))
    for m in range(M):
        for bi in range(n_bits):
            vals = Omega_avg[m, bi][valid[m, bi]]
            Omega_global[m, bi] = np.mean(vals) if len(vals) > 0 else 0.0

    Omega_shrunk = np.zeros_like(Omega_avg)
    for m in range(M):
        for bi in range(n_bits):
            for j in range(J_bins):
                for k in range(K_bins):
                    n_jk = Omega_count[m, bi, j, k]
                    alpha = n_jk / (n_jk + tau_shr)
                    Omega_shrunk[m, bi, j, k] = (
                        alpha * Omega_avg[m, bi, j, k]
                        + (1 - alpha) * Omega_global[m, bi]
                    )

    # ---- Monotone calibration ----
    Omega_cal = monotone_calibrate(Omega_shrunk, J_bins, K_bins)

    # ---- Compute kappa table ----
    # Try to load param counts from HAWQ CSV for accurate BOPs
    kappa = {}
    hawq_path = os.path.join(base_dir, "results", "csv", "hawq_importance_split.csv")
    if os.path.exists(hawq_path):
        hawq_df = pd.read_csv(hawq_path)
        layer_params = {r["Layer"]: int(r["Params"]) for _, r in hawq_df.iterrows()}
        act_bits = 16
        bops_fp32 = sum(p * 32 * act_bits for p in layer_params.values())
        for bname in block_names:
            kappa[bname] = {}
            p = layer_params.get(bname, 1000)  # fallback
            for b in bit_options:
                kappa[bname][b] = (p * b * act_bits) / bops_fp32
        print(f"[INFO] Loaded param counts from HAWQ CSV ({len(layer_params)} layers)")
    else:
        # Fallback: equal kappa per block, proportional to bit-width
        print("[WARN] HAWQ CSV not found, using approximate kappa")
        for bname in block_names:
            kappa[bname] = {}
            for b in bit_options:
                kappa[bname][b] = b / 32.0 / M

    result = {
        "Omega": Omega_cal,
        "snr_edges": snr_edges,
        "zeta_edges": zeta_edges,
        "block_names": block_names,
        "bit_options": bit_options,
        "kappa": kappa,
        "J_bins": J_bins,
        "K_bins": K_bins,
    }

    # Save to disk
    np.savez(os.path.join(output_dir, "rpmpq_v2_omega.npz"),
             Omega=Omega_cal,
             snr_edges=snr_edges,
             zeta_edges=zeta_edges,
             block_names=block_names,
             bit_options=bit_options,
             J_bins=J_bins,
             K_bins=K_bins)

    # Save kappa separately (dict)
    kappa_df_rows = []
    for bname in block_names:
        for b in bit_options:
            kappa_df_rows.append({"block": bname, "bits": b,
                                  "kappa": kappa[bname][b]})
    pd.DataFrame(kappa_df_rows).to_csv(
        os.path.join(output_dir, "rpmpq_v2_kappa.csv"), index=False)

    print(f"[INFO] Importance surface saved -> {output_dir}/rpmpq_v2_omega.npz")
    return result


def monotone_calibrate(Omega: np.ndarray,
                       J_bins: int, K_bins: int) -> np.ndarray:
    """Enforce monotonicity constraints on Omega.

    - Non-increasing in SNR (higher SNR -> lower importance/shortfall)
    - Non-decreasing in zeta (higher zeta -> more compressible -> less shortfall,
      so importance of aggressive quantisation is lower... but paper says
      nondecreasing, meaning harder samples have more shortfall).

    We use isotonic regression-like cumulative max/min sweeps.
    """
    Omega_cal = Omega.copy()
    M, n_bits = Omega_cal.shape[:2]

    for m in range(M):
        for bi in range(n_bits):
            # Non-increasing in j (SNR): sweep j from high to low, enforce cummax
            for k in range(K_bins):
                col = Omega_cal[m, bi, :, k]
                # Sweep from last j to first: each should be <= previous
                for j in range(J_bins - 2, -1, -1):
                    col[j] = max(col[j], col[j + 1])
                Omega_cal[m, bi, :, k] = col

            # Non-decreasing in k (zeta): sweep k from low to high
            for j in range(J_bins):
                row = Omega_cal[m, bi, j, :]
                for kk in range(1, K_bins):
                    row[kk] = max(row[kk], row[kk - 1])
                Omega_cal[m, bi, j, :] = row

    return Omega_cal


def optimize_per_state_policy(omega_data: dict,
                              lambda_mult: float = 1.0,
                              policy_lut_path: str | None = None) -> dict:
    """For each state cell (j, k), find the optimal policy.

    Decomposes into M independent block-level minimisations:
        b*(m) = argmin_b  [ Omega(m, b, j, k) + lambda * kappa(m, b) ]

    Returns:
        policy_map : dict with keys (j, k) -> policy dict {block_name: bits}
    """
    Omega      = omega_data["Omega"]
    block_names = omega_data["block_names"]
    bit_options = omega_data["bit_options"]
    kappa      = omega_data["kappa"]
    J_bins     = omega_data["J_bins"]
    K_bins     = omega_data["K_bins"]

    M = len(block_names)
    policy_map: dict = {}

    for j in range(J_bins):
        for k in range(K_bins):
            policy = {}
            for m_idx, bname in enumerate(block_names):
                best_b = bit_options[0]
                best_cost = float("inf")
                for bi, b in enumerate(bit_options):
                    cost = Omega[m_idx, bi, j, k] + lambda_mult * kappa[bname][b]
                    if cost < best_cost:
                        best_cost = cost
                        best_b = b
                policy[bname] = best_b
            policy_map[(j, k)] = policy

    print(f"[INFO] Policy map constructed: {J_bins} x {K_bins} = "
          f"{J_bins * K_bins} state cells")

    # Save policy map
    rows = []
    for (j, k), pol in policy_map.items():
        rows.append({"j": j, "k": k, "policy": str(pol)})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_CSV, "rpmpq_v2_policy_map.csv"), index=False)

    return policy_map


# =============================================================================
# SECTION 4 : Online Stage  --  Policy Lookup
# =============================================================================

class RPMPQv2Controller:
    """Online controller: given a CSI sample and SNR, return the policy."""

    def __init__(self,
                 policy_map: dict,
                 snr_edges: np.ndarray,
                 zeta_edges: np.ndarray,
                 K_d: np.ndarray,
                 K_a: np.ndarray,
                 lambda_d: float = 0.5,
                 lambda_a: float = 0.5):
        self.policy_map = policy_map
        self.snr_edges  = snr_edges
        self.zeta_edges = zeta_edges
        self.K_d = K_d
        self.K_a = K_a
        self.lambda_d = lambda_d
        self.lambda_a = lambda_a
        self.J_bins = len(snr_edges) - 1
        self.K_bins = len(zeta_edges) - 1

    def select_policy(self, x_csi: np.ndarray, snr: float) -> dict:
        """Select policy for a single CSI sample at a given SNR.

        Args:
            x_csi : (2, 32, 32) normalised [0,1] CSI
            snr   : SNR in dB

        Returns:
            policy : dict {block_name: bit_width}
        """
        zeta = compute_zeta_proxy(x_csi, self.K_d, self.K_a,
                                  self.lambda_d, self.lambda_a)
        j = int(np.clip(np.digitize(snr, self.snr_edges) - 1,
                        0, self.J_bins - 1))
        k = int(np.clip(np.digitize(zeta, self.zeta_edges) - 1,
                        0, self.K_bins - 1))
        return self.policy_map[(j, k)]

    def select_policy_batched(self, zeta_vals: np.ndarray,
                              snr: float) -> list[dict]:
        """Select policies for a batch of samples at a fixed SNR."""
        j = int(np.clip(np.digitize(snr, self.snr_edges) - 1,
                        0, self.J_bins - 1))
        k_indices = np.clip(
            np.digitize(zeta_vals, self.zeta_edges) - 1, 0, self.K_bins - 1
        ).astype(int)
        return [self.policy_map[(j, k)] for k in k_indices]


class StaticMPController:
    """Static MP baseline: pick a single policy based on average state."""

    def __init__(self, policy: dict):
        self.policy = policy

    def select_policy(self, x_csi=None, snr=None) -> dict:
        return self.policy


class OnlineHoyerController:
    """Online RP-MPQ with 1D Hoyer state (v1 baseline)."""

    def __init__(self, policy_lut_df: pd.DataFrame,
                 fitting_csv: str | None = None,
                 lambda_budget: float = 1.0,
                 num_bins: int = 20):
        self.lut_df = policy_lut_df.copy()
        if isinstance(self.lut_df["Policy"].iloc[0], str):
            self.lut_df["Policy"] = self.lut_df["Policy"].apply(ast.literal_eval)
        self.lambda_budget = lambda_budget

        # Build violation cost table from fitting data (if available)
        self.has_fitting = False
        if fitting_csv and os.path.exists(fitting_csv):
            df = pd.read_csv(fitting_csv)
            self._build_cost_table(df, num_bins)
            self.has_fitting = True

    def _build_cost_table(self, df, num_bins):
        """Build per-policy per-sparsity-bin (mu, sigma) table."""
        df["Config_k"] = df["B"].round(4)
        self.policy_ids = sorted(df["Config_k"].unique())
        self.s_grids = np.linspace(0.0, 1.0, num_bins + 1)
        self.cost_lut: dict = {}

        for k in self.policy_ids:
            sub = df[df["Config_k"] == k].copy()
            sub["Bin"] = pd.cut(sub["S"], bins=self.s_grids,
                                labels=False, include_lowest=True)
            stats = []
            for b_idx in range(num_bins):
                bd = sub[sub["Bin"] == b_idx]
                sc = (self.s_grids[b_idx] + self.s_grids[b_idx + 1]) / 2.0
                if len(bd) > 0:
                    mu = bd["NMSE_linear"].mean()
                    sigma = bd["NMSE_linear"].std() if len(bd) > 1 else 0.0
                else:
                    mu, sigma = np.nan, np.nan
                stats.append({"S": sc, "mu": mu, "sigma": sigma})
            sdf = pd.DataFrame(stats).interpolate(method="linear",
                                                   limit_direction="both")
            sdf = sdf.bfill().ffill()
            self.cost_lut[k] = sdf

    def select_policy(self, hoyer_s: float,
                      target_threshold: float = 0.1) -> dict:
        """Select policy based on Hoyer sparsity."""
        if not self.has_fitting:
            # Fall back to middle policy
            mid = len(self.lut_df) // 2
            return self.lut_df["Policy"].iloc[mid]

        best_id = self.policy_ids[0]
        for k in reversed(self.policy_ids):
            prof = self.cost_lut[k]
            s = np.clip(hoyer_s, prof["S"].min(), prof["S"].max())
            mu = np.interp(s, prof["S"], prof["mu"])
            sig = np.interp(s, prof["S"], prof["sigma"])
            cost = mu + self.lambda_budget * sig
            if cost <= target_threshold:
                best_id = k
                break

        row = self.lut_df.iloc[
            (self.lut_df["Actual_Saving"] - best_id).abs().argsort()[:1]
        ]
        return row["Policy"].values[0]


# =============================================================================
# SECTION 5 : Evaluation
# =============================================================================

def evaluate_scheme(model, test_loader, device, norm_params, snr_list,
                    scheme_name: str,
                    policy_lut_path: str,
                    controller=None,
                    zeta_vals: np.ndarray | None = None,
                    hoyer_vals: np.ndarray | None = None,
                    uniform_bits: int | None = None,
                    aq_bits: int = 0,
                    target_threshold: float = 0.1) -> dict:
    """Evaluate a quantisation scheme on the test set.

    Schemes:
      - uniform_intXX : apply uniform bit-width to all encoder weights
      - static_mp     : single MP policy for all samples
      - online_rpmpq_hoyer : 1D Hoyer-based per-sample selection
      - online_rpmpq_zeta  : 2D (SNR, zeta) per-sample selection

    Returns dict with per-sample NMSE and rates.
    """
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    min_val, range_val = norm_params
    N = len(test_loader.dataset)

    # Load LUT
    df_lut = pd.read_csv(policy_lut_path)
    if isinstance(df_lut["Policy"].iloc[0], str):
        df_lut["Policy"] = df_lut["Policy"].apply(ast.literal_eval)

    # Get block names from first policy
    block_names = list(df_lut["Policy"].iloc[0].keys())

    results = {
        "scheme": scheme_name,
        "nmse_linear": [],
        "nmse_db": [],
        "policies_used": [],
    }
    for snr in snr_list:
        results[f"rate_{snr}"] = []

    # ---- Uniform schemes ----
    if scheme_name.startswith("uniform_int"):
        bits = uniform_bits or int(scheme_name.replace("uniform_int", ""))
        policy = {bn: bits for bn in block_names}

        real_model.load_state_dict(original_state)
        apply_precision_policy(model, policy, device)

        nmse_l, rates = _compute_per_sample_nmse_rate(
            model, test_loader, device, norm_params, snr_list, aq_bits)
        results["nmse_linear"] = nmse_l
        results["nmse_db"] = [10 * np.log10(x + 1e-15) for x in nmse_l]
        for snr in snr_list:
            results[f"rate_{snr}"] = rates[snr]
        results["policies_used"] = [policy] * N

        real_model.load_state_dict(original_state)
        return results

    # ---- Static MP ----
    if scheme_name.startswith("static_mp"):
        # Use the median-saving policy from LUT
        mid_idx = len(df_lut) // 2
        policy = df_lut["Policy"].iloc[mid_idx]

        real_model.load_state_dict(original_state)
        apply_precision_policy(model, policy, device)

        nmse_l, rates = _compute_per_sample_nmse_rate(
            model, test_loader, device, norm_params, snr_list, aq_bits)
        results["nmse_linear"] = nmse_l
        results["nmse_db"] = [10 * np.log10(x + 1e-15) for x in nmse_l]
        for snr in snr_list:
            results[f"rate_{snr}"] = rates[snr]
        results["policies_used"] = [policy] * N

        real_model.load_state_dict(original_state)
        return results

    # ---- Online per-sample schemes ----
    # These need per-sample policy selection -> inference sample-by-sample
    # For efficiency, group samples by policy and batch-process each group.

    if "rpmpq" in scheme_name:
        assert controller is not None, "Controller required for online schemes"

        # Assign policies to all samples
        sample_policies: list[dict] = []

        if "zeta" in scheme_name:
            # 2D controller (SNR, zeta)
            assert zeta_vals is not None
            # We evaluate at each SNR separately, so assign per SNR
            # For the overall eval, use the first SNR for policy selection
            # (actual per-SNR selection happens in the rate computation loop)
            for i in range(N):
                # Use the middle SNR for policy assignment
                # (per-SNR results will be computed separately)
                policy = controller.select_policy(
                    test_loader.dataset[i].numpy(),
                    snr_list[len(snr_list) // 2])
                sample_policies.append(policy)

        elif "hoyer" in scheme_name:
            assert hoyer_vals is not None
            for i in range(N):
                policy = controller.select_policy(
                    hoyer_vals[i], target_threshold)
                sample_policies.append(policy)

        # Group samples by policy (for batched inference)
        policy_groups: dict[str, list[int]] = {}
        for i, pol in enumerate(sample_policies):
            key = str(sorted(pol.items()))
            if key not in policy_groups:
                policy_groups[key] = []
            policy_groups[key].append(i)

        # Pre-allocate results arrays
        nmse_arr = np.zeros(N)
        rate_arr = {snr: np.zeros(N) for snr in snr_list}

        print(f"  [{scheme_name}] {len(policy_groups)} distinct policies "
              f"for {N} samples")

        for pol_key, indices in tqdm(policy_groups.items(),
                                     desc=f"Eval {scheme_name}",
                                     leave=False):
            policy = sample_policies[indices[0]]

            # Create subset loader
            subset = Subset(test_loader.dataset, indices)
            sub_loader = DataLoader(subset,
                                    batch_size=test_loader.batch_size or 256,
                                    shuffle=False, num_workers=0)

            real_model.load_state_dict(original_state)
            apply_precision_policy(model, policy, device)

            nmse_l, rates = _compute_per_sample_nmse_rate(
                model, sub_loader, device, norm_params, snr_list, aq_bits)

            for local_i, global_i in enumerate(indices):
                nmse_arr[global_i] = nmse_l[local_i]
                for snr in snr_list:
                    rate_arr[snr][global_i] = rates[snr][local_i]

        results["nmse_linear"] = nmse_arr.tolist()
        results["nmse_db"] = (10 * np.log10(nmse_arr + 1e-15)).tolist()
        for snr in snr_list:
            results[f"rate_{snr}"] = rate_arr[snr].tolist()
        results["policies_used"] = sample_policies

        real_model.load_state_dict(original_state)
        return results

    raise ValueError(f"Unknown scheme: {scheme_name}")


def evaluate_all_schemes(model, test_loader, device, norm_params,
                         snr_list, policy_lut_path,
                         omega_path=None, zeta_csv=None,
                         fitting_csv=None,
                         aq_bits=0, lambda_mult=1.0,
                         K_d=None, K_a=None,
                         lambda_d=0.5, lambda_a=0.5,
                         target_threshold=0.1) -> dict:
    """Run all comparison schemes and collect results."""

    all_results: dict[str, dict] = {}

    # ---- zeta + hoyer values ----
    if zeta_csv and os.path.exists(zeta_csv):
        zdf = pd.read_csv(zeta_csv)
        zeta_vals = zdf["zeta_proxy"].values
    else:
        if K_d is None:
            K_d = build_kernel(32, 1.0)
        if K_a is None:
            K_a = build_kernel(32, 1.0)
        zeta_vals = compute_all_zeta(test_loader.dataset, K_d, K_a,
                                     use_proxy=True,
                                     lambda_d=lambda_d, lambda_a=lambda_a)

    _, hoyer_vals = compute_hoyer_sparsity_bins(test_loader.dataset, device)

    # ---- Load LUT ----
    df_lut = pd.read_csv(policy_lut_path)
    if isinstance(df_lut["Policy"].iloc[0], str):
        df_lut["Policy"] = df_lut["Policy"].apply(ast.literal_eval)

    # ---- 1. Uniform baselines ----
    for bits in [16, 8, 4]:
        name = f"uniform_int{bits}"
        print(f"\n[EVAL] {name}")
        all_results[name] = evaluate_scheme(
            model, test_loader, device, norm_params, snr_list,
            scheme_name=name, policy_lut_path=policy_lut_path,
            uniform_bits=bits, aq_bits=aq_bits)

    # ---- 2. Static MP (median policy) ----
    print(f"\n[EVAL] static_mp")
    all_results["static_mp"] = evaluate_scheme(
        model, test_loader, device, norm_params, snr_list,
        scheme_name="static_mp", policy_lut_path=policy_lut_path,
        aq_bits=aq_bits)

    # ---- 3. Online RP-MPQ (Hoyer, v1 baseline) ----
    print(f"\n[EVAL] online_rpmpq_hoyer")
    hoyer_ctrl = OnlineHoyerController(
        df_lut, fitting_csv=fitting_csv, lambda_budget=1.0, num_bins=20)
    all_results["online_rpmpq_hoyer"] = evaluate_scheme(
        model, test_loader, device, norm_params, snr_list,
        scheme_name="online_rpmpq_hoyer",
        policy_lut_path=policy_lut_path,
        controller=hoyer_ctrl,
        hoyer_vals=hoyer_vals,
        aq_bits=aq_bits,
        target_threshold=target_threshold)

    # ---- 4. Online RP-MPQ (zeta, v2 proposed) ----
    if omega_path and os.path.exists(omega_path):
        print(f"\n[EVAL] online_rpmpq_zeta")
        omega_data = dict(np.load(omega_path, allow_pickle=True))
        # Reconstruct the omega data structure
        omega_dict = {
            "Omega": omega_data["Omega"],
            "snr_edges": omega_data["snr_edges"],
            "zeta_edges": omega_data["zeta_edges"],
            "block_names": list(omega_data["block_names"]),
            "bit_options": list(omega_data["bit_options"]),
            "J_bins": int(omega_data["J_bins"]),
            "K_bins": int(omega_data["K_bins"]),
        }

        # Load kappa
        kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_kappa.csv")
        if os.path.exists(kappa_csv):
            kdf = pd.read_csv(kappa_csv)
            kappa = {}
            for _, r in kdf.iterrows():
                bn = r["block"]
                if bn not in kappa:
                    kappa[bn] = {}
                kappa[bn][int(r["bits"])] = r["kappa"]
            omega_dict["kappa"] = kappa
        else:
            # Fallback kappa
            bn_list = omega_dict["block_names"]
            M = len(bn_list)
            kappa = {bn: {b: b / 32.0 / M
                          for b in omega_dict["bit_options"]}
                     for bn in bn_list}
            omega_dict["kappa"] = kappa

        policy_map = optimize_per_state_policy(omega_dict,
                                               lambda_mult=lambda_mult)

        if K_d is None:
            K_d = build_kernel(32, 1.0)
        if K_a is None:
            K_a = build_kernel(32, 1.0)

        zeta_ctrl = RPMPQv2Controller(
            policy_map=policy_map,
            snr_edges=omega_dict["snr_edges"],
            zeta_edges=omega_dict["zeta_edges"],
            K_d=K_d, K_a=K_a,
            lambda_d=lambda_d, lambda_a=lambda_a)

        all_results["online_rpmpq_zeta"] = evaluate_scheme(
            model, test_loader, device, norm_params, snr_list,
            scheme_name="online_rpmpq_zeta",
            policy_lut_path=policy_lut_path,
            controller=zeta_ctrl,
            zeta_vals=zeta_vals,
            aq_bits=aq_bits)
    else:
        print("[WARN] Omega file not found -- skipping online_rpmpq_zeta")

    # ---- FP32 reference ----
    print(f"\n[EVAL] fp32_reference")
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    real_model.load_state_dict(original_state)
    nmse_l, rates = _compute_per_sample_nmse_rate(
        model, test_loader, device, norm_params, snr_list, aq_bits=0)
    all_results["fp32"] = {
        "scheme": "fp32",
        "nmse_linear": nmse_l,
        "nmse_db": [10 * np.log10(x + 1e-15) for x in nmse_l],
        **{f"rate_{snr}": rates[snr] for snr in snr_list},
    }

    # ---- Save summary ----
    summary_rows = []
    for name, res in all_results.items():
        row = {"scheme": name}
        nl = np.array(res["nmse_linear"])
        row["nmse_db_mean"] = 10 * np.log10(np.mean(nl) + 1e-15)
        row["nmse_db_median"] = np.median(res["nmse_db"])
        for snr in snr_list:
            rk = f"rate_{snr}"
            if rk in res:
                row[f"rate_{snr}_mean"] = np.mean(res[rk])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_eval_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[INFO] Evaluation summary saved -> {summary_csv}")
    print(summary_df.to_string(index=False))

    return all_results


# =============================================================================
# SECTION 5b : Plot Functions
# =============================================================================

def plot_nmse_vs_saving(policy_lut_path: str, eval_results: dict,
                        output_dir: str | None = None):
    """Plot NMSE (dB) Pareto curves for all schemes vs BOPs saving."""
    if output_dir is None:
        output_dir = RESULTS_PLOT

    with plt.rc_context(PAPER_FIG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        markers = {"uniform_int16": "s", "uniform_int8": "D",
                    "uniform_int4": "^", "static_mp": "o",
                    "online_rpmpq_hoyer": "v", "online_rpmpq_zeta": "*",
                    "fp32": "x"}
        colors = {"uniform_int16": "#1f77b4", "uniform_int8": "#ff7f0e",
                  "uniform_int4": "#d62728", "static_mp": "#9467bd",
                  "online_rpmpq_hoyer": "#2ca02c",
                  "online_rpmpq_zeta": "#e377c2", "fp32": "gray"}

        for name, res in eval_results.items():
            nmse_db = np.mean(res["nmse_db"])
            marker = markers.get(name, "o")
            color = colors.get(name, "black")
            ax.scatter(0, nmse_db, marker=marker, color=color,
                       s=100, label=name, zorder=5)

        ax.set_ylabel("Average NMSE (dB)")
        ax.set_title("RP-MPQ v2: Scheme Comparison")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, "rpmpq_v2_nmse_comparison.png")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved: {path}")


def plot_outage_vs_saving(eval_results: dict, snr_list: list[int],
                          gamma_list: list[float],
                          perf_csv: str,
                          output_dir: str | None = None):
    """Plot outage probability P(r < gamma * r_ref) for each SNR and gamma."""
    if output_dir is None:
        output_dir = RESULTS_PLOT

    df_perf = pd.read_csv(perf_csv)
    n_snr = len(snr_list)
    n_gamma = len(gamma_list)

    with plt.rc_context(PAPER_FIG_STYLE):
        fig, axes = plt.subplots(n_gamma, n_snr,
                                 figsize=(5 * n_snr, 4 * n_gamma),
                                 squeeze=False)

        scheme_styles = {
            "uniform_int16": ("s-", "#1f77b4"),
            "uniform_int8":  ("D-", "#ff7f0e"),
            "uniform_int4":  ("^-", "#d62728"),
            "static_mp":     ("o-", "#9467bd"),
            "online_rpmpq_hoyer": ("v-", "#2ca02c"),
            "online_rpmpq_zeta":  ("*-", "#e377c2"),
            "fp32":          ("x-", "gray"),
        }

        for gi, gamma in enumerate(gamma_list):
            for si, snr in enumerate(snr_list):
                ax = axes[gi][si]
                r_ref = df_perf[f"r_perf_{snr}"].values

                for name, res in eval_results.items():
                    rk = f"rate_{snr}"
                    if rk not in res:
                        continue
                    r_pol = np.array(res[rk])
                    N = min(len(r_pol), len(r_ref))
                    outage = np.mean(r_pol[:N] < gamma * r_ref[:N])

                    style, color = scheme_styles.get(name, ("o-", "black"))
                    ax.scatter(0, outage, label=name if (gi == 0 and si == 0) else "",
                               marker=style[0], color=color, s=60)

                ax.set_title(f"SNR={snr}dB, gamma={gamma}")
                ax.set_ylabel("Outage Probability")
                ax.set_ylim(-0.05, 1.05)

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        path = os.path.join(output_dir, "rpmpq_v2_outage.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved: {path}")


def plot_conditional_nmse(eval_results: dict, zeta_csv: str,
                          n_percentiles: int = 10,
                          output_dir: str | None = None):
    """Plot E[NMSE | zeta percentile] for validation of zeta descriptor."""
    if output_dir is None:
        output_dir = RESULTS_PLOT

    zdf = pd.read_csv(zeta_csv)
    zeta_vals = zdf["zeta_proxy"].values
    N = len(zeta_vals)

    pct_edges = np.linspace(0, 100, n_percentiles + 1)
    pct_centers = (pct_edges[:-1] + pct_edges[1:]) / 2
    zeta_thresholds = np.percentile(zeta_vals, pct_edges)

    with plt.rc_context(PAPER_FIG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        for name, res in eval_results.items():
            if name == "fp32":
                continue
            nmse_db = np.array(res["nmse_db"])
            n_res = min(len(nmse_db), N)

            cond_nmse = []
            for p in range(n_percentiles):
                lo, hi = zeta_thresholds[p], zeta_thresholds[p + 1]
                mask = (zeta_vals[:n_res] >= lo) & (zeta_vals[:n_res] < hi)
                if np.sum(mask) > 0:
                    cond_nmse.append(np.mean(nmse_db[:n_res][mask]))
                else:
                    cond_nmse.append(np.nan)

            ax.plot(pct_centers, cond_nmse, "o-", label=name, markersize=4)

        ax.set_xlabel("Zeta Percentile")
        ax.set_ylabel("E[NMSE (dB) | zeta percentile]")
        ax.set_title("Conditional NMSE by Zeta Descriptor")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, "rpmpq_v2_conditional_nmse.png")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved: {path}")


def plot_importance_surface(omega_path: str,
                            block_indices: list[int] | None = None,
                            output_dir: str | None = None):
    """Plot heatmaps of Omega for selected blocks."""
    if output_dir is None:
        output_dir = RESULTS_PLOT

    data = dict(np.load(omega_path, allow_pickle=True))
    Omega = data["Omega"]
    block_names = list(data["block_names"])
    bit_options = list(data["bit_options"])
    J = int(data["J_bins"])
    K = int(data["K_bins"])

    if block_indices is None:
        # Pick top-4 most important blocks (highest max Omega)
        max_omega = np.max(Omega, axis=(1, 2, 3))
        block_indices = np.argsort(max_omega)[-4:][::-1].tolist()

    n_blocks = len(block_indices)
    n_bits = len(bit_options)

    with plt.rc_context(PAPER_FIG_STYLE):
        fig, axes = plt.subplots(n_blocks, n_bits,
                                 figsize=(4 * n_bits, 3 * n_blocks),
                                 squeeze=False)

        for row, m_idx in enumerate(block_indices):
            bname = block_names[m_idx] if m_idx < len(block_names) else f"block_{m_idx}"
            for col, (bi, b) in enumerate(enumerate(bit_options)):
                ax = axes[row][col]
                im = ax.imshow(Omega[m_idx, bi], aspect="auto",
                               origin="lower", cmap="YlOrRd")
                ax.set_xlabel("Zeta bin (k)")
                ax.set_ylabel("SNR bin (j)")
                ax.set_title(f"{bname}\nINT{b}", fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle("Importance Surface Omega[m, b, j, k]", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        path = os.path.join(output_dir, "rpmpq_v2_importance_surface.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[PLOT] Saved: {path}")


def plot_budget_consistency(eval_results: dict,
                            policy_lut_path: str,
                            output_dir: str | None = None):
    """Print and save target vs achieved saving for each scheme."""
    if output_dir is None:
        output_dir = RESULTS_PLOT

    df_lut = pd.read_csv(policy_lut_path)
    if isinstance(df_lut["Policy"].iloc[0], str):
        df_lut["Policy"] = df_lut["Policy"].apply(ast.literal_eval)

    block_names = list(df_lut["Policy"].iloc[0].keys())
    bit_options = [16, 8, 4, 2]

    rows = []
    for name, res in eval_results.items():
        # Compute average saving from policies used
        if "policies_used" in res and res["policies_used"]:
            savings = []
            for pol in res["policies_used"]:
                # Count bits distribution
                bits_vals = list(pol.values())
                avg_bits = np.mean(bits_vals)
                saving_approx = (1.0 - avg_bits / 32.0) * 100
                savings.append(saving_approx)
            mean_saving = np.mean(savings)
            std_saving = np.std(savings)
        else:
            mean_saving = 0
            std_saving = 0

        nmse_db_mean = np.mean(res["nmse_db"]) if "nmse_db" in res else np.nan
        rows.append({
            "scheme": name,
            "avg_saving_pct": f"{mean_saving:.1f}",
            "std_saving_pct": f"{std_saving:.2f}",
            "nmse_db": f"{nmse_db_mean:.2f}",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_CSV, "rpmpq_v2_budget_consistency.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[TABLE] Budget Consistency:")
    print(df.to_string(index=False))
    print(f"Saved -> {csv_path}")


# =============================================================================
# SECTION 6 : Main  --  Full Pipeline
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="RP-MPQ v2: Rate-adaptive Precision Mixed-Precision "
                    "Quantization with zeta-proxy state descriptor")

    # ---- step selection ----
    p.add_argument("--step", type=str, default="all",
                   choices=["collect", "build", "eval", "plot", "all"],
                   help="Pipeline step to run")

    # ---- model / data ----
    p.add_argument("--encoder", type=str, default="mamba")
    p.add_argument("--decoder", type=str, default="transnet")
    p.add_argument("--encoded_dim", type=int, default=512)
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--encoder_layers", type=int, default=2)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--use_chunking", action="store_true", default=False)
    p.add_argument("--act_quant", type=int, default=16,
                   help="Activation quantisation bits (fixed)")
    p.add_argument("--aq", type=int, default=8,
                   help="Feedback (latent z) quantisation bits")

    p.add_argument("--train_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "DATA_Htrainout.mat"))
    p.add_argument("--test_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "DATA_Htestout.mat"))
    p.add_argument("--train_key", type=str, default="HT")
    p.add_argument("--test_key", type=str, default="HT")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint (.pth.tar)")

    # ---- hardware ----
    p.add_argument("--no_cuda", action="store_true", default=False,
                   help="Disable CUDA (use CPU only)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)

    # ---- zeta parameters ----
    p.add_argument("--alpha_d", type=float, default=1.0,
                   help="Kernel decay for delay domain")
    p.add_argument("--alpha_a", type=float, default=1.0,
                   help="Kernel decay for angular domain")
    p.add_argument("--lambda_d", type=float, default=0.5,
                   help="Proxy weight for delay coherence")
    p.add_argument("--lambda_a", type=float, default=0.5,
                   help="Proxy weight for angular coherence")

    # ---- importance surface ----
    p.add_argument("--J_bins", type=int, default=3,
                   help="Number of SNR bins")
    p.add_argument("--K_bins", type=int, default=10,
                   help="Number of zeta bins")
    p.add_argument("--tau_shr", type=float, default=10.0,
                   help="Shrinkage parameter")
    p.add_argument("--lambda_mult", type=float, default=1.0,
                   help="Lagrange multiplier for policy optimisation")
    p.add_argument("--anchor_policy_id", type=int, default=0,
                   help="Index of anchor policy in LUT (lowest saving)")

    # ---- SNR / gamma ----
    p.add_argument("--snr_list", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--gamma_list", type=float, nargs="+",
                   default=[0.99, 0.98, 0.95])

    # ---- evaluation ----
    p.add_argument("--target_threshold", type=float, default=0.1,
                   help="Violation cost threshold for Hoyer controller")

    # ---- policy LUT ----
    p.add_argument("--policy_lut", type=str, default=None,
                   help="Path to pruned policy LUT CSV")
    p.add_argument("--fitting_csv", type=str, default=None,
                   help="Path to fitting raw data CSV (for Hoyer controller)")

    return p.parse_args()


def _resolve_checkpoint(args) -> str | None:
    """Find the model checkpoint automatically if not specified."""
    if args.checkpoint:
        return args.checkpoint

    # Try the default location for mamba_transnet baseline
    candidates = [
        os.path.join(PROJECT_ROOT, "saved_models",
                     "mamba_transnet_L2_dim512_baseline", "best.pth"),
        os.path.join(PROJECT_ROOT, "saved_models",
                     "mamba_transnet_L2_dim512_baseline", "checkpoint.pth"),
        os.path.join(PROJECT_ROOT, "saved_models",
                     f"{args.encoder}_{args.decoder}_L{args.decoder_layers}"
                     f"_dim{args.encoded_dim}_baseline", "best.pth"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _resolve_policy_lut(args) -> str:
    """Find the policy LUT CSV automatically if not specified."""
    if args.policy_lut:
        return args.policy_lut
    candidates = [
        os.path.join(RESULTS_CSV, "mp_policy_lut_mamba_pruned.csv"),
        os.path.join(RESULTS_CSV, "mp_policy_lut_mamba_wide_pruned.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Policy LUT not found. Specify with --policy_lut or run "
        "train_ae.py --analyze_all first.")


def _load_model_and_data(args):
    """Load model and datasets (shared setup for all steps)."""
    device = "cuda" if (not args.no_cuda) and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device.upper()}")

    # ---- data ----
    train_set = CsiDataset(args.train_path, args.train_key)
    test_set  = CsiDataset(args.test_path, args.test_key,
                            normalization_params=train_set.normalization_params)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    norm_params = train_set.normalization_params

    # ---- model ----
    net = ModularAE(
        encoder_type=args.encoder,
        decoder_type=args.decoder,
        encoded_dim=args.encoded_dim,
        M=args.M,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        use_chunking=args.use_chunking,
        quant_act_bits=0,      # no training-time quant
        quant_param_bits=0,
    ).to(device)

    ckpt_path = _resolve_checkpoint(args)
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("state_dict", state)
        net.load_state_dict(sd, strict=False)
    else:
        print("[WARN] No checkpoint found. Using initialised model.")

    if device == "cuda" and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    return net, test_loader, norm_params, device


def main():
    args = parse_args()
    step = args.step

    print("=" * 70)
    print(f"  RP-MPQ v2  |  step={step}")
    print("=" * 70)

    # Resolve paths
    policy_lut_path = _resolve_policy_lut(args)
    print(f"[INFO] Policy LUT: {policy_lut_path}")

    # Kernel matrices (shared)
    K_d = build_kernel(32, args.alpha_d)
    K_a = build_kernel(32, args.alpha_a)

    # Derived paths
    collected_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_collected.csv")
    zeta_csv      = os.path.join(RESULTS_CSV, "rpmpq_v2_zeta.csv")
    fp32_csv      = os.path.join(RESULTS_CSV, "rpmpq_v2_fp32_ref.csv")
    perf_csv      = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    omega_path    = os.path.join(RESULTS_CSV, "rpmpq_v2_omega.npz")
    fitting_csv   = args.fitting_csv or os.path.join(
        RESULTS_CSV, f"fitting_raw_data_{args.encoder}.csv")

    # ================================================================
    # Step 1 : COLLECT  (GPU)
    # ================================================================
    if step in ("collect", "all"):
        print("\n" + "=" * 60)
        print("  STEP 1: Collect per-sample data  (GPU)")
        print("=" * 60)

        net, test_loader, norm_params, device = _load_model_and_data(args)

        collect_per_sample_data(
            model=net,
            test_loader=test_loader,
            policy_lut_path=policy_lut_path,
            device=device,
            norm_params=norm_params,
            snr_list=args.snr_list,
            aq_bits=args.aq,
            output_csv=collected_csv,
            zeta_csv=zeta_csv,
            K_d=K_d, K_a=K_a,
            lambda_d=args.lambda_d,
            lambda_a=args.lambda_a,
        )

    # ================================================================
    # Step 2 : BUILD  (CPU — no GPU needed)
    # ================================================================
    if step in ("build", "all"):
        print("\n" + "=" * 60)
        print("  STEP 2: Build importance surface + policy map  (CPU)")
        print("=" * 60)

        # Verify collected data exists
        for req in (collected_csv, fp32_csv, perf_csv):
            if not os.path.exists(req):
                raise FileNotFoundError(
                    f"Required file not found: {req}\n"
                    f"Run --step collect first.")

        omega_data = build_importance_surface(
            collected_csv=collected_csv,
            fp32_csv=fp32_csv,
            perf_csv=perf_csv,
            policy_lut_path=policy_lut_path,
            snr_list=args.snr_list,
            gamma_list=args.gamma_list,
            J_bins=args.J_bins,
            K_bins=args.K_bins,
            tau_shr=args.tau_shr,
            anchor_policy_id=args.anchor_policy_id,
        )

        policy_map = optimize_per_state_policy(
            omega_data, lambda_mult=args.lambda_mult)

        # Plot importance surface
        plot_importance_surface(omega_path)

    # ================================================================
    # Step 3 : EVAL  (GPU)
    # ================================================================
    if step in ("eval", "all"):
        print("\n" + "=" * 60)
        print("  STEP 3: Evaluate all schemes  (GPU)")
        print("=" * 60)

        net, test_loader, norm_params, device = _load_model_and_data(args)

        all_results = evaluate_all_schemes(
            model=net,
            test_loader=test_loader,
            device=device,
            norm_params=norm_params,
            snr_list=args.snr_list,
            policy_lut_path=policy_lut_path,
            omega_path=omega_path if os.path.exists(omega_path) else None,
            zeta_csv=zeta_csv if os.path.exists(zeta_csv) else None,
            fitting_csv=fitting_csv if os.path.exists(fitting_csv) else None,
            aq_bits=args.aq,
            lambda_mult=args.lambda_mult,
            K_d=K_d, K_a=K_a,
            lambda_d=args.lambda_d,
            lambda_a=args.lambda_a,
            target_threshold=args.target_threshold,
        )

        # ---- generate plots ----
        plot_nmse_vs_saving(policy_lut_path, all_results)

        if os.path.exists(perf_csv):
            plot_outage_vs_saving(all_results, args.snr_list,
                                  args.gamma_list, perf_csv)

        if os.path.exists(zeta_csv):
            plot_conditional_nmse(all_results, zeta_csv)

        plot_budget_consistency(all_results, policy_lut_path)

    # ================================================================
    # Step 4 : PLOT only  (no GPU)
    # ================================================================
    if step == "plot":
        print("\n" + "=" * 60)
        print("  STEP: Generate plots from cached results  (CPU)")
        print("=" * 60)

        if os.path.exists(omega_path):
            plot_importance_surface(omega_path)

        # Re-load eval results from summary if available
        summary_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_eval_summary.csv")
        if os.path.exists(summary_csv):
            print(f"\n[INFO] Summary from {summary_csv}:")
            print(pd.read_csv(summary_csv).to_string(index=False))
        else:
            print("[WARN] No evaluation summary found. Run --step eval first.")

    print("\n" + "=" * 70)
    print("  RP-MPQ v2  |  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
