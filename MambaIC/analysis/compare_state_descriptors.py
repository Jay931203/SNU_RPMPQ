#!/usr/bin/env python3
"""
compare_state_descriptors.py
=============================
Stage 1: Compute ζ_full, ζ_proxy, Hoyer for COST2100 CSI samples.
Stage 2: Match with existing per-sample NMSE (fitting_raw_data_mamba.csv)
         and compare which state descriptor better predicts quantization sensitivity.

Key insight: fitting_raw_data_mamba.csv already has per-sample × per-policy NMSE,
and sample ordering is deterministic (np.linspace), so NO NEW INFERENCE is needed.
Just compute ζ on the same samples and re-correlate.

Usage:
  python compare_state_descriptors.py                     # Full analysis
  python compare_state_descriptors.py --stage 1           # Feature computation only
  python compare_state_descriptors.py --alpha_d 1.5       # Custom decay exponent

Requirements: scipy, numpy, pandas, matplotlib (no PyTorch/GPU needed)
"""

import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


# ============================================================
# Path Resolution
# ============================================================

def get_base_dir(override=None):
    """Auto-detect MambaIC base directory."""
    if override:
        return Path(override)
    candidates = [
        Path(__file__).resolve().parent.parent,                    # MambaIC/analysis/ → MambaIC/
        Path("G:/내 드라이브/MambaCompression/MambaIC"),
        Path("/content/drive/MyDrive/MyProjects/01_CL_PEFT_WIRELESS/MambaCompression/MambaIC"),
    ]
    for c in candidates:
        if (c / "data").exists():
            return c
    return candidates[0]


# ============================================================
# Kernel Construction
# ============================================================

def build_kernel(n, alpha):
    """Build soft-locality kernel K ∈ R^{n×n}.

    K[i,i'] = (1 + |i-i'|)^{-α} / Σ_k (1 + |i-k|)^{-α}
    Each row normalized to sum to 1.
    """
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])  # (n, n)
    raw = (1.0 + dist) ** (-alpha)
    K = raw / raw.sum(axis=1, keepdims=True)
    return K


# ============================================================
# Feature Computation
# ============================================================

def compute_energy_map(x):
    """Normalized energy map P from CSI sample.

    Data is [0,1]-normalized; subtract 0.5 to center before computing
    complex magnitude (matches train_ae.py's 0-Mean normalization).

    Args:
        x: shape (2, N_d, N_a) — [real/imag, delay, angular], values in [0,1]
    Returns:
        P: shape (N_d, N_a), sums to 1
    """
    x_c = x - 0.5  # center: 0-mean normalization
    energy = x_c[0]**2 + x_c[1]**2  # complex magnitude squared
    total = energy.sum()
    if total < 1e-12:
        return np.ones_like(energy) / energy.size
    return energy / total


def compute_zeta_full(P, K_d, K_a):
    """ζ_full = 1 - tr(P^T K_d P K_a).  O(N^3)."""
    return 1.0 - np.trace(P.T @ K_d @ P @ K_a)


def compute_zeta_proxy(P, K_d, K_a, lambda_d=0.5, lambda_a=0.5):
    """ζ_proxy = 1 - λ_d c_d - λ_a c_a.  O(N^2)."""
    p_d = P.sum(axis=1)  # delay marginal
    p_a = P.sum(axis=0)  # angular marginal
    c_d = p_d @ K_d @ p_d
    c_a = p_a @ K_a @ p_a
    return 1.0 - lambda_d * c_d - lambda_a * c_a, c_d, c_a


def compute_hoyer(x):
    """Hoyer = ||x||_1 / ||x||_2 (same as train_ae.py)."""
    l1 = np.abs(x).sum()
    l2 = np.sqrt((x**2).sum())
    return l1 / (l2 + 1e-8)


def compute_all_features(csi_data, K_d, K_a, lambda_d, lambda_a, sample_indices=None):
    """Compute ζ_full, ζ_proxy, Hoyer for selected samples."""
    if sample_indices is None:
        sample_indices = np.arange(len(csi_data))

    records = []
    for i, idx in enumerate(sample_indices):
        x = csi_data[idx]
        P = compute_energy_map(x)
        zf = compute_zeta_full(P, K_d, K_a)
        zp, c_d, c_a = compute_zeta_proxy(P, K_d, K_a, lambda_d, lambda_a)
        h = compute_hoyer(x)
        records.append({
            "sample_pos": i,           # position in calibration set (0-999)
            "global_idx": int(idx),     # index in full dataset
            "hoyer": h,
            "zeta_full": zf,
            "zeta_proxy": zp,
            "c_d": c_d,
            "c_a": c_a,
        })
        if (i + 1) % 200 == 0:
            print(f"  Computed {i+1}/{len(sample_indices)} samples")

    return pd.DataFrame(records)


# ============================================================
# Stage 1: Feature Analysis
# ============================================================

def run_stage1(csi_data, K_d, K_a, args, base_dir, fit_indices):
    """Compute features and analyze distributions/correlations."""
    print("=" * 60)
    print("STAGE 1: Feature Computation")
    print("=" * 60)

    df = compute_all_features(csi_data, K_d, K_a, args.lambda_d, args.lambda_a, fit_indices)

    # Save
    csv_dir = base_dir / "results" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "zeta_features.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFeatures saved: {out_csv}")

    # Statistics
    print("\n--- Feature Statistics ---")
    for col in ["hoyer", "zeta_full", "zeta_proxy", "c_d", "c_a"]:
        v = df[col]
        print(f"  {col:12s}: mean={v.mean():.6f}  std={v.std():.6f}  "
              f"range=[{v.min():.6f}, {v.max():.6f}]")

    # Correlations
    print("\n--- Rank Correlations (Spearman) ---")
    rho_fp, p_fp = spearmanr(df["zeta_full"], df["zeta_proxy"])
    rho_fh, p_fh = spearmanr(df["zeta_full"], df["hoyer"])
    rho_ph, p_ph = spearmanr(df["zeta_proxy"], df["hoyer"])
    print(f"  ζ_full  vs ζ_proxy : ρ = {rho_fp:.4f}  (p = {p_fp:.2e})")
    print(f"  ζ_full  vs Hoyer   : ρ = {rho_fh:.4f}  (p = {p_fh:.2e})")
    print(f"  ζ_proxy vs Hoyer   : ρ = {rho_ph:.4f}  (p = {p_ph:.2e})")

    # Plots
    plot_dir = base_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) ζ_full vs ζ_proxy
    ax = axes[0]
    ax.scatter(df["zeta_full"], df["zeta_proxy"], s=6, alpha=0.4, c='tab:blue')
    ax.set_xlabel(r"$\zeta_{\mathrm{full}}$", fontsize=11)
    ax.set_ylabel(r"$\zeta_{\mathrm{proxy}}$", fontsize=11)
    ax.set_title(f"Proxy Fidelity (Spearman ρ = {rho_fp:.3f})")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, lw=0.8)

    # (b) ζ_full vs Hoyer
    ax = axes[1]
    ax.scatter(df["zeta_full"], df["hoyer"], s=6, alpha=0.4, c='tab:orange')
    ax.set_xlabel(r"$\zeta_{\mathrm{full}}$", fontsize=11)
    ax.set_ylabel("Hoyer Sparsity", fontsize=11)
    ax.set_title(f"ζ_full vs Hoyer (Spearman ρ = {rho_fh:.3f})")

    # (c) Distributions
    ax = axes[2]
    ax.hist(df["zeta_full"], bins=50, alpha=0.5, density=True,
            label=r"$\zeta_{\mathrm{full}}$", color='tab:blue')
    ax.hist(df["zeta_proxy"], bins=50, alpha=0.5, density=True,
            label=r"$\zeta_{\mathrm{proxy}}$", color='tab:cyan')
    ax.set_xlabel("Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax2 = ax.twinx()
    ax2.hist(df["hoyer"], bins=50, alpha=0.3, density=True, color='tab:orange')
    ax2.set_ylabel("Density (Hoyer)", fontsize=11, color='tab:orange')
    ax.set_title("Feature Distributions")
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    fig.savefig(plot_dir / "stage1_feature_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_dir / 'stage1_feature_comparison.png'}")

    return df


# ============================================================
# Stage 2: Correlation with Quantization Sensitivity
# ============================================================

def run_stage2(feature_df, base_dir):
    """Match features with per-sample NMSE, compare correlations."""
    print("\n" + "=" * 60)
    print("STAGE 2: Quantization Sensitivity Correlation")
    print("=" * 60)

    csv_dir = base_dir / "results" / "csv"
    plot_dir = base_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fitting_csv = csv_dir / "fitting_raw_data_mamba.csv"
    if not fitting_csv.exists():
        print(f"ERROR: {fitting_csv} not found.")
        print("Need existing calibration data. Run profiling in train_ae.py first.")
        return None

    cal_df = pd.read_csv(fitting_csv)
    policies = sorted(cal_df['B'].unique())
    n_policies = len(policies)
    savings = np.array(policies)

    # Extract per-policy groups — each group has samples in same order
    first_group = cal_df[cal_df['B'] == policies[0]].reset_index(drop=True)
    n_samples = len(first_group)
    n_features = len(feature_df)

    print(f"Calibration data: {len(cal_df)} rows, {n_policies} policies, {n_samples} samples/policy")

    # --- Sample Matching ---
    hoyer_csv = first_group['S'].values
    hoyer_computed = feature_df['hoyer'].values

    if n_features != n_samples:
        print(f"WARNING: sample count mismatch (features={n_features}, CSV={n_samples})")
        n_use = min(n_features, n_samples)
        print(f"  Using first {n_use} samples from each")
    else:
        n_use = n_samples

    # Sanity check: Hoyer values should match
    hoyer_diff = np.abs(hoyer_csv[:n_use] - hoyer_computed[:n_use])
    print(f"\nHoyer sanity check: max_diff={hoyer_diff.max():.6f}, mean_diff={hoyer_diff.mean():.6f}")
    if hoyer_diff.max() < 0.01:
        print("  [OK] Sample matching confirmed!")
    elif hoyer_diff.max() < 1.0:
        print("  [~] Approximate match (likely small normalization difference)")
    else:
        print("  [FAIL] Hoyer values differ significantly. Check dataset/normalization.")
        print(f"    CSV range:  [{hoyer_csv[:n_use].min():.4f}, {hoyer_csv[:n_use].max():.4f}]")
        print(f"    Computed:   [{hoyer_computed[:n_use].min():.4f}, {hoyer_computed[:n_use].max():.4f}]")

    # --- Build NMSE matrix: (n_samples, n_policies) ---
    nmse_matrix = np.zeros((n_use, n_policies))
    for k, b in enumerate(policies):
        group = cal_df[cal_df['B'] == b].reset_index(drop=True)
        nmse_matrix[:, k] = group['NMSE_linear'].values[:n_use]

    zeta_full = feature_df['zeta_full'].values[:n_use]
    zeta_proxy = feature_df['zeta_proxy'].values[:n_use]
    hoyer = feature_df['hoyer'].values[:n_use]

    # --- Quantization Sensitivity ---
    # Sensitivity = NMSE at most aggressive policy - NMSE at mildest policy
    nmse_mild = nmse_matrix[:, 0]        # lowest saving
    nmse_aggressive = nmse_matrix[:, -1]  # highest saving
    sensitivity = nmse_aggressive - nmse_mild

    print(f"\n--- Quantization Sensitivity ---")
    print(f"  Mild policy:       saving={savings[0]:.4f}")
    print(f"  Aggressive policy: saving={savings[-1]:.4f}")
    print(f"  Sensitivity range: [{sensitivity.min():.6f}, {sensitivity.max():.6f}]")

    # --- Feature vs Sensitivity Correlation ---
    print(f"\n--- Feature vs Sensitivity (Aggressive - Mild) ---")
    descriptors = [
        ("ζ_full",  zeta_full),
        ("ζ_proxy", zeta_proxy),
        ("Hoyer",   hoyer),
    ]
    summary = {}
    for name, vals in descriptors:
        rho_s, p_s = spearmanr(vals, sensitivity)
        rho_p, p_p = pearsonr(vals, sensitivity)
        summary[name] = {"spearman": rho_s, "pearson": rho_p}
        print(f"  {name:10s}: Spearman ρ = {rho_s:+.4f} (p={p_s:.2e}), "
              f"Pearson r = {rho_p:+.4f} (p={p_p:.2e})")

    # --- Per-Policy Correlation Sweep ---
    print(f"\n--- Per-Policy Spearman ρ (feature vs NMSE) ---")
    print(f"  {'Saving':>8s}  {'ζ_full':>8s}  {'ζ_proxy':>8s}  {'Hoyer':>8s}")

    per_policy = []
    for k in range(n_policies):
        nmse_k = nmse_matrix[:, k]
        rho_zf, _ = spearmanr(zeta_full, nmse_k)
        rho_zp, _ = spearmanr(zeta_proxy, nmse_k)
        rho_h, _ = spearmanr(hoyer, nmse_k)
        per_policy.append({
            "saving": savings[k],
            "rho_zeta_full": rho_zf,
            "rho_zeta_proxy": rho_zp,
            "rho_hoyer": rho_h,
        })
        # Print every 10th + last
        if k % max(1, n_policies // 10) == 0 or k == n_policies - 1:
            print(f"  {savings[k]:8.4f}  {rho_zf:+8.4f}  {rho_zp:+8.4f}  {rho_h:+8.4f}")

    corr_df = pd.DataFrame(per_policy)

    # --- Tercile Discrimination ---
    print(f"\n--- Tercile Discrimination (NMSE separation by feature bins) ---")
    for name, vals in descriptors:
        t33, t66 = np.percentile(vals, [33, 66])
        low = nmse_aggressive[vals <= t33]
        mid = nmse_aggressive[(vals > t33) & (vals <= t66)]
        high = nmse_aggressive[vals > t66]
        sep = high.mean() - low.mean()  # mean NMSE difference between top/bottom tercile
        print(f"  {name:10s}: Low={low.mean():.6f}, Mid={mid.mean():.6f}, "
              f"High={high.mean():.6f}, Separation={sep:.6f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY: Average |Spearman ρ| across all {n_policies} policies")
    print(f"{'='*60}")
    avg_zf = corr_df['rho_zeta_full'].abs().mean()
    avg_zp = corr_df['rho_zeta_proxy'].abs().mean()
    avg_h = corr_df['rho_hoyer'].abs().mean()
    print(f"  ζ_full  : {avg_zf:.4f}")
    print(f"  ζ_proxy : {avg_zp:.4f}")
    print(f"  Hoyer   : {avg_h:.4f}")

    winner = "ζ" if avg_zf > avg_h else "Hoyer"
    ratio = max(avg_zf, avg_h) / (min(avg_zf, avg_h) + 1e-8)
    print(f"\n  → {winner} has {ratio:.1f}× higher average correlation with per-sample NMSE")

    # --- Plots ---
    # Use a moderate policy (~90% saving) for cleaner scatter
    mid_idx = n_policies // 2
    nmse_mid = nmse_matrix[:, mid_idx]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) ζ_proxy vs NMSE (moderate policy)
    ax = axes[0, 0]
    ax.scatter(zeta_proxy, nmse_mid, s=6, alpha=0.4, c='tab:blue')
    ax.set_xlabel(r"$\zeta_{\mathrm{proxy}}$", fontsize=11)
    ax.set_ylabel(f"NMSE (saving={savings[mid_idx]:.1%})", fontsize=11)
    rho, _ = spearmanr(zeta_proxy, nmse_mid)
    ax.set_title(r"$\zeta_{\mathrm{proxy}}$" + f" vs NMSE ({rho:.3f})")

    # (b) Hoyer vs NMSE (same policy)
    ax = axes[0, 1]
    ax.scatter(hoyer, nmse_mid, s=6, alpha=0.4, c='tab:orange')
    ax.set_xlabel("Hoyer Sparsity", fontsize=11)
    ax.set_ylabel(f"NMSE (saving={savings[mid_idx]:.1%})", fontsize=11)
    rho, _ = spearmanr(hoyer, nmse_mid)
    ax.set_title(f"Hoyer vs NMSE ({rho:.3f})")

    # (c) Per-policy correlation sweep
    ax = axes[1, 0]
    ax.plot(corr_df['saving'], corr_df['rho_zeta_full'],
            'b-', lw=1.5, label=r"$\zeta_{\mathrm{full}}$")
    ax.plot(corr_df['saving'], corr_df['rho_zeta_proxy'],
            'b--', lw=1.5, label=r"$\zeta_{\mathrm{proxy}}$")
    ax.plot(corr_df['saving'], corr_df['rho_hoyer'],
            'r-', lw=1.5, label="Hoyer")
    ax.set_xlabel("BOPs Saving", fontsize=11)
    ax.set_ylabel("Spearman ρ (feature vs NMSE)", fontsize=11)
    ax.set_title("Per-Policy Correlation Sweep")
    ax.legend(fontsize=9)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.grid(alpha=0.2)

    # (d) Tercile boxplot comparison (moderate policy, ζ_proxy vs Hoyer)
    ax = axes[1, 1]
    for idx, (name, vals, color) in enumerate([
        (r"$\zeta_{\mathrm{proxy}}$", zeta_proxy, 'tab:blue'),
        ("Hoyer", hoyer, 'tab:orange'),
    ]):
        t33, t66 = np.percentile(vals, [33, 66])
        groups = [
            nmse_mid[vals <= t33],
            nmse_mid[(vals > t33) & (vals <= t66)],
            nmse_mid[vals > t66],
        ]
        positions = [1 + idx*4, 2 + idx*4, 3 + idx*4]
        bp = ax.boxplot(groups, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
    ax.set_xticks([2, 6])
    ax.set_xticklabels([r"$\zeta_{\mathrm{proxy}}$ terciles", "Hoyer terciles"])
    ax.set_ylabel(f"NMSE (saving={savings[mid_idx]:.1%})", fontsize=11)
    ax.set_title("Tercile Discrimination: NMSE Separation")

    plt.tight_layout()
    fig.savefig(plot_dir / "stage2_sensitivity_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {plot_dir / 'stage2_sensitivity_correlation.png'}")

    # Save correlation data
    corr_df.to_csv(csv_dir / "per_policy_correlation.csv", index=False)
    print(f"Correlation data saved: {csv_dir / 'per_policy_correlation.csv'}")

    return corr_df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare ζ vs Hoyer as state descriptors for RP-MPQ")
    parser.add_argument("--stage", type=int, default=0,
                        help="1=features only, 2=correlation only (needs precomputed), 0=both")
    parser.add_argument("--alpha_d", type=float, default=1.0,
                        help="Delay-axis decay exponent (default: 1.0)")
    parser.add_argument("--alpha_a", type=float, default=1.0,
                        help="Angular-axis decay exponent (default: 1.0)")
    parser.add_argument("--lambda_d", type=float, default=0.5,
                        help="Proxy delay weight (default: 0.5)")
    parser.add_argument("--lambda_a", type=float, default=0.5,
                        help="Proxy angular weight (default: 0.5)")
    parser.add_argument("--data", type=str, default="outdoor_test",
                        choices=["outdoor_test", "outdoor_train", "indoor_test"],
                        help="Dataset to use (default: outdoor_test)")
    parser.add_argument("--n_cal", type=int, default=1000,
                        help="Number of calibration samples (default: 1000)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="MambaIC base directory override")
    args = parser.parse_args()

    base_dir = get_base_dir(args.base_dir)
    print(f"Base directory: {base_dir}")
    print(f"Parameters: α_d={args.alpha_d}, α_a={args.alpha_a}, "
          f"λ_d={args.lambda_d}, λ_a={args.lambda_a}")

    # Build kernels (constant, independent of data)
    N_d, N_a = 32, 32
    K_d = build_kernel(N_d, args.alpha_d)
    K_a = build_kernel(N_a, args.alpha_a)
    print(f"Kernels: K_d ({N_d}×{N_d}, α={args.alpha_d}), "
          f"K_a ({N_a}×{N_a}, α={args.alpha_a})")

    # --- Load Data ---
    if args.stage != 2:  # Stage 1 or both
        data_map = {
            "outdoor_test":  ("DATA_Htestout.mat",  "HT"),
            "outdoor_train": ("DATA_Htrainout.mat", "HT"),
            "indoor_test":   ("DATA_Htestin.mat",   "HT"),
        }
        mat_name, data_key = data_map[args.data]
        mat_path = base_dir / "data" / mat_name

        if not mat_path.exists():
            print(f"ERROR: {mat_path} not found.")
            print("Ensure COST2100 .mat files are in MambaIC/data/")
            return

        print(f"\nLoading: {mat_path}")
        mat_data = sio.loadmat(str(mat_path))
        csi_raw = np.array(mat_data[data_key], dtype=np.float32)
        if csi_raw.ndim == 2:
            csi_raw = csi_raw.reshape(-1, 2, 32, 32)
        N = len(csi_raw)
        print(f"Data shape: {csi_raw.shape}")

        # Normalize to [0, 1] (matches CsiDataset)
        min_val = csi_raw.min()
        range_val = csi_raw.max() - min_val + 1e-9
        csi_data = (csi_raw - min_val) / range_val
        print(f"Normalized: [{min_val:.4f}, {min_val + range_val:.4f}] → [0, 1]")

        # Calibration indices (matches train_ae.py exactly)
        fit_indices = np.linspace(0, N - 1, args.n_cal, dtype=int)
        print(f"Calibration: {len(fit_indices)} samples from {N} total")

    # --- Run Stages ---
    if args.stage in (0, 1):
        feature_df = run_stage1(csi_data, K_d, K_a, args, base_dir, fit_indices)

    if args.stage in (0, 2):
        if args.stage == 2:
            # Load pre-computed features
            feat_csv = base_dir / "results" / "csv" / "zeta_features.csv"
            if not feat_csv.exists():
                print(f"ERROR: {feat_csv} not found. Run Stage 1 first.")
                return
            feature_df = pd.read_csv(feat_csv)
            print(f"Loaded pre-computed features: {len(feature_df)} rows")

        run_stage2(feature_df, base_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
