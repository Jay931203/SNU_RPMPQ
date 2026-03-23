"""
HAWQ Validation Figures for Research Meeting
---
Figure 1: Sensitivity heatmap (Layer x Bit-width)
Figure 2: ILP Surrogate quality — S(π) vs actual NMSE degradation
Figure 3: Hutchinson estimator convergence — nc=4/8/32 comparison
Figure 4: Diagonal dominance (indirect) — layer sensitivity profile

Data: hawq_importance_split*.csv, mp_policy_lut_mamba.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_DIR    = os.path.join(RESULT_DIR, "csv")
PLOT_DIR   = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────
df_sens = pd.read_csv(os.path.join(CSV_DIR, "hawq_importance_split.csv"))
df_lut  = pd.read_csv(os.path.join(CSV_DIR, "mp_policy_lut_mamba.csv"))

# Hutchinson convergence comparison
nc_files = {
    "nc=4":  "hawq_importance_split_nc4.csv",
    "nc=8":  "hawq_importance_split_nc8.csv",
    "nc=32": "hawq_importance_split.csv",
}
dfs_nc = {}
for label, fname in nc_files.items():
    fpath = os.path.join(CSV_DIR, fname)
    if os.path.exists(fpath):
        dfs_nc[label] = pd.read_csv(fpath)

print("Loaded data:")
print(f"  hawq_importance_split: {len(df_sens)} layers, columns: {list(df_sens.columns)}")
print(f"  mp_policy_lut_mamba:   {len(df_lut)} policies, columns: {list(df_lut.columns)}")
for k, v in dfs_nc.items():
    print(f"  {k}: {len(v)} layers")

# ─────────────────────────────────────────────────────
# Figure 1: Sensitivity Heatmap (Layer × Bit-width)
# ─────────────────────────────────────────────────────
def plot_sensitivity_heatmap(df, save_path):
    bit_cols = ["Omg_INT16", "Omg_INT8", "Omg_INT4", "Omg_INT2"]
    bit_labels = ["INT16", "INT8", "INT4", "INT2"]

    # Use only columns that exist
    available = [c for c in bit_cols if c in df.columns]
    available_labels = [bit_labels[bit_cols.index(c)] for c in available]

    # Top-20 layers by max sensitivity
    df_top = df.copy()
    df_top["max_omega"] = df_top[available].max(axis=1)
    df_top = df_top.nlargest(20, "max_omega").reset_index(drop=True)

    data = df_top[available].values.astype(float)

    # Log scale (add small epsilon to avoid log(0))
    data_log = np.log10(data + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data_log.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(df_top)))
    ax.set_xticklabels(df_top["Layer"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(available_labels)))
    ax.set_yticklabels(available_labels, fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log₁₀(Ω_i)", fontsize=10)

    ax.set_title("Layer Sensitivity Heatmap — Ω_i per Bit-width\n"
                 "(MT-AE encoder, COST2100 outdoor, top-20 layers by max Ω)",
                 fontsize=11)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Bit-width", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────
# Figure 2: ILP Surrogate Quality — S(π) vs ΔL(π)
# ─────────────────────────────────────────────────────
def plot_surrogate_quality(df_lut, save_path):
    """
    S(π) = Total_Omega (sum of per-layer Omega for chosen bit-widths)
    ΔL(π) = NMSE degradation relative to full-precision baseline
    """
    col_omega = None
    col_nmse  = None
    for c in df_lut.columns:
        if "omega" in c.lower() or "Omega" in c or "omega" in c:
            col_omega = c
        if "nmse" in c.lower():
            col_nmse = c

    print(f"  Using columns: omega='{col_omega}', nmse='{col_nmse}'")
    if col_omega is None or col_nmse is None:
        print("  [WARN] Could not find omega/nmse columns. Available:", list(df_lut.columns))
        return

    df_plot = df_lut[[col_omega, col_nmse]].dropna()
    nmse_vals = df_plot[col_nmse].values.astype(float)
    omega_vals = df_plot[col_omega].values.astype(float)

    # Baseline = most negative NMSE = full precision (best quality)
    nmse_baseline = nmse_vals.min()       # e.g., -15.34 dB (full precision)
    delta_L = nmse_vals - nmse_baseline   # positive = degradation from FP

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter S(π) vs ΔL(π)
    ax = axes[0]
    sc = ax.scatter(omega_vals, delta_L,
                    c=delta_L, cmap="RdYlGn_r", edgecolors="k", linewidths=0.4,
                    s=60, zorder=3)
    plt.colorbar(sc, ax=ax, label="NMSE degradation (dB)")

    # Fit line
    mask = np.isfinite(omega_vals) & np.isfinite(delta_L)
    if mask.sum() > 2:
        p = np.polyfit(omega_vals[mask], delta_L[mask], 1)
        x_fit = np.linspace(omega_vals[mask].min(), omega_vals[mask].max(), 100)
        ax.plot(x_fit, np.polyval(p, x_fit), "k--", lw=1.5, label="Linear fit")
        corr = np.corrcoef(omega_vals[mask], delta_L[mask])[0, 1]
        ax.set_title(f"ILP Surrogate Quality\nPearson r = {corr:.3f}", fontsize=11)

    ax.set_xlabel("S(π) = Σ Ω_i^(b_i)  [ILP objective]", fontsize=10)
    ax.set_ylabel("ΔL(π) = NMSE degradation (dB)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: NMSE vs BOPs saving
    ax2 = axes[1]
    col_saving = None
    for c in df_lut.columns:
        if "actual" in c.lower() and "sav" in c.lower():
            col_saving = c
            break
        if "saving" in c.lower():
            col_saving = c

    if col_saving:
        saving_vals = df_lut[col_saving].values.astype(float)
        valid = np.isfinite(saving_vals) & np.isfinite(nmse_vals)
        ax2.plot(saving_vals[valid], nmse_vals[valid], "o-", color="steelblue",
                 markersize=5, linewidth=1.5, label="ILP policy")
        ax2.axhline(nmse_baseline, color="gray", linestyle="--", lw=1.5,
                    label=f"FP baseline ({nmse_baseline:.1f} dB)")
        ax2.set_xlabel("BOPs Saving (%)", fontsize=10)
        ax2.set_ylabel("NMSE (dB)", fontsize=10)
        ax2.set_title("NMSE vs BOPs Saving\n(Pareto frontier via ILP)", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.suptitle("ILP Surrogate Validation — MT-AE (COST2100 Outdoor)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────
# Figure 3: Hutchinson Estimator Convergence
# ─────────────────────────────────────────────────────
def plot_hutchinson_convergence(dfs_nc, save_path):
    """
    Compare Ω_i^(INT8) across nc=4, nc=8, nc=32
    Shows that even coarse nc is sufficient for layer ranking.
    """
    if len(dfs_nc) < 2:
        print("  [SKIP] Not enough nc variants found")
        return

    # Find common layers
    keys = list(dfs_nc.keys())
    df_ref = dfs_nc[keys[-1]]   # nc=32 as reference

    bit_col = "Omg_INT8"
    if bit_col not in df_ref.columns:
        bit_col = [c for c in df_ref.columns if "Omg" in c][0]

    fig, axes = plt.subplots(1, len(keys) - 1, figsize=(5 * (len(keys) - 1), 5))
    if len(keys) - 1 == 1:
        axes = [axes]

    ref_vals = df_ref.set_index("Layer")[bit_col]

    for i, key in enumerate(keys[:-1]):
        ax = axes[i]
        df_cmp = dfs_nc[key].set_index("Layer")
        common = ref_vals.index.intersection(df_cmp.index)

        x = ref_vals.loc[common].values.astype(float)
        y = df_cmp.loc[common, bit_col].values.astype(float)

        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x, y = x[valid], y[valid]

        ax.scatter(np.log10(x), np.log10(y), s=40, alpha=0.7,
                   edgecolors="k", linewidths=0.3, color="steelblue")

        lims = [min(np.log10(x).min(), np.log10(y).min()) - 0.2,
                max(np.log10(x).max(), np.log10(y).max()) + 0.2]
        ax.plot(lims, lims, "r--", lw=1.5, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)

        corr = np.corrcoef(np.log10(x), np.log10(y))[0, 1]
        ax.set_title(f"Hutchinson: {key} vs {keys[-1]}\nr = {corr:.4f}", fontsize=11)
        ax.set_xlabel(f"log₁₀(Ω_INT8), {keys[-1]}", fontsize=9)
        ax.set_ylabel(f"log₁₀(Ω_INT8), {key}", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Hutchinson Estimator Convergence — INT8 Sensitivity\n"
                 "Coarse nc suffices for layer ranking", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────
# Figure 4: Trace(H) vs Omega — Dirichlet contribution
# ─────────────────────────────────────────────────────
def plot_trace_vs_omega(df, save_path):
    """
    Shows the two components of Omega:
      Ω_i^(b) = Tr(H_ii) × ||Δθ_i^(b)||²
    Separate effect of curvature (Tr) vs quantization error (||Δθ||²)
    """
    bit_pairs = [("Omg_INT8", "INT8"), ("Omg_INT4", "INT4"), ("Omg_INT2", "INT2")]
    available = [(c, l) for c, l in bit_pairs if c in df.columns]

    if "Trace(H)" not in df.columns:
        print("  [SKIP] Trace(H) column not found")
        return

    tr_vals = df["Trace(H)"].values.astype(float)

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "crimson"]

    for i, (col, label) in enumerate(available):
        ax = axes[i]
        omega_vals = df[col].values.astype(float)
        valid = np.isfinite(tr_vals) & np.isfinite(omega_vals) & (tr_vals > 0) & (omega_vals > 0)

        x, y = tr_vals[valid], omega_vals[valid]
        ax.scatter(np.log10(x), np.log10(y), s=50, alpha=0.75,
                   edgecolors="k", linewidths=0.3, color=colors[i])

        # fit
        p = np.polyfit(np.log10(x), np.log10(y), 1)
        xfit = np.linspace(np.log10(x).min(), np.log10(x).max(), 100)
        ax.plot(xfit, np.polyval(p, xfit), "k--", lw=1.5,
                label=f"slope={p[0]:.2f}")

        ax.set_title(f"Tr̄(H_ii) vs Ω_{label}\n(correlation driven by curvature)", fontsize=10)
        ax.set_xlabel("log₁₀(Tr̄(H_ii))  [curvature]", fontsize=9)
        ax.set_ylabel(f"log₁₀(Ω_{label})", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Curvature vs Sensitivity — Ω_i = Tr(H_ii) × ||Δθ_i||²\n"
                 "MT-AE encoder (COST2100 outdoor)", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Generating HAWQ Validation Figures ===\n")

    print("[Fig 1] Sensitivity Heatmap...")
    plot_sensitivity_heatmap(
        df_sens,
        os.path.join(PLOT_DIR, "hawq_fig1_sensitivity_heatmap.png")
    )

    print("[Fig 2] ILP Surrogate Quality...")
    plot_surrogate_quality(
        df_lut,
        os.path.join(PLOT_DIR, "hawq_fig2_surrogate_quality.png")
    )

    print("[Fig 3] Hutchinson Convergence...")
    plot_hutchinson_convergence(
        dfs_nc,
        os.path.join(PLOT_DIR, "hawq_fig3_hutchinson_convergence.png")
    )

    print("[Fig 4] Trace vs Omega...")
    plot_trace_vs_omega(
        df_sens,
        os.path.join(PLOT_DIR, "hawq_fig4_trace_vs_omega.png")
    )

    print("\nAll done. Figures saved to:", PLOT_DIR)
