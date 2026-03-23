"""
HAWQ Validation: S+R Decomposition Bar + Hutchinson Convergence
---
(a) ΔL(π) = S_part + R_part — stacked bar, each bar = one policy π
    S_part = α·S(π), R_part = ΔL(π) − α·S(π)
    α fitted by OLS on non-R-dominant policies
(b) Hutchinson convergence: nc=4 vs nc=32 (Omega_INT8, SSM layers only)

Output:
  results/plots/hawq_figA_SR_clean.png
  results/plots/hawq_figB_hutchinson_clean.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR    = os.path.join(SCRIPT_DIR, "results", "csv")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 10,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

# ─────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────
df_lut  = pd.read_csv(os.path.join(CSV_DIR, "mp_policy_lut_mamba.csv"))
df_nc4  = pd.read_csv(os.path.join(CSV_DIR, "hawq_importance_split_nc4.csv"))
df_nc32 = pd.read_csv(os.path.join(CSV_DIR, "hawq_importance_split.csv"))

print(f"LUT : {len(df_lut)} policies")
print(f"nc4 : {len(df_nc4)} layers")
print(f"nc32: {len(df_nc32)} layers\n")


# ─────────────────────────────────────────────────────────────────────────
# Fig A  ΔL(π) = S(π) + R(π)  stacked bar  (square-ish, clean)
# ─────────────────────────────────────────────────────────────────────────
def plot_SR_stacked(df_lut, save_path):
    nmse_db = df_lut['NMSE_dB'].values.astype(float)
    omega   = df_lut['Total_Omega'].values.astype(float)

    # Baseline = all-INT16 policy (smallest Ω, best NMSE)
    baseline_db  = nmse_db[np.argmin(omega)]
    baseline_lin = 10 ** (baseline_db / 10)

    # ΔL(π) in linear NMSE scale
    delta_L = 10 ** (nmse_db / 10) - baseline_lin

    # High-compression regime: Ω > 10 (INT4/INT2 bits present, savings ≈ 88%+)
    saving  = df_lut['Actual_Saving'].values.astype(float)
    mask    = (omega > 10) & (delta_L > 1e-6)
    omega_v  = omega[mask]
    delta_v  = delta_L[mask]
    saving_v = saving[mask]

    # α: scale Ω (Hessian·param² units) → ΔL (linear NMSE) units
    # Use median ratio across all policies (robust to outliers)
    alpha = np.median(delta_v / omega_v)

    S_part = alpha * omega_v                     # Tr(H)·‖Δθ‖² proxy
    R_part = np.maximum(delta_v - S_part, 0.0)  # cross-layer residual (≥ 0)
    S_part = np.minimum(S_part, delta_v)         # cap at ΔL

    # Sort by BOPs saving (ascending)
    order    = np.argsort(saving_v)
    S_s      = S_part[order]
    R_s      = R_part[order]
    saving_s = saving_v[order]

    x      = np.arange(len(S_s))
    xlabs  = [f'{s:.1f}' for s in saving_s]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(x, S_s, color='steelblue', label=r'$S(\pi)$')
    ax.bar(x, R_s, bottom=S_s, color='salmon',  label=r'$R(\pi)$')

    # x-axis: BOPs saving (%)
    tick_step = 2  # show every 2nd label to avoid clutter
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([xlabs[i] for i in range(0, len(xlabs), tick_step)],
                       rotation=45, fontsize=9)
    ax.set_xlabel('BOPs Saving (%)', fontsize=11)
    ax.set_ylabel(r'$\Delta L(\pi)$', fontsize=13)
    ax.set_title(r'$\Delta L(\pi) = S(\pi) + R(\pi)$'
                 '\n(MT-AE encoder, COST2100 outdoor)',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Fig A] Saved: {save_path}")
    print(f"  n={mask.sum()},  α={alpha:.3e}")
    for i in order:
        sf = S_part[i] / delta_v[i] * 100
        sv = saving_v[i]
        print(f"  saving={sv:.1f}%  ΔL={delta_v[i]:.4f}  S={sf:.0f}%  R={100-sf:.0f}%")


# ─────────────────────────────────────────────────────────────────────────
# Fig B  Hutchinson Convergence  nc=4 vs nc=32  (5x5 square)
#   Match layers by (Layer, Params) — only physically identical layers
# ─────────────────────────────────────────────────────────────────────────
def plot_hutchinson_square(df_nc4, df_nc32, save_path):
    bit_col = 'Omg_INT8'

    # Build (Layer, Params) -> Omg_INT8 for each nc version
    kv4  = {(r['Layer'], int(r['Params'])): float(r[bit_col])
            for _, r in df_nc4.iterrows()}
    kv32 = {(r['Layer'], int(r['Params'])): float(r[bit_col])
            for _, r in df_nc32.iterrows()}

    common = set(kv4) & set(kv32)
    keys   = sorted(common, key=lambda k: kv32[k], reverse=True)

    x = np.array([kv32[k] for k in keys])   # nc=32 reference
    y = np.array([kv4[k]  for k in keys])   # nc=4  fast

    valid = (x > 0) & (y > 0)
    x, y   = x[valid], y[valid]
    layers = [k[0] for k, v in zip(keys, valid) if v]

    lx, ly = np.log10(x), np.log10(y)
    corr   = np.corrcoef(lx, ly)[0, 1]
    n      = len(x)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(lx, ly, s=70, color='steelblue',
               edgecolors='k', linewidths=0.5, alpha=0.85, zorder=3)

    lim = (min(lx.min(), ly.min()) - 0.4,
           max(lx.max(), ly.max()) + 0.4)
    ax.plot(lim, lim, 'r--', lw=1.5, label='$y = x$  (perfect agreement)')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(r'$\log_{10}(\Omega_{\mathrm{INT8}})$,  $n_c=32$  (reference)',
                  fontsize=11)
    ax.set_ylabel(r'$\log_{10}(\Omega_{\mathrm{INT8}})$,  $n_c=4$  (fast)',
                  fontsize=11)
    ax.set_title(
        fr'(b) Hutchinson Convergence: $n_c=4$ vs $n_c=32$' + '\n'
        fr'$r={corr:.3f}$ across {n} shared SSM layers',
        fontsize=10.5
    )
    ax.legend(fontsize=9.5, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Fig B] Saved: {save_path}")
    print(f"  n={n}, Pearson r (log-log)={corr:.4f}")
    for lyr, lx_v, ly_v in zip(layers, lx, ly):
        print(f"  {lyr}: nc32={10**lx_v:.5f}, nc4={10**ly_v:.5f}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== HAWQ: S+R Scatter + Hutchinson Convergence ===\n")

    print("[Fig A] ΔL = S + R stacked bar ...")
    plot_SR_stacked(
        df_lut,
        os.path.join(PLOT_DIR, "hawq_figA_SR_clean.png")
    )

    print("\n[Fig B] Hutchinson nc=4 vs nc=32 (square, SSM layers) ...")
    plot_hutchinson_square(
        df_nc4, df_nc32,
        os.path.join(PLOT_DIR, "hawq_figB_hutchinson_clean.png")
    )

    print("\nDone.")
