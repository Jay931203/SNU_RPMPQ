#!/usr/bin/env python3
"""
plot_corrected_comparison.py
============================
Corrected conditional NMSE and exp3-style plots.

Key fix: Hoyer and zeta tercile boundaries are computed on the FULL 20K test set,
not just the 1000 calibration samples. Percentiles on x-axis also use the full
20K ranking.

Outputs:
  Plot A: results/plots/conditional_nmse_continuous.png  (+ figures/*.pdf)
  Plot B: results/plots/exp3_style_comparison.png
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────
base = Path(__file__).resolve().parent.parent  # MambaIC/
data_dir = base / "data"
csv_dir = base / "results" / "csv"
plot_dir = base / "results" / "plots"
fig_dir = base.parent / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# ── rc params ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
})

# =====================================================================
# STEP 1: Load FULL 20K test set and normalize
# =====================================================================
print("=" * 60)
print("STEP 1: Loading full 20K test set")
print("=" * 60)

mat_path = data_dir / "DATA_Htestout.mat"
mat = sio.loadmat(str(mat_path))
csi_raw = np.array(mat['HT'], dtype=np.float32)
if csi_raw.ndim == 2:
    csi_raw = csi_raw.reshape(-1, 2, 32, 32)
N_full = len(csi_raw)
print(f"Loaded: {mat_path}, shape={csi_raw.shape}")

# Normalize to [0, 1] (matches CsiDataset)
min_val = csi_raw.min()
range_val = csi_raw.max() - min_val + 1e-9
csi_data = (csi_raw - min_val) / range_val
print(f"Normalized: [{min_val:.6f}, {min_val + range_val:.6f}] -> [0, 1]")

# =====================================================================
# STEP 2: Compute descriptors for ALL 20K samples
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2: Computing descriptors for ALL 20K samples")
print("=" * 60)


def build_kernel(n, alpha):
    """K[i,i'] = (1 + |i-i'|)^{-alpha} / row_sum, shape (n, n)."""
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    raw = (1.0 + dist) ** (-alpha)
    K = raw / raw.sum(axis=1, keepdims=True)
    return K


# Build kernels once
K_d = build_kernel(32, alpha=1.0)
K_a = build_kernel(32, alpha=1.0)

# Pre-allocate arrays for 20K samples
hoyer_full = np.zeros(N_full)
zeta_proxy_full = np.zeros(N_full)
zeta_full_full = np.zeros(N_full)

for i in range(N_full):
    x = csi_data[i]  # (2, 32, 32)

    # Hoyer sparsity (same formula as train_ae.py)
    l1 = np.abs(x).sum()
    l2 = np.sqrt((x**2).sum())
    hoyer_full[i] = l1 / (l2 + 1e-8)

    # Energy map (centered)
    x_c = x - 0.5
    energy = x_c[0]**2 + x_c[1]**2  # (32, 32)
    total = energy.sum()
    if total < 1e-12:
        P = np.ones_like(energy) / energy.size
    else:
        P = energy / total

    # zeta_full = 1 - tr(P^T K_d P K_a)
    zeta_full_full[i] = 1.0 - np.trace(P.T @ K_d @ P @ K_a)

    # zeta_proxy = 1 - 0.5*c_d - 0.5*c_a
    p_d = P.sum(axis=1)  # delay marginal
    p_a = P.sum(axis=0)  # angular marginal
    c_d = p_d @ K_d @ p_d
    c_a = p_a @ K_a @ p_a
    zeta_proxy_full[i] = 1.0 - 0.5 * c_d - 0.5 * c_a

    if (i + 1) % 5000 == 0:
        print(f"  Computed {i+1}/{N_full} samples")

print(f"\nFull 20K descriptor stats:")
print(f"  Hoyer:      mean={hoyer_full.mean():.4f}, std={hoyer_full.std():.4f}, "
      f"range=[{hoyer_full.min():.4f}, {hoyer_full.max():.4f}]")
print(f"  zeta_proxy: mean={zeta_proxy_full.mean():.6f}, std={zeta_proxy_full.std():.6f}, "
      f"range=[{zeta_proxy_full.min():.6f}, {zeta_proxy_full.max():.6f}]")
print(f"  zeta_full:  mean={zeta_full_full.mean():.6f}, std={zeta_full_full.std():.6f}, "
      f"range=[{zeta_full_full.min():.6f}, {zeta_full_full.max():.6f}]")

# =====================================================================
# STEP 3: Identify the 1000 calibration samples
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3: Extracting 1000 calibration samples")
print("=" * 60)

fit_indices = np.linspace(0, N_full - 1, 1000, dtype=int)
print(f"fit_indices: {len(fit_indices)} samples, range [{fit_indices[0]}, {fit_indices[-1]}]")

# Calibration subset descriptors
hoyer_cal = hoyer_full[fit_indices]
zeta_proxy_cal = zeta_proxy_full[fit_indices]
zeta_full_cal = zeta_full_full[fit_indices]

# =====================================================================
# STEP 4: Compute FULL 20K tercile boundaries
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4: Computing tercile boundaries from FULL 20K distribution")
print("=" * 60)

descriptors_info = {
    'hoyer': {
        'full': hoyer_full,
        'cal': hoyer_cal,
        'label': 'Hoyer',
        'tex_label': r'Hoyer',
    },
    'zeta_proxy': {
        'full': zeta_proxy_full,
        'cal': zeta_proxy_cal,
        'label': 'zeta_proxy',
        'tex_label': r'$\zeta_{\mathrm{proxy}}$',
    },
    'zeta_full': {
        'full': zeta_full_full,
        'cal': zeta_full_cal,
        'label': 'zeta_full',
        'tex_label': r'$\zeta_{\mathrm{full}}$',
    },
}

for key, info in descriptors_info.items():
    t33 = np.percentile(info['full'], 33.33)
    t66 = np.percentile(info['full'], 66.67)
    info['t33'] = t33
    info['t66'] = t66

    # Assign calibration samples to terciles using FULL boundaries
    cal_vals = info['cal']
    n_lo = np.sum(cal_vals <= t33)
    n_mi = np.sum((cal_vals > t33) & (cal_vals <= t66))
    n_hi = np.sum(cal_vals > t66)
    info['idx_lo'] = cal_vals <= t33
    info['idx_mi'] = (cal_vals > t33) & (cal_vals <= t66)
    info['idx_hi'] = cal_vals > t66

    print(f"  {key:12s}: t33={t33:.6f}, t66={t66:.6f} | "
          f"cal: Lo={n_lo}, Mid={n_mi}, Hi={n_hi}")

# =====================================================================
# STEP 5: Load per-sample NMSE data from fitting_raw_data_mamba.csv
# =====================================================================
print("\n" + "=" * 60)
print("STEP 5: Loading per-sample NMSE data")
print("=" * 60)

cal_df = pd.read_csv(csv_dir / "fitting_raw_data_mamba.csv")
policies = sorted(cal_df['B'].unique())
n_pol = len(policies)
savings = np.array(policies) * 100  # BOPs Saving (%)

# Build NMSE matrix (1000 x n_policies)
nmse_lin = np.zeros((1000, n_pol))
for k, b in enumerate(policies):
    grp = cal_df[cal_df['B'] == b].reset_index(drop=True)
    nmse_lin[:, k] = grp['NMSE_linear'].values[:1000]

nmse_db = 10.0 * np.log10(nmse_lin + 1e-12)

print(f"Policies: {n_pol}, Savings range: {savings[0]:.1f}% - {savings[-1]:.1f}%")

# Pick mid policy for Plot A
mid_idx = n_pol // 2
nmse_db_mid = nmse_db[:, mid_idx]
saving_pct = savings[mid_idx]
print(f"Using policy index {mid_idx} (saving={saving_pct:.1f}%) for continuous plot")

# =====================================================================
# STEP 6: Compute percentiles from FULL 20K ranking
# =====================================================================
print("\n" + "=" * 60)
print("STEP 6: Computing percentiles from FULL 20K ranking")
print("=" * 60)

# For each calibration sample, find its percentile in the full 20K distribution
# percentile = (number of full samples <= this value) / N_full * 100


def compute_full_percentiles(cal_vals, full_vals):
    """For each calibration value, compute its percentile in the full distribution."""
    full_sorted = np.sort(full_vals)
    # Use searchsorted to find rank
    ranks = np.searchsorted(full_sorted, cal_vals, side='right')
    pctiles = ranks / len(full_sorted) * 100.0
    return pctiles


for key, info in descriptors_info.items():
    info['cal_percentile'] = compute_full_percentiles(info['cal'], info['full'])
    print(f"  {key}: cal percentile range [{info['cal_percentile'].min():.1f}%, "
          f"{info['cal_percentile'].max():.1f}%]")

# =====================================================================
# PLOT A: Conditional NMSE Continuous (rolling mean)
# =====================================================================
print("\n" + "=" * 60)
print("PLOT A: Conditional NMSE Continuous")
print("=" * 60)

window = 80

fig, ax = plt.subplots(figsize=(6.5, 4.5))

plot_configs = [
    ('zeta_proxy', '#1f77b4', '-', 2.5),
    ('zeta_full',  '#17becf', '--', 2.0),
    ('hoyer',      '#ff7f0e', '-.', 2.0),
]

for key, color, ls, lw in plot_configs:
    info = descriptors_info[key]
    cal_vals = info['cal']
    cal_pctiles = info['cal_percentile']

    # Sort calibration samples by descriptor value
    order = np.argsort(cal_vals)
    nmse_sorted = nmse_db_mid[order]
    pctile_sorted = cal_pctiles[order]

    # Rolling mean and std
    roll_mean = pd.Series(nmse_sorted).rolling(
        window, center=True, min_periods=window // 2).mean().values
    roll_std = pd.Series(nmse_sorted).rolling(
        window, center=True, min_periods=window // 2).std().values

    ax.plot(pctile_sorted, roll_mean, color=color, ls=ls, lw=lw,
            label=info['tex_label'])
    ax.fill_between(pctile_sorted, roll_mean - roll_std, roll_mean + roll_std,
                    alpha=0.08, color=color)

    # Print swing
    valid = ~np.isnan(roll_mean)
    swing = roll_mean[valid][-1] - roll_mean[valid][0]
    print(f"  {info['tex_label']}: swing = {swing:.1f} dB")

ax.set_xlabel('Descriptor Percentile (%, from full 20K distribution)')
ax.set_ylabel(f'NMSE (dB)   [BOPs saving = {saving_pct:.1f}%]')
ax.legend(loc='upper left')
ax.grid(alpha=0.25)
ax.set_xlim(0, 100)

plt.tight_layout()
out_png = plot_dir / 'conditional_nmse_continuous.png'
out_pdf = fig_dir / 'conditional_nmse_continuous.pdf'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out_png}")
fig.savefig(out_pdf, bbox_inches='tight')
print(f"Saved: {out_pdf}")
plt.close()

# =====================================================================
# PLOT B: Exp3-style Tercile Divergence (3 panels)
# =====================================================================
print("\n" + "=" * 60)
print("PLOT B: Exp3-style Tercile Divergence")
print("=" * 60)


def monotonic_smooth(vals):
    """Right-to-left sweep: keep running min (best NMSE).
    Ensures non-decreasing NMSE as saving increases."""
    smoothed = list(vals)
    current_best = float('inf')
    for i in range(len(smoothed) - 1, -1, -1):
        if smoothed[i] < current_best:
            current_best = smoothed[i]
        else:
            smoothed[i] = current_best
    return smoothed


plt.rcParams.update({
    'legend.fontsize': 9,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
})

panel_order = ['zeta_proxy', 'zeta_full', 'hoyer']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
every = max(1, n_pol // 6)

for ax_idx, key in enumerate(panel_order):
    info = descriptors_info[key]
    idx_lo = info['idx_lo']
    idx_mi = info['idx_mi']
    idx_hi = info['idx_hi']

    # Per-group mean NMSE (dB) at each policy
    raw_lo = nmse_db[idx_lo].mean(axis=0)
    raw_mi = nmse_db[idx_mi].mean(axis=0)
    raw_hi = nmse_db[idx_hi].mean(axis=0)
    std_lo = nmse_db[idx_lo].std(axis=0)

    # Monotonic smoothing
    mean_lo = monotonic_smooth(raw_lo)
    mean_mi = monotonic_smooth(raw_mi)
    mean_hi = monotonic_smooth(raw_hi)

    ax = axes[ax_idx]
    ax.plot(savings, mean_hi, 'r-o', markevery=every, ms=5,
            label='High (top 33%)')
    ax.plot(savings, mean_mi, 'g-D', markevery=every, ms=5,
            label='Mid')
    ax.plot(savings, mean_lo, 'b-^', markevery=every, ms=5,
            label='Low (bot 33%)')

    gap_mild = mean_hi[0] - mean_lo[0]
    gap_aggr = mean_hi[-1] - mean_lo[-1]
    within_std = std_lo.mean()

    n_lo_count = idx_lo.sum()
    n_mi_count = idx_mi.sum()
    n_hi_count = idx_hi.sum()

    ax.set_xlabel('BOPs Saving (%)')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title(f'{info["tex_label"]}  (Lo={n_lo_count}/Mid={n_mi_count}/Hi={n_hi_count})\n'
                 f'Gap: {gap_mild:.1f} / {gap_aggr:.1f} dB,  '
                 f'Std: {within_std:.1f} dB')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(savings[0] - 0.5, savings[-1] + 0.5)

    print(f"  {info['tex_label']}: Lo={n_lo_count}, Mid={n_mi_count}, Hi={n_hi_count} | "
          f"Gap mild={gap_mild:.1f} dB, Gap aggr={gap_aggr:.1f} dB")

plt.tight_layout()
out_b = plot_dir / 'exp3_style_comparison.png'
fig.savefig(out_b, dpi=150, bbox_inches='tight')
print(f"Saved: {out_b}")
plt.close()

# =====================================================================
# Summary comparison: old (1000-only) vs new (20K) tercile boundaries
# =====================================================================
print("\n" + "=" * 60)
print("SUMMARY: Old (1000-only) vs New (20K) tercile boundaries")
print("=" * 60)

for key, info in descriptors_info.items():
    cal_vals = info['cal']
    t33_old = np.percentile(cal_vals, 33.33)
    t66_old = np.percentile(cal_vals, 66.67)
    t33_new = info['t33']
    t66_new = info['t66']

    n_lo_old = np.sum(cal_vals <= t33_old)
    n_lo_new = np.sum(cal_vals <= t33_new)

    print(f"  {key:12s}:")
    print(f"    Old (1000): t33={t33_old:.6f}, t66={t66_old:.6f}  |  Lo={n_lo_old}")
    print(f"    New (20K):  t33={t33_new:.6f}, t66={t66_new:.6f}  |  Lo={n_lo_new}")

print("\nDone!")
