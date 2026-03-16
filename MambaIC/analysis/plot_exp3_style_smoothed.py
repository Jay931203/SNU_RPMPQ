"""
Exp3-style comparison plot: tercile NMSE vs BOPs Saving for three state descriptors.

Monotonic smoothing: right-to-left sweep keeping running minimum (best NMSE).
This produces the lower envelope — outlier bad policies are replaced by the
better value from the right, ensuring non-decreasing NMSE as saving increases.
Same algorithm as the original exp3 in train_ae.py.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
base = Path(__file__).resolve().parent.parent          # MambaIC/
feat = pd.read_csv(base / 'results/csv/zeta_features.csv')
cal  = pd.read_csv(base / 'results/csv/fitting_raw_data_mamba.csv')

# ── build NMSE matrix (1000 samples x N policies) ─────────────────────
policies = sorted(cal['B'].unique())
n_pol    = len(policies)
savings  = np.array(policies) * 100                    # BOPs Saving (%)

nmse_lin = np.zeros((1000, n_pol))
for k, b in enumerate(policies):
    grp = cal[cal['B'] == b].reset_index(drop=True)
    nmse_lin[:, k] = grp['NMSE_linear'].values[:1000]

nmse_db = 10.0 * np.log10(nmse_lin + 1e-12)

print(f'Policies: {n_pol}, Savings range: {savings[0]:.1f}% - {savings[-1]:.1f}%')


# ── monotonic smoothing (right-to-left running min) ────────────────────
def monotonic_smooth(vals):
    """Right-to-left sweep: keep running min (best NMSE).
    Ensures non-decreasing NMSE as saving increases."""
    smoothed = list(vals)
    current_best = float('inf')
    for i in range(len(smoothed) - 1, -1, -1):  # right to left
        if smoothed[i] < current_best:
            current_best = smoothed[i]
        else:
            smoothed[i] = current_best
    return smoothed


# ── descriptors ────────────────────────────────────────────────────────
descriptors = {
    r'$\zeta_{\mathrm{proxy}}$': feat['zeta_proxy'].values,
    r'$\zeta_{\mathrm{full}}$':  feat['zeta_full'].values,
    'Hoyer':                      feat['hoyer'].values,
}

# ── rc params ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       11,
    'axes.labelsize':  12,
    'axes.titlesize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
})

# ── plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

every = max(1, n_pol // 6)

for ax_idx, (name, vals) in enumerate(descriptors.items()):
    # tercile thresholds
    t33, t66 = np.percentile(vals, [33.33, 66.67])
    idx_lo = vals <= t33
    idx_mi = (vals > t33) & (vals <= t66)
    idx_hi = vals > t66

    # Per-group mean NMSE (dB) at each policy
    raw_lo = nmse_db[idx_lo].mean(axis=0)
    raw_mi = nmse_db[idx_mi].mean(axis=0)
    raw_hi = nmse_db[idx_hi].mean(axis=0)
    std_lo = nmse_db[idx_lo].std(axis=0)

    # Apply correct monotonic smoothing (right-to-left running min)
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

    gap_mild = mean_hi[0]  - mean_lo[0]
    gap_aggr = mean_hi[-1] - mean_lo[-1]
    within_std = std_lo.mean()

    ax.set_xlabel('BOPs Saving (%)')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title(f'{name}\n'
                 f'Gap: {gap_mild:.1f} / {gap_aggr:.1f} dB,  '
                 f'Std: {within_std:.1f} dB')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(84.5, 95.5)

plt.tight_layout()
out = base / 'results/plots/exp3_style_comparison.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

# ── summary stats ──────────────────────────────────────────────────────
print(f'\nPolicies: {n_pol}, Savings range: {savings[0]:.1f}% - {savings[-1]:.1f}%')
for name, vals in descriptors.items():
    t33, t66 = np.percentile(vals, [33.33, 66.67])
    print(f'  {name:>25s}  t33={t33:.4f}  t66={t66:.4f}')
