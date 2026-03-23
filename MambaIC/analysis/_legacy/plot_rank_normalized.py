import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

base = Path(__file__).resolve().parent.parent
feat = pd.read_csv(base / 'results/csv/zeta_features.csv')
cal = pd.read_csv(base / 'results/csv/fitting_raw_data_mamba.csv')

policies = sorted(cal['B'].unique())
n_pol = len(policies)
nmse_lin = np.zeros((1000, n_pol))
for k, b in enumerate(policies):
    grp = cal[cal['B'] == b].reset_index(drop=True)
    nmse_lin[:, k] = grp['NMSE_linear'].values[:1000]

savings = np.array(policies) * 100
mid_idx = n_pol // 2
nmse_db = 10 * np.log10(nmse_lin[:, mid_idx] + 1e-12)

zp = feat['zeta_proxy'].values
zf = feat['zeta_full'].values
ho = feat['hoyer'].values

to_pct = lambda x: rankdata(x) / len(x) * 100
zp_pct = to_pct(zp)
zf_pct = to_pct(zf)
ho_pct = to_pct(-ho)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Scatter
ax = axes[0]
ax.scatter(ho_pct, nmse_db, s=6, alpha=0.25, c='tab:orange', label='Hoyer (flipped)')
ax.scatter(zf_pct, nmse_db, s=6, alpha=0.25, c='tab:cyan', label='zeta_full')
ax.scatter(zp_pct, nmse_db, s=6, alpha=0.35, c='tab:blue', label='zeta_proxy')
ax.set_xlabel('Descriptor Rank Percentile (%)', fontsize=11)
ax.set_ylabel(f'NMSE (dB), saving={savings[mid_idx]:.1f}%', fontsize=11)
ax.set_title('Rank-Normalized Scatter')
ax.legend(fontsize=9, markerscale=3)
ax.grid(alpha=0.2)

# (b) Smoothed binned mean + std
ax = axes[1]
n_bins = 25
sigma = 1.8

for name, pct, color, ls in [
    ('zeta_proxy', zp_pct, 'tab:blue', '-'),
    ('zeta_full', zf_pct, 'tab:cyan', '--'),
    ('Hoyer (flip)', ho_pct, 'tab:orange', '-.'),
]:
    bin_edges = np.linspace(0, 100, n_bins + 1)
    bc, bm, bs = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (pct >= lo) & (pct < hi) if i < n_bins - 1 else (pct >= lo) & (pct <= hi)
        if mask.sum() > 0:
            bc.append((lo + hi) / 2)
            bm.append(nmse_db[mask].mean())
            bs.append(nmse_db[mask].std())

    bc = np.array(bc)
    bm = gaussian_filter1d(np.array(bm), sigma=sigma)
    bs = gaussian_filter1d(np.array(bs), sigma=sigma)

    ax.plot(bc, bm, color=color, ls=ls, lw=2.2, marker='o', ms=3.5, label=name)
    ax.fill_between(bc, bm - bs, bm + bs, alpha=0.1, color=color)

ax.set_xlabel('Descriptor Rank Percentile (%)', fontsize=11)
ax.set_ylabel(f'NMSE (dB), saving={savings[mid_idx]:.1f}%', fontsize=11)
ax.set_title('Binned Mean +/- Std (Smoothed)')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

plt.tight_layout()
out = base / 'results/plots/stage2_rank_normalized.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
