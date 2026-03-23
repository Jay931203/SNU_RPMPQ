import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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

base = Path(__file__).resolve().parent.parent  # MambaIC/
feat = pd.read_csv(base / 'results/csv/zeta_features.csv')
cal = pd.read_csv(base / 'results/csv/fitting_raw_data_mamba.csv')

policies = sorted(cal['B'].unique())
n_pol = len(policies)
nmse_lin = np.zeros((1000, n_pol))
for k, b in enumerate(policies):
    grp = cal[cal['B'] == b].reset_index(drop=True)
    nmse_lin[:, k] = grp['NMSE_linear'].values[:1000]

mid_idx = n_pol // 2
nmse_db = 10 * np.log10(nmse_lin[:, mid_idx] + 1e-12)
saving_pct = policies[mid_idx] * 100

zp = feat['zeta_proxy'].values
zf = feat['zeta_full'].values
ho = -feat['hoyer'].values  # flip so higher = harder

window = 80

fig, ax = plt.subplots(figsize=(6.5, 4.5))

for name, vals, color, ls, lw in [
    (r'$\zeta_{\mathrm{proxy}}$', zp, '#1f77b4', '-', 2.5),
    (r'$\zeta_{\mathrm{full}}$', zf, '#17becf', '--', 2.0),
    (r'Hoyer (flipped)', ho, '#ff7f0e', '-.', 2.0),
]:
    order = np.argsort(vals)
    nmse_sorted = nmse_db[order]

    roll_mean = pd.Series(nmse_sorted).rolling(window, center=True, min_periods=window//2).mean().values
    roll_std = pd.Series(nmse_sorted).rolling(window, center=True, min_periods=window//2).std().values

    pct = np.linspace(0, 100, len(nmse_sorted))

    ax.plot(pct, roll_mean, color=color, ls=ls, lw=lw, label=name)
    ax.fill_between(pct, roll_mean - roll_std, roll_mean + roll_std, alpha=0.08, color=color)

    # Compute and print swing
    valid = ~np.isnan(roll_mean)
    swing = roll_mean[valid][-1] - roll_mean[valid][0]
    print(f'{name}: swing = {swing:.1f} dB')

ax.set_xlabel('Descriptor Percentile (%)')
ax.set_ylabel(f'NMSE (dB)   [BOPs saving = {saving_pct:.1f}%]')
ax.legend(loc='upper left')
ax.grid(alpha=0.25)
ax.set_xlim(0, 100)

plt.tight_layout()
out_png = base / 'results/plots/conditional_nmse_continuous.png'
out_pdf = base.parent / 'figures' / 'conditional_nmse_continuous.pdf'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png}')
try:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'Saved: {out_pdf}')
except:
    pass
plt.close()
