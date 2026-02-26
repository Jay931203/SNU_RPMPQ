"""
Fig. 1  —  (a) Uniform Quantization  |  (b) RP-MPQ Pareto
CR=1/4, Outdoor.  4 models: CsiNet / CRNet / CLNet / MT-AE
(a) solid lines connecting INT16 → INT8 → INT4
(b) RP-MPQ Pareto solid lines (85–95% saving)
"""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})

BASE = 'MambaIC/results/csv/'

rpmpq = {
    'CRNet':  pd.read_csv(BASE + 'mp_policy_lut_crnet_cr4_out.csv'),
    'CLNet':  pd.read_csv(BASE + 'mp_policy_lut_clnet_cr4_out.csv'),
    'CsiNet': pd.read_csv(BASE + 'mp_policy_lut_csinet_cr4_out.csv'),
    'MT-AE':  pd.read_csv(BASE + 'mp_policy_lut_mamba_pruned.csv'),
}
NMSE_COL = {'CRNet': 'NMSE_dB', 'CLNet': 'NMSE_dB', 'CsiNet': 'NMSE_dB', 'MT-AE': 'NMSE_KL'}

ACT_BITS = 16
WB_LIST  = [16, 8, 4]

NMSE_UNIF = {
    'CsiNet': {16: -8.74,  8:  1.46,  4: 19.40},
    'CRNet':  {16: -12.71, 8: -3.57,  4: 10.36},
    'CLNet':  {16: -12.82, 8:  0.15,  4: 23.36},
    'MT-AE':  {16: -15.37, 8: -15.19, 4:  0.03},
}

MARKERS = {'CsiNet': 's', 'CRNet': 'v', 'CLNet': 'D', 'MT-AE': 'o'}
COLORS  = {'CsiNet': '#1f77b4', 'CRNet': '#9467bd', 'CLNet': '#ff7f0e', 'MT-AE': '#d62728'}
LABELS  = {'CsiNet': 'CsiNet', 'CRNet': 'CRNet', 'CLNet': 'CLNet', 'MT-AE': 'MT-AE (Ours)'}

def calc_saving(wb):
    return (1 - wb * ACT_BITS / (32 * 32)) * 100  # 75 / 87.5 / 93.75

def smooth_pareto(saving_arr, nmse_arr):
    df = pd.DataFrame({'sv': saving_arr, 'nm': nmse_arr}).sort_values('sv').reset_index(drop=True)
    best, smoothed = float('inf'), []
    for v in reversed(df['nm'].tolist()):
        if v < best: best = v
        smoothed.append(best)
    df['nm_s'] = smoothed[::-1]
    return df['sv'].values, df['nm_s'].values

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# ── (a) Uniform Quantization ──────────────────────────────────────────────────
ax = axes[0]
for model in ['CsiNet', 'CRNet', 'CLNet', 'MT-AE']:
    c = COLORS[model]; m = MARKERS[model]; lbl = LABELS[model]
    sv_u = [calc_saving(b) for b in WB_LIST]
    nm_u = [NMSE_UNIF[model][b] for b in WB_LIST]
    lw = 2.0 if model == 'MT-AE' else 1.6
    ax.plot(sv_u, nm_u, '-', color=c, marker=m, markersize=7,
            linewidth=lw, label=lbl, zorder=5)

ax.axhline(0, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)

ax.set_xlabel('BOPs Saving vs. FP32 (%)')
ax.set_ylabel('NMSE (dB)')
ax.set_xlim(70, 100)
ax.set_title('(a) Uniform Quantization', fontsize=10)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)

# ── (b) RP-MPQ Pareto ─────────────────────────────────────────────────────────
ax = axes[1]
for model in ['CsiNet', 'CRNet', 'CLNet', 'MT-AE']:
    c = COLORS[model]; m = MARKERS[model]; lbl = LABELS[model]
    df = rpmpq[model]
    mask = (df['Actual_Saving'] >= 85.0) & (df['Actual_Saving'] <= 97.0)
    sv_p, nm_p = smooth_pareto(df.loc[mask, 'Actual_Saving'].values,
                               df.loc[mask, NMSE_COL[model]].values)
    n = len(sv_p)
    marks = list(range(0, n, 10)) + ([n - 1] if (n - 1) % 10 != 0 else [])
    lw = 2.0 if model == 'MT-AE' else 1.6
    ax.plot(sv_p, nm_p, '-', color=c, linewidth=lw, marker=m,
            markersize=5, markevery=marks, label=lbl)

ax.axhline(0, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)

ax.set_xlabel('BOPs Saving vs. FP32 (%)')
ax.set_ylabel('NMSE (dB)')
ax.set_xlim(84, 97.5)
ax.set_title('(b) Offline RP-MPQ', fontsize=10)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_uniform_rpmpq.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_uniform_rpmpq.png', bbox_inches='tight', dpi=300)
print('Saved -> figures/fig_uniform_rpmpq.pdf/png')
