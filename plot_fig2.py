"""
Fig. 2  —  RP-MPQ NMSE Gain over Uniform Quantization
ΔNMSE = NMSE_Uniform_interp(s) − NMSE_RP-MPQ(s)   [positive = RP-MPQ better]
CR=1/4, Outdoor. 4 models.
"""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif'})

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

def calc_saving(wb):
    return (1 - wb * ACT_BITS / (32 * 32)) * 100

def uniform_interp(model, s_arr):
    """Linearly interpolate uniform NMSE at arbitrary saving values."""
    sv_pts = np.array([calc_saving(b) for b in WB_LIST])   # [75, 87.5, 93.75]
    nm_pts = np.array([NMSE_UNIF[model][b] for b in WB_LIST])
    return np.interp(s_arr, sv_pts, nm_pts)

def smooth_pareto(saving_arr, nmse_arr):
    df = pd.DataFrame({'sv': saving_arr, 'nm': nmse_arr}).sort_values('sv').reset_index(drop=True)
    best, smoothed = float('inf'), []
    for v in reversed(df['nm'].tolist()):
        if v < best: best = v
        smoothed.append(best)
    df['nm_s'] = smoothed[::-1]
    return df['sv'].values, df['nm_s'].values

MARKERS = {'CsiNet': 's', 'CRNet': 'v', 'CLNet': 'D', 'MT-AE': 'o'}
COLORS  = {'CsiNet': '#1f77b4', 'CRNet': '#9467bd', 'CLNet': '#ff7f0e', 'MT-AE': '#d62728'}
LABELS  = {'CsiNet': 'CsiNet', 'CRNet': 'CRNet', 'CLNet': 'CLNet', 'MT-AE': 'MT-AE (Ours)'}

fig, ax = plt.subplots(figsize=(5.5, 4.2))

for model in ['CsiNet', 'CRNet', 'CLNet', 'MT-AE']:
    c = COLORS[model]; m = MARKERS[model]; lbl = LABELS[model]

    df = rpmpq[model]
    mask = (df['Actual_Saving'] >= 85.0) & (df['Actual_Saving'] <= 95.0)
    sv_p, nm_p = smooth_pareto(df.loc[mask, 'Actual_Saving'].values,
                               df.loc[mask, NMSE_COL[model]].values)

    # ΔNMSE = NMSE_Uniform(s) − NMSE_RP-MPQ(s)  (positive = RP-MPQ better)
    nm_u_interp = uniform_interp(model, sv_p)
    delta = nm_u_interp - nm_p

    n = len(sv_p)
    marks = list(range(0, n, 10)) + ([n - 1] if (n - 1) % 10 != 0 else [])
    lw = 2.0 if model == 'MT-AE' else 1.6
    ax.plot(sv_p, delta, '-', color=c, linewidth=lw, marker=m,
            markersize=5, markevery=marks, label=lbl)

# Reference line at Δ=0 (= uniform quant level)
ax.axhline(0, color='k', linestyle='--', linewidth=1.0, alpha=0.7, label='Uniform Quant (ref)')

# INT8 / INT4 vertical reference lines
for wb, label in [(8, 'INT8'), (4, 'INT4')]:
    sv_x = calc_saving(wb)
    if 85 <= sv_x <= 95:
        ax.axvline(sv_x, color='gray', linestyle=':', linewidth=0.9, alpha=0.6)
        ax.text(sv_x + 0.2, 2, label, ha='left', va='top', fontsize=8, color='dimgray')

ax.set_xlabel('BOPs Saving vs. FP32 (%)')
ax.set_ylabel(r'$\Delta$NMSE: Uniform $-$ RP-MPQ (dB)')
ax.set_xlim(84, 96)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_rpmpq_gain.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_rpmpq_gain.png', bbox_inches='tight', dpi=300)
print('Saved → figures/fig_rpmpq_gain.pdf/png')

# ── Print table of delta at INT8 and INT4 reference ──────────────────────────
print()
print('dNMSE (Uniform - RP-MPQ) at INT8/INT4 reference savings:')
print(f"{'Model':10}  {'@INT8(87.5%)':>14}  {'@INT4(93.75%)':>14}")
print('-' * 44)
for model in ['CsiNet', 'CRNet', 'CLNet', 'MT-AE']:
    df = rpmpq[model]
    mask = (df['Actual_Saving'] >= 85.0) & (df['Actual_Saving'] <= 95.0)
    sv_p, nm_p = smooth_pareto(df.loc[mask, 'Actual_Saving'].values,
                               df.loc[mask, NMSE_COL[model]].values)
    nm_u_interp = uniform_interp(model, sv_p)
    delta = nm_u_interp - nm_p

    idx8  = np.argmin(np.abs(sv_p - calc_saving(8)))
    idx4  = np.argmin(np.abs(sv_p - calc_saving(4)))
    print(f"{model:10}  {delta[idx8]:>+12.2f} dB  {delta[idx4]:>+12.2f} dB")
