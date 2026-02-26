"""
Fig. 2  —  Online RP-MPQ vs. Static MP: Rate-Based Outage Probability
3 panels: SNR = 10 / 20 / 30 dB
Per panel: 3 QoS thresholds (γ = 0.99 / 0.98 / 0.95)
Solid thick  = Online RP-MPQ
Dashed thin  = Static MP (same budget)
"""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})

static = pd.read_csv('MambaIC/results/csv/exp3_5_results_mamba.csv')
ranc   = pd.read_csv('MambaIC/results/csv/ranc_simulation_results_mamba.csv')

SNR_LIST    = [10, 20, 30]
QOS_LIST    = [0.99, 0.98, 0.95]
QOS_COLORS  = {0.99: '#d62728', 0.98: '#ff7f0e', 0.95: '#1f77b4'}
QOS_LABELS  = {0.99: r'$\gamma=0.99$', 0.98: r'$\gamma=0.98$', 0.95: r'$\gamma=0.95$'}

# Column names for each (SNR, QoS) pair
def out_col(snr, qos):
    q = int(qos * 100)
    return f'Outage_{snr}dB_{q}'

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

for col_idx, snr in enumerate(SNR_LIST):
    ax = axes[col_idx]

    for qos in QOS_LIST:
        c = QOS_COLORS[qos]
        col = out_col(snr, qos)

        # ── Static MP: best (min-outage) policy per saving level, then smooth
        sv_s = static['Actual_Saving'].values
        raw  = static[col].values
        # rolling min (window=3) removes isolated collapse spikes, keeps trend
        best = pd.Series(raw).rolling(3, center=True, min_periods=1).min().values
        # light smooth to remove remaining zigzag
        nm_s = pd.Series(best).rolling(9, center=True, min_periods=1).mean().values
        ax.plot(sv_s, nm_s, '--', color=c, linewidth=1.2, alpha=0.7)

        # ── Online RP-MPQ ────────────────────────────────────────────────────
        sub = ranc[(ranc['SNR_Context'] == snr) & (ranc['QoS_Target'] == qos)
                   ].sort_values('Target_Saving')
        sv_r = sub['Target_Saving'].values
        nm_r = sub[col].values
        ax.plot(sv_r, nm_r, '-', color=c, linewidth=2.0)

    ax.set_xlim(85, 95.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f'SNR = {snr} dB', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel('')   # cleared; shared label added below

axes[0].set_ylabel('Outage Probability', fontsize=10)

# shared x-axis label
fig.supxlabel('BOPs Saving vs. FP32 (%)', fontsize=10, y=0.01)

# ── Legend: Static first, then Online ────────────────────────────────────────
color_handles = [
    Line2D([0], [0], color=QOS_COLORS[q], linewidth=2.0, label=QOS_LABELS[q])
    for q in QOS_LIST
]
style_handles = [
    Line2D([0], [0], color='k', linewidth=1.2, linestyle='--', label='Static MP'),
    Line2D([0], [0], color='k', linewidth=2.0, linestyle='-',  label='Online RP-MPQ'),
]
axes[0].legend(handles=color_handles + style_handles,
               fontsize=8, loc='upper left', ncol=1)

plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_online_outage.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_online_outage.png', bbox_inches='tight', dpi=300)
print('Saved -> figures/fig_online_outage.pdf/png')
