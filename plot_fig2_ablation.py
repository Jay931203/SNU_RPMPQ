"""
Fig. 2 — Offline Policy Refinement Ablation (CR=1/4, outdoor, Mamba-Trans)
(b) only: Discrepancy |NMSE_ILP - NMSE_KL-Ref|, full range, no title
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'MambaIC/results/csv/'
PRN_PATH = BASE + 'mp_policy_lut_mamba_wide_pruned.csv'

df = pd.read_csv(PRN_PATH).sort_values('Actual_Saving').reset_index(drop=True)
print(f'Loaded {len(df)} rows, range {df["Actual_Saving"].iloc[0]:.1f}% ~ {df["Actual_Saving"].iloc[-1]:.1f}%')

x        = df['Actual_Saving'].values
nmse_ilp = df['NMSE_ILP'].values
nmse_kl  = df['NMSE_KL'].values
disc     = np.abs(nmse_ilp - nmse_kl)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12,
})
fig, ax = plt.subplots(figsize=(6.5, 4.0))

ax.fill_between(x, 0, disc, color='steelblue', alpha=0.3)
ax.plot(x, disc, 'b-', linewidth=1.8, alpha=0.85)
ax.set_xlabel('BOPs Saving vs. FP32 (%)')
ax.set_ylabel(r'$|\mathrm{NMSE}_{\mathrm{ILP}} - \mathrm{NMSE}_{\mathrm{KL-Ref}}|$ (dB)')
ax.grid(True, linestyle='--', alpha=0.35)

fig.tight_layout()

os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig2_offline_ablation_mamba.pdf', bbox_inches='tight', dpi=300)
fig.savefig('figures/fig2_offline_ablation_mamba.png', bbox_inches='tight', dpi=300)
os.makedirs('MambaIC/results/plots', exist_ok=True)
fig.savefig('MambaIC/results/plots/fig2_offline_ablation_mamba.png', bbox_inches='tight', dpi=300)
print('Saved -> figures/fig2_offline_ablation_mamba.pdf/png')
