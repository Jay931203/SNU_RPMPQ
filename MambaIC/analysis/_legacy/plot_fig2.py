"""
Temporary script: generate Fig. 2 raw vs monotonic separate figures
from existing mp_policy_lut_mamba_pruned.csv (or raw if available).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_PLOT = 'results/plots'
FIGURES_DIR  = 'results/figures'
os.makedirs(RESULTS_PLOT, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

pruned_path = 'results/csv/mp_policy_lut_mamba_pruned.csv'
raw_path    = 'results/csv/mp_policy_lut_mamba_raw.csv'

# Load raw if available, else fall back to pruned
load_path = raw_path if os.path.exists(raw_path) else pruned_path
is_truly_raw = os.path.exists(raw_path)
if not is_truly_raw:
    print('[WARN] Raw LUT not found -- using pruned (already monotonic).')
    print('       Raw/monotonic plots will look identical.')
    print('       Re-run offline scan with updated code to get raw data.')
else:
    print('[INFO] Raw LUT loaded for comparison.')

df_raw = pd.read_csv(load_path).sort_values('Actual_Saving').reset_index(drop=True)
print(f'Loaded {len(df_raw)} rows from: {load_path}')

# Monotonicity check
kl_vals  = df_raw['NMSE_KL'].tolist()
ilp_vals = df_raw['NMSE_ILP'].tolist()
kl_mono  = all(kl_vals[i] <= kl_vals[i+1]  for i in range(len(kl_vals)-1))
ilp_mono = all(ilp_vals[i] <= ilp_vals[i+1] for i in range(len(ilp_vals)-1))
print(f'Raw NMSE_KL  non-decreasing (monotonic): {kl_mono}')
print(f'Raw NMSE_ILP non-decreasing (monotonic): {ilp_mono}')

# Monotonic smoothing
def monotonic_smooth(vals):
    pruned, best = [], float('inf')
    for v in reversed(vals):
        if v < best:
            best = v
        pruned.append(best)
    return pruned[::-1]

smooth_kl  = monotonic_smooth(kl_vals)
smooth_ilp = monotonic_smooth(ilp_vals)
changed_kl  = sum(1 for a, b in zip(kl_vals,  smooth_kl)  if abs(a - b) > 1e-6)
changed_ilp = sum(1 for a, b in zip(ilp_vals, smooth_ilp) if abs(a - b) > 1e-6)
print(f'Points changed by KL  smoothing: {changed_kl}/{len(kl_vals)}')
print(f'Points changed by ILP smoothing: {changed_ilp}/{len(ilp_vals)}')

df_smooth = df_raw.copy()
df_smooth['NMSE_KL']  = smooth_kl
df_smooth['NMSE_ILP'] = smooth_ilp

# Plot
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def plot_single(df, title, suffix):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df['Actual_Saving'], df['NMSE_ILP'],
            'b--s', label='ILP Prediction', alpha=0.7, markersize=4, linewidth=1.5)
    ax.plot(df['Actual_Saving'], df['NMSE_KL'],
            'r-o', label='KL-Refined', linewidth=2, markersize=4)
    ax.set_xlabel('BOPs Saving (%)')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)
    fig.tight_layout()
    png_out = os.path.join(RESULTS_PLOT, f'exp1_pareto_accuracy_mamba_{suffix}.png')
    pdf_out = os.path.join(FIGURES_DIR,  f'kl_vs_ilp_{suffix}.pdf')
    fig.savefig(png_out, dpi=300)
    fig.savefig(pdf_out, dpi=300)
    print(f'Saved: {png_out}')
    print(f'Saved: {pdf_out}')
    plt.close(fig)

plot_single(df_raw,    '(a) Raw',                'raw')
plot_single(df_smooth, '(b) Monotonic Smoothed', 'monotonic')

# -------------------------------------------------------
# Idea C: ILP Prediction Error vs BOPs Saving
# -------------------------------------------------------
df_raw['pred_error']        = df_raw['NMSE_ILP'] - df_raw['NMSE_KL']   # signed
df_raw['pred_error_abs']    = df_raw['pred_error'].abs()
df_smooth['pred_error']     = df_smooth['NMSE_ILP'] - df_smooth['NMSE_KL']
df_smooth['pred_error_abs'] = df_smooth['pred_error'].abs()

fig, ax = plt.subplots(figsize=(7, 4.5))

x = df_smooth['Actual_Saving']
ax.bar(x, df_smooth['pred_error_abs'], width=0.08, color='steelblue', alpha=0.6, label='|ILP − KL| (abs)')
ax.set_xlabel('BOPs Saving (%)', fontsize=13)
ax.set_ylabel('|ILP Prediction Error| (dB)', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(fontsize=11)

fig.tight_layout()
png_out = os.path.join(RESULTS_PLOT, 'exp1_ilp_pred_error_mamba.png')
pdf_out = os.path.join(FIGURES_DIR,  'ilp_pred_error.pdf')
fig.savefig(png_out, dpi=300)
fig.savefig(pdf_out, dpi=300)
print(f'Saved: {png_out}')

# Print summary stats
print('\n--- Prediction Error Stats (Raw) ---')
print(f'  Mean |error|:  {df_raw["pred_error_abs"].mean():.4f} dB')
print(f'  Max  |error|:  {df_raw["pred_error_abs"].max():.4f} dB')
print(f'  Corr (saving vs |error|): {df_raw["Actual_Saving"].corr(df_raw["pred_error_abs"]):.3f}')
print(f'  ILP over-estimates (ILP > KL): {(df_raw["pred_error"] > 0).sum()} / {len(df_raw)} policies')

print('Done.')
