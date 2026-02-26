"""
Fig. kl_vs_ilp  —  (a) ILP vs KL-Refined (smoothed) + (b) |Prediction Error|
Outputs: figures/kl_vs_ilp.pdf / .png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────
BASE     = 'MambaIC/results/csv/'
RAW_PATH = BASE + 'mp_policy_lut_mamba_raw.csv'
PRN_PATH = BASE + 'mp_policy_lut_mamba_pruned.csv'
OUT_DIR  = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load (prefer pruned = latest extended range) ─────────────────────────
if os.path.exists(PRN_PATH):
    df = pd.read_csv(PRN_PATH).sort_values('Actual_Saving').reset_index(drop=True)
    print(f'[INFO] Loaded pruned LUT: {len(df)} rows')
else:
    df = pd.read_csv(RAW_PATH).sort_values('Actual_Saving').reset_index(drop=True)
    print(f'[WARN] Pruned not found, using raw LUT: {len(df)} rows')

# column names may differ between raw/pruned versions
kl_col  = 'NMSE_KL'
ilp_col = 'NMSE_ILP'

# ── Monotonic smoothing (Pareto hull from high-saving side) ──────────────
def monotonic_smooth(vals):
    out, best = [], float('inf')
    for v in reversed(vals):
        if v < best:
            best = v
        out.append(best)
    return out[::-1]

x        = df['Actual_Saving'].values
kl_raw   = df[kl_col].values
ilp_raw  = df[ilp_col].values

kl_smooth  = np.array(monotonic_smooth(kl_raw.tolist()))
ilp_smooth = np.array(monotonic_smooth(ilp_raw.tolist()))
pred_err   = np.abs(ilp_smooth - kl_smooth)   # |ILP − KL| after smoothing

print(f'Mean |ILP - KL| (smoothed): {pred_err.mean():.4f} dB')
print(f'Max  |ILP - KL| (smoothed): {pred_err.max():.4f} dB')
print(f'Fraction > 0.01 dB: {(pred_err > 0.01).mean()*100:.1f}%')

# ── Plot ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12, 'axes.titlesize': 13,
})

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

# ── (a) ILP vs KL-Refined — solid lines ──────────────────────────────────
ax_a.plot(x, ilp_smooth, 'b-s', label='ILP-predicted', markersize=3, linewidth=1.8, alpha=0.85)
ax_a.plot(x, kl_smooth,  'r-o', label='KL-refined',    markersize=3, linewidth=1.8, alpha=0.85)
ax_a.set_xlabel('BOPs Saving vs. FP32 (%)')
ax_a.set_ylabel('NMSE (dB)')
ax_a.set_title('(a) ILP vs. KL-Refined Policy')
ax_a.legend(fontsize=10, loc='upper left')
ax_a.grid(True, linestyle='--', alpha=0.35)

# ── (b) Discrepancy — fill_between + line ─────────────────────────────────
ax_b.fill_between(x, 0, pred_err, color='steelblue', alpha=0.3)
ax_b.plot(x, pred_err, 'b-', linewidth=1.8, alpha=0.85)
ax_b.set_xlabel('BOPs Saving vs. FP32 (%)')
ax_b.set_ylabel('|NMSE_ILP - NMSE_KL-Ref| (dB)')
ax_b.set_title('(b) ILP vs. KL-Refined Discrepancy')
ax_b.grid(True, linestyle='--', alpha=0.35)

fig.tight_layout()

pdf_out = os.path.join(OUT_DIR, 'kl_vs_ilp.pdf')
png_out = os.path.join(OUT_DIR, 'kl_vs_ilp.png')
fig.savefig(pdf_out, bbox_inches='tight', dpi=300)
fig.savefig(png_out, bbox_inches='tight', dpi=300)
print(f'\nSaved → {pdf_out}')
print(f'Saved → {png_out}')
plt.close(fig)
