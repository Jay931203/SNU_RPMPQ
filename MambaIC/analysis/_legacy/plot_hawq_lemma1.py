"""
Lemma 1 Validation Figure
  Key message:
    Trace approximation distorts the inter-layer sensitivity RATIO.
    Exact Omega correctly shows Stem >> VSS >> FC.
    Trace incorrectly shows FC >= VSS (due to parameter-count bias).
    => This is why ILP (trace-based) alone is insufficient; KL correction needed.

  Figure: bar chart of Omega_i / FC_avg ratio, exact vs trace, INT4
  Saves: results/plots/hawq_lemma1_scatter.png
         ../figures/hawq_lemma1_scatter.pdf
"""
import matplotlib
matplotlib.use('Agg')   # no GUI popup

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(SCRIPT_DIR, 'results', 'csv', 'hawq_exact_omega.csv')
PLOT_DIR   = os.path.join(SCRIPT_DIR, 'results', 'plots')
FIG_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── Load ─────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ── Representative non-FC layers ─────────────────────────────
names = [
    'stem.0',
    'layers.1.vss.1.out_proj',
    'layers.1.vss.1.in_proj',
    'layers.0.vss.1.out_proj',
    'proj_conv',
    'layers.0.vss.1.in_proj',
]
short_labels = [
    'Stem',
    'VSS out\n(L1)',
    'VSS in\n(L1)',
    'VSS out\n(L0)',
    'Proj\nconv',
    'VSS in\n(L0)',
]

# ── Exact & Trace for each layer (INT4) ───────────────────────
def lookup(col, name):
    return float(df.loc[df['Layer'] == name, col].values[0])

exact_vals = np.array([max(lookup('ExactOmg_INT4', n), 1e-8) for n in names])
trace_vals = np.array([lookup('Omg_INT4', n) for n in names])

# FC aggregate (use median to avoid fc_part31 outlier in trace)
fc_rows      = df[df['Layer'].str.startswith('fc_part')]
fc_exact_ref = fc_rows['ExactOmg_INT4'].median()
fc_trace_ref = fc_rows['Omg_INT4'].median()

# ── Ratios: Omega_i / FC_ref  (i.e., "how much more important than FC?") ────
exact_ratio = exact_vals / fc_exact_ref
trace_ratio = trace_vals / fc_trace_ref

# ── Console sanity check ─────────────────────────────────────
print('── Sensitivity ratio  (Omega_i / FC_median),  INT4 ──')
print(f'  {"Layer":25s}  {"Exact ratio":>12s}  {"Trace ratio":>12s}')
for i, lbl in enumerate(short_labels):
    print(f'  {lbl.replace(chr(10)," "):25s}  '
          f'{exact_ratio[i]:12.1f}  {trace_ratio[i]:12.2f}')
print(f'  {"FC (median)":25s}  {"1.0":>12s}  {"1.0":>12s}')

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 10,
    'legend.fontsize': 9,
})

bar_colors = [
    '#D32F2F',   # Stem
    '#1565C0',   # VSS out L1
    '#1565C0',   # VSS in L1
    '#1565C0',   # VSS out L0
    '#E65100',   # Proj conv
    '#1565C0',   # VSS in L0
]

n   = len(names)
x   = np.arange(n)
w   = 0.34

fig, ax = plt.subplots(figsize=(6.4, 4.2))

# Exact bars (solid)
ax.bar(x - w/2, exact_ratio, w, color=bar_colors, alpha=0.90,
       zorder=3, label=r'Exact $\Omega_i / \Omega_{\rm FC}$')

# Trace bars (hatched)
ax.bar(x + w/2, trace_ratio, w, color=bar_colors, alpha=0.35,
       hatch='//', edgecolor='gray', zorder=2,
       label=r'Trace approx. $\bar{\Omega}_i / \bar{\Omega}_{\rm FC}$')

# FC = 1.0 reference line
ax.axhline(1.0, color='#2E7D32', lw=1.4, ls='--', zorder=4,
           label='FC baseline  = 1')
ax.text(n - 0.1, 1.18, 'FC = 1', color='#2E7D32', fontsize=8.5, ha='right')

# Log y-axis
ax.set_yscale('log')
ax.set_ylim(5e-2, 8e3)

# ── Annotation: trace mis-ranks VSS vs FC ────────────────────
# Trace: VSS in L0 is BELOW 1 (less than FC), while exact is > 1
vss_in_L0_idx = 5   # last bar
t_val = trace_ratio[vss_in_L0_idx]
e_val = exact_ratio[vss_in_L0_idx]

ax.annotate(
    f'Trace: VSS ≈ FC\n(ratio ≈ {t_val:.2f})',
    xy=(vss_in_L0_idx + w/2, t_val),
    xytext=(vss_in_L0_idx - 1.2, 0.18),
    fontsize=8, color='#555',
    arrowprops=dict(arrowstyle='->', color='#999', lw=1.0),
    ha='center',
)

ax.annotate(
    f'Exact: {e_val:.1f}× FC',
    xy=(vss_in_L0_idx - w/2, e_val),
    xytext=(vss_in_L0_idx - 2.0, e_val * 3.5),
    fontsize=8, color='#1565C0',
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.0),
    ha='center',
)

# ── Axes ─────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(short_labels)
ax.set_ylabel(r'$\Omega_i \;/\; \Omega_{\rm FC}$  (log scale)',
              fontsize=10.5)
ax.set_title(
    r'Inter-layer sensitivity ratio: exact $\Omega_i$ vs trace approx. $\bar{\Omega}_i$  (INT4)',
    fontweight='bold', fontsize=10.5,
)

legend_handles = [
    mpatches.Patch(facecolor='#888', alpha=0.90,
                   label=r'Exact $\Omega_i$'),
    mpatches.Patch(facecolor='#888', alpha=0.35, hatch='//', edgecolor='gray',
                   label=r'Trace approx. $\bar{\Omega}_i$'),
    plt.Line2D([0], [0], color='#2E7D32', ls='--', lw=1.4,
               label='FC median (reference)'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=8.5,
          framealpha=0.88, handlelength=1.8)

ax.grid(axis='y', alpha=0.18, which='both', ls=':')
fig.tight_layout(pad=1.1)

# ── Save ─────────────────────────────────────────────────────
out_paths = [
    os.path.join(PLOT_DIR, 'hawq_lemma1_scatter.png'),
    os.path.join(FIG_DIR,  'hawq_lemma1_scatter.pdf'),
    os.path.join(FIG_DIR,  'hawq_lemma1_scatter.png'),
]
for p in out_paths:
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f'Saved: {p}')

plt.close(fig)
print('\nDone — no popup.')
