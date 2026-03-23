"""
Delta-L Decomposition Figure — 3-panel (paper-quality)
  (a) S(pi) vs DeltaL(pi)        : policy-level, ordering preserved despite large R
  (b) Omega_exact vs Omega_trace  : per-layer, trace badly ranks layers (rho~0.21)
  (c) Omega_exact vs DeltaL_meas  : per-layer, exact correctly ranks layers (rho~0.49)

  Data: results/csv/delta_L_decomposition.csv
        results/csv/hawq_exact_omega.csv
        results/csv/per_layer_dL.csv
  Saves: results/plots/delta_L_decomposition.png
         ../figures/delta_L_decomposition.pdf
"""
import matplotlib
matplotlib.use('Agg')

import os, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

SCRIPT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DL        = os.path.join(SCRIPT_DIR, 'results', 'csv', 'delta_L_decomposition.csv')
CSV_OMEGA     = os.path.join(SCRIPT_DIR, 'results', 'csv', 'hawq_exact_omega.csv')
CSV_PER_LAYER = os.path.join(SCRIPT_DIR, 'results', 'csv', 'per_layer_dL.csv')
PLOT_DIR      = os.path.join(SCRIPT_DIR, 'results', 'plots')
FIG_DIR       = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── Load policy-level ──────────────────────────────────────────
with open(CSV_DL, newline='') as f:
    data = {int(r['bits']): {'dL': float(r['delta_L']), 'S': float(r['S_exact'])}
            for r in csv.DictReader(f)}

with open(CSV_OMEGA, newline='') as f:
    omega_rows = {r['Layer']: r for r in csv.DictReader(f)}

def layer_type(name):
    if 'fc_part' in name: return 'FC'
    elif 'conv' in name.lower() or name in ('stem.0', 'proj_conv'): return 'Conv'
    else: return 'Linear proj.'

def match_omega_name(param_base, omega_dict):
    for hawq_name in sorted(omega_dict.keys(), key=len, reverse=True):
        if param_base.endswith(hawq_name) or param_base == hawq_name:
            return hawq_name
    return None

# ── Panel (b): Omega_exact vs Omega_trace ────────────────────
ex_b, tr_b, lt_b = [], [], []
for name, r in omega_rows.items():
    e, t = float(r['ExactOmg_INT4']), float(r['Omg_INT4'])
    if e > 1e-10 and t > 0:
        ex_b.append(e); tr_b.append(t); lt_b.append(layer_type(name))
ex_b  = np.array(ex_b); tr_b = np.array(tr_b); lt_b = np.array(lt_b)
rho_b, _ = spearmanr(ex_b, tr_b)

# ── Panel (c): Omega_exact vs DeltaL_measured ─────────────────
ex_c, dl_c, lt_c = [], [], []
if os.path.isfile(CSV_PER_LAYER):
    with open(CSV_PER_LAYER, newline='') as f:
        per_layer = list(csv.DictReader(f))
    for row in per_layer:
        dL_val = float(row['delta_L_INT4'])
        if dL_val <= 0:
            continue
        base = row['param']
        for suf in ('.weight', '.bias'):
            if base.endswith(suf):
                base = base[:-len(suf)]; break
        hawq_name = match_omega_name(base, omega_rows)
        if hawq_name is None: continue
        om_exact = float(omega_rows[hawq_name]['ExactOmg_INT4'])
        if om_exact <= 1e-10: continue
        ex_c.append(om_exact); dl_c.append(dL_val)
        lt_c.append(layer_type(hawq_name))
ex_c = np.array(ex_c); dl_c = np.array(dl_c); lt_c = np.array(lt_c)
rho_c, _ = spearmanr(ex_c, dl_c)

print(f'Panel (b): rho={rho_b:.3f}  ({len(ex_b)} layers)')
print(f'Panel (c): rho={rho_c:.3f}  ({len(ex_c)} layers)')

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 9.5, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_S  = '#1565C0'
C_R  = '#E65100'
TYPE_STYLE = {
    'FC':           {'color': '#1565C0', 'marker': 'o', 's': 28},
    'Conv':         {'color': '#E65100', 'marker': 's', 's': 50},
    'Linear proj.': {'color': '#2E7D32', 'marker': '^', 's': 50},
}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.1))

# ── Panel (a): S(pi) vs DeltaL(pi) ───────────────────────────
dL8, dL4 = data[8]['dL'], data[4]['dL']
S8,  S4  = data[8]['S'],  data[4]['S']

xy_lo = min(S8, dL8) * 0.3
xy_hi = max(S4, dL4) * 3
xy_r  = np.logspace(np.log10(xy_lo), np.log10(xy_hi), 100)
ax1.plot(xy_r, xy_r, color='#aaa', lw=1.0, ls='--', zorder=1)

x_fit = np.logspace(np.log10(S8) - 0.3, np.log10(S4) + 0.3, 100)
slope = (np.log10(dL4) - np.log10(dL8)) / (np.log10(S4) - np.log10(S8))
ic    = np.log10(dL8) - slope * np.log10(S8)
y_fit = 10 ** (slope * np.log10(x_fit) + ic)
ax1.plot(x_fit, y_fit, 'k-', lw=1.0, alpha=0.35, zorder=2)
ax1.text(x_fit[-1] * 0.55, y_fit[-1] * 1.6,
         fr'slope $= {slope:.2f}$', fontsize=8, color='#555', ha='right')

ax1.text(S8 * 3.2,  dL8 * 0.85, 'INT8', fontsize=9.5, color=C_S,
         fontweight='bold', va='center')
ax1.text(S4 * 0.32, dL4 * 1.30, 'INT4', fontsize=9.5, color=C_R,
         fontweight='bold', ha='right')

ax1.legend(handles=[
    mpatches.Patch(color=C_S, label=r'$S(\pi)$'),
    mpatches.Patch(color=C_R, label=r'$R(\pi)$'),
], fontsize=9, framealpha=0.9, handlelength=1.2, loc='upper left', borderpad=0.5)

ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel(r'$S(\pi)$')
ax1.set_ylabel(r'$\Delta\mathcal{L}(\pi)$')
ax1.set_title(r'(a) $S(\pi)$ vs. $\Delta\mathcal{L}(\pi)$')
ax1.grid(alpha=0.2, which='both', ls=':', zorder=0)

# ── Panel (b): Omega_exact vs Omega_trace ────────────────────
def draw_scatter(ax, x_arr, y_arr, lt_arr, rho, xlabel, ylabel, title):
    for ltype, style in TYPE_STYLE.items():
        mask = lt_arr == ltype
        if not mask.any(): continue
        ax.scatter(x_arr[mask], y_arr[mask],
                   color=style['color'], marker=style['marker'],
                   s=style['s'], alpha=0.82, zorder=3,
                   linewidths=0.4, edgecolors='white', label=ltype)
    lo = 10 ** (min(np.log10(x_arr).min(), np.log10(y_arr).min()) - 0.5)
    hi = 10 ** (max(np.log10(x_arr).max(), np.log10(y_arr).max()) + 0.5)
    ref = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(ref, ref, color='#aaa', lw=1.0, ls='--', zorder=1)
    ax.text(0.05, 0.97, fr'Spearman $\rho_s = {rho:.2f}$',
            transform=ax.transAxes, fontsize=9.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='none'))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85,
              handletextpad=0.4, borderpad=0.5)
    ax.grid(alpha=0.2, which='both', ls=':', zorder=0)

draw_scatter(ax2, ex_b, tr_b, lt_b, rho_b,
             r'$\Omega_i^{\rm exact}$ (INT4)',
             r'$\bar{\Omega}_i^{\rm trace}$ (INT4)',
             r'(b) $\Omega^{\rm exact}$ vs. $\bar{\Omega}^{\rm trace}$')

draw_scatter(ax3, ex_c, dl_c, lt_c, rho_c,
             r'$\Omega_i^{\rm exact}$ (INT4)',
             r'$\Delta\mathcal{L}_i$ measured (INT4)',
             r'(c) $\Omega^{\rm exact}$ vs. $\Delta\mathcal{L}_i$')

fig.tight_layout(pad=1.3)

# ── Pie chart insets on panel (a) ─────────────────────────────
PIE = 0.11
for sx, dy in [(S8, dL8), (S4, dL4)]:
    sf   = sx / dy
    disp = ax1.transData.transform((sx, dy))
    axy  = ax1.transAxes.inverted().transform(disp)
    axp  = ax1.inset_axes([axy[0] - PIE/2, axy[1] - PIE/2, PIE, PIE])
    axp.pie([sf, 1 - sf], colors=[C_S, C_R], startangle=90,
            wedgeprops=dict(linewidth=0.7, edgecolor='white'))
    axp.set_aspect('equal')

# ── Save ──────────────────────────────────────────────────────
for p in [
    os.path.join(PLOT_DIR, 'delta_L_decomposition.png'),
    os.path.join(FIG_DIR,  'delta_L_decomposition.pdf'),
    os.path.join(FIG_DIR,  'delta_L_decomposition.png'),
]:
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f'Saved: {p}')

plt.close(fig)
print(f'\n  S/dL: INT8={S8/dL8*100:.1f}%  INT4={S4/dL4*100:.1f}%')
print(f'  Panel (b) rho={rho_b:.3f}  Panel (c) rho={rho_c:.3f}')
