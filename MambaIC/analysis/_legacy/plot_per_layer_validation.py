"""
Per-layer HAWQ validation figure (paper-quality)
  x: Ωᵢ_exact  — HAWQ block-diagonal Hessian quadratic form (INT4)
  y: ΔLᵢ        — actual NMSE change when layer i alone is INT4-quantized

  Message: Ωᵢ_exact correctly ranks layers by true sensitivity
           → justifies using Ωᵢ as per-layer score in ILP

  Data required:
    results/csv/hawq_exact_omega.csv   (Ωᵢ_exact per layer)
    results/csv/per_layer_dL.csv       (ΔLᵢ_measured per param)

  Saves:
    results/plots/per_layer_validation.png
    ../figures/per_layer_validation.pdf
    ../figures/per_layer_validation.png
"""
import matplotlib
matplotlib.use('Agg')

import os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_OMEGA  = os.path.join(SCRIPT_DIR, 'results', 'csv', 'hawq_exact_omega.csv')
CSV_DL     = os.path.join(SCRIPT_DIR, 'results', 'csv', 'per_layer_dL.csv')
PLOT_DIR   = os.path.join(SCRIPT_DIR, 'results', 'plots')
FIG_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── Load hawq_exact_omega ──────────────────────────────────────
with open(CSV_OMEGA, newline='') as f:
    omega_rows = {r['Layer']: r for r in csv.DictReader(f)}

# ── Load per_layer_dL ──────────────────────────────────────────
with open(CSV_DL, newline='') as f:
    dL_rows = list(csv.DictReader(f))

# ── Name matching: hawq short name ↔ full param name ──────────
# hawq names: "stem.0", "layers.0.vss.1.in_proj", "fc_part0", ...
# param names: "encoder.stem.0.weight", "fc_part0.weight", etc.
def match_omega(param_name, omega_dict):
    """Return (omega_exact, hawq_layer_name) or None if no match."""
    # Strip trailing .weight / .bias
    base = param_name
    for suffix in ('.weight', '.bias'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    # Try direct suffix match (most specific first)
    for hawq_name in sorted(omega_dict.keys(), key=len, reverse=True):
        if base.endswith(hawq_name) or base == hawq_name:
            return omega_dict[hawq_name], hawq_name
    return None, None

# ── Build matched pairs ────────────────────────────────────────
omega_vals, dL_vals, layer_types, matched_names = [], [], [], []

for row in dL_rows:
    param  = row['param']
    dL_val = float(row['delta_L_INT4'])

    if dL_val <= 0:
        continue  # skip layers that actually improved (likely noise)

    om_row, hawq_name = match_omega(param, omega_rows)
    if om_row is None:
        continue

    om_exact = float(om_row['ExactOmg_INT4'])
    if om_exact <= 1e-10:
        continue  # skip near-zero exact values

    # Layer type
    if 'fc_part' in param:
        ltype = 'FC'
    elif 'conv' in param.lower():
        ltype = 'Conv'
    else:
        ltype = 'Linear proj.'

    omega_vals.append(om_exact)
    dL_vals.append(dL_val)
    layer_types.append(ltype)
    matched_names.append(hawq_name)

omega_vals  = np.array(omega_vals)
dL_vals     = np.array(dL_vals)
layer_types = np.array(layer_types)
matched_names = np.array(matched_names)

print(f'Matched {len(omega_vals)} layers.')
rho_s, p_val = spearmanr(omega_vals, dL_vals)
print(f'Spearman rho = {rho_s:.3f}  (p={p_val:.3f})')

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

TYPE_STYLE = {
    'FC':           {'color': '#1565C0', 'marker': 'o', 's': 32},
    'Conv':         {'color': '#E65100', 'marker': 's', 's': 60},
    'Linear proj.': {'color': '#2E7D32', 'marker': '^', 's': 60},
}

fig, ax = plt.subplots(figsize=(4.8, 4.2))

# ── Scatter ────────────────────────────────────────────────────
for ltype, style in TYPE_STYLE.items():
    mask = layer_types == ltype
    if not mask.any():
        continue
    ax.scatter(omega_vals[mask], dL_vals[mask],
               color=style['color'], marker=style['marker'],
               s=style['s'], alpha=0.82, zorder=3,
               linewidths=0.4, edgecolors='white', label=ltype)

# ── y = x dashed reference ────────────────────────────────────
log_o = np.log10(omega_vals)
log_d = np.log10(dL_vals)
lo = 10 ** (min(log_o.min(), log_d.min()) - 0.5)
hi = 10 ** (max(log_o.max(), log_d.max()) + 0.5)
ref = np.logspace(np.log10(lo), np.log10(hi), 200)
ax.plot(ref, ref, color='#aaa', lw=1.0, ls='--', zorder=1)

# ── Spearman annotation ───────────────────────────────────────
ax.text(0.05, 0.97, fr'Spearman $\rho_s = {rho_s:.2f}$',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='none'))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\Omega_i^{\rm exact}$ (INT4)')
ax.set_ylabel(r'$\Delta\mathcal{L}_i$ measured (INT4)')
ax.set_title(r'Per-layer: $\Omega^{\rm exact}$ vs. $\Delta\mathcal{L}_i$')
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.85,
          handletextpad=0.4, borderpad=0.5)
ax.grid(alpha=0.2, which='both', ls=':', zorder=0)

fig.tight_layout(pad=1.2)

# ── Save ──────────────────────────────────────────────────────
for p in [
    os.path.join(PLOT_DIR, 'per_layer_validation.png'),
    os.path.join(FIG_DIR,  'per_layer_validation.pdf'),
    os.path.join(FIG_DIR,  'per_layer_validation.png'),
]:
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f'Saved: {p}')

plt.close(fig)

# ── Diagnostics ───────────────────────────────────────────────
print('\nTop 10 by Ωᵢ_exact:')
idx_s = np.argsort(-omega_vals)
for k in idx_s[:10]:
    print(f'  {matched_names[k]:40s}  Ω={omega_vals[k]:.4e}  ΔL={dL_vals[k]:.6f}')

print('\nTop 10 by ΔLᵢ_measured:')
idx_d = np.argsort(-dL_vals)
for k in idx_d[:10]:
    print(f'  {matched_names[k]:40s}  Ω={omega_vals[k]:.4e}  ΔL={dL_vals[k]:.6f}')
