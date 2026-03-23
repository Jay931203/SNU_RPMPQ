"""
Compression Difficulty Metric (ζ) — Illustrative Example Figure
Shows P (normalized energy), K (locality kernels), and trace computation
for compact vs diffuse CSI, demonstrating why compact → small ζ (easy).

Layout:
  Row 0: K_d, K_a, formula box
  Row 1: P_compact, P^T K_d P K_a (compact), diag bar → ζ_compact
  Row 2: P_diffuse, P^T K_d P K_a (diffuse), diag bar → ζ_diffuse

Saves: results/plots/zeta_metric_example.png
       ../../figures/zeta_metric_example.pdf
"""
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'plots')
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), '..', 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "mathtext.fontset": "cm",
})

# ── Parameters ───────────────────────────────────────────────────────────────
N = 8          # Grid size (paper uses N_d=N_a=32; 8 for visual clarity)
tau = 2.0      # Exponential decay constant

# ── Build exponential locality kernel ────────────────────────────────────────
def build_exp_kernel(n, tau):
    """K[i,i'] = exp(-|i-i'|/τ)"""
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    return np.exp(-diff / tau)

K_d = build_exp_kernel(N, tau)
K_a = build_exp_kernel(N, tau)

# ── Synthetic normalized energy maps P ───────────────────────────────────────
def make_P(n, centers, sigmas, weights=None):
    """Create normalized energy map P with Gaussian peaks (sums to 1)."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    P = np.zeros((n, n))
    if weights is None:
        weights = [1.0] * len(centers)
    for (cy, cx), sig, w in zip(centers, sigmas, weights):
        P += w * np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sig**2))
    P = np.maximum(P, 1e-10)
    return P / P.sum()

# Compact: very sharp single peak (dominant LoS path)
P_compact = make_P(N, centers=[(2, 2)], sigmas=[0.45])

# Diffuse: many scattered peaks + strong uniform floor (rich NLoS scattering)
P_diffuse_raw = make_P(N,
    centers=[(0, 6), (2, 0), (4, 4), (6, 1), (7, 7), (1, 3), (5, 6)],
    sigmas=[0.5, 0.6, 0.5, 0.7, 0.4, 0.5, 0.6],
    weights=[1.0, 0.7, 0.9, 0.5, 0.8, 0.6, 0.7])
P_diffuse = P_diffuse_raw + np.full((N, N), 0.008)
P_diffuse /= P_diffuse.sum()

# ── Compute ζ ────────────────────────────────────────────────────────────────
def compute_zeta(P, K_d, K_a):
    M = P.T @ K_d @ P @ K_a       # (N_a × N_a)
    tr = np.trace(M)
    zeta = 1.0 - tr
    diag = np.diag(M)
    return M, tr, zeta, diag

M_c, tr_c, zeta_c, diag_c = compute_zeta(P_compact, K_d, K_a)
M_d, tr_d, zeta_d, diag_d = compute_zeta(P_diffuse, K_d, K_a)

print(f"Compact: tr = {tr_c:.4f}, ζ = {zeta_c:.4f}")
print(f"Diffuse: tr = {tr_d:.4f}, ζ = {zeta_d:.4f}")

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))

gs = GridSpec(3, 4, figure=fig,
             width_ratios=[1, 1, 1, 0.7],
             height_ratios=[0.85, 1, 1],
             hspace=0.40, wspace=0.40)

# -- Helper: heatmap with optional cell annotations ---
def plot_heatmap(ax, data, title, cmap='YlOrRd', annotate_K=False, vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, fontweight='bold', pad=8)
    if annotate_K:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = 'white' if val > 0.55 * (data.max() + data.min()) else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

# ===================== Row 0: Kernels + Formula =====================
ax_kd = fig.add_subplot(gs[0, 0])
plot_heatmap(ax_kd, K_d, r'$\mathbf{K}_d$  (delay kernel)',
             cmap='Blues', annotate_K=True)
ax_kd.set_xlabel(r"$i'$")
ax_kd.set_ylabel(r"$i$")

ax_ka = fig.add_subplot(gs[0, 1])
plot_heatmap(ax_ka, K_a, r'$\mathbf{K}_a$  (angular kernel)',
             cmap='Blues', annotate_K=True)
ax_ka.set_xlabel(r"$j'$")
ax_ka.set_ylabel(r"$j$")

# Formula box
ax_f = fig.add_subplot(gs[0, 2:])
ax_f.axis('off')
formula = (
    r'$\zeta(\mathbf{X}_a) = 1 \;-\; '
    r'\mathrm{tr}\!\left(\mathbf{P}^{\!\top} '
    r'\mathbf{K}_d\, \mathbf{P}\, \mathbf{K}_a\right)$'
    '\n\n'
    r'$\mathbf{P}_{ij} = \dfrac{|X_{ij}|^2}{\sum_{u,v} |X_{uv}|^2}$'
    '    (normalized energy)\n\n'
    r'$[\mathbf{K}]_{i,i\prime} = '
    r'e^{-|i - i\prime|/\tau}$'
    '    (locality kernel)'
    '\n\n'
    r'Small $\zeta$ $\rightarrow$ compact, easy to compress'
    '\n'
    r'Large $\zeta$ $\rightarrow$ diffuse, hard to compress'
)
ax_f.text(0.08, 0.50, formula, transform=ax_f.transAxes,
          fontsize=13, va='center',
          bbox=dict(boxstyle='round,pad=0.6', fc='lightyellow', alpha=0.85))

# ===================== Row 1: Compact CSI =====================
# P_compact
ax_pc = fig.add_subplot(gs[1, 0])
plot_heatmap(ax_pc, P_compact, r'$\mathbf{P}$  (compact)', cmap='YlOrRd')
ax_pc.set_xlabel('Angular $j$')
ax_pc.set_ylabel('Delay $i$')

# P^T K_d P K_a  (compact)
ax_mc = fig.add_subplot(gs[1, 1])
plot_heatmap(ax_mc, M_c, r'$\mathbf{P}^\top \mathbf{K}_d \mathbf{P} \mathbf{K}_a$',
             cmap='Greens')
ax_mc.set_xlabel("$j'$")
ax_mc.set_ylabel("$j$")
# Highlight diagonal with red outlines
for k in range(N):
    ax_mc.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                    fill=False, ec='red', lw=1.8, ls='--'))

# Diagonal bar plot (compact)
ax_dc = fig.add_subplot(gs[1, 2])
ax_dc.bar(range(N), diag_c, color='seagreen', alpha=0.85, edgecolor='darkgreen',
          linewidth=0.8)
ax_dc.set_title(r'$\mathrm{diag}(\cdot)$  — trace elements', fontweight='bold', pad=8)
ax_dc.set_xlabel('Index $j$')
ax_dc.set_ylabel('Value')
ax_dc.set_xticks(range(N))
y_max_c = diag_c.max() * 1.3
ax_dc.set_ylim(0, y_max_c)
for k, v in enumerate(diag_c):
    if v > 0.005:
        ax_dc.text(k, v + y_max_c * 0.02, f'{v:.3f}', ha='center', fontsize=7,
                   color='darkgreen')

# ζ result (compact)
ax_zc = fig.add_subplot(gs[1, 3])
ax_zc.axis('off')
ax_zc.text(0.50, 0.70, r'$\mathrm{tr} = $' + f'{tr_c:.4f}',
           transform=ax_zc.transAxes, fontsize=15, ha='center', fontweight='bold',
           color='seagreen')
ax_zc.text(0.50, 0.42, r'$\zeta = $' + f'{zeta_c:.4f}',
           transform=ax_zc.transAxes, fontsize=22, ha='center', fontweight='bold',
           color='seagreen',
           bbox=dict(boxstyle='round,pad=0.3', fc='#c8f7c5', alpha=0.6))
ax_zc.text(0.50, 0.15, r'$\Leftarrow$ Easy to compress',
           transform=ax_zc.transAxes, fontsize=12, ha='center',
           style='italic', color='seagreen')

# ===================== Row 2: Diffuse CSI =====================
# P_diffuse
ax_pd = fig.add_subplot(gs[2, 0])
plot_heatmap(ax_pd, P_diffuse, r'$\mathbf{P}$  (diffuse)', cmap='YlOrRd')
ax_pd.set_xlabel('Angular $j$')
ax_pd.set_ylabel('Delay $i$')

# P^T K_d P K_a  (diffuse)
ax_md = fig.add_subplot(gs[2, 1])
plot_heatmap(ax_md, M_d, r'$\mathbf{P}^\top \mathbf{K}_d \mathbf{P} \mathbf{K}_a$',
             cmap='Greens')
ax_md.set_xlabel("$j'$")
ax_md.set_ylabel("$j$")
for k in range(N):
    ax_md.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                    fill=False, ec='red', lw=1.8, ls='--'))

# Diagonal bar plot (diffuse)
ax_dd = fig.add_subplot(gs[2, 2])
ax_dd.bar(range(N), diag_d, color='indianred', alpha=0.85, edgecolor='darkred',
          linewidth=0.8)
ax_dd.set_title(r'$\mathrm{diag}(\cdot)$  — trace elements', fontweight='bold', pad=8)
ax_dd.set_xlabel('Index $j$')
ax_dd.set_ylabel('Value')
ax_dd.set_xticks(range(N))
y_max_d = diag_d.max() * 1.3
ax_dd.set_ylim(0, y_max_d)
for k, v in enumerate(diag_d):
    if v > 0.002:
        ax_dd.text(k, v + y_max_d * 0.02, f'{v:.4f}', ha='center', fontsize=7,
                   color='darkred')

# ζ result (diffuse)
ax_zd = fig.add_subplot(gs[2, 3])
ax_zd.axis('off')
ax_zd.text(0.50, 0.70, r'$\mathrm{tr} = $' + f'{tr_d:.4f}',
           transform=ax_zd.transAxes, fontsize=15, ha='center', fontweight='bold',
           color='indianred')
ax_zd.text(0.50, 0.42, r'$\zeta = $' + f'{zeta_d:.4f}',
           transform=ax_zd.transAxes, fontsize=22, ha='center', fontweight='bold',
           color='indianred',
           bbox=dict(boxstyle='round,pad=0.3', fc='#f7c5c5', alpha=0.6))
ax_zd.text(0.50, 0.15, r'$\Leftarrow$ Hard to compress',
           transform=ax_zd.transAxes, fontsize=12, ha='center',
           style='italic', color='indianred')

# ── Row labels on the left margin ────────────────────────────────────────────
fig.text(0.01, 0.82, 'Locality\nKernels', fontsize=12, fontweight='bold',
         rotation=90, va='center', ha='center', color='steelblue')
fig.text(0.01, 0.50, 'Compact\nCSI', fontsize=12, fontweight='bold',
         rotation=90, va='center', ha='center', color='seagreen')
fig.text(0.01, 0.18, 'Diffuse\nCSI', fontsize=12, fontweight='bold',
         rotation=90, va='center', ha='center', color='indianred')

# ── Subtle column-to-column flow labels ──────────────────────────────────────
# (arrows removed — they overlapped with subplots)

plt.suptitle(r'Compression Difficulty Metric $\zeta$: Compact vs Diffuse CSI'
             f'  ($N = {N}$, $\\tau = {tau}$)',
             fontsize=15, fontweight='bold', y=0.99)

# ── Save ─────────────────────────────────────────────────────────────────────
out_png = os.path.join(PLOT_DIR, 'zeta_metric_example.png')
out_pdf = os.path.join(FIG_DIR, 'zeta_metric_example.pdf')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
