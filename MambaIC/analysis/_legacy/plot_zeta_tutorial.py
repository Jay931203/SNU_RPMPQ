"""
zeta tutorial -- diverse P examples, all intermediate matrices.

5 columns (cases) x 5 rows:
  Row 0: P
  Row 1: P^T K_d          (delay-smoothed)
  Row 2: P^T K_d P        (angular correlation via delay)
  Row 3: P^T K_d P K_a    (full product, diagonal highlighted)
  Row 4: diag bar + zeta
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
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "mathtext.fontset": "cm",
})

N = 8
tau = 2.0

def build_K(n, tau):
    idx = np.arange(n)
    return np.exp(-np.abs(idx[:, None] - idx[None, :]) / tau)

K_d = build_K(N, tau)
K_a = build_K(N, tau)

def make_P(n, centers, sigmas, weights=None, floor=0.0):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    P = np.zeros((n, n))
    if weights is None:
        weights = [1.0] * len(centers)
    for (cy, cx), sig, w in zip(centers, sigmas, weights):
        P += w * np.exp(-((y-cy)**2 + (x-cx)**2) / (2*sig**2))
    P = np.maximum(P, 1e-10) + floor
    return P / P.sum()

def compute_all(P):
    S1 = P.T @ K_d             # P^T K_d        (N_a x N_d)
    S2 = S1 @ P                # P^T K_d P      (N_a x N_a)
    S3 = S2 @ K_a              # P^T K_d P K_a  (N_a x N_a)
    d  = np.diag(S3)
    return S1, S2, S3, d, d.sum(), 1 - d.sum()

# ── 5 cases ──────────────────────────────────────────────────────────────────
cases = [
    ('(a) Single peak',
     make_P(N, [(2,2)], [0.45])),
    ('(b) Two peaks (close)',
     make_P(N, [(2,2),(3,4)], [0.5,0.5], [1.0,0.8])),
    ('(c) Two peaks (far)',
     make_P(N, [(1,1),(6,6)], [0.5,0.5])),
    ('(d) Delay band',
     make_P(N, [(2,j) for j in range(8)], [0.4]*8)),
    ('(e) Diffuse',
     make_P(N, [(0,6),(2,0),(4,4),(6,1),(7,7),(1,3),(5,6)],
            [0.5,0.6,0.5,0.7,0.4,0.5,0.6],
            [1.0,0.7,0.9,0.5,0.8,0.6,0.7], floor=0.008)),
]

results = []
for label, P in cases:
    S1, S2, S3, d, tr, z = compute_all(P)
    results.append((label, P, S1, S2, S3, d, tr, z))

zetas = [r[7] for r in results]
z_min, z_max = min(zetas), max(zetas)

def zeta_color(z):
    t = (z - z_min) / (z_max - z_min + 1e-9)
    r = int(46 + t * (198 - 46))
    g = int(125 - t * (125 - 40))
    b = int(50 - t * (50 - 40))
    return f'#{r:02x}{g:02x}{b:02x}'

# ── figure: 6 rows x 5 cols ─────────────────────────────────────────────────
NC = len(cases)
fig = plt.figure(figsize=(4 * NC, 21))
gs = GridSpec(6, NC, figure=fig,
             height_ratios=[1, 1, 1, 1, 1, 0.75],
             hspace=0.32, wspace=0.30)

def hm(ax, data, title, cmap, diag=False):
    im = ax.imshow(data, cmap=cmap, aspect='equal', interpolation='nearest')
    ax.set_title(title, fontweight='bold', fontsize=10, pad=5)
    if diag:
        for k in range(min(data.shape)):
            ax.add_patch(plt.Rectangle((k-.5, k-.5), 1, 1,
                         fill=False, ec='red', lw=1.5, ls='--'))
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── Row 0: Kernels K_d, K_a ───────────────────────────────────────────────────
ax_kd = fig.add_subplot(gs[0, 0])
hm(ax_kd, K_d, r'$\mathbf{K}_d$  ($\tau\!=\!' + f'{tau}$)', 'Blues')
ax_kd.set_ylabel('$i$'); ax_kd.set_xlabel("$i'$")

ax_ka = fig.add_subplot(gs[0, 1])
hm(ax_ka, K_a, r'$\mathbf{K}_a$  ($\tau\!=\!' + f'{tau}$)', 'Blues')
ax_ka.set_ylabel('$j$'); ax_ka.set_xlabel("$j'$")

# remaining cols: formula / description
for ci in range(2, NC):
    ax = fig.add_subplot(gs[0, ci])
    ax.axis('off')
if NC > 2:
    ax_txt = fig.add_subplot(gs[0, 2:])
    ax_txt.axis('off')
    ax_txt.text(0.05, 0.55,
        r'$[\mathbf{K}]_{i,i\prime} = e^{-|i - i\prime|/\tau}$'
        '\n\nLocality kernel: nearby indices '
        r'$\rightarrow$ high,  far $\rightarrow$ low',
        transform=ax_txt.transAxes, fontsize=13, va='center',
        bbox=dict(boxstyle='round,pad=0.5', fc='#e3f2fd', alpha=0.85))

# ── Row 1: P ─────────────────────────────────────────────────────────────────
for i, (label, P, S1, S2, S3, d, tr, z) in enumerate(results):
    ax = fig.add_subplot(gs[1, i])
    hm(ax, P, label, 'YlOrRd')
    if i == 0: ax.set_ylabel('Delay $i$')
    ax.set_xlabel('Angular $j$')

# ── Row 2: P^T K_d ───────────────────────────────────────────────────────────
for i, (label, P, S1, S2, S3, d, tr, z) in enumerate(results):
    ax = fig.add_subplot(gs[2, i])
    hm(ax, S1, r'$\mathbf{P}^\top\!\mathbf{K}_d$', 'Purples')
    if i == 0: ax.set_ylabel('$j$')
    ax.set_xlabel("$i'$")

# ── Row 3: P^T K_d P ─────────────────────────────────────────────────────────
for i, (label, P, S1, S2, S3, d, tr, z) in enumerate(results):
    ax = fig.add_subplot(gs[3, i])
    hm(ax, S2, r'$\mathbf{P}^\top\!\mathbf{K}_d\mathbf{P}$', 'Purples', diag=True)
    if i == 0: ax.set_ylabel('$j$')
    ax.set_xlabel("$j'$")

# ── Row 4: P^T K_d P K_a ─────────────────────────────────────────────────────
for i, (label, P, S1, S2, S3, d, tr, z) in enumerate(results):
    ax = fig.add_subplot(gs[4, i])
    hm(ax, S3, r'$\mathbf{P}^\top\!\mathbf{K}_d\mathbf{P}\mathbf{K}_a$', 'Greens', diag=True)
    if i == 0: ax.set_ylabel('$j$')
    ax.set_xlabel("$j'$")

# ── Row 5: diag bars + zeta ──────────────────────────────────────────────────
ym = max(r[5].max() for r in results) * 1.25
for i, (label, P, S1, S2, S3, d, tr, z) in enumerate(results):
    ax = fig.add_subplot(gs[5, i])
    col = zeta_color(z)
    ax.bar(range(N), d, color=col, alpha=0.85, edgecolor='k', lw=0.5)
    ax.set_ylim(0, ym)
    ax.set_xticks(range(N))
    ax.set_xlabel('$j$')
    if i == 0: ax.set_ylabel('diag value')
    for k, v in enumerate(d):
        if v > d.max() * 0.12:
            ax.text(k, v + ym*0.015, f'{v:.3f}', ha='center', fontsize=6, color=col)
    ax.text(0.98, 0.95,
            f'tr = {tr:.4f}\n' + r'$\zeta$ = ' + f'{z:.4f}',
            transform=ax.transAxes, fontsize=11, ha='right', va='top',
            fontweight='bold', color=col,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec=col, lw=1.5))

# ── row labels ───────────────────────────────────────────────────────────────
labels = [
    (0.92, r'$\mathbf{K}$',                                        '#1565c0'),
    (0.78, r'$\mathbf{P}$',                                        '#d84315'),
    (0.63, r'$\mathbf{P}^\top\!\mathbf{K}_d$',                     '#4a148c'),
    (0.48, r'$\mathbf{P}^\top\!\mathbf{K}_d\mathbf{P}$',           '#4a148c'),
    (0.33, r'$\times\mathbf{K}_a$',                                 '#2e7d32'),
    (0.11, r'diag $\rightarrow \zeta$',                             '#1565c0'),
]
for y, txt, c in labels:
    fig.text(0.005, y, txt, fontsize=11, fontweight='bold',
             rotation=90, va='center', color=c)

fig.text(0.50, 0.005,
    r'Easy to compress  $\longleftarrow$   $\zeta$   $\longrightarrow$  Hard to compress',
    fontsize=13, ha='center', va='bottom', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray'))

fig.suptitle(r'$\zeta$ Metric: Step-by-Step Matrix Products  ($N\!=\!8,\;\tau\!=\!2$)',
             fontsize=14, fontweight='bold', y=0.995)

out_png = os.path.join(PLOT_DIR, 'zeta_tutorial.png')
out_pdf = os.path.join(FIG_DIR, 'zeta_tutorial.pdf')
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
for label, P, S1, S2, S3, d, tr, z in results:
    print(f"  {label:25s}  tr={tr:.4f}  zeta={z:.4f}")
plt.close()
