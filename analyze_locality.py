"""
Empirical validation: long-tailed angular locality in COST2100 outdoor CSI.

Figure 1: (a) Angular power decay profile (standalone)
Figure 2: (b) Energy capture — CNN hard truncation vs SSM soft aggregation
          on the SAME y-axis (% captured), plus table output
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

DATA_DIR = "MambaIC/data"
OUT_DIR  = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
Na, Nd = 32, 32

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'font.family': 'serif',
})

def load_csi_centred(sc="out"):
    path = os.path.join(DATA_DIR, f"DATA_Htest{sc}.mat")
    d = sio.loadmat(path)["HT"].astype(np.float32)
    if d.ndim == 2: d = d.reshape(-1, 2, Na, Nd)
    return d - 0.5

def to_power(csi):
    return csi[:, 0]**2 + csi[:, 1]**2


def angular_profile_ew(power):
    max_d = Na // 2
    pf = power.transpose(0, 2, 1).reshape(-1, Na)
    M = pf.shape[0]
    i0 = np.argmax(pf, axis=1)
    col_e = np.sum(pf, axis=1)
    valid = pf[np.arange(M), i0] > 1e-12
    pf, i0, col_e = pf[valid], i0[valid], col_e[valid]
    M = pf.shape[0]

    idx = (i0[:, None] + np.arange(Na)[None, :]) % Na
    sh = np.take_along_axis(pf, idx, axis=1)
    prof = np.zeros((M, max_d + 1))
    prof[:, 0] = sh[:, 0]
    for d in range(1, max_d + 1):
        prof[:, d] = (sh[:, d] + sh[:, Na - d]) / 2.0

    norm = prof / (prof[:, 0:1] + 1e-30)
    w = col_e / (col_e.sum() + 1e-30)
    avg_ew = np.sum(norm * w[:, None], axis=0)
    return avg_ew


def angular_energy_capture(power):
    """Hard truncation: fraction of energy within L bins of peak."""
    max_d = Na // 2
    pf = power.transpose(0, 2, 1).reshape(-1, Na)
    M = pf.shape[0]
    i0 = np.argmax(pf, axis=1)
    total_e = np.sum(pf, axis=1) + 1e-30
    col_e = total_e.copy()

    idx = (i0[:, None] + np.arange(Na)[None, :]) % Na
    sh = np.take_along_axis(pf, idx, axis=1)
    folded = np.zeros((M, max_d + 1))
    folded[:, 0] = sh[:, 0]
    for d in range(1, max_d + 1):
        folded[:, d] = sh[:, d] + sh[:, Na - d]

    cum = np.cumsum(folded, axis=1)
    frac = np.clip(cum / total_e[:, None], 0, 1)
    w = col_e / (col_e.sum() + 1e-30)
    frac_ew = np.sum(frac * w[:, None], axis=0)
    return frac_ew


def main():
    csi = load_csi_centred("out")
    power = to_power(csi)
    print(f"Outdoor: {csi.shape[0]} samples")

    avg_ew = angular_profile_ew(power)
    frac_hard = angular_energy_capture(power)
    d_arr = np.arange(Na // 2 + 1)

    # ════════════════════════════════════════
    # Figure 1: (a) Angular power decay
    # ════════════════════════════════════════
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))

    color_main = '#D84315'
    color_theory = '#555555'

    ax1.semilogy(d_arr, avg_ew, 's-', color=color_main,
                  ms=6, lw=2.2, label='COST2100 Outdoor', zorder=3)

    d_ref = np.arange(0, Na // 2 + 1).astype(float)
    ax1.semilogy(d_ref, 1.0 / (1 + d_ref)**2, '--', color=color_theory,
                  lw=2.5, alpha=0.5, label='$c_0/(1{+}d)^2$')

    ax1.set_xlabel('Angular distance $d = |i - i_0|$ from peak')
    ax1.set_ylabel('Normalised power  $|x_d|^2 / |x_0|^2$')
    ax1.set_title('Angular power decay', fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 16])
    ax1.set_ylim([5e-4, 2])

    fig1.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'fig_angular_decay.{ext}')
        fig1.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig1)

    # ════════════════════════════════════════
    # Figure 2: (b) Energy capture vs receptive field (CNN only)
    # ════════════════════════════════════════
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))

    ax2.plot(d_arr, frac_hard * 100, 's-', color=color_main,
             ms=6, lw=2.2, label='CNN')

    ax2.axvline(x=1, color='#37474F', ls=':', lw=1.5, alpha=0.5)
    ax2.annotate(f'3x3$\\times$1L, 1x params, {frac_hard[1]*100:.0f}% ([1]CsiNet)',
                 (1.3, frac_hard[1]*100 - 6),
                 fontsize=9.5, color='#37474F')
    ax2.axvline(x=3, color='#607D8B', ls=':', lw=1.5, alpha=0.5)
    ax2.annotate(f'7x7$\\times$1L, 5.4x params, {frac_hard[3]*100:.0f}%',
                 (3.3, frac_hard[3]*100 - 4),
                 fontsize=9.5, color='#607D8B')

    # SSM effective capture — learned alpha from trained model
    # Extracted from csi_mamba_AE_M32_bits0_chunked_ss2d_222_out
    # alpha = softplus(dt_bias) * exp(A_log) per sequence step
    alpha_med = 0.079   # median across encoder layers
    alpha_q25 = 0.035   # 25th pct (slow decay channels)
    alpha_q75 = 0.273   # 75th pct (fast decay channels)

    color_ssm = '#1565C0'
    ssm_caps = {}
    for a in [alpha_q25, alpha_med, alpha_q75]:
        w = np.exp(-a * d_arr)
        ssm_caps[a] = np.sum(w * avg_ew) / (np.sum(avg_ew) + 1e-30)

    ax2.axhline(y=ssm_caps[alpha_med]*100, color=color_ssm, ls='--', lw=2.2,
                label=f'MT-AE ($\\alpha$={alpha_med}): '
                      f'{ssm_caps[alpha_med]*100:.0f}%')

    cross_L = float(np.interp(ssm_caps[alpha_med], frac_hard, d_arr))
    cross_L_ceil = int(np.ceil(cross_L))
    cross_kernel = 2 * cross_L_ceil + 1
    print(f"\nSSM (trained alpha={alpha_med}): {ssm_caps[alpha_med]*100:.1f}%,"
          f" equiv CNN L={cross_L:.1f}")

    # Mark intersection: same format as L=1, L=3
    ax2.axvline(x=cross_L_ceil, color=color_ssm, ls=':', lw=1.5, alpha=0.5)
    ax2.annotate(
        f'3x3$\\times${cross_L_ceil}L, {cross_L_ceil}x params, '
        f'{frac_hard[cross_L_ceil]*100:.0f}%',
        (cross_L_ceil + 0.3, ssm_caps[alpha_med]*100 - 5),
        fontsize=9.5, color=color_ssm,
    )

    ax2.set_xlabel('Receptive-field radius $L$ (angular bins)')
    ax2.set_ylabel('Angular energy captured (%)')
    ax2.set_title('(b) Energy capture vs receptive field (COST2100 Outdoor)', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 16])
    ax2.set_ylim([0, 108])

    fig2.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'fig_locality_analysis.{ext}')
        fig2.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig2)

    # ════════════════════════════════════════
    # Figure 3: (c) Bell curve — CNN hard truncation only
    # ════════════════════════════════════════
    d_sym = np.concatenate([-d_arr[:0:-1], d_arr])       # -16..0..16
    p_sym = np.concatenate([avg_ew[:0:-1], avg_ew])      # symmetric bell

    cnn1_fill = np.where(np.abs(d_sym) <= 1, p_sym, 0)
    ssm_fill  = p_sym * np.exp(-alpha_med * np.abs(d_sym))

    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))

    # Full profile (gray = total energy)
    ax3.fill_between(d_sym, p_sym, alpha=0.12, color='#BDBDBD')
    ax3.plot(d_sym, p_sym, '-', color='#555', lw=2, zorder=3,
             label='Angular profile (100%)')

    # SSM captured (draw first — extends everywhere)
    ax3.fill_between(d_sym, ssm_fill, alpha=0.35, color=color_ssm,
                     label=f'MT-AE ($\\alpha$={alpha_med}): '
                           f'{ssm_caps[alpha_med]*100:.0f}%')

    # CNN L=1 captured (narrower, on top)
    ax3.fill_between(d_sym, cnn1_fill, alpha=0.45, color=color_main,
                     label=f'CNN $L$=1 (3x3): {frac_hard[1]*100:.0f}%')

    # Percentage annotations ON the graph
    import matplotlib.patheffects as pe
    ax3.text(0, 0.45, f'{frac_hard[1]*100:.0f}%',
             ha='center', va='center',
             fontsize=15, fontweight='bold', color='white',
             path_effects=[pe.withStroke(linewidth=3, foreground=color_main)])
    ax3.text(5.5, 0.06, f'{ssm_caps[alpha_med]*100:.0f}%',
             ha='center', va='center',
             fontsize=15, fontweight='bold', color='white',
             path_effects=[pe.withStroke(linewidth=3, foreground=color_ssm)])

    ax3.set_xlabel('Angular distance $d$ from peak')
    ax3.set_ylabel('Normalised power')
    ax3.set_title('CNN truncation vs SSM coverage', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9.5)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim([-16, 16])
    ax3.set_ylim([0, 1.15])

    fig3.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'fig_bell_coverage.{ext}')
        fig3.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig3)

    # ════════════════════════════════════════
    # Summary table
    # ════════════════════════════════════════
    print("\n" + "="*55)
    print("  CNN receptive-field energy capture")
    print("-"*55)
    for L in [1, 2, 3, 5, 8, 16]:
        pct = frac_hard[L] * 100
        kernel = 2 * L + 1
        print(f"  L={L:2d}  ({kernel:2d}x{kernel:2d}):  {pct:5.1f}%")
    print("="*55)


if __name__ == "__main__":
    np.random.seed(42)
    main()
