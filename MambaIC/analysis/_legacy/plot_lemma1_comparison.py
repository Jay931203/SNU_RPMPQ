"""
Lemma 1 Empirical Validation - Two-Panel Figure
  (a) SSM state error saturation  (W2/W4/W8/W16)
  (b) W8: Mamba (contractive) vs CsiNet-equiv (no recurrence)

Data sources:
  ACTUAL (preferred):
    results/csv/lemma1_state_errors.csv    ← from extract_lemma1_data.py on Colab
    results/csv/lemma1_output_errors.csv   ← from extract_lemma1_data.py on Colab
  SYNTHETIC fallback:
    Monte Carlo simulation using rho_mamba=0.6016 from trained model

Usage:
  python plot_lemma1_comparison.py

Output:
  results/plots/lemma1_combined.png
  ../figures/lemma1_combined.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(SCRIPT_DIR, "results", "plots")
FIG_DIR  = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
CSV_DIR  = os.path.join(SCRIPT_DIR, "results", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

# ─── SSM parameters from trained Mamba model ────────────
RHO_MAMBA = 0.6016   # mean spectral radius  (csi_mamba_AE_M32_bits0_chunked_ss2d_222_out)
RHO_MAX   = 0.9995   # max  spectral radius

T_MAX    = 64        # sequence length (stage2 = 8×8 positions)
N_TRIALS = 300       # Monte Carlo trials for synthetic fallback


# =====================================================================
# Synthetic simulation helpers (fallback if CSV not available)
# =====================================================================
def quant_noise_amplitude(bits, signal_range=2.0):
    """Half-range of uniform quantization noise: delta/2."""
    return signal_range / (2 ** (bits + 1))


def simulate_ssm_error(rho, bits, T=T_MAX, n_trials=N_TRIALS, seed=0):
    """
    Scalar contractive SSM: e_t = rho * e_{t-1} + nu_t
    nu_t ~ Uniform[-amp, +amp]

    Returns: mean_e (T,), std_e (T,)
    """
    rng  = np.random.default_rng(seed)
    amp  = quant_noise_amplitude(bits)
    noise = rng.uniform(-amp, amp, (n_trials, T))
    e = np.zeros(n_trials)
    abs_e = np.zeros((n_trials, T))
    for t in range(T):
        e = rho * e + noise[:, t]
        abs_e[:, t] = np.abs(e)
    return abs_e.mean(axis=0), abs_e.std(axis=0)


def theoretical_bound(rho, bits, T=T_MAX):
    amp = quant_noise_amplitude(bits)
    t   = np.arange(T)
    return amp * (1 - rho ** (t + 1)) / (1 - rho)


# =====================================================================
# Data loading (actual CSV or synthetic fallback)
# =====================================================================
def load_panel_a_data():
    csv_path = os.path.join(CSV_DIR, "lemma1_state_errors.csv")
    if os.path.exists(csv_path):
        print(f"[Panel a] Using ACTUAL data: {csv_path}")
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        # columns: position, W16, W8, W4, W2
        return {
            16: data[:, 1],
            8:  data[:, 2],
            4:  data[:, 3],
            2:  data[:, 4],
        }, "actual"
    else:
        print("[Panel a] CSV not found - using synthetic simulation (rho from trained model)")
        result = {}
        for b in [16, 8, 4, 2]:
            mean_e, _ = simulate_ssm_error(RHO_MAMBA, b)
            result[b] = mean_e
        return result, "synthetic"


def load_panel_b_data():
    csv_path = os.path.join(CSV_DIR, "lemma1_output_errors.csv")
    if os.path.exists(csv_path):
        print(f"[Panel b] Using ACTUAL data: {csv_path}")
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        # columns: position, mamba_w8, csinet_w8
        mamba_err  = data[:, 1]
        csinet_err = data[:, 2]
        valid = ~np.isnan(csinet_err)
        if valid.all():
            return mamba_err, csinet_err, "actual"
        else:
            print("  CsiNet column has NaN - using synthetic for CsiNet")
            _, csinet_syn = simulate_ssm_error(1.0, 8, seed=2)
            return mamba_err, csinet_syn, "hybrid"
    else:
        print("[Panel b] CSV not found - using synthetic simulation")
        mamba_mean,  _ = simulate_ssm_error(RHO_MAMBA, 8, seed=1)
        csinet_mean, _ = simulate_ssm_error(1.0,       8, seed=2)
        return mamba_mean, csinet_mean, "synthetic"


# =====================================================================
# Panel (a): W2 / W4 / W8 / W16
# =====================================================================
def plot_panel_a(ax, data, mode):
    """State error |e_t| for four bit-widths at rho_mamba."""

    BIT_CONFIGS = [
        (16, '#1A237E', 'W16'),
        (8,  '#1565C0', 'W8'),
        (4,  '#E65100', 'W4'),
        (2,  '#B71C1C', 'W2'),
    ]

    t_arr = np.arange(T_MAX)

    for bits, color, label in BIT_CONFIGS:
        err = data[bits][:T_MAX]
        ax.semilogy(t_arr, np.maximum(err, 1e-8), '-',
                    color=color, lw=2, label=label, zorder=3)

    # Theoretical bound for W2 (smooth line - contrasts with bumpy simulation)
    bound_w2 = theoretical_bound(RHO_MAMBA, 2)
    ax.semilogy(t_arr, bound_w2, '--', color='#B71C1C', lw=1.4, alpha=0.55,
                label='Lemma 1 bound (W2)')

    # Saturation dashed lines for W2 and W8
    for bits, color in [(2, '#B71C1C'), (8, '#1565C0')]:
        sat = quant_noise_amplitude(bits) / (1 - RHO_MAMBA)
        ax.axhline(sat, color=color, ls=':', lw=1.0, alpha=0.35)

    tag = "(actual)" if mode == "actual" else "(synthetic)"
    ax.set_xlabel('Scan position $t$')
    ax.set_ylabel(r'Feature error  (log scale)')
    ax.set_title(fr'(a) SSM error saturation $\rho={RHO_MAMBA}$ {tag}',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_xlim([0, T_MAX - 1])
    ax.grid(True, alpha=0.2, which='both')


# =====================================================================
# Panel (b): W8 fixed - Mamba vs CsiNet
# =====================================================================
def plot_panel_b(ax, mamba_err, csinet_err, mode):
    """Compare contractive Mamba vs CsiNet (no recurrence) at W8."""

    t_arr = np.arange(len(mamba_err))
    c_mamba  = '#1565C0'
    c_csinet = '#C62828'

    # ── Mamba ──
    ax.plot(t_arr, mamba_err, '-', color=c_mamba, lw=2.2, zorder=3,
            label=fr'MT-AE ($\rho={RHO_MAMBA}$, W8)')

    # Saturation level
    sat_mb = quant_noise_amplitude(8) / (1 - RHO_MAMBA)
    ax.axhline(sat_mb, color=c_mamba, ls='--', lw=1.2, alpha=0.5)
    ax.text(t_arr[-1] * 0.55, sat_mb * 1.12,
            f'Saturation ({sat_mb:.4f})',
            fontsize=8.5, color=c_mamba, va='bottom')

    # ── CsiNet / non-contractive ──
    csinet_label = r'CsiNet-equiv ($\rho\!=\!1$, W8)' if mode != "actual" \
                   else 'CsiNet (W8, actual)'
    ax.plot(t_arr, csinet_err[:len(t_arr)], '-', color=c_csinet, lw=2.2, zorder=3,
            label=csinet_label)

    tag = f"({mode})"
    ax.set_xlabel('Scan position $t$')
    ax.set_ylabel(r'Feature error $\|e_t\|$')
    ax.set_title(f'(b) W8: contractive vs non-contractive {tag}\n'
                 r'MT-AE ($\rho<1$)  vs  CsiNet ($\rho=1$)',
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax.set_xlim([0, len(t_arr) - 1])
    ax.grid(True, alpha=0.2)


# =====================================================================
# Main
# =====================================================================
def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Load data (actual CSV or synthetic fallback)
    data_a, mode_a          = load_panel_a_data()
    mamba_b, csinet_b, mode_b = load_panel_b_data()

    plot_panel_a(ax1, data_a, mode_a)
    plot_panel_b(ax2, mamba_b, csinet_b, mode_b)

    fig.tight_layout(pad=1.5, w_pad=3.0)

    for ext, d in [('png', PLOT_DIR), ('pdf', FIG_DIR), ('png', FIG_DIR)]:
        p = os.path.join(d, f'lemma1_combined.{ext}')
        fig.savefig(p, dpi=300)
        print(f"Saved: {p}")

    plt.close(fig)

    print(f"\n[Mode] Panel a: {mode_a}  |  Panel b: {mode_b}")
    if mode_a != "actual" or mode_b != "actual":
        print(">>> Run extract_lemma1_data.py on Colab to get actual model data.")
        print("    Then copy results/csv/lemma1_*.csv here and re-run.")


if __name__ == "__main__":
    main()
