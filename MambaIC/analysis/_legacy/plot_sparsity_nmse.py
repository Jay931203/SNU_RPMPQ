"""
Plot sparsity-stratified NMSE for Mamba encoder.
3 bins: High / Mid / Low sparsity (Hoyer measure, sorted ascending → descending).
X range: 85–95 % BOPs saving (INT8/INT4 reference lines shown).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_CSV  = os.path.join(os.path.dirname(__file__), "results", "csv")
RESULTS_PLOT = os.path.join(os.path.dirname(__file__), "results", "plots")

# ── Paper style (matches plot_fig2.py convention) ────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.labelsize':   12,
    'axes.titlesize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'lines.linewidth':  1.8,
    'lines.markersize': 5,
})

# INT8 / INT4 uniform-quant reference savings
# calc_saving(wb) = (1 - wb * 16 / 1024) * 100
INT8_SAVING  = (1 - 8  * 16 / 1024) * 100   # 87.5 %
INT4_SAVING  = (1 - 4  * 16 / 1024) * 100   # 93.75 %


def main():
    csv_path = os.path.join(RESULTS_CSV, "exp3_results_mamba.csv")
    df = pd.read_csv(csv_path)
    df = df[df['encoder'] == 'mamba'].copy()

    # ── Restrict to 85–95 % range ────────────────────────────────────────────
    mask = (df['Actual_Saving'] >= 85.0) & (df['Actual_Saving'] <= 95.0)
    df   = df[mask].reset_index(drop=True)
    print(f"Plotting {len(df)} rows in 85-95 % range")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    n = len(df)
    every = max(1, n // 6)   # ~6 markers along the curve

    ax.plot(df['Actual_Saving'], df['High_S'],
            color='#d62728', linestyle='-', marker='o', markevery=every,
            label='High Sparsity')
    ax.plot(df['Actual_Saving'], df['Mid_S'],
            color='#2ca02c', linestyle='-', marker='D', markevery=every,
            label='Mid Sparsity')
    ax.plot(df['Actual_Saving'], df['Low_S'],
            color='#1f77b4', linestyle='-', marker='^', markevery=every,
            label='Low Sparsity')

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xlim(84.5, 95.5)
    ax.set_xlabel('BOPs Saving vs. FP32 (%)')
    ax.set_ylabel('NMSE (dB)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    png_out = os.path.join(RESULTS_PLOT, "exp3_nmse_sparsity_mamba_3level.png")
    fig.savefig(png_out, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {png_out}")

    figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    pdf_out = os.path.join(figures_dir, "fig_sparsity_nmse_3level.pdf")
    fig.savefig(pdf_out, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {pdf_out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
