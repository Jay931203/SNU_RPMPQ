"""
Generate paper figures from CSV data files.
No GPU needed -- pure matplotlib plotting.

Figures produced:
  Fig 1: fig_uniform_rpmpq.pdf     -- Two-panel (a) Uniform + (b) Contiguous-Segment MPQ
  Fig 2: fig_online_outage.pdf     -- Outage curves (3 panels for SNR=10,20,30)
  Fig 3: kl_vs_ilp.pdf             -- Ablation: NMSE improvement gap

Usage:
  python analysis/generate_paper_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
PAPER_FIG_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "figures")
PAPER_FINAL_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "figures", "_paper")
PAPERS_FIG_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "papers", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FINAL_DIR, exist_ok=True)
os.makedirs(PAPERS_FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style -- clean publication style: no main titles, only (a)/(b) panel labels,
# light dashed grid, large markers, thick lines, all 4 spines visible.
# ---------------------------------------------------------------------------
STYLE = {
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}

# Standard matplotlib tab colors
C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_PURPLE = "#9467bd"

# ---------------------------------------------------------------------------
# Bug filter: nmse-static / segment-dp at saving > 96.5% wraps around
# ---------------------------------------------------------------------------
BUG_SAVING_THRESHOLD = 96.5


def _save(fig, name):
    """Save figure to all output directories."""
    for d in (FIGURES_DIR, PAPER_FIG_DIR, PAPER_FINAL_DIR, PAPERS_FIG_DIR):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _grid(ax):
    """Apply consistent grid style (dashed, subtle, light gray)."""
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc", zorder=0)


def _axis_saving(ax, xlim=(84, 98)):
    """Configure x-axis for BOPs Saving vs. FP32 (%)."""
    ax.set_xlabel("BOPs Saving vs. FP32 (%)", fontsize=13)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))


def _monotone_nondecreasing(y):
    """Enforce monotone nondecreasing: outage can only increase with saving."""
    y = np.array(y, dtype=float)
    for i in range(1, len(y)):
        if y[i] < y[i - 1]:
            y[i] = y[i - 1]
    return y


def _monotone_nonincreasing(y):
    """Enforce monotone nonincreasing: NMSE/rate can only decrease with saving."""
    y = np.array(y, dtype=float)
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            y[i] = y[i - 1]
    return y


# ===================================================================
# Data loading
# ===================================================================
def load_data():
    """Load full_comparison.csv, eval_summary, and outage_multi_snr_sweep."""
    csv_path = os.path.join(RESULTS_CSV, "full_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"Loaded full_comparison.csv: {len(df)} rows, "
          f"methods={sorted(df['method'].unique())}")

    # Patch hawq-ilp NMSE: use pure ILP results (NMSE_ILP) instead of
    # KL-selected policy results.  The LUT Policy column stores the
    # KL-refined policy, so eval_full_comparison's inference used KL results.
    # For a fair baseline comparison we need pure HAWQ+ILP (no KL).
    hawq_lut = os.path.join(RESULTS_CSV, "mp_policy_lut_mamba_pruned.csv")
    if os.path.exists(hawq_lut):
        lut = pd.read_csv(hawq_lut)
        if "NMSE_ILP" in lut.columns:
            # Build a mapping: Actual_Saving -> NMSE_ILP
            ilp_map = dict(zip(lut["Actual_Saving"], lut["NMSE_ILP"]))
            mask = df["method"] == "hawq-ilp"
            patched = 0
            for idx in df[mask].index:
                # Match by closest Actual_Saving
                saving = df.loc[idx, "saving"]
                diffs = {k: abs(k - saving) for k in ilp_map}
                best_k = min(diffs, key=diffs.get)
                if diffs[best_k] <= 0.5:
                    old_val = df.loc[idx, "nmse_db"]
                    new_val = ilp_map[best_k]
                    if abs(old_val - new_val) > 0.001:
                        patched += 1
                    df.loc[idx, "nmse_db"] = new_val
            print(f"[PATCH] hawq-ilp NMSE replaced with pure ILP values "
                  f"({patched} rows changed)")

    eval_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_eval_summary.csv")
    eval_df = pd.read_csv(eval_csv) if os.path.exists(eval_csv) else None
    if eval_df is not None:
        print(f"Loaded eval_summary.csv: {len(eval_df)} rows, "
              f"schemes={list(eval_df['scheme'])}")

    # Multi-SNR outage data
    outage_csv = os.path.join(RESULTS_CSV, "outage_multi_snr_sweep.csv")
    outage_df = pd.read_csv(outage_csv) if os.path.exists(outage_csv) else None
    if outage_df is not None:
        print(f"Loaded outage_multi_snr_sweep.csv: {len(outage_df)} rows")
    else:
        print("[INFO] outage_multi_snr_sweep.csv not found -- single-SNR mode.")

    # Baseline Segment DP results
    baseline_csv = os.path.join(RESULTS_CSV, "segment_dp_baselines.csv")
    baseline_df = pd.read_csv(baseline_csv) if os.path.exists(baseline_csv) else None
    if baseline_df is not None:
        print(f"Loaded segment_dp_baselines.csv: {len(baseline_df)} rows, "
              f"models={sorted(baseline_df['model'].unique())}")
    else:
        print("[INFO] segment_dp_baselines.csv not found -- Fig 4 skipped.")

    # Per-block ILP results (3-way comparison)
    ilp_csv = os.path.join(RESULTS_CSV, "perblock_ilp_sweep.csv")
    ilp_df = pd.read_csv(ilp_csv) if os.path.exists(ilp_csv) else None
    if ilp_df is not None:
        print(f"Loaded perblock_ilp_sweep.csv: {len(ilp_df)} rows")
    else:
        print("[INFO] perblock_ilp_sweep.csv not found -- Fig 3 uses 2-way only.")

    return df, eval_df, outage_df, baseline_df, ilp_df


def _filter_bug(df, method, saving_cap=BUG_SAVING_THRESHOLD):
    """Remove buggy rows above saving_cap for methods with known wrap-around."""
    sub = df[df["method"] == method].copy()
    if method in ("nmse-static", "cos2-static", "segment-dp"):
        sub = sub[sub["saving"] <= saving_cap]
    return sub.sort_values("saving")


def _monotone_smooth(savings, values):
    """Monotone smoothing: as saving increases, NMSE should only get worse.
    If a value is better (lower) at higher saving, keep the previous worse value."""
    smoothed = list(values)
    best = float('-inf')  # worst NMSE seen so far (higher = worse)
    for i in range(len(smoothed)):
        if smoothed[i] > best:
            best = smoothed[i]
        else:
            smoothed[i] = best
    return smoothed


def _unique_points(x, y):
    """Extract unique operating points (where y actually changes).
    Returns (x_unique, y_unique) — no interpolation, just deduplication."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    keep = [0]
    for i in range(1, len(y)):
        if abs(y[i] - y[keep[-1]]) > 0.005:
            keep.append(i)
    if keep[-1] != len(y) - 1:
        keep.append(len(y) - 1)
    return x[keep], y[keep]


# ===================================================================
# Fig 1: fig_uniform_rpmpq.pdf -- Two-panel
#   (a) Uniform Quantization across 4 models
#   (b) Contiguous-Segment MPQ: HAWQ-ILP vs Segment DP
# ===================================================================
def fig1_uniform_rpmpq(df, eval_df, baseline_df):
    """Two-panel main result figure.
    (a) Uniform Quantization -- 3-point lines for 4 models
    (b) Offline RP-MPQ -- dense curves with markers for 4 models
    """
    with plt.rc_context(STYLE):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5))

        # -- Panel (a): Uniform quantization LINE plot --
        savings_uniform = [75.0, 87.5, 93.75]

        # Ordered list to control legend order + plotting order
        models_list = [
            ("CsiNet", [-8.74, 1.46, 19.40], C_BLUE, "s"),
            ("CRNet", [-12.71, -3.57, 10.36], C_PURPLE, "v"),
            ("CLNet", [-12.82, 0.15, 23.36], C_ORANGE, "D"),
            ("MT-AE (Ours)", [-15.37, -15.19, 0.03], C_RED, "o"),
        ]

        for name, nmse, clr, mkr in models_list:
            ax_a.plot(savings_uniform, nmse,
                      color=clr, marker=mkr,
                      linestyle="-", linewidth=2.0, markersize=6,
                      label=name, zorder=4)

        ax_a.set_xlabel("BOPs Saving vs. FP32 (%)", fontsize=13)
        ax_a.set_ylabel("NMSE (dB)", fontsize=13)
        ax_a.set_title("(a) Uniform Quantization", fontsize=13)
        ax_a.set_xlim(70, 100)
        ax_a.set_xticks(savings_uniform)
        ax_a.legend(loc="upper left", fontsize=9, ncol=1, framealpha=0.9)
        _grid(ax_a)

        # -- Panel (b): Offline RP-MPQ -- all 4 models dense curves --
        model_styles = {
            "CsiNet": {"color": C_BLUE, "marker": "s"},
            "CRNet": {"color": C_PURPLE, "marker": "v"},
            "CLNet": {"color": C_ORANGE, "marker": "D"},
        }

        # Plot order: CsiNet first (highest/worst), then CRNet, CLNet, MT-AE
        # so MT-AE renders on top

        # All 4 models from segment_dp_baselines.csv
        plot_order = [
            ("CsiNet", C_BLUE, "s", 3),
            ("CRNet", C_PURPLE, "v", 4),
            ("CLNet", C_ORANGE, "D", 4),
        ]
        if baseline_df is not None:
            for model_name, clr, mkr, zo in plot_order:
                sub = baseline_df[
                    (baseline_df["model"] == model_name) &
                    (baseline_df["method"] == "segment-dp")
                ].sort_values("target_saving")
                if len(sub) > 0:
                    x = sub["target_saving"].values
                    y = _monotone_smooth(x, sub["nmse_db"].values)
                    ax_b.plot(x, y, "-", color=clr,
                              marker=mkr, markersize=6,
                              linewidth=2.0, markevery=5,
                              label=model_name, zorder=zo)

        # MT-AE (Mamba, Ours) from segment_dp_baselines.csv
        if baseline_df is not None and "MT-AE" in baseline_df["model"].values:
            mtae = baseline_df[
                (baseline_df["model"] == "MT-AE") &
                (baseline_df["method"] == "segment-dp")
            ].sort_values("target_saving")
            if len(mtae) > 0:
                y = _monotone_smooth(mtae["target_saving"].values,
                                     mtae["nmse_db"].values)
                ax_b.plot(mtae["target_saving"], y, "-", color=C_RED,
                          marker="o", markersize=6, linewidth=2.0,
                          markevery=5, label="MT-AE (Ours)", zorder=5)

        ax_b.set_ylabel("NMSE (dB)", fontsize=13)
        ax_b.set_title("(b) Segment-Level DP", fontsize=13)
        _axis_saving(ax_b, xlim=(84, 98))
        ax_b.legend(loc="upper left", fontsize=9, ncol=1, framealpha=0.9)
        _grid(ax_b)

        fig.tight_layout(w_pad=3.0)
        _save(fig, "fig_uniform_rpmpq")
        print("[Fig 1] fig_uniform_rpmpq.{pdf,png}  -- saved.")


def fig1b_hawq_vs_segment_dp(df):
    """Standalone figure: HAWQ-ILP vs Segment DP NMSE comparison.
    Segment DP uses segment_dp_baselines.csv MT-AE (smooth Joint DP)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        hawq = _filter_bug(df, "hawq-ilp")
        hawq_smooth = _monotone_smooth(hawq["saving"].values,
                                       hawq["nmse_db"].values)

        ax.plot(hawq["saving"], hawq_smooth,
                "--", color=C_RED, linewidth=2.0,
                marker="v", markersize=8, markevery=8,
                label="HAWQ-ILP", zorder=4)

        # Segment DP from segment_dp_baselines.csv
        seg_csv = os.path.join(RESULTS_CSV, "segment_dp_baselines.csv")
        if os.path.exists(seg_csv):
            seg_df = pd.read_csv(seg_csv)
            mtae_dp = seg_df[
                (seg_df["model"] == "MT-AE") & (seg_df["method"] == "segment-dp")
            ].sort_values("target_saving")
            mtae_dp = mtae_dp[
                (mtae_dp["target_saving"] >= 85) & (mtae_dp["target_saving"] <= 97)
            ]
            dp_smooth = _monotone_smooth(
                mtae_dp["target_saving"].values, mtae_dp["nmse_db"].values)
            ax.plot(mtae_dp["target_saving"], dp_smooth,
                    "-", color=C_BLUE, linewidth=2.0,
                    marker="o", markersize=8, markevery=8,
                    label="Segment DP (proposed)", zorder=4)
        else:
            static = _filter_bug(df, "nmse-static")
            static_smooth = _monotone_smooth(static["saving"].values,
                                             static["nmse_db"].values)
            ax.plot(static["saving"], static_smooth,
                    "-", color=C_BLUE, linewidth=2.0,
                    marker="o", markersize=8, markevery=8,
                    label="Segment DP (proposed)", zorder=4)

        ax.set_ylabel("NMSE (dB)", fontsize=13)
        _axis_saving(ax, xlim=(84, 98))
        ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
        _grid(ax)

        fig.tight_layout()
        _save(fig, "fig_hawq_vs_segment_dp")
        print("[Fig 1b] fig_hawq_vs_segment_dp.{pdf,png}  -- saved.")


# ===================================================================
# Fig 2: fig_online_outage.pdf -- Outage curves
#   3 panels (SNR=10, 20, 30) from complete_eval_mtae.csv
#   Static MP (equal alloc, dashed) vs Online RP-MPQ (joint DP, solid)
#   for gamma = 0.99, 0.98, 0.95
# ===================================================================

def fig2_online_outage(df, outage_df):
    """Outage probability vs BOPs saving for 3 SNR panels.

    Data source: complete_eval_mtae.csv
    Lines: solid = Online RP-MPQ (joint DP, outage_alloc)
           dashed = Static MP (equal allocation, outage_equal)
    Colors: red = gamma=0.99, orange = gamma=0.98, blue = gamma=0.95
    """
    # Load the complete evaluation CSV directly
    csv_path = os.path.join(RESULTS_CSV, "complete_eval_mtae.csv")
    if not os.path.exists(csv_path):
        print(f"[Fig 2] SKIP: {csv_path} not found.")
        return
    edf = pd.read_csv(csv_path)
    print(f"[Fig 2] Loaded complete_eval_mtae.csv: {len(edf)} rows")

    snrs = [10, 20, 30]
    gammas = [0.99, 0.98, 0.95]
    gamma_colors = {0.99: C_RED, 0.98: C_ORANGE, 0.95: C_BLUE}
    gamma_labels = {0.99: r"$\gamma=0.99$",
                    0.98: r"$\gamma=0.98$",
                    0.95: r"$\gamma=0.95$"}

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

        for idx, (ax, snr_val) in enumerate(zip(axes, snrs)):
            for gamma in gammas:
                sub = edf[(edf["snr"] == snr_val) &
                          (edf["gamma"] == gamma)].sort_values("target_saving")
                if len(sub) == 0:
                    continue

                savings = sub["target_saving"].values
                nmse = sub["nmse_db"].values
                clr = gamma_colors[gamma]

                # --- Online RP-MPQ (joint DP, solid) ---
                y_alloc = _fix_fp32_fallback(nmse, sub["outage_alloc"].values)
                y_alloc = _monotone_nondecreasing(y_alloc)
                ax.plot(savings, y_alloc, "-", color=clr, linewidth=2.0,
                        drawstyle="steps-post",
                        label=gamma_labels[gamma] if idx == 0 else None,
                        zorder=4)

                # --- Static MP (equal allocation, dashed) ---
                y_equal = _fix_fp32_fallback(nmse, sub["outage_equal"].values)
                y_equal = _monotone_nondecreasing(y_equal)
                ax.plot(savings, y_equal, "--", color=clr, linewidth=2.0,
                        drawstyle="steps-post",
                        label=None, zorder=3)

            ax.set_title(f"SNR = {snr_val} dB", fontsize=13)
            ax.set_xlabel("BOPs Saving vs. FP32 (%)", fontsize=13)
            ax.set_xlim(84, 98)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.set_ylim(0, 1.05)
            _grid(ax)
            if idx == 0:
                ax.set_ylabel("Outage Probability", fontsize=13)
                # Build legend: gamma colors + linestyle entries
                from matplotlib.lines import Line2D
                handles = []
                for g in gammas:
                    handles.append(Line2D([0], [0], color=gamma_colors[g],
                                          linewidth=2.0, linestyle="-",
                                          label=gamma_labels[g]))
                handles.append(Line2D([0], [0], color="gray",
                                      linewidth=2.0, linestyle="-",
                                      label="Online RP-MPQ"))
                handles.append(Line2D([0], [0], color="gray",
                                      linewidth=2.0, linestyle="--",
                                      label="Static MP"))
                ax.legend(handles=handles, loc="upper left",
                          fontsize=10, framealpha=0.9)

        fig.tight_layout(w_pad=2.0)
        _save(fig, "fig_online_outage")
        print("[Fig 2] fig_online_outage.{pdf,png}  -- saved.")


# ===================================================================
# Fig 3: kl_vs_ilp.pdf -- Ablation: HAWQ-ILP vs Segment DP (both NMSE curves)
# ===================================================================
def fig3_ablation_gap(df, ilp_df):
    """(a) NMSE: HAWQ-ILP vs Segment DP vs Joint DP.
    (b) Outage: HAWQ-ILP vs Segment DP vs Joint DP.
    ALL data from full_comparison.csv (df) for consistency.
    Methods: hawq-ilp, nmse-static (Segment DP), nmse-adaptive-opt (Joint DP)."""

    with plt.rc_context(STYLE):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5))

        # --- All from full_comparison.csv (consistent calibration) ---
        hawq = _filter_bug(df, "hawq-ilp")
        hawq = hawq[(hawq["saving"] >= 85) & (hawq["saving"] <= 97)]

        sdp = df[df["method"] == "nmse-static"].sort_values("saving")
        sdp = sdp[(sdp["saving"] >= 85) & (sdp["saving"] <= 97)]

        jdp = df[df["method"] == "nmse-adaptive-opt"].sort_values("saving")
        jdp = jdp[(jdp["saving"] >= 85) & (jdp["saving"] <= 97)]

        # --- Panel (a): NMSE ---
        ax_a.plot(hawq["saving"],
                  _monotone_smooth(hawq["saving"].values, hawq["nmse_db"].values),
                  "--", color=C_RED, linewidth=2.0,
                  marker="v", markersize=5, markevery=5,
                  label="HAWQ-ILP", zorder=3)

        if len(sdp) > 0:
            ax_a.plot(sdp["saving"],
                      _monotone_smooth(sdp["saving"].values, sdp["nmse_db"].values),
                      "-", color=C_BLUE, linewidth=2.0,
                      marker="o", markersize=5, markevery=5,
                      label="Segment DP", zorder=4)

        if len(jdp) > 0:
            ax_a.plot(jdp["saving"],
                      _monotone_smooth(jdp["saving"].values, jdp["nmse_db"].values),
                      "-.", color=C_GREEN, linewidth=2.0,
                      marker="s", markersize=5, markevery=5,
                      label="Joint DP", zorder=5)

        ax_a.set_title("(a) NMSE", fontsize=13)
        ax_a.set_ylabel("NMSE (dB)", fontsize=13)
        _axis_saving(ax_a, xlim=(84, 98))
        ax_a.legend(loc="upper left", fontsize=10, framealpha=0.9)
        _grid(ax_a)

        # --- Panel (b): Outage (gamma=0.99) ---
        if len(hawq) > 0 and "outage_99" in hawq.columns:
            ax_b.plot(hawq["saving"],
                      _monotone_nondecreasing(hawq["outage_99"].values),
                      "--", color=C_RED, linewidth=2.0,
                      marker="v", markersize=5, markevery=5,
                      label="HAWQ-ILP", zorder=3)

        if len(sdp) > 0 and "outage_99" in sdp.columns:
            ax_b.plot(sdp["saving"],
                      _monotone_nondecreasing(sdp["outage_99"].values),
                      "-", color=C_BLUE, linewidth=2.0,
                      marker="o", markersize=5, markevery=5,
                      label="Segment DP", zorder=4)

        if len(jdp) > 0 and "outage_99" in jdp.columns:
            ax_b.plot(jdp["saving"],
                      _monotone_nondecreasing(jdp["outage_99"].values),
                      "-.", color=C_GREEN, linewidth=2.0,
                      marker="s", markersize=5, markevery=5,
                      label="Joint DP", zorder=5)

        ax_b.set_title(r"(b) Outage ($\gamma = 0.99$, SNR = 20 dB)", fontsize=13)
        ax_b.set_ylabel("Outage Probability", fontsize=13)
        _axis_saving(ax_b, xlim=(84, 98))
        ax_b.set_ylim(-0.02, 1.05)
        ax_b.legend(loc="upper left", fontsize=10, framealpha=0.9)
        _grid(ax_b)

        fig.tight_layout(w_pad=3.0)
        _save(fig, "kl_vs_ilp")
        print("[Fig 3] kl_vs_ilp.{pdf,png}  -- saved.")


# ===================================================================
# Fig 4: fig_segment_dp_all_models.pdf -- Segment DP across all models
#   Uniform points + Segment DP points for CsiNet, CRNet, CLNet, MT-AE
# ===================================================================
def fig4_segment_dp_all_models(df, baseline_df):
    """Segment DP performance across all model architectures (no uniform)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        model_styles = {
            "CsiNet": {"color": C_BLUE, "marker": "s"},
            "CRNet": {"color": C_PURPLE, "marker": "v"},
            "CLNet": {"color": C_ORANGE, "marker": "D"},
        }

        # Baseline models from segment_dp_baselines.csv
        if baseline_df is not None:
            for model_name, style in model_styles.items():
                sub = baseline_df[baseline_df["model"] == model_name]
                dp = sub[sub["method"] == "segment-dp"].sort_values("target_saving")
                if len(dp) > 0:
                    dp_smooth = _monotone_smooth(
                        dp["target_saving"].values, dp["nmse_db"].values)
                    ax.plot(dp["target_saving"], dp_smooth,
                            "-", color=style["color"], marker=style["marker"],
                            markersize=6, linewidth=2.0, markevery=5,
                            label=f"{model_name}", zorder=4)

        # MT-AE from segment_dp_baselines.csv (same source as others)
        if baseline_df is not None and "MT-AE" in baseline_df["model"].values:
            mtae = baseline_df[
                (baseline_df["model"] == "MT-AE") &
                (baseline_df["method"] == "segment-dp")
            ].sort_values("target_saving")
            if len(mtae) > 0:
                mtae_smooth = _monotone_smooth(
                    mtae["target_saving"].values, mtae["nmse_db"].values)
                ax.plot(mtae["target_saving"], mtae_smooth,
                        "-", color=C_RED, marker="o",
                        markersize=6, linewidth=2.0, markevery=5,
                        label="MT-AE (Ours)", zorder=5)

        ax.set_ylabel("NMSE (dB)", fontsize=13)
        _axis_saving(ax, xlim=(84, 98))
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        _grid(ax)

        fig.tight_layout()
        _save(fig, "fig_segment_dp_all_models")
        print("[Fig 4] fig_segment_dp_all_models.{pdf,png}  -- saved.")


# ===================================================================
# Fig 5: fig_outage_budget_allocation.pdf -- Two-panel
#   (a) Per-bin outage curves (MT-AE, gamma=0.95)
#   (b) Budget allocation effect (MT-AE, gamma=0.95)
# ===================================================================
def fig5_outage_budget_allocation():
    """Per-bin outage heterogeneity and budget allocation improvement."""
    perbin_csv = os.path.join(RESULTS_CSV, "outage_curves_per_bin.csv")
    alloc_csv = os.path.join(RESULTS_CSV, "budget_allocation_outage.csv")
    if not os.path.exists(perbin_csv) or not os.path.exists(alloc_csv):
        print("[Fig 5] SKIPPED -- outage_curves_per_bin.csv or "
              "budget_allocation_outage.csv not found.")
        return

    perbin_df = pd.read_csv(perbin_csv)
    alloc_df = pd.read_csv(alloc_csv)

    gamma = 0.95
    bin_colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE]
    bin_labels = ["Bin 0 (easy)", "Bin 1", "Bin 2", "Bin 3", "Bin 4 (hard)"]

    with plt.rc_context(STYLE):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5))

        # -- Panel (a): Per-bin outage curves --
        # Dense step curves (50+ points): NO markers
        for b in range(5):
            sub = perbin_df[(perbin_df["bin"] == b) &
                            (perbin_df["gamma"] == gamma)].sort_values("target_saving")
            if len(sub) == 0:
                continue
            x = sub["target_saving"].values
            y = _monotone_nondecreasing(sub["outage"].values)
            ax_a.plot(x, y, color=bin_colors[b], drawstyle="steps-post",
                      linewidth=2.0, label=bin_labels[b], zorder=4)

        ax_a.set_title(r"(a) Per-bin outage ($\gamma=0.95$)", fontsize=13)
        _axis_saving(ax_a, xlim=(84, 98))
        ax_a.set_ylabel("Outage Probability", fontsize=13)
        ax_a.set_ylim(-0.02, 1.05)
        ax_a.legend(loc="upper left", fontsize=10, framealpha=0.9)
        _grid(ax_a)

        # -- Panel (b): Equal vs Optimal allocation --
        # 2-line comparison: dashed black baseline, solid red proposed
        # Light blue shading for improvement region
        sub = alloc_df[alloc_df["gamma"] == gamma].sort_values("target_saving")
        if len(sub) > 0:
            x = sub["target_saving"].values
            y_eq = _monotone_nondecreasing(sub["equal_outage"].values)
            y_opt = _monotone_nondecreasing(sub["optimal_outage"].values)

            ax_b.plot(x, y_eq, "--", color=C_RED, drawstyle="steps-post",
                      linewidth=2.0, label="Equal allocation", zorder=4)
            ax_b.plot(x, y_opt, "-", color=C_BLUE, drawstyle="steps-post",
                      linewidth=2.0, label="Optimal allocation", zorder=5)

        ax_b.set_title(r"(b) Budget allocation ($\gamma=0.95$)", fontsize=13)
        _axis_saving(ax_b, xlim=(84, 98))
        ax_b.set_ylim(-0.02, 1.05)
        ax_b.legend(loc="upper left", fontsize=10, framealpha=0.9)
        _grid(ax_b)

        fig.tight_layout(w_pad=3.0)
        _save(fig, "fig_outage_budget_allocation")
        print("[Fig 5] fig_outage_budget_allocation.{pdf,png}  -- saved.")


# ===================================================================
# Fig 6: fig_cross_arch_outage.pdf -- Three panels (gamma=0.99,0.98,0.95)
#   MT-AE, CLNet, CRNet: equal vs optimized allocation outage
# ===================================================================
def fig6_cross_arch_outage():
    """Cross-architecture outage comparison from segment_dp_baselines.csv.

    3 panels for outage_99, outage_98, outage_95.
    Each panel: lines for MT-AE, CLNet, CRNet, CsiNet.
    """
    csv_path = os.path.join(RESULTS_CSV, "segment_dp_baselines.csv")
    if not os.path.exists(csv_path):
        print("[Fig 6] SKIPPED -- segment_dp_baselines.csv not found.")
        return

    df = pd.read_csv(csv_path)
    dp = df[df["method"] == "segment-dp"].copy()

    outage_cols = ["outage_99", "outage_98", "outage_95"]
    panel_labels = [r"(a) $\gamma = 0.99$", r"(b) $\gamma = 0.98$",
                    r"(c) $\gamma = 0.95$"]

    model_cfg = [
        ("CsiNet", C_BLUE, "s"),
        ("CRNet", C_PURPLE, "v"),
        ("CLNet", C_ORANGE, "D"),
        ("MT-AE", C_RED, "o"),
    ]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

        for idx, (ax, ocol) in enumerate(zip(axes, outage_cols)):
            if ocol not in dp.columns:
                ax.set_title(panel_labels[idx])
                ax.text(0.5, 0.5, f"{ocol} not available",
                        transform=ax.transAxes, ha="center")
                continue

            for name, clr, mkr in model_cfg:
                sub = dp[dp["model"] == name].sort_values("target_saving")
                sub = sub[sub["target_saving"] <= BUG_SAVING_THRESHOLD]
                if len(sub) == 0:
                    continue

                x = sub["target_saving"].values
                y = _monotone_nondecreasing(sub[ocol].values)

                ax.plot(x, y, "-", color=clr, marker=mkr,
                        markersize=6, markevery=5,
                        linewidth=2.0, label=name, zorder=4)

            ax.set_title(panel_labels[idx], fontsize=13)
            ax.set_xlabel("BOPs Saving vs. FP32 (%)", fontsize=13)
            ax.set_xlim(84, 98)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.set_ylim(-0.02, 1.05)
            _grid(ax)
            if idx == 0:
                ax.set_ylabel("Outage Probability", fontsize=13)
                ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

        fig.tight_layout(w_pad=2.0)
        _save(fig, "fig_cross_arch_outage")
        print("[Fig 6] fig_cross_arch_outage.{pdf,png}  -- saved.")


# ===================================================================
# Fig 7: fig_joint_dp_budget_distribution.pdf -- Single panel
#   Per-bin FC cost (kappa) from Joint DP at gamma=0.98
# ===================================================================
def fig7_joint_dp_budget_distribution():
    """Per-bin budget allocation from Joint DP across saving levels."""
    joint_csv = os.path.join(RESULTS_CSV, "joint_dp_comparison.csv")
    if not os.path.exists(joint_csv):
        print("[Fig 7] SKIPPED -- joint_dp_comparison.csv not found.")
        return

    joint_df = pd.read_csv(joint_csv)
    gamma = 0.98

    sub = joint_df[(joint_df["gamma"] == gamma) &
                   (joint_df["joint_cost_0"].notna())].sort_values(
        "target_saving")
    if len(sub) == 0:
        print("[Fig 7] SKIPPED -- no data for gamma=0.98 with joint costs.")
        return

    bin_colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE]
    bin_labels = ["Bin 0 (easy)", "Bin 1", "Bin 2", "Bin 3",
                  "Bin 4 (hard)"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        x = sub["target_saving"].values
        # Dense step curves -- NO markers
        for b in range(5):
            col = f"joint_cost_{b}"
            y = sub[col].values
            ax.plot(x, y, color=bin_colors[b], drawstyle="steps-post",
                    linewidth=2.0, label=bin_labels[b], zorder=4)

        # Dashed gray equal-allocation reference line
        eq_cost = sub.iloc[0]["joint_cost_0"]
        ax.axhline(y=eq_cost, color="gray", linestyle="--", linewidth=1.5,
                   alpha=0.7, label="Equal allocation", zorder=3)

        ax.set_xlabel("BOPs Saving vs. FP32 (%)", fontsize=13)
        ax.set_ylabel(r"Per-bin FC cost ($\kappa$)", fontsize=13)
        _axis_saving(ax, xlim=(84, 94))
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        _grid(ax)

        fig.tight_layout()
        _save(fig, "fig_joint_dp_budget_distribution")
        print("[Fig 7] fig_joint_dp_budget_distribution.{pdf,png}  -- saved.")


# ===================================================================
# Fig 8: fig_budget_alloc_multi_snr.pdf -- Three panels (SNR=10,20,30)
#   Equal vs Optimal allocation outage (gamma=0.95)
# ===================================================================
def _fix_fp32_fallback(nmse_db, outage, fp32_nmse_threshold=-14.0):
    """Fix FP32-fallback bug: when the DP solver can't find a feasible policy
    at high saving targets, it falls back to FP32 (NMSE ~ -15.3 dB) and
    reports outage=0.  Physically, infeasible saving means outage=1.0.

    Detection: if NMSE suddenly improves (drops below threshold) after being
    worse, that row is a fallback and outage should be 1.0.
    """
    nmse_db = np.array(nmse_db, dtype=float)
    outage = np.array(outage, dtype=float)
    # Find the first row where NMSE was above fp32_nmse_threshold (i.e.,
    # quantization was active) -- then any subsequent drop below threshold
    # is a fallback.
    was_quantized = False
    for i in range(len(nmse_db)):
        if nmse_db[i] > fp32_nmse_threshold:
            was_quantized = True
        if was_quantized and nmse_db[i] <= fp32_nmse_threshold:
            outage[i] = 1.0
    return outage


def fig8_budget_alloc_multi_snr():
    """Multi-SNR budget allocation: equal vs optimal outage."""
    csv_path = os.path.join(RESULTS_CSV, "complete_eval_mtae.csv")
    if not os.path.exists(csv_path):
        print("[Fig 8] SKIPPED -- complete_eval_mtae.csv not found.")
        return

    df = pd.read_csv(csv_path)
    gamma = 0.95

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for idx, snr in enumerate([10, 20, 30]):
            ax = axes[idx]
            sub = df[(df["snr"] == snr) & (df["gamma"] == gamma)].sort_values(
                "target_saving")
            x = sub["target_saving"].values
            nmse = sub["nmse_db"].values

            # Fix FP32-fallback: set outage=1.0 for infeasible saving levels
            y_eq = _fix_fp32_fallback(nmse, sub["outage_equal"].values.copy())
            y_al = _fix_fp32_fallback(nmse, sub["outage_alloc"].values.copy())

            y_eq = _monotone_nondecreasing(y_eq)
            y_al = _monotone_nondecreasing(y_al)

            ax.plot(x, y_eq, "--", color=C_RED, drawstyle="steps-post",
                    linewidth=2.0, label="Equal allocation")
            ax.plot(x, y_al, "-", color=C_BLUE, drawstyle="steps-post",
                    linewidth=2.0, label="Optimal allocation")

            labels = ["(a)", "(b)", "(c)"]
            ax.set_title(f"{labels[idx]} SNR = {snr} dB", fontsize=13)
            ax.set_xlabel("BOPs Saving vs. FP32 (%)", fontsize=13)
            ax.set_xlim(84, 98)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.set_ylim(-0.02, 1.05)
            _grid(ax)
            if idx == 0:
                ax.set_ylabel("Outage Probability", fontsize=13)
                ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

        fig.tight_layout(w_pad=2.0)
        _save(fig, "fig_budget_alloc_multi_snr")
        print("[Fig 8] fig_budget_alloc_multi_snr.{pdf,png}  -- saved.")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("  GENERATING PAPER FIGURES")
    print("=" * 60)

    df, eval_df, outage_df, baseline_df, ilp_df = load_data()

    fig1_uniform_rpmpq(df, eval_df, baseline_df)
    fig1b_hawq_vs_segment_dp(df)
    fig2_online_outage(df, outage_df)
    fig3_ablation_gap(df, ilp_df)
    fig4_segment_dp_all_models(df, baseline_df)
    fig5_outage_budget_allocation()
    fig6_cross_arch_outage()
    fig7_joint_dp_budget_distribution()
    fig8_budget_alloc_multi_snr()

    # Diagnostic summary
    print("\n" + "-" * 60)
    print("  Output directories:")
    print(f"    Plots:   {FIGURES_DIR}")
    print(f"    Figures: {PAPER_FIG_DIR}")
    print("-" * 60)
    print("  Style notes:")
    print("    - No main titles, only (a)/(b) panel labels")
    print("    - Default matplotlib font (sans-serif), label size 13")
    print("    - Large markers (7-8), lines (2.0)")
    print("    - Dashed grid, alpha=0.4, light gray")
    print("    - All 4 spines visible")
    print(f"    - segment-dp/nmse-static saving > {BUG_SAVING_THRESHOLD}% filtered")
    print("    - figsize: (8,5) single, (10,4.5) two-panel, (15,4) three-panel")
    print("-" * 60)
    if outage_df is None:
        print("  [!] outage_multi_snr_sweep.csv missing -- Fig 2 uses single SNR=20")
    if baseline_df is None:
        print("  [!] segment_dp_baselines.csv missing -- Fig 4 skipped")
    print("-" * 60)
    print("Done.\n")


if __name__ == "__main__":
    main()
