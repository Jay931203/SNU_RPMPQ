"""
Joint DP for Segment-Level Mixed-Precision Quantization across ALL zeta bins.

Instead of the 2-step approach (per-bin segment DP + outer water-filling),
this solves a SINGLE DP where the state includes the bin dimension, so budget
sharing happens naturally inside the DP.

State: (j, m, c) where
    j = bin index (0 to K-1)
    m = block position within bin j (0 to M)
    c = remaining TOTAL budget (discretized, 0 to C_steps)

The budget cost for bin j's segment is weighted by p_j (population fraction)
because the total budget constraint is: sum_j p_j * c_j <= c_bar.

The distortion contribution is also weighted by p_j because we minimize
population-weighted outage.

Usage:
    python analysis/joint_dp_outage.py                 # full pipeline (GPU)
    python analysis/joint_dp_outage.py --dp-only       # joint DP only (CPU)
    python analysis/joint_dp_outage.py --plot-only      # plot from cached results

Requires: cached segment_dp_omegas.csv, rpmpq_v2_zeta.csv,
          rpmpq_v2_perfect_rates.csv, rpmpq_v2_step1_nmse_kappa.csv,
          and model checkpoint + data for GPU inference.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rpmpq_v2 import RESULTS_CSV
from analysis.segment_dp_baselines import (
    enumerate_segments,
    solve_dp,
    segmentation_to_policy,
)
from analysis.budget_allocation import load_cached_omegas
from analysis.budget_allocation_outage import (
    _setup_kappa_from_csv,
    _load_zeta_and_bins,
    _load_r_ref,
    _load_model_and_data,
    _run_inference_rates,
    _build_outage_lookup,
    optimize_allocation_greedy,
    optimize_allocation_grid,
    K_BINS,
    L_MAX,
    BIT_OPTIONS,
    ANCHOR_BITS,
    SNR,
    AQ_BITS,
    GAMMAS,
    OUTAGE_CURVES_CSV,
)

os.makedirs(RESULTS_CSV, exist_ok=True)

# Output files
JOINT_DP_CSV = os.path.join(RESULTS_CSV, "joint_dp_comparison.csv")
JOINT_DP_POLICIES_CSV = os.path.join(RESULTS_CSV, "joint_dp_policies.csv")

# Budget sweep: same range as budget_allocation_outage.py
CURVE_SAVINGS = np.arange(85.0, 97.01, 0.25).tolist()


# ===================================================================
# Joint DP solver
# ===================================================================

def solve_joint_dp(
    K: int,
    M: int,
    segments: List[Tuple[int, int]],
    omega_per_bin: Dict[int, Dict[Tuple[int, int, int], float]],
    kappa_seg: Dict[Tuple[int, int, int], float],
    total_budget: float,
    bit_options: List[int],
    anchor_bits: int,
    p_bins: List[float],
    C_steps: int = 2000,
) -> Tuple[float, Dict[int, List[Tuple[int, int, int]]], Dict[int, float]]:
    """Joint DP across all bins with shared budget.

    State: (j, m, c) -- bin index, block position, remaining budget.

    All bins share the SAME encoder architecture (same M blocks) but have
    DIFFERENT omega values per bin.  The budget is shared across bins via
    population weighting: bin j's segment cost is scaled by p_j.

    Memory optimisation: process bins in reverse order (j = K-1 down to 0).
    For each bin only keep F_curr[m][c] and F_next[c] = F[j+1][0][c].

    Parameters
    ----------
    K : int
        Number of zeta bins.
    M : int
        Number of FC blocks per bin (same architecture for all bins).
    segments : list of (l, r)
        All valid contiguous segments [l:r).
    omega_per_bin : {j: {(l, r, b): omega}}
        Per-bin distortion dict.  omega is the NMSE-delta.
    kappa_seg : {(l, r, b): cost}
        Segment cost in [0, 1] range (same for all bins -- same architecture).
    total_budget : float
        Total FC budget (NOT saving percentage).
    bit_options : list of int
        e.g. [16, 8, 4, 2].
    anchor_bits : int
        e.g. 16.
    p_bins : list of float
        Population weights [p_0, ..., p_{K-1}], sum = 1.
    C_steps : int
        Budget discretization granularity.

    Returns
    -------
    total_distortion : float
        Population-weighted total distortion (sum of p_j * dist_j).
    per_bin_segmentations : {j: [(l, r, b), ...]}
        Optimal segmentation for each bin.
    per_bin_costs : {j: float}
        Actual cost allocated to each bin (in kappa units, NOT budget-weighted).
    """
    if total_budget <= 0:
        total_budget = 1e-6
    c_step = total_budget / C_steps

    INF = float("inf")

    # Build segment lookup by endpoint (same for all bins)
    segs_ending_at: Dict[int, List[int]] = {}
    for (l, r) in segments:
        segs_ending_at.setdefault(r, []).append(l)

    # F_next[c] = value function from bin j+1 at m=0, remaining budget c
    # Base case: after all K bins are done, distortion = 0 for any remaining budget
    F_next = [0.0] * (C_steps + 1)

    # Back-tracking storage: per-bin
    # back_all[j] = back[m][c] = (l, b) or None
    back_all: Dict[int, List[List[Optional[Tuple[int, int]]]]] = {}
    # Also store F[j][M][c] to recover which c was used at the boundary
    F_boundary: Dict[int, List[float]] = {}

    # Process bins in reverse order: j = K-1 down to 0
    for j in range(K - 1, -1, -1):
        p_j = p_bins[j]
        omega_j = omega_per_bin[j]

        # F_curr[m][c] for current bin j
        F_curr = [[INF] * (C_steps + 1) for _ in range(M + 1)]
        back = [[None] * (C_steps + 1) for _ in range(M + 1)]

        # Base case for this bin: F_curr[0][c] = F_next[c]
        # (starting bin j at block 0 with budget c = continuing from bin j+1)
        for c in range(C_steps + 1):
            F_curr[0][c] = F_next[c]

        # Fill DP for blocks 1..M within bin j
        for m in range(1, M + 1):
            if m not in segs_ending_at:
                continue
            for l in segs_ending_at[m]:
                for b in bit_options:
                    seg_kappa = kappa_seg.get((l, m, b), 0)
                    # Budget cost for this bin's segment, weighted by p_j
                    weighted_cost = p_j * seg_kappa
                    seg_cost_idx = math.ceil(weighted_cost / c_step) if c_step > 0 else 0

                    # Distortion contribution, weighted by p_j
                    seg_omega = omega_j.get((l, m, b), 0)
                    weighted_dist = p_j * seg_omega

                    if seg_cost_idx > C_steps:
                        continue

                    for c in range(seg_cost_idx, C_steps + 1):
                        prev_c = c - seg_cost_idx
                        if prev_c < 0:
                            continue
                        candidate = F_curr[l][prev_c] + weighted_dist
                        if candidate < F_curr[m][c]:
                            F_curr[m][c] = candidate
                            back[m][c] = (l, b)

        # Store back-tracking and boundary values for this bin
        back_all[j] = back
        F_boundary[j] = list(F_curr[M])

        # F_next for the previous bin (j-1) is F_curr[M][c]
        # i.e., the cost of optimally handling bins [j..K-1] starting from
        # block M of bin j (= block 0 of bin j+1) with remaining budget c
        F_next = list(F_curr[M])

    # After processing all bins (j = K-1 down to 0):
    # F_boundary[0][c] = F[0][M][c] = pop-weighted total distortion for
    # bins [0..K-1] when c budget-steps remain (i.e., C_steps - c were used).
    # The optimum is at c = C_steps (full budget available) due to monotonicity,
    # but we search all c for robustness against discretization edge cases.

    best_c = 0
    best_val = INF
    for c in range(C_steps + 1):
        if F_boundary[0][c] < best_val:
            best_val = F_boundary[0][c]
            best_c = c

    if best_val >= INF:
        # Infeasible
        return INF, {j: [] for j in range(K)}, {j: 0.0 for j in range(K)}

    # Backtrack to recover per-bin segmentations
    per_bin_segmentations: Dict[int, List[Tuple[int, int, int]]] = {}
    per_bin_costs: Dict[int, float] = {}

    c = best_c
    for j in range(K):
        p_j = p_bins[j]
        back = back_all[j]
        segmentation = []
        m = M
        while m > 0 and back[m][c] is not None:
            l, b = back[m][c]
            segmentation.append((l, m, b))
            seg_kappa = kappa_seg.get((l, m, b), 0)
            weighted_cost = p_j * seg_kappa
            seg_cost_idx = (
                math.ceil(weighted_cost / c_step) if c_step > 0 else 0
            )
            c = c - seg_cost_idx
            m = l

        # If we did not reach m=0, fill remaining with anchor
        if m > 0:
            segmentation.append((0, m, anchor_bits))

        segmentation.reverse()
        per_bin_segmentations[j] = segmentation

        # Compute actual (unweighted) cost for this bin
        bin_cost = sum(
            kappa_seg.get((l, r, b), 0) for (l, r, b) in segmentation
        )
        per_bin_costs[j] = bin_cost

    return best_val, per_bin_segmentations, per_bin_costs


# ===================================================================
# Evaluation pipeline
# ===================================================================

def _compute_equal_policy(
    M: int,
    segments: List[Tuple[int, int]],
    omega_per_bin: Dict[int, Dict],
    kappa_seg: Dict,
    fc_budget: float,
    bit_options: List[int],
    anchor_bits: int,
    p_bins: List[float],
) -> Tuple[List[Tuple[int, int, int]], float]:
    """Equal allocation: use median bin's omega, same policy for all bins.

    Returns (segmentation, dp_distortion).
    """
    # Find the median bin (bin with median population weight, or just bin K//2)
    median_j = len(p_bins) // 2
    omega_median = omega_per_bin[median_j]

    dist, seg = solve_dp(
        M, segments, omega_median, kappa_seg,
        fc_budget, bit_options, anchor_bits,
    )
    return seg, dist


def run_joint_dp_evaluation(
    target_savings_list: Optional[List[float]] = None,
    gammas: Optional[List[float]] = None,
    objective: str = "nmse",
) -> pd.DataFrame:
    """Full pipeline: for each target saving, run joint DP, apply policies,
    measure outage.

    Compares:
    1. Equal allocation (single median-bin DP policy applied to all)
    2. 2-step: segment DP + greedy outer allocation (existing)
    3. Joint DP (new)

    Saves results to joint_dp_comparison.csv.
    """
    import torch
    import torch.nn as nn
    from train_ae import apply_precision_policy

    if target_savings_list is None:
        target_savings_list = CURVE_SAVINGS
    if gammas is None:
        gammas = GAMMAS

    print("=" * 70)
    print("  JOINT DP vs 2-STEP COMPARISON")
    print("=" * 70)

    # Load infrastructure (CPU only for kappa/omega)
    fc_blocks, non_fc_blocks, M, kappa_seg, non_fc_cost, segments = (
        _setup_kappa_from_csv()
    )

    # Load zeta bins and population weights
    zeta_vals, k_indices, zeta_edges = _load_zeta_and_bins()
    r_ref = _load_r_ref()

    N = len(zeta_vals)
    bin_counts = [int(np.sum(k_indices == j)) for j in range(K_BINS)]
    p_bins = [bc / N for bc in bin_counts]
    print(f"  Test samples: {N}")
    for j in range(K_BINS):
        print(f"    Bin {j}: {bin_counts[j]} samples (p={p_bins[j]:.3f})")

    # Load cached omegas
    print("\n  Loading cached segment omegas...")
    omega_nmse, omega_cos2 = load_cached_omegas(
        K_BINS, segments, BIT_OPTIONS, ANCHOR_BITS,
    )
    omega_map = {"nmse": omega_nmse, "cos2": omega_cos2}
    omega_per_bin = omega_map.get(objective, omega_nmse)

    # Load model and data (GPU)
    print("\n  Loading model and data...")
    net, test_set, test_loader, norm_params, device = _load_model_and_data()

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {
        k: v.clone().cpu() for k, v in real_model.state_dict().items()
    }

    # Load existing 2-step outage curves if available
    df_2step_alloc = None
    if os.path.exists(OUTAGE_CURVES_CSV):
        print(f"  Loading existing outage curves from {OUTAGE_CURVES_CSV}")
        df_curves = pd.read_csv(OUTAGE_CURVES_CSV)
    else:
        df_curves = None
        print("  WARNING: No existing outage curves found. 2-step results"
              " will be recomputed.")

    # Results accumulator
    all_results = []
    policy_rows = []

    total_combos = len(target_savings_list) * len(gammas)
    print(f"\n  Savings levels: {len(target_savings_list)}")
    print(f"  Gammas: {gammas}")
    print(f"  DP objective: {objective}")
    print()

    # Pre-compute per-bin sample indices
    indices_per_bin = {
        j: np.where(k_indices == j)[0] for j in range(K_BINS)
    }

    pbar = tqdm(
        total=len(target_savings_list), desc="Joint DP sweep",
    )

    for target_saving in target_savings_list:
        total_budget = 1.0 - target_saving / 100.0
        fc_budget = total_budget - non_fc_cost
        if fc_budget < 0:
            fc_budget = 0.001

        # ---------------------------------------------------------------
        # Method 1: Equal allocation (median-bin DP, same policy for all)
        # ---------------------------------------------------------------
        seg_equal, _ = _compute_equal_policy(
            M, segments, omega_per_bin, kappa_seg,
            fc_budget, BIT_OPTIONS, ANCHOR_BITS, p_bins,
        )
        pol_equal = segmentation_to_policy(
            seg_equal, fc_blocks, non_fc_blocks, ANCHOR_BITS,
        )

        # Run inference for equal allocation (all samples, single policy)
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, pol_equal, device)
        rates_equal_all = _run_inference_rates(
            net, test_set, list(range(N)), norm_params, device,
        )

        # ---------------------------------------------------------------
        # Method 2: 2-step (per-bin DP + outer allocation)
        # Solve per-bin DP independently, each with fc_budget
        # Then measure outage per bin
        # ---------------------------------------------------------------
        rates_2step_all = np.zeros(N)
        twostep_feasible = True
        for j in range(K_BINS):
            dist_j, seg_j = solve_dp(
                M, segments, omega_per_bin[j], kappa_seg,
                fc_budget, BIT_OPTIONS, ANCHOR_BITS,
            )
            if dist_j == float("inf"):
                twostep_feasible = False
                break

            pol_j = segmentation_to_policy(
                seg_j, fc_blocks, non_fc_blocks, ANCHOR_BITS,
            )
            real_model.load_state_dict(original_state)
            apply_precision_policy(net, pol_j, device)

            idx_j = indices_per_bin[j]
            if len(idx_j) > 0:
                rates_j = _run_inference_rates(
                    net, test_set, idx_j.tolist(), norm_params, device,
                )
                rates_2step_all[idx_j] = rates_j

        # ---------------------------------------------------------------
        # Method 3: Joint DP (new)
        # ---------------------------------------------------------------
        total_dist, joint_segs, joint_costs = solve_joint_dp(
            K=K_BINS,
            M=M,
            segments=segments,
            omega_per_bin=omega_per_bin,
            kappa_seg=kappa_seg,
            total_budget=fc_budget,
            bit_options=BIT_OPTIONS,
            anchor_bits=ANCHOR_BITS,
            p_bins=p_bins,
        )
        joint_feasible = total_dist < float("inf")

        rates_joint_all = np.zeros(N)
        if joint_feasible:
            for j in range(K_BINS):
                seg_j = joint_segs[j]
                pol_j = segmentation_to_policy(
                    seg_j, fc_blocks, non_fc_blocks, ANCHOR_BITS,
                )
                real_model.load_state_dict(original_state)
                apply_precision_policy(net, pol_j, device)

                idx_j = indices_per_bin[j]
                if len(idx_j) > 0:
                    rates_j = _run_inference_rates(
                        net, test_set, idx_j.tolist(), norm_params, device,
                    )
                    rates_joint_all[idx_j] = rates_j

            # Save per-bin policies for this saving level
            for j in range(K_BINS):
                seg_str = " ".join(
                    f"[{l}:{r}]b{b}" for (l, r, b) in joint_segs[j]
                )
                policy_rows.append({
                    "target_saving": target_saving,
                    "bin": j,
                    "segmentation": seg_str,
                    "cost": joint_costs[j],
                    "weighted_cost": p_bins[j] * joint_costs[j],
                })

        # ---------------------------------------------------------------
        # Compute outage for each gamma
        # ---------------------------------------------------------------
        for gamma in gammas:
            # Equal
            outage_equal = float(
                np.mean(rates_equal_all < gamma * r_ref[:N])
            )

            # 2-step (equal budget per bin -- each bin gets fc_budget)
            if twostep_feasible:
                outage_2step = float(
                    np.mean(rates_2step_all < gamma * r_ref[:N])
                )
            else:
                outage_2step = 1.0

            # Joint DP
            if joint_feasible:
                outage_joint = float(
                    np.mean(rates_joint_all < gamma * r_ref[:N])
                )
            else:
                outage_joint = 1.0

            # 2-step with outer allocation (if curves available)
            outage_2step_alloc = np.nan
            if df_curves is not None:
                outage_lookup, sorted_savings, p_bins_lu = (
                    _build_outage_lookup(df_curves, gamma)
                )
                _, opt_outage_g, _ = optimize_allocation_greedy(
                    outage_lookup, sorted_savings, p_bins_lu, target_saving,
                )
                # Also try grid
                if target_saving in sorted_savings:
                    _, opt_outage_grid = optimize_allocation_grid(
                        outage_lookup, sorted_savings, p_bins_lu,
                        target_saving, max_deviation_steps=4,
                    )
                    outage_2step_alloc = min(opt_outage_g, opt_outage_grid)
                else:
                    outage_2step_alloc = opt_outage_g

            result = {
                "target_saving": target_saving,
                "gamma": gamma,
                "outage_equal": outage_equal,
                "outage_2step_equal_budget": outage_2step,
                "outage_2step_alloc": outage_2step_alloc,
                "outage_joint_dp": outage_joint,
                "joint_dp_distortion": total_dist if joint_feasible else np.nan,
                "improvement_vs_equal": outage_equal - outage_joint,
                "improvement_vs_2step": outage_2step - outage_joint,
            }
            # Per-bin costs from joint DP
            if joint_feasible:
                for j in range(K_BINS):
                    result[f"joint_cost_{j}"] = joint_costs[j]
                    result[f"joint_wcost_{j}"] = p_bins[j] * joint_costs[j]

            all_results.append(result)

        pbar.update(1)

        # Incremental save every 5 savings levels
        if pbar.n % 5 == 0 and all_results:
            pd.DataFrame(all_results).to_csv(JOINT_DP_CSV, index=False)

    pbar.close()

    # Restore model
    real_model.load_state_dict(original_state)

    # Save final results
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(JOINT_DP_CSV, index=False)
    print(f"\n  Saved comparison results -> {JOINT_DP_CSV}")

    if policy_rows:
        df_pol = pd.DataFrame(policy_rows)
        df_pol.to_csv(JOINT_DP_POLICIES_CSV, index=False)
        print(f"  Saved policies -> {JOINT_DP_POLICIES_CSV}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for gamma in gammas:
        sub = df_out[df_out["gamma"] == gamma]
        print(f"\n  gamma = {gamma}")
        print(f"  {'Saving':>8s} | {'Equal':>8s} | {'2Step-Eq':>8s} | "
              f"{'2Step-Alloc':>10s} | {'Joint DP':>8s} | "
              f"{'vs Equal':>8s} | {'vs 2Step':>8s}")
        print("  " + "-" * 78)
        for _, row in sub.iterrows():
            s = row["target_saving"]
            oe = row["outage_equal"]
            o2e = row["outage_2step_equal_budget"]
            o2a = row["outage_2step_alloc"]
            oj = row["outage_joint_dp"]
            dve = row["improvement_vs_equal"]
            dv2 = row["improvement_vs_2step"]
            o2a_str = f"{o2a:.4f}" if not np.isnan(o2a) else "   N/A"
            print(f"  {s:8.1f} | {oe:8.4f} | {o2e:8.4f} | "
                  f"{o2a_str:>10s} | {oj:8.4f} | "
                  f"{dve:+8.4f} | {dv2:+8.4f}")

    return df_out


# ===================================================================
# Plotting
# ===================================================================

def plot_comparison(csv_path: str = JOINT_DP_CSV):
    """Plot comparison: Equal vs 2-Step vs Joint DP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found. Run evaluation first.")
        return

    df = pd.read_csv(csv_path)

    gammas_to_plot = [g for g in [0.99, 0.98, 0.95] if g in df["gamma"].unique()]
    n_panels = len(gammas_to_plot)
    if n_panels == 0:
        print("  No gamma values found in data.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax_idx, gamma in enumerate(gammas_to_plot):
        ax = axes[ax_idx]
        sub = df[df["gamma"] == gamma].sort_values("target_saving")

        # Equal allocation
        ax.plot(
            sub["target_saving"], sub["outage_equal"],
            "k--", linewidth=2, label="Equal (median-bin DP)",
        )

        # 2-step equal budget
        ax.plot(
            sub["target_saving"], sub["outage_2step_equal_budget"],
            color="#1f77b4", linestyle="-.", linewidth=1.5,
            label="2-Step (equal budget)",
        )

        # 2-step with outer allocation (if available)
        has_alloc = sub["outage_2step_alloc"].notna().any()
        if has_alloc:
            sub_alloc = sub[sub["outage_2step_alloc"].notna()]
            ax.plot(
                sub_alloc["target_saving"], sub_alloc["outage_2step_alloc"],
                color="#2ca02c", linestyle="-", linewidth=2,
                label="2-Step + outer alloc",
            )

        # Joint DP
        ax.plot(
            sub["target_saving"], sub["outage_joint_dp"],
            color="#d62728", linestyle="-", linewidth=2.5,
            label="Joint DP (proposed)",
        )

        # Shade improvement over equal
        ax.fill_between(
            sub["target_saving"],
            sub["outage_joint_dp"],
            sub["outage_equal"],
            where=sub["outage_equal"] > sub["outage_joint_dp"],
            alpha=0.15, color="green",
        )

        ax.set_xlabel("Average BOPs Saving (%)", fontsize=11)
        ax.set_ylabel("Population-Weighted Outage", fontsize=11)
        ax.set_title(f"$\\gamma$ = {gamma}", fontsize=13)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.02, 1.02])

    fig.suptitle(
        "Joint DP vs 2-Step Budget Allocation",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    plot_dir = os.path.join(PROJECT_ROOT, "results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, "joint_dp_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot -> {out_path}")

    # Also save PDF
    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved plot -> {pdf_path}")

    plt.close(fig)

    # Budget allocation breakdown plot
    _plot_budget_allocation(df, gammas_to_plot, plot_dir)


def _plot_budget_allocation(
    df: pd.DataFrame,
    gammas: List[float],
    plot_dir: str,
):
    """Plot per-bin budget allocation from Joint DP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Check if per-bin cost columns exist
    if "joint_cost_0" not in df.columns:
        return

    n_panels = len(gammas)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, K_BINS))

    for ax_idx, gamma in enumerate(gammas):
        ax = axes[ax_idx]
        sub = df[df["gamma"] == gamma].sort_values("target_saving")

        for j in range(K_BINS):
            col = f"joint_cost_{j}"
            if col in sub.columns:
                vals = sub[col].values
                label = f"Bin {j} ({'easy' if j == 0 else 'hard' if j == K_BINS - 1 else ''})"
                ax.plot(
                    sub["target_saving"], vals,
                    "o-", color=colors[j], markersize=2, label=label,
                )

        ax.set_xlabel("Average BOPs Saving (%)", fontsize=11)
        ax.set_ylabel("Per-Bin FC Budget (kappa)", fontsize=11)
        ax.set_title(f"$\\gamma$ = {gamma}", fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Joint DP: Per-Bin Budget Allocation",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    out_path = os.path.join(plot_dir, "joint_dp_budget_allocation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot -> {out_path}")
    plt.close(fig)


# ===================================================================
# Entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Joint DP for segment-level mixed-precision quantization."
    )
    parser.add_argument(
        "--dp-only", action="store_true",
        help="Run Joint DP solver only (CPU, no inference).",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Plot from cached results only.",
    )
    parser.add_argument(
        "--objective", type=str, default="nmse", choices=["nmse", "cos2"],
        help="DP objective for policy selection (default: nmse).",
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.plot_only:
        plot_comparison()
    elif args.dp_only:
        # CPU-only: solve joint DP and print policies, no outage eval
        _run_dp_only(args.objective)
    else:
        # Full pipeline: joint DP + inference + outage
        df = run_joint_dp_evaluation(objective=args.objective)
        plot_comparison()

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done.")


def _run_dp_only(objective: str = "nmse"):
    """CPU-only: solve joint DP for each saving level, print policies."""
    print("=" * 70)
    print("  JOINT DP SOLVER (CPU only, no inference)")
    print("=" * 70)

    fc_blocks, non_fc_blocks, M, kappa_seg, non_fc_cost, segments = (
        _setup_kappa_from_csv()
    )

    zeta_vals, k_indices, zeta_edges = _load_zeta_and_bins()
    N = len(zeta_vals)
    p_bins = [float(np.sum(k_indices == j)) / N for j in range(K_BINS)]

    print(f"  M={M} FC blocks, K={K_BINS} bins, {len(segments)} segments")
    print(f"  Population weights: {[f'{p:.3f}' for p in p_bins]}")

    omega_nmse, omega_cos2 = load_cached_omegas(
        K_BINS, segments, BIT_OPTIONS, ANCHOR_BITS,
    )
    omega_map = {"nmse": omega_nmse, "cos2": omega_cos2}
    omega_per_bin = omega_map.get(objective, omega_nmse)

    print(f"\n  DP objective: {objective}")
    print(f"  Non-FC cost: {non_fc_cost:.6f}")

    policy_rows = []

    for target_saving in [87.5, 90.0, 92.5, 95.0]:
        total_budget = 1.0 - target_saving / 100.0
        fc_budget = total_budget - non_fc_cost
        if fc_budget < 0:
            fc_budget = 0.001

        print(f"\n  --- Target saving: {target_saving}% (FC budget: {fc_budget:.6f}) ---")

        # Equal: per-bin DP with equal budget
        print("    Equal allocation (per-bin DP):")
        total_equal_dist = 0
        for j in range(K_BINS):
            dist_j, seg_j = solve_dp(
                M, segments, omega_per_bin[j], kappa_seg,
                fc_budget, BIT_OPTIONS, ANCHOR_BITS,
            )
            total_equal_dist += p_bins[j] * dist_j
            bits_used = set(b for (_, _, b) in seg_j)
            print(f"      Bin {j}: dist={dist_j:.6f}  bits={bits_used}")
        print(f"      Total weighted dist: {total_equal_dist:.6f}")

        # Joint DP
        print("    Joint DP:")
        total_dist, joint_segs, joint_costs = solve_joint_dp(
            K=K_BINS, M=M, segments=segments,
            omega_per_bin=omega_per_bin, kappa_seg=kappa_seg,
            total_budget=fc_budget, bit_options=BIT_OPTIONS,
            anchor_bits=ANCHOR_BITS, p_bins=p_bins,
        )

        if total_dist < float("inf"):
            total_weighted_cost = sum(
                p_bins[j] * joint_costs[j] for j in range(K_BINS)
            )
            print(f"      Total weighted dist: {total_dist:.6f}")
            print(f"      Total weighted cost: {total_weighted_cost:.6f} "
                  f"(budget: {fc_budget:.6f})")
            print(f"      Improvement vs equal: "
                  f"{total_equal_dist - total_dist:+.6f}")

            for j in range(K_BINS):
                seg_j = joint_segs[j]
                seg_str = " ".join(
                    f"[{l}:{r}]b{b}" for (l, r, b) in seg_j
                )
                bits_used = set(b for (_, _, b) in seg_j)
                print(f"      Bin {j}: cost={joint_costs[j]:.6f}  "
                      f"bits={bits_used}  segs={seg_str}")
                policy_rows.append({
                    "target_saving": target_saving,
                    "bin": j,
                    "segmentation": seg_str,
                    "cost": joint_costs[j],
                    "weighted_cost": p_bins[j] * joint_costs[j],
                })
        else:
            print("      INFEASIBLE")

    if policy_rows:
        df_pol = pd.DataFrame(policy_rows)
        df_pol.to_csv(JOINT_DP_POLICIES_CSV, index=False)
        print(f"\n  Saved policies -> {JOINT_DP_POLICIES_CSV}")


if __name__ == "__main__":
    main()
