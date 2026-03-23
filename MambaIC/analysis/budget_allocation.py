"""
Outer Budget Allocation Optimization.

Given total average budget B_th, find per-zeta-bin budget allocation
that minimizes total distortion via marginal distortion equalization.

Uses cached segment DP omegas — CPU only, runs in seconds.

Usage: python analysis/budget_allocation.py
"""
import os, sys, re
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rpmpq_v2 import RESULTS_CSV
from analysis.segment_dp_baselines import enumerate_segments_joint, solve_dp

os.makedirs(RESULTS_CSV, exist_ok=True)


def load_cached_omegas(K_bins, segments, bit_options, anchor_bits):
    """Load cached segment omegas and build per-bin omega dicts.

    Supports two formats:
      - v1: segment_dp_omegas.csv with per-bin 'j' column + omega_cos2
      - v2: segment_dp_omegas_v2_mt-ae.csv without 'j' (bin-averaged omega_nmse only)
    Falls back to v2 if v1 not found. For v2, all bins share the same omega.
    """
    v1_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas.csv")
    v2_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas_v2_mt-ae.csv")

    # Prefer v2 (40-block joint indexing) over v1 (32-block FC-only, legacy)
    if os.path.exists(v2_csv):
        df = pd.read_csv(v2_csv)
        has_j = False
        has_cos2 = False
        print(f"[load_cached_omegas] Using v2 omegas (40-block): {v2_csv}")
    elif os.path.exists(v1_csv):
        df = pd.read_csv(v1_csv)
        has_j = "j" in df.columns
        has_cos2 = "omega_cos2" in df.columns
        print(f"[load_cached_omegas] WARNING: Using v1 omegas (32-block FC-only): {v1_csv}")
    else:
        raise FileNotFoundError(
            f"No omega cache found. Expected:\n  {v1_csv}\n  or {v2_csv}")

    if has_j:
        # v1 format: per-bin omegas
        omega_nmse_all = {}
        omega_cos2_all = {}
        for _, row in df.iterrows():
            key = (int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))
            omega_nmse_all[key] = row["omega_nmse"]
            if has_cos2:
                omega_cos2_all[key] = row["omega_cos2"]

        for (l, r) in segments:
            for j in range(K_bins):
                omega_nmse_all[(l, r, anchor_bits, j)] = 0.0
                if has_cos2:
                    omega_cos2_all[(l, r, anchor_bits, j)] = 0.0

        omega_per_bin_nmse = {}
        omega_per_bin_cos2 = {}
        for j in range(K_bins):
            omega_per_bin_nmse[j] = {(l, r, b): omega_nmse_all.get((l, r, b, j), 0)
                                      for (l, r) in segments for b in bit_options}
            omega_per_bin_cos2[j] = {(l, r, b): omega_cos2_all.get((l, r, b, j), 0)
                                      for (l, r) in segments for b in bit_options}
    else:
        # v2 format: bin-averaged (same omega for all bins)
        omega_flat = {}
        for _, row in df.iterrows():
            omega_flat[(int(row["l"]), int(row["r"]), int(row["b"]))] = row["omega_nmse"]
        for (l, r) in segments:
            omega_flat[(l, r, anchor_bits)] = 0.0

        shared = {(l, r, b): omega_flat.get((l, r, b), 0)
                  for (l, r) in segments for b in bit_options}
        omega_per_bin_nmse = {j: dict(shared) for j in range(K_bins)}
        omega_per_bin_cos2 = {j: {k: 0.0 for k in shared} for j in range(K_bins)}

    return omega_per_bin_nmse, omega_per_bin_cos2


def compute_distortion_curve(omega_per_bin, kappa_seg, segments, M,
                              bit_options, anchor_bits, budget_range):
    """For each bin, compute distortion at each budget level."""
    # D[j][budget_idx] = optimal distortion
    K_bins = len(omega_per_bin)
    D = {j: [] for j in range(K_bins)}
    policies = {j: [] for j in range(K_bins)}

    for j in range(K_bins):
        for budget in budget_range:
            dist, seg = solve_dp(M, segments, omega_per_bin[j], kappa_seg,
                                  budget, bit_options, anchor_bits)
            D[j].append(dist)
            policies[j].append(seg)

    return D, policies


def optimize_allocation(D_curves, budget_range, K_bins, target_budget,
                         p_bins=None):
    """Find optimal per-bin budget allocation.

    Grid search over allocations where mean(B_j) = target_budget.
    """
    if p_bins is None:
        p_bins = [1.0 / K_bins] * K_bins

    n_budgets = len(budget_range)
    budget_arr = np.array(budget_range)

    best_total_D = float('inf')
    best_alloc = [n_budgets // 2] * K_bins  # default: median budget for all

    # For 5 bins, grid search is feasible if we limit to ~20 budget levels
    # Total combinations: 20^5 = 3.2M — too much
    # Use greedy/iterative approach instead

    # Start with equal allocation
    # Find the budget index closest to target for each bin
    target_idx = np.argmin(np.abs(budget_arr - target_budget))
    alloc = [target_idx] * K_bins

    # Iterative improvement: move budget from bin with lowest marginal gain
    # to bin with highest marginal gain
    for iteration in range(200):
        improved = False

        # Compute marginal gains for each bin
        marginals = []
        for j in range(K_bins):
            idx = alloc[j]
            # Marginal gain of increasing budget by 1 step
            if idx + 1 < n_budgets:
                gain = D_curves[j][idx] - D_curves[j][idx + 1]  # positive = improvement
            else:
                gain = 0
            marginals.append(gain)

        # Marginal cost of decreasing budget by 1 step
        marginal_costs = []
        for j in range(K_bins):
            idx = alloc[j]
            if idx - 1 >= 0:
                cost = D_curves[j][idx - 1] - D_curves[j][idx]  # positive = worsening
            else:
                cost = float('inf')
            marginal_costs.append(cost)

        # Find best swap: decrease lowest-cost bin, increase highest-gain bin
        donor = np.argmin(marginal_costs)
        receiver = np.argmax(marginals)

        if donor == receiver:
            break
        if marginals[receiver] <= marginal_costs[donor]:
            break  # no beneficial swap
        if alloc[donor] <= 0:
            break

        alloc[donor] -= 1
        alloc[receiver] += 1
        improved = True

        if not improved:
            break

    # Compute final metrics
    total_D = sum(p_bins[j] * D_curves[j][alloc[j]] for j in range(K_bins))
    alloc_budgets = [budget_arr[alloc[j]] for j in range(K_bins)]
    avg_budget = sum(p_bins[j] * alloc_budgets[j] for j in range(K_bins))

    return alloc_budgets, total_D, avg_budget


def main():
    print("=" * 70)
    print("  OUTER BUDGET ALLOCATION (CPU only)")
    print("=" * 70)

    # Setup from cached kappa CSV (no model/GPU needed)
    kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_step1_nmse_kappa.csv")
    if not os.path.exists(kappa_csv):
        kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_kappa.csv")
    kdf = pd.read_csv(kappa_csv)
    all_blocks = sorted(kdf["block"].unique())
    fc_blocks = sorted([b for b in all_blocks if "fc_part" in b],
                       key=lambda x: int(re.search(r'(\d+)$', x).group()))
    # Non-FC: only blocks with dim>=2 weights (exclude LayerNorm, BatchNorm)
    # Match segment_dp_baselines.py get_encoder_modules() filter
    EXCLUDE_NONFC = {"out_norm", "stem.1"}  # 1D weight layers
    non_fc_blocks = [b for b in all_blocks if "fc_part" not in b
                     and not any(ex in b for ex in EXCLUDE_NONFC)]
    # Joint 40-block ordering: FC first, then non-FC
    all_block_names = fc_blocks + non_fc_blocks
    M_fc = len(fc_blocks)
    M_nonfc = len(non_fc_blocks)
    M = M_fc + M_nonfc

    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    K_bins = 5
    L_max = 6

    segments = enumerate_segments_joint(M_fc, M_nonfc, L_max)

    # Kappa from CSV: joint 40-block
    block_kappa = {}
    for _, row in kdf.iterrows():
        block_kappa[(row["block"], int(row["bits"]))] = row["kappa"]

    # Renormalize kappa: CSV uses W*A/(W32*A16), baselines uses W*A/(W32*A32)
    # So CSV INT16 total = 0.5, baselines INT16 total = 0.25. Scale = 0.5
    total_int16 = sum(block_kappa.get((bn, anchor_bits), 0) for bn in all_block_names)
    TARGET_INT16_TOTAL = 0.25  # segment_dp_baselines: params*16*16/(params*32*32)
    if total_int16 > 0:
        scale = TARGET_INT16_TOTAL / total_int16
    else:
        scale = 1.0
    print(f"  Kappa renorm: sum(INT16)={total_int16:.4f}, scale={scale:.2f}")

    kappa_seg = {}
    for (l, r) in segments:
        for b in bit_options:
            kappa_seg[(l, r, b)] = scale * sum(
                block_kappa.get((all_block_names[i], b), 0) for i in range(l, r))

    # Load cached omegas
    print("\n[1] Loading cached segment omegas...")
    omega_nmse, omega_cos2 = load_cached_omegas(
        K_bins, segments, bit_options, anchor_bits)

    # Budget range for distortion curves (total budget, not FC-only)
    budget_range = np.linspace(0.005, 0.20, 200).tolist()

    # Compute distortion curves
    print("\n[2] Computing distortion-budget curves for each bin...")
    for obj_name, omega in [("nmse", omega_nmse), ("cos2", omega_cos2)]:
        D_curves, pol_curves = compute_distortion_curve(
            omega, kappa_seg, segments, M, bit_options, anchor_bits, budget_range)

        print(f"\n  --- {obj_name} objective ---")
        print(f"  Budget range: {budget_range[0]:.3f} ~ {budget_range[-1]:.3f}")
        for j in range(K_bins):
            print(f"  Bin {j}: D_min={min(D_curves[j]):.6f}  D_max={max(D_curves[j]):.6f}")

        # Optimal allocation for each target saving
        print(f"\n  Optimal budget allocation:")
        for target_saving in np.arange(85, 97.01, 0.1).tolist():
            target_budget = 1.0 - target_saving / 100.0
            if target_budget < 0.005:
                target_budget = 0.005

            equal_D = sum(D_curves[j][np.argmin(np.abs(
                np.array(budget_range) - target_budget))] for j in range(K_bins)) / K_bins

            opt_budgets, opt_D, avg_budget = optimize_allocation(
                D_curves, budget_range, K_bins, target_budget)

            savings_per_bin = [(1.0 - b) * 100 for b in opt_budgets]

            print(f"\n  Target: {target_saving}% (budget={target_budget:.4f})")
            print(f"    Equal:   total_D={equal_D:.6f}")
            print(f"    Optimal: total_D={opt_D:.6f}  (Δ={opt_D-equal_D:+.6f})")
            print(f"    Allocation: ", end="")
            for j in range(K_bins):
                print(f"bin{j}={opt_budgets[j]:.4f}({savings_per_bin[j]:.1f}%) ", end="")
            print()

    # Save allocation results
    results = []
    for obj_name, omega in [("nmse", omega_nmse), ("cos2", omega_cos2)]:
        D_curves, _ = compute_distortion_curve(
            omega, kappa_seg, segments, M, bit_options, anchor_bits, budget_range)
        for target_saving in np.arange(85, 97.01, 0.1).tolist():
            target_budget = 1.0 - target_saving / 100.0
            if target_budget < 0.005:
                target_budget = 0.005
            opt_budgets, opt_D, avg_budget = optimize_allocation(
                D_curves, budget_range, K_bins, target_budget)
            equal_idx = np.argmin(np.abs(np.array(budget_range) - target_budget))
            equal_D = sum(D_curves[j][equal_idx] for j in range(K_bins)) / K_bins
            results.append({
                "objective": obj_name,
                "target_saving": target_saving,
                "equal_D": equal_D,
                "optimal_D": opt_D,
                "improvement": equal_D - opt_D,
                **{f"B_{j}": opt_budgets[j] for j in range(K_bins)},
            })

    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_CSV, "budget_allocation.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
