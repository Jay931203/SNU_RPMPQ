"""
Outage-Based Outer Budget Allocation Optimization.

Changes the objective from mean-NMSE (budget_allocation.py) to outage probability
minimization.  Outage curves exhibit cliff-like behaviour at different budget
positions per zeta bin, enabling meaningful unequal allocation across bins.

Pipeline
--------
Step 1: For each zeta bin and budget level, run segment DP and measure outage
        probability P(rate < gamma * r_ref) via GPU inference.  The per-bin
        outage-vs-budget curves are cached to a CSV for reuse.

Step 2: Given a total average budget constraint, optimise per-bin budgets to
        minimise population-weighted outage using greedy marginal-exchange
        and (for K=5 bins) exhaustive grid search.

Step 3: Save results to budget_allocation_outage.csv.

Usage:  python analysis/budget_allocation_outage.py          # full pipeline
        python analysis/budget_allocation_outage.py --curves-only  # Step 1 only
        python analysis/budget_allocation_outage.py --opt-only     # Steps 2-3 only

Requires: cached segment_dp_omegas.csv, rpmpq_v2_zeta.csv, rpmpq_v2_perfect_rates.csv,
          rpmpq_v2_kappa.csv, and model checkpoint + data for GPU inference.
"""
import os
import sys
import re
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rpmpq_v2 import get_encoder_block_names, get_encoder_layer_params, RESULTS_CSV
from analysis.segment_dp_baselines import (
    enumerate_segments_joint,
    enumerate_segments,
    solve_dp,
    segmentation_to_policy,
)
from analysis.budget_allocation import load_cached_omegas

os.makedirs(RESULTS_CSV, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K_BINS = 5
L_MAX = 6
BIT_OPTIONS = [16, 8, 4, 2]
ANCHOR_BITS = 16
SNR = 20
AQ_BITS = 8  # activation quantization bits
GAMMAS = [0.99, 0.98, 0.95]  # outage thresholds

# Budget sweep: savings from 85% to 97% in 0.1% steps for smooth curves
CURVE_SAVINGS = np.arange(85.0, 97.01, 0.1).tolist()
# Budget sweep for final allocation optimization
ALLOC_SAVINGS = np.arange(85.0, 97.01, 0.1).tolist()

OUTAGE_CURVES_CSV = os.path.join(RESULTS_CSV, "outage_curves_per_bin.csv")
ALLOC_OUT_CSV = os.path.join(RESULTS_CSV, "budget_allocation_outage.csv")


# ===================================================================
# Step 0: Infrastructure helpers (model, data, kappa, zeta)
# ===================================================================

def _setup_kappa_from_csv() -> Tuple[
    List[str], List[str], int, Dict, float, Dict
]:
    """Load block structure and kappa from cached CSVs (no model needed).

    Returns
    -------
    fc_blocks      : ordered list of fc_part block names
    non_fc_blocks  : ordered list of non-fc block names
    M              : number of FC chunks
    kappa_seg      : dict[(l, r, b)] -> float
    non_fc_cost    : total kappa of non-FC blocks at anchor
    segments       : list of (l, r) tuples
    """
    kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_step1_nmse_kappa.csv")
    if not os.path.exists(kappa_csv):
        kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_kappa.csv")
    kdf = pd.read_csv(kappa_csv)

    all_blocks = sorted(kdf["block"].unique())
    fc_blocks = sorted(
        [b for b in all_blocks if "fc_part" in b],
        key=lambda x: int(re.search(r'(\d+)$', x).group()),
    )
    non_fc_blocks = [b for b in all_blocks if "fc_part" not in b]
    M = len(fc_blocks)

    segments = enumerate_segments(M, L_MAX)

    # Build per-block kappa lookup
    block_kappa: Dict[Tuple[str, int], float] = {}
    for _, row in kdf.iterrows():
        block_kappa[(row["block"], int(row["bits"]))] = row["kappa"]

    # Segment kappa
    kappa_seg: Dict[Tuple[int, int, int], float] = {}
    for (l, r) in segments:
        for b in BIT_OPTIONS:
            kappa_seg[(l, r, b)] = sum(
                block_kappa.get((fc_blocks[i], b), 0) for i in range(l, r)
            )

    non_fc_cost = sum(
        block_kappa.get((bn, ANCHOR_BITS), 0) for bn in non_fc_blocks
    )

    return fc_blocks, non_fc_blocks, M, kappa_seg, non_fc_cost, segments


def _load_zeta_and_bins() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load zeta values and compute bin assignments.

    Returns
    -------
    zeta_vals   : (N,) array
    k_indices   : (N,) int array of bin indices in [0, K_BINS)
    zeta_edges  : (K_BINS+1,) array of bin edges
    """
    zeta_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_zeta.csv")
    zeta_vals = pd.read_csv(zeta_csv)["zeta_proxy"].values
    zeta_edges = np.quantile(zeta_vals, np.linspace(0, 1, K_BINS + 1))
    zeta_edges[0] -= 1e-6
    zeta_edges[-1] += 1e-6
    k_indices = np.clip(np.digitize(zeta_vals, zeta_edges) - 1, 0, K_BINS - 1)
    return zeta_vals, k_indices, zeta_edges


def _load_r_ref() -> np.ndarray:
    """Load perfect-CSI rates at the configured SNR."""
    r_ref = pd.read_csv(
        os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    )[f"r_perf_{SNR}"].values
    return r_ref


def _load_model_and_data():
    """Load model, datasets, loaders -- requires GPU-capable environment.

    Returns
    -------
    net, test_set, test_loader, norm_params, device
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from train_ae import CsiDataset
    from ModularModels import ModularAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")

    train_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htrainout.mat"), "HT"
    )
    test_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htestout.mat"), "HT",
        normalization_params=train_set.normalization_params,
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    norm_params = train_set.normalization_params

    net = ModularAE(
        encoder_type="mamba",
        decoder_type="transnet",
        encoded_dim=512,
        M=32,
        encoder_layers=2,
        decoder_layers=2,
    ).to(device)

    ckpt = os.path.join(
        PROJECT_ROOT, "saved_models",
        "mamba_transnet_L2_dim512_baseline", "best.pth",
    )
    state = torch.load(ckpt, map_location=device)
    net.load_state_dict(state.get("state_dict", state), strict=False)
    net.eval()

    return net, test_set, test_loader, norm_params, device


# ===================================================================
# Step 1: Build per-bin outage-vs-budget curves (GPU)
# ===================================================================

def _run_inference_rates(model, dataset, indices, norm_params, device):
    """Run inference on a subset and return per-sample rates at SNR.

    Returns
    -------
    rates : np.ndarray of shape (len(indices),)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    from train_ae import (
        apply_precision_policy,
        quantize_feedback_torch,
        calculate_su_miso_rate_mrt,
    )

    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)

    rate_all = []
    with torch.no_grad():
        for batch in loader:
            d = batch.to(device)
            z = real_model.encoder(d)
            if AQ_BITS > 0:
                z = quantize_feedback_torch(z, AQ_BITS)
            x_hat = real_model.decoder(z)
            h_true = (d * range_val) + min_val - 0.5
            h_hat = (x_hat * range_val) + min_val - 0.5
            r = calculate_su_miso_rate_mrt(h_true, h_hat, SNR, device)
            rate_all.extend(r.cpu().numpy().tolist())

    return np.array(rate_all)


def build_outage_curves(
    net,
    test_set,
    norm_params,
    device,
    fc_blocks: List[str],
    non_fc_blocks: List[str],
    M: int,
    segments,
    kappa_seg: Dict,
    non_fc_cost: float,
    omega_per_bin: Dict[int, Dict],
    k_indices: np.ndarray,
    r_ref: np.ndarray,
    budget_savings: List[float],
    gammas: List[float],
    cache_csv: str = OUTAGE_CURVES_CSV,
) -> pd.DataFrame:
    """Build outage-vs-budget curves for each zeta bin via GPU inference.

    For each (bin, budget) pair:
      1. Run segment DP with the bin's omega dict to find the policy.
      2. Apply the policy and run inference on that bin's test samples.
      3. Compute outage probability for each gamma threshold.

    Results are saved incrementally to *cache_csv*.

    Parameters
    ----------
    net           : PyTorch model
    test_set      : CsiDataset
    norm_params   : (min_val, range_val)
    device        : 'cuda' or 'cpu'
    fc_blocks     : ordered FC block names
    non_fc_blocks : non-FC block names
    M             : number of FC chunks
    segments      : list of (l, r)
    kappa_seg     : dict[(l,r,b)] -> cost
    non_fc_cost   : total anchor cost for non-FC blocks
    omega_per_bin : {j: {(l,r,b): omega}} -- DP distortion dict per bin
    k_indices     : (N,) bin assignments
    r_ref         : (N,) perfect-CSI rates
    budget_savings: list of target saving percentages
    gammas        : list of outage thresholds
    cache_csv     : path to save the curves

    Returns
    -------
    DataFrame with columns [bin, target_saving, fc_budget, gamma, outage, n_samples]
    """
    import torch
    import torch.nn as nn
    from train_ae import apply_precision_policy

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    N = len(test_set)

    # Always recompute from scratch (no cache reuse)
    if os.path.exists(cache_csv):
        os.remove(cache_csv)
        print(f"  Removed old cache: {cache_csv}")

    rows = []
    total_combos = K_BINS * len(budget_savings)

    print(f"  Total (bin x saving) combinations: {total_combos}")

    pbar = tqdm(total=remaining, desc="Outage curves")

    for j in range(K_BINS):
        indices_j = np.where(k_indices == j)[0]
        r_ref_j = r_ref[indices_j]
        n_j = len(indices_j)

        for target_saving in budget_savings:
            if (j, target_saving) in done_keys:
                continue

            # Convert to FC budget
            total_budget = 1.0 - target_saving / 100.0
            fc_budget = total_budget - non_fc_cost
            if fc_budget < 0:
                fc_budget = 0.001

            # Solve DP for this bin
            dist, seg = solve_dp(
                M, segments, omega_per_bin[j], kappa_seg,
                fc_budget, BIT_OPTIONS, ANCHOR_BITS,
            )

            if dist == float("inf"):
                # DP infeasible -- outage = 1.0 for all gammas
                for gamma in gammas:
                    rows.append({
                        "bin": j,
                        "target_saving": target_saving,
                        "fc_budget": fc_budget,
                        "gamma": gamma,
                        "outage": 1.0,
                        "n_samples": n_j,
                    })
            else:
                # Apply policy and run inference
                pol = segmentation_to_policy(seg, fc_blocks, non_fc_blocks, ANCHOR_BITS)
                real_model.load_state_dict(original_state)
                apply_precision_policy(net, pol, device)

                rates_j = _run_inference_rates(
                    net, test_set, indices_j.tolist(), norm_params, device,
                )

                for gamma in gammas:
                    outage = float(np.mean(rates_j < gamma * r_ref_j))
                    rows.append({
                        "bin": j,
                        "target_saving": target_saving,
                        "fc_budget": fc_budget,
                        "gamma": gamma,
                        "outage": outage,
                        "n_samples": n_j,
                    })

            pbar.update(1)

            # Incremental save every 5 combinations
            if pbar.n % 5 == 0:
                pd.DataFrame(rows).to_csv(cache_csv, index=False)

    pbar.close()

    # Restore model and final save
    real_model.load_state_dict(original_state)
    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"  Saved outage curves -> {cache_csv}")
    return df


# ===================================================================
# Step 2: Optimise per-bin budget allocation
# ===================================================================

def _build_outage_lookup(
    df_curves: pd.DataFrame,
    gamma: float,
) -> Tuple[Dict[int, List[float]], List[float], List[float]]:
    """From the outage curves DataFrame, build per-bin outage arrays.

    Returns
    -------
    outage_lookup : {j: list of outage values, aligned to sorted_savings}
    sorted_savings: sorted list of target_saving values
    p_bins        : population weight per bin (n_samples[j] / sum)
    """
    df_g = df_curves[df_curves["gamma"] == gamma].copy()
    sorted_savings = sorted(df_g["target_saving"].unique())

    # Population weights from n_samples
    bin_counts = {}
    for j in range(K_BINS):
        sub = df_g[df_g["bin"] == j]
        if len(sub) > 0:
            bin_counts[j] = sub["n_samples"].iloc[0]
        else:
            bin_counts[j] = 0
    total_n = sum(bin_counts.values())
    p_bins = [bin_counts[j] / total_n if total_n > 0 else 1.0 / K_BINS
              for j in range(K_BINS)]

    outage_lookup: Dict[int, List[float]] = {}
    for j in range(K_BINS):
        sub = df_g[df_g["bin"] == j].sort_values("target_saving")
        # Align to sorted_savings
        saving_to_outage = dict(zip(sub["target_saving"], sub["outage"]))
        outage_lookup[j] = [
            saving_to_outage.get(s, 1.0) for s in sorted_savings
        ]

    return outage_lookup, sorted_savings, p_bins


def _saving_to_fc_budget(saving: float, non_fc_cost: float) -> float:
    """Convert a target saving percentage to FC-only budget."""
    fc = (1.0 - saving / 100.0) - non_fc_cost
    return max(fc, 0.001)


def optimize_allocation_greedy(
    outage_lookup: Dict[int, List[float]],
    sorted_savings: List[float],
    p_bins: List[float],
    target_saving: float,
) -> Tuple[List[float], float, float]:
    """Greedy marginal-exchange allocation (same structure as budget_allocation.py).

    Operates in index space over sorted_savings (coarser than FC budget).

    Returns
    -------
    alloc_savings : per-bin saving values
    opt_outage    : population-weighted outage under optimal allocation
    equal_outage  : population-weighted outage under equal allocation
    """
    n_levels = len(sorted_savings)
    savings_arr = np.array(sorted_savings)

    # Start with equal allocation (each bin gets target_saving)
    target_idx = int(np.argmin(np.abs(savings_arr - target_saving)))
    alloc = [target_idx] * K_BINS

    # Equal-allocation outage
    equal_outage = sum(
        p_bins[j] * outage_lookup[j][target_idx] for j in range(K_BINS)
    )

    # Iterative marginal exchange
    for _iteration in range(500):
        # Marginal gain of DECREASING saving by 1 step (= more budget)
        # for each bin.  Lower saving = more bits = (potentially) lower outage.
        gains = []
        for j in range(K_BINS):
            idx = alloc[j]
            if idx > 0:
                gain = outage_lookup[j][idx] - outage_lookup[j][idx - 1]
            else:
                gain = 0.0  # already at max budget
            gains.append(gain)

        # Marginal cost of INCREASING saving by 1 step (= less budget)
        costs = []
        for j in range(K_BINS):
            idx = alloc[j]
            if idx + 1 < n_levels:
                cost = outage_lookup[j][idx + 1] - outage_lookup[j][idx]
            else:
                cost = float("inf")  # cannot reduce further
            costs.append(cost)

        # Weight by population fraction for a fair comparison:
        # net improvement = p[receiver]*gain[receiver] - p[donor]*cost[donor]
        # We want to find the (donor, receiver) pair with largest net improvement.
        best_net = 0.0
        best_donor, best_receiver = -1, -1
        for donor in range(K_BINS):
            for receiver in range(K_BINS):
                if donor == receiver:
                    continue
                if alloc[donor] + 1 >= n_levels:
                    continue
                if alloc[receiver] <= 0:
                    continue
                net = p_bins[receiver] * gains[receiver] - p_bins[donor] * costs[donor]
                if net > best_net:
                    best_net = net
                    best_donor = donor
                    best_receiver = receiver

        if best_net <= 1e-12:
            break  # no beneficial swap

        alloc[best_donor] += 1      # less budget for donor
        alloc[best_receiver] -= 1   # more budget for receiver

    opt_outage = sum(
        p_bins[j] * outage_lookup[j][alloc[j]] for j in range(K_BINS)
    )
    alloc_savings = [sorted_savings[alloc[j]] for j in range(K_BINS)]

    return alloc_savings, opt_outage, equal_outage


def optimize_allocation_grid(
    outage_lookup: Dict[int, List[float]],
    sorted_savings: List[float],
    p_bins: List[float],
    target_saving: float,
    max_deviation_steps: int = 4,
) -> Tuple[List[float], float]:
    """Exhaustive grid search over feasible per-bin allocations.

    Limits to allocations within +/- max_deviation_steps of the equal budget
    to keep the search tractable for K=5 bins.

    The budget balance constraint is:
        sum_j p_j * saving_j = target_saving  (approximately)

    Returns
    -------
    alloc_savings : per-bin saving values
    opt_outage    : population-weighted outage
    """
    n_levels = len(sorted_savings)
    savings_arr = np.array(sorted_savings)
    target_idx = int(np.argmin(np.abs(savings_arr - target_saving)))

    lo = max(0, target_idx - max_deviation_steps)
    hi = min(n_levels - 1, target_idx + max_deviation_steps)
    candidates = list(range(lo, hi + 1))

    # Weighted savings constraint: sum_j p_j * savings[idx_j] ~ target_saving
    # Allow tolerance of half a step
    step = savings_arr[1] - savings_arr[0] if n_levels > 1 else 1.0
    tol = step * 0.6

    best_outage = float("inf")
    best_alloc = [target_idx] * K_BINS

    # Recursive enumeration with pruning
    def _search(j: int, partial: list, weighted_sum: float):
        nonlocal best_outage, best_alloc

        if j == K_BINS:
            if abs(weighted_sum - target_saving) <= tol:
                outage = sum(
                    p_bins[jj] * outage_lookup[jj][partial[jj]]
                    for jj in range(K_BINS)
                )
                if outage < best_outage:
                    best_outage = outage
                    best_alloc = list(partial)
            return

        # Remaining weight
        remaining_weight = sum(p_bins[jj] for jj in range(j, K_BINS))

        for idx in candidates:
            new_sum = weighted_sum + p_bins[j] * savings_arr[idx]
            # Pruning: check if remaining bins can bring sum within target
            min_remaining = remaining_weight * savings_arr[lo] if j + 1 < K_BINS else 0
            max_remaining = remaining_weight * savings_arr[hi] if j + 1 < K_BINS else 0
            # Remaining contribution from bins j+1..K-1
            if j + 1 < K_BINS:
                rem_wt = sum(p_bins[jj] for jj in range(j + 1, K_BINS))
                lo_sum = rem_wt * savings_arr[lo]
                hi_sum = rem_wt * savings_arr[hi]
            else:
                lo_sum = hi_sum = 0.0

            if new_sum + hi_sum < target_saving - tol:
                continue  # even max remaining cannot reach target
            if new_sum + lo_sum > target_saving + tol:
                continue  # even min remaining exceeds target

            partial.append(idx)
            _search(j + 1, partial, new_sum)
            partial.pop()

    _search(0, [], 0.0)

    alloc_savings = [sorted_savings[best_alloc[j]] for j in range(K_BINS)]
    return alloc_savings, best_outage


# ===================================================================
# Step 3: Main pipeline
# ===================================================================

def run_step1_curves(objective: str = "nmse"):
    """Step 1: Build outage-vs-budget curves (requires GPU + data)."""
    print("=" * 70)
    print("  STEP 1: Building per-bin outage-vs-budget curves")
    print("=" * 70)

    fc_blocks, non_fc_blocks, M, kappa_seg, non_fc_cost, segments = (
        _setup_kappa_from_csv()
    )

    # Load model and data
    net, test_set, test_loader, norm_params, device = _load_model_and_data()

    # Load zeta bins
    zeta_vals, k_indices, zeta_edges = _load_zeta_and_bins()
    r_ref = _load_r_ref()

    N = len(test_set)
    print(f"  Test samples: {N}")
    for j in range(K_BINS):
        print(f"    Bin {j}: {np.sum(k_indices == j)} samples")

    # Load cached omegas (for DP objective)
    print("\n  Loading cached segment omegas...")
    omega_nmse, omega_cos2 = load_cached_omegas(
        K_BINS, segments, BIT_OPTIONS, ANCHOR_BITS,
    )

    omega_map = {"nmse": omega_nmse, "cos2": omega_cos2}
    omega_per_bin = omega_map.get(objective, omega_nmse)

    print(f"\n  DP objective for policy selection: {objective}")
    print(f"  Budget savings range: {CURVE_SAVINGS[0]}% -- {CURVE_SAVINGS[-1]}%")
    print(f"  Gammas: {GAMMAS}")
    print()

    df_curves = build_outage_curves(
        net=net,
        test_set=test_set,
        norm_params=norm_params,
        device=device,
        fc_blocks=fc_blocks,
        non_fc_blocks=non_fc_blocks,
        M=M,
        segments=segments,
        kappa_seg=kappa_seg,
        non_fc_cost=non_fc_cost,
        omega_per_bin=omega_per_bin,
        k_indices=k_indices,
        r_ref=r_ref,
        budget_savings=CURVE_SAVINGS,
        gammas=GAMMAS,
    )

    # Print summary
    print("\n  Outage curve summary:")
    for gamma in GAMMAS:
        print(f"\n    gamma = {gamma}")
        sub = df_curves[df_curves["gamma"] == gamma]
        for j in range(K_BINS):
            bsub = sub[sub["bin"] == j].sort_values("target_saving")
            if len(bsub) == 0:
                continue
            o_min = bsub["outage"].min()
            o_max = bsub["outage"].max()
            # Find the "cliff": largest single-step increase
            outages = bsub["outage"].values
            diffs = np.diff(outages)
            cliff_idx = np.argmax(diffs) if len(diffs) > 0 else 0
            cliff_saving = bsub["target_saving"].values[cliff_idx]
            print(
                f"      Bin {j}: outage [{o_min:.4f}, {o_max:.4f}]  "
                f"cliff near {cliff_saving:.1f}% saving"
            )

    return df_curves


def run_step2_optimize(df_curves: Optional[pd.DataFrame] = None):
    """Step 2-3: Optimise allocation and save results (CPU only)."""
    print("\n" + "=" * 70)
    print("  STEP 2: Optimising per-bin budget allocation (outage objective)")
    print("=" * 70)

    # Load curves if not provided
    if df_curves is None:
        if not os.path.exists(OUTAGE_CURVES_CSV):
            print(f"  ERROR: Outage curves not found at {OUTAGE_CURVES_CSV}")
            print("  Run with --curves-only first (requires GPU + data).")
            return
        df_curves = pd.read_csv(OUTAGE_CURVES_CSV)
        print(f"  Loaded outage curves from {OUTAGE_CURVES_CSV}")

    fc_blocks, non_fc_blocks, M, kappa_seg, non_fc_cost, segments = (
        _setup_kappa_from_csv()
    )

    all_results = []

    for gamma in GAMMAS:
        print(f"\n  --- gamma = {gamma} ---")

        outage_lookup, sorted_savings, p_bins = _build_outage_lookup(
            df_curves, gamma
        )

        print(f"    Population weights: {[f'{p:.3f}' for p in p_bins]}")
        print(f"    Savings levels: {len(sorted_savings)}")

        for target_saving in ALLOC_SAVINGS:
            # Greedy allocation
            alloc_greedy, opt_outage_g, equal_outage = optimize_allocation_greedy(
                outage_lookup, sorted_savings, p_bins, target_saving,
            )

            # Grid search allocation (if feasible with curves)
            if target_saving in sorted_savings:
                alloc_grid, opt_outage_grid = optimize_allocation_grid(
                    outage_lookup, sorted_savings, p_bins, target_saving,
                    max_deviation_steps=4,
                )
                # Take the better of greedy and grid
                if opt_outage_grid < opt_outage_g:
                    alloc_final = alloc_grid
                    opt_outage = opt_outage_grid
                    method_chosen = "grid"
                else:
                    alloc_final = alloc_greedy
                    opt_outage = opt_outage_g
                    method_chosen = "greedy"
            else:
                alloc_final = alloc_greedy
                opt_outage = opt_outage_g
                method_chosen = "greedy"

            improvement = equal_outage - opt_outage
            improvement_pct = (
                (improvement / equal_outage * 100) if equal_outage > 1e-9 else 0.0
            )

            result = {
                "target_saving": target_saving,
                "gamma": gamma,
                "equal_outage": equal_outage,
                "optimal_outage": opt_outage,
                "improvement": improvement,
                "improvement_pct": improvement_pct,
                "method": method_chosen,
            }
            # Per-bin savings under optimal allocation
            for j in range(K_BINS):
                result[f"B_{j}"] = _saving_to_fc_budget(alloc_final[j], non_fc_cost)
                result[f"saving_{j}"] = alloc_final[j]

            all_results.append(result)

            # Print notable results
            if improvement > 1e-5:
                per_bin_str = "  ".join(
                    f"b{j}={alloc_final[j]:.1f}%"
                    for j in range(K_BINS)
                )
                print(
                    f"    {target_saving:5.1f}%: equal={equal_outage:.4f}  "
                    f"opt={opt_outage:.4f}  "
                    f"D={improvement:.4f} ({improvement_pct:.1f}%)  "
                    f"[{method_chosen}]  {per_bin_str}"
                )

    # Save
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(ALLOC_OUT_CSV, index=False)
    print(f"\n  Saved allocation results -> {ALLOC_OUT_CSV}")

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: Best improvements per gamma")
    print("=" * 70)
    for gamma in GAMMAS:
        sub = df_out[df_out["gamma"] == gamma]
        if len(sub) == 0:
            continue
        best_row = sub.loc[sub["improvement"].idxmax()]
        print(
            f"  gamma={gamma}: best at {best_row['target_saving']:.1f}% saving  "
            f"equal_outage={best_row['equal_outage']:.4f}  "
            f"optimal_outage={best_row['optimal_outage']:.4f}  "
            f"improvement={best_row['improvement']:.4f} "
            f"({best_row['improvement_pct']:.1f}%)"
        )

    return df_out


# ===================================================================
# Entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Outage-based outer budget allocation for RP-MPQ."
    )
    parser.add_argument(
        "--curves-only", action="store_true",
        help="Only build outage curves (Step 1, requires GPU).",
    )
    parser.add_argument(
        "--opt-only", action="store_true",
        help="Only run allocation optimization (Steps 2-3, CPU only).",
    )
    parser.add_argument(
        "--objective", type=str, default="nmse", choices=["nmse", "cos2"],
        help="DP objective for policy selection (default: nmse).",
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.opt_only:
        run_step2_optimize()
    elif args.curves_only:
        run_step1_curves(objective=args.objective)
    else:
        # Full pipeline
        df_curves = run_step1_curves(objective=args.objective)
        run_step2_optimize(df_curves)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
