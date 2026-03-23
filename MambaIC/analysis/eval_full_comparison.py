"""
Full Comparison Evaluation for Paper Figures.

Compares:
1. Static: single policy from median zeta bin
2. Adaptive-equal: per-zeta-bin DP policy, equal budget
3. Adaptive-opt: per-zeta-bin DP policy, optimal budget allocation
4. HAWQ-ILP: old HAWQ + ILP baseline from mp_policy_lut_mamba_pruned.csv
5. Re-anchored adaptive: per-zeta-bin DP with re-anchored omegas
6. Zeta vs Hoyer breakdown: per-bin NMSE for static policy under both binnings

Uses cached segment_dp_omegas.csv + budget_allocation.csv.

Output columns: saving, method, nmse_db, cos2, rate, outage_99, outage_95
Methods: {nmse,cos2}-static, {nmse,cos2}-adaptive-equal, {nmse,cos2}-adaptive-opt,
         hawq-ilp, {nmse,cos2}-reanchor-adaptive,
         {nmse,cos2}-zeta-breakdown, {nmse,cos2}-hoyer-breakdown

Usage: !python analysis/eval_full_comparison.py
"""
import os, sys, re, ast
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from train_ae import (
    apply_precision_policy, quantize_feedback_torch,
    calculate_su_miso_rate_mrt, CsiDataset,
)
from ModularModels import ModularAE
from rpmpq_v2 import get_encoder_block_names, get_encoder_layer_params, RESULTS_CSV
from analysis.segment_dp_baselines import (
    enumerate_segments_joint, solve_dp, segmentation_to_policy,
)
from analysis.budget_allocation import load_cached_omegas

os.makedirs(RESULTS_CSV, exist_ok=True)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    train_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htrainout.mat"), "HT")
    test_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htestout.mat"), "HT",
        normalization_params=train_set.normalization_params)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    norm_params = train_set.normalization_params
    net = ModularAE(
        encoder_type='mamba', decoder_type='transnet',
        encoded_dim=512, M=32, encoder_layers=2, decoder_layers=2,
    ).to(device)
    ckpt = os.path.join(PROJECT_ROOT, "saved_models",
                        "mamba_transnet_L2_dim512_baseline", "best.pth")
    state = torch.load(ckpt, map_location=device)
    net.load_state_dict(state.get("state_dict", state), strict=False)
    net.eval()
    return net, test_set, test_loader, norm_params, device


def run_inference_subset(model, dataset, indices, norm_params, device,
                          snr=20, aq_bits=8):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)
    nmse_all, rate_all = [], []
    with torch.no_grad():
        for batch in loader:
            d = batch.to(device)
            z = real_model.encoder(d)
            if aq_bits > 0:
                z = quantize_feedback_torch(z, aq_bits)
            x_hat = real_model.decoder(z)
            h_true = (d * range_val) + min_val - 0.5
            h_hat = (x_hat * range_val) + min_val - 0.5
            error = torch.sum((h_true - h_hat)**2, dim=[1,2,3])
            power = torch.sum(h_true**2, dim=[1,2,3])
            nmse_all.extend((error / (power + 1e-9)).cpu().numpy().tolist())
            r = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
            rate_all.extend(r.cpu().numpy().tolist())
    return np.array(nmse_all), np.array(rate_all)


def eval_adaptive(net, test_set, norm_params, device, policies_per_bin,
                   k_indices, K_bins, r_ref, snr=20):
    """Evaluate with per-sample bin-specific policies."""
    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    N = len(test_set)
    nmse_all = np.zeros(N)
    rate_all = np.zeros(N)

    for j in range(K_bins):
        indices = np.where(k_indices == j)[0].tolist()
        if len(indices) == 0:
            continue
        policy = policies_per_bin[j]
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, policy, device)
        nmse_sub, rate_sub = run_inference_subset(
            net, test_set, indices, norm_params, device, snr)
        for li, gi in enumerate(indices):
            nmse_all[gi] = nmse_sub[li]
            rate_all[gi] = rate_sub[li]

    real_model.load_state_dict(original_state)
    cos2_all = np.clip((2**rate_all - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

    return {
        "nmse_db": 10 * np.log10(np.mean(nmse_all) + 1e-15),
        "cos2": np.mean(cos2_all),
        "rate": np.mean(rate_all),
        "outage_99": np.mean(rate_all < 0.99 * r_ref[:N]),
        "outage_95": np.mean(rate_all < 0.95 * r_ref[:N]),
        "nmse_per_bin": [10*np.log10(np.mean(nmse_all[k_indices==j])+1e-15)
                         for j in range(K_bins)],
        "cos2_per_bin": [np.mean(cos2_all[k_indices==j]) for j in range(K_bins)],
    }


def main():
    print("=" * 70)
    print("  FULL COMPARISON FOR PAPER FIGURES")
    print("=" * 70)

    net, test_set, test_loader, norm_params, device = load_model()
    block_names = get_encoder_block_names(net, fc_chunks=32)
    fc_blocks = sorted([b for b in block_names if "fc_part" in b],
                       key=lambda x: int(re.search(r'(\d+)$', x).group()))
    # Non-FC: only blocks with dim>=2 weights (match segment_dp_baselines)
    EXCLUDE_NONFC = {"out_norm", "stem.1"}
    non_fc_blocks = [b for b in block_names if "fc_part" not in b
                     and not any(ex in b for ex in EXCLUDE_NONFC)]

    # Joint 40-block ordering: FC first, then non-FC
    all_block_names = fc_blocks + non_fc_blocks
    M_fc = len(fc_blocks)
    M_nonfc = len(non_fc_blocks)
    M = M_fc + M_nonfc

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    K_bins = 5
    L_max = 6
    snr = 20
    segments = enumerate_segments_joint(M_fc, M_nonfc, L_max)

    r_ref = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
                         )[f"r_perf_{snr}"].values
    N = len(test_set)

    zeta_vals = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_zeta.csv")
                             )["zeta_proxy"].values
    zeta_edges = np.quantile(zeta_vals, np.linspace(0, 1, K_bins + 1))
    zeta_edges[0] -= 1e-6
    zeta_edges[-1] += 1e-6
    k_indices = np.clip(np.digitize(zeta_vals, zeta_edges) - 1, 0, K_bins - 1)

    # Kappa: joint 40-block
    layer_params = get_encoder_layer_params(net, fc_chunks=32)
    total_fp32 = sum(layer_params.get(bn, 0) * 32 * 32 for bn in block_names)
    kappa_seg = {}
    for (l, r) in segments:
        for b in bit_options:
            bops = sum(layer_params.get(all_block_names[i], 0) * b * 16
                       for i in range(l, r))
            kappa_seg[(l, r, b)] = bops / total_fp32 if total_fp32 > 0 else 0

    # Load omegas and budget allocation
    omega_nmse, omega_cos2 = load_cached_omegas(
        K_bins, segments, bit_options, anchor_bits)

    alloc_csv = os.path.join(RESULTS_CSV, "budget_allocation.csv")
    if os.path.exists(alloc_csv):
        alloc_df = pd.read_csv(alloc_csv)
    else:
        alloc_df = None
        print("[WARN] No budget allocation data. Run budget_allocation.py first.")

    budget_savings = np.arange(85, 97.01, 0.1).tolist()

    # --- Load HAWQ ILP LUT (Method 4) ---
    hawq_lut_csv = os.path.join(RESULTS_CSV, "mp_policy_lut_mamba_pruned.csv")
    hawq_df = None
    if os.path.exists(hawq_lut_csv):
        hawq_df = pd.read_csv(hawq_lut_csv)
        if isinstance(hawq_df["Policy"].iloc[0], str):
            hawq_df["Policy"] = hawq_df["Policy"].apply(ast.literal_eval)
        print(f"[INFO] Loaded HAWQ LUT with {len(hawq_df)} policies")
    else:
        print(f"[WARN] HAWQ LUT not found: {hawq_lut_csv}. Skipping hawq-ilp method.")

    # --- Load re-anchored omegas (Method 5) ---
    reanchor_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas_adaptive.csv")
    omega_reanchor_nmse, omega_reanchor_cos2 = None, None
    if os.path.exists(reanchor_csv):
        reanchor_df = pd.read_csv(reanchor_csv)
        omega_ra_nmse_all, omega_ra_cos2_all = {}, {}
        for _, row in reanchor_df.iterrows():
            key = (int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))
            omega_ra_nmse_all[key] = row["omega_nmse"]
            omega_ra_cos2_all[key] = row["omega_cos2"]
        for (l, r) in segments:
            for j in range(K_bins):
                omega_ra_nmse_all[(l, r, anchor_bits, j)] = 0.0
                omega_ra_cos2_all[(l, r, anchor_bits, j)] = 0.0
        omega_reanchor_nmse = {}
        omega_reanchor_cos2 = {}
        for j in range(K_bins):
            omega_reanchor_nmse[j] = {(l, r, b): omega_ra_nmse_all.get((l, r, b, j), 0)
                                       for (l, r) in segments for b in bit_options}
            omega_reanchor_cos2[j] = {(l, r, b): omega_ra_cos2_all.get((l, r, b, j), 0)
                                       for (l, r) in segments for b in bit_options}
        print(f"[INFO] Loaded re-anchored omegas ({len(reanchor_df)} entries)")
    else:
        print(f"[WARN] Re-anchored omegas not found: {reanchor_csv}. Skipping reanchor method.")

    # --- Hoyer binning (Method 6) ---
    fp32_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_fp32_ref.csv")
    hoyer_vals = None
    k_hoyer = None
    hoyer_edges = None
    if os.path.exists(fp32_csv):
        fp32_df = pd.read_csv(fp32_csv)
        if "hoyer" in fp32_df.columns:
            hoyer_vals = fp32_df["hoyer"].values
            hoyer_edges = np.quantile(hoyer_vals, np.linspace(0, 1, K_bins + 1))
            hoyer_edges[0] -= 1e-6
            hoyer_edges[-1] += 1e-6
            k_hoyer = np.clip(np.digitize(hoyer_vals, hoyer_edges) - 1, 0, K_bins - 1)
            print(f"[INFO] Hoyer binning ready (range [{hoyer_vals.min():.2f}, {hoyer_vals.max():.2f}])")
        else:
            print(f"[WARN] 'hoyer' column not in {fp32_csv}. Skipping hoyer breakdown.")
    else:
        print(f"[WARN] FP32 ref not found: {fp32_csv}. Skipping hoyer breakdown.")

    all_results = []

    for target_saving in budget_savings:
        print(f"\n{'='*60}")
        print(f"  Target Saving = {target_saving}%")
        print(f"{'='*60}")

        # Total budget (joint FC + non-FC)
        target_budget = 1.0 - target_saving / 100.0
        if target_budget < 0.005:
            target_budget = 0.005

        # --- Method 4: HAWQ + ILP baseline (objective-agnostic) ---
        if hawq_df is not None:
            # Find closest policy within 0.5% tolerance
            diffs = np.abs(hawq_df["Actual_Saving"].values - target_saving)
            best_idx = np.argmin(diffs)
            if diffs[best_idx] <= 0.5:
                hawq_policy = hawq_df["Policy"].iloc[best_idx]
                hawq_actual = hawq_df["Actual_Saving"].iloc[best_idx]

                real_model.load_state_dict(original_state)
                apply_precision_policy(net, hawq_policy, device)
                nmse_hawq, rate_hawq = run_inference_subset(
                    net, test_set, list(range(N)), norm_params, device, snr)
                cos2_hawq = np.clip(
                    (2**rate_hawq - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

                res_hawq = {
                    "nmse_db": 10*np.log10(np.mean(nmse_hawq)+1e-15),
                    "cos2": np.mean(cos2_hawq),
                    "rate": np.mean(rate_hawq),
                    "outage_99": np.mean(rate_hawq < 0.99 * r_ref[:N]),
                    "outage_95": np.mean(rate_hawq < 0.95 * r_ref[:N]),
                }
                print(f"  [hawq-ilp] actual={hawq_actual:.2f}%  "
                      f"NMSE={res_hawq['nmse_db']:.2f}dB  "
                      f"cos2={res_hawq['cos2']:.6f}  "
                      f"out99={res_hawq['outage_99']:.4f}")

                all_results.append({
                    "saving": target_saving,
                    "method": "hawq-ilp",
                    "nmse_db": res_hawq["nmse_db"],
                    "cos2": res_hawq["cos2"],
                    "rate": res_hawq["rate"],
                    "outage_99": res_hawq["outage_99"],
                    "outage_95": res_hawq["outage_95"],
                })
            else:
                print(f"  [hawq-ilp] No policy within 0.5% of {target_saving}% "
                      f"(closest: {hawq_df['Actual_Saving'].iloc[best_idx]:.2f}%)")

        for obj_name, omega in [("nmse", omega_nmse), ("cos2", omega_cos2)]:

            # --- Method 1: Static (single policy, median bin omega) ---
            mid_bin = K_bins // 2
            _, seg_static = solve_dp(M, segments, omega[mid_bin], kappa_seg,
                                      target_budget, bit_options, anchor_bits)
            pol_static = segmentation_to_policy(seg_static, all_block_names)

            real_model.load_state_dict(original_state)
            apply_precision_policy(net, pol_static, device)
            nmse_s, rate_s = run_inference_subset(
                net, test_set, list(range(N)), norm_params, device, snr)
            cos2_s = np.clip((2**rate_s - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

            res_static = {
                "nmse_db": 10*np.log10(np.mean(nmse_s)+1e-15),
                "cos2": np.mean(cos2_s),
                "rate": np.mean(rate_s),
                "outage_99": np.mean(rate_s < 0.99 * r_ref[:N]),
            }

            # --- Method 2: Adaptive equal budget ---
            policies_equal = {}
            for j in range(K_bins):
                _, seg_j = solve_dp(M, segments, omega[j], kappa_seg,
                                     target_budget, bit_options, anchor_bits)
                policies_equal[j] = segmentation_to_policy(seg_j, all_block_names)

            res_equal = eval_adaptive(net, test_set, norm_params, device,
                                       policies_equal, k_indices, K_bins, r_ref, snr)

            # --- Method 3: Adaptive optimal budget ---
            if alloc_df is not None:
                row = alloc_df[(alloc_df["objective"] == obj_name) &
                               (alloc_df["target_saving"] == target_saving)]
                if len(row) > 0:
                    opt_budgets = [row[f"B_{j}"].values[0] for j in range(K_bins)]
                else:
                    opt_budgets = [target_budget] * K_bins
            else:
                opt_budgets = [target_budget] * K_bins

            policies_opt = {}
            for j in range(K_bins):
                _, seg_j = solve_dp(M, segments, omega[j], kappa_seg,
                                     opt_budgets[j], bit_options, anchor_bits)
                policies_opt[j] = segmentation_to_policy(seg_j, all_block_names)

            res_opt = eval_adaptive(net, test_set, norm_params, device,
                                     policies_opt, k_indices, K_bins, r_ref, snr)

            # Print
            print(f"\n  [{obj_name}] Static:        NMSE={res_static['nmse_db']:.2f}dB  "
                  f"cos2={res_static['cos2']:.6f}  out99={res_static['outage_99']:.4f}")
            print(f"  [{obj_name}] Adaptive-equal: NMSE={res_equal['nmse_db']:.2f}dB  "
                  f"cos2={res_equal['cos2']:.6f}  out99={res_equal['outage_99']:.4f}")
            print(f"  [{obj_name}] Adaptive-opt:   NMSE={res_opt['nmse_db']:.2f}dB  "
                  f"cos2={res_opt['cos2']:.6f}  out99={res_opt['outage_99']:.4f}")

            # Per-bin for optimal
            print(f"    Per-bin (opt): ", end="")
            for j in range(K_bins):
                print(f"b{j}={res_opt['nmse_per_bin'][j]:.1f}dB ", end="")
            print()

            for method, res, label in [
                ("static", res_static, f"{obj_name}-static"),
                ("equal", res_equal, f"{obj_name}-adaptive-equal"),
                ("optimal", res_opt, f"{obj_name}-adaptive-opt"),
            ]:
                all_results.append({
                    "saving": target_saving,
                    "method": label,
                    "nmse_db": res["nmse_db"],
                    "cos2": res["cos2"],
                    "rate": res.get("rate", 0),
                    "outage_99": res.get("outage_99", 0),
                    "outage_95": res.get("outage_95", 0),
                })

            # --- Method 5: Re-anchored adaptive (per-zeta-bin, equal budget) ---
            if obj_name == "nmse" and omega_reanchor_nmse is not None:
                omega_ra = omega_reanchor_nmse
            elif obj_name == "cos2" and omega_reanchor_cos2 is not None:
                omega_ra = omega_reanchor_cos2
            else:
                omega_ra = None

            if omega_ra is not None:
                policies_ra = {}
                for j in range(K_bins):
                    _, seg_j = solve_dp(M, segments, omega_ra[j], kappa_seg,
                                         target_budget, bit_options, anchor_bits)
                    policies_ra[j] = segmentation_to_policy(seg_j, all_block_names)

                res_ra = eval_adaptive(net, test_set, norm_params, device,
                                        policies_ra, k_indices, K_bins, r_ref, snr)

                print(f"  [{obj_name}] Reanchor-adpt:  NMSE={res_ra['nmse_db']:.2f}dB  "
                      f"cos2={res_ra['cos2']:.6f}  out99={res_ra['outage_99']:.4f}")

                all_results.append({
                    "saving": target_saving,
                    "method": f"{obj_name}-reanchor-adaptive",
                    "nmse_db": res_ra["nmse_db"],
                    "cos2": res_ra["cos2"],
                    "rate": res_ra.get("rate", 0),
                    "outage_99": res_ra.get("outage_99", 0),
                    "outage_95": res_ra.get("outage_95", 0),
                })

            # --- Method 6: Zeta vs Hoyer per-bin breakdown for static policy ---
            # Zeta breakdown
            for j in range(K_bins):
                idx_j = np.where(k_indices == j)[0]
                if len(idx_j) == 0:
                    continue
                nmse_j = nmse_s[idx_j]
                rate_j = rate_s[idx_j]
                cos2_j = np.clip((2**rate_j - 1) / (2**r_ref[idx_j] - 1 + 1e-12), 0, 1)
                all_results.append({
                    "saving": target_saving,
                    "method": f"{obj_name}-zeta-breakdown",
                    "bin": j,
                    "nmse_db": 10*np.log10(np.mean(nmse_j)+1e-15),
                    "cos2": np.mean(cos2_j),
                    "rate": np.mean(rate_j),
                    "outage_99": np.mean(rate_j < 0.99 * r_ref[idx_j]),
                    "outage_95": np.mean(rate_j < 0.95 * r_ref[idx_j]),
                })

            # Hoyer breakdown (same static policy, different bin assignment)
            if k_hoyer is not None:
                for j in range(K_bins):
                    idx_j = np.where(k_hoyer == j)[0]
                    if len(idx_j) == 0:
                        continue
                    nmse_j = nmse_s[idx_j]
                    rate_j = rate_s[idx_j]
                    cos2_j = np.clip((2**rate_j - 1) / (2**r_ref[idx_j] - 1 + 1e-12), 0, 1)
                    all_results.append({
                        "saving": target_saving,
                        "method": f"{obj_name}-hoyer-breakdown",
                        "bin": j,
                        "nmse_db": 10*np.log10(np.mean(nmse_j)+1e-15),
                        "cos2": np.mean(cos2_j),
                        "rate": np.mean(rate_j),
                        "outage_99": np.mean(rate_j < 0.99 * r_ref[idx_j]),
                        "outage_95": np.mean(rate_j < 0.95 * r_ref[idx_j]),
                    })

    # Save
    df = pd.DataFrame(all_results)
    out_csv = os.path.join(RESULTS_CSV, "full_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    real_model.load_state_dict(original_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
