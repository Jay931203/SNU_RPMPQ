"""
Segment DP with ADAPTIVE anchor (static policy as anchor).

Step 1: Run segment DP with INT16 anchor → get static policy at ~90% saving
Step 2: Use that static policy as anchor → re-collect segment omegas
Step 3: Re-run DP + budget allocation + full eval

This should fix the adaptive allocation by measuring omega at the actual
operating point instead of INT16.

Usage: !python analysis/segment_dp_adaptive.py
"""
import os, sys, re
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
from analysis.segment_dp_policy import (
    enumerate_segments, solve_dp, segmentation_to_policy, run_inference,
)

os.makedirs(RESULTS_CSV, exist_ok=True)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    train_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htrainout.mat"), "HT")
    test_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htestout.mat"), "HT",
        normalization_params=train_set.normalization_params)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
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


def main():
    print("=" * 70)
    print("  SEGMENT DP WITH ADAPTIVE ANCHOR")
    print("  Step 1: INT16 anchor → static policy")
    print("  Step 2: Static policy as anchor → re-collect omegas → adaptive")
    print("=" * 70)

    net, test_set, test_loader, norm_params, device = load_model()
    block_names = get_encoder_block_names(net, fc_chunks=32)
    fc_blocks = sorted([b for b in block_names if "fc_part" in b],
                       key=lambda x: int(re.search(r'(\d+)$', x).group()))
    non_fc_blocks = [b for b in block_names if "fc_part" not in b]
    M = len(fc_blocks)

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    snr = 20
    L_max = 6
    K_bins = 5

    r_ref = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
                         )[f"r_perf_{snr}"].values
    zeta_vals = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_zeta.csv")
                             )["zeta_proxy"].values
    N = len(zeta_vals)

    zeta_edges = np.quantile(zeta_vals, np.linspace(0, 1, K_bins + 1))
    zeta_edges[0] -= 1e-6
    zeta_edges[-1] += 1e-6
    k_indices = np.clip(np.digitize(zeta_vals, zeta_edges) - 1, 0, K_bins - 1)

    segments = enumerate_segments(M, L_max)

    layer_params = get_encoder_layer_params(net, fc_chunks=32)
    total_fp32 = sum(layer_params.get(bn, 0) * 32 * 32 for bn in block_names)
    kappa_seg = {}
    for (l, r) in segments:
        for b in bit_options:
            bops = sum(layer_params.get(fc_blocks[i], 0) * b * 16
                       for i in range(l, r))
            kappa_seg[(l, r, b)] = bops / total_fp32 if total_fp32 > 0 else 0
    non_fc_cost = sum((layer_params.get(bn, 0) * anchor_bits * 16) / total_fp32
                      for bn in non_fc_blocks)

    # ==================================================================
    # STEP 1: Get static policy from cached INT16-anchor omegas
    # ==================================================================
    print("\n[Step 1] Loading INT16-anchor omegas → static policy...")
    cache_s1 = os.path.join(RESULTS_CSV, "segment_dp_omegas.csv")
    assert os.path.exists(cache_s1), f"Run segment_dp_policy.py first: {cache_s1}"

    df_s1 = pd.read_csv(cache_s1)
    omega_s1 = {}
    for _, row in df_s1.iterrows():
        omega_s1[(int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))] = row["omega_nmse"]
    for (l, r) in segments:
        for j in range(K_bins):
            omega_s1[(l, r, anchor_bits, j)] = 0.0

    # Solve DP at ~90% for median bin → static policy
    target_fc = (1.0 - 90.0 / 100.0) - non_fc_cost
    mid_bin = K_bins // 2
    omega_mid = {(l, r, b): omega_s1.get((l, r, b, mid_bin), 0)
                 for (l, r) in segments for b in bit_options}
    _, seg_static = solve_dp(M, segments, omega_mid, kappa_seg,
                              target_fc, bit_options, anchor_bits)
    anchor_policy = segmentation_to_policy(seg_static, fc_blocks, non_fc_blocks, anchor_bits)

    print(f"  Static policy at ~90%:")
    print(f"  Segments: ", end="")
    for (l, r, b) in seg_static:
        print(f"[{l}:{r}]INT{b} ", end="")
    print()

    # ==================================================================
    # STEP 2: Re-collect segment omegas with static policy as anchor
    # ==================================================================
    cache_s2 = os.path.join(RESULTS_CSV, "segment_dp_omegas_adaptive.csv")

    if os.path.exists(cache_s2):
        print(f"\n[Step 2] Loading cached adaptive omegas from {cache_s2}")
        df_s2 = pd.read_csv(cache_s2)
        omega_s2_nmse, omega_s2_cos2, omega_s2_sf99, omega_s2_sf95 = {}, {}, {}, {}
        for _, row in df_s2.iterrows():
            key = (int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))
            omega_s2_nmse[key] = row["omega_nmse"]
            omega_s2_cos2[key] = row["omega_cos2"]
            omega_s2_sf99[key] = row.get("omega_sf99", 0.0)
            omega_s2_sf95[key] = row.get("omega_sf95", 0.0)
    else:
        print(f"\n[Step 2] Collecting segment omegas with static-policy anchor...")

        # Run anchor policy
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, anchor_policy, device)
        nmse_anc, rates_anc = run_inference(net, test_loader, norm_params, device)
        r_anc = rates_anc[snr]
        cos2_anc = np.clip((2**r_anc - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)
        print(f"  Anchor NMSE: {10*np.log10(np.mean(nmse_anc)+1e-15):.2f} dB")

        omega_s2_nmse, omega_s2_cos2, omega_s2_sf99, omega_s2_sf95 = {}, {}, {}, {}
        cache_rows = []

        # Shortfall for anchor
        gamma_99, gamma_95 = 0.99, 0.95
        V_anc_99 = np.maximum(0, gamma_99 * r_ref[:N] - r_anc) / (gamma_99 * r_ref[:N] + 1e-6)
        V_anc_95 = np.maximum(0, gamma_95 * r_ref[:N] - r_anc) / (gamma_95 * r_ref[:N] + 1e-6)

        # For each segment, perturb from anchor_policy
        total_runs = len(segments) * len(bit_options)
        print(f"  {len(segments)} segments × {len(bit_options)} bits = {total_runs} runs")

        for (l, r) in tqdm(segments, desc="Adaptive segments"):
            seg_blocks = fc_blocks[l:r]
            for b in bit_options:
                # Check if this is the anchor bit for ALL blocks in segment
                all_anchor = all(anchor_policy.get(fc_blocks[i], anchor_bits) == b
                                 for i in range(l, r))
                if all_anchor:
                    for j in range(K_bins):
                        omega_s2_nmse[(l, r, b, j)] = 0.0
                        omega_s2_cos2[(l, r, b, j)] = 0.0
                        omega_s2_sf99[(l, r, b, j)] = 0.0
                        omega_s2_sf95[(l, r, b, j)] = 0.0
                    continue

                # Build perturbation: segment at bit b, rest at anchor_policy
                pert_policy = dict(anchor_policy)
                for chunk in seg_blocks:
                    pert_policy[chunk] = b

                real_model.load_state_dict(original_state)
                apply_precision_policy(net, pert_policy, device)
                nmse_p, rates_p = run_inference(net, test_loader, norm_params, device)
                r_p = rates_p[snr]
                cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

                # Shortfall for perturbation
                V_pert_99 = np.maximum(0, gamma_99 * r_ref[:N] - r_p) / (gamma_99 * r_ref[:N] + 1e-6)
                V_pert_95 = np.maximum(0, gamma_95 * r_ref[:N] - r_p) / (gamma_95 * r_ref[:N] + 1e-6)

                for j in range(K_bins):
                    mask = k_indices == j
                    if mask.sum() == 0:
                        omega_s2_nmse[(l, r, b, j)] = 0.0
                        omega_s2_cos2[(l, r, b, j)] = 0.0
                        omega_s2_sf99[(l, r, b, j)] = 0.0
                        omega_s2_sf95[(l, r, b, j)] = 0.0
                    else:
                        omega_s2_nmse[(l, r, b, j)] = float(
                            np.mean(nmse_p[mask] - nmse_anc[mask]))
                        omega_s2_cos2[(l, r, b, j)] = float(
                            np.mean(cos2_anc[mask] - cos2_p[mask]))
                        omega_s2_sf99[(l, r, b, j)] = float(
                            np.mean(V_pert_99[mask] - V_anc_99[mask]))
                        omega_s2_sf95[(l, r, b, j)] = float(
                            np.mean(V_pert_95[mask] - V_anc_95[mask]))
                    cache_rows.append({
                        "l": l, "r": r, "b": b, "j": j,
                        "omega_nmse": omega_s2_nmse[(l, r, b, j)],
                        "omega_cos2": omega_s2_cos2[(l, r, b, j)],
                        "omega_sf99": omega_s2_sf99[(l, r, b, j)],
                        "omega_sf95": omega_s2_sf95[(l, r, b, j)],
                    })

        real_model.load_state_dict(original_state)
        pd.DataFrame(cache_rows).to_csv(cache_s2, index=False)
        print(f"  Cached → {cache_s2}")

    # Fill anchor bits
    for (l, r) in segments:
        for j in range(K_bins):
            if (l, r, anchor_bits, j) not in omega_s2_nmse:
                omega_s2_nmse[(l, r, anchor_bits, j)] = 0.0
                omega_s2_cos2[(l, r, anchor_bits, j)] = 0.0
                omega_s2_sf99[(l, r, anchor_bits, j)] = 0.0
                omega_s2_sf95[(l, r, anchor_bits, j)] = 0.0

    # ==================================================================
    # STEP 3: DP + eval with adaptive omegas
    # ==================================================================
    print(f"\n[Step 3] DP + eval with adaptive-anchor omegas...")

    budget_savings = [87.5, 90.0, 92.5, 95.0]

    for target_saving in budget_savings:
        target_fc = (1.0 - target_saving / 100.0) - non_fc_cost
        if target_fc < 0:
            target_fc = 0.001

        print(f"\n{'='*60}")
        print(f"  Target Saving = {target_saving}%")
        print(f"{'='*60}")

        for obj_name, omega_dict in [("nmse", omega_s2_nmse), ("cos2", omega_s2_cos2),
                                     ("sf99", omega_s2_sf99), ("sf95", omega_s2_sf95)]:
            # Per-bin DP
            policies = {}
            for j in range(K_bins):
                omega_j = {(l, r, b): omega_dict.get((l, r, b, j), 0)
                           for (l, r) in segments for b in bit_options}
                _, seg_j = solve_dp(M, segments, omega_j, kappa_seg,
                                     target_fc, bit_options, anchor_bits)
                policies[j] = segmentation_to_policy(
                    seg_j, fc_blocks, non_fc_blocks, anchor_bits)

            # Per-sample adaptive eval
            nmse_all = np.zeros(N)
            rate_all = np.zeros(N)

            for j in range(K_bins):
                indices = np.where(k_indices == j)[0].tolist()
                if not indices:
                    continue
                real_model.load_state_dict(original_state)
                apply_precision_policy(net, policies[j], device)
                subset = Subset(test_set, indices)
                sub_loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=0)
                nmse_sub, rates_sub = run_inference(net, sub_loader, norm_params, device)
                for li, gi in enumerate(indices):
                    nmse_all[gi] = nmse_sub[li]
                    rate_all[gi] = rates_sub[snr][li]

            cos2_all = np.clip((2**rate_all - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)
            nmse_db = 10 * np.log10(np.mean(nmse_all) + 1e-15)
            outage_99 = np.mean(rate_all < 0.99 * r_ref[:N])

            print(f"  [{obj_name}] Adaptive (re-anchored): "
                  f"NMSE={nmse_db:.2f}dB  cos2={np.mean(cos2_all):.6f}  "
                  f"out99={outage_99:.4f}")

            # Per-bin breakdown
            for j in range(K_bins):
                mask = k_indices == j
                if mask.sum() > 0:
                    print(f"    bin {j}: NMSE={10*np.log10(np.mean(nmse_all[mask])+1e-15):.1f}dB  "
                          f"cos2={np.mean(cos2_all[mask]):.6f}")

        # Also compare with static (INT16-anchor omega)
        omega_mid_s1 = {(l, r, b): omega_s1.get((l, r, b, mid_bin), 0)
                        for (l, r) in segments for b in bit_options}
        _, seg_st = solve_dp(M, segments, omega_mid_s1, kappa_seg,
                              target_fc, bit_options, anchor_bits)
        pol_st = segmentation_to_policy(seg_st, fc_blocks, non_fc_blocks, anchor_bits)
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, pol_st, device)
        nmse_st, rates_st = run_inference(net, test_loader, norm_params, device)
        r_st = rates_st[snr]
        cos2_st = np.clip((2**r_st - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)
        print(f"\n  [ref] Static (INT16-anchor): "
              f"NMSE={10*np.log10(np.mean(nmse_st)+1e-15):.2f}dB  "
              f"cos2={np.mean(cos2_st):.6f}  "
              f"out99={np.mean(r_st < 0.99 * r_ref[:N]):.4f}")

    real_model.load_state_dict(original_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
