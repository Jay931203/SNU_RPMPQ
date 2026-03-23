"""
LUT-based Per-Sample Adaptive Evaluation.

Each sample gets its OWN policy based on its zeta bin.
Uses cached segment_dp_omegas.csv (no re-collection needed).

Usage: !python analysis/eval_lut_adaptive.py
"""
import os, sys, re
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, Subset

from train_ae import (
    apply_precision_policy, quantize_feedback_torch,
    calculate_su_miso_rate_mrt, CsiDataset,
)
from ModularModels import ModularAE
from rpmpq_v2 import get_encoder_block_names, get_encoder_layer_params, RESULTS_CSV
from analysis.segment_dp_baselines import (
    enumerate_segments, solve_dp, segmentation_to_policy,
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
                          snr_list=[10, 20, 30], aq_bits=8):
    """Run inference on a subset of samples."""
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)

    nmse_all = []
    rates_all = {s: [] for s in snr_list}
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
            for snr in snr_list:
                r = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
                rates_all[snr].extend(r.cpu().numpy().tolist())
    return np.array(nmse_all), {s: np.array(v) for s, v in rates_all.items()}


import torch.nn as nn


def main():
    print("=" * 70)
    print("  LUT-BASED PER-SAMPLE ADAPTIVE EVALUATION")
    print("  Each sample → zeta bin → bin-specific policy")
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
    N = len(test_set)

    zeta_vals = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_zeta.csv")
                             )["zeta_proxy"].values

    zeta_edges = np.quantile(zeta_vals, np.linspace(0, 1, K_bins + 1))
    zeta_edges[0] -= 1e-6
    zeta_edges[-1] += 1e-6
    k_indices = np.clip(np.digitize(zeta_vals, zeta_edges) - 1, 0, K_bins - 1)

    # Load cached omegas
    cache_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas.csv")
    assert os.path.exists(cache_csv), f"Run segment_dp_policy.py first: {cache_csv}"
    df_c = pd.read_csv(cache_csv)
    omega_nmse, omega_cos2 = {}, {}
    for _, row in df_c.iterrows():
        key = (int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))
        omega_nmse[key] = row["omega_nmse"]
        omega_cos2[key] = row["omega_cos2"]
    for (l, r) in enumerate_segments(M, L_max):
        for j in range(K_bins):
            omega_nmse[(l, r, anchor_bits, j)] = 0.0
            omega_cos2[(l, r, anchor_bits, j)] = 0.0

    # Kappa
    layer_params = get_encoder_layer_params(net, fc_chunks=32)
    total_fp32 = sum(layer_params.get(bn, 0) * 32 * 32 for bn in block_names)
    kappa_seg = {}
    segments = enumerate_segments(M, L_max)
    for (l, r) in segments:
        for b in bit_options:
            bops = sum(layer_params.get(fc_blocks[i], 0) * b * 16
                       for i in range(l, r))
            kappa_seg[(l, r, b)] = bops / total_fp32 if total_fp32 > 0 else 0

    non_fc_cost = sum((layer_params.get(bn, 0) * anchor_bits * 16) / total_fp32
                      for bn in non_fc_blocks)

    budget_savings = [87.5, 90.0, 92.5, 95.0]

    print(f"\nSamples: {N}, Zeta bins: {K_bins}")
    for j in range(K_bins):
        n_in_bin = np.sum(k_indices == j)
        print(f"  Bin {j}: {n_in_bin} samples ({100*n_in_bin/N:.1f}%)")

    # ---- Build LUT: per (zeta_bin, budget, objective) → policy ----
    print("\n[1] Building LUT...")
    LUT = {}  # (objective, j, saving) → policy_dict

    for obj_name, omega in [("nmse", omega_nmse), ("cos2", omega_cos2)]:
        for j in range(K_bins):
            omega_j = {(l, r, b): omega.get((l, r, b, j), 0)
                       for (l, r) in segments for b in bit_options}
            for saving in budget_savings:
                fc_budget = (1.0 - saving / 100.0) - non_fc_cost
                if fc_budget < 0:
                    fc_budget = 0.001
                _, seg = solve_dp(M, segments, omega_j, kappa_seg,
                                   fc_budget, bit_options, anchor_bits)
                pol = segmentation_to_policy(seg, fc_blocks, non_fc_blocks,
                                             anchor_bits)
                LUT[(obj_name, j, saving)] = pol

    print(f"  LUT size: {len(LUT)} entries")

    # ---- Evaluate: per-sample adaptive ----
    print("\n[2] Per-sample adaptive evaluation...")

    for saving in budget_savings:
        print(f"\n{'='*60}")
        print(f"  Target Saving = {saving}%")
        print(f"{'='*60}")

        for obj_name in ["nmse", "cos2"]:
            # Group samples by zeta bin → same policy
            bin_groups = {}
            for j in range(K_bins):
                indices = np.where(k_indices == j)[0].tolist()
                if len(indices) > 0:
                    bin_groups[j] = indices

            # Per-bin inference, then aggregate
            nmse_all = np.zeros(N)
            rate_all = np.zeros(N)

            for j, indices in tqdm(bin_groups.items(),
                                    desc=f"{obj_name} s={saving}%",
                                    leave=False):
                policy = LUT[(obj_name, j, saving)]

                real_model.load_state_dict(original_state)
                apply_precision_policy(net, policy, device)

                nmse_sub, rates_sub = run_inference_subset(
                    net, test_set, indices, norm_params, device,
                    snr_list=[snr])

                for li, gi in enumerate(indices):
                    nmse_all[gi] = nmse_sub[li]
                    rate_all[gi] = rates_sub[snr][li]

            cos2_all = np.clip(
                (2**rate_all - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

            nmse_db = 10 * np.log10(np.mean(nmse_all) + 1e-15)
            outage_99 = np.mean(rate_all < 0.99 * r_ref[:N])
            outage_95 = np.mean(rate_all < 0.95 * r_ref[:N])

            print(f"  {obj_name:5s}-LUT: NMSE={nmse_db:.2f}dB  "
                  f"cos2={np.mean(cos2_all):.6f}  "
                  f"rate={np.mean(rate_all):.4f}  "
                  f"out99={outage_99:.4f}  out95={outage_95:.4f}")

            # Per-bin breakdown
            for j in range(K_bins):
                mask = k_indices == j
                if mask.sum() == 0:
                    continue
                bin_nmse = 10 * np.log10(np.mean(nmse_all[mask]) + 1e-15)
                bin_cos2 = np.mean(cos2_all[mask])
                print(f"    bin {j}: NMSE={bin_nmse:.2f}dB  cos2={bin_cos2:.6f}  "
                      f"(n={mask.sum()})")

        # Also run static (single policy for all samples, no adaptation)
        print(f"\n  --- Static baseline (no zeta adaptation) ---")
        for obj_name, omega in [("nmse", omega_nmse), ("cos2", omega_cos2)]:
            # Use median bin (bin 2) policy for all samples
            policy = LUT[(obj_name, K_bins // 2, saving)]
            real_model.load_state_dict(original_state)
            apply_precision_policy(net, policy, device)

            nmse_static, rates_static = run_inference_subset(
                net, test_set, list(range(N)), norm_params, device,
                snr_list=[snr])
            rate_s = rates_static[snr]
            cos2_s = np.clip((2**rate_s - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

            nmse_db_s = 10 * np.log10(np.mean(nmse_static) + 1e-15)
            print(f"  {obj_name:5s}-static: NMSE={nmse_db_s:.2f}dB  "
                  f"cos2={np.mean(cos2_s):.6f}  "
                  f"rate={np.mean(rate_s):.4f}")

    real_model.load_state_dict(original_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
