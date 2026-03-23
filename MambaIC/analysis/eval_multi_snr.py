"""
Multi-SNR evaluation for paper table.
Evaluates Segment DP and HAWQ-ILP at SNR=10, 20, 30 for key saving levels.
Only evaluates at 4 representative points (not full sweep).

Usage: !python analysis/eval_multi_snr.py
"""
import os, sys, re, ast
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


def run_inference_multi_snr(model, loader, norm_params, device,
                             snr_list=[10, 20, 30], aq_bits=8):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params
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


def main():
    print("=" * 60)
    print("  MULTI-SNR EVALUATION (SNR=10, 20, 30)")
    print("=" * 60)

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
    snr_list = [10, 20, 30]
    L_max = 6
    segments = enumerate_segments(M, L_max)

    # Load perfect rates for all SNRs
    perf_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    perf_df = pd.read_csv(perf_csv)
    N = len(test_set)

    # Load segment omegas (SNR=20 based, but policies are SNR-independent)
    cache_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas.csv")
    df_cache = pd.read_csv(cache_csv)
    omega_nmse = {}
    for _, row in df_cache.iterrows():
        omega_nmse[(int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))] = row["omega_nmse"]
    K_bins = 5
    for (l, r) in segments:
        for j in range(K_bins):
            omega_nmse[(l, r, anchor_bits, j)] = 0.0

    # Kappa
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

    # Load HAWQ LUT
    hawq_csv = os.path.join(RESULTS_CSV, "mp_policy_lut_mamba_pruned.csv")
    hawq_df = None
    if os.path.exists(hawq_csv):
        hawq_df = pd.read_csv(hawq_csv)
        if isinstance(hawq_df["Policy"].iloc[0], str):
            hawq_df["Policy"] = hawq_df["Policy"].apply(ast.literal_eval)

    key_savings = [87.5, 90.0, 92.5, 95.0]
    mid_bin = K_bins // 2

    results = []

    for target_saving in key_savings:
        print(f"\n{'='*50}")
        print(f"  Target Saving = {target_saving}%")
        print(f"{'='*50}")

        # --- Segment DP policy ---
        target_fc = (1.0 - target_saving / 100.0) - non_fc_cost
        if target_fc < 0:
            target_fc = 0.001
        omega_mid = {(l, r, b): omega_nmse.get((l, r, b, mid_bin), 0)
                     for (l, r) in segments for b in bit_options}
        _, seg = solve_dp(M, segments, omega_mid, kappa_seg,
                           target_fc, bit_options, anchor_bits)
        pol_dp = segmentation_to_policy(seg, fc_blocks, non_fc_blocks, anchor_bits)

        # Evaluate Segment DP at all SNRs
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, pol_dp, device)
        nmse_dp, rates_dp = run_inference_multi_snr(
            net, test_loader, norm_params, device, snr_list)

        nmse_db_dp = 10 * np.log10(np.mean(nmse_dp) + 1e-15)

        for snr in snr_list:
            r_ref = perf_df[f"r_perf_{snr}"].values[:N]
            r_dp = rates_dp[snr]
            out99 = np.mean(r_dp < 0.99 * r_ref)
            out95 = np.mean(r_dp < 0.95 * r_ref)
            rate_mean = np.mean(r_dp)

            results.append({
                "saving": target_saving, "method": "segment-dp",
                "snr": snr, "nmse_db": nmse_db_dp,
                "rate": rate_mean, "outage_99": out99, "outage_95": out95,
            })
            print(f"  [Segment DP] SNR={snr}: NMSE={nmse_db_dp:.2f}dB  "
                  f"rate={rate_mean:.4f}  out99={out99:.4f}")

        # --- HAWQ-ILP policy ---
        if hawq_df is not None:
            closest = hawq_df.iloc[
                (hawq_df["Actual_Saving"] - target_saving).abs().argsort()[:1]]
            if abs(closest["Actual_Saving"].values[0] - target_saving) < 1.0:
                pol_hawq = closest["Policy"].values[0]

                real_model.load_state_dict(original_state)
                apply_precision_policy(net, pol_hawq, device)
                nmse_h, rates_h = run_inference_multi_snr(
                    net, test_loader, norm_params, device, snr_list)

                nmse_db_h = 10 * np.log10(np.mean(nmse_h) + 1e-15)

                for snr in snr_list:
                    r_ref = perf_df[f"r_perf_{snr}"].values[:N]
                    r_h = rates_h[snr]
                    out99 = np.mean(r_h < 0.99 * r_ref)
                    out95 = np.mean(r_h < 0.95 * r_ref)
                    rate_mean = np.mean(r_h)

                    results.append({
                        "saving": target_saving, "method": "hawq-ilp",
                        "snr": snr, "nmse_db": nmse_db_h,
                        "rate": rate_mean, "outage_99": out99, "outage_95": out95,
                    })
                    print(f"  [HAWQ-ILP]   SNR={snr}: NMSE={nmse_db_h:.2f}dB  "
                          f"rate={rate_mean:.4f}  out99={out99:.4f}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_CSV, "multi_snr_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    real_model.load_state_dict(original_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
