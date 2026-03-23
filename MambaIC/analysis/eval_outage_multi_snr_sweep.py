"""
Outage curves at SNR=10,20,30 with 0.25% step.
Only HAWQ-ILP and Segment DP Static (paper main curves).

Usage: !python analysis/eval_outage_multi_snr_sweep.py
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
from analysis.segment_dp_policy import enumerate_segments, solve_dp, segmentation_to_policy

os.makedirs(RESULTS_CSV, exist_ok=True)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def run_inference_all_snr(model, loader, norm_params, device, aq_bits=8):
    """One forward pass, compute rate at SNR=10,20,30."""
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params
    snr_list = [10, 20, 30]
    nmse_all = []
    rates = {s: [] for s in snr_list}
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
                rates[snr].extend(r.cpu().numpy().tolist())
    return np.array(nmse_all), {s: np.array(v) for s, v in rates.items()}


def main():
    print("=" * 60)
    print("  OUTAGE SWEEP: SNR=10,20,30 at 0.25% step")
    print("  Methods: HAWQ-ILP + Segment DP only")
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
    K_bins = 5
    segments = enumerate_segments(M, L_max)

    # Perfect rates
    perf_df = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv"))
    N = len(test_set)

    # Segment DP setup
    cache_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas.csv")
    df_cache = pd.read_csv(cache_csv)
    omega_nmse = {}
    for _, row in df_cache.iterrows():
        omega_nmse[(int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))] = row["omega_nmse"]
    for (l, r) in segments:
        for j in range(K_bins):
            omega_nmse[(l, r, anchor_bits, j)] = 0.0

    layer_params = get_encoder_layer_params(net, fc_chunks=32)
    total_fp32 = sum(layer_params.get(bn, 0) * 32 * 32 for bn in block_names)
    kappa_seg = {}
    for (l, r) in segments:
        for b in bit_options:
            bops = sum(layer_params.get(fc_blocks[i], 0) * b * 16 for i in range(l, r))
            kappa_seg[(l, r, b)] = bops / total_fp32 if total_fp32 > 0 else 0
    non_fc_cost = sum((layer_params.get(bn, 0) * anchor_bits * 16) / total_fp32
                      for bn in non_fc_blocks)

    # HAWQ LUT
    hawq_csv = os.path.join(RESULTS_CSV, "mp_policy_lut_mamba_pruned.csv")
    hawq_df = pd.read_csv(hawq_csv)
    if isinstance(hawq_df["Policy"].iloc[0], str):
        hawq_df["Policy"] = hawq_df["Policy"].apply(ast.literal_eval)

    savings = np.arange(85, 97.01, 0.1).tolist()
    mid_bin = K_bins // 2
    results = []

    # Cache policies to avoid redundant DP solves
    dp_policy_cache = {}
    hawq_policy_cache = {}

    print(f"\n{len(savings)} saving levels × 2 methods = {len(savings)*2} evals")

    for si, target_saving in enumerate(tqdm(savings, desc="Sweep")):
        # --- Segment DP policy ---
        target_fc = (1.0 - target_saving / 100.0) - non_fc_cost
        if target_fc < 0:
            target_fc = 0.001
        fc_key = round(target_fc, 6)
        if fc_key not in dp_policy_cache:
            omega_mid = {(l, r, b): omega_nmse.get((l, r, b, mid_bin), 0)
                         for (l, r) in segments for b in bit_options}
            _, seg = solve_dp(M, segments, omega_mid, kappa_seg,
                               target_fc, bit_options, anchor_bits)
            dp_policy_cache[fc_key] = segmentation_to_policy(
                seg, fc_blocks, non_fc_blocks, anchor_bits)
        pol_dp = dp_policy_cache[fc_key]

        real_model.load_state_dict(original_state)
        apply_precision_policy(net, pol_dp, device)
        nmse_dp, rates_dp = run_inference_all_snr(
            net, test_loader, norm_params, device)
        nmse_db_dp = 10 * np.log10(np.mean(nmse_dp) + 1e-15)

        for snr in snr_list:
            r_ref = perf_df[f"r_perf_{snr}"].values[:N]
            r_dp = rates_dp[snr]
            results.append({
                "saving": target_saving, "method": "segment-dp", "snr": snr,
                "nmse_db": nmse_db_dp, "rate": np.mean(r_dp),
                "outage_99": np.mean(r_dp < 0.99 * r_ref),
                "outage_98": np.mean(r_dp < 0.98 * r_ref),
                "outage_95": np.mean(r_dp < 0.95 * r_ref),
            })

        # --- HAWQ-ILP policy ---
        closest = hawq_df.iloc[
            (hawq_df["Actual_Saving"] - target_saving).abs().argsort()[:1]]
        actual_sav = closest["Actual_Saving"].values[0]
        sav_key = round(actual_sav, 4)

        if sav_key not in hawq_policy_cache:
            hawq_policy_cache[sav_key] = closest["Policy"].values[0]

            real_model.load_state_dict(original_state)
            apply_precision_policy(net, hawq_policy_cache[sav_key], device)
            nmse_h, rates_h = run_inference_all_snr(
                net, test_loader, norm_params, device)
            hawq_policy_cache[f"{sav_key}_nmse"] = nmse_h
            hawq_policy_cache[f"{sav_key}_rates"] = rates_h

        nmse_h = hawq_policy_cache[f"{sav_key}_nmse"]
        rates_h = hawq_policy_cache[f"{sav_key}_rates"]
        nmse_db_h = 10 * np.log10(np.mean(nmse_h) + 1e-15)

        for snr in snr_list:
            r_ref = perf_df[f"r_perf_{snr}"].values[:N]
            r_h = rates_h[snr]
            results.append({
                "saving": target_saving, "method": "hawq-ilp", "snr": snr,
                "nmse_db": nmse_db_h, "rate": np.mean(r_h),
                "outage_99": np.mean(r_h < 0.99 * r_ref),
                "outage_98": np.mean(r_h < 0.98 * r_ref),
                "outage_95": np.mean(r_h < 0.95 * r_ref),
            })

    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_CSV, "outage_multi_snr_sweep.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df)} rows)")

    real_model.load_state_dict(original_state)
    print("Done.")


if __name__ == "__main__":
    main()
