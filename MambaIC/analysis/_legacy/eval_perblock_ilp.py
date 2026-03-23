"""
Per-block ILP evaluation (L_max=1) for 3-way comparison:
  HAWQ (Hessian trace) vs ILP (our Ω, per-block) vs DP (our Ω, segment)

Uses existing segment_dp_omegas.csv (single-block entries [m:m+1])
and solves DP with L_max=1 (= per-block assignment).

Usage: !python analysis/eval_perblock_ilp.py
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


def run_inference(model, loader, norm_params, device, aq_bits=8):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params
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
            error = torch.sum((h_true - h_hat)**2, dim=[1, 2, 3])
            power = torch.sum(h_true**2, dim=[1, 2, 3])
            nmse_all.extend((error / (power + 1e-9)).cpu().numpy().tolist())
            r = calculate_su_miso_rate_mrt(h_true, h_hat, 20, device)
            rate_all.extend(r.cpu().numpy().tolist())
    return np.array(nmse_all), np.array(rate_all)


def main():
    print("=" * 60)
    print("  PER-BLOCK ILP (L_max=1) -- 3-way comparison data")
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
    K_bins = 5
    mid_bin = K_bins // 2

    # Load segment omegas (same as segment DP)
    cache_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas.csv")
    if not os.path.exists(cache_csv):
        print(f"[ERROR] {cache_csv} not found. Run segment_dp_policy.py first.")
        return
    df_cache = pd.read_csv(cache_csv)
    omega_nmse = {}
    for _, row in df_cache.iterrows():
        omega_nmse[(int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))] = row["omega_nmse"]

    # L_max=1: only single-block segments [m:m+1)
    L_max_ilp = 1
    segments_ilp = enumerate_segments(M, L_max_ilp)
    print(f"Segments (L_max=1): {len(segments_ilp)} (= {M} blocks × 1)")

    # Fill anchor entries
    for (l, r) in segments_ilp:
        for j in range(K_bins):
            omega_nmse[(l, r, anchor_bits, j)] = 0.0

    # Kappa
    layer_params = get_encoder_layer_params(net, fc_chunks=32)
    total_fp32 = sum(layer_params.get(bn, 0) * 32 * 32 for bn in block_names)
    kappa_seg = {}
    for (l, r) in segments_ilp:
        for b in bit_options:
            bops = sum(layer_params.get(fc_blocks[i], 0) * b * 16 for i in range(l, r))
            kappa_seg[(l, r, b)] = bops / total_fp32 if total_fp32 > 0 else 0
    non_fc_cost = sum((layer_params.get(bn, 0) * anchor_bits * 16) / total_fp32
                      for bn in non_fc_blocks)

    # Perfect rates
    perf_df = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv"))
    N = len(test_set)
    r_ref = perf_df["r_perf_20"].values[:N]

    # Sweep
    savings = np.arange(85, 97.01, 0.25).tolist()
    print(f"\nSweep: {len(savings)} saving levels")

    results = []
    policy_cache = {}

    omega_mid = {(l, r, b): omega_nmse.get((l, r, b, mid_bin), 0)
                 for (l, r) in segments_ilp for b in bit_options}

    for target_saving in tqdm(savings, desc="Per-block ILP"):
        target_fc = (1.0 - target_saving / 100.0) - non_fc_cost
        if target_fc < 0:
            target_fc = 0.001

        _, seg = solve_dp(M, segments_ilp, omega_mid, kappa_seg,
                          target_fc, bit_options, anchor_bits)
        pol = segmentation_to_policy(seg, fc_blocks, non_fc_blocks, anchor_bits)
        pol_key = str(sorted(pol.items()))

        if pol_key not in policy_cache:
            real_model.load_state_dict(original_state)
            apply_precision_policy(net, pol, device)
            nmse_arr, rate_arr = run_inference(
                net, test_loader, norm_params, device)
            nmse_db = 10 * np.log10(np.mean(nmse_arr) + 1e-15)
            out99 = float(np.mean(rate_arr[:N] < 0.99 * r_ref))
            out95 = float(np.mean(rate_arr[:N] < 0.95 * r_ref))
            policy_cache[pol_key] = (nmse_db, float(np.mean(nmse_arr)),
                                      float(np.mean(rate_arr)), out99, out95)

        nmse_db, nmse_lin, rate_mean, out99, out95 = policy_cache[pol_key]

        # Compute actual saving
        seg_bops = sum(layer_params.get(bn, 0) * anchor_bits * 16 for bn in non_fc_blocks)
        for (sl, sr, sb) in seg:
            for i in range(sl, sr):
                seg_bops += layer_params.get(fc_blocks[i], 0) * sb * 16
        actual_saving = (1.0 - seg_bops / total_fp32) * 100

        results.append({
            "saving": target_saving, "actual_saving": actual_saving,
            "method": "omega-ilp",
            "nmse_db": nmse_db, "nmse_linear": nmse_lin,
            "rate": rate_mean, "outage_99": out99, "outage_95": out95,
        })

    n_unique = len(policy_cache)
    print(f"{len(savings)} targets -> {n_unique} unique policies evaluated")

    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_CSV, "perblock_ilp_sweep.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df)} rows)")

    real_model.load_state_dict(original_state)
    print("Done.")


if __name__ == "__main__":
    main()
