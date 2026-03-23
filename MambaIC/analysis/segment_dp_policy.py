"""
Segment-Level DP Policy Optimization.

Contiguous segments of FC chunks + DP for optimal segmentation under budget.
Captures inter-block interaction naturally via segment-level perturbation.

Usage: !python analysis/segment_dp_policy.py
"""
import os, sys, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

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
from rpmpq_v2 import (
    get_encoder_block_names, get_encoder_layer_params,
    compute_all_zeta, build_kernel, RESULTS_CSV,
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


def run_inference(model, loader, norm_params, device, aq_bits=8):
    """Return per-sample NMSE and rates."""
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params
    snr_list = [10, 20, 30]
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


def enumerate_segments(M, L_max):
    """Enumerate all contiguous segments [l:r] with length <= L_max."""
    segments = []
    for l in range(M):
        for r in range(l, min(l + L_max, M)):
            segments.append((l, r + 1))  # [l, r+1) python-style
    return segments


def collect_segment_omegas(net, test_loader, norm_params, device,
                            fc_blocks, block_names, segments, bit_options,
                            anchor_bits, nmse_anc, cos2_anc, zeta_vals,
                            zeta_edges, snr=20, r_ref=None):
    """Collect Omega for each (segment, bit, zeta-bin).

    Caches to CSV. If cache exists, skips GPU inference.

    Returns:
        omega_nmse: dict[(l,r,b,j)] -> float
        omega_cos2: dict[(l,r,b,j)] -> float
        kappa_seg:  dict[(l,r,b)] -> float (BOPs cost)
    """
    cache_csv = os.path.join(RESULTS_CSV, "segment_dp_omegas.csv")

    # Kappa (always compute, CPU only)
    layer_params = get_encoder_layer_params(net, fc_chunks=32)
    total_fp32_bops = sum(layer_params.get(bn, 0) * 32 * 32 for bn in block_names)
    kappa_seg = {}
    for (l, r) in segments:
        seg_blocks = fc_blocks[l:r]
        for b in bit_options:
            bops = sum(layer_params.get(bn, 0) * b * 16 for bn in seg_blocks)
            kappa_seg[(l, r, b)] = bops / total_fp32_bops if total_fp32_bops > 0 else 0

    N = len(nmse_anc)
    K_bins = len(zeta_edges) - 1

    # Try cache (check if shortfall columns exist)
    if os.path.exists(cache_csv):
        df_c = pd.read_csv(cache_csv)
        has_shortfall = "omega_sf99" in df_c.columns
        if has_shortfall:
            print(f"  Loading cached segment omegas (with shortfall) from {cache_csv}")
            omega_nmse, omega_cos2, omega_sf99, omega_sf95 = {}, {}, {}, {}
            for _, row in df_c.iterrows():
                key = (int(row["l"]), int(row["r"]), int(row["b"]), int(row["j"]))
                omega_nmse[key] = row["omega_nmse"]
                omega_cos2[key] = row["omega_cos2"]
                omega_sf99[key] = row["omega_sf99"]
                omega_sf95[key] = row["omega_sf95"]
            for (l, r) in segments:
                for j in range(K_bins):
                    omega_nmse[(l, r, anchor_bits, j)] = 0.0
                    omega_cos2[(l, r, anchor_bits, j)] = 0.0
                    omega_sf99[(l, r, anchor_bits, j)] = 0.0
                    omega_sf95[(l, r, anchor_bits, j)] = 0.0
            print(f"  Loaded {len(df_c)} entries")
            return omega_nmse, omega_cos2, omega_sf99, omega_sf95, kappa_seg
        else:
            print(f"  Cache exists but missing shortfall columns. Re-collecting...")

    # Collect via GPU
    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    k_indices = np.clip(np.digitize(zeta_vals, zeta_edges) - 1, 0, K_bins - 1)

    # Anchor shortfall
    r_anc_snr = None  # will compute after anchor inference below
    # We need anchor rate for shortfall — use the anchor rates passed via cos2_anc
    # Actually, we need raw r_anc. Recompute from cos2_anc and r_ref:
    # cos2 = (2^r - 1) / (2^r_ref - 1) → 2^r = 1 + cos2*(2^r_ref - 1) → r = log2(...)
    # Simpler: just run anchor inference to get rates
    real_model.load_state_dict(original_state)
    anc_policy = {bn: anchor_bits for bn in block_names}
    apply_precision_policy(net, anc_policy, device)
    _, rates_anc_full = run_inference(net, test_loader, norm_params, device)
    r_anc_raw = rates_anc_full[snr]

    gamma_99, gamma_95 = 0.99, 0.95
    V_anc_99 = np.maximum(0, gamma_99 * r_ref[:N] - r_anc_raw) / (gamma_99 * r_ref[:N] + 1e-6)
    V_anc_95 = np.maximum(0, gamma_95 * r_ref[:N] - r_anc_raw) / (gamma_95 * r_ref[:N] + 1e-6)

    omega_nmse, omega_cos2, omega_sf99, omega_sf95 = {}, {}, {}, {}
    cache_rows = []
    total_runs = len(segments) * (len(bit_options) - 1)
    print(f"  Collecting {len(segments)} segments × {len(bit_options)-1} bits = {total_runs} runs")

    for seg_idx, (l, r) in enumerate(tqdm(segments, desc="Segments")):
        seg_blocks = fc_blocks[l:r]
        for b in bit_options:
            if b == anchor_bits:
                for j in range(K_bins):
                    omega_nmse[(l, r, b, j)] = 0.0
                    omega_cos2[(l, r, b, j)] = 0.0
                    omega_sf99[(l, r, b, j)] = 0.0
                    omega_sf95[(l, r, b, j)] = 0.0
                continue

            policy = {bn: anchor_bits for bn in block_names}
            for chunk in seg_blocks:
                policy[chunk] = b

            real_model.load_state_dict(original_state)
            apply_precision_policy(net, policy, device)
            nmse_p, rates_p = run_inference(net, test_loader, norm_params, device)
            r_p = rates_p[snr]
            cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

            V_pert_99 = np.maximum(0, gamma_99 * r_ref[:N] - r_p) / (gamma_99 * r_ref[:N] + 1e-6)
            V_pert_95 = np.maximum(0, gamma_95 * r_ref[:N] - r_p) / (gamma_95 * r_ref[:N] + 1e-6)

            for j in range(K_bins):
                mask = k_indices == j
                if mask.sum() == 0:
                    omega_nmse[(l, r, b, j)] = 0.0
                    omega_cos2[(l, r, b, j)] = 0.0
                    omega_sf99[(l, r, b, j)] = 0.0
                    omega_sf95[(l, r, b, j)] = 0.0
                else:
                    omega_nmse[(l, r, b, j)] = float(np.mean(nmse_p[mask] - nmse_anc[mask]))
                    omega_cos2[(l, r, b, j)] = float(np.mean(cos2_anc[mask] - cos2_p[mask]))
                    omega_sf99[(l, r, b, j)] = float(np.mean(V_pert_99[mask] - V_anc_99[mask]))
                    omega_sf95[(l, r, b, j)] = float(np.mean(V_pert_95[mask] - V_anc_95[mask]))
                cache_rows.append({"l": l, "r": r, "b": b, "j": j,
                                   "omega_nmse": omega_nmse[(l, r, b, j)],
                                   "omega_cos2": omega_cos2[(l, r, b, j)],
                                   "omega_sf99": omega_sf99[(l, r, b, j)],
                                   "omega_sf95": omega_sf95[(l, r, b, j)]})

    real_model.load_state_dict(original_state)
    pd.DataFrame(cache_rows).to_csv(cache_csv, index=False)
    print(f"  Cached -> {cache_csv}")
    return omega_nmse, omega_cos2, omega_sf99, omega_sf95, kappa_seg


def solve_dp(M, segments, omega, kappa, budget, bit_options, anchor_bits):
    """DP to find optimal contiguous segmentation under budget.

    F(m, c) = min distortion using blocks [0..m) with cost <= c

    Args:
        M: total number of FC blocks
        segments: list of (l, r) tuples
        omega: dict[(l,r,b,j)] -> distortion (for a specific j)
        kappa: dict[(l,r,b)] -> cost
        budget: max total cost
        bit_options: [16, 8, 4, 2]
        anchor_bits: 16

    Returns:
        best_distortion, segmentation list [(l, r, b), ...]
    """
    # Discretize budget into steps
    import math
    C_steps = 3000  # match segment_dp_baselines.py resolution
    c_step = budget / C_steps

    INF = float('inf')
    # F[m][c] = min distortion using blocks [0..m) with cost index <= c
    F = [[INF] * (C_steps + 1) for _ in range(M + 1)]
    # Backtrack: which (l, b) produced F[m][c]
    back = [[None] * (C_steps + 1) for _ in range(M + 1)]

    # Base case: 0 blocks, 0 distortion
    for c in range(C_steps + 1):
        F[0][c] = 0.0

    # Build segment lookup by endpoint
    segs_ending_at = {}  # m -> [(l, r=m)]
    for (l, r) in segments:
        if r not in segs_ending_at:
            segs_ending_at[r] = []
        segs_ending_at[r].append(l)

    for m in range(1, M + 1):
        # Every block must be covered by exactly one segment.
        # F[m][c] starts at INF — can only be reached via a segment ending at m.

        # For each segment ending at m
        if m in segs_ending_at:
            for l in segs_ending_at[m]:
                for b in bit_options:
                    seg_cost = kappa.get((l, m, b), 0)
                    seg_cost_idx = math.ceil(seg_cost / c_step) if c_step > 0 else 0
                    seg_dist = omega.get((l, m, b), 0)

                    if seg_cost_idx > C_steps:
                        continue

                    for c in range(seg_cost_idx, C_steps + 1):
                        prev_c = c - seg_cost_idx
                        if prev_c < 0:
                            continue
                        candidate = F[l][prev_c] + seg_dist
                        if candidate < F[m][c]:
                            F[m][c] = candidate
                            back[m][c] = (l, b)

    # Find best solution
    best_c = 0
    best_val = INF
    for c in range(C_steps + 1):
        if F[M][c] < best_val:
            best_val = F[M][c]
            best_c = c

    # Backtrack to recover segmentation
    segmentation = []
    m = M
    c = best_c
    while m > 0 and back[m][c] is not None:
        l, b = back[m][c]
        segmentation.append((l, m, b))
        seg_cost_idx = math.ceil(kappa.get((l, m, b), 0) / c_step) if c_step > 0 else 0
        c = c - seg_cost_idx
        m = l

    # If we didn't reach 0, fill remaining with anchor
    if m > 0:
        segmentation.append((0, m, anchor_bits))

    segmentation.reverse()
    return best_val, segmentation


def segmentation_to_policy(segmentation, fc_blocks, non_fc_blocks, anchor_bits=16):
    """Convert segmentation list to full policy dict."""
    policy = {bn: anchor_bits for bn in non_fc_blocks}
    for (l, r, b) in segmentation:
        for i in range(l, r):
            policy[fc_blocks[i]] = b
    return policy


def main():
    print("=" * 70)
    print("  SEGMENT-LEVEL DP POLICY OPTIMIZATION")
    print("  Contiguous segments + DP (not per-block ILP)")
    print("=" * 70)

    net, test_set, test_loader, norm_params, device = load_model()
    block_names = get_encoder_block_names(net, fc_chunks=32)
    fc_blocks = sorted([b for b in block_names if "fc_part" in b],
                       key=lambda x: int(re.search(r'(\d+)$', x).group()))
    non_fc_blocks = [b for b in block_names if "fc_part" not in b]
    M = len(fc_blocks)  # 32

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    snr = 20
    L_max = 6  # max segment length
    K_bins = 5  # zeta bins (fewer for speed)

    # Perfect rates
    r_ref = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
                         )[f"r_perf_{snr}"].values

    # Zeta values
    zeta_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_zeta.csv")
    if os.path.exists(zeta_csv):
        zeta_vals = pd.read_csv(zeta_csv)["zeta_proxy"].values
    else:
        K_d = build_kernel(32, 1.0)
        K_a = build_kernel(32, 1.0)
        zeta_vals = compute_all_zeta(test_set, K_d, K_a, use_proxy=True)

    zeta_edges = np.quantile(zeta_vals, np.linspace(0, 1, K_bins + 1))
    zeta_edges[0] -= 1e-6
    zeta_edges[-1] += 1e-6

    # ---- Anchor ----
    print("\n[1] Anchor (all INT16)...")
    real_model.load_state_dict(original_state)
    apply_precision_policy(net, {bn: anchor_bits for bn in block_names}, device)
    nmse_anc, rates_anc = run_inference(net, test_loader, norm_params, device)
    N = len(nmse_anc)
    r_anc = rates_anc[snr]
    cos2_anc = np.clip((2**r_anc - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)
    print(f"  NMSE: {10*np.log10(np.mean(nmse_anc)+1e-15):.2f} dB, cos²θ: {np.mean(cos2_anc):.6f}")

    # ---- Enumerate segments ----
    segments = enumerate_segments(M, L_max)
    print(f"\n[2] Segments: {len(segments)} (M={M}, L_max={L_max})")
    print(f"  Examples: {segments[:3]} ... {segments[-3:]}")

    # ---- Collect segment Omegas ----
    print(f"\n[3] Collecting segment-level distortion...")
    omega_nmse, omega_cos2, omega_sf99, omega_sf95, kappa_seg = collect_segment_omegas(
        net, test_loader, norm_params, device,
        fc_blocks, block_names, segments, bit_options, anchor_bits,
        nmse_anc, cos2_anc, zeta_vals, zeta_edges, snr, r_ref)

    # Non-FC blocks: add their kappa (fixed at anchor for now)
    layer_params = get_encoder_layer_params(net, fc_chunks=32)
    total_fp32_bops = sum(layer_params.get(bn, 0) * 32 * 32 for bn in block_names)
    non_fc_kappa = {}
    for bn in non_fc_blocks:
        for b in bit_options:
            non_fc_kappa[(bn, b)] = (layer_params.get(bn, 0) * b * 16) / total_fp32_bops

    # ---- DP for each zeta-bin and budget ----
    print(f"\n[4] Running DP optimization...")

    budget_savings = [87.5, 90.0, 92.5, 95.0]

    for j in range(K_bins):
        print(f"\n--- Zeta bin {j}/{K_bins} ---")

        for target_saving in budget_savings:
            # Budget for FC blocks only (non-FC at anchor)
            non_fc_cost = sum(non_fc_kappa.get((bn, anchor_bits), 0) for bn in non_fc_blocks)
            total_budget = 1.0 - target_saving / 100.0
            fc_budget = total_budget - non_fc_cost
            if fc_budget < 0:
                fc_budget = 0.001

            # Extract omega for this zeta-bin
            objectives = {
                "nmse": {(l, r, b): omega_nmse.get((l, r, b, j), 0)
                         for (l, r) in segments for b in bit_options},
                "cos2": {(l, r, b): omega_cos2.get((l, r, b, j), 0)
                         for (l, r) in segments for b in bit_options},
                "sf99": {(l, r, b): omega_sf99.get((l, r, b, j), 0)
                         for (l, r) in segments for b in bit_options},
                "sf95": {(l, r, b): omega_sf95.get((l, r, b, j), 0)
                         for (l, r) in segments for b in bit_options},
            }

            for obj_name, omega_j in objectives.items():
                dist, seg = solve_dp(M, segments, omega_j, kappa_seg,
                                      fc_budget, bit_options, anchor_bits)
                pol = segmentation_to_policy(seg, fc_blocks, non_fc_blocks, anchor_bits)

                real_model.load_state_dict(original_state)
                apply_precision_policy(net, pol, device)
                res = run_inference(net, test_loader, norm_params, device)
                r_eval = res[1][snr]
                cos2_eval = np.clip((2**r_eval - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)
                nmse_db = 10*np.log10(np.mean(res[0])+1e-15)
                outage_99 = np.mean(r_eval < 0.99 * r_ref[:N])
                outage_95 = np.mean(r_eval < 0.95 * r_ref[:N])

                seg_str = " ".join(f"[{l}:{r}]INT{b}" for (l, r, b) in seg)
                print(f"    {obj_name:5s}-DP: NMSE={nmse_db:.2f}dB  cos2={np.mean(cos2_eval):.6f}  "
                      f"out99={outage_99:.4f}  out95={outage_95:.4f}")

            print()  # separator between savings

    real_model.load_state_dict(original_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
