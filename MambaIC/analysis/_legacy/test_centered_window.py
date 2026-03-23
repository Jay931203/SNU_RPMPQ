"""
Centered Window Importance (sizes 1, 3, 5).

Each block m gets ONE centered window per scale — no overlapping averaging.
- Size 1: {m}
- Size 3: {m-1, m, m+1}
- Size 5: {m-2, m-1, m, m+1, m+2}
Boundary blocks get truncated windows.

Ω_m^eff = ⅓·[ΔV_1/1 + ΔV_3/3 + ΔV_5/5]

Usage: !python analysis/test_centered_window.py
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
from rpmpq_v2 import get_encoder_block_names, RESULTS_CSV
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

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
    return net, test_loader, norm_params, device


def run_inference(model, loader, norm_params, device, aq_bits=8):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params
    snr_list = [10, 20, 30]
    nmse_all, rates_all = [], {s: [] for s in snr_list}
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


def centered_window(m, size, N_fc=32):
    """Return centered window indices, clipped at boundaries."""
    half = size // 2
    start = max(0, m - half)
    end = min(N_fc, m + half + 1)
    # If clipped, extend the other side
    if end - start < size:
        if start == 0:
            end = min(N_fc, start + size)
        elif end == N_fc:
            start = max(0, end - size)
    return list(range(start, end))


def main():
    print("=" * 70)
    print("  CENTERED WINDOW IMPORTANCE (sizes 1, 3, 5)")
    print("  Each block = center of its own window. No overlap averaging.")
    print("=" * 70)

    net, test_loader, norm_params, device = load_model()
    block_names = get_encoder_block_names(net, fc_chunks=32)
    fc_blocks = sorted([b for b in block_names if "fc_part" in b],
                       key=lambda x: int(re.search(r'(\d+)$', x).group()))
    non_fc_blocks = [b for b in block_names if "fc_part" not in b]
    N_fc = len(fc_blocks)

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    snr = 20

    r_ref = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
                         )[f"r_perf_{snr}"].values

    # ---- Anchor ----
    print("\n[1] Anchor (all INT16)...")
    real_model.load_state_dict(original_state)
    apply_precision_policy(net, {bn: anchor_bits for bn in block_names}, device)
    nmse_anc, rates_anc = run_inference(net, test_loader, norm_params, device)
    r_anc = rates_anc[snr]
    N = len(nmse_anc)
    cos2_anc = np.clip((2**r_anc - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)
    print(f"  NMSE: {10*np.log10(np.mean(nmse_anc)+1e-15):.2f} dB, cos²θ: {np.mean(cos2_anc):.6f}")

    # ---- Load single-block (size 1) ----
    print("\n[2] Loading single-block data (size=1)...")
    pert_df = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perturbation.csv"))

    # omega_centered[(m, size, bit)] = (d_nmse, d_cos2)
    omega_centered = {}

    for mi, bname in enumerate(fc_blocks):
        for b in bit_options:
            if b == anchor_bits:
                omega_centered[(mi, 1, b)] = (0.0, 0.0)
                continue
            mask = (pert_df["block_name"] == bname) & (pert_df["bits"] == b)
            if mask.sum() == 0:
                omega_centered[(mi, 1, b)] = (0.0, 0.0)
                continue
            df_mb = pert_df[mask].sort_values("sample_idx")
            nmse_p = df_mb["nmse_linear"].values
            r_p = df_mb[f"rate_{snr}"].values
            cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:len(r_p)] - 1 + 1e-12), 0, 1)
            omega_centered[(mi, 1, b)] = (
                float(np.mean(nmse_p - nmse_anc[:len(nmse_p)])),
                float(np.mean(cos2_anc[:len(cos2_p)] - cos2_p)))

    # Non-FC single-block
    omega_nonfc_nmse, omega_nonfc_cos2 = {}, {}
    for bname in non_fc_blocks:
        for b in bit_options:
            if b == anchor_bits:
                omega_nonfc_nmse[(bname, b)] = 0.0
                omega_nonfc_cos2[(bname, b)] = 0.0
                continue
            mask = (pert_df["block_name"] == bname) & (pert_df["bits"] == b)
            if mask.sum() == 0:
                omega_nonfc_nmse[(bname, b)] = 0.0
                omega_nonfc_cos2[(bname, b)] = 0.0
                continue
            df_mb = pert_df[mask].sort_values("sample_idx")
            nmse_p = df_mb["nmse_linear"].values
            r_p = df_mb[f"rate_{snr}"].values
            cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:len(r_p)] - 1 + 1e-12), 0, 1)
            omega_nonfc_nmse[(bname, b)] = float(np.mean(nmse_p - nmse_anc[:len(nmse_p)]))
            omega_nonfc_cos2[(bname, b)] = float(np.mean(cos2_anc[:len(cos2_p)] - cos2_p))

    # ---- Sizes 3, 5: Centered windows ----
    for size in [3, 5]:
        # Build unique windows (deduplicate)
        windows = {}  # {frozenset(indices): [list of block indices that use this window]}
        for mi in range(N_fc):
            w = centered_window(mi, size, N_fc)
            key = tuple(w)
            if key not in windows:
                windows[key] = []
            windows[key].append(mi)

        unique_windows = list(windows.keys())
        total_runs = len(unique_windows) * 3
        print(f"\n[Size={size}] {len(unique_windows)} unique windows × 3 bits = {total_runs} runs")

        # Cache results by window
        window_results = {}  # {(tuple(w), bit): (d_nmse, d_cos2)}

        for wi, w_indices in enumerate(tqdm(unique_windows, desc=f"Size {size}")):
            w_blocks = [fc_blocks[i] for i in w_indices]
            for b in bit_options:
                if b == anchor_bits:
                    window_results[(w_indices, b)] = (0.0, 0.0)
                    continue

                policy = {bn: anchor_bits for bn in block_names}
                for chunk in w_blocks:
                    policy[chunk] = b

                real_model.load_state_dict(original_state)
                apply_precision_policy(net, policy, device)
                nmse_p, rates_p = run_inference(net, test_loader, norm_params, device)
                r_p = rates_p[snr]
                cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

                window_results[(w_indices, b)] = (
                    float(np.mean(nmse_p - nmse_anc)),
                    float(np.mean(cos2_anc - cos2_p)))

        # Assign to each block (centered)
        for mi in range(N_fc):
            w = tuple(centered_window(mi, size, N_fc))
            for b in bit_options:
                omega_centered[(mi, size, b)] = window_results.get((w, b), (0.0, 0.0))

        # Print examples
        for mi in [0, N_fc//2, N_fc-1]:
            w = centered_window(mi, size, N_fc)
            d_n, d_c = omega_centered.get((mi, size, 2), (0, 0))
            print(f"  m={mi} ({fc_blocks[mi]}), window={fc_blocks[w[0]]}~{fc_blocks[w[-1]]}: "
                  f"Ω_n={d_n:.6f}  Ω_c={d_c:.6f}  ratio={d_n/(d_c+1e-12):.4f}")

    # ---- Combine ----
    print("\n" + "=" * 70)
    print("  COMBINING: Ω_m^eff = ⅓·[ΔV_1/1 + ΔV_3/3 + ΔV_5/5]")
    print("=" * 70)

    omega_eff_nmse, omega_eff_cos2 = {}, {}
    scales = [1, 3, 5]
    w_scale = 1.0 / len(scales)

    for mi, bname in enumerate(fc_blocks):
        for b in bit_options:
            if b == anchor_bits:
                omega_eff_nmse[(bname, b)] = 0.0
                omega_eff_cos2[(bname, b)] = 0.0
                continue

            total_n, total_c = 0.0, 0.0
            for size in scales:
                d_n, d_c = omega_centered.get((mi, size, b), (0.0, 0.0))
                w_len = len(centered_window(mi, size, N_fc))
                total_n += w_scale * d_n / w_len
                total_c += w_scale * d_c / w_len

            omega_eff_nmse[(bname, b)] = total_n
            omega_eff_cos2[(bname, b)] = total_c

    for bname in non_fc_blocks:
        for b in bit_options:
            omega_eff_nmse[(bname, b)] = omega_nonfc_nmse.get((bname, b), 0.0)
            omega_eff_cos2[(bname, b)] = omega_nonfc_cos2.get((bname, b), 0.0)

    # ---- Per-block table ----
    print(f"\n{'Block':20s}  {'S1_nmse':>10s}  {'Eff_nmse':>10s}  "
          f"{'S1_cos2':>10s}  {'Eff_cos2':>10s}  {'n/c':>8s}")
    for mi, bname in enumerate(fc_blocks):
        s1_n = omega_centered.get((mi, 1, 2), (0, 0))[0]
        eff_n = omega_eff_nmse.get((bname, 2), 0)
        s1_c = omega_centered.get((mi, 1, 2), (0, 0))[1]
        eff_c = omega_eff_cos2.get((bname, 2), 0)
        ratio = eff_n / (eff_c + 1e-12)
        print(f"  {bname:20s}  {s1_n:10.6f}  {eff_n:10.6f}  {s1_c:10.6f}  {eff_c:10.6f}  {ratio:8.4f}")

    # Spearman
    imp_n = np.array([omega_eff_nmse.get((bn, 2), 0) for bn in block_names])
    imp_c = np.array([omega_eff_cos2.get((bn, 2), 0) for bn in block_names])
    rho, _ = spearmanr(imp_n, imp_c)
    mask_ns = [i for i, bn in enumerate(block_names) if bn != "stem.0"]
    rho_ns, _ = spearmanr(imp_n[mask_ns], imp_c[mask_ns])

    imp_s1_n = np.array([omega_centered.get((fc_blocks.index(bn) if bn in fc_blocks else -1, 1, 2), (0,0))[0]
                          if bn in fc_blocks else omega_nonfc_nmse.get((bn, 2), 0)
                          for bn in block_names])
    imp_s1_c = np.array([omega_centered.get((fc_blocks.index(bn) if bn in fc_blocks else -1, 1, 2), (0,0))[1]
                          if bn in fc_blocks else omega_nonfc_cos2.get((bn, 2), 0)
                          for bn in block_names])
    rho_s1, _ = spearmanr(imp_s1_n, imp_s1_c)
    rho_s1_ns, _ = spearmanr(imp_s1_n[mask_ns], imp_s1_c[mask_ns])

    print(f"\nSingle-block Spearman(nmse, cos2): {rho_s1:.4f} (no stem: {rho_s1_ns:.4f})")
    print(f"Centered-win Spearman(nmse, cos2): {rho:.4f} (no stem: {rho_ns:.4f})")

    # ---- ILP ----
    print("\n" + "=" * 70)
    print("  ILP: Centered-window NMSE vs cos²θ")
    print("=" * 70)

    kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_step1_nmse_kappa.csv")
    if not os.path.exists(kappa_csv):
        kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_kappa.csv")
    kappa = {}
    if os.path.exists(kappa_csv):
        kdf = pd.read_csv(kappa_csv)
        for _, r in kdf.iterrows():
            kappa[(r["block"], int(r["bits"]))] = r["kappa"]

    M = len(block_names)

    def solve_ilp(omega_dict, budget, label):
        prob = LpProblem(f"cw_{label}", LpMinimize)
        x = {}
        for m in range(M):
            x[m] = {}
            for bi in range(len(bit_options)):
                x[m][bi] = LpVariable(f"x_{m}_{bi}", cat="Binary")
        prob += lpSum(omega_dict.get((block_names[m], bit_options[bi]), 0) * x[m][bi]
                      for m in range(M) for bi in range(len(bit_options)))
        for m in range(M):
            prob += lpSum(x[m][bi] for bi in range(len(bit_options))) == 1
        prob += lpSum(kappa.get((block_names[m], bit_options[bi]), 0) * x[m][bi]
                      for m in range(M) for bi in range(len(bit_options))) <= budget
        prob.solve(PULP_CBC_CMD(msg=0))
        policy = {}
        for m in range(M):
            for bi, b in enumerate(bit_options):
                if x[m][bi].varValue is not None and x[m][bi].varValue > 0.5:
                    policy[block_names[m]] = b
                    break
        return policy

    for target in [87.5, 90.0, 92.5, 95.0]:
        budget = 1.0 - target / 100.0
        pol_n = solve_ilp(omega_eff_nmse, budget, f"n_{target}")
        pol_c = solve_ilp(omega_eff_cos2, budget, f"c_{target}")

        diffs = [(k, pol_n.get(k), pol_c.get(k))
                 for k in block_names if pol_n.get(k) != pol_c.get(k)]

        real_model.load_state_dict(original_state)
        apply_precision_policy(net, pol_n, device)
        res_n = run_inference(net, test_loader, norm_params, device)
        r_n = res_n[1][snr]
        cos2_n = np.clip((2**r_n - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

        real_model.load_state_dict(original_state)
        apply_precision_policy(net, pol_c, device)
        res_c = run_inference(net, test_loader, norm_params, device)
        r_c = res_c[1][snr]
        cos2_c = np.clip((2**r_c - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

        n_db = 10*np.log10(np.mean(res_n[0])+1e-15)
        c_db = 10*np.log10(np.mean(res_c[0])+1e-15)
        out_n = np.mean(r_n < 0.99 * r_ref[:N])
        out_c = np.mean(r_c < 0.99 * r_ref[:N])

        print(f"\n--- Target: {target}% ({len(diffs)} blocks differ) ---")
        print(f"  NMSE-ILP:  NMSE={n_db:.2f}dB  cos2={np.mean(cos2_n):.6f}  "
              f"rate={np.mean(r_n):.4f}  outage99={out_n:.4f}")
        print(f"  cos2-ILP:  NMSE={c_db:.2f}dB  cos2={np.mean(cos2_c):.6f}  "
              f"rate={np.mean(r_c):.4f}  outage99={out_c:.4f}")
        print(f"  Δcos²θ={np.mean(cos2_c)-np.mean(cos2_n):+.6f}  "
              f"ΔNMSE={c_db-n_db:+.3f}dB  "
              f"Δrate={np.mean(r_c)-np.mean(r_n):+.4f}  "
              f"Δoutage={out_c-out_n:+.4f}")
        if 0 < len(diffs) <= 15:
            for bname, bn, bc in diffs:
                print(f"    {bname:30s}: {bn} -> {bc}")

    real_model.load_state_dict(original_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
