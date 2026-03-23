"""
Local Contiguous Window Importance for FC chunks.

Instead of dyadic (1,2,4,8) fixed groups, use sliding contiguous windows
of length 1~4. For each block m, average over all windows containing m.

Example for m=5:
  L=1: {5}
  L=2: {4,5}, {5,6}
  L=3: {3,4,5}, {4,5,6}, {5,6,7}
  L=4: {2,3,4,5}, {3,4,5,6}, {4,5,6,7}, {5,6,7,8}

Ω_m^eff = ¼ · [ avg_L1(m) + avg_L2(m) + avg_L3(m) + avg_L4(m) ]

where avg_Lk(m) = mean of ΔV(window)/|window| over all length-k windows containing m.

Windows:
  L=1: 32 (already have from single-block perturbation)
  L=2: 31 (new)
  L=3: 30 (new)
  L=4: 29 (new)
  Total: 122 windows, new collection: 90 windows × 3 bits = 270 runs

Non-FC blocks: single-block importance as-is.

Usage (Colab):
    !python analysis/test_dyadic_importance.py
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
from rpmpq_v2 import get_encoder_block_names, get_encoder_layer_params, RESULTS_CSV
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
            nmse = (error / (power + 1e-9)).cpu().numpy()
            nmse_all.extend(nmse.tolist())
            for snr in snr_list:
                r = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
                rates_all[snr].extend(r.cpu().numpy().tolist())
    return np.array(nmse_all), {s: np.array(v) for s, v in rates_all.items()}


def main():
    print("=" * 70)
    print("  LOCAL CONTIGUOUS WINDOW IMPORTANCE (L=1,2,3,4)")
    print("  Sliding windows, equal scale weights")
    print("=" * 70)

    net, test_loader, norm_params, device = load_model()
    block_names = get_encoder_block_names(net, fc_chunks=32)
    fc_blocks = sorted([b for b in block_names if "fc_part" in b],
                       key=lambda x: int(re.search(r'(\d+)$', x).group()))
    non_fc_blocks = [b for b in block_names if "fc_part" not in b]
    N_fc = len(fc_blocks)  # 32

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    snr = 20

    perf_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    r_ref = pd.read_csv(perf_csv)[f"r_perf_{snr}"].values

    # ---- Anchor ----
    print("\n[1] Running anchor (all INT16)...")
    real_model.load_state_dict(original_state)
    apply_precision_policy(net, {bn: anchor_bits for bn in block_names}, device)
    nmse_anc, rates_anc = run_inference(net, test_loader, norm_params, device)
    r_anc = rates_anc[snr]
    N = len(nmse_anc)
    cos2_anc = np.clip((2**r_anc - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)
    print(f"  NMSE: {10*np.log10(np.mean(nmse_anc)+1e-15):.2f} dB, cos²θ: {np.mean(cos2_anc):.6f}")

    # ---- Scale 1: Load existing single-block data ----
    print("\n[2] Loading single-block data (L=1, 32 windows)...")
    pert_df = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perturbation.csv"))

    # omega_window[(L, start_idx, bit)] = (delta_nmse, delta_cos2)
    omega_window = {}

    for mi, bname in enumerate(fc_blocks):
        for b in bit_options:
            if b == anchor_bits:
                omega_window[(1, mi, b)] = (0.0, 0.0)
                continue
            mask = (pert_df["block_name"] == bname) & (pert_df["bits"] == b)
            if mask.sum() == 0:
                omega_window[(1, mi, b)] = (0.0, 0.0)
                continue
            df_mb = pert_df[mask].sort_values("sample_idx")
            nmse_p = df_mb["nmse_linear"].values
            r_p = df_mb[f"rate_{snr}"].values
            cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:len(r_p)] - 1 + 1e-12), 0, 1)
            omega_window[(1, mi, b)] = (
                float(np.mean(nmse_p - nmse_anc[:len(nmse_p)])),
                float(np.mean(cos2_anc[:len(cos2_p)] - cos2_p)))

    # Non-FC single-block
    omega_nonfc_nmse = {}
    omega_nonfc_cos2 = {}
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

    print(f"  L=1: 32 windows loaded")

    # ---- Scales 2, 3, 4: Sliding window perturbation ----
    for L in [2, 3, 4]:
        n_windows = N_fc - L + 1
        total_runs = n_windows * 3  # 3 non-anchor bits
        print(f"\n[L={L}] Collecting {n_windows} windows × 3 bits = {total_runs} runs...")

        for start in tqdm(range(n_windows), desc=f"L={L} windows"):
            window = fc_blocks[start : start + L]
            for b in bit_options:
                if b == anchor_bits:
                    omega_window[(L, start, b)] = (0.0, 0.0)
                    continue

                policy = {bn: anchor_bits for bn in block_names}
                for chunk in window:
                    policy[chunk] = b

                real_model.load_state_dict(original_state)
                apply_precision_policy(net, policy, device)
                nmse_p, rates_p = run_inference(net, test_loader, norm_params, device)
                r_p = rates_p[snr]
                cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

                omega_window[(L, start, b)] = (
                    float(np.mean(nmse_p - nmse_anc)),
                    float(np.mean(cos2_anc - cos2_p)))

        # Print a few examples
        for si in [0, n_windows//2, n_windows-1]:
            d_n, d_c = omega_window.get((L, si, 2), (0, 0))
            w = fc_blocks[si:si+L]
            print(f"  [{w[0]}~{w[-1]}] Ω_nmse@2={d_n:.6f}  Ω_cos2@2={d_c:.6f}")

    # ---- Combine: Per-block effective importance ----
    print("\n" + "=" * 70)
    print("  COMBINING: Local contiguous window importance")
    print("  Ω_m^eff = ¼·[avg_L1 + avg_L2 + avg_L3 + avg_L4]")
    print("=" * 70)

    omega_eff_nmse = {}
    omega_eff_cos2 = {}
    n_scales = 4
    w_scale = 1.0 / n_scales

    for mi, bname in enumerate(fc_blocks):
        for b in bit_options:
            if b == anchor_bits:
                omega_eff_nmse[(bname, b)] = 0.0
                omega_eff_cos2[(bname, b)] = 0.0
                continue

            total_nmse = 0.0
            total_cos2 = 0.0

            for L in [1, 2, 3, 4]:
                # All windows of length L that contain block mi
                contributions_nmse = []
                contributions_cos2 = []
                for start in range(max(0, mi - L + 1), min(N_fc - L + 1, mi + 1)):
                    d_n, d_c = omega_window.get((L, start, b), (0.0, 0.0))
                    # Per-block share = window delta / window length
                    contributions_nmse.append(d_n / L)
                    contributions_cos2.append(d_c / L)

                # Average over all windows at this scale
                if contributions_nmse:
                    total_nmse += w_scale * np.mean(contributions_nmse)
                    total_cos2 += w_scale * np.mean(contributions_cos2)

            omega_eff_nmse[(bname, b)] = total_nmse
            omega_eff_cos2[(bname, b)] = total_cos2

    # Non-FC: single-block as-is
    for bname in non_fc_blocks:
        for b in bit_options:
            omega_eff_nmse[(bname, b)] = omega_nonfc_nmse.get((bname, b), 0.0)
            omega_eff_cos2[(bname, b)] = omega_nonfc_cos2.get((bname, b), 0.0)

    # ---- Print per-block comparison ----
    print(f"\n{'Block':20s}  {'Single_n':>10s}  {'Window_n':>10s}  "
          f"{'Single_c':>10s}  {'Window_c':>10s}  {'n/c ratio':>10s}")
    for mi, bname in enumerate(fc_blocks):
        s_n = omega_window.get((1, mi, 2), (0, 0))[0]
        w_n = omega_eff_nmse.get((bname, 2), 0)
        s_c = omega_window.get((1, mi, 2), (0, 0))[1]
        w_c = omega_eff_cos2.get((bname, 2), 0)
        ratio = w_n / (w_c + 1e-12)
        print(f"  {bname:20s}  {s_n:10.6f}  {w_n:10.6f}  {s_c:10.6f}  {w_c:10.6f}  {ratio:10.4f}")

    # Ranking comparison
    imp_nmse = np.array([omega_eff_nmse.get((bn, 2), 0) for bn in block_names])
    imp_cos2 = np.array([omega_eff_cos2.get((bn, 2), 0) for bn in block_names])
    rho, _ = spearmanr(imp_nmse, imp_cos2)
    mask_ns = [i for i, bn in enumerate(block_names) if bn != "stem.0"]
    rho_ns, _ = spearmanr(imp_nmse[mask_ns], imp_cos2[mask_ns])
    print(f"\nSpearman(Ω_nmse, Ω_cos2) all blocks: {rho:.4f}")
    print(f"Spearman(Ω_nmse, Ω_cos2) no stem.0:  {rho_ns:.4f}")

    # Compare with single-block only ranking
    imp_s1_nmse = np.array([omega_window.get((1, fc_blocks.index(bn) if bn in fc_blocks else -1, 2), (0,0))[0]
                             if bn in fc_blocks else omega_nonfc_nmse.get((bn, 2), 0)
                             for bn in block_names])
    imp_s1_cos2 = np.array([omega_window.get((1, fc_blocks.index(bn) if bn in fc_blocks else -1, 2), (0,0))[1]
                             if bn in fc_blocks else omega_nonfc_cos2.get((bn, 2), 0)
                             for bn in block_names])
    rho_s1, _ = spearmanr(imp_s1_nmse, imp_s1_cos2)
    rho_s1_ns, _ = spearmanr(imp_s1_nmse[mask_ns], imp_s1_cos2[mask_ns])
    print(f"\n[Comparison] Single-block Spearman(nmse, cos2): {rho_s1:.4f} (no stem: {rho_s1_ns:.4f})")
    print(f"[Comparison] Window-avg  Spearman(nmse, cos2): {rho:.4f} (no stem: {rho_ns:.4f})")

    # ---- ILP comparison ----
    print("\n" + "=" * 70)
    print("  ILP COMPARISON: Window NMSE vs Window cos²θ")
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
        prob = LpProblem(f"win_{label}", LpMinimize)
        x = {}
        for m in range(M):
            x[m] = {}
            for bi in range(len(bit_options)):
                x[m][bi] = LpVariable(f"x_{m}_{bi}", cat="Binary")
        prob += lpSum(
            omega_dict.get((block_names[m], bit_options[bi]), 0) * x[m][bi]
            for m in range(M) for bi in range(len(bit_options)))
        for m in range(M):
            prob += lpSum(x[m][bi] for bi in range(len(bit_options))) == 1
        prob += lpSum(
            kappa.get((block_names[m], bit_options[bi]), 0) * x[m][bi]
            for m in range(M) for bi in range(len(bit_options))) <= budget
        prob.solve(PULP_CBC_CMD(msg=0))
        policy = {}
        for m in range(M):
            for bi, b in enumerate(bit_options):
                if x[m][bi].varValue is not None and x[m][bi].varValue > 0.5:
                    policy[block_names[m]] = b
                    break
        return policy

    for target_saving in [87.5, 90.0, 92.5, 95.0]:
        budget = 1.0 - target_saving / 100.0

        pol_nmse = solve_ilp(omega_eff_nmse, budget, f"nmse_{target_saving}")
        pol_cos2 = solve_ilp(omega_eff_cos2, budget, f"cos2_{target_saving}")

        diffs = [(k, pol_nmse.get(k), pol_cos2.get(k))
                 for k in block_names if pol_nmse.get(k) != pol_cos2.get(k)]

        # Evaluate both
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, pol_nmse, device)
        nmse_n, rates_n = run_inference(net, test_loader, norm_params, device)
        r_n = rates_n[snr]
        cos2_n = np.clip((2**r_n - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

        real_model.load_state_dict(original_state)
        apply_precision_policy(net, pol_cos2, device)
        nmse_c, rates_c = run_inference(net, test_loader, norm_params, device)
        r_c = rates_c[snr]
        cos2_c = np.clip((2**r_c - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

        nmse_db_n = 10*np.log10(np.mean(nmse_n)+1e-15)
        nmse_db_c = 10*np.log10(np.mean(nmse_c)+1e-15)
        outage_n = np.mean(r_n < 0.99 * r_ref[:N])
        outage_c = np.mean(r_c < 0.99 * r_ref[:N])

        print(f"\n--- Target: {target_saving}% ({len(diffs)} blocks differ) ---")
        print(f"  NMSE-ILP:  NMSE={nmse_db_n:.2f}dB  cos2={np.mean(cos2_n):.6f}  "
              f"rate={np.mean(r_n):.4f}  outage99={outage_n:.4f}")
        print(f"  cos2-ILP:  NMSE={nmse_db_c:.2f}dB  cos2={np.mean(cos2_c):.6f}  "
              f"rate={np.mean(r_c):.4f}  outage99={outage_c:.4f}")
        print(f"  Δcos²θ={np.mean(cos2_c)-np.mean(cos2_n):+.6f}  "
              f"ΔNMSE={nmse_db_c-nmse_db_n:+.3f}dB  "
              f"Δrate={np.mean(r_c)-np.mean(r_n):+.4f}  "
              f"Δoutage={outage_c-outage_n:+.4f}")

        if 0 < len(diffs) <= 15:
            for bname, bn, bc in diffs:
                print(f"    {bname:30s}: {bn} -> {bc}")

    real_model.load_state_dict(original_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
