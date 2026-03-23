"""
Group-based perturbation test.

Instead of per-block perturbation (linear regime), perturb groups of blocks
simultaneously to capture multi-block interaction effects.

Groups: FC chunks split into 4 groups of 8 chunks each.
For each group at bit b: set ALL 8 chunks to b, rest at anchor.
Measure Omega_group → split equally to per-block Omega → ILP.

Compare with single-block Omega ILP at same BOPs budget.

Usage (Colab):
    !python analysis/test_group_perturbation.py
"""
import os, sys, argparse, re
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
    compute_shortfall, RESULTS_CSV,
)
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
    sd = state.get("state_dict", state)
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net, test_loader, norm_params, device


def run_inference(model, loader, norm_params, device, aq_bits=8):
    """Run inference, return per-sample NMSE, rate, cos²θ."""
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


def make_groups(fc_blocks, group_size=8):
    """Split FC chunks into groups of group_size."""
    groups = []
    for i in range(0, len(fc_blocks), group_size):
        groups.append(fc_blocks[i:i+group_size])
    return groups


def main():
    print("=" * 70)
    print("  GROUP PERTURBATION TEST")
    print("  Groups of FC chunks perturbed together → captures interaction")
    print("=" * 70)

    net, test_loader, norm_params, device = load_model()
    block_names = get_encoder_block_names(net, fc_chunks=32)
    layer_params = get_encoder_layer_params(net, fc_chunks=32)
    fc_blocks = [b for b in block_names if "fc_part" in b]
    non_fc_blocks = [b for b in block_names if "fc_part" not in b]

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    bit_options = [16, 8, 4, 2]
    anchor_bits = 16
    snr = 20

    # Perfect rates for cos²θ
    perf_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    r_ref = pd.read_csv(perf_csv)[f"r_perf_{snr}"].values
    N = len(r_ref)

    # ---- Anchor ----
    print("\n[1] Running anchor (all INT16)...")
    real_model.load_state_dict(original_state)
    anchor_policy = {bn: anchor_bits for bn in block_names}
    apply_precision_policy(net, anchor_policy, device)
    nmse_anc, rates_anc = run_inference(net, test_loader, norm_params, device)
    r_anc = rates_anc[snr]
    cos2_anc = np.clip((2**r_anc - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)
    print(f"  NMSE: {10*np.log10(np.mean(nmse_anc)+1e-15):.2f} dB")
    print(f"  cos²θ: {np.mean(cos2_anc):.6f}")

    # ---- Group perturbation ----
    group_sizes = [4, 8, 16]  # test different group sizes

    for gs in group_sizes:
        groups = make_groups(fc_blocks, gs)
        print(f"\n{'='*70}")
        print(f"  GROUP SIZE = {gs} ({len(groups)} groups of FC chunks)")
        print(f"{'='*70}")

        # Omega_group[group_idx][bit] = mean delta per sample
        Omega_group_nmse = {}
        Omega_group_cos2 = {}
        Omega_group_rate = {}

        for gi, group in enumerate(groups):
            for b in bit_options:
                if b == anchor_bits:
                    continue

                # Set entire group to bit b, rest at anchor
                policy = {bn: anchor_bits for bn in block_names}
                for chunk in group:
                    policy[chunk] = b

                real_model.load_state_dict(original_state)
                apply_precision_policy(net, policy, device)
                nmse_p, rates_p = run_inference(net, test_loader, norm_params, device)
                r_p = rates_p[snr]
                cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:N] - 1 + 1e-12), 0, 1)

                # Group-level delta
                d_nmse = np.mean(nmse_p - nmse_anc)
                d_cos2 = np.mean(cos2_anc - cos2_p)
                d_rate = np.mean(r_anc - r_p)

                Omega_group_nmse[(gi, b)] = d_nmse
                Omega_group_cos2[(gi, b)] = d_cos2
                Omega_group_rate[(gi, b)] = d_rate

            print(f"  Group {gi} ({group[0]}~{group[-1]}): "
                  f"Ω_nmse@2={Omega_group_nmse.get((gi,2),0):.6f}  "
                  f"Ω_cos2@2={Omega_group_cos2.get((gi,2),0):.6f}")

        # ---- Split group Omega to per-block (equal split) ----
        Omega_block_nmse = {}
        Omega_block_cos2 = {}

        for gi, group in enumerate(groups):
            for b in bit_options:
                if b == anchor_bits:
                    for bname in group:
                        Omega_block_nmse[(bname, b)] = 0.0
                        Omega_block_cos2[(bname, b)] = 0.0
                    continue
                group_omega_n = Omega_group_nmse.get((gi, b), 0.0)
                group_omega_c = Omega_group_cos2.get((gi, b), 0.0)
                for bname in group:
                    Omega_block_nmse[(bname, b)] = group_omega_n / len(group)
                    Omega_block_cos2[(bname, b)] = group_omega_c / len(group)

        # Non-FC blocks: use existing single-block data
        pert_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perturbation.csv")
        if os.path.exists(pert_csv):
            pert_df = pd.read_csv(pert_csv)
            for bname in non_fc_blocks:
                for b in bit_options:
                    if b == anchor_bits:
                        Omega_block_nmse[(bname, b)] = 0.0
                        Omega_block_cos2[(bname, b)] = 0.0
                        continue
                    mask = (pert_df["block_name"] == bname) & (pert_df["bits"] == b)
                    if mask.sum() > 0:
                        df_mb = pert_df[mask].sort_values("sample_idx")
                        nmse_p = df_mb["nmse_linear"].values
                        r_p = df_mb[f"rate_{snr}"].values
                        cos2_p = np.clip((2**r_p - 1) / (2**r_ref[:len(r_p)] - 1 + 1e-12), 0, 1)
                        Omega_block_nmse[(bname, b)] = float(np.mean(nmse_p - nmse_anc[:len(nmse_p)]))
                        Omega_block_cos2[(bname, b)] = float(np.mean(cos2_anc[:len(cos2_p)] - cos2_p))
                    else:
                        Omega_block_nmse[(bname, b)] = 0.0
                        Omega_block_cos2[(bname, b)] = 0.0

        # ---- Load kappa ----
        kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_step1_nmse_kappa.csv")
        if not os.path.exists(kappa_csv):
            kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_kappa.csv")
        kappa = {}
        if os.path.exists(kappa_csv):
            kdf = pd.read_csv(kappa_csv)
            for _, r in kdf.iterrows():
                kappa[(r["block"], int(r["bits"]))] = r["kappa"]

        # ---- ILP comparison: NMSE-ILP vs cos²θ-ILP (with group Omega) ----
        print(f"\n  --- ILP comparison (group_size={gs}) ---")

        def solve_ilp(omega_dict, budget, label):
            prob = LpProblem(f"grp_{label}", LpMinimize)
            x = {}
            for m, bname in enumerate(block_names):
                x[m] = {}
                for bi, b in enumerate(bit_options):
                    x[m][bi] = LpVariable(f"x_{m}_{bi}", cat="Binary")
            prob += lpSum(
                omega_dict.get((block_names[m], bit_options[bi]), 0) * x[m][bi]
                for m in range(len(block_names))
                for bi in range(len(bit_options)))
            for m in range(len(block_names)):
                prob += lpSum(x[m][bi] for bi in range(len(bit_options))) == 1
            prob += lpSum(
                kappa.get((block_names[m], bit_options[bi]), 0) * x[m][bi]
                for m in range(len(block_names))
                for bi in range(len(bit_options))) <= budget
            prob.solve(PULP_CBC_CMD(msg=0))
            policy = {}
            for m in range(len(block_names)):
                for bi, b in enumerate(bit_options):
                    if x[m][bi].varValue is not None and x[m][bi].varValue > 0.5:
                        policy[block_names[m]] = b
                        break
            return policy

        for target_saving in [90.0, 92.5]:
            budget = 1.0 - target_saving / 100.0

            pol_nmse = solve_ilp(Omega_block_nmse, budget, f"nmse_{gs}_{target_saving}")
            pol_cos2 = solve_ilp(Omega_block_cos2, budget, f"cos2_{gs}_{target_saving}")

            diffs = [(k, pol_nmse.get(k), pol_cos2.get(k))
                     for k in block_names if pol_nmse.get(k) != pol_cos2.get(k)]

            print(f"\n  Target: {target_saving}% | Group size: {gs}")
            print(f"  Policy differences: {len(diffs)}/{len(block_names)} blocks")

            # Evaluate both policies with actual inference
            real_model.load_state_dict(original_state)
            apply_precision_policy(net, pol_nmse, device)
            res_nmse = run_inference(net, test_loader, norm_params, device)
            r_n = res_nmse[1][snr]
            cos2_n = np.clip((2**r_n - 1) / (2**r_ref[:len(r_n)] - 1 + 1e-12), 0, 1)

            real_model.load_state_dict(original_state)
            apply_precision_policy(net, pol_cos2, device)
            res_cos2_eval = run_inference(net, test_loader, norm_params, device)
            r_c = res_cos2_eval[1][snr]
            cos2_c = np.clip((2**r_c - 1) / (2**r_ref[:len(r_c)] - 1 + 1e-12), 0, 1)

            nmse_n = 10*np.log10(np.mean(res_nmse[0])+1e-15)
            nmse_c = 10*np.log10(np.mean(res_cos2_eval[0])+1e-15)

            print(f"  NMSE-ILP:  NMSE={nmse_n:.2f}dB  cos2={np.mean(cos2_n):.6f}  "
                  f"rate={np.mean(r_n):.4f}")
            print(f"  cos2-ILP:  NMSE={nmse_c:.2f}dB  cos2={np.mean(cos2_c):.6f}  "
                  f"rate={np.mean(r_c):.4f}")
            print(f"  Δcos²θ = {np.mean(cos2_c)-np.mean(cos2_n):+.6f}  "
                  f"ΔNMSE = {nmse_c-nmse_n:+.3f}dB  "
                  f"Δrate = {np.mean(r_c)-np.mean(r_n):+.4f}")

            if len(diffs) > 0 and len(diffs) <= 10:
                for bname, bn, bc in diffs:
                    print(f"    {bname:30s}: {bn} -> {bc}")

    # Restore
    real_model.load_state_dict(original_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
