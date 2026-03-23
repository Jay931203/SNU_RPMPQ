"""
Multi-chunk interaction test — GPU forward pass version.

Additive approximation이 아닌 실제 inference로 검증:
- 50 random multi-chunk policies (16 FC chunks INT2, 16 INT16)
- 각 policy로 실제 forward pass → actual NMSE, cos²θ 측정
- additive 추정치와 비교 → interaction 크기 확인
- 같은 NMSE에서 cos²θ가 갈라지는지 확인

Usage (Colab):
    !python analysis/test_interaction_gpu.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from train_ae import (
    apply_precision_policy, quantize_feedback_torch,
    calculate_su_miso_rate_mrt, CsiDataset,
    compute_hoyer_sparsity_bins,
)
from ModularModels import ModularAE
from rpmpq_v2 import get_encoder_block_names, get_encoder_layer_params

RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "csv")


def load_model():
    """Load model and test data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    train_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htrainout.mat"), "HT")
    test_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htestout.mat"), "HT",
        normalization_params=train_set.normalization_params)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=0)
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


def run_policy(net, test_loader, norm_params, device, policy, aq_bits=8):
    """Run inference with a specific policy, return per-sample NMSE and rate."""
    real_model = net.module if isinstance(net, torch.nn.DataParallel) else net
    original_state = {k: v.clone().cpu()
                      for k, v in real_model.state_dict().items()}

    real_model.load_state_dict(original_state)
    apply_precision_policy(net, policy, device)
    real_model.eval()

    min_val, range_val = norm_params
    snr_list = [10, 20, 30]
    nmse_all = []
    rates_all = {snr: [] for snr in snr_list}

    with torch.no_grad():
        for batch in test_loader:
            d = batch.to(device)
            z = real_model.encoder(d)
            if aq_bits > 0:
                z = quantize_feedback_torch(z, aq_bits)
            x_hat = real_model.decoder(z)

            h_true = (d * range_val) + min_val - 0.5
            h_hat = (x_hat * range_val) + min_val - 0.5

            error = torch.sum((h_true - h_hat) ** 2, dim=[1, 2, 3])
            power = torch.sum(h_true ** 2, dim=[1, 2, 3])
            nmse = (error / (power + 1e-9)).cpu().numpy()
            nmse_all.extend(nmse.tolist())

            for snr in snr_list:
                r = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
                rates_all[snr].extend(r.cpu().numpy().tolist())

    # Restore
    real_model.load_state_dict(original_state)

    return np.array(nmse_all), {s: np.array(v) for s, v in rates_all.items()}


def main():
    print("=" * 70)
    print("  MULTI-CHUNK INTERACTION TEST (GPU forward pass)")
    print("=" * 70)

    net, test_loader, norm_params, device = load_model()
    block_names = get_encoder_block_names(net, fc_chunks=32)
    fc_blocks = [b for b in block_names if "fc_part" in b]
    non_fc_blocks = [b for b in block_names if "fc_part" not in b]

    print(f"FC chunks: {len(fc_blocks)}")
    print(f"Non-FC blocks: {len(non_fc_blocks)}")

    # ---- Anchor (all INT16) ----
    print("\n[1/3] Running anchor policy (all INT16)...")
    anchor_policy = {bn: 16 for bn in block_names}
    nmse_anc, rates_anc = run_policy(net, test_loader, norm_params, device,
                                      anchor_policy)
    N = len(nmse_anc)

    snr = 20
    r_anc = rates_anc[snr]
    r_ref_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    r_ref = pd.read_csv(r_ref_csv)[f"r_perf_{snr}"].values
    cos2_anc = np.clip((2**r_anc - 1) / (2**r_ref - 1 + 1e-12), 0, 1)

    print(f"  Anchor NMSE: {10*np.log10(np.mean(nmse_anc)+1e-15):.2f} dB")
    print(f"  Anchor cos²θ: {np.mean(cos2_anc):.6f}")

    # ---- Load additive deltas for comparison ----
    print("\n[2/3] Loading single-block perturbation data for additive comparison...")
    pert = pd.read_csv(os.path.join(RESULTS_CSV, "rpmpq_v2_perturbation.csv"))
    delta_nmse_single = {}
    delta_cos2_single = {}
    for bname in fc_blocks:
        mask = (pert["block_name"] == bname) & (pert["bits"] == 2)
        if mask.sum() == 0:
            continue
        df_mb = pert[mask].sort_values("sample_idx")
        if len(df_mb) != N:
            continue
        nmse_p = df_mb["nmse_linear"].values
        r_p = df_mb[f"rate_{snr}"].values
        cos2_p = np.clip((2**r_p - 1) / (2**r_ref - 1 + 1e-12), 0, 1)
        delta_nmse_single[bname] = nmse_p - nmse_anc
        delta_cos2_single[bname] = cos2_anc - cos2_p

    valid_fc = sorted(delta_nmse_single.keys())
    print(f"  Valid FC chunks: {len(valid_fc)}")

    # ---- Random multi-chunk policies ----
    n_trials = 50
    np.random.seed(42)

    print(f"\n[3/3] Running {n_trials} random multi-chunk policies (GPU)...")
    print(f"  Each: 16 FC chunks @ INT2, 16 @ INT16, non-FC @ INT16")

    results = []

    for trial in tqdm(range(n_trials), desc="Multi-chunk policies"):
        # Random split: 16 chunks get INT2, 16 stay at INT16
        perm = np.random.permutation(len(valid_fc))
        quant_chunks = [valid_fc[i] for i in perm[:16]]
        keep_chunks = [valid_fc[i] for i in perm[16:]]

        # Build policy
        policy = {bn: 16 for bn in block_names}  # all INT16 base
        for chunk in quant_chunks:
            policy[chunk] = 2  # these get INT2

        # Actual forward pass
        nmse_actual, rates_actual = run_policy(
            net, test_loader, norm_params, device, policy)
        r_actual = rates_actual[snr]
        cos2_actual = np.clip(
            (2**r_actual - 1) / (2**r_ref - 1 + 1e-12), 0, 1)

        # Additive estimate
        est_nmse = nmse_anc.copy()
        est_cos2 = cos2_anc.copy()
        for chunk in quant_chunks:
            est_nmse = est_nmse + delta_nmse_single[chunk]
            est_cos2 = est_cos2 - delta_cos2_single[chunk]

        actual_nmse_db = 10 * np.log10(np.mean(nmse_actual) + 1e-15)
        est_nmse_db = 10 * np.log10(np.mean(est_nmse) + 1e-15)
        actual_cos2_mean = np.mean(cos2_actual)
        est_cos2_mean = np.mean(est_cos2)
        actual_rate_mean = np.mean(r_actual)
        outage_99 = np.mean(r_actual < 0.99 * r_ref)
        outage_95 = np.mean(r_actual < 0.95 * r_ref)

        results.append({
            "trial": trial,
            "actual_nmse_db": actual_nmse_db,
            "est_nmse_db": est_nmse_db,
            "nmse_error": actual_nmse_db - est_nmse_db,
            "actual_cos2": actual_cos2_mean,
            "est_cos2": est_cos2_mean,
            "cos2_error": actual_cos2_mean - est_cos2_mean,
            "actual_rate": actual_rate_mean,
            "outage_99": outage_99,
            "outage_95": outage_95,
            "quant_chunks": str(sorted([valid_fc.index(c) for c in quant_chunks])),
        })

    df = pd.DataFrame(results)

    # ---- Analysis ----
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n1. Additive approximation accuracy:")
    print(f"   NMSE error (actual - est): "
          f"mean={df['nmse_error'].mean():+.3f} dB, "
          f"std={df['nmse_error'].std():.3f} dB, "
          f"range=[{df['nmse_error'].min():+.3f}, {df['nmse_error'].max():+.3f}]")
    print(f"   cos²θ error (actual - est): "
          f"mean={df['cos2_error'].mean():+.6f}, "
          f"std={df['cos2_error'].std():.6f}, "
          f"range=[{df['cos2_error'].min():+.6f}, {df['cos2_error'].max():+.6f}]")

    print(f"\n2. Actual variation across policies:")
    print(f"   NMSE range: [{df['actual_nmse_db'].min():.3f}, "
          f"{df['actual_nmse_db'].max():.3f}] dB "
          f"(spread: {df['actual_nmse_db'].max()-df['actual_nmse_db'].min():.3f} dB)")
    print(f"   cos²θ range: [{df['actual_cos2'].min():.6f}, "
          f"{df['actual_cos2'].max():.6f}] "
          f"(spread: {df['actual_cos2'].max()-df['actual_cos2'].min():.6f})")
    print(f"   Rate range: [{df['actual_rate'].min():.4f}, "
          f"{df['actual_rate'].max():.4f}] "
          f"(spread: {df['actual_rate'].max()-df['actual_rate'].min():.4f})")

    print(f"\n3. NMSE vs cos²θ correlation (ACTUAL, not additive):")
    rho_nc, _ = spearmanr(df["actual_nmse_db"], df["actual_cos2"])
    rho_nr, _ = spearmanr(df["actual_nmse_db"], df["actual_rate"])
    print(f"   Spearman(actual_NMSE, actual_cos²θ) = {rho_nc:.4f}")
    print(f"   Spearman(actual_NMSE, actual_rate) = {rho_nr:.4f}")

    # Within-NMSE-bin analysis
    df_sorted = df.sort_values("actual_nmse_db")
    n_bins = min(5, len(df) // 5)
    if n_bins > 0:
        df_sorted["nmse_bin"] = pd.qcut(df_sorted["actual_nmse_db"],
                                         n_bins, labels=False,
                                         duplicates="drop")
        print(f"\n4. Within-NMSE-bin variation (ACTUAL):")
        for bin_id in sorted(df_sorted["nmse_bin"].unique()):
            sub = df_sorted[df_sorted["nmse_bin"] == bin_id]
            if len(sub) < 3:
                continue
            nmse_mean = sub["actual_nmse_db"].mean()
            cos2_spread = sub["actual_cos2"].max() - sub["actual_cos2"].min()
            rate_spread = sub["actual_rate"].max() - sub["actual_rate"].min()
            outage_spread = sub["outage_99"].max() - sub["outage_99"].min()
            print(f"   NMSE~{nmse_mean:.2f}dB (n={len(sub)}): "
                  f"cos2_spread={cos2_spread:.6f}, "
                  f"rate_spread={rate_spread:.4f}, "
                  f"outage99_spread={outage_spread:.4f}")

    # Best vs worst rate at similar NMSE
    print(f"\n5. Best vs worst cos²θ at similar NMSE:")
    mid = df_sorted.iloc[len(df_sorted)//4 : 3*len(df_sorted)//4]
    if len(mid) >= 10:
        best = mid.nlargest(5, "actual_cos2")
        worst = mid.nsmallest(5, "actual_cos2")
        print(f"   NMSE range: [{mid['actual_nmse_db'].min():.2f}, "
              f"{mid['actual_nmse_db'].max():.2f}] dB")
        print(f"\n   Best cos²θ (same NMSE):")
        for _, r in best.iterrows():
            print(f"     NMSE={r['actual_nmse_db']:.3f}dB  "
                  f"cos2={r['actual_cos2']:.6f}  "
                  f"rate={r['actual_rate']:.4f}  "
                  f"outage99={r['outage_99']:.4f}")
        print(f"\n   Worst cos²θ (same NMSE):")
        for _, r in worst.iterrows():
            print(f"     NMSE={r['actual_nmse_db']:.3f}dB  "
                  f"cos2={r['actual_cos2']:.6f}  "
                  f"rate={r['actual_rate']:.4f}  "
                  f"outage99={r['outage_99']:.4f}")

        cos2_gap = best["actual_cos2"].mean() - worst["actual_cos2"].mean()
        rate_gap = best["actual_rate"].mean() - worst["actual_rate"].mean()
        outage_gap = worst["outage_99"].mean() - best["outage_99"].mean()
        print(f"\n   cos²θ gap: {cos2_gap:.6f}")
        print(f"   Rate gap: {rate_gap:.4f} bps/Hz")
        print(f"   Outage99 gap: {outage_gap:.4f}")

    # Save
    out_csv = os.path.join(RESULTS_CSV, "interaction_gpu_test.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if abs(rho_nc) > 0.95:
        print("  NMSE ≈ cos²θ 상관관계 유지 (>0.95)")
        print("  → 이 시스템에서 방향성 분리 약함")
    elif abs(rho_nc) > 0.8:
        print("  NMSE-cos²θ 상관관계 부분적 (~0.8-0.95)")
        print("  → 방향성 차이 일부 존재, 알고리즘 개선 여지 있음")
    else:
        print("  NMSE-cos²θ 상관관계 낮음 (<0.8)")
        print("  → 방향성 분리 명확, rate-aware 알고리즘이 유효")


if __name__ == "__main__":
    main()
