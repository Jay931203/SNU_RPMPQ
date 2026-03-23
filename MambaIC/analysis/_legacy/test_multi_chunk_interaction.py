"""
Multi-chunk interaction test.

Single-block perturbation으로 안 보이던 FC chunk 조합 효과가 존재하는지 검증.

방법:
- 32 FC chunks를 랜덤하게 반반 (16 high-bit, 16 low-bit)으로 나눔
- 같은 BOPs인데 NMSE vs direction(cos²θ) 관계가 조합마다 다르면 → interaction 존재
- non-FC blocks (stem, mamba, proj_conv)는 고정

기존 perturbation 데이터에서는 못 하고, 직접 forward pass가 필요하지만
→ GPU 없이 CPU로 가능 (20000 samples × ~500 random policies → 느리지만 가능)

대신 더 간단한 접근: 기존 single-block perturbation 데이터로
"additive approximation vs actual" 차이를 추정.

실제로는: 기존 데이터의 per-sample (NMSE, Rate) 분포를 block별로 분석해서
multi-block 시나리오를 시뮬레이션.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

RESULTS_CSV = "results/csv"

pert = pd.read_csv(f"{RESULTS_CSV}/rpmpq_v2_perturbation.csv")
anc = pd.read_csv(f"{RESULTS_CSV}/rpmpq_v2_anchor.csv")
perf = pd.read_csv(f"{RESULTS_CSV}/rpmpq_v2_perfect_rates.csv")
N = len(anc)

nmse_anc = anc["nmse_linear"].values
snr = 20
r_anc = anc[f"rate_{snr}"].values
r_ref = perf[f"r_perf_{snr}"].values
cos2_anc = np.clip((2**r_anc - 1) / (2**r_ref - 1 + 1e-12), 0, 1)

block_names = sorted(pert["block_name"].unique())
fc_blocks = [b for b in block_names if "fc_part" in b]
non_fc_blocks = [b for b in block_names if "fc_part" not in b]

print(f"FC chunks: {len(fc_blocks)}")
print(f"Non-FC blocks: {len(non_fc_blocks)}: {non_fc_blocks}")
print(f"Samples: {N}")

# ============================================================
# 1. Per-sample, per-block delta vectors
# ============================================================
# For each FC chunk at bit=2 (most aggressive):
# delta_nmse[chunk][sample] = nmse_pert - nmse_anc
# delta_cos2[chunk][sample] = cos2_anc - cos2_pert

print("\nLoading per-sample deltas for FC chunks @ 2-bit...")
delta_nmse = {}  # {chunk_name: array(N,)}
delta_cos2 = {}
delta_rate = {}

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

    delta_nmse[bname] = nmse_p - nmse_anc
    delta_cos2[bname] = cos2_anc - cos2_p
    delta_rate[bname] = r_anc - r_p

valid_fc = sorted(delta_nmse.keys())
print(f"Valid FC chunks with delta data: {len(valid_fc)}")

# ============================================================
# 2. Additive approximation of multi-chunk effect
# ============================================================
# If we quantize chunks {A, B, C} to 2-bit simultaneously:
# Additive approx: delta_total ≈ delta_A + delta_B + delta_C
# (this is what single-block perturbation assumes)
#
# If interaction exists: actual delta ≠ sum of individual deltas
# We can't test this without GPU, but we CAN test:
# "If additive holds, do random half-splits give the same NMSE-rate ratio?"

print("\n" + "=" * 70)
print("  TEST: Random half-splits of FC chunks (additive approximation)")
print("  16 chunks @ 2-bit, 16 chunks @ 16-bit (anchor)")
print("  Does the NMSE vs cos²θ tradeoff vary across splits?")
print("=" * 70)

np.random.seed(42)
n_trials = 2000

results = []
for trial in range(n_trials):
    # Random split: 16 chunks get 2-bit, 16 stay at anchor
    perm = np.random.permutation(len(valid_fc))
    quant_chunks = [valid_fc[i] for i in perm[:16]]

    # Additive per-sample delta
    total_d_nmse = np.zeros(N)
    total_d_cos2 = np.zeros(N)
    total_d_rate = np.zeros(N)

    for chunk in quant_chunks:
        total_d_nmse += delta_nmse[chunk]
        total_d_cos2 += delta_cos2[chunk]
        total_d_rate += delta_rate[chunk]

    # Estimated metrics after multi-chunk quantization
    est_nmse = nmse_anc + total_d_nmse
    est_cos2 = cos2_anc - total_d_cos2
    est_rate = r_anc - total_d_rate

    mean_nmse = np.mean(est_nmse)
    mean_cos2 = np.mean(est_cos2)
    mean_rate = np.mean(est_rate)
    nmse_db = 10 * np.log10(mean_nmse + 1e-15)

    # Outage
    outage_99 = np.mean(est_rate < 0.99 * r_ref)
    outage_95 = np.mean(est_rate < 0.95 * r_ref)

    results.append({
        "trial": trial,
        "nmse_db": nmse_db,
        "mean_cos2": mean_cos2,
        "mean_rate": mean_rate,
        "outage_99": outage_99,
        "outage_95": outage_95,
        "chunks": str(sorted([valid_fc.index(c) for c in quant_chunks])),
    })

df = pd.DataFrame(results)

print(f"\n{n_trials} random half-splits:")
print(f"  NMSE range:    [{df['nmse_db'].min():.3f}, {df['nmse_db'].max():.3f}] dB  (spread: {df['nmse_db'].max()-df['nmse_db'].min():.3f} dB)")
print(f"  cos²θ range:   [{df['mean_cos2'].min():.6f}, {df['mean_cos2'].max():.6f}]  (spread: {df['mean_cos2'].max()-df['mean_cos2'].min():.6f})")
print(f"  Rate range:    [{df['mean_rate'].min():.4f}, {df['mean_rate'].max():.4f}]  (spread: {df['mean_rate'].max()-df['mean_rate'].min():.4f})")
print(f"  Outage99 range: [{df['outage_99'].min():.4f}, {df['outage_99'].max():.4f}]")

# Key: correlation between NMSE and cos²θ across splits
rho_nc, _ = spearmanr(df["nmse_db"], df["mean_cos2"])
rho_nr, _ = spearmanr(df["nmse_db"], df["mean_rate"])
print(f"\n  Spearman(NMSE, cos²θ) across splits: {rho_nc:.4f}")
print(f"  Spearman(NMSE, Rate) across splits:  {rho_nr:.4f}")

# Can we find splits that are Pareto-better in rate at same NMSE?
# Sort by NMSE, then check if rate varies at similar NMSE
df_sorted = df.sort_values("nmse_db")
# Group into NMSE bins
df_sorted["nmse_bin"] = pd.qcut(df_sorted["nmse_db"], 20, labels=False, duplicates="drop")
print(f"\n  Within-NMSE-bin rate variation:")
for bin_id in sorted(df_sorted["nmse_bin"].unique()):
    sub = df_sorted[df_sorted["nmse_bin"] == bin_id]
    if len(sub) < 10:
        continue
    nmse_mean = sub["nmse_db"].mean()
    rate_spread = sub["mean_rate"].max() - sub["mean_rate"].min()
    cos2_spread = sub["mean_cos2"].max() - sub["mean_cos2"].min()
    outage_spread = sub["outage_99"].max() - sub["outage_99"].min()
    print(f"    NMSE~{nmse_mean:.2f}dB (n={len(sub)}): rate_spread={rate_spread:.4f}, cos2_spread={cos2_spread:.6f}, outage99_spread={outage_spread:.4f}")

# ============================================================
# 3. Find best/worst rate splits at similar NMSE
# ============================================================
print("\n" + "=" * 70)
print("  BEST vs WORST rate splits at similar NMSE")
print("=" * 70)

# Pick the median NMSE bin
median_bin = df_sorted["nmse_bin"].median()
mid_sub = df_sorted[(df_sorted["nmse_bin"] >= median_bin - 1) &
                     (df_sorted["nmse_bin"] <= median_bin + 1)]
if len(mid_sub) > 20:
    best_rate = mid_sub.nlargest(5, "mean_rate")
    worst_rate = mid_sub.nsmallest(5, "mean_rate")

    print(f"\nNMSE range: [{mid_sub['nmse_db'].min():.2f}, {mid_sub['nmse_db'].max():.2f}] dB")
    print(f"\nBest rate splits (same NMSE, highest rate):")
    for _, r in best_rate.iterrows():
        print(f"  NMSE={r['nmse_db']:.3f}dB  Rate={r['mean_rate']:.4f}  cos2={r['mean_cos2']:.6f}  outage99={r['outage_99']:.4f}")

    print(f"\nWorst rate splits (same NMSE, lowest rate):")
    for _, r in worst_rate.iterrows():
        print(f"  NMSE={r['nmse_db']:.3f}dB  Rate={r['mean_rate']:.4f}  cos2={r['mean_cos2']:.6f}  outage99={r['outage_99']:.4f}")

    rate_gap = best_rate["mean_rate"].mean() - worst_rate["mean_rate"].mean()
    cos2_gap = best_rate["mean_cos2"].mean() - worst_rate["mean_cos2"].mean()
    outage_gap = worst_rate["outage_99"].mean() - best_rate["outage_99"].mean()
    print(f"\n  Rate gap: {rate_gap:.4f} bps/Hz")
    print(f"  cos²θ gap: {cos2_gap:.6f}")
    print(f"  Outage99 gap: {outage_gap:.4f}")

# ============================================================
# 4. Which FC chunks characterize good vs bad splits?
# ============================================================
print("\n" + "=" * 70)
print("  Which FC chunks appear more in high-rate vs low-rate splits?")
print("=" * 70)

import ast

# Among all splits, correlate "chunk i is quantized" with rate
chunk_presence = np.zeros((n_trials, len(valid_fc)), dtype=int)
for idx, row in df.iterrows():
    chunk_ids = ast.literal_eval(row["chunks"])
    for cid in chunk_ids:
        chunk_presence[idx, cid] = 1

rates_arr = df["mean_rate"].values
cos2_arr = df["mean_cos2"].values

print(f"\n  Per-chunk correlation with rate (negative = quantizing hurts rate more):")
chunk_rate_corr = []
for ci, cname in enumerate(valid_fc):
    rho_r, _ = spearmanr(chunk_presence[:, ci], rates_arr)
    rho_c, _ = spearmanr(chunk_presence[:, ci], cos2_arr)
    chunk_rate_corr.append((cname, rho_r, rho_c))

chunk_rate_corr.sort(key=lambda x: x[1])
print(f"\n  Most rate-damaging FC chunks (quantizing hurts rate most):")
for name, rho_r, rho_c in chunk_rate_corr[:10]:
    print(f"    {name:20s}: corr_rate={rho_r:+.4f}  corr_cos2={rho_c:+.4f}")

print(f"\n  Least rate-damaging FC chunks (quantizing hurts rate least):")
for name, rho_r, rho_c in chunk_rate_corr[-10:]:
    print(f"    {name:20s}: corr_rate={rho_r:+.4f}  corr_cos2={rho_c:+.4f}")

# Compare with NMSE correlation
chunk_nmse_corr = []
nmse_arr = df["nmse_db"].values
for ci, cname in enumerate(valid_fc):
    rho_n, _ = spearmanr(chunk_presence[:, ci], nmse_arr)
    chunk_nmse_corr.append((cname, rho_n))

# Do rate-corr and nmse-corr rankings agree?
rate_ranks = [x[0] for x in sorted(chunk_rate_corr, key=lambda x: x[1])]
nmse_ranks = [x[0] for x in sorted(chunk_nmse_corr, key=lambda x: x[1], reverse=True)]

from scipy.stats import kendalltau
rate_order = [rate_ranks.index(c) for c in valid_fc]
nmse_order = [nmse_ranks.index(c) for c in valid_fc]
tau, _ = kendalltau(rate_order, nmse_order)
print(f"\n  Kendall tau (rate ranking vs NMSE ranking): {tau:.4f}")
print(f"  (1.0 = identical, <0.8 = meaningful divergence)")
