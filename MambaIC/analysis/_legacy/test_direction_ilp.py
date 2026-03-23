"""Compare NMSE-based ILP vs Direction-based ILP using existing perturbation data."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

RESULTS_CSV = "results/csv"

pert = pd.read_csv(f"{RESULTS_CSV}/rpmpq_v2_perturbation.csv")
anc = pd.read_csv(f"{RESULTS_CSV}/rpmpq_v2_anchor.csv")
perf = pd.read_csv(f"{RESULTS_CSV}/rpmpq_v2_perfect_rates.csv")
N = len(anc)

nmse_anc = anc["nmse_linear"].values
block_names = sorted(pert["block_name"].unique())
bit_options = [16, 8, 4, 2]
M = len(block_names)

# === Build importance: NMSE vs Direction (cos^2 theta) ===
snr = 20
r_anc = anc[f"rate_{snr}"].values
r_ref = perf[f"r_perf_{snr}"].values
cos2_anc = np.clip((2**r_anc - 1) / (2**r_ref - 1 + 1e-12), 0, 1)

Omega_nmse = {}
Omega_dir = {}

for bname in block_names:
    for b in bit_options:
        mask = (pert["block_name"] == bname) & (pert["bits"] == b)
        if mask.sum() == 0:
            Omega_nmse[(bname, b)] = 0.0
            Omega_dir[(bname, b)] = 0.0
            continue
        df_mb = pert[mask].sort_values("sample_idx")
        if len(df_mb) != N:
            Omega_nmse[(bname, b)] = 0.0
            Omega_dir[(bname, b)] = 0.0
            continue
        nmse_pert = df_mb["nmse_linear"].values
        r_pert = df_mb[f"rate_{snr}"].values
        cos2_pert = np.clip((2**r_pert - 1) / (2**r_ref - 1 + 1e-12), 0, 1)
        Omega_nmse[(bname, b)] = float(np.mean(nmse_pert - nmse_anc))
        Omega_dir[(bname, b)] = float(np.mean(cos2_anc - cos2_pert))

# === Kappa ===
kdf = pd.read_csv(f"{RESULTS_CSV}/rpmpq_v2_step1_nmse_kappa.csv")
kappa = {}
for _, r in kdf.iterrows():
    kappa[(r["block"], int(r["bits"]))] = r["kappa"]

# === ILP solver ===
def solve_ilp(omega_dict, budget, label):
    prob = LpProblem(f"policy_{label}", LpMinimize)
    x = {}
    for m_idx, bname in enumerate(block_names):
        x[m_idx] = {}
        for bi, b in enumerate(bit_options):
            x[m_idx][bi] = LpVariable(f"x_{m_idx}_{bi}", cat="Binary")
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

# === Compare at multiple savings ===
print("=" * 80)
print("  NMSE-based ILP vs Direction-based ILP (SNR=20dB)")
print("=" * 80)

for target_saving in [87.5, 90.0, 92.5, 95.0]:
    budget = 1.0 - target_saving / 100.0
    pol_nmse = solve_ilp(Omega_nmse, budget, f"nmse_{target_saving}")
    pol_dir = solve_ilp(Omega_dir, budget, f"dir_{target_saving}")

    diffs = [(k, pol_nmse.get(k), pol_dir.get(k))
             for k in block_names if pol_nmse.get(k) != pol_dir.get(k)]

    print(f"\n--- Target saving: {target_saving}% (budget={budget:.4f}) ---")
    print(f"  Policy differences: {len(diffs)}/{M} blocks differ")

    if len(diffs) > 0:
        print(f"  {'Block':35s}  NMSE-ILP  Dir-ILP")
        for bname, b_nmse, b_dir in diffs[:15]:
            marker = ""
            if bname == "stem.0":
                marker = " <-- STEM"
            elif "fc_part" in bname:
                marker = " <-- FC"
            print(f"    {bname:35s}  {b_nmse:>4d}      {b_dir:>4d} {marker}")
        if len(diffs) > 15:
            print(f"    ... and {len(diffs)-15} more")

    # Estimate using single-block approximation
    est_nmse_n = np.mean(nmse_anc)
    est_nmse_d = np.mean(nmse_anc)
    est_cos2_n = np.mean(cos2_anc)
    est_cos2_d = np.mean(cos2_anc)
    for bname in block_names:
        b_n = pol_nmse.get(bname, 16)
        b_d = pol_dir.get(bname, 16)
        est_nmse_n += Omega_nmse.get((bname, b_n), 0)
        est_nmse_d += Omega_nmse.get((bname, b_d), 0)
        est_cos2_n -= Omega_dir.get((bname, b_n), 0)
        est_cos2_d -= Omega_dir.get((bname, b_d), 0)

    nmse_db_n = 10 * np.log10(est_nmse_n + 1e-15)
    nmse_db_d = 10 * np.log10(est_nmse_d + 1e-15)
    print(f"  Est NMSE:  nmse-ILP={nmse_db_n:.2f}dB  dir-ILP={nmse_db_d:.2f}dB  (diff={nmse_db_d - nmse_db_n:+.3f}dB)")
    print(f"  Est cos2:  nmse-ILP={est_cos2_n:.6f}  dir-ILP={est_cos2_d:.6f}  (diff={est_cos2_d - est_cos2_n:+.6f})")

# === Pattern analysis at 90% saving ===
print("\n" + "=" * 80)
print("  PATTERN: Which blocks flip at 90% saving?")
print("=" * 80)

budget = 0.10
pol_nmse = solve_ilp(Omega_nmse, budget, "nmse_90")
pol_dir = solve_ilp(Omega_dir, budget, "dir_90")

upgraded = [(k, pol_nmse[k], pol_dir[k]) for k in block_names
            if pol_dir.get(k, 16) > pol_nmse.get(k, 16)]
downgraded = [(k, pol_nmse[k], pol_dir[k]) for k in block_names
              if pol_dir.get(k, 16) < pol_nmse.get(k, 16)]

print(f"\nBlocks UPGRADED by dir-ILP (more bits for direction):")
for bname, bn, bd in upgraded:
    ratio = Omega_dir.get((bname, 2), 0) / (Omega_nmse.get((bname, 2), 0) + 1e-12)
    print(f"  {bname:35s}  {bn} -> {bd}  (dir/nmse ratio@2bit: {ratio:.4f})")

print(f"\nBlocks DOWNGRADED by dir-ILP (fewer bits, sacrificed):")
for bname, bn, bd in downgraded:
    ratio = Omega_dir.get((bname, 2), 0) / (Omega_nmse.get((bname, 2), 0) + 1e-12)
    print(f"  {bname:35s}  {bn} -> {bd}  (dir/nmse ratio@2bit: {ratio:.4f})")

# === Importance vector comparison ===
imp_nmse = np.array([Omega_nmse.get((bn, 2), 0) for bn in block_names])
imp_dir = np.array([Omega_dir.get((bn, 2), 0) for bn in block_names])
rho, _ = spearmanr(imp_nmse, imp_dir)
cos_sim = np.dot(imp_nmse, imp_dir) / (np.linalg.norm(imp_nmse) * np.linalg.norm(imp_dir) + 1e-12)
print(f"\nImportance vector (bit=2):")
print(f"  Cosine(Omega_nmse, Omega_dir) = {cos_sim:.4f}")
print(f"  Spearman = {rho:.4f}")

# Without stem.0 (which dominates everything)
mask_no_stem = [i for i, bn in enumerate(block_names) if bn != "stem.0"]
imp_nmse_ns = imp_nmse[mask_no_stem]
imp_dir_ns = imp_dir[mask_no_stem]
rho_ns, _ = spearmanr(imp_nmse_ns, imp_dir_ns)
cos_ns = np.dot(imp_nmse_ns, imp_dir_ns) / (np.linalg.norm(imp_nmse_ns) * np.linalg.norm(imp_dir_ns) + 1e-12)
print(f"\nWithout stem.0:")
print(f"  Cosine(Omega_nmse, Omega_dir) = {cos_ns:.4f}")
print(f"  Spearman = {rho_ns:.4f}")
