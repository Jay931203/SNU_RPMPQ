import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from tqdm import tqdm
import scipy.io as sio
import time
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import ast
import pulp # ILP optimization library

# TF32 Enable
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.append(project_root)

# --- Output directories ---
RESULTS_CSV  = os.path.join(project_root, "results", "csv")
RESULTS_PLOT = os.path.join(project_root, "results", "plots")
FIGURES_DIR  = os.path.join(project_root, "..", "figures")
os.makedirs(RESULTS_CSV, exist_ok=True)
os.makedirs(RESULTS_PLOT, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Imports ---
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

import os
import numpy as np
import torch
import pandas as pd
import torch.utils
import re

class RPMPQOnlineController:
    """RP-MPQ online stage: per-sample policy selection (Section V)"""
    def __init__(self, calibration_data_path, policy_lut_path, lambda_budget=1.0, num_bins=20):
        """
        [RP-MPQ Offline] Violation Cost LUT Construction (Section V-E)
        Build per-bin (mu, sigma) table from calibration data.
        """
        self.lambda_budget = lambda_budget

        if not os.path.exists(calibration_data_path):
            raise FileNotFoundError(f"[ERROR] Calibration data not found: {calibration_data_path}")

        df = pd.read_csv(calibration_data_path)
        self.lut_df = pd.read_csv(policy_lut_path)

        if isinstance(self.lut_df['Policy'].iloc[0], str):
            self.lut_df['Policy'] = self.lut_df['Policy'].apply(ast.literal_eval)

        # Identify policy IDs (sorted by saving ascending)
        df['Config_k'] = df['B'].round(4)
        self.policy_ids = sorted(df['Config_k'].unique())

        # Binning & profiling
        self.s_grids = np.linspace(0.0, 1.0, num_bins + 1)
        self.violation_cost_lut = {}

        print(f"   [INFO] RP-MPQ Online Controller: Building Violation Cost LUT (lambda={lambda_budget}, Bins={num_bins})...")

        for k in self.policy_ids:
            sub_df = df[df['Config_k'] == k]
            sub_df = sub_df.copy()
            sub_df['Bin'] = pd.cut(sub_df['S'], bins=self.s_grids, labels=False, include_lowest=True)

            stats = []
            for bin_idx in range(num_bins):
                bin_data = sub_df[sub_df['Bin'] == bin_idx]
                s_center = (self.s_grids[bin_idx] + self.s_grids[bin_idx+1]) / 2.0

                if len(bin_data) > 0:
                    mu = bin_data['NMSE_linear'].mean()
                    sigma = bin_data['NMSE_linear'].std() if len(bin_data) > 1 else 0.0
                else:
                    mu, sigma = np.nan, np.nan

                stats.append({'S': s_center, 'mu': mu, 'sigma': sigma})

            stat_df = pd.DataFrame(stats)
            stat_df = stat_df.interpolate(method='linear', limit_direction='both')
            stat_df = stat_df.fillna(method='bfill').fillna(method='ffill')
            self.violation_cost_lut[k] = stat_df

    def estimate_violation_cost(self, s_curr, policy_id):
        """
        [RP-MPQ Online] Per-Sample Violation Cost Estimation (Eq.33)
        V_bar(pi; xi_t) = mu + lambda * sigma
        """
        profile = self.violation_cost_lut[policy_id]
        s = np.clip(s_curr, profile['S'].min(), profile['S'].max())

        mu_hat = np.interp(s, profile['S'], profile['mu'])
        sigma_hat = np.interp(s, profile['S'], profile['sigma'])

        violation_cost_surrogate = mu_hat + self.lambda_budget * sigma_hat
        return violation_cost_surrogate

    def select_online_policy(self, s_curr, target_violation_threshold):
        """
        [RP-MPQ Online] Policy Selection under Budget (Eq.34)
        Greedy: scan from highest saving to lowest, pick first policy whose cost <= threshold
        """
        best_policy_id = self.policy_ids[0]

        for k in reversed(self.policy_ids):
            violation_cost_surrogate = self.estimate_violation_cost(s_curr, k)
            if violation_cost_surrogate <= target_violation_threshold:
                best_policy_id = k
                break

        row = self.lut_df.iloc[(self.lut_df['Actual_Saving'] - best_policy_id).abs().argsort()[:1]]
        policy = row['Policy'].values[0]

        return policy, best_policy_id


def compute_hoyer_sparsity_bins(dataset, device):
    """Compute Hoyer's sparsity measure (Eq.34) and return tercile bins + continuous values"""
    all_sparsity = []
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    for batch in loader:
        l1 = torch.norm(batch, p=1, dim=(1,2,3))
        l2 = torch.norm(batch, p=2, dim=(1,2,3))
        sparsity = (l1 / (l2 + 1e-8)).cpu().numpy()
        all_sparsity.extend(sparsity)

    all_sparsity = np.array(all_sparsity)

    bins = {
        "High": np.where(all_sparsity > np.percentile(all_sparsity, 66))[0].tolist(),
        "Mid":  np.where((all_sparsity <= np.percentile(all_sparsity, 66)) &
                         (all_sparsity > np.percentile(all_sparsity, 33)))[0].tolist(),
        "Low":  np.where(all_sparsity <= np.percentile(all_sparsity, 33))[0].tolist()
    }

    return bins, all_sparsity

def verify_fc_quantization(model, lut_path, device):
    import pandas as pd
    import ast
    import torch
    import numpy as np

    print("\n" + "="*60)
    print("[INFO] FC Layer Verification")
    print("="*60)

    # 1. Load policy
    df = pd.read_csv(lut_path)
    policy_str = df.iloc[0]['Policy']
    policy = ast.literal_eval(policy_str) if isinstance(policy_str, str) else policy_str

    # 2. Get FC layer from model
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Find fc layer based on model structure
    fc_layer = None
    if hasattr(real_model, 'encoder'):
        if hasattr(real_model.encoder, 'fc'): fc_layer = real_model.encoder.fc
        elif hasattr(real_model.encoder, 'layers'):
             for n, m in real_model.encoder.named_modules():
                 if 'fc' in n: fc_layer = m; break
    
    if fc_layer is None:
        print("[ERROR] Could not find 'fc' layer in model!")
        return

    # 3. Clone pre-quantization weights
    w_before = fc_layer.weight.data.clone()
    print(f"[INFO] FC Layer Shape: {w_before.shape}")

    # 4. Apply precision policy
    print("[INFO] Applying precision policy...")
    apply_precision_policy(model, policy, device)

    # 5. Compare weights after quantization
    w_after = fc_layer.weight.data
    
    diff = torch.norm(w_before - w_after).item()

    fc_keys = [k for k in policy.keys() if 'fc_part' in k]
    print(f"[INFO] Policy contains {len(fc_keys)} split parts for FC.")
    print(f"[INFO] Weight Difference (L2): {diff:.6f}")

    if diff == 0:
        print("[FAIL] FC layer weights unchanged (still FP32)")
    else:
        print("[OK] FC layer weights changed (split quantization applied)")

def run_exp3_sparsity_frontier(model, loader, lut_path, device, norm_params, args):
    """
    [Exp 3] NMSE & Stability Analysis (Final Version)
    - Fix: Added model weight reset per iteration to prevent cumulative quantization error.
    - Fix: 'Avg NMSE' now uses re-calculated values for consistency.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset

    # Cache setup
    cache_path = os.path.join(RESULTS_CSV, f"exp3_nmse_cache_{args.encoder}.npz")
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    min_val, range_val = norm_params
    
    # Load LUT
    df_lut = pd.read_csv(lut_path)
    if isinstance(df_lut['Policy'].iloc[0], str):
        df_lut['Policy'] = df_lut['Policy'].apply(ast.literal_eval)

    # 1. Load cached data or compute
    if os.path.exists(cache_path):
        print(f"\n[INFO] Loading cached NMSE data: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        fp32_nmses_linear = data['fp32_nmses']
        all_raw_nmses = data['raw_nmses'] 
        savings = data['savings']
        base_kl_nmses = data['kl_nmses']
    else:
        print(f"\n[INFO] No cache found. Calculating NMSEs...")
        
        bins, _ = compute_hoyer_sparsity_bins(loader.dataset, device)
        sub_indices = []
        for g in ["High", "Mid", "Low"]: sub_indices.extend(bins[g][:200]) # total 600 samples
        analysis_loader = DataLoader(Subset(loader.dataset, sub_indices), batch_size=args.test_batch_size, shuffle=False)
        
        # (1) FP32 Baseline NMSE
        real_model.load_state_dict(original_state)
        real_model.eval()
        fp32_nmses_linear = []
        with torch.no_grad():
            for batch in analysis_loader:
                d = batch.to(device)
                z = real_model.encoder(d)
                if args.aq > 0: z = quantize_feedback_torch(z, args.aq)
                x_hat = real_model.decoder(z)
                
                orig_c = (d * range_val) + min_val - 0.5
                rec_c = (x_hat * range_val) + min_val - 0.5
                nmse = torch.sum((orig_c - rec_c)**2, dim=(1,2,3)) / (torch.sum(orig_c**2, dim=(1,2,3)) + 1e-8)
                fp32_nmses_linear.extend(nmse.cpu().numpy())
        
        # (2) LUT Policy Scan
        all_raw_nmses = []
        savings = []
        base_kl_nmses = []

        for _, row in tqdm(df_lut.iterrows(), total=len(df_lut), desc="NMSE Analysis"):
            # Reset to FP32 weights before each policy evaluation
            real_model.load_state_dict(original_state)
            
            apply_precision_policy(model, row['Policy'], device)
            real_model.eval()
            
            policy_nmses = []
            with torch.no_grad():
                for batch in analysis_loader:
                    d = batch.to(device)
                    z = real_model.encoder(d)
                    if args.aq > 0: z = quantize_feedback_torch(z, args.aq)
                    x_hat = real_model.decoder(z)
                    
                    orig_c = (d * range_val) + min_val - 0.5
                    rec_c = (x_hat * range_val) + min_val - 0.5
                    nmse = torch.sum((orig_c - rec_c)**2, dim=(1,2,3)) / (torch.sum(orig_c**2, dim=(1,2,3)) + 1e-8)
                    policy_nmses.extend(nmse.cpu().numpy())
            
            all_raw_nmses.append(policy_nmses)
            savings.append(row['Actual_Saving'])
            base_kl_nmses.append(row['NMSE_KL'])

        # Restore model weights after loop
        real_model.load_state_dict(original_state)

        np.savez(cache_path, fp32_nmses=np.array(fp32_nmses_linear), 
                 raw_nmses=np.array(all_raw_nmses), savings=np.array(savings), kl_nmses=np.array(base_kl_nmses))

    # 2. Statistical analysis
    final_results = []
    fp32_avg = np.mean(fp32_nmses_linear)
    all_raw_nmses = np.array(all_raw_nmses)
    savings = np.array(savings)
    
    # Pareto Smoothing
    for s_idx in range(all_raw_nmses.shape[1]):
        curve = all_raw_nmses[:, s_idx]
        best_val = float('inf')
        smoothed = []
        for val in reversed(curve):
            best_val = min(best_val, val)
            smoothed.append(best_val)
        all_raw_nmses[:, s_idx] = smoothed[::-1]

    TARGET_CDF = 90.0
    closest_idx = np.abs(savings - TARGET_CDF).argmin()
    target_cdf_data = []

    for i, saving in enumerate(savings):
        s_nmses = all_raw_nmses[i]
        s_nmses_db = 10 * np.log10(s_nmses + 1e-15)
        
        # Use computed average (not CSV value) for consistency
        calculated_avg_nmse = 10 * np.log10(np.mean(s_nmses) + 1e-15)
        
        ratios = fp32_avg / (s_nmses + 1e-12)
        outage = np.mean(ratios < 0.9)
        
        if i == closest_idx:
            target_cdf_data = s_nmses_db

        final_results.append({
            "Actual_Saving": saving,
            "Base_NMSE": calculated_avg_nmse, # Use computed value instead of CSV
            "NMSE_Sigma": np.std(s_nmses_db),
            "High_S": 10 * np.log10(np.mean(s_nmses[:200]) + 1e-15),
            "Mid_S":  10 * np.log10(np.mean(s_nmses[200:400]) + 1e-15),
            "Low_S":  10 * np.log10(np.mean(s_nmses[400:]) + 1e-15),
            "Outage_Prob": outage
        })

    res_df = pd.DataFrame(final_results)

    # 3. Plot results
    # [1] NMSE Performance
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['Actual_Saving'], res_df['Base_NMSE'], 'k--', label='Avg NMSE')
    plt.plot(res_df['Actual_Saving'], res_df['High_S'], 'r-o', label='High Sparsity')
    plt.plot(res_df['Actual_Saving'], res_df['Low_S'], 'b-^', label='Low Sparsity')
    plt.xlabel("Encoder BOPs Saving (%)"); plt.ylabel("NMSE (dB)"); plt.legend(); plt.grid(True, alpha=0.5)
    plt.title(f"Sparsity-Stratified NMSE under MP Policies ({args.encoder})")
    plt.savefig(os.path.join(RESULTS_PLOT, f"exp3_nmse_sparsity_{args.encoder}.png"))

    # [2] Stability
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['Actual_Saving'], res_df['NMSE_Sigma'], 'm-D')
    plt.xlabel("Encoder BOPs Saving (%)"); plt.ylabel("Sigma (dB)"); plt.grid(True, alpha=0.5)
    plt.title(f"NMSE Stability ({args.encoder})")
    plt.savefig(os.path.join(RESULTS_PLOT, f"exp3_nmse_sigma_{args.encoder}.png"))

    # [3] CDF
    if len(target_cdf_data) > 0:
        plt.figure(figsize=(10, 6))
        sorted_data = np.sort(target_cdf_data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, 'k-', linewidth=2, label=f'CDF @ {res_df.loc[closest_idx, "Actual_Saving"]:.1f}%')
        plt.axvline(x=res_df.loc[closest_idx, "Base_NMSE"], color='r', linestyle='--', label='Avg NMSE')
        plt.xlabel("NMSE (dB)"); plt.ylabel("CDF"); plt.grid(True, alpha=0.5); plt.legend(loc='lower right')
        plt.title(f"NMSE Distribution CDF ({args.encoder})")
        plt.savefig(os.path.join(RESULTS_PLOT, f"exp3_nmse_cdf_{args.encoder}.png"))

    # [4] NMSE Outage
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['Actual_Saving'], res_df['Outage_Prob'], 'r-X', label='NMSE Outage (< 90% of FP32)')
    plt.axhline(0.1, color='k', linestyle=':', label='10% Limit')
    plt.ylim(-0.05, 1.05); plt.xlabel("Encoder BOPs Saving (%)"); plt.ylabel("Outage Probability"); plt.grid(True, alpha=0.5); plt.legend()
    plt.title(f"NMSE-Based Outage Probability ({args.encoder})")
    plt.savefig(os.path.join(RESULTS_PLOT, f"exp3_nmse_outage_{args.encoder}.png"))
    
    print(f"[INFO] Exp 3: NMSE analysis complete (4 graphs saved).")
    return res_df
    
    
    
def run_exp3_5_rate_outage_analysis(model, loader, lut_path, device, norm_params, args):
    """
    [Exp 3.5] Generalized Rigorous Sum-Rate & Multi-Threshold Outage Analysis
    - Generalization: Dynamically handles any number of SNR levels and Outage Thresholds.
    - Layout: Adjusts figure size and subplot grid automatically.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pandas as pd
    import numpy as np
    import os
    import torch
    import ast
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset 

    print("\n" + "="*80)
    print("[INFO] Exp 3.5: Rate-Based Outage under Identical Budgets (Section VI-D)")
    print("="*80)

    # ---------------------------------------------------------
    # [Helper Function] Pareto Frontier Smoothing
    # ---------------------------------------------------------
# ---------------------------------------------------------
    # [Helper Function] Pareto Frontier Smoothing (revised)
    # ---------------------------------------------------------
    def get_pareto_frontier_max(bops_list, metric_list):
        # Handle pandas Series input: check .empty or convert to list
        if isinstance(bops_list, pd.Series):
            if bops_list.empty: return [], []
            bops_list = bops_list.tolist()
        elif not bops_list: # list case
            return [], []

        if isinstance(metric_list, pd.Series):
            metric_list = metric_list.tolist()
            
        data = sorted(zip(bops_list, metric_list), key=lambda x: x[0])
        pareto_x, pareto_y = [], []
        current_max = -np.inf
        for bops, val in reversed(data):
            if val >= current_max:
                pareto_x.append(bops)
                pareto_y.append(val)
                current_max = val
        return pareto_x[::-1], pareto_y[::-1]

    def get_pareto_frontier_min(bops_list, metric_list):
        # Same Series/list handling as get_pareto_frontier_max
        if isinstance(bops_list, pd.Series):
            if bops_list.empty: return [], []
            bops_list = bops_list.tolist()
        elif not bops_list: 
            return [], []

        if isinstance(metric_list, pd.Series):
            metric_list = metric_list.tolist()
            
        data = sorted(zip(bops_list, metric_list), key=lambda x: x[0])
        pareto_x, pareto_y = [], []
        current_min = np.inf
        for bops, val in reversed(data):
            if val <= current_min:
                pareto_x.append(bops)
                pareto_y.append(val)
                current_min = val
        return pareto_x[::-1], pareto_y[::-1]

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------
    # SNR levels and outage thresholds
    snr_list = [10, 20, 30] 
    outage_thresholds = [0.99, 0.98, 0.95] # 99%, 98%, 95%
    
    min_val, range_val = norm_params
    real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    cache_path = os.path.join(RESULTS_CSV, f"exp3_5_rate_baseline_{args.encoder}_final.npz")

    # Balanced Sampling
    try:
        bins, _ = compute_hoyer_sparsity_bins(loader.dataset, device)
        sub_indices = []
        for g in ["High", "Mid", "Low"]: 
            if g in bins: sub_indices.extend(bins[g][:100])
        if not sub_indices: raise ValueError
    except:
        print("[WARN] Balanced sampling failed or bins empty. Using first 300 samples.")
        sub_indices = list(range(min(300, len(loader.dataset))))
        
    eval_loader = DataLoader(Subset(loader.dataset, sub_indices), batch_size=args.test_batch_size, shuffle=False)
    print(f"[INFO] Evaluated on {len(sub_indices)} balanced samples")

    # ---------------------------------------------------------
    # Phase 0 & 1: Baseline Calculation (Perfect CSI & FP32)
    # ---------------------------------------------------------
    perfect_rates = {snr: [] for snr in snr_list}
    fp32_rates = {}
    
    # Load cache
    if os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            fp32_rates = loaded['fp32_rates'].item()
            # Recompute if cached SNR keys differ from current snr_list
            if not all(k in fp32_rates for k in snr_list): raise ValueError
            print(f"[INFO] Loading cached baseline rates.")
        except:
            fp32_rates = {}

    # Compute if not cached
    if not fp32_rates or not perfect_rates[snr_list[0]]:
        print(f"[INFO] Calculating baselines (Perfect CSI & FP32)...")
        real_model.load_state_dict(original_state)
        real_model.eval()
        
        fp32_rates = {snr: [] for snr in snr_list}
        perfect_rates = {snr: [] for snr in snr_list}
        
        with torch.no_grad():
            for d in eval_loader:
                d = d.to(device)
                x_hat = real_model.decoder(real_model.encoder(d))
                
                h_true = (d * range_val) + min_val - 0.5
                h_hat = (x_hat * range_val) + min_val - 0.5
                
                for snr in snr_list:
                    r_perf = calculate_su_miso_rate_mrt(h_true, h_true, snr, device)
                    perfect_rates[snr].extend(r_perf.cpu().numpy())
                    
                    r_fp32 = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
                    fp32_rates[snr].extend(r_fp32.cpu().numpy())
        np.savez(cache_path, fp32_rates=fp32_rates)
    
    # Ensure perfect rates are in memory (when loaded from cache)
    if not perfect_rates[snr_list[0]]:
         with torch.no_grad():
            for d in eval_loader:
                d = d.to(device)
                h_true = (d * range_val) + min_val - 0.5
                for snr in snr_list:
                    r_perf = calculate_su_miso_rate_mrt(h_true, h_true, snr, device)
                    perfect_rates[snr].extend(r_perf.cpu().numpy())

    # ---------------------------------------------------------
    # Phase 2: Quantized Policy Rate Calculation
    # ---------------------------------------------------------
    df_lut = pd.read_csv(lut_path)
    if isinstance(df_lut['Policy'].iloc[0], str):
        df_lut['Policy'] = df_lut['Policy'].apply(ast.literal_eval)
        
    analysis_results = []
    print(f"[INFO] 2. Scanning {len(df_lut)} policies...")
    
    for _, row in tqdm(df_lut.iterrows(), total=len(df_lut), desc="Rate Sweep"):
        real_model.load_state_dict(original_state)
        apply_precision_policy(model, row['Policy'], device)
        real_model.eval()
        
        current_rates = {snr: [] for snr in snr_list}
        with torch.no_grad():
            for d in eval_loader:
                d = d.to(device)
                z = real_model.encoder(d)
                if args.aq > 0: z = quantize_feedback_torch(z, args.aq)
                x_hat = real_model.decoder(z)
                
                h_true = (d * range_val) + min_val - 0.5
                h_hat = (x_hat * range_val) + min_val - 0.5
                
                for snr in snr_list:
                    r_tensor = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
                    current_rates[snr].extend(r_tensor.cpu().numpy())
        
        res_entry = {"Actual_Saving": row['Actual_Saving']}
        
        for snr in snr_list:
            q_r = np.array(current_rates[snr])
            p_r = np.array(perfect_rates[snr])
            
            # Sum Rate
            res_entry[f"Rate_{snr}dB"] = np.mean(q_r)
            
            # Multi-Threshold Outage
            for th in outage_thresholds:
                is_outage = q_r < (p_r * th)
                res_entry[f"Outage_{snr}dB_{int(th*100)}"] = np.mean(is_outage)
            
        analysis_results.append(res_entry)
        
    res_df = pd.DataFrame(analysis_results).sort_values(by="Actual_Saving")
    real_model.load_state_dict(original_state)

    # ---------------------------------------------------------
    # Phase 3: Plotting Sum-Rate (Dynamic Colors)
    # ---------------------------------------------------------
    print(f"[INFO] 3. Plotting transmission rate...")
    plt.figure(figsize=(10, 7))
    
    # Generate colormap scaled to number of SNR values
    cmap = plt.get_cmap('tab10') 
    snr_colors = {snr: cmap(i % 10) for i, snr in enumerate(snr_list)}
    
    for snr in snr_list:
        c = snr_colors[snr]
        perf_val = np.mean(perfect_rates[snr])
        base_val = np.mean(fp32_rates[snr])

        # Reference Lines
        plt.axhline(perf_val, color='k', linestyle='--', linewidth=1.0, alpha=0.4) 
        plt.axhline(base_val, color=c, linestyle=':', linewidth=1.5, alpha=0.7)    
        
        # RP-MPQ
        smooth_x, smooth_y = get_pareto_frontier_max(res_df['Actual_Saving'], res_df[f"Rate_{snr}dB"])
        plt.plot(smooth_x, smooth_y,
                 marker='o', linestyle='-', linewidth=2.0, markersize=4,
                 color=c, label=f'RP-MPQ ({snr}dB)')
        
        # Legend entries (drawn only for first SNR)
        if snr == snr_list[0]:
             plt.plot([], [], 'k--', label='Perfect CSI')
             plt.plot([], [], 'k:', label='FP32 Baseline')

    plt.xlabel("Encoder BOPs Saving (%)", fontsize=12)
    plt.ylabel("Sum Rate (bps/Hz)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize='small', loc='lower left', ncol=min(3, len(snr_list)))
    plt.title(f"Transmission Rate vs UE Encoder BOPs Saving - {args.encoder}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOT, f"exp3_5_rate_final_{args.encoder}.png"), dpi=300)
    print(f"   Saved Rate Plot: exp3_5_rate_final_{args.encoder}.png")

    # ---------------------------------------------------------
    # Phase 4: Plotting Outage (Dynamic Subplots)
    # ---------------------------------------------------------
    print(f"[INFO] 4. Plotting outage (dynamic subplots)...")
    
    # Create one subplot column per SNR level
    n_plots = len(snr_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=True, constrained_layout=True)
    
    # Ensure axes is iterable even with a single subplot
    if n_plots == 1: axes = [axes]
    
    # Generate threshold-dependent styles and colors (Reds gradient for severity)
    th_cmap = plt.get_cmap('Reds')
    # Use intensity range [0.4, 1.0] for visibility
    th_colors_list = [th_cmap(x) for x in np.linspace(0.4, 1.0, len(outage_thresholds))]
    th_styles_list = ['-', '--', '-.', ':'] * (len(outage_thresholds)//4 + 1) # cyclic styles

    # Sort thresholds in descending order (strictest first)
    sorted_ths = sorted(outage_thresholds, reverse=True)

    for i, (ax, snr) in enumerate(zip(axes, snr_list)):
        c = snr_colors[snr]
        
        for j, th in enumerate(sorted_ths):
            th_int = int(th * 100)
            col_name = f"Outage_{snr}dB_{th_int}"
            
            if col_name not in res_df.columns: continue

            smooth_x, smooth_y = get_pareto_frontier_min(res_df['Actual_Saving'], res_df[col_name])
            
            # Dynamic style assignment
            line_color = th_colors_list[j] 
            line_style = th_styles_list[j]
            
            ax.plot(smooth_x, smooth_y, 
                    color=line_color, 
                    linestyle=line_style, 
                    linewidth=2.0, 
                    marker='o', markersize=4,
                    label=f'Limit {th_int}%')

        ax.set_title(f"SNR {snr}dB", fontsize=14, fontweight='bold', color=c)
        ax.set_xlabel("Encoder BOPs Saving (%)", fontsize=12)
        ax.grid(True, alpha=0.3, which='both')

        if i == 0:
            ax.set_ylabel("Outage Probability", fontsize=12)
            ax.legend(loc='upper left', fontsize='small', title="Perf. Threshold")

    plt.suptitle(f"Rate-Based Outage under Identical Budgets ({args.encoder})", fontsize=16)
    
    # Build save path
    plot_path = os.path.join(RESULTS_PLOT, f"exp3_5_outage_subplots_{args.encoder}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"   Saved Subplot: {plot_path}")

    return res_df
    
def run_exp4_rp_mpq_online(net, test_loader, lut_path, device, baseline_df, args):
    """
    [RP-MPQ Online] Reliability-Aware Policy Selection (Section VI-E)
    1. Granularity: 0.1% saving steps for smooth visualization.
    2. Optimization: Uses linear MSE for Lagrangian cost (heavy outlier penalty).
    3. Sampling: Balanced 300 samples (consistent with Exp 3.5).
    """
    import pandas as pd
    import numpy as np
    import torch
    import ast
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset

    print("\n" + "="*80)
    print("[INFO] RP-MPQ Online Simulation (High-Resolution)")
    print("="*80)

    # ---------------------------------------------------------
    # [Step 0] Balanced Sampling
    # ---------------------------------------------------------
    print("[INFO] Preparing balanced samples...")
    try:
        bins, _ = compute_hoyer_sparsity_bins(test_loader.dataset, device)
        sub_indices = []
        for g in ["High", "Mid", "Low"]: 
            if g in bins: sub_indices.extend(bins[g][:100])
        if not sub_indices: raise ValueError("No indices found")
        eval_loader = DataLoader(Subset(test_loader.dataset, sub_indices), 
                                 batch_size=args.test_batch_size, shuffle=False)
        print(f"   -> Evaluated on {len(sub_indices)} Balanced Samples.")
    except:
        print(f"[WARN] Balanced sampling failed. Using full test loader.")
        eval_loader = test_loader

    # ---------------------------------------------------------
    # Phase 1: Data Collection (Linear MSE + Rates)
    # ---------------------------------------------------------
    print("[INFO] Phase 1: Collecting raw data (linear MSE)...")
    
    df_lut = pd.read_csv(lut_path)
    if isinstance(df_lut['Policy'].iloc[0], str):
        df_lut['Policy'] = df_lut['Policy'].apply(ast.literal_eval)
    
    real_model = net.module if isinstance(net, torch.nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    norm_params = test_loader.dataset.normalization_params if hasattr(test_loader.dataset, 'normalization_params') else (0, 1)
    min_val, range_val = norm_params

    snr_list = [10, 20, 30]
    raw_data = [] 
    
    # 1. Perfect Rates Calculation (for Outage Check)
    perfect_rates_map = {}
    sample_idx = 0
    with torch.no_grad():
        for d in eval_loader:
            d = d.to(device)
            h_true = (d * range_val) + min_val - 0.5
            rates_batch = {snr: calculate_su_miso_rate_mrt(h_true, h_true, snr, device).cpu().numpy() for snr in snr_list}
            for i in range(d.shape[0]):
                perfect_rates_map[sample_idx] = {snr: rates_batch[snr][i] for snr in snr_list}
                sample_idx += 1
                
    # 2. Policy Inference
    for idx, row in tqdm(df_lut.iterrows(), total=len(df_lut), desc="Collecting Data"):
        policy = row['Policy']
        policy_id = idx
        saving_pct = row['Actual_Saving']
        bops_cost = (100 - saving_pct) / 100.0 
        
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, policy, device) 
        real_model.eval()
        
        sample_idx = 0
        with torch.no_grad():
            for d in eval_loader:
                d = d.to(device)
                z = real_model.encoder(d)
                if args.aq > 0: z = quantize_feedback_torch(z, args.aq)
                x_hat = real_model.decoder(z)
                
                h_true = (d * range_val) + min_val - 0.5
                h_hat = (x_hat * range_val) + min_val - 0.5
                
                # Linear-scale NMSE computation (used for optimization)
                # MSE = ||h - h^||^2 / ||h||^2
                error = torch.sum((h_true - h_hat)**2, dim=[1, 2, 3])
                power = torch.sum(h_true**2, dim=[1, 2, 3])
                mse_linear = (error / (power + 1e-9)).cpu().numpy() # Linear Scale
                
                # NMSE (dB) for Logging
                nmse_db = 10 * np.log10(mse_linear + 1e-15)

                # Rates
                rates_batch = {snr: calculate_su_miso_rate_mrt(h_true, h_hat, snr, device).cpu().numpy() for snr in snr_list}
                
                batch_size = len(mse_linear)
                for i in range(batch_size):
                    entry = {
                        'sample_id': sample_idx,
                        'config_id': policy_id,
                        'bops': bops_cost,
                        'saving': saving_pct,
                        'mse_linear': mse_linear[i], # optimization variable
                        'nmse_db': nmse_db[i]        # for result reporting
                    }
                    for snr in snr_list:
                        entry[f'rate_{snr}'] = rates_batch[snr][i]
                        entry[f'perf_{snr}'] = perfect_rates_map[sample_idx][snr]
                    
                    raw_data.append(entry)
                    sample_idx += 1
                    
    rate_df = pd.DataFrame(raw_data)
    
# ---------------------------------------------------------
    # [Final Generalized] Phase 2: Optimization (Margin-Normalized)
    # ---------------------------------------------------------
    
def run_exp4_rp_mpq_online(net, test_loader, lut_path, device, baseline_df, args):
    """
    [RP-MPQ Online] Multi-Target Reliability-Aware Policy Selection (Section VI-E)
    1. Optimization: Reliability-weighted safe optimization (violation cost, Eq.29)
    2. Multi-Target: Runs for reliability targets gamma in {0.99, 0.98, 0.95} independently.
    3. Visualization: Distinct curves for each gamma target.
    """
    import pandas as pd
    import numpy as np
    import torch
    import ast
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset

    print("\n" + "="*80)
    print("[INFO] RP-MPQ Online Simulation (Multi-Target: gamma=0.99, 0.98, 0.95)")
    print("="*80)

    # ---------------------------------------------------------
    # [Step 0] Balanced Sampling
    # ---------------------------------------------------------
    print("[INFO] Preparing balanced samples...")
    try:
        bins, _ = compute_hoyer_sparsity_bins(test_loader.dataset, device)
        sub_indices = []
        for g in ["High", "Mid", "Low"]: 
            if g in bins: sub_indices.extend(bins[g][:100])
        if not sub_indices: raise ValueError("No indices found")
        eval_loader = DataLoader(Subset(test_loader.dataset, sub_indices), 
                                 batch_size=args.test_batch_size, shuffle=False)
        print(f"   -> Evaluated on {len(sub_indices)} Balanced Samples.")
    except:
        print(f"[WARN] Balanced sampling failed. Using full test loader.")
        eval_loader = test_loader

    # ---------------------------------------------------------
    # Phase 1: Data Collection (Linear MSE + Rates)
    # ---------------------------------------------------------
    print("[INFO] Phase 1: Collecting raw data (linear MSE)...")
    
    df_lut = pd.read_csv(lut_path)
    if isinstance(df_lut['Policy'].iloc[0], str):
        df_lut['Policy'] = df_lut['Policy'].apply(ast.literal_eval)
    
    real_model = net.module if isinstance(net, torch.nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    norm_params = test_loader.dataset.normalization_params if hasattr(test_loader.dataset, 'normalization_params') else (0, 1)
    min_val, range_val = norm_params

    snr_list = [10, 20, 30]
    raw_data = [] 
    
    # 1. Perfect Rates Calculation
    perfect_rates_map = {}
    sample_idx = 0
    with torch.no_grad():
        for d in eval_loader:
            d = d.to(device)
            h_true = (d * range_val) + min_val - 0.5
            rates_batch = {snr: calculate_su_miso_rate_mrt(h_true, h_true, snr, device).cpu().numpy() for snr in snr_list}
            for i in range(d.shape[0]):
                perfect_rates_map[sample_idx] = {snr: rates_batch[snr][i] for snr in snr_list}
                sample_idx += 1
                
    # 2. Policy Inference
    for idx, row in tqdm(df_lut.iterrows(), total=len(df_lut), desc="Collecting Data"):
        policy = row['Policy']
        policy_id = idx
        saving_pct = row['Actual_Saving']
        bops_cost = (100 - saving_pct) / 100.0 
        
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, policy, device) 
        real_model.eval()
        
        sample_idx = 0
        with torch.no_grad():
            for d in eval_loader:
                d = d.to(device)
                z = real_model.encoder(d)
                if args.aq > 0: z = quantize_feedback_torch(z, args.aq)
                x_hat = real_model.decoder(z)
                
                h_true = (d * range_val) + min_val - 0.5
                h_hat = (x_hat * range_val) + min_val - 0.5
                
                # Linear MSE
                error = torch.sum((h_true - h_hat)**2, dim=[1, 2, 3])
                power = torch.sum(h_true**2, dim=[1, 2, 3])
                mse_linear = (error / (power + 1e-9)).cpu().numpy()
                
                nmse_db = 10 * np.log10(mse_linear + 1e-15)

                # Rates
                rates_batch = {snr: calculate_su_miso_rate_mrt(h_true, h_hat, snr, device).cpu().numpy() for snr in snr_list}
                
                batch_size = len(mse_linear)
                for i in range(batch_size):
                    entry = {
                        'sample_id': sample_idx,
                        'config_id': policy_id,
                        'bops': bops_cost,
                        'saving': saving_pct,
                        'mse_linear': mse_linear[i],
                        'nmse_db': nmse_db[i]
                    }
                    for snr in snr_list:
                        entry[f'rate_{snr}'] = rates_batch[snr][i]
                        entry[f'perf_{snr}'] = perfect_rates_map[sample_idx][snr]
                    
                    raw_data.append(entry)
                    sample_idx += 1
                    
    rate_df = pd.DataFrame(raw_data)
    
    # ---------------------------------------------------------
    # [Phase 2] Optimization Function (Safe Version v2)
    # ---------------------------------------------------------
    
    def calibrate_lagrange_multiplier(df, target_avg_bops, rate_col, perf_col, gamma, tolerance=0.0005):
        """
        Calibrate Lagrange multiplier lambda for RP-MPQ online policy selection (Section V-D).
        Uses sparsity-weighted violation cost to protect weak samples.
        """
        lambda_min, lambda_max = 0.0, 1e9
        best_lambda, best_avg = 0.0, 0.0
        selected = None

        rates = df[rate_col].values
        perfs = df[perf_col].values
        bops = df['bops'].values
        mse_linear = df['mse_linear'].values

        # Sparsity weight w_t (Eq.30): low perf -> high sensitivity -> larger weight
        if 'sparsity_weight' in df.columns:
            sparsity_weight = df['sparsity_weight'].values
        else:
            norm_perf = (perfs - perfs.min()) / (perfs.max() - perfs.min() + 1e-9)
            sparsity_weight = 1.0 - norm_perf

        # Importance weight: w_t = 1 + alpha * s_t (Eq.30)
        alpha = 2.0
        importance_weight = 1.0 + (alpha * sparsity_weight)
        
        # (A) Outage indicator (frequency control)
        is_outage = (rates < gamma * perfs).astype(float)

        # (B) Soft quadratic penalty (margin control)
        deficit = np.maximum(0, gamma * perfs - rates)
        scale_factor = np.mean(perfs) * (1 - gamma) + 1e-9
        soft_penalty = (deficit / scale_factor) ** 2

        beta = 0.5
        violation_cost = importance_weight * (is_outage + beta * soft_penalty)
        
        drift_cost = 1e-9 * mse_linear

        for _ in range(100):
            curr_lambda = (lambda_min + lambda_max) / 2

            # Total cost: violation + lambda * kappa_pi (Eq.31)
            total_cost = violation_cost + (curr_lambda * bops) + drift_cost
            
            df['temp_cost'] = total_cost
            
            idx_selected = df.groupby('sample_id')['temp_cost'].idxmin()
            curr_selected = df.loc[idx_selected]
            
            current_avg_bops = curr_selected['bops'].mean()
            
            if abs(current_avg_bops - target_avg_bops) < tolerance:
                return curr_lambda, current_avg_bops, curr_selected
            
            if current_avg_bops > target_avg_bops:
                lambda_min = curr_lambda
            else:
                lambda_max = curr_lambda
                
            best_lambda, best_avg, selected = curr_lambda, current_avg_bops, curr_selected
            
        return best_lambda, best_avg, selected
    
    print("\n[INFO] Phase 2: Optimizing control policy (multi-target)...")
    
    min_save = rate_df['saving'].min()
    max_save = rate_df['saving'].max()
    target_savings = np.arange(np.ceil(min_save), np.floor(max_save) + 0.1, 0.1)
    
    # Reliability targets gamma (Section V)
    reliability_targets = [0.99, 0.98, 0.95]
    
    ranc_results = []
    # Outage thresholds for analysis
    outage_thresholds = [0.99, 0.98, 0.95]

    for snr in snr_list:
        print(f"\n[INFO] Tuning for SNR {snr}dB...")
        rate_col = f'rate_{snr}'
        perf_col = f'perf_{snr}'
        
        for qos_target in reliability_targets:
            desc_str = f"Sim {snr}dB | Target QoS {int(qos_target*100)}%"
            
            for sav in tqdm(target_savings, desc=desc_str):
                target_bops_cost = (100 - sav) / 100.0
                
                # Run optimization
                opt_lambda, realized_bops, final_selection = calibrate_lagrange_multiplier(
                    rate_df, target_bops_cost, rate_col, perf_col, 
                    gamma=qos_target 
                )
                
                realized_saving = (1 - realized_bops) * 100
                res_entry = {
                    'Target_Saving': sav,
                    'Realized_Saving': realized_saving,
                    'Lambda': opt_lambda,
                    'SNR_Context': snr,
                    'QoS_Target': qos_target # grouping key
                }
                
                r_quant = final_selection[rate_col].values
                r_perf = final_selection[perf_col].values
                
                res_entry[f'Rate_{snr}'] = np.mean(r_quant)
                
                # Record outage for all thresholds (for analysis)
                for th in outage_thresholds:
                    is_outage = r_quant < (r_perf * th)
                    res_entry[f'Outage_{snr}dB_{int(th*100)}'] = np.mean(is_outage)
                
                ranc_results.append(res_entry)

    ranc_df = pd.DataFrame(ranc_results)
    print("[INFO] RP-MPQ online simulation completed.")

    # ---------------------------------------------------------
    # Phase 3: Plotting (Updated for Multi-Target)
    # ---------------------------------------------------------
    print("\n[INFO] Phase 3: Plotting multi-target results...")
    cmap = plt.get_cmap('tab10')
    snr_colors = {snr: cmap(i%10) for i, snr in enumerate(snr_list)}
    
    # Per-target line styles
    # 99%: solid/star, 98%: dashed/triangle, 95%: dotted/circle
    target_styles = {
        0.99: {'ls': '-', 'marker': '*', 'alpha': 1.0},
        0.98: {'ls': '--', 'marker': '^', 'alpha': 0.7},
        0.95: {'ls': ':', 'marker': 'o', 'alpha': 0.5}
    }

    def get_pareto_max(x, y):
        data = sorted(zip(x, y), key=lambda k: k[0])
        px, py = [], []
        c_max = -np.inf
        for bx, by in reversed(data):
            if by >= c_max: px.append(bx); py.append(by); c_max = by
        return px[::-1], py[::-1]
    
    def get_pareto_min(x, y):
        data = sorted(zip(x, y), key=lambda k: k[0])
        px, py = [], []
        c_min = np.inf
        for bx, by in reversed(data):
            if by <= c_min: px.append(bx); py.append(by); c_min = by
        return px[::-1], py[::-1]

    # [Plot 1] Rate Overlay
    plt.figure(figsize=(10, 7))
    
    # Plot baselines (once)
    if baseline_df is not None:
        for snr in snr_list:
            bx, by = get_pareto_max(baseline_df['Actual_Saving'], baseline_df[f"Rate_{snr}dB"])
            plt.plot(bx, by, color=snr_colors[snr], linestyle='-', linewidth=3, alpha=0.3, label=f'Static MP ({snr}dB)' if snr==10 else "")
            
    # Plot RP-MPQ results (separated by target)
    for snr in snr_list:
        for qos_t in reliability_targets:
            # Filter data matching this SNR and target
            subset = ranc_df[(ranc_df['SNR_Context'] == snr) & (ranc_df['QoS_Target'] == qos_t)]
            subset = subset.sort_values('Realized_Saving')
            
            style = target_styles.get(qos_t, {'ls': '-', 'marker': '.'})
            label_str = f"RP-MPQ {snr}dB (gamma={qos_t})"
            
            plt.plot(subset['Realized_Saving'], subset[f'Rate_{snr}'], 
                     color=snr_colors[snr], linestyle=style['ls'], marker=style['marker'], 
                     markersize=5, alpha=style['alpha'], label=label_str)
                 
    plt.xlabel("Encoder BOPs Saving (%)")
    plt.ylabel("Sum Rate (bps/Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Rate: Static MP vs RP-MPQ (gamma targets) - {args.encoder}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOT, f"exp4_final_ranc_rate_multi_{args.encoder}.png"), dpi=300)

    # [Plot 2] Outage Overlay (Complex)
    # Create one subplot per SNR
    n_plots = len(snr_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=True)
    if n_plots == 1: axes = [axes]
    
    # Colors per outage threshold
    th_cmap = plt.get_cmap('Reds')
    th_colors = [th_cmap(x) for x in np.linspace(0.4, 1.0, len(outage_thresholds))]
    
    sorted_thresholds = sorted(outage_thresholds, reverse=True)
    
    for i, (ax, snr) in enumerate(zip(axes, snr_list)):
        c = snr_colors[snr]
        ax.set_title(f"SNR {snr}dB (Outage Analysis)", color=c, fontweight='bold')
        
        # 1. Baseline Outage (one curve per threshold)
        for j, th in enumerate(sorted(outage_thresholds, reverse=True)):
            th_int = int(th*100)
            col = f"Outage_{snr}dB_{th_int}"
            lc = th_colors[j]
            
            if baseline_df is not None and col in baseline_df.columns:
                bx, by = get_pareto_min(baseline_df['Actual_Saving'], baseline_df[col])
                ax.plot(bx, by, color=lc, linestyle='-', alpha=0.3, linewidth=2, 
                        label=f'Base {th_int}% Limit' if j==0 else "")
        
        # 2. RP-MPQ Outage (plot only the outage matching each policy's own target)
        # e.g., gamma=99% policy shows Outage_99; gamma=95% shows Outage_95
        for qos_t in reliability_targets:
            subset = ranc_df[(ranc_df['SNR_Context'] == snr) & (ranc_df['QoS_Target'] == qos_t)]
            subset = subset.sort_values('Realized_Saving')
            
            # Show only the outage corresponding to the intended target (most meaningful)
            target_col = f"Outage_{snr}dB_{int(qos_t*100)}"
            
            style = target_styles.get(qos_t, {'ls': '-', 'marker': '.'})
            # Color follows per-target style; use matching threshold color
            
            if qos_t in sorted_thresholds:
                j = sorted_thresholds.index(qos_t) # find index
                lc = th_colors[j]                  # use same color as baseline
            else:
                lc = 'black'
                
            ax.plot(subset['Realized_Saving'], subset[target_col], 
                    color=lc, linestyle=style['ls'], marker=style['marker'], markersize=5, 
                    alpha=0.8, label=f'RP-MPQ gamma={qos_t}')

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Encoder BOPs Saving (%)")
        if i==0:
            ax.set_ylabel("Outage Probability")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOT, f"exp4_final_ranc_outage_multi_{args.encoder}.png"), dpi=300)
    print("[INFO] Exp 4 (RP-MPQ multi-target) completed.")

    real_model.load_state_dict(original_state)
    return ranc_df
    

    
# =========================================================
# [0] Dataset & Loss
# =========================================================
class CsiDataset(Dataset):
    def __init__(self, mat_file_path, data_key='HT', max_samples=None, normalization_params=None):
        super().__init__()
        try:
            mat_data = sio.loadmat(mat_file_path)
            self.csi_data = np.array(mat_data[data_key], dtype=np.float32)
            if self.csi_data.ndim == 2: 
                 self.csi_data = np.reshape(self.csi_data, (self.csi_data.shape[0], 2, 32, 32))
        except:
            print("[WARN] Data file not found. Using dummy data.")
            self.csi_data = np.random.randn(100, 2, 32, 32).astype(np.float32)

        if normalization_params is None:
            self.min_val, self.max_val = np.min(self.csi_data), np.max(self.csi_data)
            self.range_val = self.max_val - self.min_val + 1e-9
            self.normalization_params = (self.min_val, self.range_val)
        else:
            self.min_val, self.range_val = normalization_params
            self.normalization_params = normalization_params
        self.csi_data = (self.csi_data - self.min_val) / self.range_val
        if max_samples is not None: self.csi_data = self.csi_data[:max_samples]
    def __len__(self): return self.csi_data.shape[0]
    def __getitem__(self, idx): return torch.from_numpy(self.csi_data[idx])

class NMSELoss(nn.Module):
    def __init__(self, eps=1e-8): super().__init__(); self.eps = eps
    def forward(self, rec, orig):
        orig_c, rec_c = orig - 0.5, rec - 0.5
        pow_orig = torch.sum(orig_c ** 2, dim=[1, 2, 3])
        mse = torch.sum((orig_c - rec_c) ** 2, dim=[1, 2, 3])
        valid = pow_orig > self.eps
        if torch.sum(valid) == 0: return torch.tensor(0.0, device=orig.device, requires_grad=True)
        return torch.mean(mse[valid] / pow_orig[valid])

def calculate_nmse_db(orig, rec, params):
    min_val, range_val = params
    orig = (orig * range_val) + min_val
    rec = (rec * range_val) + min_val
    orig_c, rec_c = orig - 0.5, rec - 0.5
    pow_orig = torch.sum(orig_c ** 2, dim=[1, 2, 3])
    mse = torch.sum((orig_c - rec_c) ** 2, dim=[1, 2, 3])
    valid = pow_orig > 1e-8
    if torch.sum(valid) == 0: return torch.tensor(-100.0)
    nmse = torch.mean(mse[valid] / pow_orig[valid])
    return 10 * torch.log10(nmse) if nmse.item() > 1e-10 else torch.tensor(-100.0)

# =========================================================
# [1] RP-MPQ Offline: Policy Set Construction (Section IV)
# =========================================================

def quantize_int_asym(w, bits):
    """Helper for internal usage in apply_precision_policy"""
    q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
    w_min, w_max = w.min(), w.max()
    if w_max == w_min: return w
    scale = (w_max - w_min) / (q_max - q_min)
    zp = torch.round(q_min - w_min / scale)
    w_q = torch.round(w / scale + zp)
    w_q = torch.clamp(w_q, q_min, q_max)
    return (w_q - zp) * scale

def apply_precision_policy(model, policy, device=None):
    """Apply mixed-precision quantization policy pi to encoder weights (Section IV-A)"""
    import torch
    import torch.nn as nn
    import re

    real_model = model.module if isinstance(model, nn.DataParallel) else model
    model_map = {}
    
    # Model layer mapping
    for name, module in real_model.encoder.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            clean_name = name.replace("encoder.", "").replace("module.", "")
            model_map[clean_name] = module
            model_map[name] = module

    # Group policy entries (handle split FC layers)
    policy_groups = {}
    for p_key, bits in policy.items():
        clean_key = p_key.replace("encoder.", "").replace("module.", "").replace(".weight", "")
        match = re.search(r'(.+)_part(\d+)$', clean_key)
        if match:
            base_name = match.group(1)
            part_idx = int(match.group(2))
            if base_name not in policy_groups: policy_groups[base_name] = {}
            policy_groups[base_name][part_idx] = bits
        else:
            policy_groups[clean_key] = bits

    applied_count = 0
    
    # Apply quantization
    for base_name, bits_info in policy_groups.items():
        target_module = model_map.get(base_name)
        if target_module is None:
            for m_name, module in model_map.items():
                if m_name.endswith(base_name):
                    target_module = module
                    break
        if target_module is None: continue

        w = target_module.weight.data
        
        # Split FC layer quantization
        if isinstance(bits_info, dict) and 'fc' in base_name:
            sorted_indices = sorted(bits_info.keys())
            max_idx = max(sorted_indices)
            total_chunks = max_idx + 1
            w_chunks = torch.chunk(w, total_chunks, dim=0)
            quantized_chunks = []
            
            # Log only the first 3 chunks as samples
            for i, chunk in enumerate(w_chunks):
                b = bits_info.get(i, 32)
                if b < 32:
                    q_chunk = quantize_int_asym(chunk, b)
                    quantized_chunks.append(q_chunk)
                else:
                    quantized_chunks.append(chunk)
            
            target_module.weight.data = torch.cat(quantized_chunks, dim=0)

        # Standard layer (single bit-width)
        else:
            bit = bits_info
            if bit < 32:
                target_module.weight.data = quantize_int_asym(w, bit)
        
        applied_count += 1        

def restore_fp32_weights(model, original_weights):
    """Restore encoder weights to FP32 reference state"""
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    with torch.no_grad():
        for name, param in real_model.encoder.named_parameters():
            if name in original_weights:
                param.data = original_weights[name].to(param.device)

class ILPCoarseCandidateGenerator:
    """ILP-based coarse candidate generation (Section IV-C)"""
    def __init__(self, hawq_df, layer_params, act_bits=16):
        self.df = hawq_df
        self.layer_params = layer_params
        self.act_bits = act_bits
        self.bit_options = [16, 8, 4, 2]
        self.bops_fp32_baseline = sum(p * 32 * 32 for p in layer_params.values())

    def solve_top_k(self, target_savings_pct, top_k=20):
        """Generate top-k candidate policies via ILP (Section IV-C)"""
        prob = pulp.LpProblem("MP_TopK_Search", pulp.LpMinimize)
        layers = self.df['Layer'].tolist()
        choices = pulp.LpVariable.dicts("x", (layers, self.bit_options), cat=pulp.LpBinary)

        # 1. Objective: Minimize total Omega (Hessian-weighted distortion)
        objective_terms = []
        for name in layers:
            row = self.df[self.df['Layer'] == name].iloc[0]
            for b in self.bit_options:
                objective_terms.append(choices[name][b] * row[f'Omg_INT{b}'])
        prob += pulp.lpSum(objective_terms)

        # 2. Constraint C1: Unique bit-width selection per layer
        for name in layers:
            prob += pulp.lpSum([choices[name][b] for b in self.bit_options]) == 1

        # 3. Constraint C2: UE encoder BOPs budget (Eq.28)
        target_limit = self.bops_fp32_baseline * (1 - target_savings_pct / 100.0)
        cost_terms = []
        for name in layers:
            params = self.layer_params.get(name, 0)
            for b in self.bit_options:
                cost_terms.append(choices[name][b] * (params * b * self.act_bits))
        prob += pulp.lpSum(cost_terms) <= target_limit

        solutions = []

        for k in range(top_k):
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] != 'Optimal': break

            selected_policy = {}
            selected_vars = []
            total_omega = pulp.value(prob.objective)
            encoder_bops = 0

            for name in layers:
                for b in self.bit_options:
                    if pulp.value(choices[name][b]) == 1:
                        selected_policy[name] = b
                        selected_vars.append(choices[name][b])
                        encoder_bops += self.layer_params[name] * b * self.act_bits

            actual_saving = (1 - (encoder_bops / self.bops_fp32_baseline)) * 100
            solutions.append({
                "Policy": selected_policy,
                "Actual_Saving": actual_saving,
                "Total_Omega": total_omega
            })

            # Integer cut to exclude current solution
            prob += pulp.lpSum(selected_vars) <= len(layers) - 1

        return solutions

def kl_distributional_refinement(model, candidates, loader, device, loss_fn, nmse_params):
    """KL-divergence based distributional refinement among ILP candidates (Section IV-D)"""
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    original_weights = {name: p.clone().detach().cpu() for name, p in real_model.encoder.named_parameters()}
    
    # 1. FP32 Data Cache
    fp32_zs = []
    inputs_cache = []
    real_model.eval()
    with torch.no_grad():
        for i, d in enumerate(loader):
            if i >= 5: break
            d = d.to(device)
            z = real_model.encoder(d)
            fp32_zs.append(z) 
            inputs_cache.append(d)
            
    kl_criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
    best_kl_val = float('inf')
    best_candidate_idx = 0

    # 2. Evaluate Candidates (KL)
    for idx, cand in enumerate(candidates):
        apply_precision_policy(model, cand['Policy'], device)
        
        current_kl = 0.0
        with torch.no_grad():
            for z_fp32, inp in zip(fp32_zs, inputs_cache):
                z_quant = real_model.encoder(inp)
                p_target = F.softmax(z_fp32.view(z_fp32.size(0), -1), dim=1)
                q_input = F.log_softmax(z_quant.view(z_quant.size(0), -1), dim=1)
                current_kl += kl_criterion(q_input, p_target).item()
        
        current_kl /= len(fp32_zs)
        
        if current_kl < best_kl_val:
            best_kl_val = current_kl
            best_candidate_idx = idx
            
        restore_fp32_weights(model, original_weights)

    # 3. NMSE Check
    def get_nmse(policy_dict):
        apply_precision_policy(model, policy_dict, device)
        val_nmse = test_epoch_ae(0, loader, model, loss_fn, None, nmse_params, device, verbose=False)
        restore_fp32_weights(model, original_weights)
        return val_nmse

    ilp_best_cand = candidates[0]              
    kl_best_cand = candidates[best_candidate_idx] 
    
    nmse_ilp = get_nmse(ilp_best_cand['Policy'])
    nmse_kl = get_nmse(kl_best_cand['Policy'])
    
    kl_best_cand['NMSE_KL'] = nmse_kl
    kl_best_cand['NMSE_ILP'] = nmse_ilp
    return kl_best_cand
    
def construct_offline_policy_set(model, loader, hawq_df, layer_params, args, device, norm_params):
    """
    [RP-MPQ Offline] Policy Set Construction Pi_C (Section IV)
    Stage 1: Pareto-optimal policy search via ILP + KL refinement
    Stage 2: Calibration data collection (1000 samples) for online stage
    """
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset
    import os

    # --- [1. Setup] ---
    solver = ILPCoarseCandidateGenerator(hawq_df, layer_params, args.act_quant)
    targets = np.arange(75.0, 95.1, 0.5) # 0.5% Step
    targets = [round(x, 1) for x in targets]

    # Pre-compute Hoyer's sparsity for all samples
    _, all_sparsity = compute_hoyer_sparsity_bins(loader.dataset, device)
    
    lut_results = []        # Stage 1: Pareto LUT
    calibration_data = []   # Stage 2: Calibration data for online stage
    
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}
    min_v, range_v = norm_params

    # Fixed 1000-sample loader for statistical calibration
    fit_indices = np.linspace(0, len(loader.dataset)-1, 1000, dtype=int)
    fit_loader = DataLoader(Subset(loader.dataset, fit_indices), batch_size=args.test_batch_size, shuffle=False)

    print(f"\n[INFO] Offline Policy Search: Range 75-95% | Step 0.5% | Points: {len(targets)}")

    # --- [2. Search & Calibration Loop] ---
    for target_b in tqdm(targets, desc="Scanning Pareto & Calibration"):
        # (1) ILP top-k candidate generation (Section IV-C)
        candidates = solver.solve_top_k(target_b, top_k=10)
        if not candidates: continue
        
        # (2) KL distributional refinement (Section IV-D)
        best_entry = kl_distributional_refinement(model, candidates, loader, device, NMSELoss(), norm_params)
        
        policy = best_entry['Policy']
        actual_b = best_entry['Actual_Saving'] / 100.0

        # (3) Per-sample NMSE measurement for calibration (linear domain)
        real_model.load_state_dict(original_state)
        apply_precision_policy(real_model, policy, device)
        real_model.eval()
        
        policy_nmses_linear = []
        with torch.no_grad():
            for batch in fit_loader:
                d = batch.to(device)
                z = real_model.encoder(d)
                if args.aq > 0: z = quantize_feedback_torch(z, args.aq)
                x_hat = real_model.decoder(z)
                
                orig_c = (d * range_v) + min_v - 0.5
                rec_c = (x_hat * range_v) + min_v - 0.5
                mse = torch.sum((orig_c - rec_c)**2, dim=(1,2,3))
                pow_orig = torch.sum(orig_c**2, dim=(1,2,3))
                nmse_l = (mse / (pow_orig + 1e-8)).cpu().numpy()
                policy_nmses_linear.extend(nmse_l)

        # (4) Record results
        lut_results.append({
            "Target_Saving": target_b,
            "Actual_Saving": best_entry['Actual_Saving'],
            "NMSE_KL": best_entry['NMSE_KL'],
            "NMSE_ILP": best_entry['NMSE_ILP'],
            "Policy": policy,
            "Summary": {b: list(policy.values()).count(b) for b in [16, 8, 4, 2]}
        })

        # Per-sample calibration data (1000 samples)
        for i in range(len(policy_nmses_linear)):
            g_idx = fit_indices[i]
            calibration_data.append({
                "S": all_sparsity[g_idx],
                "B": actual_b,
                "NMSE_linear": policy_nmses_linear[i]
            })
        
        real_model.load_state_dict(original_state)

    # --- [3. Post-processing: Pareto Smoothing] ---
    df_lut = pd.DataFrame(lut_results).sort_values(by='Actual_Saving')
    
    # Monotonic smoothing for both (fair comparison)
    for col in ['NMSE_KL', 'NMSE_ILP']:
        vals = df_lut[col].tolist()
        pruned = []
        current_best = float('inf')
        for val in reversed(vals):
            if val < current_best:
                current_best = val
            pruned.append(current_best)
        df_lut[col] = pruned[::-1]

    # --- [4. Save Results] ---
    df_lut.to_csv(os.path.join(RESULTS_CSV, f"mp_policy_lut_{args.encoder}_pruned.csv"), index=False)
    pd.DataFrame(calibration_data).to_csv(os.path.join(RESULTS_CSV, f"fitting_raw_data_{args.encoder}.csv"), index=False)

    print(f"\n[INFO] Offline policy set + calibration data saved.")
    return lut_results
    
import torch
import numpy as np
def calculate_su_miso_rate_mrt(h_true, h_hat, snr_db, device):
    """
    FFT-based SU-MISO Rate Calculation (Section VI-A)

    Pipeline:
    1. Delay Domain CSI (Tap) -> FFT -> Frequency Domain CSI (Subcarrier)
    2. Per-subcarrier MRT Precoding
    3. Frequency-domain rate computation -> average (Average Spectral Efficiency)

    Args:
        h_true: (Batch, 2, 32, 32) - (Real/Imag, Angular(Tx), Delay(Tap))
        h_hat:  (Batch, 2, 32, 32) - Quantized Estimated Channel
    """
    import torch
    import torch.fft

    # 1. Convert to complex: (Batch, 2, Nt, L) -> (Batch, Nt, L)
    if h_true.shape[1] == 2:
        h_true_c = torch.complex(h_true[:, 0, ...], h_true[:, 1, ...])
        h_hat_c  = torch.complex(h_hat[:, 0, ...], h_hat[:, 1, ...])
    else:
        h_true_c, h_hat_c = h_true, h_hat

    # Shape: (Batch, Angular/Tx, Delay/Tap)
    # Convention: dim=1 -> Tx Antennas (32), dim=2 -> Delay Taps (32)

    # 2. FFT: Delay Domain -> Frequency Domain
    # Apply FFT along delay axis (dim=2) -> (Batch, Tx, Subcarriers)
    # Power normalization applied later instead of norm='ortho'
    
    H_true_freq = torch.fft.fft(h_true_c, dim=-1) # (B, 32, 32)
    H_hat_freq  = torch.fft.fft(h_hat_c,  dim=-1) # (B, 32, 32)
    
    # After FFT, dim=2 represents Subcarrier(k), not Delay Tap.
    # Reshape to (Batch * Subcarrier, 1, Tx) to treat each subcarrier
    # as an independent MISO channel.
    
    B, Nt, K = H_true_freq.shape
    
    # (B, Nt, K) -> permute(0, 2, 1) -> (B, K, Nt) -> reshape -> (B*K, 1, Nt)
    # H_true_flat: (Total_Samples, 1, 32)
    H_true_flat = H_true_freq.permute(0, 2, 1).reshape(-1, 1, Nt)
    H_hat_flat  = H_hat_freq.permute(0, 2, 1).reshape(-1, 1, Nt)

    # 3. Subcarrier-wise MRT Precoding
    # Beamform using estimated channel H_hat at each frequency k
    # w = H_hat^H / |H_hat|
    
    w_precoder = H_hat_flat.conj().mT # (Total, 32, 1)
    
    # Power Normalization (Unit Norm per Subcarrier)
    w_norm = torch.norm(w_precoder, dim=1, keepdim=True)
    w_precoder = w_precoder / (w_norm + 1e-8)

    # 4. Effective Channel (in Frequency Domain)
    # y = H_true(k) * w(k)
    # Captures frequency-selective fading
    h_eff = torch.matmul(H_true_flat, w_precoder) # (Total, 1, 1)
    
    # 5. Rate Calculation
    # R_k = log2(1 + SNR * |h_eff_k|^2)
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = torch.abs(h_eff.squeeze()) ** 2 # (Total,)
    
    rates_per_subcarrier = torch.log2(1 + snr_linear * signal_power) # (B * K,)
    
    # 6. Average Spectral Efficiency
    # Average over subcarriers per batch sample
    # (B * K,) -> (B, K) -> mean(dim=1) -> (B,)
    rates = rates_per_subcarrier.view(B, K).mean(dim=1)
    
    return rates
    
    
# [3] Weight & Feedback Quantization
# =========================================================
def quantize_tensor(w, bits, q_type='asym'):
    if q_type == 'sym': 
        q_max = (2**(bits-1)) - 1
        max_val = w.abs().max()
        if max_val == 0: return w
        scale = max_val / q_max
        w_q = torch.round(w / scale)
        w_q = torch.clamp(w_q, -q_max, q_max)
        return w_q * scale
    return quantize_int_asym(w, bits)

def quantize_feedback_torch(y, bits):
    if bits <= 0: return y
    min_val = y.min(dim=1, keepdim=True)[0]; max_val = y.max(dim=1, keepdim=True)[0]
    range_val = max_val - min_val + 1e-9; levels = 2 ** bits - 1
    y_norm = (y - min_val) / range_val
    y_q_norm = torch.round(y_norm * levels) / levels
    return y_q_norm * range_val + min_val

def apply_weight_quantization(model, mode, q_type, hybrid=False, hybrid_strategy='step'):
    if mode == 'FP32': return 0, 0
    print(f"\n[INFO] Applying PTQ: {mode} ({q_type}) | Hybrid: {hybrid}")
    
    target_modules = (nn.Conv2d, nn.Linear, nn.Conv1d)
    count_base, count_protected = 0, 0
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    
    with torch.no_grad():
        for name, module in real_model.encoder.named_modules():
            if isinstance(module, target_modules):
                w = module.weight.data
                target_mode = mode
                
                if hybrid:
                    if 'conv' in name or 'stem' in name or 'dt_proj' in name or 'A_log' in name or 'ssm' in name:
                        if hybrid_strategy == 'fixed': target_mode = 'INT16'
                        else:
                            if mode == 'INT2': target_mode = 'INT4'
                            elif mode == 'INT4': target_mode = 'INT8'
                            elif mode == 'INT8': target_mode = 'INT16'
                            elif mode == 'INT16': target_mode = 'FP32'
                
                if target_mode != 'FP32':
                    bits = int(target_mode.replace('INT', '')) if 'INT' in target_mode else 8
                    module.weight.data = quantize_tensor(w, bits, q_type)
                
                if target_mode == mode: count_base += 1
                else: count_protected += 1
    
    return count_base, count_protected

# =========================================================
# [4] Hessian Sensitivity Analysis (Section IV-B)
# =========================================================
class HessianSensitivityAnalyzer:
    def __init__(self, model, loader, loss_fn, device):
        self.model = model
        self.loader = loader
        self.loss_fn = loss_fn
        self.device = device

    def get_gradients(self, inputs):
        self.model.zero_grad()
        output = self.model(inputs)
        loss = self.loss_fn(output, inputs)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        return params, grads

    def compute_importance(self, bit_widths=[16, 8, 4, 2], split_threshold=20000, num_chunks=32):
        """Compute Hessian-weighted importance Omega per layer/block (Section IV-B)"""
        print("\n" + "="*120)
        print(f"[INFO] Hessian Sensitivity Analysis: Computing Omega with block-wise splitting")
        print(f"       Layers > {split_threshold} params split into {num_chunks} blocks.")
        print("="*120)
        
        real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        real_model.eval()
        
        try: inputs = next(iter(self.loader)).to(self.device)
        except StopIteration: return

        params, grads = self.get_gradients(inputs)
        param_grad_map = {id(p): g for p, g in zip(params, grads)}

        results = []
        target_modules = (nn.Conv2d, nn.Linear, nn.Conv1d)
        
        for name, module in tqdm(real_model.encoder.named_modules(), desc="Analyzing Layers"):
            if isinstance(module, target_modules):
                w = module.weight
                if id(w) not in param_grad_map: continue
                grad_w = param_grad_map[id(w)]

                is_large_layer = w.numel() > split_threshold
                
                if is_large_layer:
                    w_chunks = torch.chunk(w, num_chunks, dim=0)
                    g_chunks = torch.chunk(grad_w, num_chunks, dim=0)
                    sub_names = [f"{name}_part{i}" for i in range(len(w_chunks))]
                else:
                    w_chunks = [w]
                    g_chunks = [grad_w]
                    sub_names = [name]

                for i, (sub_w, sub_grad, sub_name) in enumerate(zip(w_chunks, g_chunks, sub_names)):
                    v = torch.randint_like(sub_w, high=2, device=self.device) * 2 - 1.0
                    grad_v_prod = torch.sum(sub_grad * v)
                    
                    if is_large_layer:
                        hv_full = torch.autograd.grad(grad_v_prod, w, retain_graph=True)[0]
                        hv_chunk = torch.chunk(hv_full, num_chunks, dim=0)[i]
                        trace_val = abs(torch.sum(v * hv_chunk).item())
                    else:
                        hv = torch.autograd.grad(grad_v_prod, w, retain_graph=True)[0]
                        trace_val = abs(torch.sum(v * hv).item())

                    layer_res = {"Layer": sub_name, "Params": sub_w.numel(), "Trace(H)": trace_val}
                    for b in bit_widths:
                        w_q = quantize_tensor(sub_w, b, q_type='asym')
                        l2_err = torch.norm(sub_w - w_q, p=2).item() ** 2
                        layer_res[f"Omg_INT{b}"] = trace_val * l2_err

                    results.append(layer_res)
        
        df = pd.DataFrame(results)
        if not df.empty:
            sort_key = f"Omg_INT{min(bit_widths)}"
            df = df.sort_values(by=sort_key, ascending=False).reset_index(drop=True)
            
            print("\n[INFO] Layer Importance Table (Top 10 Chunks)")
            display_df = df.copy()
            display_df['Trace(H)'] = display_df['Trace(H)'].map('{:.2e}'.format)
            for b in bit_widths:
                col = f"Omg_INT{b}"
                display_df[col] = display_df[col].map('{:.2e}'.format)
            display_df['Params'] = display_df['Params'].map('{:,}'.format)

            print(display_df.head(10).to_string(index=False))
            
            csv_path = os.path.join(RESULTS_CSV, "hawq_importance_split.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n[INFO] Results saved to: {csv_path}")

# =========================================================
# [5] ONNX Benchmark
# =========================================================
def run_onnx_benchmark(model, args, input_shape=(1, 2, 32, 32)):
    if not HAS_ONNX:
        print("[WARN] onnx or onnxruntime not installed. Skipping benchmark."); return

    print("\n" + "="*80)
    print("[INFO] ONNX Benchmark: Attempting Mamba Export (Experimental)")
    print("="*80)

    os.makedirs("onnx_models", exist_ok=True)
    model_name = f"onnx_models/{args.encoder}_enc"
    fp32_path = f"{model_name}_fp32.onnx"
    int8_path = f"{model_name}_int8.onnx"

    real_model = model.module if isinstance(model, nn.DataParallel) else model
    encoder_only = copy.deepcopy(real_model.encoder).cpu().eval()
    dummy_input = torch.randn(*input_shape).cpu()

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
        original_ss_fn = selective_scan_fn
        import mamba_ssm.ops.selective_scan_interface as ssi
        ssi.selective_scan_fn = selective_scan_ref
    except ImportError:
        original_ss_fn = None

    try:
        torch.onnx.export(
            encoder_only, dummy_input, fp32_path, export_params=True, opset_version=17, 
            do_constant_folding=True, input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    except Exception as e:
        print(f"[ERROR] Export Failed: {e}"); return
    finally:
        if original_ss_fn: ssi.selective_scan_fn = original_ss_fn

    try:
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
    except Exception as e:
        print(f"[WARN] Quantization Failed: {e}")

    def bench_ort(path, name):
        if not os.path.exists(path): return 0.0
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        try: session = ort.InferenceSession(path, sess_options, providers=['CPUExecutionProvider'])
        except: return 0.0
        input_name = session.get_inputs()[0].name
        for _ in range(5): session.run(None, {input_name: dummy_input.numpy()})
        latencies = []
        for _ in range(30):
            start = time.time()
            session.run(None, {input_name: dummy_input.numpy()})
            latencies.append((time.time() - start) * 1000)
        avg_lat = sum(latencies) / len(latencies)
        print(f"    [{name}] Average Latency: {avg_lat:.4f} ms")
        return avg_lat

    bench_ort(fp32_path, "ONNX FP32")
    bench_ort(int8_path, "ONNX INT8")

# =========================================================
# [6] Training & Evaluation Pipeline
# =========================================================
def measure_latency(model, input_tensor, runs=100):
    model.eval()
    with torch.no_grad():
        for _ in range(5): _ = model(input_tensor)
        start = time.time()
        for _ in range(runs): _ = model(input_tensor)
        return (time.time() - start) * 1000 / runs

def count_parameters_real(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch_ae(model, loss_fn, loader, optimizer, epoch, clip_norm, scaler, device):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [Train]")
    for d in pbar:
        d = d.to(device); optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device=='cuda')): output = model(d); loss = loss_fn(output, d)
        if device == 'cuda':
            scaler.scale(loss).backward()
            if clip_norm > 0: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer); scaler.update()
        else: loss.backward(); optimizer.step()
        pbar.set_postfix(NMSE_Loss=f"{loss.item():.6f}")

def test_epoch_ae(epoch, loader, model, loss_fn, args, norm_params, device, verbose=True):
    model.eval()
    nmse_sum, count = 0, 0
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    iterator = tqdm(loader, desc=f"Epoch {epoch + 1} [Benchmark]" if verbose else "Val") if verbose else loader
    with torch.no_grad():
        for d in iterator:
            d = d.to(device)
            with torch.amp.autocast('cuda', enabled=(device=='cuda')): 
                z = real_model.encoder(d)
                if args and args.aq > 0: z = quantize_feedback_torch(z, args.aq)
                x_hat = real_model.decoder(z)
            nmse_db = calculate_nmse_db(d, x_hat, norm_params)
            nmse_sum += nmse_db.item(); count += 1
            if verbose: iterator.set_postfix(NMSE_dB=f"{nmse_sum/count:.4f}")
    return nmse_sum/count

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="mamba")
    parser.add_argument("--decoder", type=str, default="csinet")
    parser.add_argument("--encoded_dim", type=int, default=512)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--encoder_layers", type=int, default=2) 
    parser.add_argument("--decoder_layers", type=int, default=2)

    parser.add_argument('--pq', type=str, default='FP32', help='Weight Quantization (e.g., INT4, INT8)')
    parser.add_argument('--quant_type', type=str, default='asym') 
    parser.add_argument('--aq', type=int, default=0, help='Feedback (z) Quantization bits')
    parser.add_argument('--act_quant', type=int, default=32, help='Activation & Input Quantization bits')
    
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--hybrid_strategy', type=str, default='step')
    parser.add_argument('--analyze_all', action='store_true', help='Offline Search (LUT Gen) + Online Sim')
    parser.add_argument('--run_online_sim', action='store_true', help='Run Online Simulation only')
    parser.add_argument('--use_chunking', action='store_true', default=False, help='Enable chunking')

    parser.add_argument("-e", "--epochs", default=0, type=int) 
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float) 
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    
    parser.add_argument("--train_path", type=str, default="data/DATA_Htrainin.mat")
    parser.add_argument("--test_path", type=str, default="data/DATA_Htestin.mat")
    parser.add_argument("--train_key", type=str, default="HT")
    parser.add_argument("--test_key", type=str, default="HT")
    
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--checkpoint", type=str)
    
    return parser.parse_args(argv)

def inspect_layer_names(model, lut_path):
    import pandas as pd
    import ast
    import torch.nn as nn

    print("\n" + "="*80)
    print("[INFO] Layer Name Inspection (raw comparison)")
    print("="*80)

    # 1. Extract actual model layer names
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Filter modules with weights
    model_keys = []
    for name, module in real_model.encoder.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            model_keys.append(name)
    
    print(f"[INFO] Real Model Keys ({len(model_keys)} total):")
    for k in model_keys[:]:
        print(f"   Model: '{k}'")

    # 2. Extract CSV policy keys
    try:
        df = pd.read_csv(lut_path)
        policy_str = df.iloc[0]['Policy']
        policy = ast.literal_eval(policy_str) if isinstance(policy_str, str) else policy_str
        policy_keys = list(policy.keys())
        
        print(f"\n[INFO] CSV Policy Keys ({len(policy_keys)} total):")
        for k in policy_keys[:]:
            print(f"   Policy: '{k}'")
            
    except Exception as e:
        print(f"[ERROR] CSV read failed: {e}")
        return

    print("\n" + "="*80)
    
def plot_offline_pareto(lut, encoder_name):
    """
    [Exp 1: Offline LUT Accuracy & Pareto Pruning]
    - Flexible key name handling for 'Actual(%)'
    - Monotonic smoothing applied to both NMSE_KL and NMSE_ILP
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if not lut:
        print("[WARN] LUT data is empty. Skipping plot."); return
    
    # 1. Data load (flexible key references)
    lut_df = pd.DataFrame([{
        "Target_Saving": e.get('Target(%)', e.get('Target_Saving', 0)),
        "Actual_Saving": e.get('Actual(%)', e.get('Actual_Saving', 0)),
        "NMSE_ILP": e.get('NMSE_ILP', 0),
        "NMSE_KL": e.get('NMSE_KL', e.get('NMSE(dB)', 0)),
        "Policy": e.get('Policy', {}),
        **{f"INT{b}": e.get('Summary', {}).get(b, 0) for b in [16, 8, 4, 2]}
    } for e in lut])
    
    # Sort by Actual_Saving (BOPs Saving)
    lut_df = lut_df.sort_values(by="Actual_Saving").reset_index(drop=True)

    # 2. Plot uses RAW data (fair: both unsmoothed, ILP spikes visible)
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(lut_df["Actual_Saving"], lut_df["NMSE_ILP"],
            'b--s', label='ILP Prediction', alpha=0.7, markersize=5, linewidth=1.5)
    ax.plot(lut_df["Actual_Saving"], lut_df["NMSE_KL"],
            'r-o', label='KL-Refined', linewidth=2, markersize=5)

    ax.set_xlabel("BOPs Saving (%)")
    ax.set_ylabel("NMSE (dB)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=12)
    fig.tight_layout()

    fig.savefig(os.path.join(RESULTS_PLOT, f"exp1_pareto_accuracy_{encoder_name}.png"), dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, "kl_vs_ilp.pdf"), dpi=300)
    plt.show()

    # 3. Monotonic smoothing for LUT CSV (used by online stage)
    for col in ['NMSE_KL', 'NMSE_ILP']:
        vals = lut_df[col].tolist()
        pruned = []
        current_best = float('inf')
        for i in range(len(vals)-1, -1, -1):
            if vals[i] < current_best:
                current_best = val = vals[i]
            pruned.append(current_best)
        lut_df[col] = pruned[::-1]

    # 4. Save smoothed LUT
    output_file = os.path.join(RESULTS_CSV, f"mp_policy_lut_{encoder_name}_pruned.csv")
    lut_df.to_csv(output_file, index=False)
    print(f"[INFO] Pruned LUT saved to: {output_file}")
    
def main(argv):
    args = parse_args(argv)
    save_dir = f"saved_models/{args.encoder}_{args.decoder}"
    print(f"--- Start: {datetime.now()} ---")
    print(f"[INFO] Config: W:[{args.pq}] A:[INT{args.act_quant}] FB:[{args.aq}-bit]")
    print(f"    Hybrid: {args.hybrid} | Chunking: {args.use_chunking}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device.upper()}")

    try:
        train_set = CsiDataset(args.train_path, args.train_key)
        test_set = CsiDataset(args.test_path, args.test_key, normalization_params=train_set.normalization_params)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device=='cuda'))
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device=='cuda'))
    except: 
        print("[ERROR] Dataset loading failed"); return

    # Model initialization
    try:
        from MambaAE import MambaAE
        if args.encoder == 'mamba':
            from ModularModels import ModularAE
            net = ModularAE(encoder_type='mamba', decoder_type=args.decoder, 
                            encoded_dim=args.encoded_dim, M=args.M,
                            use_chunking=args.use_chunking,
                            act_quant_bits=args.act_quant).to(device)
        else:
            from ModularModels import ModularAE
            net = ModularAE(encoder_type=args.encoder, decoder_type=args.decoder, 
                            encoded_dim=args.encoded_dim, M=args.M).to(device)
            
    except Exception as e:
        print(f"[WARN] Model Init: {e}. Trying basic ModularAE.")
        from ModularModels import ModularAE
        net = ModularAE(encoder_type=args.encoder, decoder_type=args.decoder,
                        encoded_dim=args.encoded_dim, M=args.M).to(device)

    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    mse_loss = NMSELoss()
    scaler = GradScaler(enabled=(device == 'cuda'))
    
    if args.checkpoint and os.path.isfile(args.checkpoint):
        try: net.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"], strict=False)
        except: pass
    elif args.epochs == 0: print("[WARN] No checkpoint. Evaluating initialized model.")

    if device == 'cuda' and torch.cuda.device_count() > 1: net = nn.DataParallel(net)
    real_model = net.module if isinstance(net, nn.DataParallel) else net


    if args.epochs == 0:
        # Option 1: RP-MPQ Offline Analysis + Online Simulation
        if args.analyze_all:
            pruned_path = os.path.join(RESULTS_CSV, f"mp_policy_lut_{args.encoder}_pruned.csv")
            fitting_data_path = os.path.join(RESULTS_CSV, f"fitting_raw_data_{args.encoder}.csv")

            # --- [Step 1] Acquire HAWQ & Layer Params ---
            # Load existing HAWQ results (compute if not found)
            hawq_path = os.path.join(RESULTS_CSV, 'hawq_importance_split.csv')
            if os.path.exists(hawq_path):
                hawq_df = pd.read_csv(hawq_path)
                print(f"[INFO] Loaded existing HAWQ results: {hawq_path}")
            else:
                print("\n[INFO] No HAWQ results found. Computing importance...")
                HessianSensitivityAnalyzer(net, train_loader, mse_loss, device).compute_importance()
                hawq_df = pd.read_csv(hawq_path)
            
            # Extract per-layer parameter info for ILP solver
            real_model = net.module if isinstance(net, nn.DataParallel) else net
            l_params = {}
            for n, m in real_model.encoder.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if m.weight.numel() > 20000:
                        for i, c in enumerate(torch.chunk(m.weight, 32, dim=0)): 
                            l_params[f"{n}_part{i}"] = c.numel()
                    else: 
                        l_params[n] = m.weight.numel()

            # --- [Step 2] Unified Search (75-95%, 0.5% Step) ---
            # Run unified scan if LUT(.csv) or calibration data(.csv) is missing
            if not os.path.exists(pruned_path) or not os.path.exists(fitting_data_path):
                print(f"\n[INFO] Starting unified scan...")
                print(f"   Range: 75.0% - 95.0% | Resolution: 0.5% Step | Sampling: 1,000 per policy")
                
                # Unified call replacing previous two-function workflow
                lut = construct_offline_policy_set(
                    net, test_loader, hawq_df, l_params, args, device, train_set.normalization_params
                )
            else:
                print(f"\n[INFO] All offline data found (LUT & calibration data).")
                df_existing = pd.read_csv(pruned_path)
                
                # Restore Policy column from string to dict (also Summary if present)
                if 'Policy' in df_existing.columns and isinstance(df_existing['Policy'].iloc[0], str):
                    df_existing['Policy'] = df_existing['Policy'].apply(ast.literal_eval)
                if 'Summary' in df_existing.columns and isinstance(df_existing['Summary'].iloc[0], str):
                    df_existing['Summary'] = df_existing['Summary'].apply(ast.literal_eval)
                
                # Convert to list of dicts for plot_offline_pareto
                lut = df_existing.to_dict('records')

            plot_offline_pareto(lut, args.encoder)
            # --- [Step 3] Result visualization & validation (Exp 3) ---
            # Read generated mp_policy_lut..._pruned.csv and produce 4 performance graphs
            print("\n[INFO] Generating Exp 3 analysis graphs (Performance, Stability, CDF, Outage)...")
            res_df = run_exp3_sparsity_frontier(net, test_loader, pruned_path, device, train_set.normalization_params, args)

            #verify_fc_quantization(net, "mp_policy_lut_mamba_pruned.csv", device)
            #inspect_layer_names(net, pruned_path)
            
            print("\n[INFO] Step 3.5: Running rate-outage analysis...")
            rate_df = run_exp3_5_rate_outage_analysis(net, test_loader, pruned_path, device, train_set.normalization_params, args)
        
            print("\n[INFO] Step 4: Running RP-MPQ online simulation...")
            ranc_df = run_exp4_rp_mpq_online(net, test_loader, pruned_path, device, rate_df, args)
        
            print(f"\n[INFO] All analysis results for {args.encoder} are ready.")
            return
        

        # 1. Manual Quantization
        if args.pq != 'FP32':
            apply_weight_quantization(net, args.pq, args.quant_type, 
                                      hybrid=args.hybrid, hybrid_strategy=args.hybrid_strategy)

        # 2. Benchmark
        nmse = test_epoch_ae(0, test_loader, net, mse_loss, args, train_set.normalization_params, device)
        dummy_input = torch.randn(1, 2, 32, 32).to(device)
        enc_latency = measure_latency(real_model.encoder, dummy_input, runs=300)
        

        # ---------------------------------------------------------
        # FLOPs Measurement (Encoder & Total)
        # ---------------------------------------------------------
        try:
            from thop import profile
            print("\n" + "="*60)
            print("[INFO] FLOPs Measurement")
            
            # 1. Dummy input (same shape as latency measurement, batch=1)
            flops_input = torch.randn(1, 2, 32, 32).to(device)
    
            # 2. Encoder FLOPs (thop takes (model, inputs) tuple)
            macs_enc, params_enc = profile(real_model.encoder, inputs=(flops_input, ), verbose=False)
            
            # 3. Total (Encoder + Decoder) FLOPs
            macs_total, params_total = profile(real_model, inputs=(flops_input, ), verbose=False)
    
            # 4. Print results (GMACs = MACs / 1e9)
            # thop returns MACs; recent papers report MACs directly as GFLOPs(MACs)
            print(f" - Encoder MACs:   {macs_enc / 1e9:.4f} G (Giga MACs)")
            print(f" - Total MACs:     {macs_total / 1e9:.4f} G (Giga MACs)")
            print("="*60)
    
        except ImportError:
            print("[WARN] 'thop' library not installed. Skipping FLOPs measurement.")
            print("   -> Run '!pip install thop' to enable this.")
        except Exception as e:
            print(f"[WARN] FLOPs measurement failed: {e}")
        # ---------------------------------------------------------

        # 3. Stats & Logs
        params_total = count_parameters_real(real_model.encoder)
        avg_weight_bits, weight_count, total_bits_raw = 0, 0, 0
        for name, module in real_model.encoder.named_modules():
            if hasattr(module, 'weight'): 
                w = module.weight.data
                uniq = torch.unique(w).numel()
                bits = 32
                if uniq <= 4: bits = 2
                elif uniq <= 16: bits = 4
                elif uniq <= 256: bits = 8
                elif uniq <= 65536: bits = 16 
                avg_weight_bits += bits * w.numel()
                weight_count += w.numel()
        
        if weight_count > 0: avg_weight_bits /= weight_count
        else: avg_weight_bits = 32

        
        # ---------------------------------------------------------
        # Absolute BOPs Calculation (for Table II comparison)
        # ---------------------------------------------------------
        # Formula: BOPs = MACs * Weight_Bits * Activation_Bits (Eq.28)
        # Requires macs_enc from thop profiling above
        
        if 'macs_enc' in locals():
            # Current activation bit-width (e.g., 16)
            current_act_bit = args.act_quant if args.act_quant < 32 else 32
            
            # 1. FP32 Baseline BOPs
            # MACs * 32(Weight) * 32(Act)
            bops_fp32 = macs_enc * 32 * 32
            
            # 2. Current model BOPs (Quantized)
            # MACs * avg_weight_bits * act_bits (avg_weight_bits computed above)
            bops_quant = macs_enc * avg_weight_bits * current_act_bit
            
            print("\n" + "="*60)
            print("[INFO] BOPs Measurement")
            print(f" - Config:       W-Avg {avg_weight_bits:.2f} bit | A {current_act_bit} bit")
            print(f" - Encoder FP32: {bops_fp32 / 1e9:.4f} G-BOPs ({bops_fp32 / 1e6:.2f} M-BOPs)")
            print(f" - Encoder Ours: {bops_quant / 1e9:.4f} G-BOPs ({bops_quant / 1e6:.2f} M-BOPs)")
            print("="*60)
        else:
            print("[WARN] FLOPs(macs_enc) not measured. Cannot calculate BOPs.")

        act_bits = args.act_quant if args.act_quant < 32 else 32
        bops_reduction = (1 - ( (avg_weight_bits * act_bits) / (32 * 32) )) * 100

        print("\n" + "="*60)
        print(f" [PYTORCH RESULT] {args.encoder.upper()} | W: {args.pq} | A: INT{act_bits} | FB: {args.aq}-bit")
        print("="*60)
        print(f" - Params:         {params_total:,}")
        print(f" - Avg W-Bit:      {avg_weight_bits:.1f} bits")
        if 'macs_enc' in locals():
            print(f" - Enc FLOPs:      {macs_enc / 1e6:.2f} M")
            print(f" - Total FLOPs:    {macs_total / 1e6:.2f} M")
        print(f" - BOPs Saving:    {bops_reduction:.2f} % (Theoretical)")
        print(f" - NMSE:           {nmse:.2f} dB")
        print(f" - Encoder Time:   {enc_latency:.4f} ms/sample")
        print("="*60)

        #run_onnx_benchmark(net, args)
        return

    # Train Mode
    if args.save: os.makedirs(save_dir, exist_ok=True)
    best_nmse = float("inf")
    for epoch in range(args.epochs):
        train_one_epoch_ae(net, mse_loss, train_loader, optimizer, epoch, args.clip_max_norm, scaler, device)
        nmse = test_epoch_ae(epoch, test_loader, net, mse_loss, args, train_set.normalization_params, device)
        if nmse < best_nmse:
            best_nmse = nmse
            if args.save: torch.save({"state_dict": net.state_dict(), "epoch": epoch}, f"{save_dir}/best.pth")

if __name__ == "__main__":
    main(sys.argv[1:])