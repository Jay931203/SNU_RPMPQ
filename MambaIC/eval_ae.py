# eval_ae.py
# Evaluation script for Asymmetric CSI Feedback Autoencoder

import os, sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

models_path = os.path.join(project_root, "models")
if models_path not in sys.path:
    sys.path.append(models_path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from tqdm import tqdm
import scipy.io as sio
import math
from torch.cuda.amp import autocast

try:
    from thop import profile, clever_format
    print("[INFO] thop (FLOPs/Params) imported successfully.")
except ImportError:
    print("[WARN] thop library not found. (pip install thop)")
    profile, clever_format = None, None

try:
    from MambaAE import MambaAE
    print("[INFO] MambaAE imported successfully.")
except ImportError as e:
    print(f"[ERROR] {e}. Ensure MambaAE.py is in the project root.")
    exit()

# ----------------------------------------------------
# 1. CsiDataset
# ----------------------------------------------------
class CsiDataset(Dataset):
    def __init__(self, mat_file_path, data_key='HT', max_samples=None, normalization_params=None):
        super(CsiDataset, self).__init__()
        self.mat_file_path = mat_file_path
        print(f"Loading .mat file: {self.mat_file_path} (key: {data_key}) ...")
        try:
            mat_data = sio.loadmat(self.mat_file_path)
            self.csi_data = np.array(mat_data[data_key], dtype=np.float32)
        except Exception as e:
            print(f"Error loading {mat_file_path}: {e}"); raise
        
        try: csi_data_reshaped = np.reshape(self.csi_data, (self.csi_data.shape[0], 2, 32, 32))
        except Exception as e: print(f"Reshape Error: {e}"); raise
        
        if normalization_params is None:
            self.min_val = np.min(csi_data_reshaped); self.max_val = np.max(csi_data_reshaped)
            self.range_val = self.max_val - self.min_val
            print(f"Calculated Min-Max. Min: {self.min_val:.4f}, Max: {self.max_val:.4f}, Range: {self.range_val:.4f}")
            self.normalization_params = (self.min_val, self.range_val)
        else:
            self.min_val, self.range_val = normalization_params
            self.normalization_params = normalization_params
            print(f"Using Provided Min-Max. Min: {self.min_val:.4f}, Range: {self.range_val:.4f}")
            
        self.csi_data_normalized = (csi_data_reshaped - self.min_val) / self.range_val
        
        if max_samples is not None:
            self.csi_data = self.csi_data_normalized[:max_samples]
            print(f"--- Test Data limited to {max_samples} samples ---")
        else:
            self.csi_data = self.csi_data_normalized
            
    def __len__(self): return self.csi_data.shape[0]
    def __getitem__(self, idx): return torch.from_numpy(self.csi_data[idx])

class AverageMeter:
    def __init__(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

# ----------------------------------------------------
# Metrics: NMSE (dB) and Rho (cross-correlation)
# NMSE = 10*log10(||X - X_hat||^2 / ||X||^2)
# Rho = frequency-domain cross-correlation (Section VI-A)
# ----------------------------------------------------
def calculate_nmse_db(original_norm_0_1, reconstructed_norm_0_1, normalization_params):
    min_val, range_val = normalization_params
    original_data = (original_norm_0_1 * range_val) + min_val
    reconstructed_data = (reconstructed_norm_0_1 * range_val) + min_val
    
    original_centered = original_data - 0.5
    reconstructed_centered = reconstructed_data - 0.5
    
    original_power = torch.sum(original_centered**2, dim=[1, 2, 3])
    mse = torch.sum((original_centered - reconstructed_centered)**2, dim=[1, 2, 3])
    
    valid_indices = original_power > 1e-8
    if torch.sum(valid_indices) == 0: return torch.tensor(-100.0)
    
    nmse = torch.mean(mse[valid_indices] / original_power[valid_indices])
    if nmse.item() <= 1e-10: return torch.tensor(-100.0)
    
    nmse_db = 10 * torch.log10(nmse)
    return nmse_db

def calculate_rho(original_norm_0_1, reconstructed_norm_0_1, normalization_params):
    min_val, range_val = normalization_params
    original_data = (original_norm_0_1 * range_val) + min_val
    reconstructed_data = (reconstructed_norm_0_1 * range_val) + min_val
    
    original_flat = original_data.reshape(original_data.shape[0], -1)
    reconstructed_flat = reconstructed_data.reshape(reconstructed_data.shape[0], -1)
    
    x_mean = original_flat.mean(dim=1, keepdim=True)
    y_mean = reconstructed_flat.mean(dim=1, keepdim=True)
    x_centered = original_flat - x_mean
    y_centered = reconstructed_flat - y_mean
    
    cov = (x_centered * y_centered).sum(dim=1)
    x_std = torch.sqrt((x_centered**2).sum(dim=1))
    y_std = torch.sqrt((y_centered**2).sum(dim=1))
    
    rho_batch = cov / (x_std * y_std + 1e-8) 
    return rho_batch.mean().item()

def calculate_metrics(original_norm_0_1, reconstructed_norm_0_1, normalization_params):
    nmse_db = calculate_nmse_db(original_norm_0_1, reconstructed_norm_0_1, normalization_params).item()
    rho = calculate_rho(original_norm_0_1, reconstructed_norm_0_1, normalization_params)
    return nmse_db, rho

# ----------------------------------------------------
# Parse Arguments
# ----------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Asymmetric CSI Feedback AE")
    
    parser.add_argument("--test_path", type=str, default="data/DATA_Htestin.mat")
    parser.add_argument("--test_key", type=str, default="HT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth.tar file")
    
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--depths", nargs='+', type=int, default=[2, 2, 9, 2])
    parser.add_argument("--chunk_sizes", nargs='+', type=int, default=[8, 8, 4])
    parser.add_argument("--scan_mode", type=str, default="chunked_ss2d")
    
    # Beam mask and compression mode options
    parser.add_argument("--use_beam_mask", action="store_true", help="Enable Soft Beam Masking")
    parser.add_argument("--compression_mode", type=str, default="default", choices=["default", "hybrid"])
    parser.add_argument("--beam_rank_K", type=int, default=4, help="Beam rank K for hybrid mode")
    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--max_test_samples", type=int, default=None)
    
    # Normalization Params (Optional manual override)
    parser.add_argument("--norm_min", type=float, default=None)
    parser.add_argument("--norm_range", type=float, default=None)
    
    return parser.parse_args()

# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"--- Evaluation Mode: {device} ---")
    print(f"[INFO] Loading Checkpoint: {args.checkpoint}")
    print(f"[INFO] Model Config: N={args.N}, M={args.M}, Mode={args.scan_mode}")
    print(f"[INFO] Compression: {args.compression_mode.upper()} (K={args.beam_rank_K})")
    print(f"[INFO] Beam Masking: {args.use_beam_mask}")

    # 1. Dataset Load
    norm_params = None
    if args.norm_min is not None and args.norm_range is not None:
        norm_params = (args.norm_min, args.norm_range)

    test_dataset = CsiDataset(args.test_path, args.test_key, max_samples=args.max_test_samples, normalization_params=norm_params)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 2. Model Build
    print(f"Instantiating MambaAE (Chunks: {args.chunk_sizes})...")
    net = MambaAE(
        depths=args.depths, N=args.N, M=args.M, bits=args.bits,
        chunk_sizes=args.chunk_sizes,
        scan_mode=args.scan_mode,
        use_beam_mask=args.use_beam_mask,       
        compression_mode=args.compression_mode, 
        beam_rank_K=args.beam_rank_K            
    ).to(device)
    
    # 3. Load Checkpoint (Cleanly)
    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # Handle DataParallel prefix & THOP artifact cleaning
    new_state_dict = {}
    for k, v in state_dict.items():
        # Filter out thop artifacts
        if "total_ops" in k or "total_params" in k:
            continue
        # Handle module prefix
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    try:
        net.load_state_dict(new_state_dict, strict=True)
        print("[INFO] Weights loaded successfully (strict=True).")
    except RuntimeError as e:
        print(f"[WARN] Strict loading failed. Retrying with strict=False.\nError: {e}")
        net.load_state_dict(new_state_dict, strict=False)

    # 4. FLOPs Calculation
    if profile:
        dummy_input = torch.randn(1, 2, 32, 32).to(device)
        # Quantizer disable for profiling if needed
        if hasattr(net, 'disable_quant'): net.disable_quant = False
        
        print("Calculating FLOPs and Parameters...")
        macs, params = profile(net, inputs=(dummy_input, ), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"[INFO] Total Parameters: {params}")
        print(f"[INFO] Total FLOPs (MACs): {macs}")
        print("-------------------------------------------------------")
    else:
        print("Warning: 'thop' not installed. Skipping FLOPs/Params calculation.")

    # 5. Inference Loop
    net.eval()
    nmse_meter = AverageMeter()
    rho_meter = AverageMeter()
    
    # Warm-up Quantizer off
    if hasattr(net, 'module'): net.module.disable_quant = False
    else: net.disable_quant = False

    print(f"--- Starting Evaluation on {len(test_dataset)} samples ---")
    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc="Evaluating")
        for i, d in enumerate(pbar):
            d = d.to(device)
            
            # Forward
            with autocast():
                x_hat = net(d)
            
            # Calculate NMSE (dB) and Rho
            nmse_db, rho = calculate_metrics(d.cpu(), x_hat.cpu(), test_dataset.normalization_params)
            
            # Update Meters
            nmse_meter.update(nmse_db, d.size(0))
            rho_meter.update(rho, d.size(0))
            
            pbar.set_postfix(NMSE_dB=f"{nmse_meter.avg:.4f}", Rho=f"{rho_meter.avg:.4f}")
            
    print("\n========================================")
    print(f"FINAL RESULTS:")
    print(f"  Test NMSE: {nmse_meter.avg:.4f} dB")
    print(f"  Test Rho : {rho_meter.avg:.4f}")
    print("========================================")

if __name__ == "__main__":
    main()
    