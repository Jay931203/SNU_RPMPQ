"""
Directional Rounding for Mixed-Precision Quantization.

Standard quantization uses nearest rounding. Directional rounding chooses
floor/ceil per weight element to preserve encoder output DIRECTION (cosine)
or reconstruction quality (NMSE).

Usage (Colab):
    !python analysis/directional_rounding.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train_ae import (
    apply_precision_policy, quantize_feedback_torch,
    calculate_su_miso_rate_mrt, CsiDataset,
)
from ModularModels import ModularAE
from rpmpq_v2 import get_encoder_block_names, RESULTS_CSV

os.makedirs(RESULTS_CSV, exist_ok=True)


def load_model_and_data(n_cal=1000):
    """Load model, test data, and calibration subset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    train_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htrainout.mat"), "HT")
    test_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htestout.mat"), "HT",
        normalization_params=train_set.normalization_params)

    # Calibration subset (smaller for gradient computation)
    cal_indices = list(range(min(n_cal, len(test_set))))
    cal_set = Subset(test_set, cal_indices)
    cal_loader = DataLoader(cal_set, batch_size=128, shuffle=False, num_workers=0)
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

    return net, cal_loader, test_loader, norm_params, device


def get_reference_output(model, loader, device):
    """Get encoder output with original FP32 weights."""
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    outputs = []
    inputs = []
    with torch.no_grad():
        for batch in loader:
            d = batch.to(device)
            y = real_model.encoder(d)
            outputs.append(y.detach())
            inputs.append(d.detach())
    return torch.cat(inputs, 0), torch.cat(outputs, 0)


def directional_quantize_block(model, block_name, bits, cal_input, y_ref,
                                device, objective="cosine"):
    """Compute directionally-rounded weights for one block.

    Args:
        model: the full model
        block_name: which block to quantize
        bits: target bit-width
        cal_input: calibration input tensor (N, 2, 32, 32)
        y_ref: reference encoder output (N, C, H, W)
        objective: "cosine" or "nmse"

    Returns:
        rounded_weight: optimally rounded weight tensor
    """
    real_model = model.module if isinstance(model, nn.DataParallel) else model

    # Find the module
    target_module = None
    for name, module in real_model.encoder.named_modules():
        clean = name.replace("encoder.", "").replace("module.", "")
        if clean == block_name or name == block_name:
            if hasattr(module, 'weight') and module.weight is not None:
                target_module = module
                break

    if target_module is None:
        # Handle fc_part names
        import re
        match = re.search(r'(.+)_part(\d+)$', block_name)
        if match:
            base_name = match.group(1)
            part_idx = int(match.group(2))
            for name, module in real_model.encoder.named_modules():
                clean = name.replace("encoder.", "").replace("module.", "")
                if clean == base_name or name == base_name:
                    if hasattr(module, 'weight') and module.weight is not None:
                        target_module = module
                        break
            if target_module is None:
                return None
            # For FC parts, we only round the specific chunk
            W = target_module.weight.data
            chunk_size = W.shape[0] // 32
            start = part_idx * chunk_size
            end = start + chunk_size
            W_chunk = W[start:end].clone()

            return _directional_round_weight(
                real_model, target_module, W_chunk, bits,
                cal_input, y_ref, device, objective,
                is_chunk=True, chunk_start=start, chunk_end=end)
        return None

    W = target_module.weight.data.clone()
    return _directional_round_weight(
        real_model, target_module, W, bits,
        cal_input, y_ref, device, objective)


def _directional_round_weight(real_model, target_module, W, bits,
                               cal_input, y_ref, device, objective,
                               is_chunk=False, chunk_start=0, chunk_end=0):
    """Core directional rounding logic."""
    original_weight = target_module.weight.data.clone()

    # Quantization grid
    w_min = W.min().item()
    w_max = W.max().item()
    n_levels = 2**bits - 1
    if n_levels == 0:
        return W
    scale = (w_max - w_min) / n_levels
    if scale < 1e-10:
        return W

    # Floor/ceil
    W_norm = (W - w_min) / scale
    W_floor_int = torch.floor(W_norm).clamp(0, n_levels)
    W_floor = W_floor_int * scale + w_min
    W_ceil = (W_floor_int + 1).clamp(0, n_levels) * scale + w_min

    # Fractional part as differentiable variable
    frac = (W_norm - W_floor_int).clamp(0, 1).detach().clone().requires_grad_(True)

    # Forward with soft-rounded weight
    W_soft = W_floor + frac * scale

    if is_chunk:
        new_weight = original_weight.clone()
        new_weight[chunk_start:chunk_end] = W_soft
        target_module.weight = nn.Parameter(new_weight)
    else:
        target_module.weight = nn.Parameter(W_soft)

    # Forward
    y_hat = real_model.encoder(cal_input)

    # Loss
    B = y_hat.shape[0]
    if objective == "cosine":
        loss = -F.cosine_similarity(
            y_hat.reshape(B, -1), y_ref.reshape(B, -1), dim=1).mean()
    else:  # nmse
        loss = F.mse_loss(y_hat, y_ref)

    # Backward
    loss.backward()

    if frac.grad is None:
        # Fallback to nearest rounding
        target_module.weight = nn.Parameter(original_weight)
        return torch.where(W_norm - W_floor_int >= 0.5, W_ceil, W_floor)

    # Decision: grad < 0 means rounding up helps the objective
    round_up = frac.grad < 0
    W_rounded = torch.where(round_up, W_ceil, W_floor)

    # Restore original weight
    target_module.weight = nn.Parameter(original_weight)
    real_model.zero_grad()

    return W_rounded.detach()


def apply_directional_policy(model, policy, rounded_weights_cache, device):
    """Apply policy using pre-computed directionally-rounded weights."""
    import re
    real_model = model.module if isinstance(model, nn.DataParallel) else model

    # Group fc_parts
    policy_groups = {}
    for p_key, bits in policy.items():
        clean = p_key.replace("encoder.", "").replace("module.", "").replace(".weight", "")
        match = re.search(r'(.+)_part(\d+)$', clean)
        if match:
            base = match.group(1)
            idx = int(match.group(2))
            if base not in policy_groups:
                policy_groups[base] = {}
            policy_groups[base][idx] = (bits, p_key)
        else:
            policy_groups[clean] = (bits, p_key)

    for name, module in real_model.encoder.named_modules():
        if not (hasattr(module, 'weight') and module.weight is not None):
            continue
        clean = name.replace("encoder.", "").replace("module.", "")

        if clean in policy_groups:
            info = policy_groups[clean]
            if isinstance(info, dict):
                # FC with chunks
                W = module.weight.data.clone()
                chunk_size = W.shape[0] // 32
                for part_idx, (bits, p_key) in info.items():
                    key = (p_key, bits)
                    start = part_idx * chunk_size
                    end = start + chunk_size
                    if key in rounded_weights_cache:
                        W[start:end] = rounded_weights_cache[key]
                    else:
                        # Fallback to standard quantization
                        from train_ae import quantize_int_asym
                        if bits < 32:
                            W[start:end] = quantize_int_asym(W[start:end], bits)
                module.weight.data = W
            else:
                bits, p_key = info
                key = (p_key, bits)
                if key in rounded_weights_cache:
                    module.weight.data = rounded_weights_cache[key]
                else:
                    from train_ae import quantize_int_asym
                    if bits < 32:
                        module.weight.data = quantize_int_asym(module.weight.data, bits)


def evaluate_policy(model, test_loader, norm_params, device, aq_bits=8):
    """Evaluate current model state: NMSE, rate, cos^2 theta."""
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    min_val, range_val = norm_params
    snr = 20

    nmse_all = []
    rate_all = []

    # Load perfect rates
    perf_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv")
    r_ref = pd.read_csv(perf_csv)[f"r_perf_{snr}"].values

    with torch.no_grad():
        for batch in test_loader:
            d = batch.to(device)
            z = real_model.encoder(d)
            if aq_bits > 0:
                z = quantize_feedback_torch(z, aq_bits)
            x_hat = real_model.decoder(z)

            h_true = (d * range_val) + min_val - 0.5
            h_hat = (x_hat * range_val) + min_val - 0.5

            error = torch.sum((h_true - h_hat)**2, dim=[1, 2, 3])
            power = torch.sum(h_true**2, dim=[1, 2, 3])
            nmse = (error / (power + 1e-9)).cpu().numpy()
            nmse_all.extend(nmse.tolist())

            r = calculate_su_miso_rate_mrt(h_true, h_hat, snr, device)
            rate_all.extend(r.cpu().numpy().tolist())

    nmse_arr = np.array(nmse_all)
    rate_arr = np.array(rate_all)
    cos2 = np.clip((2**rate_arr - 1) / (2**r_ref[:len(rate_arr)] - 1 + 1e-12), 0, 1)

    return {
        "nmse_db": 10 * np.log10(np.mean(nmse_arr) + 1e-15),
        "mean_cos2": np.mean(cos2),
        "mean_rate": np.mean(rate_arr),
        "outage_99": np.mean(rate_arr < 0.99 * r_ref[:len(rate_arr)]),
        "outage_95": np.mean(rate_arr < 0.95 * r_ref[:len(rate_arr)]),
    }


def main():
    print("=" * 70)
    print("  DIRECTIONAL ROUNDING TEST")
    print("  Standard vs NMSE-directed vs Cosine-directed")
    print("=" * 70)

    net, cal_loader, test_loader, norm_params, device = load_model_and_data(n_cal=500)
    block_names = get_encoder_block_names(net, fc_chunks=32)
    fc_blocks = [b for b in block_names if "fc_part" in b]

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    # Get reference output on calibration data
    print("\n[1] Computing reference encoder output...")
    cal_input, y_ref = get_reference_output(net, cal_loader, device)
    print(f"  Calibration: {cal_input.shape[0]} samples")
    print(f"  Encoder output: {y_ref.shape}")

    # ============================================================
    # TEST 1: Single-block comparison (all FC chunks at INT2)
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 1: Per-block directional rounding (FC chunks @ INT2)")
    print("=" * 70)

    # Pre-compute directional weights for a few FC chunks
    n_test_chunks = min(8, len(fc_blocks))  # test first 8 chunks
    test_chunks = fc_blocks[:n_test_chunks]

    for objective in ["cosine", "nmse"]:
        print(f"\n--- Objective: {objective} ---")
        for bname in tqdm(test_chunks, desc=f"Dir-round ({objective})"):
            real_model.load_state_dict(original_state)

            # Standard quantization
            std_policy = {bn: 16 for bn in block_names}
            std_policy[bname] = 2
            apply_precision_policy(net, std_policy, device)
            res_std = evaluate_policy(net, test_loader, norm_params, device)

            # Directional quantization
            real_model.load_state_dict(original_state)
            W_dir = directional_quantize_block(
                net, bname, 2, cal_input, y_ref, device, objective)

            if W_dir is not None:
                # Apply: set this chunk to directional-rounded, rest to INT16
                real_model.load_state_dict(original_state)
                # Apply standard INT16 to all other blocks first
                apply_precision_policy(net, {bn: 16 for bn in block_names}, device)
                # Then override this specific chunk with directional weights
                # (need to handle fc_part specially)
                import re
                match = re.search(r'(.+)_part(\d+)$', bname)
                if match:
                    base = match.group(1)
                    part_idx = int(match.group(2))
                    for name, module in real_model.encoder.named_modules():
                        clean = name.replace("encoder.", "").replace("module.", "")
                        if clean == base:
                            chunk_size = module.weight.shape[0] // 32
                            start = part_idx * chunk_size
                            end = start + chunk_size
                            module.weight.data[start:end] = W_dir.to(device)
                            break
                else:
                    for name, module in real_model.encoder.named_modules():
                        clean = name.replace("encoder.", "").replace("module.", "")
                        if clean == bname:
                            module.weight.data = W_dir.to(device)
                            break

                res_dir = evaluate_policy(net, test_loader, norm_params, device)

                d_nmse = res_dir["nmse_db"] - res_std["nmse_db"]
                d_cos2 = res_dir["mean_cos2"] - res_std["mean_cos2"]
                print(f"  {bname:20s}: ΔNMSE={d_nmse:+.4f}dB  Δcos²θ={d_cos2:+.6f}  "
                      f"(std: nmse={res_std['nmse_db']:.2f}, cos2={res_std['mean_cos2']:.6f})")

    # ============================================================
    # TEST 2: Multi-chunk with directional rounding
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 2: Multi-chunk (16 FC @ INT2) — Standard vs Directional")
    print("=" * 70)

    np.random.seed(42)
    n_trials = 10  # fewer trials since each requires grad computation

    results = []
    for trial in tqdm(range(n_trials), desc="Multi-chunk"):
        perm = np.random.permutation(len(fc_blocks))
        quant_chunks = [fc_blocks[i] for i in perm[:16]]

        policy = {bn: 16 for bn in block_names}
        for chunk in quant_chunks:
            policy[chunk] = 2

        # Standard quantization
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, policy, device)
        res_std = evaluate_policy(net, test_loader, norm_params, device)

        # Directional (cosine) — compute rounded weights for each chunk
        real_model.load_state_dict(original_state)
        for chunk in quant_chunks:
            W_dir = directional_quantize_block(
                net, chunk, 2, cal_input, y_ref, device, "cosine")
            if W_dir is not None:
                import re
                match = re.search(r'(.+)_part(\d+)$', chunk)
                if match:
                    base = match.group(1)
                    part_idx = int(match.group(2))
                    for name, module in real_model.encoder.named_modules():
                        clean = name.replace("encoder.", "").replace("module.", "")
                        if clean == base:
                            chunk_size = module.weight.shape[0] // 32
                            start = part_idx * chunk_size
                            end = start + chunk_size
                            module.weight.data[start:end] = W_dir.to(device)
                            break
        # Also apply INT16 quantization to non-quantized blocks
        apply_precision_policy(net, {bn: 16 for bn in block_names
                                      if bn not in quant_chunks}, device)
        res_dir = evaluate_policy(net, test_loader, norm_params, device)

        results.append({
            "trial": trial,
            "std_nmse": res_std["nmse_db"],
            "std_cos2": res_std["mean_cos2"],
            "std_rate": res_std["mean_rate"],
            "dir_nmse": res_dir["nmse_db"],
            "dir_cos2": res_dir["mean_cos2"],
            "dir_rate": res_dir["mean_rate"],
            "d_nmse": res_dir["nmse_db"] - res_std["nmse_db"],
            "d_cos2": res_dir["mean_cos2"] - res_std["mean_cos2"],
            "d_rate": res_dir["mean_rate"] - res_std["mean_rate"],
        })

        print(f"  Trial {trial}: ΔNMSE={results[-1]['d_nmse']:+.3f}dB  "
              f"Δcos²θ={results[-1]['d_cos2']:+.6f}  "
              f"Δrate={results[-1]['d_rate']:+.4f}")

    df = pd.DataFrame(results)
    print(f"\n  Summary ({n_trials} trials):")
    print(f"    ΔNMSE:  mean={df['d_nmse'].mean():+.3f}dB  std={df['d_nmse'].std():.3f}")
    print(f"    Δcos²θ: mean={df['d_cos2'].mean():+.6f}  std={df['d_cos2'].std():.6f}")
    print(f"    Δrate:  mean={df['d_rate'].mean():+.4f}  std={df['d_rate'].std():.4f}")

    out_csv = os.path.join(RESULTS_CSV, "directional_rounding_test.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # Restore
    real_model.load_state_dict(original_state)


if __name__ == "__main__":
    main()
