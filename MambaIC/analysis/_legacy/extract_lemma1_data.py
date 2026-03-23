"""
extract_lemma1_data.py
======================
Colab/GPU에서 실행하여 Lemma 1 Figure에 쓸 실제 모델 데이터 추출.

Usage (Colab):
  %cd /path/to/MambaIC
  !python extract_lemma1_data.py

Output:
  results/csv/lemma1_state_errors.csv   → Panel (a): per-position state error (W2/W4/W8/W16)
  results/csv/lemma1_output_errors.csv  → Panel (b): Mamba vs CsiNet W8 output error per position

방법:
  - Mamba g_a_stage2 출력 (N,128,8,8 → 64 positions) 에 forward hook
  - FP32 vs W{bits} 양자화 모델 비교
  - per-position 오차 = mean over (batch, channels) of L2 norm

측정 레이어: g_a_stage2 (스트라이드 2+2=4로 다운샘플 후 8×8=64 spatial positions)
"""

import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio

# ─── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from MambaAE import MambaAE

# ─── quantize_int_asym (copied from train_ae.py to avoid full import) ──
def quantize_int_asym(w, bits):
    q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
    w_min, w_max = w.min(), w.max()
    if w_max == w_min: return w
    scale = (w_max - w_min) / (q_max - q_min)
    zp = torch.round(q_min - w_min / scale)
    w_q = torch.round(w / scale + zp)
    w_q = torch.clamp(w_q, q_min, q_max)
    return (w_q - zp) * scale

# ─── Config ────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(SCRIPT_DIR,
    "saved_models/csi_mamba_AE_M32_bits0_chunked_ss2d_222_out/best.pth")
DATA_PATH  = os.path.join(SCRIPT_DIR, "data/DATA_Htestout.mat")
N_SAMPLES  = 1000
BATCH_SIZE = 50
BIT_WIDTHS = [2, 4, 8, 16]
OUT_DIR    = os.path.join(SCRIPT_DIR, "results", "csv")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# ─── Model loading ─────────────────────────────────────────────────────
def load_base_model():
    model = MambaAE(
        depths=[2, 2, 4, 2],
        drop_path_rate=0.1,
        N=128, M=32, bits=0,
        scan_mode="chunked_ss2d",
        compression_mode="default",
    )
    ckpt = torch.load(CHECKPOINT, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    # Strip DataParallel prefix if present
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


def quantize_encoder_weights(model, bits):
    """Deep copy model and quantize all encoder weights."""
    m = copy.deepcopy(model)
    # Quantize all layers in the encoder (conv + SSM projections)
    for name, module in m.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            if "g_a" in name or "g_s" not in name:  # encoder only
                with torch.no_grad():
                    module.weight.data = quantize_int_asym(module.weight.data, bits)
    return m


# ─── Feature extraction via forward hooks ──────────────────────────────
class StageHook:
    """Captures output of g_a_stage2 (shape: N, C, H, W)."""
    def __init__(self):
        self.output = None
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        self.output = out.detach().float()   # (N, C, H, W)


def get_stage2_features(model, x_batch):
    """Run encoder forward, return g_a_stage2 output (N, C, H, W)."""
    hook_obj = StageHook()
    handle = model.g_a_stage2.register_forward_hook(hook_obj.hook_fn)
    with torch.no_grad():
        _ = model.encoder(x_batch.to(device)) if hasattr(model, "encoder") \
            else model(x_batch.to(device))
    handle.remove()
    return hook_obj.output   # (N, C, H, W)


def encoder_forward(model, x):
    """
    Run encoder only — replicates model.f_theta() exactly:
      input_quant → stage1(conv1) → stage2(conv2) → stage3(conv3) → tanh
    Returns (z_latent, z_stage2) where z_stage2 is the 8×8 feature map.
    """
    z = model.input_quant(x)
    z = model.g_a_stage1(model.g_a_conv1(z))
    z_stage2 = model.g_a_stage2(model.g_a_conv2(z))   # (N, 128, 8, 8)
    z = model.g_a_stage3(model.g_a_conv3(z_stage2))
    z = model.latent_tanh(z)
    return z, z_stage2


# ─── Data loading ──────────────────────────────────────────────────────
def load_test_csi(path, n_samples):
    raw = sio.loadmat(path)["HT"].astype(np.float32)
    data = raw.reshape(-1, 2, 32, 32) - 0.5  # centre
    data = torch.from_numpy(data[:n_samples])
    print(f"[INFO] Loaded {data.shape[0]} test samples from {path}")
    return data


# ─── Panel (a): per-position state error (g_a_stage2 output) ───────────
def extract_panel_a(fp32_model, test_data):
    """
    For W = 2, 4, 8, 16: measure per-scan-position stage2 feature error.

    stage2 output shape: (N, C=128, H=8, W=8) → 64 spatial positions.
    Raster scan order: t = row*8 + col.

    Returns: dict {bits: per_pos_err (T=64,)}
    """
    T = 64  # 8×8 positions

    print("[Panel a] Extracting FP32 stage2 features...")
    fp32_feats = []  # list of (N, C, 8, 8)
    for i in range(0, len(test_data), BATCH_SIZE):
        batch = test_data[i:i+BATCH_SIZE]
        _, feat = encoder_forward(fp32_model, batch.to(device))
        fp32_feats.append(feat.cpu())
    fp32_feats = torch.cat(fp32_feats, dim=0)  # (N, C, 8, 8)

    results = {}
    for bits in BIT_WIDTHS:
        print(f"[Panel a] W{bits} quantization...")
        q_model = quantize_encoder_weights(fp32_model, bits)

        q_feats = []
        for i in range(0, len(test_data), BATCH_SIZE):
            batch = test_data[i:i+BATCH_SIZE]
            _, feat = encoder_forward(q_model, batch.to(device))
            q_feats.append(feat.cpu())
        q_feats = torch.cat(q_feats, dim=0)  # (N, C, 8, 8)

        # Per-position error: ||fp32 - q||_2 over C channels, mean over N samples
        diff = fp32_feats - q_feats          # (N, C, 8, 8)
        err  = diff.norm(dim=1)              # (N, 8, 8)  L2 over channels
        err  = err.mean(dim=0)               # (8, 8)  mean over batch
        err_flat = err.reshape(-1).numpy()   # (64,)  raster order

        results[bits] = err_flat
        print(f"  W{bits}: mean_err={err_flat.mean():.5f}, max={err_flat.max():.5f}")

    # Save CSV: columns = [position, W16, W8, W4, W2]
    T_arr = np.arange(T)
    rows = np.column_stack([T_arr] + [results[b] for b in BIT_WIDTHS])
    header = "position," + ",".join(f"W{b}" for b in BIT_WIDTHS)
    path = os.path.join(OUT_DIR, "lemma1_state_errors.csv")
    np.savetxt(path, rows, delimiter=",", header=header, comments="")
    print(f"[Panel a] Saved → {path}")
    return results


# ─── Panel (b): Mamba W8 vs CsiNet W8 per-position output error ────────
def extract_panel_b_mamba(fp32_model, test_data):
    """
    Mamba W8: per-scan-position stage2 error.
    (Same as extract_panel_a but only W8, returned separately.)
    """
    print("[Panel b] Mamba FP32 stage2 features...")
    fp32_feats = []
    for i in range(0, len(test_data), BATCH_SIZE):
        batch = test_data[i:i+BATCH_SIZE]
        _, feat = encoder_forward(fp32_model, batch.to(device))
        fp32_feats.append(feat.cpu())
    fp32_feats = torch.cat(fp32_feats, dim=0)

    q_model = quantize_encoder_weights(fp32_model, 8)
    q_feats = []
    print("[Panel b] Mamba W8 stage2 features...")
    for i in range(0, len(test_data), BATCH_SIZE):
        batch = test_data[i:i+BATCH_SIZE]
        _, feat = encoder_forward(q_model, batch.to(device))
        q_feats.append(feat.cpu())
    q_feats = torch.cat(q_feats, dim=0)

    diff = fp32_feats - q_feats
    err  = diff.norm(dim=1).mean(dim=0).reshape(-1).numpy()  # (64,)
    print(f"  Mamba W8: mean_err={err.mean():.5f}")
    return err


def extract_panel_b_csinet_keras(test_data_path, n_samples):
    """
    CsiNet W8: per-spatial-position conv output error.
    Runs with Keras (TF2) on CPU — no GPU needed.

    NOTE: CsiNet conv output at each spatial position is independent
    (3×3 receptive field, no sequential dependency).
    Expected: flat error across all 1024 spatial positions.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load
        import h5py
    except ImportError:
        print("[Panel b] TensorFlow not available, skipping CsiNet.")
        return None

    keras_path = os.path.join(SCRIPT_DIR, "..",
        "Python_CsiNet-master", "saved_model",
        "model_CsiNet_outdoor_dim512.h5")
    if not os.path.exists(keras_path):
        print(f"[Panel b] CsiNet model not found: {keras_path}")
        return None

    print("[Panel b] Loading CsiNet Keras model...")
    model = keras_load(keras_path, compile=False)

    # Load CSI data
    raw = sio.loadmat(test_data_path)["HT"].astype(np.float32)
    x = raw.reshape(-1, 2, 32, 32)[:n_samples] - 0.5   # (N, 2, 32, 32)
    x_tf = x.transpose(0, 2, 3, 1)  # channels_last: (N, 32, 32, 2)

    # Extract conv layer (first conv before flatten)
    # Name from csinet: 'conv2d_1' with kernel (3,3,2,2)
    conv_layer_name = None
    for layer in model.layers:
        if "conv2d" in layer.name:
            conv_layer_name = layer.name
            break
    if conv_layer_name is None:
        print("[Panel b] Could not find CsiNet conv layer.")
        return None

    # Build submodel: input → conv output
    from tensorflow.keras.models import Model
    conv_layer = model.get_layer(conv_layer_name)
    conv_model = Model(inputs=model.input, outputs=conv_layer.output)

    # FP32 conv output
    fp32_out = conv_model.predict(x_tf, batch_size=50, verbose=0)  # (N, 32, 32, 2)

    # Apply W8 quantization to conv weights
    w, b_k = conv_layer.get_weights()  # w: (3,3,2,2)

    def quant_np(arr, bits=8):
        q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
        a_min, a_max = arr.min(), arr.max()
        if a_max == a_min: return arr
        scale = (a_max - a_min) / (q_max - q_min)
        zp = np.round(q_min - a_min / scale)
        w_q = np.round(arr / scale + zp).clip(q_min, q_max)
        return (w_q - zp) * scale

    w_q = quant_np(w, bits=8)
    conv_layer.set_weights([w_q, b_k])

    q_out = conv_model.predict(x_tf, batch_size=50, verbose=0)  # (N, 32, 32, 2)

    # Per-position error: L2 over 2 channels, mean over N
    diff = fp32_out - q_out         # (N, 32, 32, 2)
    err  = np.linalg.norm(diff, axis=-1).mean(axis=0)  # (32, 32)
    err_flat = err.reshape(-1)      # 1024 positions (raster)

    print(f"  CsiNet W8: mean_err={err_flat.mean():.5f}, max={err_flat.max():.5f}")
    return err_flat


def save_panel_b(mamba_err, csinet_err):
    """Save Panel (b) data as CSV."""
    T_mamba = len(mamba_err)
    T_csinet = len(csinet_err) if csinet_err is not None else 0

    # Use the shorter length for fair comparison (64 vs 64 or 64 vs 1024)
    # We downsample CsiNet 1024 → 64 by averaging blocks of 16
    if csinet_err is not None and len(csinet_err) != T_mamba:
        block = len(csinet_err) // T_mamba
        csinet_err_ds = csinet_err[:T_mamba*block].reshape(T_mamba, block).mean(axis=1)
        print(f"[Panel b] Downsampled CsiNet {len(csinet_err)} → {T_mamba} positions")
    else:
        csinet_err_ds = csinet_err

    rows = []
    for t in range(T_mamba):
        cs = csinet_err_ds[t] if csinet_err_ds is not None else np.nan
        rows.append([t, mamba_err[t], cs])

    rows = np.array(rows)
    header = "position,mamba_w8,csinet_w8"
    path = os.path.join(OUT_DIR, "lemma1_output_errors.csv")
    np.savetxt(path, rows, delimiter=",", header=header, comments="")
    print(f"[Panel b] Saved → {path}")


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Lemma 1 Data Extraction")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data not found: {DATA_PATH}")
        sys.exit(1)

    # Load base model and test data
    print("[INFO] Loading model...")
    fp32_model = load_base_model()
    test_data  = load_test_csi(DATA_PATH, N_SAMPLES)

    # Panel (a)
    print("\n[Panel a] Per-position stage2 error for W2/W4/W8/W16")
    extract_panel_a(fp32_model, test_data)

    # Panel (b) — Mamba side
    print("\n[Panel b] Mamba W8 per-position error")
    mamba_err = extract_panel_b_mamba(fp32_model, test_data)

    # Panel (b) — CsiNet side (Keras, CPU)
    print("\n[Panel b] CsiNet W8 per-position error (Keras)")
    csinet_err = extract_panel_b_csinet_keras(DATA_PATH, N_SAMPLES)

    save_panel_b(mamba_err, csinet_err)

    print("\n[DONE] Files saved to results/csv/")
    print("  → lemma1_state_errors.csv  (Panel a: W2/W4/W8/W16)")
    print("  → lemma1_output_errors.csv (Panel b: Mamba vs CsiNet at W8)")
    print("\nNow copy these CSVs and run plot_lemma1_comparison.py")


if __name__ == "__main__":
    main()
