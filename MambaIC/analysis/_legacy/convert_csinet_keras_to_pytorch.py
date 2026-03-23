"""
Convert CsiNet Keras .h5 weights to PyTorch ModularAE state_dict.

Keras model (channels_first):
  conv2d_1(2→2, 3×3) → BN_1 → LeakyReLU → Flatten → dense_1(2048→512)
  dense_2(512→2048) → Reshape → [RefineBlock×2] → conv2d_8(2→2, sigmoid)

PyTorch ModularAE CsiNetEncoder + CsiNetDecoder:
  encoder.conv(2→2) → encoder.bn → encoder.act → encoder.fc(2048→512)
  decoder.fc_dec(512→2048) → decoder.refine.{0,1}.seq → decoder.final_conv → sigmoid

Usage:
  python analysis/convert_csinet_keras_to_pytorch.py
  # Or on Colab: !python analysis/convert_csinet_keras_to_pytorch.py
"""
import os, sys
import numpy as np
import h5py
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Use channels_last version — BN axis=-1 = channels (correct)
KERAS_H5 = os.path.join(
    os.path.dirname(PROJECT_ROOT), "baselines",
    "Python_CsiNet-master", "channels_last", "keras", "model_CsiNet_outdoor_dim512.h5")
OUT_PTH = os.path.join(
    PROJECT_ROOT, "saved_models", "csinet_csinet_dim512", "best.pth")


def load_keras_weights(h5_path):
    """Extract all weights from Keras .h5 file as numpy arrays."""
    weights = {}
    with h5py.File(h5_path, "r") as f:
        if "model_weights" in f:
            root = f["model_weights"]
        else:
            root = f

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)

        root.visititems(visitor)
    return weights


def convert():
    print(f"Keras H5: {KERAS_H5}")
    assert os.path.exists(KERAS_H5), f"Not found: {KERAS_H5}"

    kw = load_keras_weights(KERAS_H5)
    print(f"Loaded {len(kw)} weight arrays from Keras")

    # Print all keys for debugging
    for k, v in sorted(kw.items()):
        print(f"  {k}: {v.shape}")

    # Helper to find weight by partial key match
    def find(pattern):
        matches = [k for k in kw if pattern in k]
        assert len(matches) == 1, f"Pattern '{pattern}' matched {len(matches)}: {matches}"
        return kw[matches[0]]

    # Build PyTorch state_dict
    sd = {}

    # Keras conv kernel: (H, W, in_ch, out_ch) → PyTorch: (out_ch, in_ch, H, W)
    def conv_w(keras_kernel):
        return torch.from_numpy(keras_kernel).float().permute(3, 2, 0, 1).contiguous()

    # Helper for BN
    def bn_params(keras_name, pt_prefix):
        sd[f"{pt_prefix}.weight"] = torch.from_numpy(find(f"{keras_name}/gamma")).float()
        sd[f"{pt_prefix}.bias"] = torch.from_numpy(find(f"{keras_name}/beta")).float()
        sd[f"{pt_prefix}.running_mean"] = torch.from_numpy(find(f"{keras_name}/moving_mean")).float()
        sd[f"{pt_prefix}.running_var"] = torch.from_numpy(find(f"{keras_name}/moving_variance")).float()
        sd[f"{pt_prefix}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    # --- Encoder ---
    # conv2d → encoder.conv
    sd["encoder.conv.weight"] = conv_w(find("conv2d/kernel"))
    sd["encoder.conv.bias"] = torch.from_numpy(find("conv2d/bias")).float()

    # batch_normalization → encoder.bn
    bn_params("batch_normalization", "encoder.bn")

    # Flatten permutation: channels_last (H,W,C)=(32,32,2) → channels_first (C,H,W)=(2,32,32)
    # perm[i_cl] = i_cf: maps channels_last flat index to channels_first flat index
    # inv_perm[i_cf] = i_cl: maps channels_first flat index to channels_last flat index
    perm = [0] * 2048
    for h in range(32):
        for w in range(32):
            for c in range(2):
                i_cl = h * 64 + w * 2 + c      # channels_last: h*W*C + w*C + c
                i_cf = c * 1024 + h * 32 + w    # channels_first: c*H*W + h*W + w
                perm[i_cl] = i_cf
    # Inverse: for each channels_first index, find the channels_last index
    inv_perm = [0] * 2048
    for i, p in enumerate(perm):
        inv_perm[p] = i
    inv_perm = torch.tensor(inv_perm, dtype=torch.long)

    # dense → encoder.fc  (Keras Dense kernel: [in, out] → PyTorch Linear: [out, in])
    # Keras: y = x_cl @ W_keras, PyTorch: y = W_pt @ x_cf
    # W_pt[j, i_cf] = W_keras[cf_to_cl[i_cf], j] = W_keras[inv_perm[i_cf], j]
    enc_fc = torch.from_numpy(find("dense/kernel")).float().T  # (512, 2048) in cl order
    sd["encoder.fc.weight"] = enc_fc[:, inv_perm]  # reorder columns: cl→cf
    sd["encoder.fc.bias"] = torch.from_numpy(find("dense/bias")).float()

    # --- Decoder ---
    # dense_1 → decoder.fc_dec
    # W_pt[i_cf, :] = W_keras_T[cf_to_cl[i_cf], :] = W_keras_T[inv_perm[i_cf], :]
    dec_fc = torch.from_numpy(find("dense_1/kernel")).float().T  # (2048, 512) in cl order
    sd["decoder.fc_dec.weight"] = dec_fc[inv_perm, :]  # reorder rows: cl→cf
    sd["decoder.fc_dec.bias"] = torch.from_numpy(find("dense_1/bias")).float()[inv_perm]

    # channels_last layer naming: conv2d_{1..7}, batch_normalization_{1..6}
    # RefineBlock 0: conv2d_1(2→8)→BN_1 → conv2d_2(8→16)→BN_2 → conv2d_3(16→2)→BN_3
    # RefineBlock 1: conv2d_4(2→8)→BN_4 → conv2d_5(8→16)→BN_5 → conv2d_6(16→2)→BN_6
    refine_map = [
        ("conv2d_1", "batch_normalization_1", "decoder.refine.0.seq.0", "decoder.refine.0.seq.1"),
        ("conv2d_2", "batch_normalization_2", "decoder.refine.0.seq.3", "decoder.refine.0.seq.4"),
        ("conv2d_3", "batch_normalization_3", "decoder.refine.0.seq.6", "decoder.refine.0.seq.7"),
        ("conv2d_4", "batch_normalization_4", "decoder.refine.1.seq.0", "decoder.refine.1.seq.1"),
        ("conv2d_5", "batch_normalization_5", "decoder.refine.1.seq.3", "decoder.refine.1.seq.4"),
        ("conv2d_6", "batch_normalization_6", "decoder.refine.1.seq.6", "decoder.refine.1.seq.7"),
    ]

    for k_conv, k_bn, pt_conv, pt_bn in refine_map:
        sd[f"{pt_conv}.weight"] = conv_w(find(f"{k_conv}/kernel"))
        sd[f"{pt_conv}.bias"] = torch.from_numpy(find(f"{k_conv}/bias")).float()
        bn_params(k_bn, pt_bn)

    # conv2d_7 → decoder.final_conv (sigmoid activation is separate in PyTorch)
    sd["decoder.final_conv.weight"] = conv_w(find("conv2d_7/kernel"))
    sd["decoder.final_conv.bias"] = torch.from_numpy(find("conv2d_7/bias")).float()

    # --- Validate shapes against PyTorch model ---
    from ModularModels import ModularAE
    model = ModularAE(encoder_type='csinet', decoder_type='csinet',
                       encoded_dim=512, M=32)
    model_sd = model.state_dict()

    print(f"\n=== Shape validation ===")
    ok = True
    for k in model_sd:
        if k in sd:
            if sd[k].shape != model_sd[k].shape:
                print(f"  MISMATCH {k}: keras={sd[k].shape} vs pytorch={model_sd[k].shape}")
                ok = False
            else:
                print(f"  OK {k}: {sd[k].shape}")
        else:
            print(f"  MISSING {k}")
            ok = False

    if not ok:
        print("\n[ERROR] Shape mismatches found!")
        return

    # Save (workaround for Windows Korean path encoding)
    os.makedirs(os.path.dirname(OUT_PTH), exist_ok=True)
    import io, shutil
    buf = io.BytesIO()
    torch.save({"state_dict": sd}, buf)
    with open(OUT_PTH, "wb") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {OUT_PTH}")
    print(f"Keys: {len(sd)}")

    # Quick test
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing: {missing}")
    if unexpected:
        print(f"Unexpected: {unexpected}")
    if not missing and not unexpected:
        print("Load test: PASS")


if __name__ == "__main__":
    convert()
