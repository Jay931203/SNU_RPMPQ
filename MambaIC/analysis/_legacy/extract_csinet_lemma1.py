"""
extract_csinet_lemma1.py
========================
CsiNet W8 per-position conv output error (CPU, Keras).
No GPU needed — runs locally.

Model loading: build_model() + load_weights(by_name=True)
  (same approach as csinet_onlytest.py)

Output:
  results/csv/lemma1_csinet_w8.csv   (1024 spatial positions in raster order)
"""

import os, sys
import numpy as np
import scipy.io as sio

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSINET_DIR   = os.path.join(SCRIPT_DIR, "..", "Python_CsiNet-master")
DATA_PATH    = os.path.join(SCRIPT_DIR, "data", "DATA_Htestout.mat")
WEIGHTS_PATH = os.path.join(CSINET_DIR, "saved_model",
                            "model_CsiNet_outdoor_dim512.h5")
OUT_DIR = os.path.join(SCRIPT_DIR, "results", "csv")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES  = 1000
BATCH_SIZE = 50
ENCODED_DIM = 512


def quant_np(arr, bits=8):
    q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
    a_min, a_max = arr.min(), arr.max()
    if a_max == a_min:
        return arr
    scale = (a_max - a_min) / (q_max - q_min)
    zp = np.round(q_min - a_min / scale)
    w_q = np.round(arr / scale + zp).clip(q_min, q_max)
    return (w_q - zp) * scale


def main():
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
        LeakyReLU, Dense, Reshape, Add, Layer)
    from tensorflow.keras.models import Model

    print(f"TF version: {tf.__version__}")

    # ── Custom Keras layer (matches csinet_onlytest.py) ───────────────
    class QuantizationLayer(Layer):
        def __init__(self, bits=32, **kw):
            super().__init__(**kw)
            self.bits = bits
        def call(self, x):
            return x   # FP32 activation (act_bits=32)
        def get_config(self):
            cfg = super().get_config()
            cfg.update({"bits": self.bits})
            return cfg

    # ── Build CsiNet architecture (channels_first) ────────────────────
    img_ch, img_h, img_w = 2, 32, 32
    enc_dim  = ENCODED_DIM
    res_num  = 2
    df, bn_axis = 'channels_first', 3

    inp = Input((img_ch, img_h, img_w), name='input_1')

    ec  = Conv2D(2, (3, 3), padding='same', data_format=df, name='conv2d_1')
    eb  = BatchNormalization(axis=bn_axis, name='batch_normalization_1')
    el  = LeakyReLU(alpha=0.3, name='leaky_re_lu_1')
    er  = Reshape((img_ch * img_h * img_w,), name='reshape_1')
    ed  = Dense(enc_dim, activation='linear', name='dense_1')

    x = ec(inp)
    x = eb(x)
    x = el(x)
    x = QuantizationLayer(32, name='act_quant_1')(x)
    x = er(x)
    enc_out = ed(x)

    dd = Dense(img_ch * img_h * img_w, activation='linear', name='dense_2')
    dr = Reshape((img_ch, img_h, img_w), name='reshape_2')
    res_layers = []
    for i in range(res_num):
        idx1, idx2, idx3 = 2 + i * 3, 3 + i * 3, 4 + i * 3
        res_layers.append({
            'c1': Conv2D(8,  (3,3), padding='same', data_format=df, name=f'conv2d_{idx1}'),
            'b1': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx1}'),
            'l1': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx1}'),
            'c2': Conv2D(16, (3,3), padding='same', data_format=df, name=f'conv2d_{idx2}'),
            'b2': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx2}'),
            'l2': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx2}'),
            'c3': Conv2D(2,  (3,3), padding='same', data_format=df, name=f'conv2d_{idx3}'),
            'b3': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx3}'),
            'add': Add(name=f'add_{i+1}'),
            'lout': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx3}_out'),
        })
    dfinal = Conv2D(2, (3, 3), padding='same', activation='sigmoid',
                    data_format=df, name='conv2d_8')

    x = dd(enc_out)
    x = QuantizationLayer(32, name='act_quant_dec_dense')(x)
    x = dr(x)
    for i, b in enumerate(res_layers):
        s = x
        x = b['c1'](x); x = b['b1'](x); x = b['l1'](x); x = QuantizationLayer(32)(x)
        x = b['c2'](x); x = b['b2'](x); x = b['l2'](x); x = QuantizationLayer(32)(x)
        x = b['c3'](x); x = b['b3'](x)
        x = b['add']([s, x]); x = b['lout'](x); x = QuantizationLayer(32)(x)
    out = dfinal(x)

    ae_model = Model(inp, out, name='Autoencoder')

    # ── Load pre-trained weights ──────────────────────────────────────
    print(f"Loading weights: {WEIGHTS_PATH}")
    ae_model.load_weights(WEIGHTS_PATH, by_name=True)
    print("  Weights loaded OK")

    # ── Conv sub-model: input → conv2d_1 output ───────────────────────
    conv_out = ae_model.get_layer('conv2d_1').output   # (N, 2, 32, 32)
    conv_model = Model(inputs=ae_model.input, outputs=conv_out)

    # ── Load test CSI ─────────────────────────────────────────────────
    print(f"Loading data: {DATA_PATH}")
    raw  = sio.loadmat(DATA_PATH)["HT"].astype(np.float32)
    data = raw.reshape(-1, 2, 32, 32)[:N_SAMPLES] - 0.5  # (N, 2, 32, 32)
    # Keras channels_first model → input as-is (N, 2, 32, 32)
    print(f"  Samples: {data.shape[0]}")

    # ── FP32 conv output ──────────────────────────────────────────────
    print("Running FP32 inference (conv only)...")
    fp32_out = conv_model.predict(data, batch_size=BATCH_SIZE, verbose=0)
    # fp32_out: (N, 2, 32, 32) — channels_first

    # ── W8 quantized conv weights ─────────────────────────────────────
    conv_layer = ae_model.get_layer('conv2d_1')
    w_fp32, b = conv_layer.get_weights()    # w: (3, 3, 2, 2)
    print(f"  Conv kernel shape: {w_fp32.shape}")
    print(f"  Weight range (FP32): [{w_fp32.min():.5f}, {w_fp32.max():.5f}]")

    # Run for W2, W4, W8
    results = {}
    fp32_base_err = np.zeros(32 * 32)   # baseline (no quant → near zero)
    for bits in [8, 4, 2]:
        w_q = quant_np(w_fp32, bits=bits)
        max_err = np.abs(w_fp32 - w_q).max()
        conv_layer.set_weights([w_q, b])

        q_out = conv_model.predict(data, batch_size=BATCH_SIZE, verbose=0)
        # q_out: (N, 2, 32, 32)

        # Per-position error: L2 over 2 channels, mean over N
        diff   = fp32_out - q_out               # (N, 2, 32, 32)
        err_2d = np.linalg.norm(diff, axis=1)   # (N, 32, 32) L2 over channels (axis=1 = channels_first)
        err_mean = err_2d.mean(axis=0)          # (32, 32) mean over batch
        err_flat = err_mean.reshape(-1)         # (1024,) raster order

        print(f"  W{bits}: weight_quant_err={max_err:.5f}, "
              f"mean_pos_err={err_flat.mean():.6f}, max={err_flat.max():.6f}")
        results[bits] = err_flat

    # ── Save W8 results ───────────────────────────────────────────────
    positions = np.arange(32 * 32)
    out_arr = np.column_stack([positions, results[8], results[4], results[2]])
    path = os.path.join(OUT_DIR, "lemma1_csinet_w8.csv")
    np.savetxt(path, out_arr, delimiter=",",
               header="position,csinet_w8,csinet_w4,csinet_w2", comments="")
    print(f"\nSaved: {path}")

    # ── Statistical summary ───────────────────────────────────────────
    print("\n=== CsiNet per-position conv error (W8) ===")
    print(f"  1024 positions (32x32 raster), 2 output channels")
    print(f"  mean  = {results[8].mean():.6f}")
    print(f"  std   = {results[8].std():.6f}")
    print(f"  Range = [{results[8].min():.6f}, {results[8].max():.6f}]")
    print("  Note: CsiNet conv is 3x3 local — errors are SPATIALLY UNIFORM")
    print("        (no sequential accumulation unlike Mamba SSM)")

    print("\nDone. Copy lemma1_csinet_w8.csv to results/csv/ on Colab")
    print("Colab will merge with Mamba W8 data -> lemma1_output_errors.csv")


if __name__ == "__main__":
    main()
