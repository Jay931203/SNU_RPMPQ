import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# cuDNN DLL path for local GPU (pip install nvidia-cudnn-cu11)
_cudnn_bin = os.path.join(os.path.dirname(os.__file__), '..', 'site-packages', 'nvidia', 'cudnn', 'bin')
_cublas_bin = os.path.join(os.path.dirname(os.__file__), '..', 'site-packages', 'nvidia', 'cublas', 'bin')
for _p in [_cudnn_bin, _cublas_bin]:
    if os.path.isdir(_p):
        os.environ['PATH'] = os.path.abspath(_p) + ';' + os.environ.get('PATH', '')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, Add, LeakyReLU
from tensorflow.keras.models import Model
import scipy.io as sio 
import numpy as np
import pandas as pd
import math
import time
import argparse
import ast
import re
try:
    import pulp
except ImportError:
    pulp = None  # only needed for --analyze_all
from tqdm import tqdm
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

# =========================================================
# [1] Config
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='outdoor', choices=['indoor', 'outdoor'])
parser.add_argument('--analyze_all', action='store_true', help='Offline Pipeline')
parser.add_argument('--extend', action='store_true', help='Extend: run 95.1-97.0%% and append to existing CSV')
parser.add_argument('--wide_step', type=float, default=None, help='Wide sweep step size (e.g. 0.05). Range 85-98%%')
parser.add_argument('--run_online', action='store_true', help='Online Simulation')
parser.add_argument('--target_saving', type=float, default=75.0)
parser.add_argument('--lut_path', type=str, default=None)
parser.add_argument('--act_quant', type=int, default=32) # Sim시 16 권장
parser.add_argument('--aq', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--encoded_dim', type=int, default=512,
                    help='Latent dimension (512=CR1/4, 128=CR1/16, 64=CR1/32, 32=CR1/64)')
parser.add_argument('--weight_bits', type=int, default=32,
                    help='Uniform weight quantization bits (32=FP32)')

try: args = parser.parse_args()
except: args = parser.parse_args([])

ENV_MODE = args.env
ACT_BITS = args.act_quant
QUANTIZATION_BITS = args.aq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
data_dir = os.path.join(PROJECT_DIR, 'MambaIC', 'data') + os.sep
base_model_dir = os.path.join(SCRIPT_DIR, 'saved_model') + os.sep
img_height, img_width, img_channels = 32, 32, 2
encoded_dim = args.encoded_dim
residual_num = 2

# =========================================================
# [2] Mamba-Style Quantization Logic
# =========================================================
class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, bits, name=None, **kwargs):
        super(QuantizationLayer, self).__init__(name=name, **kwargs)
        self.bits = bits
    def call(self, inputs):
        if self.bits >= 32: return inputs
        levels = tf.cast(2**self.bits - 1, tf.float32)
        min_val, max_val = tf.reduce_min(inputs), tf.reduce_max(inputs)
        scale = (max_val - min_val) / (levels + 1e-9)
        return tf.round((inputs - min_val) / scale) * scale + min_val
    def get_config(self):
        config = super(QuantizationLayer, self).get_config(); config.update({'bits': self.bits}); return config

def quantize_int_asym_np(w, bits):
    """Mamba's Asymmetric Quantization Logic"""
    if bits >= 32: return w
    q_min, q_max = -(2**(bits-1)), (2**(bits-1))-1
    w_min, w_max = np.min(w), np.max(w)
    w_min, w_max = min(w_min, 0.0), max(w_max, 0.0) # Nudged Zero
    if w_max == w_min: return w
    scale = (w_max - w_min) / (q_max - q_min)
    zp = np.clip(np.round(q_min - w_min / scale), q_min, q_max)
    return (np.clip(np.round(w / scale + zp), q_min, q_max) - zp) * scale

def apply_mp_policy_strict(model, policy):
    """
    Applies MP policy to model layers. Handles _partN chunk splitting.
    Returns count of applied layers.
    """
    applied_count = 0
    clean_policy = {k.split('/')[0]: v for k, v in policy.items()}

    # Separate chunk policies (e.g. dense_1_part3) from regular ones
    chunk_map = {}   # {base_layer_name: {chunk_idx: bits}}
    regular_policy = {}
    for key, bits in clean_policy.items():
        m = re.match(r"^(.+)_part(\d+)$", key)
        if m:
            base, idx = m.group(1), int(m.group(2))
            chunk_map.setdefault(base, {})[idx] = bits
        else:
            regular_policy[key] = bits

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights: continue

        if layer.name in chunk_map:
            bits_info = chunk_map[layer.name]
            n_chunks = max(bits_info.keys()) + 1
            w = weights[0]
            w_chunks = np.array_split(w, n_chunks, axis=0)
            q_chunks = [quantize_int_asym_np(c, bits_info.get(i, 32))
                        if bits_info.get(i, 32) < 32 else c
                        for i, c in enumerate(w_chunks)]
            new_weights = [np.concatenate(q_chunks, axis=0)] + list(weights[1:])
            layer.set_weights(new_weights)
            applied_count += 1

        elif layer.name in regular_policy:
            bits = regular_policy[layer.name]
            w_q = quantize_int_asym_np(weights[0], bits)
            layer.set_weights([w_q] + list(weights[1:]))
            applied_count += 1

    return applied_count

def quantize_feedback(y, bits):
    if bits == 0: return y
    levels = 2**bits
    y_min, y_max = np.min(y, axis=1, keepdims=True), np.max(y, axis=1, keepdims=True)
    scale = (levels - 1) / (y_max - y_min + 1e-9)
    return (np.round((y - y_min) * scale) / scale) + y_min

# =========================================================
# [3] CsiNet Structure (Standard)
# =========================================================
def build_model(act_bits):
    df, bn_axis = 'channels_first', 3
    inp = Input((img_channels, img_height, img_width), name='input_1')
    
    # Names MUST match .h5 file exactly
    ec = Conv2D(2, (3,3), padding='same', data_format=df, name='conv2d_1')
    eb = BatchNormalization(axis=bn_axis, name='batch_normalization_1')
    el = LeakyReLU(alpha=0.3, name='leaky_re_lu_1')
    er = Reshape((img_channels*img_height*img_width,), name='reshape_1')
    ed = Dense(encoded_dim, activation='linear', name='dense_1')
    
    x = ec(inp); x = eb(x); x = el(x); x = QuantizationLayer(act_bits, name='act_quant_1')(x)
    x = er(x); enc = ed(x)
    
    dd = Dense(img_channels*img_height*img_width, activation='linear', name='dense_2')
    dr = Reshape((img_channels, img_height, img_width), name='reshape_2')
    res_layers = []
    for i in range(residual_num):
        idx1, idx2, idx3 = 2+i*3, 3+i*3, 4+i*3
        res_layers.append({
            'c1': Conv2D(8, (3,3), padding='same', data_format=df, name=f'conv2d_{idx1}'),
            'b1': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx1}'),
            'l1': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx1}'),
            'c2': Conv2D(16, (3,3), padding='same', data_format=df, name=f'conv2d_{idx2}'),
            'b2': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx2}'),
            'l2': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx2}'),
            'c3': Conv2D(2, (3,3), padding='same', data_format=df, name=f'conv2d_{idx3}'),
            'b3': BatchNormalization(axis=bn_axis, name=f'batch_normalization_{idx3}'),
            'add': Add(name=f'add_{i+1}'),
            'lout': LeakyReLU(alpha=0.3, name=f'leaky_re_lu_{idx3}_out')
        })
    
    dfinal = Conv2D(2, (3,3), padding='same', activation='sigmoid', data_format=df, name='conv2d_8')

    x = dd(enc); x = QuantizationLayer(act_bits, name='act_quant_dec_dense')(x); x = dr(x)
    for i, b in enumerate(res_layers):
        s = x
        x = b['c1'](x); x = b['b1'](x); x = b['l1'](x); x = QuantizationLayer(act_bits)(x)
        x = b['c2'](x); x = b['b2'](x); x = b['l2'](x); x = QuantizationLayer(act_bits)(x)
        x = b['c3'](x); x = b['b3'](x)
        x = b['add']([s, x]); x = b['lout'](x); x = QuantizationLayer(act_bits)(x)
    out = dfinal(x)
    
    ae = Model(inp, out, name='Autoencoder')
    enc_model = Model(inp, enc, name='Encoder')
    dec_in = Input((encoded_dim,))
    x = dd(dec_in); x = QuantizationLayer(act_bits)(x); x = dr(x)
    for i, b in enumerate(res_layers):
        s = x
        x = b['c1'](x); x = b['b1'](x); x = b['l1'](x); x = QuantizationLayer(act_bits)(x)
        x = b['c2'](x); x = b['b2'](x); x = b['l2'](x); x = QuantizationLayer(act_bits)(x)
        x = b['c3'](x); x = b['b3'](x)
        x = b['add']([s, x]); x = b['lout'](x); x = QuantizationLayer(act_bits)(x)
    dec_out = dfinal(x)
    dec_model = Model(dec_in, dec_out, name='Decoder')
    return ae, enc_model, dec_model

# =========================================================
# [3.5] FLOPs Counter
# =========================================================
def count_flops(model):
    """Count FLOPs for a Keras model via frozen graph profiler."""
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    concrete = tf.function(lambda x: model(x))
    concrete_func = concrete.get_concrete_function(
        tf.TensorSpec([1] + list(model.input_shape[1:]), model.dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'  # suppress stdout
        flops_info = tf.compat.v1.profiler.profile(graph, options=opts)
    return flops_info.total_float_ops

def print_flops(ae, encoder, decoder):
    """Print FLOPs for AE, Encoder, Decoder."""
    cr = img_channels * img_height * img_width // encoded_dim
    print(f"\n  [FLOPs] CsiNet (CR=1/{cr}, dim={encoded_dim})")
    total_params = ae.count_params()
    enc_params = encoder.count_params()
    dec_params = total_params - enc_params
    print(f"  Params  - Total: {total_params:,}  Enc: {enc_params:,}  Dec: {dec_params:,}")
    try:
        ae_flops = count_flops(ae)
        enc_flops = count_flops(encoder)
        dec_flops = ae_flops - enc_flops
        print(f"  FLOPs   - Total: {ae_flops/1e6:.2f}M  Enc: {enc_flops/1e6:.2f}M  Dec: {dec_flops/1e6:.2f}M")
    except Exception as e:
        print(f"  FLOPs measurement failed: {e}")
        print("  Falling back to manual calculation...")
        # Manual FLOPs: Conv2D = 2*H*W*Kh*Kw*Cin*Cout, Dense = 2*in*out
        enc_flops_m = (2*32*32*3*3*2*2 + 2*2048*encoded_dim) / 1e6
        dec_dense = 2*encoded_dim*2048 / 1e6
        dec_conv = 2 * residual_num * (2*32*32*3*3*2*8 + 2*32*32*3*3*8*16 + 2*32*32*3*3*16*2) / 1e6
        dec_final = 2*32*32*3*3*2*2 / 1e6
        dec_flops_m = dec_dense + dec_conv + dec_final
        total_m = enc_flops_m + dec_flops_m
        print(f"  FLOPs   - Total: {total_m:.2f}M  Enc: {enc_flops_m:.2f}M  Dec: {dec_flops_m:.2f}M")

# =========================================================
# [4] HAWQ & ILP (Corrected)
# =========================================================
class HAWQAnalyzerTF:
    def __init__(self, model, encoder, data):
        self.model = model; self.data = tf.convert_to_tensor(data)
        enc_layer_names = {l.name for l in encoder.layers}
        self.target_layers = [l for l in model.layers
                              if isinstance(l, (Conv2D, Dense))
                              and l.name in enc_layer_names]

    def compute_importance(self):
        print(f"\n[HAWQ] Computing Hessian Trace...")
        # 1. 1st Order
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                preds = self.model(self.data)
                loss = tf.reduce_mean(tf.square(self.data - preds))

            vars_ = []
            layer_names = []  # parallel list — avoids v.name collision in Keras 3.x
            for layer in self.target_layers:
                if not layer.trainable_variables: continue
                # kernel only (no bias)
                w_var = [v for v in layer.trainable_variables if 'bias' not in v.name]
                if w_var:
                    vars_.extend(w_var)
                    layer_names.extend([layer.name] * len(w_var))

            grads = tape1.gradient(loss, vars_)
            grad_v_prod = tf.constant(0.0)
            vecs = []
            for g in grads:
                if g is not None:
                    v = tf.cast(tf.random.uniform(g.shape, 0, 2, dtype=tf.int32)*2-1, tf.float32)
                    vecs.append(v); grad_v_prod += tf.reduce_sum(g * v)
                else: vecs.append(None)

        # 2. 2nd Order
        SPLIT_THRESHOLD = 20000
        NUM_CHUNKS = 32

        hvp = tape2.gradient(grad_v_prod, vars_)
        results = []
        for var, hv, v, lname in zip(vars_, hvp, vecs, layer_names):
            if hv is None: continue
            trace = tf.abs(tf.reduce_sum(v*hv)).numpy()
            w_np = var.numpy()

            if w_np.size > SPLIT_THRESHOLD:
                # Split large layer into chunks (same as rpmpq_baselines.py)
                w_chunks = np.array_split(w_np, NUM_CHUNKS, axis=0)
                trace_per_chunk = trace / NUM_CHUNKS
                for i, chunk in enumerate(w_chunks):
                    chunk_res = {"Layer": f"{lname}_part{i}",
                                 "Params": chunk.size,
                                 "Trace": trace_per_chunk}
                    for b in [16, 8, 4, 2]:
                        w_q = quantize_int_asym_np(chunk, b)
                        chunk_res[f"Omg_INT{b}"] = trace_per_chunk * np.linalg.norm(chunk - w_q)**2
                    results.append(chunk_res)
            else:
                res = {"Layer": lname, "Params": np.prod(var.shape), "Trace": trace}
                for b in [16, 8, 4, 2]:
                    w_q = quantize_int_asym_np(w_np, b)
                    res[f"Omg_INT{b}"] = trace * np.linalg.norm(w_np - w_q)**2
                results.append(res)

        return pd.DataFrame(results)

class ILP_Solver:
    def __init__(self, hawq_df, act_bits):
        self.df = hawq_df; self.act_bits = act_bits; self.bit_opts = [16, 8, 4, 2]
        # Calculate Total Baseline BOPs (FP32 W * FP32 A)
        self.total_bops = sum(r['Params']*32*32 for _, r in hawq_df.iterrows())
        print(f"   [Solver] Baseline Total BOPs: {self.total_bops/1e9:.4f} G")

    def solve_top_k(self, target_saving, top_k=10):
        solutions = []
        prob = pulp.LpProblem("MP_Search", pulp.LpMinimize)
        layers = self.df['Layer'].tolist()
        x = pulp.LpVariable.dicts("x", (layers, self.bit_opts), cat=pulp.LpBinary)
        
        # Objective: Min Perturbation
        prob += pulp.lpSum([x[r['Layer']][b] * r[f'Omg_INT{b}'] for _, r in self.df.iterrows() for b in self.bit_opts])
        
        # Constraint 1: One Bit per Layer
        for l in layers: prob += pulp.lpSum([x[l][b] for b in self.bit_opts]) == 1
        
        # Constraint 2: BOPs Limit
        # Current BOPs = Params * Selected_Bit * Act_Bit
        curr_bops = [x[r['Layer']][b] * (r['Params'] * b * self.act_bits) for _, r in self.df.iterrows() for b in self.bit_opts]
        
        # Saving = 1 - (Curr / Baseline)  ==>  Curr <= Baseline * (1 - Saving/100)
        target_bops_val = self.total_bops * (1 - target_saving/100.0)
        prob += pulp.lpSum(curr_bops) <= target_bops_val

        for k in range(top_k):
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] != 'Optimal': break
            
            policy = {l: b for l in layers for b in self.bit_opts if pulp.value(x[l][b])==1}
            
            # Calculate realized saving
            pol_bops = sum(self.df[self.df['Layer']==l]['Params'].values[0] * b * self.act_bits for l, b in policy.items())
            real_saving = (1 - pol_bops / self.total_bops) * 100
            
            solutions.append({"Policy": policy, "Actual_Saving": real_saving})
            
            # Integer Cut
            selected_vars = [x[l][b] for l, b in policy.items()]
            prob += pulp.lpSum(selected_vars) <= len(layers) - 1
            
        return solutions

# =========================================================
# [5] Evaluation Helpers
# =========================================================
def evaluate_candidates_kl(encoder, candidates, data, original_weights):
    restore_weights_tf(encoder, original_weights)
    z_fp32 = encoder.predict(data, verbose=0)
    z_fp32_logits = tf.nn.softmax(z_fp32, axis=1) # Treat latent as dist

    best_cand = None; best_kl = float('inf')
    kl_obj = tf.keras.losses.KLDivergence()
    
    for cand in candidates:
        cnt = apply_mp_policy_strict(encoder, cand['Policy'])
        # Debug: Ensure policy was applied
        if cnt == 0: print("[WARN] Policy applied to 0 layers!"); continue
            
        z_q = encoder.predict(data, verbose=0)
        z_q_log = tf.nn.log_softmax(z_q, axis=1)
        kl_val = kl_obj(z_fp32_logits, z_q_log).numpy()
        cand['KL_Score'] = kl_val
        if kl_val < best_kl: best_kl = kl_val; best_cand = cand
        restore_weights_tf(encoder, original_weights)
    return best_cand

def restore_weights_tf(model, weights_dict):
    for l_name, w in weights_dict.items(): model.get_layer(l_name).set_weights(w)

def plot_pareto_curve(df, env_mode, filename):
    plt.figure(figsize=(10, 6))
    raw_nmse_db = 10 * np.log10(df['MSE'] + 1e-15)
    plt.scatter(df['Actual_Saving'], raw_nmse_db, c='gray', alpha=0.5, label='Raw', s=20)
    
    smooth_nmse_db = 10 * np.log10(df['MSE_Smoothed'] + 1e-15)
    plt.plot(df['Actual_Saving'], smooth_nmse_db, c='r', lw=2, label='Pareto')
    
    plt.title(f"Pareto Frontier - CsiNet ({env_mode})")
    plt.xlabel("BOPs Saving (%)"); plt.ylabel("NMSE (dB)")
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.savefig(filename, dpi=300); print(f" Graph Saved: {filename}")

# =========================================================
# [6] Main Execution
# =========================================================
print("Initializing...")
# Load Data
mat_path = os.path.join(data_dir, f'DATA_Htest{"in" if ENV_MODE=="indoor" else "out"}.mat')
x_test = sio.loadmat(mat_path)['HT'].astype('float32')
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))
calib_data = x_test[:args.batch_size]

if args.analyze_all:
    print(f"[OFFLINE-STRICT] Mamba-Style Pipeline Started")
    
    # 1. HAWQ (FP32)
    print("   -> Phase 1: HAWQ Analysis (FP32 Model)...")
    with tf.device('/device:GPU:0'):
        ae_fp32, enc_fp32, _ = build_model(act_bits=32)
        w_path = os.path.join(base_model_dir, f'model_CsiNet_{ENV_MODE}_dim{encoded_dim}.h5')
        ae_fp32.load_weights(w_path, by_name=True)
        analyzer = HAWQAnalyzerTF(ae_fp32, enc_fp32, calib_data)
        hawq_df = analyzer.compute_importance()
        if hawq_df.empty: print("[ERR] Error: HAWQ results empty."); exit()
        del ae_fp32; tf.keras.backend.clear_session()

    # 2. Pareto Search (INT16 Model)
    print(f"   -> Phase 2: Pareto Search (Act: INT{ACT_BITS})...")
    with tf.device('/device:GPU:0'):
        ae, encoder, decoder = build_model(act_bits=ACT_BITS)
        ae.load_weights(w_path, by_name=True)
        orig_weights = {l.name: l.get_weights() for l in encoder.layers if l.weights}

    solver = ILP_Solver(hawq_df, act_bits=ACT_BITS)
    if args.extend:
        targets = np.arange(95.1, 97.1, 0.1)
        print("   [Extend mode] 95.1-97.0%")
    elif args.wide_step is not None:
        targets = np.arange(85.0, 98.0 + args.wide_step * 0.1, args.wide_step)
        print(f"   [Wide mode] 85.0-98.0% | step {args.wide_step}%  | {len(targets)} points")
    elif ACT_BITS < 16:
        targets = np.arange(88.0, 97.1, 0.1)
        print(f"   [act={ACT_BITS}] Target range: 88.0-97.0%")
    else:
        targets = np.arange(85.0, 95.1, 0.1)
    lut_raw = []
    
    pbar = tqdm(targets, desc="Searching")
    for tgt in pbar:
        # A. Solve ILP
        candidates = solver.solve_top_k(tgt, top_k=10)
        if not candidates: continue
        
        # B. KL Selection
        best_cand = evaluate_candidates_kl(encoder, candidates, calib_data, orig_weights)
        if not best_cand: continue

        # C. Real Metric Check
        apply_mp_policy_strict(encoder, best_cand['Policy'])
        z = encoder.predict(calib_data, verbose=0)
        z_q = quantize_feedback(z, QUANTIZATION_BITS)
        x_hat = decoder.predict(z_q, verbose=0)
        
        # Complex NMSE Calc — per-sample (mean of ratios), field standard
        x_true_c = calib_data[:,0] - 0.5 + 1j*(calib_data[:,1] - 0.5)
        x_hat_c = x_hat[:,0] - 0.5 + 1j*(x_hat[:,1] - 0.5)
        mse_per = np.sum(np.abs(x_true_c - x_hat_c)**2, axis=(1, 2))
        pwr_per = np.sum(np.abs(x_true_c)**2, axis=(1, 2))
        nmse_db = 10*np.log10(np.mean(mse_per / pwr_per))

        best_cand['MSE'] = float(np.mean(mse_per / pwr_per))
        best_cand['NMSE_dB'] = nmse_db
        best_cand['Target_Saving'] = float(tgt)
        lut_raw.append(best_cand)
        restore_weights_tf(encoder, orig_weights)
        
        avg_bits = np.mean(list(best_cand['Policy'].values()))
        pbar.set_postfix({'Sv': f"{best_cand['Actual_Saving']:.1f}%", 'dB': f"{nmse_db:.2f}", 'Bit': f"{avg_bits:.1f}"})

    # 3. Combine + Smoothing
    df_new = pd.DataFrame(lut_raw).sort_values('Actual_Saving')
    out_dir = os.path.join(PROJECT_DIR, 'MambaIC', 'results', 'csv')
    os.makedirs(out_dir, exist_ok=True)
    act_suffix = f"_a{ACT_BITS}" if ACT_BITS not in (16, 32) else ""
    csv_name = os.path.join(out_dir, f"mp_policy_lut_csinet_cr4_{ENV_MODE[:3]}{act_suffix}.csv")
    if args.extend and os.path.exists(csv_name):
        df_exist = pd.read_csv(csv_name)
        df = pd.concat([df_exist, df_new], ignore_index=True).sort_values('Actual_Saving')
    else:
        df = df_new
    print(f"\n Phase 3: Smoothing...")
    
    smoothed_mse = []
    current_min = float('inf')
    for val in reversed(df['MSE'].tolist()):
        if val < current_min: current_min = val
        smoothed_mse.append(current_min)
    df['MSE_Smoothed'] = smoothed_mse[::-1]
    
    png_name = os.path.join(PROJECT_DIR, 'figures', f"fig_rpmpq_csinet_{ENV_MODE[:3]}{act_suffix}.png")
    df.to_csv(csv_name, index=False)
    plot_pareto_curve(df, ENV_MODE, png_name)
    print(f"[OK] DONE: Saved {csv_name}")

elif args.run_online:
    # Online Logic (Same as before but uses apply_mp_policy_strict)
    pass

else:
    # =========================================================
    # Baseline Evaluation (with optional uniform quantization)
    # =========================================================
    W_BITS = args.weight_bits
    print(f"\n[Eval] CsiNet {ENV_MODE}, dim={encoded_dim}, W={W_BITS}bit, act={ACT_BITS}bit, aq={QUANTIZATION_BITS}bit")
    with tf.device('/device:GPU:0'):
        ae, encoder, decoder = build_model(act_bits=ACT_BITS)
        w_path = os.path.join(base_model_dir, f'model_CsiNet_{ENV_MODE}_dim{encoded_dim}.h5')
        ae.load_weights(w_path, by_name=True)
        print(f"   Loaded: {w_path}")

    # Uniform weight quantization
    if W_BITS < 32:
        print(f"   Applying INT{W_BITS} weight quantization...")
        for layer in ae.layers:
            if isinstance(layer, (Conv2D, Dense)):
                weights = layer.get_weights()
                if not weights: continue
                w_q = quantize_int_asym_np(weights[0], W_BITS)
                layer.set_weights([w_q] + list(weights[1:]))

    print("   Running inference...", end=" ", flush=True)
    z = encoder.predict(x_test, batch_size=args.batch_size, verbose=0)
    z_q = quantize_feedback(z, QUANTIZATION_BITS)
    x_hat = decoder.predict(z_q, batch_size=args.batch_size, verbose=0)
    print("Done.")

    # Complex NMSE — per-sample (mean of ratios), field standard
    x_true_c = x_test[:,0] - 0.5 + 1j*(x_test[:,1] - 0.5)
    x_hat_c  = x_hat[:,0]  - 0.5 + 1j*(x_hat[:,1]  - 0.5)
    mse_per_sample = np.sum(np.abs(x_true_c - x_hat_c)**2, axis=(1, 2))
    pwr_per_sample = np.sum(np.abs(x_true_c)**2, axis=(1, 2))
    nmse_db = 10 * np.log10(np.mean(mse_per_sample / pwr_per_sample))

    # BOPs calculation
    cr = img_channels * img_height * img_width // encoded_dim
    enc_params = encoder.count_params()
    w_bits_eff = W_BITS if W_BITS < 32 else 32
    a_bits_eff = ACT_BITS if ACT_BITS < 32 else 32
    enc_bops = enc_params * w_bits_eff * a_bits_eff
    fp32_bops = enc_params * 32 * 32
    saving = (1 - enc_bops / fp32_bops) * 100

    print(f"\n{'='*55}")
    print(f"  CsiNet ({ENV_MODE}, CR=1/{cr}, W=INT{w_bits_eff}, A=INT{a_bits_eff})")
    print(f"  NMSE = {nmse_db:.2f} dB")
    print(f"  Enc. Params: {enc_params:,}")
    print(f"  Enc. BOPs: {enc_bops/1e6:.2f}M  (Saving vs FP32: {saving:.2f}%)")
    print_flops(ae, encoder, decoder)
    print(f"{'='*55}")