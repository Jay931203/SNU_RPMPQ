import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, Add, LeakyReLU
from tensorflow.keras.models import Model
import scipy.io as sio 
import numpy as np
import math
import time
import argparse
from numpy.linalg import inv

# 세션 클리어
tf.keras.backend.clear_session()

# =========================================================
# [CLI Argument Parsing]
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pq', type=str, default='FP32', 
                    choices=['FP32', 'FP16', 'INT8', 'FP8', 'FP4'],
                    help='Parameter Quantization Mode')
parser.add_argument('--aq', type=int, default=0, 
                    help='Activation Quantization Bits (0=FP32)')
parser.add_argument('--sumrate', action='store_true', 
                    help='Calculate Sum Rate')

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

PARAM_QUANT_MODE = args.pq
QUANTIZATION_BITS = args.aq
CALC_SUMRATE = args.sumrate

# =========================================================
# [중요] 환경 설정 (Outdoor로 고정)
# =========================================================
envir = 'indoor'  # <--- 여기를 수정했습니다!
# =========================================================

print(f"================================================================")
print(f"   [Experiment Config]")
print(f"   - Environment:    {envir} (Changed!)")
print(f"   - Target Mode:    {PARAM_QUANT_MODE}")
print(f"   - Feedback (z):   {'FP32' if QUANTIZATION_BITS == 0 else f'{QUANTIZATION_BITS}-bit'}")
print(f"================================================================")
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

data_dir = '/content/drive/MyDrive/MambaCompression/MambaIC/data/'
base_model_dir = '/content/drive/MyDrive/MambaCompression/Python_CsiNet-master/saved_model/'

img_height = 32
img_width = 32
img_channels = 2 
encoded_dim = 512 
residual_num = 2 

# ---------------------------------------------------------
# Simulation Helper Functions
# ---------------------------------------------------------
def get_fp_grid(exponent_bits, mantissa_bits):
    bias = 2**(exponent_bits - 1) - 1
    values = []
    for s in [0, 1]:
        sign = (-1)**s
        for e in range(2**exponent_bits):
            for m in range(2**mantissa_bits):
                if e == 0 and m == 0: value = 0.0
                else:
                    mantissa_val = 1.0 + (m / (2**mantissa_bits))
                    value = sign * (2.0**(e - bias)) * mantissa_val
                values.append(value)
    grid = sorted(list(set(values)))
    return np.array(grid, dtype=np.float32)

FP8_GRID = get_fp_grid(4, 3)
FP4_GRID = get_fp_grid(2, 1)

def quantize_to_grid(weights, grid):
    w_abs_max = np.max(np.abs(weights))
    if w_abs_max == 0: return weights
    grid_max = np.max(grid)
    scale = grid_max / w_abs_max
    w_scaled = weights * scale
    idx = np.searchsorted(grid, w_scaled.flatten())
    idx = np.clip(idx, 1, len(grid)-1)
    grid_val_left = grid[idx-1]
    grid_val_right = grid[idx]
    w_flat = w_scaled.flatten()
    mask = np.abs(w_flat - grid_val_left) < np.abs(w_flat - grid_val_right)
    w_q_flat = np.where(mask, grid_val_left, grid_val_right)
    return (w_q_flat.reshape(weights.shape)) / scale

def apply_weight_quantization(model, mode):
    if mode == 'FP32': return
    print(f">> [Simulation] Applying {mode} Fake Quantization...")
    target_layers = (tf.keras.layers.Conv2D, tf.keras.layers.Dense)
    for layer in model.layers:
        if isinstance(layer, target_layers):
            weights = layer.get_weights()
            if not weights: continue
            w, b = weights[0], (weights[1] if len(weights) > 1 else None)
            
            if mode == 'FP16': w_q = w.astype(np.float16).astype(np.float32)
            elif mode == 'FP8': w_q = quantize_to_grid(w, FP8_GRID)
            elif mode == 'FP4': w_q = quantize_to_grid(w, FP4_GRID)
            elif mode == 'INT8':
                max_val = np.max(np.abs(w))
                if max_val == 0: w_q = w
                else:
                    scale = max_val / 127.0
                    w_q = np.round(w / scale) * scale
                    w_q = np.clip(w_q, -max_val, max_val)
            else: w_q = w
            
            if b is not None: layer.set_weights([w_q, b])
            else: layer.set_weights([w_q])

def quantize_uniform(y_vec, B_bits):
    Q_levels = 2**B_bits
    y_min = np.min(y_vec, axis=1, keepdims=True)
    y_max = np.max(y_vec, axis=1, keepdims=True)
    scale = (Q_levels - 1) / (y_max - y_min + 1e-9)
    y_scaled = (y_vec - y_min) * scale
    y_quantized = np.round(y_scaled)
    y_dequantized = (y_quantized / scale) + y_min
    return y_dequantized

def get_model_specs(model, weight_mode):
    bit_map = {'FP32': 32, 'FP16': 16, 'INT8': 8, 'FP8': 8, 'FP4': 4}
    w_bits = bit_map.get(weight_mode, 32)
    total_params = model.count_params()
    mem_mb = (total_params * w_bits) / (8 * 1024 * 1024)
    total_flops = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            try:
                w_shape = layer.kernel.shape
                h_out, w_out = layer.output.shape[2], layer.output.shape[3]
                macs = w_shape[0] * w_shape[1] * w_shape[2] * h_out * w_out * w_shape[3]
                total_flops += 2 * macs
            except: pass
        elif isinstance(layer, tf.keras.layers.Dense):
            try:
                w_shape = layer.kernel.shape
                total_flops += 2 * w_shape[0] * w_shape[1]
            except: pass
    total_bops_sim = (total_flops / 2) * w_bits * 32
    baseline_bops = (total_flops / 2) * 32 * 32
    bops_reduction = 1 - (total_bops_sim / baseline_bops)
    return total_params, mem_mb, total_flops, total_bops_sim, bops_reduction

# ---------------------------------------------------------
# Real CPU Benchmark
# ---------------------------------------------------------
def benchmark_tflite_real(keras_model, sample_data, target_mode):
    print("\n---------------------------------------------------------------")
    print(f"   [Real Hardware Benchmark] Measuring CPU Latency for {target_mode}")
    print("---------------------------------------------------------------")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    if target_mode == 'INT8':
        def representative_data_gen():
            for input_value in tf.data.Dataset.from_tensor_slices(sample_data[:50]).batch(1).take(50):
                yield [tf.dtypes.cast(input_value, tf.float32)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif target_mode == 'FP16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        if target_mode in ['FP8', 'FP4']:
            print(f"** Note: {target_mode} not supported in TFLite. Running as FP32 Baseline.")
        converter.optimizations = []

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"TFLite Conversion Failed: {e}")
        return 0.0, 0.0

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    
    input_data = sample_data[0:1]
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        input_data = (input_data / scale + zero_point).astype(np.int8)
    else:
        input_data = input_data.astype(np.float32)
        
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    
    runs = 200
    start = time.time()
    for _ in range(runs):
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
    end = time.time()
    
    avg_latency = (end - start) / runs * 1000
    size_kb = len(tflite_model) / 1024
    return size_kb, avg_latency

# ---------------------------------------------------------
# Build Model
# ---------------------------------------------------------
def build_csinet_shared(encoded_dim):
    data_format = 'channels_first'
    bn_axis = 3 
    input_img = Input(shape=(img_channels, img_height, img_width), name='input_1')
    
    l_enc_conv = Conv2D(2, (3, 3), padding='same', data_format=data_format, name='conv2d_1')
    l_enc_bn = BatchNormalization(axis=bn_axis, name='batch_normalization_1')
    l_enc_leaky = LeakyReLU(negative_slope=0.3, name='leaky_re_lu_1')
    l_enc_reshape = Reshape((img_channels * img_height * img_width,), name='reshape_1')
    l_enc_dense = Dense(encoded_dim, activation='linear', name='dense_1')
    
    l_dec_dense = Dense(img_channels * img_height * img_width, activation='linear', name='dense_2')
    l_dec_reshape = Reshape((img_channels, img_height, img_width), name='reshape_2')
    
    refine_layers = []
    b1 = {
        'c1': Conv2D(8, (3, 3), padding='same', data_format=data_format, name='conv2d_2'),
        'b1': BatchNormalization(axis=bn_axis, name='batch_normalization_2'),
        'l1': LeakyReLU(negative_slope=0.3, name='leaky_re_lu_2'),
        'c2': Conv2D(16, (3, 3), padding='same', data_format=data_format, name='conv2d_3'),
        'b2': BatchNormalization(axis=bn_axis, name='batch_normalization_3'),
        'l2': LeakyReLU(negative_slope=0.3, name='leaky_re_lu_3'),
        'c3': Conv2D(2, (3, 3), padding='same', data_format=data_format, name='conv2d_4'),
        'b3': BatchNormalization(axis=bn_axis, name='batch_normalization_4'),
        'add': Add(name='add_1'),
        'l_out': LeakyReLU(negative_slope=0.3, name='leaky_re_lu_4')
    }
    refine_layers.append(b1)
    
    b2 = {
        'c1': Conv2D(8, (3, 3), padding='same', data_format=data_format, name='conv2d_5'),
        'b1': BatchNormalization(axis=bn_axis, name='batch_normalization_5'),
        'l1': LeakyReLU(negative_slope=0.3, name='leaky_re_lu_5'),
        'c2': Conv2D(16, (3, 3), padding='same', data_format=data_format, name='conv2d_6'),
        'b2': BatchNormalization(axis=bn_axis, name='batch_normalization_6'),
        'l2': LeakyReLU(negative_slope=0.3, name='leaky_re_lu_6'),
        'c3': Conv2D(2, (3, 3), padding='same', data_format=data_format, name='conv2d_7'),
        'b3': BatchNormalization(axis=bn_axis, name='batch_normalization_7'),
        'add': Add(name='add_2'),
        'l_out': LeakyReLU(negative_slope=0.3, name='leaky_re_lu_7')
    }
    refine_layers.append(b2)
    l_dec_final = Conv2D(2, (3, 3), padding='same', activation='sigmoid', data_format=data_format, name='conv2d_8')

    x = l_enc_conv(input_img); x = l_enc_bn(x); x = l_enc_leaky(x)
    x = l_enc_reshape(x); encoded = l_enc_dense(x)
    x = l_dec_dense(encoded); x = l_dec_reshape(x)
    for b in refine_layers:
        s = x
        x = b['c1'](x); x = b['b1'](x); x = b['l1'](x)
        x = b['c2'](x); x = b['b2'](x); x = b['l2'](x)
        x = b['c3'](x); x = b['b3'](x); x = b['add']([s, x]); x = b['l_out'](x)
    decoded = l_dec_final(x)
    autoencoder = Model(inputs=input_img, outputs=decoded, name='Autoencoder')
    encoder = Model(inputs=input_img, outputs=encoded, name='Encoder')
    
    dec_in = Input(shape=(encoded_dim,), name='dec_in')
    x = l_dec_dense(dec_in); x = l_dec_reshape(x)
    for b in refine_layers:
        s = x
        x = b['c1'](x); x = b['b1'](x); x = b['l1'](x)
        x = b['c2'](x); x = b['b2'](x); x = b['l2'](x)
        x = b['c3'](x); x = b['b3'](x); x = b['add']([s, x]); x = b['l_out'](x)
    dec_out = l_dec_final(x)
    decoder = Model(inputs=dec_in, outputs=dec_out, name='Decoder')
    return autoencoder, encoder, decoder

with tf.device('/device:GPU:0'):
    print("Building CSINet model...")
    autoencoder, encoder, decoder = build_csinet_shared(encoded_dim)
    model_filename = 'model_CsiNet_'+(envir)+'_dim'+str(encoded_dim)+'.h5'
    weights_path = os.path.join(base_model_dir, model_filename)
    
    if os.path.exists(weights_path):
        autoencoder.load_weights(weights_path, by_name=True) 
        print("Weights loaded successfully!")
        apply_weight_quantization(encoder, PARAM_QUANT_MODE)
    else:
        print(f"[ERROR] Weight file not found")
        exit()

print(f"Loading data from: {data_dir}")
if envir == 'indoor': mat = sio.loadmat(os.path.join(data_dir, 'DATA_Htestin.mat')); x_test = mat['HT']
elif envir == 'outdoor': mat = sio.loadmat(os.path.join(data_dir, 'DATA_Htestout.mat')); x_test = mat['HT']
x_test = x_test.astype('float32')
print("Applying Simple Reshape (C-order)...")
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))

print("\n[Simulated Inference]")
with tf.device('/device:GPU:0'):
    encoded_vector = encoder.predict(x_test, batch_size=200)
if QUANTIZATION_BITS > 0:
    encoded_vector = quantize_uniform(encoded_vector, QUANTIZATION_BITS)
with tf.device('/device:GPU:0'):
    x_hat = decoder.predict(encoded_vector, batch_size=200)

if envir == 'indoor': mat = sio.loadmat(os.path.join(data_dir, 'DATA_HtestFin_all.mat')); X_test = mat['HF_all']
elif envir == 'outdoor': mat = sio.loadmat(os.path.join(data_dir, 'DATA_HtestFout_all.mat')); X_test = mat['HF_all']
X_test = np.reshape(X_test, (len(X_test), img_height, 125))
x_test_C = np.reshape(x_test[:, 0, :, :], (len(x_test), -1)) - 0.5 + 1j*(np.reshape(x_test[:, 1, :, :], (len(x_test), -1)) - 0.5)
x_hat_C = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1)) - 0.5 + 1j*(np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1)) - 0.5)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)[:, :, 0:125]

# [ComplexWarning Fix] Use np.abs()
n1 = np.sqrt(np.abs(np.sum(np.conj(X_test)*X_test, axis=1)))
n2 = np.sqrt(np.abs(np.sum(np.conj(X_hat)*X_hat, axis=1)))
rho = np.mean(np.abs(np.sum(np.conj(X_test)*X_hat, axis=1))/(n1*n2), axis=1)
power = np.sum(np.abs(x_test_C)**2, axis=1)
mse = np.sum(np.abs(x_test_C-x_hat_C)**2, axis=1)

def calculate_sum_rate_mmse(H_true, H_est, snr_db, group_size=4):
    n_samples = H_true.shape[0] // group_size * group_size
    H_true, H_est = H_true[:n_samples], H_est[:n_samples]
    snr = 10 ** (snr_db / 10.0)
    noise_var = 1.0 / snr 
    total_rate = 0.0
    num_groups = n_samples // group_size
    for g in range(num_groups):
        H_real = H_true[g*group_size:(g+1)*group_size, :, :]
        H_hat_s = H_est[g*group_size:(g+1)*group_size, :, :]
        for s in range(H_true.shape[2]):
            H_r, H_h = H_real[:, :, s], H_hat_s[:, :, s]
            gram = np.matmul(H_h, H_h.conj().T)
            inv_term = inv(gram + group_size * noise_var * np.eye(group_size))
            W = np.matmul(H_h.conj().T, inv_term)
            trace_val = np.trace(np.matmul(W, W.conj().T)).real
            W = W / (np.sqrt(trace_val) if trace_val > 1e-12 else 1e-6)
            HW = np.matmul(H_r, W)
            S = np.abs(np.diag(HW)) ** 2
            I = np.sum(np.abs(HW)**2, axis=1) - S
            total_rate += np.sum(np.log2(1 + S / (I + noise_var)))
    return total_rate / (num_groups * H_true.shape[2])

if CALC_SUMRATE:
    try:
        print("Calculating Sum Rate...")
        sum_rate_est = calculate_sum_rate_mmse(X_test, X_hat, 10, 4)
        sum_rate_ideal = calculate_sum_rate_mmse(X_test, X_test, 10, 4)
    except: sum_rate_est, sum_rate_ideal = 0.0, 0.0
else:
    sum_rate_est, sum_rate_ideal = 0.0, 0.0

enc_par, enc_mem, enc_flops, enc_bops, enc_bops_red = get_model_specs(encoder, PARAM_QUANT_MODE)
real_size, real_latency = benchmark_tflite_real(encoder, x_test[:100], PARAM_QUANT_MODE)

print("\n=======================================================")
print(f" [RESULT] Weight: {PARAM_QUANT_MODE} | Feedback: {QUANTIZATION_BITS if QUANTIZATION_BITS > 0 else 'FP32'}")
print("=======================================================")
print(f"[ENCODER] (Quantized)")
print(f" - Params:        {enc_par:,}")
print(f" - FLOPs:         {enc_flops:,.0f}")
print(f" - BOPs Count:    {enc_bops:,.0f}")
print(f" - BOPs Saving:   {enc_bops_red*100:.2f} % (Theoretical)")
print(f" - [Real HW] Size:{real_size:.2f} KB (TFLite)")
print(f" - [Real HW] Time:{real_latency:.4f} ms/sample (CPU)")
print("-------------------------------------------------------")
print(f"[PERFORMANCE]")
print(f" - NMSE:          {10*math.log10(np.mean(mse/power)):.2f} dB")
print(f" - Correlation:   {np.mean(rho):.4f}")
if CALC_SUMRATE:
    print(f" - Sum Rate:      {sum_rate_est:.4f} / {sum_rate_ideal:.4f} ({sum_rate_est/sum_rate_ideal*100:.2f}%)")
else:
    print(f" - Sum Rate:      Skipped")
print("=======================================================")
