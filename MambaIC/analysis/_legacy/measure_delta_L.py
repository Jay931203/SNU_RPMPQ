"""
ΔL 측정 스크립트 (Colab에서 실행)

목적:
  INT4 / INT8 / INT2 전체-레이어 양자화 시
  실제 loss 변화 ΔL = NMSE_quant - NMSE_fp32 를 측정.
  이후 S_exact / S_trace / R = ΔL - S_exact 를 계산하고 CSV로 저장.

사용법 (Colab):
  exec(open('measure_delta_L.py').read())
"""
import torch
import torch.nn.functional as F
import numpy as np
import csv, os
from torch.utils.data import DataLoader

# ── Setup ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from ModularModels import ModularAE
from train_ae import CsiDataset, NMSELoss

MODEL_PATH = 'saved_models/mamba_transnet_L2_dim512_baseline/best.pth'
DATA_PATH  = 'data/DATA_Htestout.mat'
BATCH_SIZE = 64
N_BATCHES  = 200     # 200 × 64 = 12800 samples → stable estimate

# ── Load model ───────────────────────────────────────────────
model = ModularAE(
    encoder_type='mamba',
    decoder_type='transnet',
    encoded_dim=512,
    decoder_layers=2
)
ck = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ck['state_dict'], strict=False)
model = model.to(device).eval()

# ── Data ─────────────────────────────────────────────────────
dataset = CsiDataset(DATA_PATH)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=False, num_workers=2)
criterion = NMSELoss()


# ── Helper: evaluate NMSE ────────────────────────────────────
def eval_nmse(model, n_batches=N_BATCHES):
    total, count = 0.0, 0
    with torch.no_grad():
        for i, x in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device)
            recon = model(x)
            loss  = criterion(recon, x)
            total += loss.item()
            count += 1
    return total / max(count, 1)


# ── Helper: apply symmetric INT-N quantization to all weights ─
def quantize_model(model, bits):
    """In-place symmetric uniform quantization of all weight params."""
    qmax = 2 ** (bits - 1) - 1
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        w = param.data
        scale = w.abs().max().clamp(min=1e-8) / qmax
        param.data = (w / scale).round().clamp(-qmax, qmax) * scale


def restore_model(original_state, model):
    model.load_state_dict(original_state, strict=False)


# ── Baseline NMSE (FP32) ─────────────────────────────────────
print('Measuring FP32 baseline ...')
nmse_fp32 = eval_nmse(model)
print(f'  FP32  NMSE: {nmse_fp32:.6f}  (loss units)')

original_state = {k: v.clone() for k, v in model.state_dict().items()}

# ── Per-bit-width measurement ────────────────────────────────
BITS_LIST = [8, 4, 2]
rows = []

for bits in BITS_LIST:
    restore_model(original_state, model)
    quantize_model(model, bits)
    nmse_q = eval_nmse(model)
    delta_L = nmse_q - nmse_fp32
    print(f'  INT{bits} NMSE: {nmse_q:.6f}  ΔL = {delta_L:.6f}')
    rows.append({'bits': bits, 'nmse_fp32': nmse_fp32,
                 'nmse_quant': nmse_q, 'delta_L': delta_L})

restore_model(original_state, model)

# ── Load S_exact and S_trace from hawq_exact_omega.csv ───────
omega_csv = 'results/csv/hawq_exact_omega.csv'
with open(omega_csv, newline='') as f:
    reader = csv.DictReader(f)
    omega_rows = list(reader)

col_map = {
    8:  ('ExactOmg_INT8',  'Omg_INT8'),
    4:  ('ExactOmg_INT4',  'Omg_INT4'),
    2:  ('ExactOmg_INT2',  'Omg_INT2'),
}

for bits in BITS_LIST:
    exact_col, trace_col = col_map[bits]
    if exact_col not in omega_rows[0]:
        continue
    s_exact = sum(max(float(r[exact_col]), 0.0) for r in omega_rows)
    s_trace = sum(float(r[trace_col]) for r in omega_rows)
    row_idx = next(i for i, r in enumerate(rows) if r['bits'] == bits)
    delta_L = rows[row_idx]['delta_L']
    rows[row_idx]['S_exact'] = s_exact
    rows[row_idx]['S_trace'] = s_trace
    rows[row_idx]['R']       = max(delta_L - s_exact, 0.0)
    print(f'\n  INT{bits}:')
    print(f'    delta_L = {delta_L:.6f}')
    print(f'    S_exact = {s_exact:.6f}  ({100*s_exact/max(abs(delta_L),1e-10):.1f}% of |delta_L|)')
    print(f'    S_trace = {s_trace:.4f}  (overestimates by {s_trace/max(s_exact,1e-10):.0f}x)')
    print(f'    R       = {rows[row_idx]["R"]:.6f}')

# ── Save ─────────────────────────────────────────────────────
out_path = 'results/csv/delta_L_decomposition.csv'
fieldnames = ['bits', 'nmse_fp32', 'nmse_quant', 'delta_L', 'S_exact', 'S_trace', 'R']
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'\nSaved: {out_path}')
for r in rows:
    print(r)
