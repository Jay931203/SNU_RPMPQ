"""
Per-layer ΔL measurement (Colab에서 실행)

목적:
  각 weight parameter를 하나씩만 INT4 양자화하고, 나머지는 FP32 유지.
  실제 per-layer 민감도 ΔLᵢ = NMSE_i - NMSE_fp32 를 직접 측정.
  → hawq_exact_omega.csv의 Ωᵢ_exact와 비교하여 HAWQ proxy 검증.

사용법 (Colab):
  exec(open('measure_per_layer_dL.py').read())

출력:
  results/csv/per_layer_dL.csv
"""
import torch
import numpy as np
import csv, os, time
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

from ModularModels import ModularAE
from train_ae import CsiDataset, NMSELoss

MODEL_PATH = 'saved_models/mamba_transnet_L2_dim512_baseline/best.pth'
DATA_PATH  = 'data/DATA_Htestout.mat'
BATCH_SIZE = 8
N_BATCHES  = 50    # 50×8 = 400 samples, ~1-2s per layer → 전체 ~2분

# ── Load model ────────────────────────────────────────────────
model = ModularAE(encoder_type='mamba', decoder_type='transnet',
                  encoded_dim=512, decoder_layers=2)
ck = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ck['state_dict'], strict=False)
model = model.to(device).eval()

dataset  = CsiDataset(DATA_PATH)
loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # num_workers=0: Colab 안정
crit     = NMSELoss()
orig_state = {k: v.clone() for k, v in model.state_dict().items()}

def eval_nmse():
    total, n = 0.0, 0
    with torch.no_grad():
        for i, x in enumerate(loader):
            if i >= N_BATCHES: break
            x = x.to(device)
            total += crit(model(x), x).item()
            n += 1
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return total / max(n, 1)

def quantize_int4(w):
    qmax = 7
    scale = w.abs().max().clamp(min=1e-8) / qmax
    return (w / scale).round().clamp(-qmax, qmax) * scale

# ── FP32 baseline ─────────────────────────────────────────────
print('Measuring FP32 baseline...')
nmse_fp32 = eval_nmse()
print(f'  NMSE_fp32 = {nmse_fp32:.6f}')

# ── Build task list ───────────────────────────────────────────
# encoder.fc.weight: split into 32 chunks matching hawq fc_part0..31
FC_CHUNK = 16   # rows per chunk (matches hawq LAYER_MAP)

tasks = []  # (save_name, param, chunk_or_None)
for name, param in model.named_parameters():
    if param.dim() < 2 or 'weight' not in name:
        continue
    if name == 'encoder.fc.weight':
        n_chunks = param.shape[0] // FC_CHUNK
        for k in range(n_chunks):
            tasks.append((f'fc_part{k}', param, (k * FC_CHUNK, (k + 1) * FC_CHUNK)))
    else:
        tasks.append((name, param, None))

print(f'\nTasks to measure ({len(tasks)} total):')
for save_name, p, chunk in tasks[:5]:
    shape = list(p[chunk[0]:chunk[1]].shape) if chunk else list(p.shape)
    print(f'  {save_name:55s} {str(shape):20s}')
print(f'  ... ({len(tasks)-5} more)')

# ── Per-layer measurement ──────────────────────────────────────
print(f'\n{"─"*80}')
print(f'{"Param":55s} {"n_params":>8} {"ΔL_INT4":>12}')
print(f'{"─"*80}')

rows = []
t0   = time.time()

for idx, (save_name, param, chunk) in enumerate(tasks):
    model.load_state_dict(orig_state, strict=False)   # restore FP32

    with torch.no_grad():
        if chunk is None:
            param.data.copy_(quantize_int4(param.data.clone()))
        else:
            r0, r1 = chunk
            param.data[r0:r1].copy_(quantize_int4(param.data[r0:r1].clone()))

    nmse_q = eval_nmse()
    dL     = nmse_q - nmse_fp32
    n_p    = param[chunk[0]:chunk[1]].numel() if chunk else param.numel()

    rows.append({'param': save_name, 'n_params': n_p, 'delta_L_INT4': dL})
    elapsed = time.time() - t0
    eta     = elapsed / (idx + 1) * (len(tasks) - idx - 1)
    print(f'[{idx+1:3d}/{len(tasks)}] {save_name:55s} '
          f'{n_p:8d} {dL:12.6f}   ETA {eta:.0f}s')

model.load_state_dict(orig_state, strict=False)
print(f'\nDone in {time.time()-t0:.1f}s')

# ── Save ──────────────────────────────────────────────────────
out_path = 'results/csv/per_layer_dL.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['param', 'n_params', 'delta_L_INT4'])
    w.writeheader()
    w.writerows(rows)
print(f'Saved: {out_path}')

# ── Summary: top sensitive layers ────────────────────────────
rows_s = sorted(rows, key=lambda r: -abs(r['delta_L_INT4']))
print('\nTop 10 most sensitive layers:')
print(f'  {"param":55s} {"n_params":>8} {"ΔL_INT4":>12}')
for r in rows_s[:10]:
    print(f'  {r["param"]:55s} {r["n_params"]:8d} {r["delta_L_INT4"]:12.6f}')

# ── Auto-plot after measurement ────────────────────────────
exec(open('plot_delta_L_decomposition.py').read())
exec(open('plot_per_layer_validation.py').read())
