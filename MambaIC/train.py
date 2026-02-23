# 파일 이름: train_ae.py
# (CsiNet 벤치마크용. "0-Mean, 1-Power" 정규화 사용)
# (❗️ 이 파일은 수정할 필요가 없습니다 ❗️)

import os, sys
# ✅ 현재 파일 기준으로 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

models_path = os.path.join(project_root, "models")
if models_path not in sys.path:
    sys.path.append(models_path)
    
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from tqdm import tqdm
import scipy.io as sio
import math
import logging
import sys
import time
from datetime import datetime
import torch.nn.functional as F # (Clamp를 위해 임포트)

# --- (핵심!) MambaAE 모델 임포트 ---
try:
    from MambaAE import MambaAE
    print("from MambaAE import MambaAE 임포트 성공.")
except ImportError as e:
    print(f"오류: {e}. MambaAE.py 파일이 프로젝트 루트에 있는지 확인하세요.")
    exit()

# ----------------------------------------------------
# 1. CsiDataset (수정: Power-Scaling 정규화)
# (❗️ CsiNet 벤치마크를 위해 "0-Mean, 1-Power"를 올바르게 구현한 버전)
# ----------------------------------------------------
class CsiDataset(Dataset):
    def __init__(self, mat_file_path, data_key='HT', max_samples=None, normalization_factor=None):
        super(CsiDataset, self).__init__()
        self.mat_file_path = mat_file_path
        print(f"Loading .mat file: {self.mat_file_path} (key: {data_key}) ...")
        try:
            mat_data = sio.loadmat(self.mat_file_path)
            self.csi_data = np.array(mat_data[data_key], dtype=np.float32)
        except Exception as e:
            print(f"Error loading {mat_file_path}: {e}")
            raise
        print(f"Data loaded (flat). Shape: {self.csi_data.shape}")
        
        # ❗️(수정) Reshape을 나중에 하고, 파워 계산은 전체 데이터로
        try:
            csi_data_reshaped = np.reshape(self.csi_data, (self.csi_data.shape[0], 2, 32, 32))
        except Exception as e:
            print(f"Reshape Error: {e}"); raise
        
        # 1. 0-Mean (Re-center)
        csi_data_centered = csi_data_reshaped - 0.5
        
        # 2. Power-Scaling (CsiNet 벤치마크 방식)
        if normalization_factor is None:
            # (훈련셋) 전체 데이터셋의 평균 파워(분산) 계산
            avg_power = np.mean(csi_data_centered ** 2)
            self.scale_factor = np.sqrt(avg_power)
            print(f"Calculated 0-Mean data. Avg Power: {avg_power:.4f}, Scale Factor (std): {self.scale_factor:.4f}")
        else:
            # (테스트셋) 훈련셋의 스케일 팩터 사용
            self.scale_factor = normalization_factor
            print(f"Using Train Scale Factor (std): {self.scale_factor:.4f}")
            
        # 3. 데이터 정규화 적용 (❗️ Z-Score 데이터 생성)
        self.csi_data_normalized = csi_data_centered / self.scale_factor
        
        # 4. (수정) max_samples 적용 (정규화 *이후에*)
        if max_samples is not None:
            if self.csi_data_normalized.shape[0] > max_samples:
                self.csi_data = self.csi_data_normalized[:max_samples]
                print(f"--- TOY PROJECT MODE: 데이터가 {max_samples}개로 제한되었습니다 ---")
            else:
                self.csi_data = self.csi_data_normalized
        else:
            self.csi_data = self.csi_data_normalized
            
        print(f"Data Normalized (0-Mean, 1-Power). Min: {np.min(self.csi_data):.4f}, Max: {np.max(self.csi_data):.4f}, Mean: {np.mean(self.csi_data):.4f}, Std: {np.std(self.csi_data):.4f}")

    def __len__(self):
        return self.csi_data.shape[0]

    def __getitem__(self, idx):
        csi_sample = self.csi_data[idx] # (2, 32, 32), 0-Mean, 1-Power (Z-Score)
        return torch.from_numpy(csi_sample) 

# --- (AverageMeter는 동일) ---
class AverageMeter:
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --- (NMSE 계산 함수는 벤치마크와 100% 동일) ---
def calculate_nmse_db(original_data, reconstructed_data):
    """
    Calculates NMSE in dB on 0-mean, 1-power normalized data.
    NMSE_dB = 10 * log10( sum( (d - d_hat)^2 ) / sum( d^2 ) )
    (이제 NMSE_dB = 10 * log10(MSE) 와 거의 동일)
    """
    original_power = torch.sum(original_data**2, dim=[1, 2, 3])
    mse = torch.sum((original_data - reconstructed_data)**2, dim=[1, 2, 3])
    
    valid_indices = original_power > 1e-8
    if torch.sum(valid_indices) == 0: return torch.tensor(-100.0)
    
    nmse = torch.mean(mse[valid_indices] / original_power[valid_indices])
    
    if nmse.item() <= 1e-10: return torch.tensor(-100.0)
    nmse_db = 10 * torch.log10(nmse)
    return nmse_db

# --- (옵티마이저) ---
def configure_optimizer_ae(net, args):
    # 🔧 (수정) 제안된 Adam Beta 값 적용
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    return optimizer

# 🔧 (신규) Warm-up 메시지 중복 방지 헬퍼
_WARMUP_PRINTED_TRAIN = False
_WARMUP_PRINTED_VAL = False

# 🔧 (신규) nn.DataParallel을 처리하는 속성 설정 헬퍼
def set_quant_status(model, disable_quant: bool):
    """모델이 DataParallel로 래핑되었는지 확인하고 disable_quant 속성을 올바르게 설정합니다."""
    if isinstance(model, nn.DataParallel):
        model.module.disable_quant = disable_quant
    else:
        model.disable_quant = disable_quant

# --- (학습 함수: 🔧 Warm-up 플래그 설정 수정) ---
def train_one_epoch_ae(model, mse_loss_fn, train_dataloader, optimizer, epoch, clip_max_norm, args):
    model.train()
    
    # 🔧 (신규) Quantization Warm-up 플래그 설정 (args.warmup_epochs 사용)
    global _WARMUP_PRINTED_TRAIN, _WARMUP_PRINTED_VAL
    if epoch < args.warmup_epochs:
        set_quant_status(model, True) # 🔧 DataParallel 버그 수정
        if not _WARMUP_PRINTED_TRAIN:
            print(f"--- [Train] Quantization WARM-UP enabled (Epoch < {args.warmup_epochs}) ---", flush=True)
            _WARMUP_PRINTED_TRAIN = True
    else:
        set_quant_status(model, False) # 🔧 DataParallel 버그 수정
        if _WARMUP_PRINTED_TRAIN: # 웜업이 끝나는 시점에 한 번만 출력
            print(f"--- [Train] Quantization WARM-UP disabled (Epoch >= {args.warmup_epochs}) ---", flush=True)
            _WARMUP_PRINTED_TRAIN = False # 플래그 리셋
            _WARMUP_PRINTED_VAL = False # 검증 플래그도 함께 리셋
    
    device = next(model.parameters()).device
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for d in pbar:
        d = d.to(device) # d는 이제 0-mean, 1-power
        optimizer.zero_grad()
        x_hat = model(d) # x_hat도 0-mean, 1-power (BatchNorm이 스케일링)
        loss = mse_loss_fn(x_hat, d)
        if torch.isnan(loss):
            print("Warning: Loss is NaN, skipping batch"); continue
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        pbar.set_postfix(MSE_Loss=f"{loss.item():.8f}")

# ----------------------------------------------------
# 2. test_epoch_ae (🔧 Warm-up 플래그 설정 수정)
# ----------------------------------------------------
def test_epoch_ae(epoch, test_dataloader, model, mse_loss_fn, args): # 🔧 args 추가
    model.eval()
    
    # 🔧 (치명적 버그 수정) 검증 함수에도 Warm-up 플래그를 동일하게 적용
    global _WARMUP_PRINTED_VAL
    if epoch < args.warmup_epochs:
        set_quant_status(model, True) # 🔧 DataParallel 버그 수정
        if not _WARMUP_PRINTED_VAL:
            print(f"--- [Val] Quantization WARM-UP enabled (Epoch < {args.warmup_epochs}) ---", flush=True)
            _WARMUP_PRINTED_VAL = True
    else:
        set_quant_status(model, False) # 🔧 DataParallel 버그 수정
        
    device = next(model.parameters()).device
    mse_loss = AverageMeter()
    nmse_db = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc=f"Epoch {epoch+1} [Val]")
        for d in pbar:
            d = d.to(device) # d는 0-mean, 1-power
            x_hat = model(d) # x_hat은 0-mean, 1-power
            
            # ❗️ (수정) clamp 제거
            
            loss_val = mse_loss_fn(x_hat, d)
            
            original_csi_norm = d
            decoded_csi_norm = x_hat
            
            nmse_val = calculate_nmse_db(original_csi_norm, decoded_csi_norm)
            
            mse_loss.update(loss_val.item())
            nmse_db.update(nmse_val.item())
            
            pbar.set_postfix(NMSE_dB=f"{nmse_db.avg:.4f}", MSE=f"{mse_loss.avg:.8f}")
            
    print(f"--- [결과] Epoch {epoch+1} ---")
    print(f"  Test MSE (성능): {mse_loss.avg:.8f}")
    print(f"  Test NMSE (성능): {nmse_db.avg:.4f} dB")
    return nmse_db.avg

# --- (main 함수는 동일) ---
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Mamba-AE (Fixed Rate) for CSI Compression.")
    parser.add_argument("--train_path", type=str, default="data/DATA_Htrainin.mat")
    parser.add_argument("--train_key", type=str, default="HT")
    parser.add_argument("--test_path", type=str, default="data/DATA_Htestin.mat")
    parser.add_argument("--test_key", type=str, default="HT")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--M", type=int, default=320, help="Latent channels (CR control)")
    parser.add_argument("--bits", type=int, default=8, help="Number of quantization bits for the latent space.")
    parser.add_argument("--depths", nargs='+', type=int, default=[2, 2, 9, 2])
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16) 
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--lr_epoch", nargs='+', type=int, default=[80, 100])
    # 🔧 (신규) Warm-up Epochs 인자 추가
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of epochs to train without quantization.")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    print(f"--- Mamba-AE (Fixed Rate) Training (2-Ch): {datetime.now()} ---")
    print(f"❗️ Fixed Rate (CsiNet-style). Latent M={args.M}, Bits={args.bits}")
    print(f"❗️ Normalization: 0-Mean, 1-Power (CsiNet Benchmark Standard)")
    print(f"❗️ Quantization Warm-up: {args.warmup_epochs} Epochs")
    
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. 데이터 로더 (수정: 훈련셋/테스트셋 정규화) ---
    print("Loading datasets...")
    # ❗️ (수정) 훈련셋 로드
    train_dataset = CsiDataset(args.train_path, args.train_key, max_samples=300)
    # ❗️ (수정) 훈련셋에서 계산된 스케일 팩터 가져오기
    train_scale_factor = train_dataset.scale_factor 
    # ❗️ (수정) 테스트셋 로드 시 스케일 팩터 전달
    test_dataset = CsiDataset(args.test_path, args.test_key, max_samples=100, normalization_factor=train_scale_factor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda")) 

    # --- 2. 모델 로드 (MambaAE) ---
    print(f"Creating MambaAE model (N={args.N}, M={args.M}, depths={args.depths}, bits={args.bits})")
    net = MambaAE(depths=args.depths, N=args.N, M=args.M, bits=args.bits)
    net = net.to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (nn.DataParallel)")
        net = nn.DataParallel(net)

    # --- 3. 옵티마이저 및 손실 함수 (MSE) ---
    optimizer = configure_optimizer_ae(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_epoch, gamma=0.1)
    mse_loss_fn = nn.MSELoss() 

    # --- 4. 체크포인트 ---
    last_epoch = 0
    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # 🔧 DataParallel 로드 호환
        state_dict = checkpoint["state_dict"]
        if isinstance(net, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not isinstance(net, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # --- 5. 학습 및 평가 루프 ---
    best_nmse = float("inf")
    # 🧩 (수정) 저장 경로에 비트 수 추가
    save_path = f"saved_models/csi_mamba_AE_M{args.M}_bits{args.bits}/" 
    if args.save:
        os.makedirs(save_path, exist_ok=True) 

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        train_one_epoch_ae(
            net, mse_loss_fn, train_dataloader, optimizer, epoch, args.clip_max_norm, args
        )
        
        # 🔧 (수정) test_epoch_ae에 args 전달
        nmse_db = test_epoch_ae(epoch, test_dataloader, net, mse_loss_fn, args)
        lr_scheduler.step()

        is_best = nmse_db < best_nmse
        best_nmse = min(nmse_db, best_nmse)

        if args.save:
            # 🔧 DataParallel 저장 호환
            state = {"epoch": epoch, "nmse_db": nmse_db,
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict()}
            if isinstance(net, nn.DataParallel):
                state["state_dict"] = net.module.state_dict()
            else:
                state["state_dict"] = net.state_dict()
                
            torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
            if is_best:
                print(f"  ✨ 최고 NMSE 갱신: {best_nmse:.4f} dB. 모델 저장.")
                torch.save(state, os.path.join(save_path, "checkpoint_best.pth.tar"))

if __name__ == "__main__":
    main(sys.argv[1:])