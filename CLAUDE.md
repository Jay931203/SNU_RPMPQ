# SNU_RPMPQ - Rate-adaptive Precision Mixed-Precision Quantization for CSI Compression

## Project Overview

Massive MIMO FDD 시스템에서 CSI (Channel State Information) 피드백 압축을 위한 연구 프로젝트.
세 가지 Neural CSI Encoder 아키텍처를 비교하고, Mamba 기반 모델에 Mixed-Precision Quantization + Rate-Adaptive 전략을 적용하는 것이 핵심 연구.

**연구 핵심:** Mamba SSM 기반 CSI 압축 + HAWQ Mixed-Precision Quantization + RANC (Rate-Adaptive Neural Compression)

**데이터셋:** COST2100 Channel Model (Indoor/Outdoor), CSI Matrix shape: `(N, 2, 32, 32)` (Real/Imag, 32 antennas, 32 subcarriers)

**GitHub:** https://github.com/Jay931203/SNU_RPMPQ

---

## Directory Structure

```
MambaCompression/
├── CLAUDE.md                    # 이 파일
├── MS_Project.ipynb             # Colab 환경 셋업 노트북 (VMamba CUDA 커널 빌드 등)
├── .gitignore                   # *.mat 데이터만 제외 (모델 웨이트는 포함)
│
├── MambaIC/                     # [주 연구] Mamba 기반 CSI 압축 + 양자화 + RANC
├── Python_CsiNet-master/        # [베이스라인] CNN 기반 CsiNet (Keras/TF/PyTorch)
└── TransNet-master/             # [베이스라인] Transformer 기반 TransNet (PyTorch)
```

---

## MambaIC/ - 주 연구 (Mamba-based CSI Compression)

### 핵심 아키텍처

**MambaAE (Autoencoder)** - `MambaAE.py`
- Encoder: Conv → 3 stages (conv + Mamba blocks) → Tanh
- Decoder: Symmetric deconv stages → Sigmoid
- Bottleneck: N=128 channels, M=32 latent dim
- Depths: [2, 2, 4, 2] blocks per stage, Drop Path Rate 0.1

**Scan Mode 변형** (MambaAE의 핵심 실험 축):
| Scan Mode | 설명 |
|-----------|------|
| `chunked_ss2d` | 2D Selective Scan (VSSBlock 사용, 4방향) |
| `chunked_hilbert` | Hilbert 공간채움곡선 순서 + 1D Mamba |
| `chunked_raster` | Raster scan (row-major) + 1D Mamba |
| `chunked_biaxial` | Row+Col 양방향 1D Mamba |
| `chunked_raster_conv` | Raster + Conv 변형 |
| `chunked_raster_conv_shuffle` | Raster + Conv + Pixel Shuffle |
| `chunked_raster_patch` | Patch 기반 Raster |

**ModularModels.py** - Encoder/Decoder 조합 프레임워크:
- Encoders: `MambaEncoder`, `CsiNetEncoder`, `TransNetEncoder`, `MobileNetEncoder`
- Decoders: `MambaTransDecoder`, `CsiNetDecoder`, `TransNetDecoder`, `MobileNetDecoder`
- `ModularAE` 클래스로 자유 조합 가능

**MambaIC (Image Compression)** - `models/MambaIC.py`
- CVPR 2025 논문 구현 (Learned Image Compression)
- VSSBlock + Swin Attention + Checkboard Context Modeling
- 5-slice progressive decoding, Entropy Bottleneck
- Latent M=320 (5 slices × 64 channels)

### 양자화 & 압축 기법

1. **STE (Straight-Through Estimator):** Forward에서 양자화, Backward에서 full precision gradient
2. **Dynamic Activation Quantization:** 비대칭 min-max, layer별 bit-width 설정 가능
3. **UniformQuantizer_0_1:** Latent space 양자화 (configurable bits)
4. **HAWQ (Hessian-Aware Quantization):** Layer별 sensitivity 분석 → ILP로 최적 bit 할당
5. **Mixed-Precision Policy LUT:** `mp_policy_lut_mamba.csv`, `mp_policy_lut_mamba_pruned.csv`
6. **BOPs (Bit Operations):** `Params × Weight_Bits × Activation_Bits` 로 효율성 측정

### RANC (Rate-Adaptive Neural Compression) - `train_ae.py`

- Offline: Sparsity bin별 reliability table (mu, sigma) 프로파일링
- Online: Risk estimation = mean + λ×std → target NMSE 만족하는 policy 선택
- 입력 특성(sparsity)에 따라 양자화 정책 적응적 선택

### 주요 실험 (eval_ae.py)

| 실험 | 내용 | 결과 파일 |
|------|------|-----------|
| Exp 1 | Blind BOPs Sweep (양자화 정책 탐색) | `exp1_pareto_accuracy_mamba.png` |
| Exp 3 | Sparsity별 NMSE, CDF, Outage 분석 | `exp3_nmse_*.png` |
| Exp 3.5 | Communication Rate 시뮬레이션 (Shannon capacity 비교) | `exp3_5_rate_final_mamba.png` |
| Exp 4 | RANC Outage/Rate Multi-policy | `exp4_final_ranc_*.png` |

### 학습 설정 (train_ae.py)

- Loss: MSE, Optimizer: Adam (betas=0.9, 0.99), Gradient Clipping: 1.0
- 정규화: 0-Mean (subtract 0.5) 또는 1-Power (divide by RMS std)
- Metric: NMSE_dB = 10×log10(MSE / Power)

### Saved Models 구조

**CSI AE 변형 (M=32):**
```
saved_models/
├── csi_mamba_AE_M32_bits0_chunked_ss2d/        # 2D scan (기본)
├── csi_mamba_AE_M32_bits0_chunked_hilbert/      # Hilbert curve
├── csi_mamba_AE_M32_bits0_chunked_raster/       # Raster scan
├── csi_mamba_AE_M32_bits0_chunked_biaxial/      # 양방향
├── csi_mamba_AE_M32_bits0_chunked_raster_conv/
├── csi_mamba_AE_M32_bits0_chunked_raster_conv_shuffle/
├── csi_mamba_AE_M32_bits0_chunked_raster_patch/
├── csi_mamba_AE_M32_bits0_chunked_ss2d_222_out/ # ss2d outdoor
├── mamba_chunked_ss2d_hybrid_M32/               # Hybrid stride
└── mamba_chunked_ss2d_default_M32_out/          # Default stride outdoor
```

**Modular Model 조합 (dim=512):**
```
saved_models/
├── csinet_csinet_dim512/                        # CsiNet baseline
├── csinet_transnet_L2_dim512/                   # CsiNet enc + Trans dec
├── transnet_transnet_L2_dim512/                 # Trans + Trans
├── mamba_transnet_L2_dim512_baseline/           # Mamba enc + Trans dec
├── mamba_transnet_L2_dim512_epoch300_mamba8_trans2048/
├── mamba_transnet_L2_dim512_proj/
├── mamba_transnet_L2_dim512_M64_conv4/
├── mamba_transnet_L3_dim512/
├── mobilenet_mamba_dim512/                      # MobileNet + Mamba
├── mobilenet_transnet_L1~L4_dim512/             # MobileNet + Trans (L1-L4)
```

### 핵심 파일

| 파일 | 역할 |
|------|------|
| `MambaAE.py` | Mamba Autoencoder 모델 정의 (scan modes, quantization) |
| `ModularModels.py` | Encoder/Decoder 모듈 조합 프레임워크 |
| `models/MambaIC.py` | CVPR 2025 Mamba Image Compression 모델 |
| `models/VSS_module.py` | SS2D (Selective Scan 2D) CUDA 커널 래퍼 |
| `train_ae.py` | AE 학습 + RANC + Exp 전체 파이프라인 (~100KB, 핵심 스크립트) |
| `eval_ae.py` | 평가 + 실험 결과 생성 |
| `train.py` / `eval.py` | Image Compression (MambaIC) 학습/평가 |
| `hawq_importance_split.csv` | HAWQ layer별 sensitivity 결과 |
| `mp_policy_lut_mamba_pruned.csv` | Pareto-optimal Mixed-Precision 정책 테이블 |
| `fitting_raw_data_mamba.csv` | 실험 raw data (gitignore에서 제외됨, .npz도) |

---

## Python_CsiNet-master/ - CNN 베이스라인

### 아키텍처 (CsiNet)

IEEE WCL 2018, Chao-Kai Wen et al.

```
Encoder: Conv2D(2, 3×3) → BN → LeakyReLU → Flatten(2048) → Dense(encoded_dim)
Decoder: Dense(2048) → Reshape → [RefineBlock × 2] → Conv2D(sigmoid)
RefineBlock: Conv(8) → BN → LeakyReLU → Conv(16) → BN → LeakyReLU → Conv(2) → BN + Skip
```

**압축률:** 1/4 (dim=512), 1/16 (128), 1/32 (64), 1/64 (32)

### 프레임워크 구현

| 폴더/파일 | 프레임워크 | 설명 |
|-----------|-----------|------|
| `CsiNet_train.py` | Keras (TF1.x) | 원본 학습 코드 |
| `CS-CsiNet_train.py` | Keras (TF1.x) | Decoder-only 변형 (random projection) |
| `channels_last/` | TF2.x | channels_last 포맷 업데이트 |
| `Improvement/CsiNet_Train.py` | PyTorch | PyTorch 재구현 |
| `csinet_onlytest.py` | TF/Keras | **커스텀 양자화 분석** (HAWQ + ILP + Pareto) |

### 양자화 분석 (csinet_onlytest.py)

MambaIC의 양자화 프레임워크와 동일한 방법론을 CsiNet에 적용:
- HAWQ Hessian trace 분석
- ILP solver로 BOPs 제약 하 최적 bit 할당
- KL divergence 기반 후보 선택
- 결과: `mp_policy_lut_csinet_outdoor_strict.csv`, `mp_policy_pareto_outdoor_strict.png`

### Saved Models

```
saved_model/           # Keras .h5 (CsiNet + CS-CsiNet, indoor/outdoor, dim 32~512)
channels_last/keras/   # Keras .h5 (channels_last 포맷)
```

---

## TransNet-master/ - Transformer 베이스라인

### 아키텍처 (TransNet)

IEEE WCL 2022, Cui et al.

```
Input (2, 32, 32) → Reshape (2048/d_model, d_model)
  → TransformerEncoder (2 layers, 2 heads) → FC_enc(2048 → 2048/CR)
  → FC_dec(2048/CR → 2048) → TransformerDecoder (2 layers, 2 heads, cross-attn)
  → Reshape (2, 32, 32)
```

- `d_model=64`, `nhead=2`, `num_layers=2`, `dim_feedforward=2048`
- 압축률: 1/4 ~ 1/64 (FC bottleneck으로 조절)
- FLOPs: ~33-36M (CR에 거의 무관)

### 성능 (README 기준)

| Scenario | CR=1/4 | CR=1/16 | CR=1/64 |
|----------|--------|---------|---------|
| Indoor | -29.22 dB | -14.98 dB | -5.77 dB |
| Outdoor | -13.99 dB | -6.90 dB | -2.20 dB |

### 양자화 분석 (csinet_onlytest.py)

CsiNet 모델에 대한 양자화 시뮬레이션 스크립트 (TransNet 자체가 아닌 비교용):
- FP32/FP16/INT8/FP8/FP4 weight quantization
- Activation quantization (uniform, configurable bits)
- TFLite 변환 + CPU latency 벤치마크
- Sum Rate 계산 (MMSE precoding)

### Saved Models

```
checkpoints/
├── 4_in.pth    # Indoor, CR=1/4
└── 4_out.pth   # Outdoor, CR=1/4
```

---

## Metrics

| Metric | 수식 | 설명 |
|--------|------|------|
| NMSE (dB) | `10×log10(MSE / Power)` | 낮을수록 좋음 (더 음수) |
| Rho (ρ) | 주파수 도메인 cross-correlation | [0,1], 높을수록 좋음 |
| BOPs | `Σ(Params_i × W_bits × A_bits)` | Bit Operations, 낮을수록 효율적 |
| Outage Prob | `P(NMSE > threshold)` | 낮을수록 안정적 |
| Sum Rate | `Σ log2(1 + SINR)` bps/Hz | 높을수록 좋음 |

## Dependencies

**MambaIC (핵심):** PyTorch 2.1+, mamba-ssm, einops, timm, compressai, thop, pulp (ILP), scipy, hilbertcurve
**CsiNet:** TensorFlow/Keras (원본), PyTorch (Improvement)
**TransNet:** PyTorch 1.6+, thop, tensorboardx, scipy

## Data (gitignore로 제외됨)

`MambaIC/data/` 아래 COST2100 `.mat` 파일 (~6.4GB 총):
- `DATA_Htrain[in|out].mat` - Training
- `DATA_Hval[in|out].mat` - Validation
- `DATA_Htest[in|out].mat` - Test (compressed)
- `DATA_HtestF[in|out]_all.mat` - Test (full frequency domain)
- `A32.mat`, `A64.mat`, `A128.mat`, `A512.mat` - Random projection matrices

## Colab 환경 설정

`MS_Project.ipynb`에서 VMamba CUDA 커널 빌드 + mamba-ssm + causal-conv1d 설치.
Google Drive 마운트 경로: `/drive/MyDrive/MyProjects/01_CL_PEFT_WIRELESS/`
