# ============================================================
# Colab Setup - MambaIC (with kernel caching)
# 매 세션 시작 시 이 셀 하나만 실행하면 됨
# ============================================================
import os, subprocess, shutil, sys

PROJECT_ROOT = "/content/drive/MyDrive/MambaCompression"
MAMBAIC_ROOT = os.path.join(PROJECT_ROOT, "MambaIC")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "_kernel_cache")

# --- Drive mount (VS Code에서는 이미 마운트, Colab 브라우저에서만 필요) ---
if not os.path.isdir(PROJECT_ROOT):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        pass

assert os.path.isdir(MAMBAIC_ROOT), f"MambaIC not found: {MAMBAIC_ROOT}"

# ── 1. Core pip packages (mamba-ssm 불필요, ss2d는 VMamba 커널만 사용) ──
print("=== 1. Core Dependencies ===")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "einops", "scipy", "tqdm", "thop", "fvcore", "pybind11"])

# ── 2. VMamba CUDA kernel (cached or build) ──
print("\n=== 2. VMamba CUDA Kernel (ss2d) ===")

# --- GPU arch check: clear cache if GPU changed since last build ---
import torch
_prop = torch.cuda.get_device_properties(0)
_curr_arch = f"sm_{_prop.major}{_prop.minor}"
print(f"Current GPU: {_prop.name} ({_curr_arch})")

_arch_file = os.path.join(CACHE_DIR, "gpu_arch.txt")
if os.path.isdir(CACHE_DIR):
    if not os.path.isfile(_arch_file):
        # gpu_arch.txt 없는 old cache → 아키텍처 모름 → 강제 재빌드
        print(f"Cache has no GPU arch record → clearing for rebuild ({_curr_arch})")
        shutil.rmtree(CACHE_DIR)
    else:
        with open(_arch_file) as _f:
            _cached_arch = _f.read().strip()
        if _cached_arch != _curr_arch:
            print(f"GPU arch mismatch: cache={_cached_arch} ≠ current={_curr_arch} → clearing for rebuild")
            shutil.rmtree(CACHE_DIR)
        else:
            print(f"Cache arch matches current GPU ({_curr_arch}) ✓")

cache_paths_file = os.path.join(CACHE_DIR, "paths.txt")
cached_so_files  = [f for f in os.listdir(CACHE_DIR) if f.endswith(".so")] if os.path.isdir(CACHE_DIR) else []

if cached_so_files and os.path.isfile(cache_paths_file):
    # --- Restore from cache ---
    print(f"Cache found! Restoring {len(cached_so_files)} kernel files...")
    with open(cache_paths_file) as fp:
        original_paths = [l.strip() for l in fp if l.strip()]

    for orig_path in original_paths:
        fname = os.path.basename(orig_path)
        src = os.path.join(CACHE_DIR, fname)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(orig_path), exist_ok=True)
            shutil.copy2(src, orig_path)
            print(f"  Restored: {fname} -> {orig_path}")

    # Import 확인 (arch mismatch는 실제 실행 시에만 발생하므로 위의 arch 체크로 사전 차단)
    try:
        import selective_scan_cuda_oflex
        print(f"selective_scan_cuda_oflex imported OK ({_curr_arch})")
    except ImportError as _e:
        print(f"WARN: import failed ({_e}) → will rebuild")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        cached_so_files = []  # trigger rebuild below

if not cached_so_files:
    # --- Build from source ---
    print("No cache found. Building from source (~2min)...")
    VMAMBA_DIR = os.path.join(PROJECT_ROOT, "VMamba")
    if not os.path.isdir(VMAMBA_DIR):
        subprocess.run(["git", "clone", "https://github.com/MzeroMiko/VMamba.git", VMAMBA_DIR])

    SELECTIVE_SCAN_DIR = os.path.join(VMAMBA_DIR, "kernels", "selective_scan")
    if os.path.isdir(SELECTIVE_SCAN_DIR):
        subprocess.run([sys.executable, "-m", "pip", "install",
                        SELECTIVE_SCAN_DIR, "--no-build-isolation", "-q"])

        # Auto-cache after build
        print("\nAuto-caching built kernels...")
        exec(open(os.path.join(PROJECT_ROOT, "cache_kernel.py")).read())

        # Record GPU arch so next session can detect mismatch
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "gpu_arch.txt"), "w") as _f:
            _f.write(_curr_arch)
        print(f"GPU arch recorded: {_curr_arch}")
    else:
        print(f"ERROR: {SELECTIVE_SCAN_DIR} not found")

print("\n=== Setup Complete ===")
print(f"Project: {PROJECT_ROOT}")
