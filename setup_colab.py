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

    # Verify import
    try:
        import selective_scan_cuda_oflex
        print("selective_scan_cuda_oflex OK")
    except ImportError:
        print("WARN: import failed, will rebuild")
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
    else:
        print(f"ERROR: {SELECTIVE_SCAN_DIR} not found")

print("\n=== Setup Complete ===")
print(f"Project: {PROJECT_ROOT}")
