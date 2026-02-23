# ============================================================
# [Cell 1] 빌드 완료 후 실행 - CUDA 커널을 Google Drive에 캐싱
# ============================================================
import subprocess, shutil, os

CACHE_DIR = "/content/drive/MyDrive/MambaCompression/_kernel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 1) selective_scan_cuda 관련 .so 파일 찾기
result = subprocess.run(
    ["find", "/usr/local/lib", "-name", "*.so", "-path", "*selective_scan*"],
    capture_output=True, text=True
)
so_files = result.stdout.strip().split("\n")
so_files = [f for f in so_files if f]

if not so_files:
    # pip show으로 대체 탐색
    result = subprocess.run(
        ["find", "/usr/local/lib", "-name", "*.so"],
        capture_output=True, text=True
    )
    all_so = result.stdout.strip().split("\n")
    so_files = [f for f in all_so if "selective_scan" in f or "csm_triton" in f]

if so_files:
    for f in so_files:
        dst = os.path.join(CACHE_DIR, os.path.basename(f))
        shutil.copy2(f, dst)
        print(f"Cached: {os.path.basename(f)}")

    # 원본 경로 기록 (복원 시 필요)
    with open(os.path.join(CACHE_DIR, "paths.txt"), "w") as fp:
        for f in so_files:
            fp.write(f + "\n")
    print(f"\nDone! {len(so_files)} files cached to {CACHE_DIR}")
else:
    print("ERROR: No selective_scan .so files found. Build may not be complete.")
