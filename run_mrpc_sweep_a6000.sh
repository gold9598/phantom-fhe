#!/usr/bin/env bash
# Launch the 408-MRPC dev sweep on A6000 (4× A6000, 256GB RAM box).
#
# Uses the rp_indep IRP cache (~3GB/layer × 32 layers = ~96GB host RAM)
# so each layer's Wo/Wgate+Wup-packed/Wdown plaintexts are encoded once
# at startup and reused across all 408 examples. Only Wq (R_P-dependent)
# is re-encoded per example.
#
# Estimated wall time on a SINGLE A6000 (with cache hot):
#   first example: ~12-15 min (cold cache, encodes all R_P-indep IRPs)
#   subsequent:    ~5-8 min/example (Wq encode + FHE compute only)
#   408 examples ≈ 30-50 hours
#
# 4× parallel option (uncomment the parallel section to use all GPUs):
#   Each GPU runs a disjoint range of 102 examples. Each gets its own
#   ~96GB RAM cache. Verify total RAM fits (4 × 96 = 384GB).

set -euo pipefail

# Resolve REPO relative to this script so the launcher works on any host.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO=${REPO:-$SCRIPT_DIR}
PYTHON=${PYTHON:-/home/yongwoo-oh/desilofhe/bin/python3}
LOG_DIR=${LOG_DIR:-/tmp}
CSV_PATH=${CSV_PATH:-/tmp/mrpc_sweep_results.csv}

cd "$REPO"

# Ensure /tmp/llama_probe_full/ probe data exists (weight .npy files,
# rope_*.npy, meta.json). One-time extract from HF LLaMA-3.1-8B.
PROBE_DIR=${PROBE_DIR:-/tmp/llama_probe_full}
if [ ! -f "$PROBE_DIR/meta.json" ]; then
    echo "[setup] Probe data not found at $PROBE_DIR — extracting from HF model..."
    "$PYTHON" -u python/llm_project/setup_probe_data.py --probe-dir "$PROBE_DIR"
fi

# Ensure pyPhantom .so is built. The Python binding is built as
# build/lib/pyPhantom.cpython-<ver>-<arch>.so by the repo's CMake.
if [ -n "${REBUILD:-}" ] || ! ls build/lib/pyPhantom.cpython-*.so >/dev/null 2>&1; then
    if [ -n "${REBUILD:-}" ]; then
        echo "[setup] REBUILD=1: wiping build/ for clean rebuild..."
        rm -rf build
    else
        echo "[setup] pyPhantom .so not found in $REPO/build/lib — building..."
    fi
    # Init the pybind11 submodule (the build needs python/pybind11/CMakeLists.txt).
    if [ ! -f python/pybind11/CMakeLists.txt ]; then
        echo "[setup] initializing git submodules (pybind11)..."
        git submodule update --init --recursive
    fi
    # CUDA_ARCH default 86 (Ampere — A6000/A100). Override with
    # CUDA_ARCH=120 for 5090 (Blackwell), 89 for 4090 (Ada), etc.
    CUDA_ARCH=${CUDA_ARCH:-86}
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DPHANTOM_USE_CUDA_PTX=ON \
        -DPHANTOM_ENABLE_PYTHON_BINDING=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
    cmake --build build --target pyPhantom -j"$(nproc)"
fi

# ---- Single-GPU (default): full 0..407 sweep ----
exec "$PYTHON" -u python/llm_project/mrpc_sweep.py \
    --start 0 --end 408 \
    2>&1 | tee "$LOG_DIR/mrpc_sweep_full.log"

# ---- 4× A6000 parallel option ----
# Uncomment below + comment the single-GPU block above. Splits 408 into
# 4 disjoint ranges (102 each). Each process gets its own CUDA device
# and its own results CSV file; merge afterwards.
#
# for i in 0 1 2 3; do
#     start=$((i * 102))
#     end=$((start + 102))
#     [ $end -gt 408 ] && end=408
#     CUDA_VISIBLE_DEVICES=$i \
#     CSV_PATH=$LOG_DIR/mrpc_sweep_gpu${i}.csv \
#     "$PYTHON" -u python/llm_project/mrpc_sweep.py \
#         --start $start --end $end \
#         > "$LOG_DIR/mrpc_sweep_gpu${i}.log" 2>&1 &
# done
# wait
# # Merge CSVs (skip duplicated headers):
# {
#     head -1 "$LOG_DIR/mrpc_sweep_gpu0.csv"
#     for i in 0 1 2 3; do tail -n +2 "$LOG_DIR/mrpc_sweep_gpu${i}.csv"; done
# } > "$CSV_PATH"
# "$PYTHON" -u python/llm_project/mrpc_sweep.py --summary
