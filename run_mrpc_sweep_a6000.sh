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
    # CUDA_ARCH:
    #   80 — A100 (Ampere data-center)   [DEFAULT — A100×4 sweep box]
    #   86 — A6000 / RTX 3090 (Ampere consumer)
    #   89 — RTX 4090 (Ada Lovelace)
    #   90 — H100 (Hopper)
    #  120 — RTX 5090 (Blackwell)
    CUDA_ARCH=${CUDA_ARCH:-80}
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DPHANTOM_USE_CUDA_PTX=ON \
        -DPHANTOM_ENABLE_PYTHON_BINDING=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
    cmake --build build --target pyPhantom -j"$(nproc)"
fi

# ---- Auto-detect GPU count and dispatch ----
# Auto-detect the number of available GPUs (override with NUM_GPUS env var)
if [ -z "${NUM_GPUS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
    else
        NUM_GPUS=1
    fi
fi
echo "[setup] Detected $NUM_GPUS GPU(s) (override with NUM_GPUS env var)"

# Dispatch to parallel or single-GPU sweep
# Set SINGLE_GPU=1 to force the single-GPU path for debugging
if [ "${SINGLE_GPU:-0}" = "1" ] || [ "$NUM_GPUS" = "1" ]; then
    echo "[setup] Using single-GPU mrpc_sweep.py"
    exec "$PYTHON" -u python/llm_project/mrpc_sweep.py \
        --start 0 --end 408 \
        2>&1 | tee "$LOG_DIR/mrpc_sweep_full.log"
else
    echo "[setup] Using $NUM_GPUS-GPU parallel sweep"
    exec "$PYTHON" -u python/llm_project/mrpc_sweep_parallel.py \
        --start 0 --end 408 --num-gpus "$NUM_GPUS" \
        2>&1 | tee "$LOG_DIR/mrpc_sweep_parallel.log"
fi
