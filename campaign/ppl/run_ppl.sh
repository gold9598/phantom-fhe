#!/usr/bin/env bash
# run_ppl.sh — WikiText-2 FHE PPL evaluation wrapper.
#
# Analogous to mrpc_campaign/resume.sh but for PPL windows.
# Handles: probe symlink regen, env setup, FHE launch, final aggregation.
#
# Usage:
#   bash run_ppl.sh [--pilot | --full] [--quant int32|int64] [--csv PATH]
#
# Examples:
#   bash run_ppl.sh --pilot --quant int32
#   bash run_ppl.sh --full  --quant int32 --csv /home/yongwoo-oh/mrpc_campaign/ppl_int32.csv
#
# Prerequisites (run prepare_ppl.py first):
#   python /home/yongwoo-oh/mrpc_campaign/ppl_prep/prepare_ppl.py
#
# Environment variables (can override):
#   PPL_USE_FHE=1        — run real FHE (default 0 = PT-only validation mode)
#   PPL_NUM_WINDOWS      — override window count (default from --pilot/--full)
#   QUANT_BRANCH         — git branch suffix if needed (informational only)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NUM_WINDOWS=32      # --pilot default
CSV_PATH=""         # auto-set below
USE_FHE="${PPL_USE_FHE:-1}"

REPO="/home/yongwoo-oh/phantom-fhe"
LLM_PROJECT="${REPO}/python/llm_project"
CAMPAIGN="/home/yongwoo-oh/mrpc_campaign"
PPL_PREP="${CAMPAIGN}/ppl_prep"
PPL_EVAL="${PPL_PREP}/code/ppl_eval.py"
PROBE_FULL_SRC="${CAMPAIGN}/llama_probe_full"
PROBE_FULL_DST="/tmp/llama_probe_full"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
QUANT="int32"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot)  NUM_WINDOWS=32;  shift ;;
        --full)   NUM_WINDOWS=256; shift ;;
        --quant)  QUANT="$2"; shift 2 ;;
        --csv)    CSV_PATH="$2"; shift 2 ;;
        --num-windows) NUM_WINDOWS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Override from env if set
if [[ -n "${PPL_NUM_WINDOWS:-}" ]]; then
    NUM_WINDOWS="${PPL_NUM_WINDOWS}"
fi

# Auto-set CSV path
if [[ -z "${CSV_PATH}" ]]; then
    CSV_PATH="${CAMPAIGN}/ppl_${QUANT}.csv"
fi

echo "=============================================="
echo " run_ppl.sh"
echo " quant:       ${QUANT}"
echo " num_windows: ${NUM_WINDOWS}"
echo " csv:         ${CSV_PATH}"
echo " use_fhe:     ${USE_FHE}"
echo "=============================================="

# ---------------------------------------------------------------------------
# Step 1: Regen probe symlink / hard copy if missing
# ---------------------------------------------------------------------------
echo ""
echo "[1] Checking probe data at ${PROBE_FULL_DST} ..."

if [[ ! -f "${PROBE_FULL_DST}/final_norm_g.npy" ]]; then
    echo "  probe not found at ${PROBE_FULL_DST}; symlinking from ${PROBE_FULL_SRC}"
    if [[ -d "${PROBE_FULL_SRC}" ]]; then
        # Try symlink first; fall back to cp if /tmp is on a different fs
        ln -sfn "${PROBE_FULL_SRC}" "${PROBE_FULL_DST}" 2>/dev/null || \
            cp -r "${PROBE_FULL_SRC}" "${PROBE_FULL_DST}"
        echo "  probe ready at ${PROBE_FULL_DST}"
    else
        echo "ERROR: probe source ${PROBE_FULL_SRC} not found."
        echo "  Run: python ${REPO}/python/llm_project/scripts/extract_llama_probe.py"
        exit 1
    fi
else
    echo "  probe OK (final_norm_g.npy present)"
fi

# Check lm_head_full.npy in probe (needed by ppl_eval.py via _PROBE_FULL)
if [[ ! -f "${PROBE_FULL_DST}/lm_head.npy" ]]; then
    echo "  WARNING: lm_head.npy missing from probe; ppl_eval.py will load from ppl_prep"
fi

# ---------------------------------------------------------------------------
# Step 2: Verify windows.npz + refs exist
# ---------------------------------------------------------------------------
echo ""
echo "[2] Checking ppl_prep artifacts ..."

if [[ ! -f "${PPL_PREP}/windows.npz" ]]; then
    echo "ERROR: ${PPL_PREP}/windows.npz not found."
    echo "  Run: python ${PPL_PREP}/prepare_ppl.py"
    exit 1
fi

N_REFS=$(ls "${PPL_PREP}/refs/ppl_window_"*.npz 2>/dev/null | wc -l)
echo "  windows.npz: OK"
echo "  refs present: ${N_REFS}"

if [[ "${N_REFS}" -lt "${NUM_WINDOWS}" ]]; then
    echo "  WARNING: only ${N_REFS} refs captured but requesting ${NUM_WINDOWS} windows."
    echo "  Some windows will fail. Run prepare_ppl.py first."
fi

# ---------------------------------------------------------------------------
# Step 3: Launch ppl_eval.py
# ---------------------------------------------------------------------------
echo ""
echo "[3] Launching ppl_eval.py ..."
echo "  cd ${LLM_PROJECT}"
echo ""

cd "${LLM_PROJECT}"

HF_HUB_OFFLINE=1 \
USE_BOOTSTRAP_17=1 \
AUTONOMOUS_FHE=1 \
PPL_USE_FHE="${USE_FHE}" \
python "${PPL_EVAL}" \
    --num-windows "${NUM_WINDOWS}" \
    --csv "${CSV_PATH}"

# ---------------------------------------------------------------------------
# Step 4: Aggregate + print final PPL
# ---------------------------------------------------------------------------
echo ""
echo "[4] Final PPL aggregation ..."
HF_HUB_OFFLINE=1 \
python "${PPL_EVAL}" \
    --summary \
    --csv "${CSV_PATH}"

echo ""
echo "=============================================="
echo " Done. Results in ${CSV_PATH}"
echo "=============================================="
