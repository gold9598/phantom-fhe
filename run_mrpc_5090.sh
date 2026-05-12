#!/usr/bin/env bash
# Run the MRPC sweep on this 5090 box (single GPU, but using the parallel
# sweep code with NUM_GPUS=1 so we get all the pre-loading optimizations:
#   - Pre-encoded rp_indep cache (disk-cached)
#   - Pre-loaded layer weights (Wq/Wk/Wv/g1/g2 subset)
#   - Pre-computed per-layer calibration (replaces per-example compute_layer_calib_n)
#   - Pre-warmed PT-ref disk cache
#   - wq_cache deduped across examples with same num_tokens, persisted to disk
#
# Runs inside a detached tmux session so it survives SSH/terminal close,
# Claude Code session end, tmux pane close, etc.
#
# Resumable: reads CSV and skips completed idx.
#
# Usage:
#   ./run_mrpc_5090.sh                 # foreground
#   ./run_mrpc_5090.sh --background    # detached tmux session 'mrpc_sweep'
#   ./run_mrpc_5090.sh --attach        # re-attach to the tmux session
#   ./run_mrpc_5090.sh --status        # alive? + last log + summary
#   ./run_mrpc_5090.sh --stop          # kill it
#
# Env overrides:
#   PYTHON=/path/to/python3   START=10   END=100   LOG=/tmp/x.log

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON=${PYTHON:-/home/yongwoo-oh/llm/bin/python3}
START=${START:-0}
END=${END:-408}
LOG=${LOG:-/tmp/local_sweep.log}
SESSION=${SESSION:-mrpc_sweep}
NUM_GPUS=${NUM_GPUS:-1}
PROBE_DIR=${PROBE_DIR:-/tmp/llama_probe_full}

# Auto-extract probe data if missing (e.g. /tmp wiped on reboot).
# Skipped for non-launching subcommands (--status / --attach / --stop).
ensure_probe() {
    if [[ ! -f "$PROBE_DIR/meta.json" || ! -f "$PROBE_DIR/rope_cos.npy" ]]; then
        echo "[run_mrpc_5090] probe data missing at $PROBE_DIR — extracting from HF cache..."
        "$PYTHON" -u python/llm_project/setup_probe_data.py --probe-dir "$PROBE_DIR"
        echo "[run_mrpc_5090] probe extraction done"
    fi
}

is_alive() { tmux has-session -t "$SESSION" 2>/dev/null; }

case "${1:-}" in
  --status)
    if is_alive; then
        echo "[run_mrpc_5090] RUNNING (tmux session '$SESSION') log=$LOG"
        echo "[run_mrpc_5090] attach with:  ./run_mrpc_5090.sh --attach"
    else
        echo "[run_mrpc_5090] NOT RUNNING (no tmux session '$SESSION')"
    fi
    echo "--- last 20 log lines ---"
    tail -20 "$LOG" 2>/dev/null || echo "(no log)"
    echo "--- summary ---"
    "$PYTHON" python/llm_project/mrpc_sweep.py --summary 2>&1 | tail -10
    exit 0
    ;;
  --attach)
    if is_alive; then exec tmux attach -t "$SESSION"
    else echo "no tmux session '$SESSION'"; exit 1; fi
    ;;
  --stop)
    is_alive && tmux kill-session -t "$SESSION" && echo "[run_mrpc_5090] killed tmux session" || true
    pkill -f 'mrpc_sweep_parallel.py\|mrpc_sweep.py' 2>/dev/null && echo "[run_mrpc_5090] pkilled stragglers" || true
    exit 0
    ;;
esac

if is_alive; then
    echo "[run_mrpc_5090] already running in tmux session '$SESSION'. Use --stop or --attach."
    exit 1
fi

ensure_probe

# Default to the simple single-GPU sweep on this box. The parallel-sweep
# code (mrpc_sweep_parallel.py) preloads the full rp_indep cache (~36 GB
# pinned host) upfront — fine on the 256 GB A100 host, but on this 5090
# box (62 GB RAM) the build-phase transients push total RSS past 60 GB
# even in subprocess isolation. The simple sweep builds the cache
# lazily during the first example's FHE compute, interleaving each
# encode with ~15 s of GPU work so the allocator can release pressure.
# Override with PARALLEL=1 to force the parallel path anyway.
if [[ "${PARALLEL:-0}" == "1" ]]; then
    CMD="$PYTHON -u python/llm_project/mrpc_sweep_parallel.py --start $START --end $END --num-gpus $NUM_GPUS"
else
    CMD="$PYTHON -u python/llm_project/mrpc_sweep.py --start $START --end $END"
fi

if [[ "${1:-}" == "--background" ]]; then
    : > "$LOG"
    echo "[run_mrpc_5090] launching in tmux session '$SESSION', log: $LOG"
    echo "[run_mrpc_5090] using parallel sweep code with NUM_GPUS=$NUM_GPUS"
    tmux new-session -d -s "$SESSION" \
        "cd '$SCRIPT_DIR' && stdbuf -oL -eL $CMD 2>&1 | tee -a '$LOG'; echo '[sweep exited rc=\$?]' | tee -a '$LOG'; sleep 600"
    echo "[run_mrpc_5090] tmux session live"
    echo "[run_mrpc_5090] monitor:  ./run_mrpc_5090.sh --status"
    echo "[run_mrpc_5090] attach:   ./run_mrpc_5090.sh --attach   (Ctrl-b d to detach)"
    echo "[run_mrpc_5090] stop:     ./run_mrpc_5090.sh --stop"
else
    echo "[run_mrpc_5090] launching in foreground"
    $CMD 2>&1 | tee "$LOG"
fi
