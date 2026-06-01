#!/bin/bash
# ppl_driver.sh — Multi-quant PPL pilot driver.
#
# ============================================================================
# BLOCKED — DO NOT RUN
# ============================================================================
# The FHE pipeline is decode-only (one query position per forward pass).
# Prefill-style WikiText-2 PPL requires running ALL positions in a window
# (up to T_MODEL=8 positions/ct with a causal mask rewrite — not yet done).
#
# run_classifier_fhe_all_positions and ppl_eval.py have been REMOVED from
# the repo (python/llm_project/). The surviving copies live in:
#   $DIR/ppl_prep/code/ppl_eval.py
#   $DIR/ppl_prep/code/run_classifier_fhe_all_positions.py  (DRAFT only)
#
# Reviving PPL requires a multi-Q prefill rewrite capped at T_MODEL=8
# positions/ct + causal mask — this is an algorithmic change, not a path fix.
#
# PT-only WikiText-2 baseline (already measured, no FHE needed):
#   token-PPL = 13.60  /  byte-PPL = 1.81  /  word-PPL = 27.89
# ============================================================================
#
# Loops over [quant-8bit, quant-16bit, quant-32bit, quant-64bit]; for each:
#   - git checkout <branch>
#   - cmake --build build --target pyPhantom -j16 (skip if pyPhantom*.so is newer than CMakeLists.txt)
#   - rm -rf cache/irp_diagonals (cross-branch contamination guard)
#   - ppl_eval.py --num-windows 32 --csv <campaign>/<branch>_ppl_pilot.csv
#   - append one-line summary to campaign.log
#
# Resumable: skips windows already in per-branch CSV (handled by ppl_eval.py);
# skips a quant whose pilot CSV already has >=32 rows of scored data.
#
# Launch (setsid+nohup, identical to resume.sh pattern):
#   setsid nohup bash /home/yongwoo-oh/mrpc_campaign/ppl_driver.sh \
#     >>/home/yongwoo-oh/mrpc_campaign/campaign.log 2>&1 &
#
# After pilots finish for all 4 quants, DOES NOT auto-launch the full sweep —
# prints summary and exits so the user can review.

set -u

REPO=/home/yongwoo-oh/phantom-fhe
PY=/home/yongwoo-oh/llm/bin/python3
DIR=/home/yongwoo-oh/mrpc_campaign
PROBE=$DIR/llama_probe_full
PILOT_WINDOWS=32

BRANCHES=(quant-8bit quant-16bit quant-32bit quant-64bit)

log() { echo "[$(date '+%F %T')] $*"; }
rows() { if [ -f "$1" ]; then grep -c '^[0-9]' "$1" 2>/dev/null || echo 0; else echo 0; fi; }
freeg() { df -h / | tail -1 | awk '{print $4}'; }

cd "$REPO" || { log "cd $REPO failed"; exit 1; }

log "=== PPL DRIVER START: pilot=${PILOT_WINDOWS} windows per quant; free=$(freeg) ==="

# ---- probe dependency (reboot-proof): ensure persistent copy + /tmp symlink ----
if [ ! -f "$PROBE/rope_cos.npy" ]; then
  log "probe missing at $PROBE — abort (regenerate via scripts/setup_probe_data.py first)"
  exit 1
fi
rm -rf /tmp/llama_probe_full
ln -sf "$PROBE" /tmp/llama_probe_full
[ -f /tmp/llama_probe_full/rope_cos.npy ] || { log "probe symlink broken — abort"; exit 1; }
log "probe ready ($(ls "$PROBE" | wc -l) entries); /tmp/llama_probe_full -> $PROBE"

# ---- ppl_prep artifacts present? ----
if [ ! -f "$DIR/ppl_prep/windows.npz" ]; then
  log "$DIR/ppl_prep/windows.npz missing — abort"
  exit 1
fi
if [ ! -f "$DIR/ppl_prep/lm_head_full.npy" ]; then
  log "$DIR/ppl_prep/lm_head_full.npy missing — abort"
  exit 1
fi
N_REFS=$(ls "$DIR"/ppl_prep/refs/ppl_window_*.npz 2>/dev/null | wc -l)
if [ "$N_REFS" -lt "$PILOT_WINDOWS" ]; then
  log "only $N_REFS PT refs present; need >= $PILOT_WINDOWS — abort"
  exit 1
fi
log "ppl_prep OK: windows.npz, lm_head_full.npy, $N_REFS PT refs"

# ---- per-branch loop ----
for br in "${BRANCHES[@]}"; do
  CSV=$DIR/${br}_ppl_pilot.csv
  done_rows=$(rows "$CSV")
  if [ "$done_rows" -ge "$PILOT_WINDOWS" ]; then
    log "[$br] PILOT already complete ($done_rows rows >= $PILOT_WINDOWS) — skip"
    continue
  fi

  log "[$br] checkout + build (free=$(freeg))"
  git -C "$REPO" checkout "$br" >"$DIR/ppl_build_${br}.log" 2>&1 || { log "[$br] CHECKOUT FAILED — abort"; exit 1; }
  log "[$br] @ $(git -C "$REPO" rev-parse --short HEAD)"

  # Cross-branch cache contamination guard (project memory: scale-hash collides on user_scale).
  rm -rf "$REPO/cache/irp_diagonals"
  log "[$br] cleared cache/irp_diagonals"

  # Build pyPhantom (skipped fast by cmake if .so already current for this branch).
  cmake --build "$REPO/build" --target pyPhantom -j16 >>"$DIR/ppl_build_${br}.log" 2>&1 \
    || { log "[$br] BUILD FAILED — abort (see $DIR/ppl_build_${br}.log)"; exit 1; }
  log "[$br] build OK (free=$(freeg))"

  # Launch ppl_eval.py
  log "[$br] launching ppl_eval.py --num-windows $PILOT_WINDOWS --csv $CSV"
  cd "$REPO/python/llm_project"
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  USE_BOOTSTRAP_17=1 \
  AUTONOMOUS_FHE=1 \
  PPL_USE_FHE=1 \
    "$PY" -u "$DIR/ppl_prep/code/ppl_eval.py" \
      --num-windows "$PILOT_WINDOWS" \
      --csv "$CSV" \
      >>"$DIR/ppl_sweep_${br}.log" 2>&1
  rc=$?
  cd "$REPO"

  after_rows=$(rows "$CSV")
  log "[$br] ppl_eval rc=$rc rows=$after_rows/$PILOT_WINDOWS"

  if [ "$after_rows" -lt 1 ]; then
    log "[$br] NO ROWS WRITTEN — abort (see $DIR/ppl_sweep_${br}.log)"
    exit 1
  fi

  # One-line summary appended to campaign.log
  summary=$(HF_HUB_OFFLINE=1 "$PY" -u "$DIR/ppl_prep/code/ppl_eval.py" --summary --csv "$CSV" 2>&1 | \
            awk '
              /FHE PPL/         { in_fhe=1; in_pt=0; next }
              /PT PPL/          { in_fhe=0; in_pt=1; next }
              in_fhe && /token-PPL/ { gsub(/[,]/, ""); tok=$3 }
              in_fhe && /byte-PPL/  { gsub(/[,]/, ""); byt=$3 }
              in_fhe && /word-PPL/  { gsub(/[,]/, ""); wrd=$3 }
              END                 { printf("token-PPL=%s byte-PPL=%s word-PPL=%s", tok, byt, wrd) }
            ')
  log "[$br] PILOT: $after_rows rows, FHE $summary" | tee -a "$DIR/campaign.log"
done

log "=== PPL DRIVER COMPLETE: pilots done for all 4 quants ==="
log "Full-sweep is NOT auto-launched. Review $DIR/{quant-*_ppl_pilot.csv,campaign.log}"

# Leave live tree on quant-8bit for inspection.
git -C "$REPO" checkout quant-8bit >/dev/null 2>&1 || true
log "live tree on $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
