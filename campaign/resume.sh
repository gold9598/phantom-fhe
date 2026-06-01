#!/bin/bash
# RESUME the MRPC FHE campaign (post-reboot). Reboot-resilient:
#  - probe data persists in $DIR/llama_probe_full; /tmp/llama_probe_full is a
#    symlink recreated here (regenerated if the persistent copy is missing).
#  - result CSVs persist in $DIR (only /tmp/mrpc_sweep_results.csv is a symlink).
#  - ABORTS (does NOT advance / delete caches) if a branch can't make progress,
#    so a startup crash never cascades into wiping the next branch's cache.
# Scope: int32 (unchunked, ~265G cold) then int64 (chunked 102x4). Free ~550G.
# Re-launch after any stop:  setsid nohup bash /home/yongwoo-oh/mrpc_campaign/resume.sh >>/home/yongwoo-oh/mrpc_campaign/campaign.log 2>&1 &
set -u
REPO=/home/yongwoo-oh/phantom-fhe
PY=/home/yongwoo-oh/llm/bin/python3
DIR=/home/yongwoo-oh/mrpc_campaign
PROBE=$DIR/llama_probe_full
CSV32=$DIR/quant-32bit_auto.csv
CSV64=$DIR/quant-64bit_auto.csv
CHUNK=102
cd "$REPO" || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
rows(){ if [ -f "$1" ]; then grep -c '^[0-9]' "$1" 2>/dev/null; else echo 0; fi; }
rows_in(){ awk -F, -v s="$2" -v e="$3" 'NR>1 && $1+0>=s && $1+0<e' "$1" 2>/dev/null | wc -l; }
freeg(){ df -h / | tail -1 | awk '{print $4}'; }

log "=== RESUME: int32 (unchunked) + int64 (chunked 102x4); persistent in $DIR; free=$(freeg) ==="

# ---- probe dependency (reboot-proof): ensure persistent copy + /tmp symlink ----
if [ ! -f "$PROBE/rope_cos.npy" ]; then
  log "probe missing — regenerating to $PROBE (~10min)"
  ( cd "$REPO/python/llm_project" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -u scripts/setup_probe_data.py --probe-dir "$PROBE" ) >>"$DIR/probe_regen.log" 2>&1 \
    || { log "PROBE REGEN FAILED — abort"; exit 1; }
fi
rm -rf /tmp/llama_probe_full; ln -sf "$PROBE" /tmp/llama_probe_full
[ -f /tmp/llama_probe_full/rope_cos.npy ] || { log "probe symlink broken — abort"; exit 1; }
log "probe ready ($(ls "$PROBE" | wc -l) entries); /tmp/llama_probe_full -> $PROBE"

# ---- PT-ref cache (reboot-proof): STANDALONE pre-capture so the PyTorch ref
#      model (~16G) never coexists with the FHE engine (~17G) on the 32G GPU.
#      precapture_ptref.py restores from persistent $DIR/ptref if present, else
#      captures all 408 (~27min). Must run in its OWN process (no engine). ----
log "ensuring PT-ref cache (standalone; restore-from-persistent or capture)"
( cd "$REPO/python/llm_project" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -u "$DIR/precapture_ptref.py" ) >>"$DIR/precapture.log" 2>&1 \
  || { log "PT-REF PRECAPTURE FAILED — abort"; exit 1; }
log "PT-ref cache ready: $(ls /tmp/mrpc_ptref_*.npz 2>/dev/null | wc -l) refs in /tmp"

# ---- int32: unchunked full re-run (cold; cache was deleted) ----
if [ "$(rows "$CSV32")" -lt 408 ]; then
  log "int32 AUTONOMOUS: checkout quant-32bit + rebuild (KEEP warm irp_diagonals — same weights) ($(rows "$CSV32")/408 done)"
  git checkout quant-32bit >"$DIR/build_32.log" 2>&1 || { log "int32 CHECKOUT FAILED — abort"; exit 1; }
  cmake --build build --target pyPhantom -j16 >>"$DIR/build_32.log" 2>&1 || { log "int32 BUILD FAILED — abort"; exit 1; }
  log "int32 build OK @ $(git rev-parse --short HEAD); free=$(freeg)"
  cd "$REPO/python/llm_project"; attempt=0
  while [ "$(rows "$CSV32")" -lt 408 ] && [ $attempt -lt 8 ]; do
    before=$(rows "$CSV32"); attempt=$((attempt+1)); ln -sf "$CSV32" /tmp/mrpc_sweep_results.csv
    log "int32 sweep attempt $attempt (from $before/408); free=$(freeg)"
    AUTONOMOUS_FHE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_BOOTSTRAP_17=1 "$PY" -u scripts/mrpc_sweep.py >>"$DIR/sweep_32_auto.log" 2>&1
    after=$(rows "$CSV32"); log "int32 attempt $attempt -> $after/408; free=$(freeg)"
    [ "$after" -le "$before" ] && { log "int32 NO PROGRESS — likely startup crash; see $DIR/sweep_32.log"; break; }
  done
  cd "$REPO"
fi
# CASCADE GUARD: only continue to int64 if int32 is genuinely complete.
if [ "$(rows "$CSV32")" -lt 408 ]; then
  log "int32 INCOMPLETE ($(rows "$CSV32")/408) — ABORT (NOT touching int64 / caches). Fix + re-launch."
  exit 1
fi
log "int32 COMPLETE: 408/408"

# ---- int64: chunked 102x4, clear only WQ between chunks ----
if [ "$(rows "$CSV64")" -lt 408 ]; then
  log "int64: checkout quant-64bit + clear irp_diagonals + rebuild"
  git checkout quant-64bit >"$DIR/build_64.log" 2>&1 || { log "int64 CHECKOUT FAILED — abort"; exit 1; }
  rm -rf "$REPO/cache/irp_diagonals"
  cmake --build build --target pyPhantom -j16 >>"$DIR/build_64.log" 2>&1 || { log "int64 BUILD FAILED — abort"; exit 1; }
  log "int64 build OK @ $(git rev-parse --short HEAD); free=$(freeg)"
  cd "$REPO/python/llm_project"
  for ((S=0;S<408;S+=CHUNK)); do
    E=$((S+CHUNK)); [ $E -gt 408 ] && E=408; need=$((E-S))
    [ "$(rows_in "$CSV64" "$S" "$E")" -ge "$need" ] && { log "int64 [$S,$E) done skip"; continue; }
    log "int64 [$S,$E): clear WQ (keep FIXED); $(rows_in "$CSV64" "$S" "$E")/$need; free=$(freeg)"
    rm -f "$REPO"/cache/irp_diagonals/wq_*.irpcv2; a=0
    while [ "$(rows_in "$CSV64" "$S" "$E")" -lt "$need" ] && [ $a -lt 6 ]; do
      b4=$(rows_in "$CSV64" "$S" "$E"); a=$((a+1)); ln -sf "$CSV64" /tmp/mrpc_sweep_results.csv
      log "  int64 [$S,$E) try $a (from $b4/$need); free=$(freeg)"
      AUTONOMOUS_FHE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_BOOTSTRAP_17=1 "$PY" -u scripts/mrpc_sweep.py --start "$S" --end "$E" >>"$DIR/sweep_64_auto.log" 2>&1
      af=$(rows_in "$CSV64" "$S" "$E"); log "  int64 [$S,$E) try $a -> $af/$need; free=$(freeg)"
      [ "$af" -le "$b4" ] && { log "  int64 [$S,$E) NO PROGRESS — stop chunk"; break; }
    done
  done
  cd "$REPO"
fi

log "=== RESUME COMPLETE: int32 $(rows "$CSV32")/408, int64 $(rows "$CSV64")/408 ==="
cd "$REPO/python/llm_project"
ln -sf "$CSV32" /tmp/mrpc_sweep_results.csv; HF_HUB_OFFLINE=1 "$PY" -u scripts/mrpc_sweep.py --summary 2>&1 | sed 's/^/[int32] /'
ln -sf "$CSV64" /tmp/mrpc_sweep_results.csv; HF_HUB_OFFLINE=1 "$PY" -u scripts/mrpc_sweep.py --summary 2>&1 | sed 's/^/[int64] /'
log "=== DONE ==="
