# RECONCILE_AND_BASELINE_RUNBOOK — `dense-bootstrap17`

Branch `dense-bootstrap17` reconciles the correct **dense Python pipeline**
(`dense-linear-rewrite@bea1cb7`) with the **bootstrap-17 native CUDA cluster**
(`bootstrap-to17-wip@92fb54c`) into a single self-contained branch.

**This branch is UNVALIDATED in the originating environment.** The originating
environment SIGTERM-kills multi-minute full CUDA rebuilds, so nothing here was
built or run — only static verification (git blob equality, 3-way-merge
conflict-freeness, py_compile, symbol-resolution greps). **Step 3 below is the
trust gate.** Until step 3 passes in a capable environment, treat the
reconciled native as unproven.

What the reconciliation did (surgical, NOT a `git merge`):

- Base = `dense-linear-rewrite@bea1cb7` (authoritative dense Python + binding's
  `bsgs_diagonals` (de)serialization + `DENSE_BSGS_DISK_CACHE`).
- 7 native files overlaid wholesale from `bootstrap-to17-wip`
  (git-blob byte-identical): `src/bootstrap.cu`, `src/ckks_engine.cu`,
  `src/evalmod.cu`, `src/evaluate.cu`, `include/bootstrap.h`,
  `include/ckks_engine.h`, `include/evalmod.h`.
- `python/src/binding.cu` = **UNION** built via 3-way merge
  (base = `merge-base 0a262fc`, ours = dense@bea1cb7, theirs = wip). Clean,
  zero conflicts (the `bsgs_diagonals` region and the bootstrap-17 region are
  textually disjoint). Final binding.cu contains BOTH
  `freshest_chain_index` AND `bsgs_diagonals` `.diagonals`/reconstruction-ctor
  AND `num_special_primes` rw.
- `python/llm_project/**` is byte-identical to `bea1cb7` (empty diff). The
  stale parked Python on `bootstrap-to17-wip` (`llama3.py`, `kv_layout.py`,
  `mrpc_sweep.py`, `probe_*.py`, `BOOTSTRAP_TO17_*` docs, `.gitignore`) was
  deliberately NOT taken.

Why this branch is needed: dense@bea1cb7's `llama3_mrpc.py:1716` hard-calls
`engine.freshest_chain_index()`, which bea1cb7's committed `binding.cu` does
NOT expose → dense@bea1cb7 is unbuildable-to-working alone. The bootstrap-17
native cluster + unioned binding.cu closes exactly that gap.

---

## Build environment expectations

- A GPU/CUDA environment that **survives a multi-minute full CUDA rebuild**
  (the originating env does not — that is the entire reason for this runbook).
- CUDA toolkit + a host C++17 compiler + CMake ≥ 3.x. The committed
  `build/CMakeCache.txt` was configured with: `CMAKE_BUILD_TYPE=Release`,
  `PHANTOM_ENABLE_PYTHON_BINDING=ON`, `CMAKE_CUDA_ARCHITECTURES=120`
  (Unix Makefiles generator). Adjust `CMAKE_CUDA_ARCHITECTURES` to the target
  GPU (e.g. `90` for H100, `80` for A100, `native` to auto-detect).
- Python deps for the MRPC sweep (already validated in the originating env):
  torch 2.11 / cu128, transformers 5.8.1, datasets 4.8.5. Run offline:
  `HF_HUB_OFFLINE=1`.

---

## Step 1 — Checkout + FULL clean rebuild

A full (NOT incremental) rebuild is mandatory: this reconciliation changed
CUDA kernels (`bootstrap.cu`, `ckks_engine.cu`, `evalmod.cu`, `evaluate.cu`)
and the shared headers (`bootstrap.h`, `ckks_engine.h`, `evalmod.h`). With
`CMAKE_CUDA_SEPARABLE_COMPILATION ON`, header changes invalidate many
translation units; an incremental build over a stale `build/` risks linking
mismatched object code.

```bash
cd /path/to/phantom-fhe
git switch dense-bootstrap17
git rev-parse HEAD            # expect the dense-bootstrap17 commit SHA

# Full clean reconfigure + build (delete prior build/ to force a clean compile)
rm -rf build
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DPHANTOM_ENABLE_PYTHON_BINDING=ON \
      -DCMAKE_CUDA_ARCHITECTURES=<your_gpu_arch>   # e.g. 90 / 80 / native
cmake --build build --target pyPhantom -j"$(nproc)"
```

- Target: `pyPhantom` (defined in `python/CMakeLists.txt`:
  `pybind11_add_module(pyPhantom src/binding.cu)` → links `Phantom`). Building
  this target transitively builds `libPhantom` (the `src/` CUDA library).
- Output artifacts: `build/lib/libPhantom.so` and
  `build/lib/pyPhantom.cpython-<ver>-<arch>-linux-gnu.so`
  (`CMAKE_LIBRARY_OUTPUT_DIRECTORY = build/lib`).
- Expected wall time: a full `src/` CUDA recompile is **~several to ~15+
  minutes** depending on the host (separable compilation + many `.cu` TUs).
  Plan for a long-lived foreground/background process; do NOT run in an
  environment that kills multi-minute jobs.

## Step 2 — Sanity: both feature sets reachable from Python

```bash
PYTHONPATH=build/lib python3 - <<'PY'
import pyPhantom as p
# bsgs_diagonals (de)serialization — from dense@bea1cb7
assert hasattr(p, "bsgs_diagonals"), "bsgs_diagonals class missing"
bd = p.bsgs_diagonals
# .diagonals accessor / reconstruction ctor — dense serialization path
assert hasattr(bd, "diagonals") or "diagonals" in dir(bd), "bsgs_diagonals.diagonals missing"
assert hasattr(p, "pre_encode_bsgs_diagonals"), "pre_encode_bsgs_diagonals missing"
# freshest_chain_index — from bootstrap-17 (the unbuildable-alone gap)
assert hasattr(p.ckks_engine, "freshest_chain_index"), "ckks_engine.freshest_chain_index missing"
# num_special_primes config rw
cfg = p.ckks_engine_config()
assert hasattr(cfg, "num_special_primes"), "num_special_primes rw missing"
assert hasattr(cfg, "use_bootstrap_to_17_levels"), "use_bootstrap_to_17_levels rw missing"
print("SANITY OK: bsgs_diagonals(.diagonals) + freshest_chain_index + num_special_primes all present")
PY
```

All asserts must pass. If `freshest_chain_index` is absent, the binding.cu
union did not take effect (rebuild from a clean `build/`). If
`bsgs_diagonals.diagonals` is absent, the dense serialization path was lost
(reconciliation regression — do not proceed).

## Step 3 — MANDATORY pre-baseline trust gate (single-layer L31, idx0, use17)

This is the ONLY gate that establishes the reconciled native is trustworthy.
Do NOT run any baseline until this passes.

```bash
rm -f /tmp/mrpc_sweep_results.csv
PROBE_MIN_LAYER=31 PROBE_MAX_LAYER=31 \
HF_HUB_OFFLINE=1 USE_BOOTSTRAP_17=1 NSL=16 NSP=8 \
PYTHONPATH=build/lib python3 python/llm_project/mrpc_sweep.py \
        --start 0 --end 1 --fixed-nt 512
```

PASS criteria (all three required):

- log shows **`Engine NSL=16`**
- **`Layer 31 rel-RMS ≈ 3.3e-3`** (single-layer teacher-forced fidelity vs
  pytorch reference; ~3.3e-3 is the expected magnitude)
- **`PT=Yes FHE=Yes (agree)`** (the L31 MRPC prediction agrees between the
  plaintext path and the FHE path)

Only if all three hold is the reconciled native CUDA trustworthy. If any fail,
STOP — the bootstrap-17 native overlay or the binding.cu union has a problem;
do not produce a baseline from an untrusted native.

(Note: `engine.freshest_chain_index()` must return `16` here — `llama3_mrpc.py`
asserts `freshest_chain_index() == 16` under `NSL=16` and aborts otherwise.
This exercises the exact bootstrap-17 binding that made dense@bea1cb7
unbuildable alone.)

## Step 4 — nt=512 / idx0 Cachemir baseline (only after Step 3 PASSES)

The dense BSGS disk cache is gated by `DENSE_BSGS_DISK_CACHE=1`
(`python/llm_project/blocks/dense_bsgs_cache.py`; cache root
`<repo>/cache/dense_bsgs/`, gitignored). Build the 32-layer cache **per layer**
(robust against an env that kills long runs — each layer is a short job), then
run the per-layer-assembled measurement.

### 4a. Build the per-layer dense BSGS cache (L = 0..31)

```bash
for L in $(seq 0 31); do
  PROBE_MIN_LAYER=$L PROBE_MAX_LAYER=$L \
  DENSE_BSGS_DISK_CACHE=1 \
  HF_HUB_OFFLINE=1 USE_BOOTSTRAP_17=1 NSL=16 NSP=8 \
  PYTHONPATH=build/lib python3 python/llm_project/mrpc_sweep.py \
          --start 0 --end 1 --fixed-nt 512
done
```

Each iteration warms `cache/dense_bsgs/` for that layer. If the target env does
NOT kill long runs, this MAY instead be run as a single full sweep
(`PROBE_MIN_LAYER`/`PROBE_MAX_LAYER` unset) — see env note below.

### 4b. Per-layer-assembled measurement (warm cache)

Re-run the same per-layer loop with the cache now warm (`DENSE_BSGS_DISK_CACHE=1`
hits disk instead of recomputing the diagonals):

- **Accuracy** = L31 MRPC prediction agrees (`PT=Yes FHE=Yes`) **plus** the
  per-layer rel-RMS across L0..L31 (each ≈ the single-layer teacher-forced
  fidelity, ~1e-3 magnitude; NOT an end-to-end FHE accuracy — the pipeline is
  per-layer reference-guided / teacher-forced from `pytorch_ref`).
- **Time** = sum of the warm cached per-layer wall times (the cache removes the
  diagonal-encode cost; the residual is the FHE linear/bootstrap evaluation).

### Environment note (single-layer-foreground vs full run)

- The originating env SIGTERM-kills ~minutes-long sweeps at turn/invocation
  boundaries → it requires **single-layer foreground**
  (`PROBE_MIN_LAYER=L PROBE_MAX_LAYER=L`, ~30s/layer with idx0 cached
  pytorch_ref) for per-layer gates and baselines.
- If the capable env does NOT kill long runs, prefer a single full 32-layer
  sweep (drop `PROBE_MIN_LAYER`/`PROBE_MAX_LAYER`) for a true end-to-end timing
  number; otherwise sum the per-layer warm-cache times as above and state that
  the timing is per-layer-assembled, not a single continuous run.

## Step 5 — Status statement (record with the baseline)

State plainly in any report derived from this branch:

> `dense-bootstrap17` is UNVALIDATED in the originating environment (no build /
> no run was possible there — full CUDA rebuilds are SIGTERM-killed). Static
> reconciliation only: native files are git-blob byte-identical to
> `bootstrap-to17-wip@92fb54c`; `python/llm_project/**` is byte-identical to
> `dense-linear-rewrite@bea1cb7`; `binding.cu` is a conflict-free 3-way union
> carrying BOTH `freshest_chain_index` and `bsgs_diagonals` (de)serialization.
> **Step 3 (single-layer L31 idx0 use17 → `Engine NSL=16`,
> `rel-RMS ≈ 3.3e-3`, `PT=Yes FHE=Yes`) is the trust gate.** Do not trust the
> reconciled native, and do not publish a baseline, until Step 3 passes in a
> capable environment.
