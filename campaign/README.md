# MRPC FHE Campaign Driver Scripts

This directory contains the operational harness that drove the MRPC FHE
quant-comparison campaign across the `quant-8bit`, `quant-16bit`, `quant-32bit`,
and `quant-64bit` branches.

The multi-GB probe data, PT-ref caches, and per-window refs live **outside**
the repo in an external state directory (`DIR=/home/yongwoo-oh/mrpc_campaign`).
Per-example MRPC result CSVs are committed under `results/` on each quant branch.

All scripts use **absolute paths** faithful to their provenance; the external
state dir is intentionally not in the repo.

---

## Script inventory

### `resume.sh`

Reboot-resilient MRPC sweep driver. Runs int32 (unchunked, full 408 examples)
followed by int64 (chunked 102×4). For each quant it:

1. Checks out the appropriate quant branch (`quant-32bit` / `quant-64bit`).
2. Builds `pyPhantom` via cmake.
3. Runs `scripts/mrpc_sweep.py` in a retry loop, appending results to a
   persistent CSV in `$DIR`; aborts — does **not** cascade — if a branch makes
   no progress (startup crash guard).

**Status**: VERIFIED end-to-end on the reorganized `fhe/helpers/scripts/` tree
(idx-0 autonomous FHE forward, FHE↔PT pred agree). Re-launch at any time:

```bash
setsid nohup bash /home/yongwoo-oh/mrpc_campaign/resume.sh \
  >> /home/yongwoo-oh/mrpc_campaign/campaign.log 2>&1 &
```

---

### `precapture_ptref.py`

Standalone PyTorch-reference pre-capture. Runs in its **own process** so the
~16 GB PT model never coexists with the ~17 GB FHE engine on the 32 GB GPU
(coexistence OOM'd the sweep after a reboot wiped `/tmp`).

For each of the 408 MRPC validation examples it:
- Restores from a persistent backup in `$DIR/ptref/` if present (fast path).
- Otherwise calls `capture_pytorch_ref` and writes both to `/tmp` (for the
  sweep) and to `$DIR/ptref/` (durable backup).

Imports: `from helpers.pytorch_ref import capture_pytorch_ref`

---

### `calib_variance_study.py`

Pure-numpy study of how stable `compute_layer_calib_n` outputs are across 20
MRPC examples × 32 decoder layers. Validates the single-example
`precomputed_calib` design (calibration frozen from idx=0, `num_tokens=60`).

Reads layer weights from `$DIR/llama_probe_full/`, PT-refs from `$DIR/ptref/`,
and the result CSV for example selection. Writes a Markdown report to
`$DIR/precomputed_calib_variance.md`. Key finding: bootstrap-critical fields
have CV < 5% across the MRPC distribution, confirming the 1.5× `BOOT_CALIB_MARGIN`
is safe.

No pyPhantom or CUDA required.

---

### `ppl/` — BLOCKED prefill-PPL scaffolding

> **DO NOT RUN** — these scripts are kept for provenance only.

The FHE pipeline is **decode-only** (one query position per forward pass).
WikiText-2 PPL requires prefill: scoring all token positions in a window.
Reviving PPL requires a multi-Q rewrite (≤ `T_MODEL=8` positions/ct + causal
mask) — an algorithmic change, not a path fix.

`run_classifier_fhe_all_positions` was **removed** from the repo
(`fhe/llama3_mrpc.py`). The surviving draft copy is preserved here for
reference.

PT-only WikiText-2 baseline (already measured, no FHE needed):

| Metric     | Value |
|------------|-------|
| token-PPL  | 13.60 |
| byte-PPL   | 1.81  |
| word-PPL   | 27.89 |

#### `ppl/prepare_ppl.py`

Deliverables 1–3 + PT smoke test. Builds `windows.npz` (256×64 strided
windows over wikitext-2-raw-v1 test), `corpus_stats.json`, `lm_head_full.npy`
(128256×4096 fp32), per-window PT reference captures in `$DIR/ppl_prep/refs/`,
and `pt_ppl_results.json`. All outputs land in `$DIR/ppl_prep/` (external).

#### `ppl/resume_capture.py`

Resumes PT reference capture for windows whose `ppl_window_NNNN.npz` is
missing or truncated (healthy file ~41 MB; truncated <35 MB). Loads the PT
model once for a contiguous `[start, end)` range, then frees it.

#### `ppl/lm_head_full.py`

Full-vocab LM-head readout helper. Provides `full_vocab_logprobs_np`
((T, D_MODEL) → (T, VOCAB) log-softmax) and `next_token_logprobs` for
per-position next-token scoring. Uses the stride-T_MODEL slot layout from
`llama3.py` / `llama3_mrpc.py`. No FHE dependency.

#### `ppl/ppl_eval.py`

FHE end-to-end PPL evaluation driver. **BLOCKED**: imports
`run_classifier_fhe_all_positions` from `fhe.llama3_mrpc`, which was removed.
Contains a PT-only validation path (`PPL_USE_FHE=0`) that exercises the
`lm_head_full` pipeline against `pytorch_pre_norm` as a stand-in.

#### `ppl/run_classifier_fhe_all_positions.py`

**DRAFT** — not patched into `llama3_mrpc.py`. Documents the exact 10-line
diff needed: capture `y_ct` after layer 31's residual2, decode all `num_tokens`
positions via `y_full[i*T_MODEL : i*T_MODEL + D_MODEL]`, return
`(num_tokens, D_MODEL)` instead of `(yes_logit, no_logit)`. The `PATCH` string
at the bottom of the file contains the verbatim diff.

#### `ppl/ppl_driver.sh`

Multi-quant PPL pilot driver. Loops over all four quant branches, builds
pyPhantom, clears the irp_diagonals cross-branch cache, and runs `ppl_eval.py`
for 32 pilot windows per quant. **BLOCKED** for the same reason as
`ppl_eval.py`. Resumable: skips branches whose pilot CSV already has ≥ 32 rows.

#### `ppl/run_ppl.sh`

Thin wrapper around `ppl_eval.py`. Handles probe-symlink setup, artifact
checks, and final PPL aggregation. Supports `--pilot` (32 windows) and
`--full` (256 windows) modes.

---

## Runtime assumptions

All scripts use absolute paths faithful to their original provenance:

| Variable | Value |
|----------|-------|
| `REPO`   | `/home/yongwoo-oh/phantom-fhe` |
| `DIR`    | `/home/yongwoo-oh/mrpc_campaign` (external; not in repo) |

The external `$DIR` holds: `llama_probe_full/` (probe weights, ~33 GB),
`ptref/` (PT-ref cache, ~28 GB), `ppl_prep/refs/` (window refs, ~9.9 GB),
`ppl_prep/lm_head_full.npy` (~2 GB), per-example CSVs, and logs. None of
these are committed.

---

## Post-reorg path references

These scripts reference the post-reorg layout (reorganization completed before
the campaign ran):

- `scripts/mrpc_sweep.py` — main sweep driver
- `scripts/setup_probe_data.py` — probe data generation
- `fhe.llama3_mrpc` — FHE LLaMA-3 MRPC classifier module
- `helpers.pytorch_ref` — PyTorch reference capture helper
