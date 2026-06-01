# MRPC Baseline Results — fp64 plaintext path

Snapshot of the in-progress 408-MRPC sweep at the point where the
`baseline` branch was forked.

## Snapshot (n=94)

```
PT  : acc=68.09  F1=81.01
FHE : acc=68.09  F1=81.01
FHE-vs-PT prediction agreement: 100.00%
Cachemir reference: acc=71.32  F1=82.19
```

- **0 disagreements** across all 94 completed examples — FHE pipeline
  is numerically faithful to the PyTorch reference.
- Numbers are converging toward Cachemir's published 71.32 / 82.19 as
  more examples accumulate (sample at n=83 hit F1=82.27 > Cachemir).
- File: `mrpc_baseline.csv` (`idx, num_tokens, label, pt_yes, pt_no,
  pt_pred, fhe_yes, fhe_no, fhe_pred, time_sec`).

## Codebase state

Branch `baseline` is forked from `parallel-sweep-4gpu` at this commit.
Includes the full streaming + disk-cache + producer-consumer pipeline:

- `python/llm_project/build_disk_cache.py` — pre-encodes the rp_indep
  IRPs per layer.
- `python/llm_project/build_wq_disk_cache.py` — pre-encodes the Wq
  IRPs per (layer, num_tokens).
- `python/llm_project/prewarm_ptref.py` — populates the PyTorch-ref
  .npz disk cache (one process, single HF load).
- `python/llm_project/mrpc_sweep.py` — streaming single-GPU sweep
  with `rp_indep_disk_root` auto-detection and `STREAM_QUEUE_DEPTH`
  env knob.
- `python/llm_project/llama3_mrpc.py` — streaming JIT load + producer
  thread in `run_classifier_fhe`.
- `python/llm_project/llama3.py` — fp64 plaintext encode path.

## Per-layer perf (5090 dev box)

| Variant | Avg/layer | Min/layer |
|---|---|---|
| No cache | 18,659 ms | — |
| Wq disk cache, no pipeline | 15,295 ms | 12,808 ms |
| + Pipeline depth=1 | 11,830 ms | 8,362 ms |
| + Pipeline depth=8 | 11,370 ms | 6,573 ms |

## Disk caches (NOT in git, gitignored under `cache/`)

- `cache/rp_indep/` — 73 GB, 32 layer subdirs, built by
  `build_disk_cache.py`.
- `cache/wq/` — ~228 GB, 57 num_tokens × 32 layer subdirs, built by
  `build_wq_disk_cache.py`.

These two caches together let the streaming sweep skip ~13 s of
per-layer encoding work that would otherwise blow past the 5090 box's
62 GB host RAM ceiling.

## Quantized plaintext branch

The follow-on work (`quantized-plaintext` or similar) will replace the
fp64 plaintext path with a quantized variant. Reuses everything in
`cache/` so don't delete those.

---

## Quant-study artifacts (int8 / int16 / int32 / int64)

Full autonomous per-example results and analysis from the quantized-plaintext
campaign (`AUTONOMOUS_FHE=1`, n=408 MRPC validation set).

### Result CSVs

| File | Branch | Rows | Notes |
|---|---|---|---|
| `mrpc_baseline.csv` | baseline | 94 | fp64, n=94 snapshot |
| `mrpc_baseline_final.csv` | baseline | 408 | fp64, full n=408 |
| `quant-32bit_auto.csv` | quant-32bit | 408 | int32 plaintext, autonomous sweep |
| `quant-64bit_auto.csv` | quant-64bit | 408 | int64 plaintext, autonomous sweep |

### Summary

`mrpc_quant_results_summary.md` — headline accuracy/F1 table across int8/16/32/64,
logit fidelity (max |err|, rel-RMS per layer), and campaign timeline.

### Design and analysis docs

- `doc/design/` — FHE design rationale migrated from in-code comments across all
  quant branches. Seven files: `README.md`, `llama3_mrpc.md`, `decoder_layer.md`,
  `engine_setup.md`, `fhe_attention_dense.md`, `pytorch_ref.md`, `diagnostics.md`.
  The quant-branch modules carry `# design: <file>` pointer comments back here.

- `doc/analysis/` — supporting analysis documents:
  - `ppl_eval_plan.md` — plan for perplexity evaluation on the FHE pipeline.
  - `precomputed_calib_variance.md` — study of variance in precomputed calibration
    data and its impact on quantized inference.
