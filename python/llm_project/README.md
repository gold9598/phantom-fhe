# python/llm_project

Phantom-FHE LLaMA-3.1-8B CKKS inference — Cachemir IRP + DAG bootstrap placement.

## Layout

```
fhe/
  llama3_mrpc.py          # production MRPC entry point (32-layer + LM head)
  decoder_layer.py        # single decoder layer (one FHE forward pass)
  engine_setup.py         # CKKS engine build, galois step assignment, calibration
  fhe_attention_dense.py  # dense IRP attention: QK^T, softmax, score·V, Wo

helpers/
  llama3.py               # constants, numpy reference helpers, weight loaders
  pytorch_ref.py          # PyTorch reference capture + on-disk cache
  diagnostics.py          # _probe (decrypt+print), _malloc_trim

scripts/
  llama3_simulation.py    # plaintext-shim reference (decrypt+re-encrypt, no bootstrap)
  mrpc_sweep.py           # MRPC sweep driver (sequential)
  mrpc_sweep_parallel.py  # MRPC sweep driver (parallel)
  build_disk_cache.py     # pre-build SCP disk cache
  build_wq_disk_cache.py  # pre-build Wq IRP disk cache
  prewarm_ptref.py        # pre-warm PyTorch reference cache
  setup_probe_data.py     # generate probe inputs
  extract_llama_probe.py  # extract per-layer probe tensors
  speed_bench.py          # wall-time benchmark harness
  probe_key_sizes.py      # report Galois/relin key sizes
  probe_key_sizes_detailed.py
  rope_extend.py          # RoPE extension experiments
  torch_mrpc_baseline.py  # plaintext MRPC baseline (no FHE)

blocks/
  irp.py                  # Cachemir §4.1 IRP (square + rect; quant-delta file)
  irp_cache.py            # IRP encoding disk cache (quant-delta file)
  scp_disk_cache.py       # SingleChainPlaintext disk cache (quant-delta file)
  attention.py            # IRP attention kernels: compute_qkt_irp, finalize_softmax_irp_t, score_times_v_irp
  bootstrap_placement.py  # Cachemir §6 DAG bootstrap placement (shortest-path)
  bootstrap.py            # mean-centered EvalMod wrapper
  rmsnorm.py              # RMSNorm forward (stride-t layout)
  softmax.py              # softmax helpers + polynomial composition
  silu.py                 # SiLU coefficient table + forward
  rope.py                 # RoPE precompute + apply
  residual.py             # residual add helper
  linear.py               # FD linear (legacy diagonal path)
  mlp.py                  # MLP forward + setup helpers

tests/
  *_test.py               # 19 regression tests (moved from blocks/ and top-level)
```

## Run

From the repo root, build first:

```bash
cmake --build build -j 8
```

Then from `python/llm_project`:

```bash
cd python/llm_project
HF_HUB_OFFLINE=1 USE_BOOTSTRAP_17=1 python3 -u fhe/llama3_mrpc.py --idx 0   # one MRPC example (real bootstrap)
python3 scripts/llama3_simulation.py                                          # plaintext-shim reference
```

Run all regression tests:

```bash
for f in python/llm_project/tests/*_test.py; do python3 "$f"; done
```

## Imports

All imports are absolute-qualified from the `llm_project` root:

```python
from fhe.engine_setup import setup_engine
from helpers.llama3 import D_MODEL
from blocks.irp import irp_matvec_host
```

No `PYTHONPATH` prefix needed — entry scripts self-insert `build/lib` on `sys.path`.

## Quant branches

The four precision branches (`quant-8bit`, `quant-16bit`, `quant-32bit`, `quant-64bit`)
differ **only** in:

- `blocks/irp.py` — SCP coefficient dtype (`IRP_COEFF_SCALE`)
- `blocks/irp_cache.py` — cache key / serialisation
- `blocks/scp_disk_cache.py` — dtype-aware load/store
- C++ kernels (expand/load path for narrower dtypes)

All Python pipeline code (`fhe/`, `helpers/`, `scripts/`, `tests/`) is identical
across branches. The three markdown docs (`README.md`, `BENCHMARKS.md`,
`BOOTSTRAP_TO17_TODO.md`) are byte-identical across all four branches.

## Docs

| Document | Contents |
|---|---|
| `BENCHMARKS.md` | Precision-sweep table, per-stage/memory breakdowns, KSK dedup, bootstrap mechanism, C++ surface, reproduce steps |
| `BOOTSTRAP_TO17_TODO.md` | Forward-looking TODO: port `for_bootstrap_to_17_levels` for ~20.5-bit precision |
| `doc/design/*.md` (repo root) | FHE design rationale (IRP, KV-cache, DAG placement) |
| `doc/analysis/*.md` (repo root) | PPL plan + calibration variance analysis |
| `results/` (main branch) | CSVs + run summaries |
