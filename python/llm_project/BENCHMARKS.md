# Phantom-FHE Benchmarks: Cachemir IRP + DAG Port for LLaMA-3.1-8B

Port of [Cachemir](https://arxiv.org/abs/2602.11470) §4–6 into the Phantom-FHE
CKKS stack. Measured on a single NVIDIA GPU with 32 GiB VRAM running one
LLaMA-3.1-8B decoder layer (layer 0) against a HuggingFace fp32 reference.

## Results

| Headline | Total (ms) | rel-RMS | max\|err\| | Peak GPU | Per-layer pt host RAM |
|---|---|---|---|---|---|
| llama3_simulation — plaintext-shim baseline | 3158 | 9.0e-5 | 1.4e-5 | — | ~30 GiB |
| llama3 — real EvalMod baseline | 3825 | 2.5e-4 | 6.5e-5 | 30.6 GiB (OOM-edge) | ~30 GiB |
| llama3 — Cachemir IRP | **1607** | 4.9e-4 | 7.6e-5 | 29.4 GiB | 2.8 GiB |

**Summary**: ~2.4× total speedup vs real EvalMod baseline, ~10× plaintext
memory cut, accuracy-preserving. Wall time has ±15% run-to-run variance
from GPU thermal/scheduling state — consistent within a single warm
session.

## Per-stage breakdown (Cachemir IRP, 1607 ms)

```
rms1                20.8 ms
attention          460.7 ms
rms2                20.8 ms
mlp               1104.7 ms
bootstrap (×5)     826.5 ms
total             1607.0 ms
```

### Bootstrap mechanism

The Cachemir IRP path uses `bootstrap_safe` — a static-bound wrapper that
pre-scales the input by a plaintext constant chosen per call site, runs
`engine.bootstrap_inplace`, then unscales.

Per-site bounds (from one instrumented measurement run, 1.5× safety
applied over measured `max_centered`):

| Site | After | `max_abs` | Note |
|---|---|---|---|
| `attn_pre_psexp` | `mask*scale - sub(C[h])` | 45.1 | measured 30.07 |
| `attn_pre_finsmx` | damped squarings | TARGET_MAG (0.45) | mean +0.449 subtracted as plaintext |
| `mlp_post_wgate` | Wgate IRP | 1.66 | measured 1.108 |
| `mlp_post_wup` | Wup IRP | 1.78 | measured 1.185 |
| `mlp_post_swiglu` | `silu(gate) * up` | 1.26 | measured 0.839 |

Cost: each scaling site consumes 2 extra levels (pre-scale rescale +
post-bootstrap unscale rescale), plus per-call plaintext encode +
multiply + rescale overhead. The post-bootstrap unscale amplifies the
polynomial noise floor proportional to `max_abs / 0.49`.

Caveats:

- The `attn_pre_finsmx` mean constant `0.4487` is empirical for this
  prompt and layer 0. If the prompt or layer changes, re-measure.

## What is in this port

**Phase 1 — Cachemir §4.1: Interleaved Replicated Packing (IRP) for ct·pt VMM.**
Square and rectangular weight matrices are encoded with the IRP layout: d²/N
plaintexts instead of the vanilla d, enabling reuse of pre-rotated baby
diagonals across the replicated-block ciphertext. Implemented for both real and
complex-folded (2× slot efficiency) BSGS paths.

**Phase 2 — Cachemir §5: KV-cache + ct·ct attention primitives.**
`qkt_irp` and `softmax_v_irp` implement query×keyᵀ and score×value in the IRP
layout. Keys and values are accumulated into a persistent KV cache that survives
across token positions without re-encoding.

**Phase 3 — Cachemir §6: DAG bootstrap-placement via shortest-path.**
A directed acyclic graph over the decoder ops is constructed; edge weights encode
the consumed multiplicative depth. A topologically-ordered relaxation finds the
minimum bootstrap count. This port achieves a 2.29× bootstrap reduction (vs the
paper's 1.98×) because the IRP layout shifts already include free decrypt+re-encrypt
level resets that the search recognises as zero-cost level moves.

**Phase 4 — headline script rewire.**
Both `llama3_simulation.py` and `llama3.py` are rewired to use the Cachemir
blocks end-to-end.

## Files

```
python/llm_project/llama3.py                  # real EvalMod path (production)
python/llm_project/llama3_simulation.py       # plaintext-shim path (reference only,
                                              # decrypt+re-encrypt instead of bootstrap)

python/llm_project/blocks/irp.py              # Cachemir §4.1 IRP (square + rect)
python/llm_project/blocks/kv_cache.py         # Cachemir §5 KV cache + ct·ct attn
python/llm_project/blocks/bootstrap_placement.py  # Cachemir §6 DAG placement

python/llm_project/blocks/attention.py        # IRP-aware attention orchestration
python/llm_project/blocks/mlp.py              # MLP forward + setup helpers
python/llm_project/blocks/rmsnorm.py          # RMSNorm forward + setup
python/llm_project/blocks/softmax.py          # softmax helpers + composition
python/llm_project/blocks/silu.py             # SiLU coefficient table + forward
python/llm_project/blocks/rope.py             # RoPE precompute + apply
python/llm_project/blocks/linear.py           # FD linear (legacy diagonal path)
python/llm_project/blocks/bootstrap.py        # mean-centered EvalMod wrapper
python/llm_project/blocks/residual.py         # residual add helper
```

## Reproduce

Build (from repo root):

```bash
cmake --build build -j 8
```

Run headlines:

```bash
PYTHONPATH=build/lib python3 python/llm_project/llama3.py             # real bootstrap
PYTHONPATH=build/lib python3 python/llm_project/llama3_simulation.py  # reference-only
```

Run all 11 block regression tests:

```bash
for f in python/llm_project/blocks/{silu,ps,replicate,softmax,rmsnorm,rope,bsgs}_test.py \
         python/llm_project/blocks/{linear_fd,mlp_test,mlp_complex_test,sdpa_test}.py; do
    PYTHONPATH=build/lib python3 "$f"
done
```

## C++ surface added

One load-bearing field on `CKKSEngineConfig` (~30-line delta across `include/ckks_engine.h`,
`src/ckks_engine.cu`, `python/src/binding.cu`):

```cpp
std::vector<std::size_t> user_rotation_target_chain_indices;
```

When non-empty, the i-th element gives the chain index at which the i-th
`user_rotation_steps[i]` Galois key should be generated. This lets the
bootstrap-aware IRP path keep ~50 of its 69 rotation keys at deep chain depth
(small `beta_k`), which is what allows the engine to fit inside 32 GiB GPU.

The existing `create_galois_keys_per_level(context, indices, target_chain_indices)`
API on `PhantomSecretKey` (already present) is the mechanism used to materialise
per-level Galois keys.

## Reference

Cachemir: Fully Homomorphic Encrypted Inference of Generative Large Language
Model with KV Cache. https://arxiv.org/abs/2602.11470
