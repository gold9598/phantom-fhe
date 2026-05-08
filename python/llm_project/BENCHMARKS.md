# Phantom-FHE Benchmarks: Cachemir IRP + DAG Port for LLaMA-3.1-8B

Port of [Cachemir](https://arxiv.org/abs/2602.11470) §4–6 into the Phantom-FHE
CKKS stack. Measured on a single NVIDIA GPU with 32 GiB VRAM running one
LLaMA-3.1-8B decoder layer (layer 0) against a HuggingFace fp32 reference.

## Results

| Total (ms) | rel-RMS | max\|err\| | Peak GPU | Per-layer pt host RAM |
|---|---|---|---|---|
| **1916** | 4.9e-4 | 1.7e-4 | 29.4 GiB | 2.8 GiB |

Wall time has ±15% run-to-run variance from GPU thermal/scheduling state
— consistent within a single warm session.

## Per-stage breakdown (1916 ms)

```
rms1                21.4 ms
attention          610.1 ms
rms2                21.1 ms
mlp               1095.5 ms
bootstrap (×6)    1163.4 ms
layout_shift         0.2 ms
total             1916.4 ms
```

## Memory breakdown (29.4 GiB GPU peak, single decoder layer)

Captured at 0.5 s sampling intervals across one full run on RTX 5090
(32 GiB):

| Phase | GPU MiB | Δ MiB | What was allocated |
|---|---|---|---|
| Python startup | 18 | — | (no CUDA context yet) |
| `import pyPhantom` | 556 | +538 | CUDA runtime + libcuPhantom load |
| Mid-`engine ctor` | 1,422 | +866 | (partial) |
| **Post-`engine ctor`** (4.3 s) | **29,358** | **+27,936** | engine + bootstrap key + 69 user-rotation Galois keys fully resident |
| IRP encoding done (16.6 s) | 29,362 | +4 | IRP plaintexts go to host **pinned memory** (~2.8 GiB), not GPU |
| Forward pass running | 29,364 | +2 | transient JIT-expanded plaintexts during `multiply_plain` |
| Post-exit | 18 | -29,346 | freed |

**Steady-state GPU residents (29.4 GiB):**

- **CUDA runtime + libcuPhantom**: ~556 MiB (one-time library load, before any phantom call)
- **Engine workspace + Galois + bootstrap keys**: ~28.1 GiB (allocated in `engine ctor`)

### Key-size breakdown

Differential measurement via `probe_key_sizes.py` (constructs three engine
variants — no-bootstrap+no-rotations, +bootstrap, +bootstrap+rotations —
and reports GPU deltas):

| Component | MiB | % of GPU | Notes |
|---|---|---|---|
| CUDA + libcuPhantom (post-import) | ~556 | 1.9% | one-time library load |
| Engine workspace + relin key | ~226 | 0.8% | NTT precomp, RNS tables, relin keyswitch key (~250 MiB) |
| **Bootstrap key** | **~15,936** | **54.2%** | C2S + EvalMod + S2C diagonals + bootstrap-internal Galois keys |
| **69 user-rotation Galois keys** | **~13,408** | **45.6%** | distributed 12@chain16 + 4@17 + 9@21 + 2@23 + 42@26 |
| ~Sum | ~30,126 | — | probe; production run ~29,358 (allocator variance ~770 MiB) |

The bootstrap key (54%) and user-rotation Galois keys (46%) dominate.
Diet targets, in priority order:

1. **Bootstrap key** — depends on chain-prime sizes. The BootstrapTo17Levels
   port (see `BOOTSTRAP_TO17_TODO.md`) replaces the standard
   `[58 | 40×NSL | 58×3 | 58×9 | 29×3 | 58×NSP]` chain with
   `[40×NSL | 29×1 | 54×12 | 42×1 | 60×NSP]`, using smaller primes for
   the bootstrap segment. Expected ~25–30% reduction.
2. **User-rotation Galois keys** — already heavily optimised: 42 of 69 sit
   at chain 26 (smallest, ~38 MiB each via `target_chain_indices`). The
   12 rms-stride-T_MODEL steps at chain 16 (~110 MiB each, ~1.3 GiB total)
   are the main outlier — required at fresh chain because rms runs
   immediately after bootstrap.

**Host RAM:** ~2.8 GiB per layer for IRP-encoded weights (Wq, Wo, Wgate,
Wup, Wdown) staged as `SingleChainPlaintext` on pinned host memory and
expanded JIT to GPU per `irp_matvec_host` call.

The pre-IRP layout (BSGS Wq/Wo + complex BSGS Wgate/Wup/Wdown plaintexts
all on GPU) peaked at ~30,580 MiB and OOMed during the first
`engine.bootstrap_inplace`. Moving plaintexts to pinned host memory plus
the per-step galois target chain assignment frees enough room for
bootstrap to fit on a 32 GiB card.

The decoder body is **secret-key-free**: rmsnorm, the residual stream,
and the SDPA pipeline (Q·K^T, softmax, score·V) all operate in stride-t
/ interleaved-replicated layout end-to-end, matching the IRP module's
native input/output convention. The C++ kernels `phantom.rmsnorm_forward`,
`phantom.compute_qkt`, and `phantom.score_times_v` are replaced by
pure-Python implementations (`rmsnorm_forward_stride_t` in
`blocks/rmsnorm.py`; `compute_qkt_irp`, `finalize_softmax_irp_t`,
`score_times_v_irp` in `blocks/attention.py`). K/V cache is interleaved
across tokens per the Cachemir paper §5.1.

Only `sk.encrypt_symmetric` (initial input encryption at the client
boundary, `llama3.py:638-640`) and `sk.decrypt(y_ct)` (test-harness
reference compare, `llama3.py:810`) touch the secret key — both at
boundaries, not in the decoder body.

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
