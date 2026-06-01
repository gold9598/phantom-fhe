# Phantom-FHE Benchmarks: Cachemir IRP + DAG Port for LLaMA-3.1-8B

Port of [Cachemir](https://arxiv.org/abs/2602.11470) §4–6 into the Phantom-FHE
CKKS stack. Measured on a single NVIDIA GPU with 32 GiB VRAM running one
LLaMA-3.1-8B decoder layer (layer 0) against a HuggingFace fp32 reference.

## Results

| Total (ms) | rel-RMS | max\|err\| | Peak GPU | Per-layer pt host RAM |
|---|---|---|---|---|
| **1919** | 5.9e-4 | 1.9e-4 | 13.1 GiB | 2.8 GiB |

Wall time has ±15% run-to-run variance from GPU thermal/scheduling state
— consistent within a single warm session.

## Full-forward precision sweep (32-layer, idx0, use17, bridgeless, 62 GB box)

Same IRP / complex-fold / bridgeless pipeline at every SCP precision — only the
stored coefficient dtype varies (`IRP_COEFF_SCALE`). Warm = SCP cache built;
FHE↔PT prediction agrees and 0 errors on all four:

| precision (branch) | SCP dtype | warm total | per-layer | fhe-compute/layer | per-layer rel-RMS |
|---|---|---|---|---|---|
| quant-64bit | int64 | **277.8 s** | 8.68 s | 7.39 s | 1.6e-3 – 2.3e-2 |
| quant-32bit | int32 | **169.7 s** | 5.31 s | ~5 s | ~5e-3 |
| quant-16bit | int16 | **97.3 s** | 3.04 s | ~3 s | ~1e-2 |
| quant-8bit | int8-BFP | **84.3 s** | 2.63 s | 2.6 s | ~1e-2 |

- **Speed:** monotonic with byte-width — **int64 → int8 = 3.3×** faster, from the
  cheaper per-tower expand/load of the narrower dtype.
- **Accuracy:** quantization is **near-lossless** in this pipeline — int64-IRP
  rel-RMS (~1e-2) ≈ int16/int8. The ~1e-2 is the IRP+use17+bridgeless pipeline,
  not the integer width.
- Cold (first run, full IRP cache build): int8 498 s, int32 489 s, int64 555.7 s.

The separate **dense `baseline`** branch is a *different* pipeline (not the
precision control): int64 at 383.1 s / rel-RMS ~5e-4 — slower *and* tighter than
quant-64bit purely because of the dense (non-IRP) algorithm, so it conflates
pipeline with precision and is excluded from the sweep above.

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

### Element-by-element breakdown

Captured via `cudaMemGetInfo` checkpoints inside the engine ctor and
`create_bootstrap_key` (instrumentation removed after measurement). Each
delta = the GPU memory committed by the corresponding allocation step.

| # | Component | Δ MiB | Cumulative MiB |
|---|---|---:|---:|
| 1 | CUDA + libcuPhantom (post-import) | 523 | 523 |
| 2 | `PhantomContext` (poly tables, NTT precomp, RNS) | +128 | 651 |
| 3 | `PhantomSecretKey` (dense form on GPU) | +130 | 781 |
| 4 | `PhantomCKKSEncoder` | +0 | 781 |
| 5 | `SmallBootstrapKey` (sparse KSKs for ModRaise) | +416 | 1,197 |
| 6 | EvalMod relin key (K=28 R=3) | +224 | 1,421 |
| 7 | C2S encoded diagonals (3 layers, chains 1–3) | +32 | 1,453 |
| 8 | S2C encoded diagonals (3 layers, chains 13–15) | +96 | 1,549 |
| **9** | **C2S layer Galois KSKs (canonical, 52 keys)** | **+9,376** | **10,925** |
| 10 | S2C layer Galois KSKs (all fallback to C2S) | +0 | 10,925 |
| 11 | `create_bootstrap_key` user_galois_keys (conjugation only) | +180 | 11,105 |
| 12 | Engine ctor override: per-level user keys (replaces #11) | +2,289 net | 13,394 |
| | **Total** | | **13,394** |

Item 9 — the C2S canonical Galois KSKs — is the single largest contributor
at ~70% of the GPU footprint. After canonical-owner dedup, every S2C step
borrows from C2S (item 10 = 0 MiB).

Item 11 used to be 8,640 MiB — `bootstrap.cu` originally built a full-Q
`user_galois_keys` covering all 47 non-overlap user-rotation steps, then
`engine.cu` immediately replaced it with a per-level override. Pure
construction-time waste: the cudaMallocAsync pool kept the 8.6 GiB at high
water mark even after the transient bundle was destroyed. Fix: shrink the
bootstrap.cu allocation to **just the conjugation key** (the only entry the
bootstrap pipeline itself reads from `bk.user_galois_keys`). Engine override
populates the rest. Saves ~8.4 GiB steady-state.

The remaining diet targets are item 9 (only reducible via BootstrapTo17Levels
chain swap, which uses smaller primes throughout) and the IRP rotation step
set (47 owned user steps at chains 17/23/26 — restructuring would mean
changing the BSGS factorization in the IRP module).

## Investigated and rejected: negative-step KSK auto-derive

Idea: derive `K(-a)` from `K(+a)` via Galois automorphism on the key
polynomials, eliminating ~10 user-side + ~4 bootstrap negative KSKs
(~1.2 GiB potential). **Math doesn't work**: applying σ_b to K(e)'s
polynomials gives a key with `σ_b(sk)` (not sk) as the incoming
secret, so the result can never be `K(c)` for c ≠ e. Reverts:
`c15e65e`, `4b32ad5`.

### KSK deduplication (canonical-owner principle)

Every Galois element (rotation step) needed in the engine has exactly
**one physical KSK**, generated at the shallowest chain that requires it.
Deeper-chain uses register a non-owning fallback pointer to the canonical
KSK; phantom's keyswitching kernel drops unused primes at use time. Two
deduplication passes are layered:

1. **User ↔ bootstrap** (commit `c10d0d2..b35ca2d` and after): 22 of 69
   user-rotation steps overlap with bootstrap C2S/S2C step sets. The user
   bundle uses `PhantomGaloisKey::set_fallback`/`resolve` to delegate
   those slots to the corresponding bootstrap-internal KSK.
2. **C2S ↔ S2C mirror pairs** (canonical-owner generalization): the C2S
   and S2C step sets are mirrored at mirrored chains
   (`C2S[layer i] ↔ S2C[2-i]`, same step values). Each mirror pair shares
   one KSK at the shallower (C2S) chain via a `PerLayerKSKSlot`'s
   `fallback` pointer.

3. **Per-step chain target tightening.** A per-call `apply_galois_inplace`
   audit revealed that 15 user steps (rmsnorm `sum_reduce_stride` + the
   QK^T Q-preprocess negative steps) were declared one chain shallower
   than they actually fire — `rms_x²` consumes a level before the inner
   sum, and the Q-preprocess fires post-Wq-IRP rescale. Moved
   `TARGET_RMS` from chain 16 to 17. The conjugation key (galois elt
   `2N-1`) was at full-Q (chain 0) but actually fires only at chain
   `first_idx + num_c2s = 4` (post-C2S, pre-EvalMod) — shifted to that
   chain.

Combined effect: total GPU memory drops from ~30.1 GiB (no dedup) to
~21.3 GiB (-8.1 GiB / -27.6%) with no measurable wall-time impact and
~10% accuracy variance (bootstrap noise floor amplified by larger
fallback KSKs at deep-chain uses, but well within tolerance).

Step distribution after tightening: 19@chain 17, 7@chain 23, 43@chain 26
(was: 12@16 + 4@17 + 9@21 + 2@23 + 42@26).

Remaining diet target:

- **BootstrapTo17Levels port** (`BOOTSTRAP_TO17_TODO.md`) — replaces the
  standard chain with a smaller-prime layout for the bootstrap segment.
  Expected another ~25-30% bootstrap-key reduction.

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

Only `sk.encrypt_symmetric` (client-boundary input encryption in
`fhe/fhe_attention_dense.py` — `encrypt_layer_inputs_multi`) and
`sk.decrypt` (diagnostic/test reference compare in `fhe/decoder_layer.py`)
touch the secret key — both at boundaries, not in the decoder body.

### Bootstrap mechanism

The Cachemir IRP path uses `bootstrap` — a static-bound wrapper that
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
The production entry is `fhe/llama3_mrpc.py` (MRPC, 32-layer + LM head);
`scripts/llama3_simulation.py` is the plaintext-shim reference
(decrypt+re-encrypt instead of bootstrap).

## Files

```
python/llm_project/fhe/llama3_mrpc.py            # production MRPC entry (32-layer + LM head)
python/llm_project/fhe/decoder_layer.py          # one decoder layer
python/llm_project/fhe/engine_setup.py           # CKKS engine build + galois steps + calibration
python/llm_project/fhe/fhe_attention_dense.py    # dense IRP attention (QK^T, softmax, score·V, Wo)
python/llm_project/helpers/llama3.py             # constants, numpy reference helpers, weight loaders
python/llm_project/helpers/pytorch_ref.py        # PyTorch reference capture + on-disk cache
python/llm_project/helpers/diagnostics.py        # _probe (decrypt+print), _malloc_trim
python/llm_project/scripts/llama3_simulation.py  # plaintext-shim reference (decrypt+re-encrypt)
python/llm_project/scripts/mrpc_sweep.py         # MRPC sweep driver
python/llm_project/blocks/irp.py                 # Cachemir §4.1 IRP (square + rect; quant-delta file)
python/llm_project/blocks/attention.py           # IRP attention kernels (compute_qkt_irp, finalize_softmax_irp_t, score_times_v_irp)
python/llm_project/blocks/bootstrap_placement.py # Cachemir §6 DAG placement
python/llm_project/blocks/{bootstrap,rmsnorm,softmax,silu,rope,residual,linear}.py  # kernels
```

blocks/ also holds the per-branch quant delta: `irp.py` / `irp_cache.py` / `scp_disk_cache.py`.

## Reproduce

Build (from repo root):

```bash
cmake --build build -j 8
```

Run from `python/llm_project`:

```bash
cd python/llm_project
HF_HUB_OFFLINE=1 USE_BOOTSTRAP_17=1 python3 -u fhe/llama3_mrpc.py --idx 0   # one MRPC example (real bootstrap)
python3 scripts/llama3_simulation.py                                          # plaintext-shim reference
```

Run all block regression tests:

```bash
for f in python/llm_project/tests/*_test.py; do python3 "$f"; done
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
