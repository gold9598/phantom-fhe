# Design rationale: `fhe_attention_dense.py`

FHE (CKKS) dense token-major attention block for the LLaMA-3.1-8B inference
pipeline (split out of `llama3_mrpc.py`). This document holds the migrated
cryptographic-engineering rationale that previously lived as inline comments
and docstring prose. Each section is anchored; the code points to it via
`# design: doc/design/fhe_attention_dense.md#<anchor>`.

---

## k-cache-scale

K cache magnitude pre-scaler. Reduces `||K_h||_2` entering the QKT ct·ct
multiply — post-QKT err is Cauchy-Schwarz-bounded by `err_Q · ||K_h||_2`.
Default 0.25 -> 4x err reduction at zero level/runtime cost. Math is
invariant: `inv_sqrt_d` divides by the same factor downstream.

---

## c-per-head-real-keys

`c_per_head` is the per-head softmax shift the ciphertext actually
receives (via `qkt_irp_per_head_sub_plaintext` at stage-A). It MUST
be computed over the same real keys `[0, real_nt)` the FHE pipeline
reduces over (block_sizes clip masks pad keys to 0). Using the
padded keys here is the primary L0/L1 blow-up: at L0 the EOS-pad
raw scores are astronomically large, `c_per_head` is wildly off, and
`ps_exp` saturates -> ~1e45. Variable-nt: `real_nt == num_tokens`
-> identical, byte-for-byte.

---

## pack-kv-real-tokens-only

Pack/encrypt K/V for the REAL tokens `[0, real_nt)` only. The query
only ever attends real keys; the EOS-pad keys `[real_nt, num_tokens)`
must never enter the encrypted KV cache. Packing the padded
`num_tokens` at fixed-nt would encode/encrypt `ceil(num_tokens/T)`
ciphertexts (64 at nt=512) instead of `ceil(real_nt/T)` (8 for
real_nt=60). Beyond the obvious slot waste, feeding those extra
pad ciphertexts through the engine corrupts accuracy at scale
(uniform ~0.3 rel-RMS + L7/L8/L31 ~1e3 blow-ups at fixed-512;
survivable but still wrong at fixed-128). `pack_kv_blocks` already
zeros slots past each block_size and fills only block_size real
tokens, so packing real_nt yields EXACTLY the variable-nt-real_nt
blocks: structurally and numerically identical to a real
`nt=real_nt` prompt. BSGS / rotation / diagonal decomposition is
unchanged — only the (real-only) block count differs.
Variable-nt: `real_nt == num_tokens` -> identical, byte-for-byte.

---

## dense-kernel-contract

FHE dense token-major attention — THE compute path (IRP machinery deleted).

Kernel contract (same primitives `attention_forward_llama` uses for Stage A):

```
phantom.bsgs_matmul_preencoded (Wq, d_pad == D_TOTAL, replicated-block I/O)
  -> phantom.compute_qkt(q, k_shard_list, d_head)  (the multi-shard loop)
  -> fused multiply_plain (1/sqrt(d_head) * per-head mask * pad-token-zero)
  -> per-head sub_plain (c_per_head centering).
```

`baby_steps=64` for the `d_pad=4096` Wq BSGS: `bsgs_required_steps(64) ==
{1,2,4,8,16,32,64} == inner_sum(D_HEAD=128)` steps, all provisioned by
`build_user_steps_mrpc` (now built directly from the dense step builders).

Slot geometry is byte-identical to the verified numpy oracle
`blocks.kv_layout_dense` (commit 744e61f); the caller validates each layer
against `kv_layout_dense.dense_qkt` on the same Q/K.

The `_DENSE_WQ_BABY_STEPS = 64` constant: `bsgs_required_steps(64)` is a
subset of the provisioned steps.

---

## dense-full-pipeline

`fhe_attention_dense_full` — Stage 4 (dense-layout rewrite): close the dense
attention block.

Runs the FULL dense token-major attention pipeline end-to-end, ENTIRELY
in FHE, returning the post-Wo attention-output ciphertext (replicated-
block period D_TOTAL) so the caller can wire it into the residual stream:

```
QK^T (compute_qkt)  ->  scale*mask + per-head sub(C)  ->  bootstrap
  ->  ps_exp_init + damped squarings  ->  bootstrap (mean-centered)
  ->  STRICT 0/1 base-slot re-mask (the poly(0) trap fix)
  ->  per-shard sum_reduce(stride=D, count=P) -> cross-shard ADD
  ->  a-reset mask + Goldschmidt softmax_correct  (== exp / Σexp)
      [softmax weights kept ENCRYPTED per shard — token-major]
  ->  score_times_v (src/attention.cu kernel, AS-IS) over the
      softmax-weight cts + token-major V shards: mask base ->
      negative-stride d_head broadcast -> xV -> +d_total accumulate
      over P -> cross-shard ADD.  Because P*D == NUM_SLOTS exactly
      (8*4096 == 32768), the kernel's step-4 accumulate doubling is a
      full cyclic sum-reduce that REPLICATES the per-head attention
      output into ALL NUM_SLOTS/D_TOTAL periods -> slot[k*D + h*H + j]
      = Σ_tok w[tok,h]·V[tok,h,j] for every period k.  That IS the
      replicated-block period-D_pad layout BSGS Wo consumes (identical
      to encrypt_x_replicated_block's Wq input) — no phantom.replicate
      needed (its -4096/-8192 galois steps are NOT provisioned).
  ->  BSGS Wo (bsgs_matmul_preencoded, d_pad == D_TOTAL,
      baby_steps == _DENSE_WQ_BABY_STEPS == 64; bsgs_required_steps(64)
      = {1,2,4,8,16,32,64} ALL already provisioned -> ZERO new keys).
      Output: replicated-block period D_TOTAL, o_ct[k*D + i] = O[i].
```

The QK^T -> softmax stages are byte-identical to `fhe_attention_dense_softmax`
(same constants, same bootstraps, same poly(0)-trap re-mask, same
safety_scale, same a-reset); the ONLY difference is the softmax
weights are NOT decrypted — they stay as per-shard token-major
ciphertexts fed straight into the score_times_v kernel.

Args:
- `xn_query` : (D_MODEL,) rmsnormed hidden at the query position.
- `Wq_baked` : (D_TOTAL, D_MODEL) Wq with R_P (rope@query) pre-applied.
- `K_full_h` : (real_nt, N_HEADS, D_HEAD) rope-applied + GQA-expanded K.
- `V_full_h` : (real_nt, N_HEADS, D_HEAD) GQA-expanded V (NO rope).
- `Wo`       : (D_MODEL, D_TOTAL) output projection (NO R_P).
- `c_per_head` : (N_HEADS,) per-head softmax shift (real-key max + 0.5).
- `real_nt`  : real token count (== num_tokens for variable-nt).
- `chain_index` : fresh chain to encode/encrypt the dense inputs at.

Returns dict:
- `'o_ct'`        : post-Wo attention-output ciphertext (replicated-block
  period D_TOTAL); `o_ct decoded[i] == attn_out[i]` for `i in [0, D_MODEL)`.
  THIS is wired into the residual.
- `'fhe_attn_o'`  : (N_HEADS, D_HEAD) decrypted score·V output (pre-Wo)
- `'oracle_attn_o'`: (N_HEADS, D_HEAD) `kv_layout_dense.dense_score_v` on
  the IDENTICAL Q/K/V softmax weights (trusted spec)
- `'fhe_out'`     : (D_MODEL,) decrypted post-Wo attention output
- `'oracle_out'`  : (D_MODEL,) numpy `Wo @ (flattened oracle score·V)`
- `'P'`, `'n_shards'`

---

## qkt-irp-wq

QK^T via IRP-Wq (Cachemir §4.1) + `compute_qkt_irp` (Cachemir §5.1).

Wq: K=d²/N=512 SCPs (8x fewer than dense BSGS's 4096); `irp_matvec_host`
computes `y = x @ M` so we pass `Wq_baked.T` to get `q = Wq_baked @ xn_query`.
BRIDGE 1 (bridgeless): unfolded square Wq — no SK decrypt.
`irp_matvec_host` emits q in IRP stride-t (`slot[i*t]=q[i]`, `t=N/D=t_k=8`),
the layout `compute_qkt_irp` consumes directly. `bootstrap` refreshes
the chain; a lossless `mod_switch` restores user_level 11 so the downstream
§5.1/Stage-A/score_v galois target chains are byte-for-byte unchanged.

`_BABY_STEPS_IRP_Q = 16`: M=16, G=32 for d=4096 K=512 (~sqrt(K)).

Standard interleaved input: `slot[i*t]=xn[i]`.

Lazy-level: unfolded square matvec + mask + rescale = 2 levels.
Target input user_level 11 -> q at ~13; bootstrap pre-scale (+1) -> ul14 < max.

---

## q-bootstrap-mean-center

Bootstrap replaces the SK bridge. Mean-center before (boot assumes
~zero mean); `q_max_abs + |mean|` over-estimates centered max safely.

---

## q-restore-user-level

Restore user_level 11 (lossless `mod_switch`) so downstream galois targets
and K-cache chain are unchanged from the bridged path.

---

## qkt-irp-k-cache

Cachemir §5.1 `compute_qkt_irp` on IRP-Wq output. K cache packs
`t_k = NUM_SLOTS//D` tokens per ct in interleaved layout
(`slot[h*d_head*t + r*t + p] = K[c*t + p, h, r]`); `compute_qkt_irp` on
`(q_ct, k_chunk_ct)` yields scores at `slot[h*d_head*t + tok_local]`.
SK bridge per chunk: decrypt §5.1 scores, repack into dense Stage-A
base-slot layout (`slot[tok_local*D + h*H]`) per `kv_layout_dense_fhe.py:158`
so downstream Stage A/softmax/score_v consumes unchanged.

`t_k = 8` for LLaMA. `n_chunks_k = ceil(real_nt / t_k)`.

Build §5.1 K cache cts (one ct per chunk of t_k tokens).

---

## path-b-irp-native

Path B: IRP-native attention chain. Eliminates dense Stage A/B/C +
C++ `score_times_v` + IRP-Wo input SK bridge. Single-ct IRP layout across
the chain: `slot[h*d_head*t + tok] = m[tok, h]`.

Pad `real_nt` to next pow2 for `finalize_softmax_irp_t` (which asserts
`pow2>=2`). For real_nt=60 -> 64; for real_nt=512 -> 512.

---

## per-chunk-qkt-mask-scale

Per-chunk `compute_qkt_irp` + per-chunk Stage A mask*scale (Section 1
partial-junk fix: mask each chunk BEFORE tree-agg so the partial-junk
in slots `[h*1024+t..h*1024+1023]` doesn't pollute valid token slots in
the global ct).

Per-chunk mask*scale plaintext shared across chunks (every chunk has
t=8 valid slots per head at offsets `[0,t)`). Encoded LAZILY at the
post-`compute_qkt_irp` chain (`q.chain + 1` after the rescale inside
`compute_qkt_irp`).

Per-chunk mask*scale: keep `slot[h*1024+p] = inv_sqrt_d * m[c*t+p, h]`
for `p in [0, t)`; zero junk slots so tree-agg is collision-safe.

---

## tree-aggregate-scores

Tree-aggregate the `n_chunks_k` per-chunk score cts into one global
ct with `slot[h*1024 + tok] = (m[tok, h] - 0) / sqrt(d_head)` for h<nH,
tok<real_nt; zero elsewhere within each head's first-nt_pad slots.
Pre-condition: `n_chunks_k` must be a power of 2 (= nt_pad / t_k).
nt_pad/t_k = 64/8 = 8 chunks (3 levels of tree).

---

## pad-score-cts

Pad `score_cts_irp` to `n_chunks_pow2` with zero ciphertexts (mask*0 trick:
encode a zero ct at the chunk-mask chain so the tree-add is a no-op).
Create a zero ct at the same chain/scale as the masked score cts.

---

## global-per-head-sub

Global per-head `sub(c_per_head)`. Keeps `slot[h*1024 + tok]` valid
for tok in `[0, real_nt)` — the helper only writes the first real_nt
slots per head, so slots `[real_nt, nt_pad)` (the pad-to-pow2 buffer)
are untouched and remain zero from the per-chunk mask above.

---

## scores-calib-bound

Bootstrap-1 (post-Stage-A). Single ct vs dense's per-shard 4x.

`_SCORES_CALIB` bounds `max|centered scores|` (scores_post_C). It is held
at **45.10** (NOT tightened to the per-run max ~22.6): a LOOSER bound keeps
the extreme scores well inside the bootstrap EvalMod polynomial's
accurate CENTRAL region. Measured at nt=512: tightening to 23.76 lands
the -22.6 extreme near the `|x|->0.49` domain EDGE where the bootstrap poly
is least accurate, recovering -22.2 instead of -22.6 (Δ0.42); `ps_exp`
then amplifies that extreme-score error catastrophically (rel-RMS 1.1e-2
-> 2.5e19). The loose bound recovers -22.5 (Δ0.12) -> stable. So this
bound is deliberately conservative, not a stale nt=60 artifact.

---

## safety-scale

Safety scale + global pre-bootstrap mean (numpy oracle; layout-
agnostic). Identical derivation to the dense path; only the mean
reduction shape changes (global, not per-shard).

---

## global-pre-bootstrap-mean

Global pre-bootstrap mean: numpy poly applied to padded `(real_nt, nH)`
= `(60, 32)` scores; the rest of the slots are zero (`mask*0` -> `poly(0)`
= `_v0`). The IRP layout populates `real_nt*nH` slots; the rest are
`_v0`-valued junk after ps_exp+squarings (poly evaluated at 0 elementwise).

---

## stage-b-irp

Stage B IRP (single ct): `ps_exp_init` + damped squarings +
mean-centered bootstrap. Layout-agnostic (elementwise polynomial).

Bootstrap-2 (mean-centered).

---

## stage-b-mask

Stage B mask: keep `slot[h*1024 + tok]` for h<nH, tok<real_nt with
value `safety_scale`; zero elsewhere (especially slots `[real_nt, nt_pad)`
within each head's first-nt_pad block — required for `finalize_softmax_irp_t`'s
cyclic-replica precondition at num_tokens=nt_pad).

---

## stage-c-irp

Stage C IRP: single `finalize_softmax_irp_t` call. Cyclic-replica
at `-nt_pad` (= -64 for real_nt=60). All rotations in provisioned set.

---

## former-bootstrap-3-removed

(Former Bootstrap-3 removed.) `finalize_softmax_irp_t` outputs
`weights_ct` at user_level 7; the only downstream consumers are
tree-distribute (3 levels for nt_pad=64 / 8 chunks), `score_times_v_irp_multi`
(2 levels), and IRP-Wo (lazy-leveled via `mod_switch` to user_level
13, independent of the input level). So the real chain budget is
`7 + 3 + 2 = 12` at the score_v output, well under max_user_level 15
(3 levels of headroom). Wo's lazy `mod_switch` only ever drops deeper
(guarded `if chain_index < target`), so entering distribute at level 7
is safe. The earlier audit's "12 levels" referred to `softmax_correct`
inside `finalize_softmax_irp_t`, which runs UPSTREAM of this point — not a
downstream constraint. Dropping this bootstrap removes its ~170ms and its
injected noise.

---

## tree-distribute-weights

Tree-distribute global weights -> `n_chunks_pow2` per-chunk cts.
Inverse of tree-agg: at each level, split one ct into "lower" and
"upper" via mask+rotate. Only positive power-of-2 rotations needed
(all in provisioned set). For 8 chunks: 3 levels (8 -> 4 -> 2 -> 1
reversed = 1 -> 2 -> 4 -> 8). Each level uses one shared mask plaintext
at this level's "lower-half" pattern.

Process levels from coarse to fine: at level L (`W = nt_pad >> L`), we
have `2^L` blocks each holding W consecutive tokens. To split one block
of W tokens into two blocks of W/2 tokens:

```
lower = mask_low(block)              # keeps slots [h*1024+0 .. +W/2-1]
upper = mask_high(block) rotate +W/2 # shifts slots [h*1024+W/2 ..] -> [h*1024+0 ..]
```

Equivalently: `lower = block * lo_mask`; `upper = (rotate(block, +W/2)) * lo_mask`.
Use a single shared `lo_mask` per level.

`lo_mask`: 1.0 at `slot[h*1024+0..half-1]` for h<nH; zero elsewhere.
Upper half: rotate left by `+_half` (source slot `h*1024+half` lands
at `h*1024+0`), then mask.

---

## truncate-weights-blocks

Truncate `weights_blocks` to actual `n_chunks_k` (drop the pad-to-pow2
trailing blocks; they'll be ignored in the sum_v anyway since V cache
only has `n_chunks_k` chunks).

---

## score-times-v-irp

IRP-native `score_times_v_irp_multi` (per-chunk; sums across chunks).
`output_mask` is applied AFTER the ct·ct multiply+rescale in
`score_times_v_irp`, so encode at `weights_chain + 1`.

Align `v_cache_cts` chain to weights chain.

---

## decrypt-pre-wo-diagnostic

Decrypt the pre-Wo score·V output for the per-stage diagnostic. The
IRP layout is `slot[(h*d_head + j)*t] = attn[h, j]` (stride-t).

---

## bridge-2-wo

BRIDGE 2 (bridgeless): unfolded square Wo — no input or output SK bridge.
`attn_h` is already in IRP stride-t_wo layout (`slot[(h*d+j)*t_wo]=attn[h,j]`)
— the exact input layout `irp_matvec_host` expects for a d=D_TOTAL square.
Output is natural stride-t_wo=stride-T_MODEL (residual1 layout).
`bootstrap` replaces the output SK bridge.

`t_wo = 8 = T_MODEL`.

---

## lazy-level-wo

Lazy-level: Wo matvec (mask + caller rescale) costs 2 levels.
`attn_h` must be at `ul<=12` so output lands at `ul<=14 < max(15)`, leaving
1 level for bootstrap pre-scale.
If `attn_h` arrives deeper than ul12 (longer tree-distribute for nt_pad>=128),
pre-bootstrap it to refresh the chain first, then `mod_switch` to ul12.

`attn_h` is deeper than ul12; bootstrap it first using the oracle mean
from the already-decoded `_av_irp` (no extra decrypt needed).

---

## o-bootstrap-bridge

`bootstrap` replaces SK bridge. Mean-center before boot.

Diagnostic decode (stride-t_wo natural order).

---

## oracle-spec

Oracle: softmax weights -> `dense_score_v` -> Wo, on the IDENTICAL
teacher-forced Q/K/V (the trusted Stage-1 spec).

`Wo @ flattened attn` (`attn_flat[h*H+j] == oracle_attn_o[h,j]`).

---

## lazy-full-weight-cache

Module-level full-weight cache + lock used as a defensive fallback by
`_LazyLayerWeights`. As of the "preload all 9 weights" fix, the parallel
sweep pre-loads every key per layer up front, so the lazy fallback path
is normally never taken — `w[k]` always hits the subset dict and returns
without touching this lock. We keep the machinery in place purely as a
safety net: if a future caller passes a partial `preloaded_weights` dict
(only some keys), the lazy path will still satisfy the missing accesses
correctly (at the cost of the global-lock serialization that motivated
the preload-all fix). Cost when unused: zero.

---

## lazy-layer-weights-class

`_LazyLayerWeights` — Dict-like wrapper around a pre-loaded per-layer
weight subset.

DEFENSIVE FALLBACK ONLY. The parallel sweep now pre-loads all 9
weights per layer, so the subset is the full set and every
`__getitem__` returns from `self._subset` without entering `_full()`.
If a caller ever passes a partial subset, missed keys trigger a
one-shot `load_layer_weights(layer_idx)` cached in `full_cache`
under `lock` — note this serializes ALL worker threads on the lock,
which is why we now avoid it via the preload-all default.

Returns values directly from the subset when present; on a miss
(Wo/Wgate/Wup/Wdown for the per-example hot path), falls back to a
one-shot full `load_layer_weights(layer_idx)` cached in `full_cache`
under `lock`. Subsequent misses for the same layer hit the cache; a
miss in one worker thread populates the cache for all workers.

Supports `__getitem__`, `__contains__`, `__iter__`, and `get()` so it
is a drop-in stand-in for the subset dict at every call site that
treats it as read-only.

---

## contains-full-keyset

Treat the full on-disk weight set as the source of truth so callers
using `if k in w` (e.g. `encode_layer_irps`' subset check) see all
9 keys without forcing a disk load.
