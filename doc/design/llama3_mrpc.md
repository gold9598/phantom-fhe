# Design rationale: `llama3_mrpc.py`

End-to-end MRPC single-example FHE forward via multi-ct K/V cache. This document
holds the design-rationale prose migrated out of the slim orchestrator module
`llama3_mrpc.py`. Each section below is anchored by a short kebab slug referenced
from a `# design:` pointer comment at the corresponding site in the code.

---

## module-overview

Stage 3b-f-1: skeleton that runs at NUM_TOKENS up to T_MODEL=8 (single
block, n_blocks=1). Verifies the multi-ct attention path reduces to the
single-ct path on a known input. Same prompt as the existing llama3.py
sanity check: [BOS, "The", " quick", " brown"] (4 tokens).

3b-f-2 will scale to NUM_TOKENS=64 (n_blocks=8) on a real MRPC prompt.

Phase 1/2/3 split (refactor, byte-identical):
  - Phase 1: engine setup, dense attention block, PyTorch reference capture,
             diagnostic helpers moved to sibling modules.
  - Phase 2: run_classifier_fhe split into _classifier_setup +
             _run_decoder_layer + _run_lm_head in-file helpers.
  - Phase 3: ClassifierCtx + run_decoder_layer promoted to decoder_layer.py;
             this file is now the slim orchestrator only.

External import paths (mrpc_sweep, mrpc_sweep_parallel, precapture_ptref, etc.)
keep working unchanged via re-export at the top of this file.

---

## reexport-diagnostics

diagnostics: _malloc_trim, _probe, plus the shared _PROBE_DUMP_LAYER list
(mutated in place by run_classifier_fhe — identity must be preserved).

---

## reexport-engine-setup

engine_setup: galois step set + engine builder + per-layer calibration.

---

## reexport-fhe-attention-dense

fhe_attention_dense: encrypt_layer_inputs_multi, _LazyLayerWeights, and
fhe_attention_dense_full. K_CACHE_SCALE / _DENSE_WQ_BABY_STEPS and the
lazy full-weight cache+lock are re-exported for back-compat / shared
state identity with the moved class.

---

## reexport-pytorch-ref

pytorch_ref: PT model forward + on-disk cache.

---

## reexport-decoder-layer

decoder_layer (Phase 3): ClassifierCtx dataclass + run_decoder_layer.

---

## classifier-setup-contract

Byte-for-byte equivalent to the head of the original run_classifier_fhe
(down to and including the weight-subset preload print).

---

## preloaded-weights-rationale

Per-layer weight accessor. The parallel sweep pre-loads the per-example
subset (Wq/Wk/Wv/g1/g2) ONCE on the main thread and passes the dict
here via preloaded_weights; serial / legacy callers leave it None and
fall back to the original per-call np.load. py-spy showed concurrent
workers stuck on disk I/O + glibc malloc contention inside
load_layer_weights (~128 MB allocations × 9 keys × 4 threads); the
pre-load eliminates that contention entirely.

The preloaded subset is missing the R_P-independent keys
(Wo/Wgate/Wup/Wdown). Most consumers (encode_layer_irps, attention/MLP
blocks) serve those from the shared rp_indep_cache and never touch
`w[...]` directly, but a few call sites (e.g. compute_layer_calib_n in
this module) do read them. We wrap the subset in _LazyLayerWeights so
any missed key triggers a one-shot full load_layer_weights() on first
access. The full-weight cache is module-level so the cost is paid ONCE
per layer across the entire sweep (all examples, all workers).

---

## engine-reuse-rationale

Engine. If caller supplies one, reuse it (required when sharing
an rp_indep_cache of plaintexts across calls — plaintexts are bound to
the engine's (ctx, encoder) and become invalid if the engine is rebuilt).

---

## freshest-chain-assert

Galois-key target chains were computed against FRESHEST_CHAIN=16; fail
fast if the actual freshest chain has moved (e.g. evalmod_r=4).

---

## irp-masks-removed

(IRP layer-independent masks removed — the dense token-major path
builds its plaintext masks per-shard inside the dense kernels.)

---

## bootstrap-placement

Bootstrap placement (same as llama3.py).

output_level here is the planner's REMAINING-BUDGET level (NSL_MAX = fresh,
0 = exhausted), the inverse of the runtime's consumed-level view.
attention output is SK-bridged (decrypted + re-encrypted at fresh_ci, see
the attn_out re-encrypt below), so it emerges FRESH — NOT at the IRP output
level. Modeling it as fresh stops the planner scheduling a redundant
bootstrap_before rms2 (which was firing on a user_level-0 ct). NOTE: revert
to OUTPUT_LEVEL_AFTER_IRP if the SK output bridge is ever removed (bridgeless
autonomous path), where attention really does emerge deep.

---

## autonomous-fhe-residual-stream

Per-layer FHE forward.

Opt-in autonomous residual stream (mirrors reverted commit 625ea9c
"llama3: bootstrap y_ct forward as next-layer x_ct"). When
AUTONOMOUS_FHE=1, layer >= 1 feeds the PREVIOUS layer's output
ciphertext y_ct forward (bootstrapped to the same fresh chain /
scale / stride-T_MODEL layout that encrypt_layer_inputs_multi
produces for x_ct) instead of re-encrypting pytorch_ref[layer_idx].
K/V/c_per_head still come from the clean numpy ref (x_btd) exactly
as in 625ea9c, so the per-layer decrypt/log now measures the TRUE
drift of the carried encrypted state vs pytorch_ref[layer_idx+1].
Default (unset) leaves the guided path byte-for-byte unchanged.

---

## weight-subset-preload

Per-layer weight-subset preload. The dense token-major pipeline
rebuilds Q/K/V/Wo/Wgate/Wup/Wdown directly from the numpy weights
each layer (no pre-encoded IRP plaintexts), so all that is needed
up front is the small per-example-hot subset (Wq/Wk/Wv + g1/g2,
used by encrypt_layer_inputs_multi & rmsnorm). The big R_P-indep
matrices (Wo/Wgate/Wup/Wdown) are loaded per-layer via
load_layer_weights_subset inside the dense attention / MLP blocks.

The shared_wq_cache* / rp_indep_cache / rp_indep_disk_root params
are retained for call-signature compatibility with the sweep
drivers but are inert now that the IRP machinery is gone (the
pre-encoded-IRP cache they fed no longer exists).

---

## run-lm-head-contract

Byte-for-byte equivalent to the tail of the original run_classifier_fhe
(after the per-layer loop).

---

## run-classifier-fhe-orchestrator

Slim orchestrator. The setup, per-layer compute, and LM head live in
`_classifier_setup` (this file), `decoder_layer.run_decoder_layer`, and
`_run_lm_head` (this file). AUTONOMOUS_FHE carry semantics, all
diagnostic prints, and all bootstrap firings are preserved byte-for-byte.

Args:
  num_tokens: actual number of tokens in the prompt (NUM_TOKENS).
  query_position: position to query for next-token logit (typically num_tokens-1).
  pytorch_ref: (33, num_tokens, D_MODEL) per-layer hidden states from PyTorch.
  pytorch_pre_norm: (num_tokens, D_MODEL) pre-final-norm last hidden state.
  cos_all_full / sin_all_full: RoPE tables of shape (>=num_tokens, D_HEAD).
  label: short string for printing.

---

## run-mrpc-example-contract

If truncate_to is set, use only the first `truncate_to` tokens (for
num_tokens-vs-error sweep).

---

## main-4tok-contract

Stage 3b-f-1 sanity: 4-token "[BOS] The quick brown" via the same
pipeline. Loads the precomputed pytorch_ref / pre_norm from probe v2
rather than re-running PyTorch.
