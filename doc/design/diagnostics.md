# Design rationale: `diagnostics.py`

Diagnostic / instrumentation helpers split out of `llama3_mrpc.py`, part of the
FHE (CKKS) LLaMA-3.1-8B inference pipeline. This document holds the hard-won
engineering rationale migrated out of the module's comments and docstrings. Each
section below is anchored by a kebab slug referenced from a `# design:` pointer
line in the code.

---

## probe-dump-layer-list-identity

Contains the `libc.malloc_trim` helper and the per-stage `_probe` decrypt
function used at ~25 call sites in `run_classifier_fhe` / `fhe_attention_dense_full`.

The module-level `_PROBE_DUMP_LAYER` list is mutated in place by
`run_classifier_fhe` (it writes `_PROBE_DUMP_LAYER[0] = layer_idx`) — keeping
that single list object identity across modules is required for the
diagnostic dump path to work. `llama3_mrpc` re-exports the name so the
mutation site continues to target the SAME list object the `_probe` reader
in this file sees.

---

## malloc-trim-rationale

`libc.malloc_trim` helper for the streaming-rp_indep path. Phantom's
`cudaMallocHost` pages live OUTSIDE glibc, but per-layer numpy
temporaries in `_build_irp_slots` and astype copies DO go through
glibc, and on a 62 GB box they accumulate uncoalesced free chunks
fast enough to push RSS past the ceiling between layers.

---

## probe-decrypt-stages-dump

DIAGNOSTIC ONLY (opt-in via `PROBE_DECRYPT_STAGES=1`). Dumps the full
decrypted slot vector to disk so an offline harness can compute the
rel-RMS vs plain-math per stage. When the flag is unset this block is
not entered: byte-identical to the original.
