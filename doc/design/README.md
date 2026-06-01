# FHE design-rationale docs

This directory holds the design-rationale prose for the FHE (CKKS) LLaMA-3.1-8B
MRPC inference pipeline. Each `.md` here pairs with one compact `.py` module under
`../code/`. During the comment-migration pass, every multi-line "why" block —
cryptographic reasoning, constant justifications, slot-layout math, failure-mode
analysis, and tuning history — was moved OUT of the code and INTO these docs. The
compact modules keep only API-contract one-liners, section markers, and
operationally necessary `NOTE`/`WARNING` comments.

The link back is mechanical: at each migrated site the code carries a
`# design: <module>.md#<anchor>` pointer line at the original indentation, and
every pointer resolves 1:1 to a `## <anchor>` section in the matching doc here.
To recover the full reasoning for any code site, follow its `# design:` anchor
into these files; to find the code a section explains, grep the matching module
for the anchor slug.

## Shared document shape

Every doc follows the same structure so they read uniformly:

1. `# Design rationale: \`<module>.py\`` title.
2. A short source-module note (which `python/llm_project/<module>.py` it was
   migrated from, and what the module does).
3. `---`, then a sequence of `## <kebab-anchor>` rationale sections, each
   terminated by a `---` separator.

## The 6 design docs

| Doc | Module it documents | Rationale sections |
| --- | --- | --- |
| [`llama3_mrpc.md`](llama3_mrpc.md) | `llama3_mrpc.py` | 18 |
| [`decoder_layer.md`](decoder_layer.md) | `decoder_layer.py` | 36 |
| [`engine_setup.md`](engine_setup.md) | `engine_setup.py` | 22 |
| [`fhe_attention_dense.md`](fhe_attention_dense.md) | `fhe_attention_dense.py` | 32 |
| [`pytorch_ref.md`](pytorch_ref.md) | `pytorch_ref.py` | 4 |
| [`diagnostics.md`](diagnostics.md) | `diagnostics.py` | 3 |

Total: 115 anchored rationale sections across 6 docs.

---

### [`llama3_mrpc.md`](llama3_mrpc.md)

Slim end-to-end orchestrator: MRPC single-example FHE forward via multi-ct K/V
cache. After the Phase 1/2/3 refactor it only wires together the sibling modules
(engine setup, dense attention, PyTorch ref, decoder layer, diagnostics) and
preserves the byte-for-byte `run_classifier_fhe` contract, the `AUTONOMOUS_FHE`
residual-stream carry, and the bootstrap-placement REMAINING-BUDGET semantics.

Top anchors: `module-overview`, `run-classifier-fhe-orchestrator`,
`bootstrap-placement`, `autonomous-fhe-residual-stream`,
`preloaded-weights-rationale`, `classifier-setup-contract`.

### [`decoder_layer.md`](decoder_layer.md)

One full decoder-layer FHE forward (rms1 → attn → resid → rms2 → mlp → resid →
decrypt/log) plus the `ClassifierCtx` per-call state bundle. The largest doc:
covers silu Chebyshev/Clenshaw degree dispatch, the IRP-rect MLP (wide gate/up +
tall Wdown) with bridgeless layouts, lazy-level RNS scheduling, per-layer
calibration, and the I/O-bound prefetch/evict machinery.

Top anchors: `module-overview`, `run-decoder-layer-contract`,
`dense-attention-block`, `silu-degree-search`, `silu-clenshaw-dispatch`,
`lazy-level-gate-up`, `wdown-tall-matvec`, `prefetch-next-layer`.

### [`engine_setup.md`](engine_setup.md)

CKKS engine + Galois user-step construction and per-layer numpy calibration. Home
of the fixed-nt causal-correctness invariant (`real_nt = query_position + 1`),
the per-step Galois target-chain assignment with min-target-wins collision
resolution, the softmax safety-scale Goldschmidt-noise bound, and the IRP/dense
rotation step sets.

Top anchors: `module-contents`, `fixed-nt-causal-invariant`,
`build-user-steps-overview`, `setup-engine-target-chain`,
`softmax-safety-goldschmidt-noise`, `chain-depth-buckets`.

### [`fhe_attention_dense.md`](fhe_attention_dense.md)

The dense token-major attention block (THE compute path; IRP machinery deleted)
end-to-end in FHE: QK^T → scale/mask + per-head centering → bootstrap → ps_exp +
damped squarings → poly(0)-trap re-mask → sum_reduce → Goldschmidt softmax →
score·V → BSGS Wo. Also documents the `_SCORES_CALIB=45.10` conservative bound,
real-key-only K/V packing, and the lazy full-weight cache.

Top anchors: `dense-kernel-contract`, `dense-full-pipeline`,
`scores-calib-bound`, `c-per-head-real-keys`, `pack-kv-real-tokens-only`,
`k-cache-scale`, `lazy-full-weight-cache`.

### [`pytorch_ref.md`](pytorch_ref.md)

PyTorch reference-capture helpers: run the LLaMA-3.1-8B forward and capture all
hidden states + the pre-final-norm last hidden state, with an on-disk cache.
Documents the re-export contract that keeps these helpers importable under the
original `llama3_mrpc` name.

Top anchors: `module-contents-and-reexport`, `capture-with-model-contract`,
`capture-pytorch-ref-returns`, `cached-ptref-disk-cache`.

### [`diagnostics.md`](diagnostics.md)

Diagnostic / instrumentation helpers (`_malloc_trim`, `_probe`, and the shared
`_PROBE_DUMP_LAYER` list). Documents the cross-module list-identity requirement,
the glibc `malloc_trim` rationale for the streaming path, and the opt-in
`PROBE_DECRYPT_STAGES` decrypt-dump.

Top anchors: `probe-dump-layer-list-identity`, `malloc-trim-rationale`,
`probe-decrypt-stages-dump`.
