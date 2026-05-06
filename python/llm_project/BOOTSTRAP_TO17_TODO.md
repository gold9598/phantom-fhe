# BootstrapTo17Levels port — TODO

Port `the_lib`'s `CKKS_42_54_29_40_60_BOOTSTRAP` (= `for_bootstrap_to_17_levels`)
into our phantom-fhe fork to close the precision gap from ~16 bits avg
(current) to the_lib's ~20.5 bits.

## Reference / measurements

| | avg \|err\| | bits |
|---|---|---|
| **the_lib** `for_bootstrap()` (CKKS_54_60_BOOTSTRAP, 54-bit primes) | 6.64e-07 | ~20.5 |
| **lapis** 29-58-4sp K=28 (matches our legacy chain) | 7.60e-06 | ~17.0 |
| **our port (legacy chain)** 29-58 K=28 | 1.48e-05 | ~16.0 |
| **gap to the_lib** | 22× | ~4.5 bits |

Root cause (per lapis `evalround-plus.md`): the_lib's 3 architectural changes
that we don't have:

1. Per-level `ckks_scale[]` (squared) + `ckks_rescaled_scale[]` (single)
   arrays — encoding scale decoupled from chain prime.
2. Encode-then-rescale for diagonals — encode at "double-prime" (squared)
   scale, rescale during encoding.
3. Rescale-first butterfly — rescale `ct` BEFORE multiplying by encoded
   diagonal.

Plus a chain layout that uses the_lib's distinct prime pools:
`[40 (small) | 29 (scale_down) | 54 (bootstrap) | 42 (coeff_to_slot) | 60 (large)]`.

---

## Phases

### ✅ Phase 1 — two-scale precomputed arrays

- [x] `CKKSEngineConfig::build_two_scale_arrays` flag (default false →
      backward compat)
- [x] `CKKSEngine::ckks_scale_at(idx)` and `ckks_rescaled_scale_at(idx)`
      accessors mirroring the_lib `precomputed_.ckks_scale_` /
      `ckks_rescaled_scale_` (`src/ckks/scale.cpp:1420 make_ckks_scales`)
- [x] Recurrence: `rescaled[i] = scale[i] / q_drop[i]`,
      `scale[i+1] = rescaled[i]^2`
- [x] Python binding (`build_two_scale_arrays`, `scale_array_size`,
      `ckks_scale_at`, `ckks_rescaled_scale_at`)
- [x] Phase-1 probe: recurrence verified across 30 adjacent pairs

### ✅ Phase 2 — chain layout

- [x] `CKKSEngineConfig::use_bootstrap_to_17_levels` flag (opt-in)
- [x] When set: build chain `[40×NSL | 29×1 | 54×12 | 42×1 | 60×NSP]`
      instead of `[58 | 40×NSL | 58×3 | 58×9 | 29×3 | 58×NSP]`
- [x] Validated config: `NSL=18, NSP=8` (size_Q=32 ✓, max_user_level=17)
- [x] Skip `create_bootstrap_key` on new chain (Phase 3 wires it up)
- [x] `freshest_chain_index_ = first_idx + 13` matches
      `1 (C2S) + 9 (ER) + 3 (S2C)`; the 29-bit scale_down prime is consumed
      by the user's first post-bootstrap rescale (level 17 → 16 in
      the_lib's level numbering)
- [x] Phase-2 probe: chain bits verified (60 / 42 / 54×12 / 29 / 40×16
      drop sequence matches expected)
- [x] Python binding (`use_bootstrap_to_17_levels`)

### 🟡 Phase 3a — single-stage C2S apply (skeleton)

- [x] `single_stage` param on `apply_linear_transform_inplace` — skips
      per-layer `rescale_to_next_inplace`, performs ONE final rescale
      mirroring the_lib `rescale_after_multiply` at
      `src/ckks/engine/bootstrap.cpp:733` inside
      `coeff_to_slot_complex_for_17_levels`
- [x] Public `apply_c2s_inplace_single_stage` wrapper in `bootstrap.h` /
      `bootstrap.cu`
- [x] Phase-3.0 guard: `bootstrap_inplace` throws clear "not implemented"
      on `use_bootstrap_to_17_levels` chain instead of segfaulting on empty
      `bk_`
- [ ] Wire `apply_c2s_inplace_single_stage` into `bootstrap()` (Phase 6)

### ⏸️ Phase 3b — single-stage diagonal pre-encoding

- [ ] `single_stage` param on `pre_encode_diagonals` — encodes ALL layer
      diagonals at the SAME `target_chain_index` (instead of
      `start_chain_index + layer`)
- [ ] Use a fixed encoding scale matching the_lib's `COEFF_TO_SLOT_SCALE`
      (= `1L << 30` per `the_lib/src/operation/bootstrap.h`)
- [ ] Mirror the_lib's `make_coeff_to_slot_stage_for_bootstrap_to_17_levels`
      at `src/ckks/engine/bootstrap.cpp:79` (passes `rescale=false`,
      `rescale_to_key=false`, fixed `COEFF_TO_SLOT_SCALE`)

### ⏸️ Phase 3c — `create_bootstrap_key` for new chain

- [ ] Variant of `create_bootstrap_key` that:
  - Encodes 1-stage C2S diagonals (Phase 3b) at chain index 1 (top of main)
  - Encodes 3-layer S2C diagonals at the 54-bit bootstrap-segment chain
    indices (post-EvalMod chain)
  - Builds relin / Galois keys at the new chain depths
- [ ] Wire into `CKKSEngine` constructor when `use_bootstrap_to_17_levels`
- [ ] Remove the Phase-2 early-return; engine now constructs `bk_` for the
      new chain too

### ⏸️ Phase 4 — encode-then-rescale for diagonals

- [ ] Mirror the_lib `encode_for_bootstrap` (`src/ckks/core/encode.cpp:244`):
  - `ckks_scale = ckks_rescaled_scale[level]` (single, post-rescale)
  - `rescale = true`, `rescale_to_key = true`
- [ ] Underlying encoder needs to fold a rescale into NTT/RNS conversion;
      may need new helper in `single_chain_plaintext.cu`
- [ ] This is the "encode at squared scale, then rescale during encoding"
      step that decouples encoding precision from chain prime size

### ⏸️ Phase 5 — rescale-first S2C

- [ ] Modify `apply_linear_transform_inplace` (or add a `rescale_first`
      flag) to mirror the_lib `slot_to_coeff_` (`bootstrap.cpp:794`):
  ```cpp
  for stage in stages:
      rescale(ct);                               // rescale FIRST
      apply_butterfly(ct, encoded_diag);         // then multiply
  ```
- [ ] S2C diagonals must be encoded at the post-rescale chain prime so
      scales line up

### ⏸️ Phase 6 — `bootstrap()` body for new chain

- [ ] Branch on `bk.use_to17_levels` flag in `phantom::bootstrap`:
  ```
  scale_up_for_bootstrap → mod_raise → apply_c2s_inplace_single_stage
    → conjugation_split → eval_round (K=28 R=3) → apply_s2c_inplace
    → saved-out align (use bootstrap_level_down_ratio from Phase 1 arrays)
    → final extra `multiply(bootstrapped, 1.0 / scale_up_ratio)` for
      to_17_levels (mirrors the_lib `bootstrap.cpp:1078`)
  ```
- [ ] Update saved/qi alignment math to use `ckks_scale_at()` /
      `ckks_rescaled_scale_at()` instead of hand-rolled D_exact computation
- [ ] Drop the Phase-3.0 guard in `CKKSEngine::bootstrap_inplace`

### ⏸️ Phase 7 — wrappers / tests / llama3.py

- [ ] `bootstrap_test.py`: tighten tolerances to ~20-bit precision
      expectation when running on the new chain
- [ ] Add new probe `probe_bootstrap_to17.py` that runs full bootstrap on
      the new chain, prints avg/max bits, compares to the_lib reference
      (`/home/yongwoo-oh/the_lib/build/bin/examples/ckks_bootstrapping_cuda`
      avg = 6.64e-07 = ~20.5 bits)
- [ ] `boot_centered`: lift the `TARGET_MAG = 0.49` ceiling — if
      bootstrap precision is now ~20 bits absolute, scale-down for
      large-magnitude inputs may not be needed for moderate ranges
- [ ] `llama3.py`: re-derive `freshest_chain_index = 14` (not 16),
      adjust per-step galois target chain indices, possibly swap
      `NUM_SCALE_LEVELS = 14` → `18` for more user-level headroom

---

## Risks / open questions

1. **128-bit security at NSL=18 + NSP=8**: total log_q = 1919 bits at
   logN=16 — above the standard 1730-bit ceiling. May need to fall back
   to `NSL=14, NSP=8` (size_Q=28, log_q=1199 bits) or accept the security
   margin if the underlying lattice tables permit.
2. **Encoder rework for encode-then-rescale**: phantom-fhe's current
   `PhantomCKKSEncoder` does not support encode-then-rescale natively;
   may need to add a helper or extend the encoder.
3. **OVER_SCALED state**: `PhantomCiphertext` has no flag for "ct is
   over-scaled, awaiting rescale". Either add one (touches phantom-fhe
   core) or track externally in our wrappers.
4. **llama3.py level budget**: pipeline currently fits in NSL=14
   (max_user_level=13); switching to NSL=18 (max=17) gives more headroom
   but per-step galois assignments need re-derivation. Some IRP step
   targets may shift.

---

## File-by-file ownership

| File | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 |
|---|---|---|---|---|---|---|
| `include/ckks_engine.h` | ✅ | ✅ | — | — | — | — |
| `src/ckks_engine.cu` | ✅ | ✅ | — | — | — | — |
| `include/bootstrap.h` | — | — | ✅ (3a) | TBD | TBD | TBD |
| `src/bootstrap.cu` | — | — | ✅ (3a), TBD (3b/3c) | TBD | TBD | TBD |
| `src/single_chain_plaintext.cu` | — | — | — | TBD | — | — |
| `python/src/binding.cu` | ✅ | ✅ | — | — | — | — |
| `python/llm_project/blocks/bootstrap.py` | — | — | — | — | — | TBD |
| `python/llm_project/blocks/bootstrap_test.py` | — | — | — | — | — | TBD |
| `python/llm_project/llama3.py` | — | — | — | — | — | TBD |
