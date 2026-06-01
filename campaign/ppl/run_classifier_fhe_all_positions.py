"""Helper: run_classifier_fhe_all_positions — returns (num_tokens, D_MODEL) hidden state.

Drop-in companion to run_classifier_fhe in llama3_mrpc.py. Same body, but
instead of calling yes_no_logits_np at the end it decodes ALL num_tokens
positions from the final-layer y_ct and returns them.

Slot layout (llama3_mrpc.py:1970-1973, llama3.py:46-62):
  slot[i * T_MODEL + d] = x[i, d]
  y_full[i*T_MODEL : i*T_MODEL + D_MODEL]  → position i hidden state
  y_full[::T_MODEL][:D_MODEL]              → position 0 (= y_full[0:D_MODEL])
  (existing MRPC code uses position P_local = query_position, but y_full[::T_MODEL]
   actually extracts positions 0, T_MODEL, 2*T_MODEL, ... not position P_local.
   For P_local == 0 they coincide; for PPL P_local = num_tokens-1 the correct
   extract is y_full[(num_tokens-1)*T_MODEL : (num_tokens-1)*T_MODEL + D_MODEL].
   The all-position form np.stack([y_full[i*T_MODEL : i*T_MODEL + D_MODEL]
   for i in range(num_tokens)]) is the correct generalization.)

This file is a DRAFT — not yet patched into llama3_mrpc.py. The function
should be added after run_classifier_fhe (around line 2007 of llama3_mrpc.py)
and before capture_pytorch_ref_with_model.

Usage in ppl_eval.py:
  from fhe.llama3_mrpc import run_classifier_fhe_all_positions
  y_per_pos = run_classifier_fhe_all_positions(
      num_tokens=64,
      pytorch_ref=pytorch_ref,       # (33, 64, D_MODEL) from ppl_window_*.npz
      pytorch_pre_norm=pytorch_pre_norm,  # (64, D_MODEL)
      cos_all_full=cos_all_full,     # from PROBE_FULL/rope_cos.npy
      sin_all_full=sin_all_full,     # from PROBE_FULL/rope_sin.npy
      label=f"ppl_w{w:04d}",
      engine=engine,                 # shared engine (reuse across windows)
      preloaded_weights=preloaded_weights,
      precomputed_calib=precomputed_calib,
  )
  # y_per_pos: (64, 4096) float64 — all-position hidden state after layer 31
"""


def run_classifier_fhe_all_positions(
        num_tokens, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label="ppl_window",
        engine=None, preloaded_weights=None, precomputed_calib=None):
    """End-to-end FHE forward returning ALL-POSITION hidden state at layer 31.

    Identical to run_classifier_fhe except:
    - query_position is fixed to num_tokens-1 (last real token).
    - After the final layer's y_ct, ALL num_tokens positions are decoded
      instead of only P_local.
    - Returns (num_tokens, D_MODEL) float64 array instead of (yes_logit, no_logit).

    The LM-head (final RMSNorm + full-vocab matvec) is NOT run here — that
    is the caller's responsibility via full_vocab_logprobs_np in lm_head_full.py.

    Sanity assertion built in:
      y_per_pos[num_tokens-1] == y_p_fhe (the existing P_local decode)
      max abs diff < 1e-9 (same decrypt, different slice).

    Args:
      num_tokens:          int — number of real tokens in the window (M=64 for PPL).
      pytorch_ref:         (33, num_tokens, D_MODEL) float64 — per-layer PT hidden states.
      pytorch_pre_norm:    (num_tokens, D_MODEL) float64 — pre-final-norm PT hidden state.
      cos_all_full:        (>=num_tokens, D_HEAD) float64 — RoPE cos table.
      sin_all_full:        (>=num_tokens, D_HEAD) float64 — RoPE sin table.
      label:               str — printed in per-layer drift log.
      engine:              phantom engine (shared across windows; None=build fresh).
      preloaded_weights:   optional preloaded layer weights dict.
      precomputed_calib:   optional precomputed calibration dict.

    Returns:
      y_per_pos: (num_tokens, D_MODEL) float64 — all-position hidden state
                 at the end of layer 31, post-residual2, stride-T_MODEL decoded.
    """
    # ---- This function body is the patch to paste into llama3_mrpc.py ----
    # Import everything from the module namespace (when pasted, these are
    # already in scope; shown here for clarity).
    import numpy as np
    import json
    from helpers.llama3 import D_MODEL, T_MODEL, NUM_SLOTS
    from fhe.llama3_mrpc import (
        run_classifier_fhe,
        PROBE_FULL,
    )

    # Delegate to the existing run_classifier_fhe with P = num_tokens-1,
    # then re-use the y_ct that was decrypted inside it. However,
    # run_classifier_fhe does not return y_ct — it only returns (yes, no).
    #
    # The minimal approach (zero change to existing code): run
    # run_classifier_fhe and separately call the FHE engine again to
    # decrypt y_ct. But run_classifier_fhe already decrypts y_ct at every
    # layer for the drift log — we just need to capture the last one.
    #
    # Cleanest minimal approach: pass a capture hook via a thread-local
    # list. The plan §5.3 notes the "zero-touch version decrypts twice on
    # layer 31 only" — that is what the patched function below does.
    #
    # For the DRAFT, we document the full patched function body here.
    # The actual implementation is inlined directly — no delegation to
    # run_classifier_fhe — to avoid a double-decrypt on layer 31.

    # ---- Inline implementation (same as run_classifier_fhe body) ----
    # NOTE: when pasting into llama3_mrpc.py, remove all the import
    # statements above and the module-path references; everything is
    # already in scope at that level.

    # The cleanest patch is to add ONE capture variable after the
    # residual2 compute and before the LM-head call:
    #
    #   # In run_classifier_fhe, after:
    #   #   y_ct = residual(ctx, x_mid_ct, mlp_out)
    #   # add:
    #   if layer_idx == NUM_DECODERS - 1:
    #       _y_ct_final = y_ct   # captured for all-position decode
    #
    # Then replace the LM-head call with:
    #   y_full_final = np.array(
    #       encoder.decode_double_vector(ctx, sk.decrypt(ctx, _y_ct_final)),
    #       dtype=np.float64)
    #   y_per_pos = np.stack(
    #       [y_full_final[i*T_MODEL : i*T_MODEL + D_MODEL]
    #        for i in range(num_tokens)], axis=0)   # (num_tokens, D_MODEL)
    #   return y_per_pos   # caller applies full_vocab_logprobs_np
    #
    # This is a 10-line diff in llama3_mrpc.py. For the PPL driver we
    # import this as a separate entry point; for MRPC backward compat
    # run_classifier_fhe stays untouched (returns yes/no logits).
    raise NotImplementedError(
        "This file is a DRAFT documenting the patch. "
        "The actual function must be added to llama3_mrpc.py. "
        "See the inline comments above for the exact diff."
    )


# ---------------------------------------------------------------------------
# Exact diff to apply to llama3_mrpc.py
# ---------------------------------------------------------------------------
# Paste AFTER the closing `return yes_logit, no_logit` of run_classifier_fhe
# (currently at llama3_mrpc.py:2007) and BEFORE capture_pytorch_ref_with_model.
#
# The diff is additive — run_classifier_fhe is NOT modified.
# ---------------------------------------------------------------------------

PATCH = r'''
def run_classifier_fhe_all_positions(
        num_tokens, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label="ppl_window",
        engine=None, preloaded_weights=None, precomputed_calib=None):
    """End-to-end FHE forward returning ALL-POSITION hidden state at layer 31.

    Identical body to run_classifier_fhe(query_position=num_tokens-1) but
    returns (num_tokens, D_MODEL) float64 instead of (yes_logit, no_logit).

    The existing MRPC per-layer drift log (y_full[::T_MODEL][:D_MODEL])
    extracts POSITION 0, not query_position, because stride-T_MODEL indexing
    means y_full[::T_MODEL] picks slots 0, T_MODEL, 2*T_MODEL, ... i.e.
    position 0 only.  The correct all-position decode is:
      y_per_pos[i] = y_full[i*T_MODEL : i*T_MODEL + D_MODEL]
    which for i=0 matches the existing y_full[::T_MODEL][:D_MODEL] exactly.

    Slot layout (llama3_mrpc.py:1970-1973):
      y_full = decoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct))
      y_p = y_full[::T_MODEL][:D_MODEL]   # position 0 only (drift log)
    """
    print(f"=== run_classifier_fhe_all_positions: {label}, NUM_TOKENS={num_tokens} ===")
    P_local = num_tokens - 1

    yes_logit, no_logit = run_classifier_fhe(
        num_tokens, P_local, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label=label,
        engine=engine,
        preloaded_weights=preloaded_weights,
        precomputed_calib=precomputed_calib,
        _expose_y_ct_final=_y_ct_final_container := [],
    )
    # NOTE: the above requires a small addition to run_classifier_fhe to
    # accept and populate _expose_y_ct_final.  See alternative below.

    # ---- SIMPLER ALTERNATIVE (zero change to run_classifier_fhe) ----
    # Re-run the final-layer decrypt directly.  run_classifier_fhe already
    # decrypts y_ct at layer 31 for the drift log; we do a second decrypt
    # here.  The extra cost is ~300 ms on a ~370 s window — noise.
    #
    # Implementation: copy run_classifier_fhe body verbatim, change the
    # final return statement:
    #
    #   # OLD (in run_classifier_fhe):
    #   yes_logit, no_logit = yes_no_logits_np(y_p_fhe, ...)
    #   return yes_logit, no_logit
    #
    #   # NEW (in run_classifier_fhe_all_positions):
    #   y_full_final = np.array(
    #       encoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct)),
    #       dtype=np.float64)
    #   y_per_pos = np.stack(
    #       [y_full_final[i * T_MODEL : i * T_MODEL + D_MODEL]
    #        for i in range(num_tokens)], axis=0)   # (num_tokens, D_MODEL)
    #   # Sanity: all-position[P_local] must match y_p_fhe (existing single-pos decode)
    #   assert np.abs(y_per_pos[P_local] - y_p_fhe).max() < 1e-9, (
    #       f"all-position decode mismatch at P_local={P_local}")
    #   return y_per_pos                            # (num_tokens, D_MODEL)
    raise NotImplementedError("See PATCH string and inline comments for the exact diff.")
'''

if __name__ == "__main__":
    print("This is a DRAFT — see PATCH variable for the diff to apply to llama3_mrpc.py.")
    print(PATCH)
