"""Full-vocab LM-head readout for PPL evaluation.

Analogous to blocks/lm_head.py:yes_no_logits_np but for all 128256 vocab
positions. The full lm_head matrix is public (model weight), so doing the
matvec in plaintext on the host leaks nothing about the prompt — same
threat-model story as the existing Yes/No readout in lm_head.py.

Slot layout note (from llama3.py:46-62, llama3_mrpc.py:1970-1973):
  slot[i * T_MODEL + d] = x[i, d]
  → y_full[::T_MODEL][:D_MODEL]  extracts position 0
  → y_full[i*T_MODEL : i*T_MODEL + D_MODEL] extracts position i

For PPL we call full_vocab_logprobs_np with y_btd of shape (T, D_MODEL)
already extracted by the caller:
  y_btd = np.stack([y_full[i*T_MODEL : i*T_MODEL + D_MODEL]
                    for i in range(num_tokens)], axis=0)
Cite: llama3_mrpc.py:1970-1973 for the slot layout.
"""
import numpy as np


def rmsnorm_np_batch(x_td, gamma, eps=1e-5):
    """Batched final-layer RMSNorm over (T, D_MODEL).

    Args:
      x_td:  (T, D_MODEL) float64 — decrypted hidden states for all positions.
      gamma: (D_MODEL,) float64 — final RMSNorm weight.
      eps:   float — epsilon for numerical stability.

    Returns:
      (T, D_MODEL) float64 — normalized and scaled.
    """
    # rms per row: sqrt(mean(x^2) + eps), shape (T, 1)
    rms = np.sqrt((x_td ** 2).mean(axis=1, keepdims=True) + eps)
    return (x_td / rms) * gamma  # (T, D_MODEL)


def full_vocab_logprobs_np(y_btd, final_norm_g, lm_head_full, eps=1e-5):
    """Compute per-position log-softmax logits over the full vocabulary.

    Args:
      y_btd:        (T, D_MODEL) float64 — decrypted last-layer output
                    at EVERY position (all T positions in the window).
                    The slot layout is stride-T_MODEL; the caller extracts
                    this from y_full via:
                      np.stack([y_full[i*T_MODEL : i*T_MODEL + D_MODEL]
                                for i in range(T)], axis=0)
                    as specified in llama3_mrpc.py:1970-1973.
      final_norm_g: (D_MODEL,) float64 — final RMSNorm gamma weight,
                    loaded from PROBE_FULL/final_norm_g.npy.
      lm_head_full: (VOCAB, D_MODEL) float32 — full LM-head weight matrix,
                    loaded from ppl_prep/lm_head_full.npy.
      eps:          float — RMSNorm epsilon (from meta.json rms_norm_eps).

    Returns:
      (T, VOCAB) float64 — log-softmax log-probabilities.
                           logprobs[i, v] = log P(v | t_0 .. t_i).
                           Standard LM convention: logprobs[i] predicts
                           the token at position i+1. The last row
                           logprobs[-1] is unused for next-token prediction
                           (no ground truth to score against).
    """
    # Step 1: final RMSNorm — (T, D_MODEL)
    y_norm = rmsnorm_np_batch(y_btd, final_norm_g.astype(np.float64), eps=eps)

    # Step 2: LM-head matvec — (T, VOCAB)
    # lm_head_full is (VOCAB, D_MODEL); y_norm is (T, D_MODEL)
    # result = y_norm @ lm_head_full.T  shape (T, VOCAB)
    logits = y_norm @ lm_head_full.astype(np.float64).T   # (T, VOCAB)

    # Step 3: numerically stable log-softmax per row
    logits_max = logits.max(axis=1, keepdims=True)          # (T, 1)
    log_sum_exp = np.log(np.exp(logits - logits_max).sum(axis=1, keepdims=True)) + logits_max
    log_probs = logits - log_sum_exp                        # (T, VOCAB)

    return log_probs.astype(np.float64)


def next_token_logprobs(logprobs, token_ids, scoring_mask):
    """Extract per-position next-token log-probs for scored positions.

    Standard LM convention: logprobs[i] = log P(token at i+1 | context up to i).
    Position M-1 (the last in the window) has no "next" to predict; its
    logprobs row is unused. Position 0 is only scored in window 0; its
    log-prob comes from logprobs[-1] of the PREVIOUS window, which is
    handled at the outer loop level — here we simply use the in-window
    logprobs[i-1] for position i.

    Args:
      logprobs:     (T, VOCAB) float64 — from full_vocab_logprobs_np.
      token_ids:    (T,) int64 — token IDs for this window.
      scoring_mask: (T,) bool — True at positions contributing to H.
                    For window 0: all T positions True (but position 0
                    uses logprobs[-1] from previous context, not available
                    here; callers skip position 0 for window 0).
                    For window w>0: last S positions True.

    Returns:
      list of (position_i, token_id, log_prob) for each scored position
      where i >= 1 (position 0 skipped — no in-window predecessor logit).
    """
    results = []
    T = len(token_ids)
    for i in range(1, T):                  # position 0 has no predecessor
        if not scoring_mask[i]:
            continue
        # logprobs[i-1] = log P(·|t_0..t_{i-1}); target = token_ids[i]
        lp = float(logprobs[i - 1, token_ids[i]])
        results.append((i, int(token_ids[i]), lp))
    return results
