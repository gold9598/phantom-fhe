"""LM-head wiring for the MRPC Yes/No classifier (Stage 3b-e).

The full LM head is D_MODEL × VOCAB = 4096 × 128256 (~2 GB at fp32). For MRPC
prompt-classification we only need 2 specific token logits (" Yes" id=7566,
" No" id=2360), so we keep just those 2 rows of the LM-head weight matrix
(extracted to /tmp/llama_probe_full/lm_head_yesno.npy in Phase 2).

The pipeline that the FHE side runs to produce the 2 logits:
  y_ct (post-residual2 of layer 31, stride-T_MODEL hidden state, D_MODEL slots)
   -> final RMSNorm with γ = final_norm_g
   -> 2 dot products: logit = (final_norm_g · y / sqrt(z + eps)) · lm_head_row
   -> decrypt 2 scalar logits

For Stage 3b-e we keep the final RMSNorm + dot-product OUT of FHE for the
first end-to-end demo: the FHE pipeline ends at y_ct, the host decrypts the
post-residual2 hidden state, and the final RMSNorm + lm_head_yesno matvec
runs in plaintext on the host. This avoids re-engineering the rmsnorm
polynomial domain for the final-layer hidden state magnitudes (||y_31||
reaches ~134 — well past the per-layer rmsnorm calibration windows).

The multiply-by-lm-head step is information-free at the boundary: the
host already has lm_head_yesno (a public model parameter), so doing the
matvec in plaintext leaks nothing about the prompt. Only the final 2
scalar logits cross the trust boundary. A future Stage 3b-e' can move
the final RMSNorm + matvec into FHE if the threat model demands it.
"""
import numpy as np


def rmsnorm_np(x_d, gamma, eps=1e-5):
    """Final-layer RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * gamma. Operates
    on a single D-dim vector."""
    rms = np.sqrt((x_d ** 2).mean() + eps)
    return (x_d / rms) * gamma


def yes_no_logits_np(y_d, final_norm_g, lm_head_yesno, eps=1e-5):
    """Compute (yes_logit, no_logit) given a single hidden state vector.

    Args:
      y_d: hidden state at the query position, shape (D_MODEL,).
      final_norm_g: final RMSNorm γ, shape (D_MODEL,).
      lm_head_yesno: 2 LM-head rows (Yes, No), shape (2, D_MODEL).
      eps: RMSNorm epsilon.

    Returns:
      (yes_logit, no_logit) floats.
    """
    y_norm = rmsnorm_np(y_d, final_norm_g, eps=eps)
    yes_logit = float(np.dot(lm_head_yesno[0], y_norm))
    no_logit  = float(np.dot(lm_head_yesno[1], y_norm))
    return yes_logit, no_logit


def fhe_decrypt_extract_y(ctx, encoder, sk, y_ct, t_model, d_model):
    """Decrypt the final FHE hidden state ct and extract the D_MODEL-sized
    vector at the query position from its stride-T_MODEL slot layout."""
    full = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct)),
                    dtype=np.float64)
    return full[::t_model][:d_model]
