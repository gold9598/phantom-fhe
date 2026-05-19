"""Multi-ciphertext K/V cache layout for variable NUM_TOKENS up to 128.

The original llama3 pipeline hard-codes NUM_TOKENS=4 and packs the entire K
and V cache into a single ciphertext using:
    slot[(h*D_HEAD + j)*T_MODEL + tok] = K[tok, h, j]    for tok in [0, NUM_TOKENS)
where T_MODEL = NUM_SLOTS / D_MODEL = 8.

For NUM_TOKENS up to 128 (covers MRPC prompt lengths up to 111), that single-ct
layout would need N_HEADS*D_HEAD*NUM_TOKENS = 32*128*128 = 524288 slots, far
exceeding NUM_SLOTS=32768. Solution: split the cache across multiple
ciphertexts, each holding T_MODEL=8 token positions, using the same in-block
layout the existing FHE attention kernel knows how to read.

Layout for NUM_TOKENS = T (variable, up to 128):
    n_blocks = ceil(T / T_MODEL)
    Block k (k in [0, n_blocks)) holds tokens [k*T_MODEL, k*T_MODEL+T_MODEL):
        slot[(h*D_HEAD + j)*T_MODEL + tok_local] = K[k*T_MODEL + tok_local, h, j]
    The final block may be partial (only block_size = min(T_MODEL, T - k*T_MODEL)
    tokens are valid; remaining slots are zero).
"""
import math

import numpy as np


def pack_kv_blocks(K_full, V_full, num_tokens, t_model, num_slots, n_heads, d_head,
                     k_scale=1.0):
    """Pack K_full[num_tokens, n_heads, d_head] and V_full into a list of
    n_blocks slot vectors each (numpy float64 arrays of length num_slots).

    Args:
        k_scale: scalar applied to K values during packing. Used to reduce
            ||K_h||_2 entering the ct·ct multiply in compute_qkt — post-QKT
            err is Cauchy-Schwarz-bounded by err_Q · ||K_h||_2, so scaling K
            down by 4× at encoding time drops cascade post-QKT err 4×
            (1.16 → 0.29). The caller MUST compensate by dividing
            inv_sqrt_d by k_scale so post-stage-A scores are bit-identical.
            V is NOT scaled — only K participates in the noise-amplifying
            QKT dot product.

    Returns:
        k_blocks: list of n_blocks numpy arrays of shape (num_slots,)
        v_blocks: same shape
    where n_blocks = ceil(num_tokens / t_model).
    """
    n_blocks = math.ceil(num_tokens / t_model)
    k_blocks = []
    v_blocks = []
    for blk in range(n_blocks):
        block_start = blk * t_model
        block_size = min(t_model, num_tokens - block_start)
        k_slots = np.zeros(num_slots, dtype=np.float64)
        v_slots = np.zeros(num_slots, dtype=np.float64)
        for h in range(n_heads):
            for j in range(d_head):
                base = (h * d_head + j) * t_model
                for tok_local in range(block_size):
                    abs_tok = block_start + tok_local
                    k_slots[base + tok_local] = K_full[abs_tok, h, j] * k_scale
                    v_slots[base + tok_local] = V_full[abs_tok, h, j]
        k_blocks.append(k_slots)
        v_blocks.append(v_slots)
    return k_blocks, v_blocks


def unpack_kv_blocks(k_blocks, v_blocks, num_tokens, t_model, n_heads, d_head):
    """Inverse of pack_kv_blocks. Reconstructs K_full[num_tokens, n_heads,
    d_head] and V_full from the per-block slot vectors. The reconstructed
    arrays are numpy float64."""
    n_blocks = len(k_blocks)
    assert len(v_blocks) == n_blocks
    assert n_blocks == math.ceil(num_tokens / t_model)
    K_full = np.zeros((num_tokens, n_heads, d_head), dtype=np.float64)
    V_full = np.zeros((num_tokens, n_heads, d_head), dtype=np.float64)
    for blk, (k_slots, v_slots) in enumerate(zip(k_blocks, v_blocks)):
        block_start = blk * t_model
        block_size = min(t_model, num_tokens - block_start)
        for h in range(n_heads):
            for j in range(d_head):
                base = (h * d_head + j) * t_model
                for tok_local in range(block_size):
                    abs_tok = block_start + tok_local
                    K_full[abs_tok, h, j] = k_slots[base + tok_local]
                    V_full[abs_tok, h, j] = v_slots[base + tok_local]
    return K_full, V_full
