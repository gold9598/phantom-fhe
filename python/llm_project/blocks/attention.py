"""
Attention orchestration ported from src/attention.cu to Python.

C++ primitives (ct x ct + lazy relin/rescale): phantom.compute_qkt, phantom.score_times_v.
Everything else here is pure orchestration over those CUDA primitives.

encode_scale convention: callers pass encode_scale (default = ct.scale()) as
the plaintext encode scale.  For BITS-uniform chains every middle prime is
~2^40 = SCALE, so set_scale(nominal) snaps the residue back exactly.
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

# Support both import styles:
#  - top-level (`from attention import ...` after adding blocks/ to sys.path,
#    used by the per-block regression tests)
#  - package-qualified (`from blocks.attention import ...`, used by headlines)
try:
    from blocks.linear import inner_sum_required_steps, replicate_required_steps
    from blocks.softmax import softmax_damping_schedule, softmax_required_steps
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from linear import inner_sum_required_steps, replicate_required_steps
    from softmax import softmax_damping_schedule, softmax_required_steps


# DIAGNOSTIC ONLY (opt-in via PROBE_DECRYPT_STAGES=1). Dumps a decrypted
# slot vector to disk so an offline harness can compute rel-RMS vs plain-math
# for the softmax internals (denominator `a`, broadcast sum). When the flag is
# unset _probe_dump_stage is a no-op: byte-identical to the original.
_PROBE_DECRYPT_STAGES = os.environ.get("PROBE_DECRYPT_STAGES") == "1"
_PROBE_DUMP_DIR = os.environ.get("PROBE_DUMP_DIR", "/tmp/probe_stage_dump")
_PROBE_DUMP_LAYER = [None]  # set by llama3_mrpc per verbose layer


def _probe_dump_stage(tag, v):
    if not (_PROBE_DECRYPT_STAGES and _PROBE_DUMP_LAYER[0] is not None):
        return
    os.makedirs(_PROBE_DUMP_DIR, exist_ok=True)
    safe = (tag.replace("/", "_").replace(" ", "_")
            .replace("[", "").replace("]", "").replace("(", "").replace(")", ""))
    np.save(f"{_PROBE_DUMP_DIR}/L{_PROBE_DUMP_LAYER[0]}__smx_{safe}.npy",
            np.asarray(v, dtype=np.float64))


# ---------------------------------------------------------------------------
# Shape / step helpers
# ---------------------------------------------------------------------------

def _is_pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def qkt_required_steps(d_head: int):
    """Galois steps for QK^T inner-sum: powers of 2 in [1, d_head)."""
    return inner_sum_required_steps(d_head)


def score_v_required_steps(d_head: int, d_total: int, positions_per_ct: int):
    """Steps for score_times_v: in-block broadcast (negative) + cross-position
    accumulation (positive)."""
    if not _is_pow2(d_head):
        raise ValueError("score_v_required_steps: d_head must be a power of 2")
    if not _is_pow2(positions_per_ct):
        raise ValueError("score_v_required_steps: positions_per_ct must be a power of 2")
    steps = []
    # Broadcast within d_head blocks: negative strides d_head/2, d_head/4, ..., 1.
    bstride = d_head // 2
    while bstride >= 1:
        steps.append(-int(bstride))
        if bstride == 1:
            break
        bstride >>= 1
    # Accumulate across packed positions: d_total, 2*d_total, ..., (positions_per_ct/2)*d_total.
    max_accumulate = positions_per_ct * d_total
    astride = d_total
    while astride < max_accumulate:
        steps.append(int(astride))
        astride <<= 1
    return steps


def broadcast_required_steps(block_size: int):
    """Steps for broadcast_within_blocks: -block_size/2, ..., -2, -1."""
    if not _is_pow2(block_size):
        raise ValueError("broadcast_required_steps: block_size must be a power of 2")
    steps = []
    bstride = block_size // 2
    while bstride >= 1:
        steps.append(-int(bstride))
        if bstride == 1:
            break
        bstride >>= 1
    return steps


def sdpa_required_steps(d_head: int, d_total: int, num_tokens: int, slot_count: int):
    """Combined Galois steps for full SDPA: QK^T | softmax | score*V."""
    steps = []
    steps.extend(qkt_required_steps(d_head))
    # Softmax sum_reduce uses cyclic-wrap count = slot_count/d_total.
    steps.extend(softmax_required_steps(slot_count // d_total, d_total))
    steps.extend(score_v_required_steps(d_head, d_total, num_tokens))
    steps = sorted(set(int(s) for s in steps))
    return steps

