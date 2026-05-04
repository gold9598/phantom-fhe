"""
Softmax orchestration (pure Python / numpy).

C++ primitives: phantom.ps_exp_init, phantom.square_iterations_inplace,
phantom.square_iterations_damped_inplace, phantom.finalize_softmax.
"""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom
import numpy as np


def softmax_damping_schedule(num_squarings, num_tokens, extra_scale, target_mag):
    """Per-step damping factors so intermediate magnitudes stay near target_mag.

    Returns a list of length num_squarings.
    """
    damps = [1.0] * num_squarings
    if num_squarings == 0:
        return damps
    t = float(num_tokens)
    scale_exp = float(2 ** num_squarings)
    t_factor = t ** (-1.0 / scale_exp)
    f = extra_scale * t_factor
    for i in range(num_squarings):
        f_sq = f * f
        d = target_mag / f_sq
        damps[i] = d
        f = f_sq * d  # = target_mag by construction
    return damps


def sum_reduce_stride(ctx, gk, ct, stride, count):
    """Sum-reduce ct over `count` positions at the given stride (power of 2).

    Returns a ciphertext with the sum broadcast into every stride-th slot.
    """
    if count == 0 or (count & (count - 1)) != 0:
        raise ValueError("sum_reduce_stride: count must be a power of two and > 0")
    if stride == 0:
        raise ValueError("sum_reduce_stride: stride must be > 0")

    acc = ct
    step = stride
    reach = 1
    while reach < count:
        rotated = phantom.rotate(ctx, acc, int(step), gk)
        acc = phantom.add(ctx, acc, rotated)
        step <<= 1
        reach <<= 1
    return acc


def softmax_required_steps(num_tokens, stride):
    """Galois rotation steps needed by sum_reduce_stride."""
    steps = []
    if num_tokens < 2 or stride == 0:
        return steps
    step = stride
    reach = 1
    while reach < num_tokens:
        steps.append(int(step))
        step <<= 1
        reach <<= 1
    return steps


def softmax_forward(ctx, encoder, relin_key, galois_key, scores_ct,
                    num_tokens, num_squarings, extra_scale, target_mag,
                    iters, reduce_count, stride):
    """ps_exp_init -> damped squarings -> finalize_softmax.

    Caller is responsible for masking non-meaningful slots before calling this.
    """
    damps = softmax_damping_schedule(num_squarings, num_tokens, extra_scale, target_mag)
    e_ct = phantom.ps_exp_init(
        ctx, encoder, relin_key, scores_ct,
        num_tokens, num_squarings, extra_scale)
    phantom.square_iterations_damped_inplace(ctx, encoder, relin_key, e_ct, damps)
    return phantom.finalize_softmax(
        ctx, encoder, relin_key, galois_key, e_ct,
        reduce_count, stride, iters)


def reference_softmax(scores):
    """Numerically-stable max-shifted softmax (numpy reference)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s.copy()
    s_shifted = s - s.max()
    exps = np.exp(s_shifted)
    return exps / exps.sum()
