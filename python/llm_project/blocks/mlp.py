"""MLP orchestration helpers (pure Python / numpy).

CUDA primitives: phantom.mlp_forward, phantom.mlp_forward_complex.
This module: required_steps, weight setup, forward wrappers, numpy reference.
"""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")

import numpy as np
import pyPhantom as phantom


def mlp_required_steps(baby_steps):
    """Galois rotation steps for real MLP (delegates to phantom.bsgs_required_steps)."""
    return phantom.bsgs_required_steps(baby_steps)


def mlp_complex_required_steps(baby_steps):
    """Galois steps for complex-folded MLP: BSGS steps + step 0 (conjugation)."""
    steps = phantom.bsgs_required_steps(baby_steps)
    steps.append(0)
    return steps


def setup_mlp_weights(ctx, encoder, w_gate_flat, w_up_flat, w_down_flat,
                      d_model, d_hidden, d_pad, baby_steps, scale):
    """Encode three MLP weight matrices (gate, up, down) into BSGS diagonals.

    w_gate, w_up: row-major (d_hidden, d_model).
    w_down:       row-major (d_model, d_hidden).
    Returns a phantom.mlp_weights struct.
    """
    if len(w_gate_flat) != d_hidden * d_model:
        raise ValueError("setup_mlp_weights: w_gate size != d_hidden * d_model")
    if len(w_up_flat) != d_hidden * d_model:
        raise ValueError("setup_mlp_weights: w_up size != d_hidden * d_model")
    if len(w_down_flat) != d_model * d_hidden:
        raise ValueError("setup_mlp_weights: w_down size != d_model * d_hidden")
    w_gate = phantom.pre_encode_bsgs_diagonals(
        ctx, encoder, list(w_gate_flat),
        d_hidden, d_model, d_pad, baby_steps, scale)
    w_up = phantom.pre_encode_bsgs_diagonals(
        ctx, encoder, list(w_up_flat),
        d_hidden, d_model, d_pad, baby_steps, scale)
    w_down = phantom.pre_encode_bsgs_diagonals(
        ctx, encoder, list(w_down_flat),
        d_model, d_hidden, d_pad, baby_steps, scale)
    return phantom.mlp_weights(w_gate, w_up, w_down)


def setup_mlp_weights_complex(ctx, encoder, w_gate_flat, w_up_flat, w_down_flat,
                               d_model, d_hidden, d_pad, baby_steps, scale):
    """Encode three MLP weight matrices into complex-folded BSGS diagonals.

    Fold modes: w_gate/w_up use Rows, w_down uses ColsConj.
    d_hidden must be even. Returns a phantom.mlp_weights_complex struct.
    """
    if len(w_gate_flat) != d_hidden * d_model:
        raise ValueError("setup_mlp_weights_complex: w_gate size != d_hidden * d_model")
    if len(w_up_flat) != d_hidden * d_model:
        raise ValueError("setup_mlp_weights_complex: w_up size != d_hidden * d_model")
    if len(w_down_flat) != d_model * d_hidden:
        raise ValueError("setup_mlp_weights_complex: w_down size != d_model * d_hidden")
    if (d_hidden % 2) != 0:
        raise ValueError("setup_mlp_weights_complex: d_hidden must be even")
    w_gate = phantom.pre_encode_bsgs_diagonals_complex(
        ctx, encoder, list(w_gate_flat),
        d_hidden, d_model, d_pad, baby_steps, scale,
        phantom.complex_fold_mode.Rows)
    w_up = phantom.pre_encode_bsgs_diagonals_complex(
        ctx, encoder, list(w_up_flat),
        d_hidden, d_model, d_pad, baby_steps, scale,
        phantom.complex_fold_mode.Rows)
    w_down = phantom.pre_encode_bsgs_diagonals_complex(
        ctx, encoder, list(w_down_flat),
        d_model, d_hidden, d_pad, baby_steps, scale,
        phantom.complex_fold_mode.ColsConj)
    return phantom.mlp_weights_complex(
        w_gate, w_up, w_down, d_model, d_hidden, d_pad)


def mlp_forward(ctx, encoder, relin_key, galois_key, ct, weights):
    """Real-valued SwiGLU MLP forward (delegates to phantom.mlp_forward)."""
    return phantom.mlp_forward(ctx, encoder, relin_key, galois_key, ct, weights)


def mlp_forward_complex(ctx, encoder, relin_key, galois_key, ct, weights):
    """Complex-folded SwiGLU MLP forward (delegates to phantom.mlp_forward_complex)."""
    return phantom.mlp_forward_complex(ctx, encoder, relin_key, galois_key, ct, weights)


def reference_mlp_forward(x, w_gate, w_up, w_down):
    """Numpy SwiGLU reference: y = W_down @ (silu(W_gate @ x) * (W_up @ x))."""
    x = np.asarray(x, dtype=np.float64)
    w_gate = np.asarray(w_gate, dtype=np.float64)
    w_up = np.asarray(w_up, dtype=np.float64)
    w_down = np.asarray(w_down, dtype=np.float64)

    gate = w_gate @ x          # (d_hidden,)
    up = w_up @ x              # (d_hidden,)
    silu_gate = gate / (1.0 + np.exp(-gate))
    h = silu_gate * up         # (d_hidden,)
    y = w_down @ h             # (d_model,)
    return y
