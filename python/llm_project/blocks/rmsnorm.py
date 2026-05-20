"""RMSNorm orchestration helpers (pure Python / numpy).

CUDA primitive: phantom.rmsnorm_forward.
This module: chebyshev_monomial_coeffs, rmsnorm_required_steps,
setup_rmsnorm_weights, rmsnorm_forward, reference_rmsnorm.
"""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")

import math
import numpy as np
import pyPhantom as phantom


def chebyshev_monomial_coeffs(z_min, z_max, exponent, degree):
    """Monomial coefficients [c_0, ..., c_degree] of the Chebyshev interpolant
    of z^exponent on [z_min, z_max].

    Algorithm: Chebyshev nodes -> DCT-I coeffs -> T_k to monomial basis ->
    affine substitution t = a*z + b via binomial expansion (matches C++).
    """
    if z_min <= 0.0:
        raise ValueError("chebyshev_monomial_coeffs: z_min must be positive")
    if z_max <= z_min:
        raise ValueError("chebyshev_monomial_coeffs: z_max must exceed z_min")
    if degree < 1:
        raise ValueError("chebyshev_monomial_coeffs: degree must be >= 1")

    n = degree + 1
    half_span = 0.5 * (z_max - z_min)
    midpoint = 0.5 * (z_min + z_max)

    # 1. Chebyshev nodes and function values.
    nodes_t = np.array([math.cos(math.pi * (j + 0.5) / n) for j in range(n)])
    nodes_z = midpoint + half_span * nodes_t
    values = np.power(nodes_z, exponent)

    # 2. Chebyshev coefficients c_k (standard DCT-I formula).
    c_cheb = np.zeros(n)
    for k in range(n):
        s = sum(values[j] * math.cos(k * math.pi * (j + 0.5) / n) for j in range(n))
        c_cheb[k] = s / n if k == 0 else 2.0 * s / n

    # 3. Build monomial representations of T_0, T_1, ..., T_{n-1} in t,
    #    then accumulate into t_coeffs[i] = coefficient of t^i.
    #    T_k stored as list of length k+1.
    t_mono = [[1.0]]  # T_0 = 1
    if n >= 2:
        t_mono.append([0.0, 1.0])  # T_1 = t
    for k in range(2, n):
        prev = t_mono[k - 1]
        prev_prev = t_mono[k - 2]
        curr = [0.0] * (k + 1)
        # 2t * T_{k-1}
        for i, v in enumerate(prev):
            curr[i + 1] += 2.0 * v
        # - T_{k-2}
        for i, v in enumerate(prev_prev):
            curr[i] -= v
        t_mono.append(curr)

    t_coeffs = np.zeros(n)
    for k in range(n):
        for i, v in enumerate(t_mono[k]):
            t_coeffs[i] += c_cheb[k] * v

    # 4. Substitute t = a*z + b via binomial expansion.
    #    t = a*z + b  =>  t^i = sum_{j=0}^{i} C(i,j) * a^j * b^{i-j} * z^j
    a = 2.0 / (z_max - z_min)
    b = -(z_min + z_max) / (z_max - z_min)
    z_coeffs = np.zeros(n)
    for i in range(n):
        if t_coeffs[i] == 0.0:
            continue
        for j in range(i + 1):
            binom = math.comb(i, j)
            term = t_coeffs[i] * binom * (a ** j) * (b ** (i - j))
            z_coeffs[j] += term

    return z_coeffs.tolist()


def rmsnorm_required_steps(d_model):
    """Galois steps for sum_of_squares: [1, 2, 4, ..., d_model/2]."""
    steps = []
    if d_model < 2:
        return steps
    stride = 1
    while stride < d_model:
        steps.append(stride)
        stride <<= 1
    return steps


def rmsnorm_required_steps_stride_t(d_model, t):
    """Galois steps for stride-t sum_of_squares: [t, 2t, 4t, ..., (d_model/2)*t]."""
    steps = []
    if d_model < 2:
        return steps
    stride = t
    reach = 1
    while reach < d_model:
        steps.append(int(stride))
        stride <<= 1
        reach <<= 1
    return steps


def setup_rmsnorm_weights(ctx, encoder, params, g, stride=1):
    """Tile gamma across all slots with 1/sqrt(zgeo) compensator folded in.

    g: list or 1-D array of length params.d_model.
    stride: 1 (default, periodic packing); >1 = stride-t packing where g[k] is
            placed at slot k*stride and other slots are zero.
    Returns a phantom.rmsnorm_weights instance.
    """
    d_model = params.d_model
    if (d_model & (d_model - 1)) != 0 or d_model == 0:
        raise ValueError("setup_rmsnorm_weights: d_model must be a power of 2")
    g_arr = np.asarray(g, dtype=np.float64)
    if g_arr.shape[0] != d_model:
        raise ValueError("setup_rmsnorm_weights: g length must equal d_model")
    zgeo = math.sqrt(params.z_min * params.z_max)
    compensator = 1.0 / math.sqrt(zgeo)
    num_slots = encoder.slot_count()
    g_tiled = np.zeros(num_slots, dtype=np.float64)
    g_scaled = g_arr * compensator
    if stride == 1:
        periods = num_slots // d_model
        for k in range(periods):
            g_tiled[k * d_model : (k + 1) * d_model] = g_scaled
    else:
        if stride * d_model > num_slots:
            raise ValueError(
                f"setup_rmsnorm_weights: stride*d_model ({stride*d_model}) must be <= num_slots ({num_slots})")
        for k in range(d_model):
            g_tiled[k * stride] = g_scaled[k]
    return phantom.rmsnorm_weights(g_tiled.tolist(), g_arr.tolist())


def rmsnorm_forward(ctx, encoder, relin_key, galois_key, ct, weights, params):
    """RMSNorm forward (delegates to phantom.rmsnorm_forward)."""
    return phantom.rmsnorm_forward(ctx, encoder, relin_key, galois_key, ct, weights, params)


def _sum_reduce_stride(ctx, gk, ct, stride, count):
    """Sum-reduce ct over `count` positions at the given stride (power of 2)."""
    if count == 0 or (count & (count - 1)) != 0:
        raise ValueError("_sum_reduce_stride: count must be a power of two and > 0")
    if stride == 0:
        raise ValueError("_sum_reduce_stride: stride must be > 0")
    acc = ct
    step = stride
    reach = 1
    while reach < count:
        rotated = phantom.rotate(ctx, acc, int(step), gk)
        acc = phantom.add(ctx, acc, rotated)
        step <<= 1
        reach <<= 1
    return acc


def rmsnorm_forward_stride_t(ctx, encoder, relin_key, galois_key,
                              x, weights_stride_t, params, t):
    """RMSNorm forward for stride-t input layout (pure Python).

    x: stride-t ciphertext — values placed at slots {0, t, 2t, ...,
       (d_model-1)*t}, zeros elsewhere.
    weights_stride_t: phantom.rmsnorm_weights with g_tiled_real encoded
       stride-t (use setup_rmsnorm_weights(..., stride=t)).
    params: phantom.rmsnorm_params (d_model, epsilon, z_min, z_max,
       poly_degree).
    t: stride (= num_slots // d_model in our setup).

    Returns: stride-t ciphertext at the same chain depth as the C++ kernel
       would produce for the periodic case.
    """
    d_model = params.d_model
    if (d_model & (d_model - 1)) != 0 or d_model == 0:
        raise ValueError("rmsnorm_forward_stride_t: d_model must be a power of 2")
    if t <= 0:
        raise ValueError("rmsnorm_forward_stride_t: t must be > 0")

    nominal = x.scale()

    # Stage 1: sum_of_squares on stride-t layout.
    x_sq = phantom.multiply_and_relin(ctx, x, x, relin_key)
    x_sq = phantom.rescale_to_next(ctx, x_sq)
    x_sq.set_scale(nominal)
    sum_sq = _sum_reduce_stride(ctx, galois_key, x_sq, stride=t, count=d_model)

    # Stage 2: (alpha/d_model) * sum_x^2 + (alpha*epsilon).
    zgeo = math.sqrt(params.z_min * params.z_max)
    alpha = 1.0 / zgeo
    scale_factor = alpha / float(d_model)
    num_slots = encoder.slot_count()
    scale_pt = encoder.encode_double_vector(
        ctx, [scale_factor] * num_slots, nominal, sum_sq.chain_index())
    sum_sq = phantom.multiply_plain(ctx, sum_sq, scale_pt)
    sum_sq = phantom.rescale_to_next(ctx, sum_sq)
    sum_sq.set_scale(nominal)

    scaled_eps = alpha * params.epsilon
    if scaled_eps != 0.0:
        # Match C++: split into pieces of <= 0.1 to keep encoding precise.
        abs_v = abs(scaled_eps)
        pieces = max(1, int(math.ceil(abs_v / 0.1)))
        piece = scaled_eps / float(pieces)
        for _ in range(pieces):
            eps_pt = encoder.encode_double_vector(
                ctx, [piece] * num_slots, sum_sq.scale(), sum_sq.chain_index())
            sum_sq = phantom.add_plain(ctx, sum_sq, eps_pt)

    # Stage 3: 1/sqrt(alpha*z) via Chebyshev PS on prescaled input.
    # Output is sqrt(zgeo) * 1/sqrt(z); g_tiled_real folds in 1/sqrt(zgeo).
    zp_min = params.z_min * alpha
    zp_max = params.z_max * alpha
    coeffs = chebyshev_monomial_coeffs(zp_min, zp_max, -0.5, params.poly_degree)
    w = phantom.eval_polynomial(ctx, encoder, relin_key, sum_sq, coeffs)

    # Stage 4: x * w (mod-switch x to w's chain first).
    x_aligned = phantom.mod_switch_to(ctx, x, w.chain_index())
    xw = phantom.multiply_and_relin(ctx, x_aligned, w, relin_key)
    xw = phantom.rescale_to_next(ctx, xw)
    xw.set_scale(nominal)

    # Stage 5: pt*ct rescale by stride-t g.
    # Model-weight γ encoded as SingleChainPlaintext (chain-agnostic, single
    # polynomial = N*8 bytes) + JIT-expanded to full-RNS PhantomPlaintext at
    # xw's chain via expand_single_chain_to_full. Mathematically identical to
    # the prior encode_double_vector path; opens the door to hoisting the SCP
    # encoding to setup-time so it's not re-encoded on every rmsnorm call.
    g_scp = phantom.encode_single_chain_plaintext(
        ctx, encoder,
        np.asarray(weights_stride_t.g_tiled_real, dtype=np.float64),
        nominal)
    g_pt = phantom.expand_single_chain_to_full(ctx, g_scp, xw.chain_index())
    xw = phantom.multiply_plain(ctx, xw, g_pt)
    xw = phantom.rescale_to_next(ctx, xw)
    xw.set_scale(nominal)

    return xw


def reference_rmsnorm(x, g, d_model, epsilon):
    """Numpy reference: out = x / sqrt(mean(x^2) + eps) * g."""
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if x.shape[0] != d_model:
        raise ValueError("reference_rmsnorm: x length must equal d_model")
    if g.shape[0] != d_model:
        raise ValueError("reference_rmsnorm: g length must equal d_model")
    mean_sq = np.sum(x * x) / d_model
    inv_rms = 1.0 / math.sqrt(mean_sq + epsilon)
    return x * inv_rms * g
