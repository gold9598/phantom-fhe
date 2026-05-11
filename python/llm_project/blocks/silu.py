"""SiLU via Chebyshev polynomial evaluation."""

import sys
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def fit_silu_coeffs(domain, deg=14, n_samples=4001, normalized=False):
    """Fit a degree-`deg` Chebyshev polynomial to silu on the given symmetric
    domain (-d, d) and return monomial coefficients suitable for
    phantom.eval_polynomial. Cost: ~5ms per fit.

    When `normalized=False` (default), coefficients are for the polynomial
    in x (the input ciphertext value). For wide domains (D > ~6), the
    high-order coefficients become smaller than CKKS encoding precision
    (1/SCALE ~ 9e-13) and silently quantize to zero, leaving an under-fit
    polynomial.

    When `normalized=True`, coefficients are for the polynomial in
    z = x/D, fit on [-1, 1]. Equivalent to the same function but with
    coefficients c'_i = c_i * D^i — large enough to survive CKKS encoding
    even at high degree on wide domain. Caller must scale ct by 1/D before
    polynomial evaluation (one extra ct·scalar multiplication, 1 level).
    """
    x = np.linspace(domain[0], domain[1], n_samples)
    y = _silu(x)
    cheb = Chebyshev.fit(x, y, deg, domain=domain)
    if normalized:
        # Coefficients for the polynomial in z = x/D, fit on [-1, 1].
        mono = cheb.convert(kind=Polynomial, domain=domain, window=(-1.0, 1.0))
    else:
        mono = cheb.convert(kind=Polynomial, domain=domain, window=domain)
    return mono.coef.tolist()

# Degree-8 Chebyshev fit of SiLU on [-2, 2]. L_inf err 2.74e-5 over [-2, 2].
# Calibrated for layer-0 gate magnitudes; saturates badly outside [-2, 2]
# (true silu(3.7)=3.6, this poly produces ~2.6 — 1.0 absolute error). Kept
# for reference / regression comparison only.
SILU_COEFFS_DEG8_R2 = [
    -4.596475110252296e-16,  # x^0
     0.4999999999999997,     # x^1
     0.24991346416127744,    # x^2
     4.3792130415770066e-16, # x^3
    -0.020536112891251988,   # x^4
    -2.1433472280957883e-16, # x^5
     0.0017936119840762485,  # x^6
     2.929755203871941e-17,  # x^7
    -9.492355514915103e-05,  # x^8
]

# Degree-16 Chebyshev fit of SiLU on [-6, 6]. L_inf err 6.78e-4 over [-6, 6].
# +1 level of depth vs deg-8 (5 levels instead of 4 via PS), but ~2.6x tighter
# precision than deg-14 on the same domain.
SILU_COEFFS_DEG16_R6 = [
    0.00028658111573443,      # x^0
    0.49999999999999933,      # x^1
    0.24844928466555605,      # x^2
    7.644143092088274e-17,    # x^3
   -0.019391892402283603,     # x^4
   -4.6676421966433063e-17,   # x^5
    0.0015295996373917546,    # x^6
    4.09757260808143e-18,     # x^7
   -9.095305360868811e-05,    # x^8
    4.028399289822532e-20,    # x^9
    3.680959289224436e-06,    # x^10
   -1.491255940355426e-20,    # x^11
   -9.369417995491622e-08,    # x^12
    5.240508411876882e-22,    # x^13
    1.3398620738119484e-09,   # x^14
   -5.626637476201116e-24,    # x^15
   -8.169035409133704e-12,    # x^16
]

# Degree-14 Chebyshev fit of SiLU on [-6, 6]. L_inf err 1.76e-3 over [-6, 6].
# Calibrated for the 32-layer LLaMA-3.1-8B prefill on MRPC: max|gate| across
# all layers is 5.88 (layer 28). Same depth (4 levels via PS) as deg 8.
SILU_COEFFS_DEG14_R6 = [
    0.0007819802969966094,    # x^0
    0.49999999999999745,      # x^1
    0.24657870157282422,      # x^2
    1.156633155728448e-15,    # x^3
   -0.018240667511804365,     # x^4
   -1.865281303653367e-16,    # x^5
    0.001261113296332196,     # x^6
    1.3561479485191877e-17,   # x^7
   -6.0337317303621734e-05,   # x^8
   -5.0781152197672e-19,      # x^9
    1.7920336481251102e-06,   # x^10
    9.819529239383292e-21,    # x^11
   -2.9330839452341827e-08,   # x^12
   -7.87785135946346e-23,     # x^13
    2.009039091924252e-10,    # x^14
]


def fit_silu_chebyshev_basis(domain, deg, n_samples=4001):
    """Return Chebyshev BASIS coefficients t_0..t_N of silu on the given
    symmetric domain. Used by silu_clenshaw which evaluates Σ t_k T_k(z),
    z = x/D, in Clenshaw recurrence — bounded intermediates ≤ max|t_k|,
    so per-mul CKKS noise stays small even at high degree (vs monomial PS
    where intermediates scale with c_top ~ silu_max·D^N).
    """
    x = np.linspace(domain[0], domain[1], n_samples)
    y = _silu(x)
    cheb = Chebyshev.fit(x, y, deg, domain=domain)
    return cheb.coef.tolist()


def silu_clenshaw(engine, ctx, encoder, relin_key, ct, D, t_coeffs, slot_count,
                   ul_max=10, max_abs_intermediate=None):
    """Evaluate silu(x) on encrypted ct via Chebyshev Clenshaw recurrence.

    silu(x) ≈ Σ_{k=0}^{N} t_k · T_k(z), z = x/D ∈ [-1, 1].
    Clenshaw recurrence: b_{N+1}=b_{N+2}=0; b_k = 2z·b_{k+1} - b_{k+2} + t_k;
    silu(x) = t_0 + z·b_1 - b_2.

    Each iteration: 1 ct·ct mul (1 chain level). Mid-flow bootstrap when
    user_level(b_curr) ≥ ul_max so the recurrence fits in NSL-1 chain budget.

    Args:
      D: half-width of silu domain (caller's silu_max·1.2).
      t_coeffs: Chebyshev basis coefficients of silu on [-D, D], from
                fit_silu_chebyshev_basis.
      ul_max: user_level threshold for bootstrap.
      max_abs_intermediate: bound on Chebyshev intermediate magnitudes
                            for bootstrap_safe scaling. Defaults to max|t_k|.
    """
    import sys as _sys
    _sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
    from blocks.bootstrap import bootstrap_safe

    N = len(t_coeffs) - 1
    if N < 2:
        raise ValueError("silu_clenshaw: degree must be >= 2")
    user_scale = ct.scale()
    if max_abs_intermediate is None:
        max_abs_intermediate = float(np.abs(t_coeffs).max() * 2.0)

    # z = x / D
    norm_pt = encoder.encode_double_vector(
        ctx, [1.0 / D] * slot_count, user_scale, ct.chain_index())
    ct_z = phantom.multiply_plain(ctx, ct, norm_pt)
    ct_z = phantom.rescale_to_next(ctx, ct_z)
    ct_z.set_scale(user_scale)

    # Init: b_{N-1} = 2 t_N · z + t_{N-1}, b_prev_value = t_N (scalar)
    sc_pt = encoder.encode_double_vector(
        ctx, [2.0 * t_coeffs[N]] * slot_count, ct_z.scale(), ct_z.chain_index())
    b_curr = phantom.multiply_plain(ctx, ct_z, sc_pt)
    b_curr = phantom.rescale_to_next(ctx, b_curr)
    b_curr.set_scale(user_scale)
    tp_pt = encoder.encode_double_vector(
        ctx, [t_coeffs[N - 1]] * slot_count, b_curr.scale(), b_curr.chain_index())
    b_curr = phantom.add_plain(ctx, b_curr, tp_pt)
    b_prev = t_coeffs[N]  # scalar

    def _maybe_bootstrap(b_curr, b_prev, ct_z):
        if engine.user_level(b_curr) >= ul_max:
            b_curr = bootstrap_safe(engine, ctx, encoder, b_curr,
                                      max_abs=max_abs_intermediate, slot_count=slot_count)
            if not isinstance(b_prev, (int, float)) and \
               b_prev.chain_index() > b_curr.chain_index():
                b_prev = bootstrap_safe(engine, ctx, encoder, b_prev,
                                          max_abs=max_abs_intermediate, slot_count=slot_count)
            # ct_z needs to be at chain_index >= b_curr.chain_index() for mod_switch.
            # bootstrap_safe ct_z to fresh chain (max_abs<=1 so no scale needed)
            if ct_z.chain_index() > b_curr.chain_index():
                ct_z = bootstrap_safe(engine, ctx, encoder, ct_z,
                                       max_abs=1.0, slot_count=slot_count)
        return b_curr, b_prev, ct_z

    for k in range(N - 2, 0, -1):
        b_curr, b_prev, ct_z = _maybe_bootstrap(b_curr, b_prev, ct_z)
        # b_k = 2z · b_{k+1} - b_{k+2} + t_k
        z_aligned = phantom.mod_switch_to(ctx, ct_z, b_curr.chain_index())
        z_aligned.set_scale(b_curr.scale())
        m = phantom.multiply_and_relin(ctx, z_aligned, b_curr, relin_key)
        m = phantom.rescale_to_next(ctx, m)
        m.set_scale(user_scale)
        m = phantom.add(ctx, m, m)  # ·2 (cheap doubling)
        if isinstance(b_prev, (int, float)):
            sub_pt = encoder.encode_double_vector(
                ctx, [b_prev] * slot_count, m.scale(), m.chain_index())
            b_new = phantom.sub_plain(ctx, m, sub_pt)
        else:
            b_prev_a = phantom.mod_switch_to(ctx, b_prev, m.chain_index())
            b_prev_a.set_scale(m.scale())
            b_new = phantom.sub(ctx, m, b_prev_a)
        tk_pt = encoder.encode_double_vector(
            ctx, [t_coeffs[k]] * slot_count, b_new.scale(), b_new.chain_index())
        b_new = phantom.add_plain(ctx, b_new, tk_pt)
        b_prev = b_curr
        b_curr = b_new

    b_curr, b_prev, ct_z = _maybe_bootstrap(b_curr, b_prev, ct_z)
    # silu(x) = t_0 + z · b_1 - b_2
    z_aligned = phantom.mod_switch_to(ctx, ct_z, b_curr.chain_index())
    z_aligned.set_scale(b_curr.scale())
    zb1 = phantom.multiply_and_relin(ctx, z_aligned, b_curr, relin_key)
    zb1 = phantom.rescale_to_next(ctx, zb1)
    zb1.set_scale(user_scale)
    if isinstance(b_prev, (int, float)):
        sub_pt = encoder.encode_double_vector(
            ctx, [b_prev] * slot_count, zb1.scale(), zb1.chain_index())
        result = phantom.sub_plain(ctx, zb1, sub_pt)
    else:
        b2_a = phantom.mod_switch_to(ctx, b_prev, zb1.chain_index())
        b2_a.set_scale(zb1.scale())
        result = phantom.sub(ctx, zb1, b2_a)
    t0_pt = encoder.encode_double_vector(
        ctx, [t_coeffs[0]] * slot_count, result.scale(), result.chain_index())
    result = phantom.add_plain(ctx, result, t0_pt)
    return result


def silu(ctx, encoder, relin_key, ct, coeffs=None, norm_factor=None,
         slot_count=None):
    """Evaluate SiLU on ct using a Chebyshev polynomial. If `coeffs` is None
    use the default degree-14 fit on [-6, 6] (covers max|gate|≈5.88 observed
    across the first 31 LLaMA-3.1-8B layers; layer 31's gate reaches 12.7,
    so the caller should pass a wider per-layer fit via fit_silu_coeffs).
    Coefficients must be in monomial order, length determines the polynomial
    degree.

    When `norm_factor` is set, ct is pre-scaled by `norm_factor` (1 level
    cost) so the polynomial evaluates on z = norm_factor·x ∈ [-1, 1]. This
    pairs with `fit_silu_coeffs(..., normalized=True)` to enable high-degree
    fits on wide domains where un-normalized coefficients would underflow
    CKKS encoding precision. `slot_count` must be passed when `norm_factor`
    is set (used to encode the scalar plaintext)."""
    if norm_factor is not None:
        if slot_count is None:
            raise ValueError("silu: slot_count required when norm_factor is set")
        user_scale = ct.scale()
        norm_pt = encoder.encode_double_vector(
            ctx, [float(norm_factor)] * slot_count,
            ct.scale(), ct.chain_index())
        ct = phantom.multiply_plain(ctx, ct, norm_pt)
        ct = phantom.rescale_to_next(ctx, ct)
        ct.set_scale(user_scale)
    return phantom.eval_polynomial(ctx, encoder, relin_key, ct,
                                    coeffs if coeffs is not None else SILU_COEFFS_DEG14_R6)
