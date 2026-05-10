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
