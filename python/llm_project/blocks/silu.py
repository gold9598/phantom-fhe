"""SiLU via Chebyshev polynomial evaluation."""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

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


def silu(ctx, encoder, relin_key, ct):
    """Evaluate SiLU on ct using the Chebyshev approximation. Default uses
    the degree-14 fit on [-6, 6]: covers the gate magnitudes observed across
    all 32 LLaMA-3.1-8B decoder layers (max|gate|=5.88), and stays at the
    same depth (4 levels via PS) as the original degree-8 polynomial — the
    higher-degree variants below cost +1 level each, which pushes the
    downstream swiglu past the per-block level budget and regresses
    accuracy at deeper layers."""
    return phantom.eval_polynomial(ctx, encoder, relin_key, ct, SILU_COEFFS_DEG14_R6)
