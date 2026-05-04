"""SiLU via degree-8 Chebyshev polynomial evaluation."""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

# Degree-8 Chebyshev fit of SiLU on [-2, 2]. L_inf err 2.74e-5 over [-2, 2].
SILU_COEFFS_DEG5_R2 = [
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


def silu(ctx, encoder, relin_key, ct):
    """Evaluate SiLU on ct using the degree-8 Chebyshev approximation."""
    return phantom.eval_polynomial(ctx, encoder, relin_key, ct, SILU_COEFFS_DEG5_R2)
