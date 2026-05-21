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
                   ul_max=10, max_abs_intermediate=None, galois_key=None):
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
            # When b_curr bootstraps it drops to fresh chain; b_prev (if a ct)
            # is at b_curr.chain - 1 typically, which is then too deep to
            # mod_switch forward to b_curr's post-boot+1 chain in the next
            # multiply. So always bootstrap b_prev too when b_curr bootstraps.
            need_b_prev = not isinstance(b_prev, (int, float))
            if need_b_prev and galois_key is not None:
                # Merge b_curr and b_prev into one bootstrap (saves ~163ms).
                from blocks.bootstrap import merge_bootstrap as _mb
                b_curr, b_prev = _mb(engine, ctx, encoder, b_curr, b_prev,
                                       max_abs=max_abs_intermediate,
                                       slot_count=slot_count, galois_key=galois_key)
            else:
                b_curr = bootstrap_safe(engine, ctx, encoder, b_curr,
                                          max_abs=max_abs_intermediate, slot_count=slot_count)
                if need_b_prev:
                    b_prev = bootstrap_safe(engine, ctx, encoder, b_prev,
                                              max_abs=max_abs_intermediate, slot_count=slot_count)
            # ct_z is at the deepest chain (carried through all iters). Bootstrap
            # it too if it now sits past b_curr's fresh chain.
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


def _cheb_split_coeffs(c, g):
    """Split a Chebyshev series c (numpy 1D, degree N) about the giant index g
    (1 <= g <= N) into (c_lo, c_hi) with deg(c_lo) < g, deg(c_hi) = N - g, so

        sum_k c_k T_k(z) = sum_j c_lo[j] T_j(z) + T_g(z) * sum_j c_hi[j] T_j(z)

    using the Chebyshev product identity T_{g+j} = 2 T_g T_j - T_{|g-j|}
    (and T_g = T_g * T_0 for j == 0). Pure plaintext bookkeeping — this is the
    intricate part validated to < 1e-10 against numpy.polynomial.Chebyshev.
    """
    N = len(c) - 1
    c_lo = np.zeros(g, dtype=np.float64)            # T_0 .. T_{g-1}
    c_hi = np.zeros(N - g + 1, dtype=np.float64)    # T_0 .. T_{N-g}
    for k in range(len(c)):
        if k < g:
            c_lo[k] += c[k]
        elif k == g:
            c_hi[0] += c[k]                          # c_g T_g  -> p_hi T_0
        else:
            j = k - g                               # >= 1
            c_hi[j] += 2.0 * c[k]                    # 2 c_k  (times T_g)
            c_lo[abs(g - j)] += -c[k]                # - c_k
    return c_lo, c_hi


def silu_cheb_bsgs(engine, ctx, encoder, relin_key, ct, D, t_coeffs, slot_count,
                    galois_key=None, max_abs_intermediate=None, m=8):
    """Evaluate silu(x) on encrypted ct via Chebyshev-basis Paterson-Stockmeyer
    (baby-step / giant-step), the depth-efficient alternative to silu_clenshaw.

    silu(x) ~= sum_{k=0}^{N} t_k T_k(z),  z = x/D in [-1, 1],  N = len(t_coeffs)-1.

    Structure (Bossuat et al. / Lattigo-style):
      * Baby steps: build T_0(z)..T_m(z) as ciphertexts via the Chebyshev
        recurrences T_{2k}=2 T_k^2-1, T_{2k+1}=2 T_k T_{k+1}-z.
      * Giant steps: T_{2m}, T_{4m}, ... up to the largest giant g <= N via
        T_{2g}=2 T_g^2-1.
      * Recursive split: p = p_lo + T_g * p_hi, with the c_k folded in
        plaintext (_cheb_split_coeffs). Base case deg < m -> direct
        sum c_k T_k (scalar multiply_plain + adds).

    For N=32, m=8: baby T_0..T_8, giants T_16, T_32. Giant-step depth ~6,
    ~11-15 ct.ct mults total — vs Clenshaw's ~30 mults / ~14 levels for the
    same bounded-coefficient accuracy. NO internal bootstrap needed at NSL=16.

    Args:
      D: half-width of the silu domain (caller's silu_max * 1.2).
      t_coeffs: Chebyshev BASIS coefficients of silu on [-D, D], from
                fit_silu_chebyshev_basis. Bounded ~O(silu_max), so per-mul
                CKKS noise stays small even at degree 32.
      m: baby-step bound (power of 2). Default 8 (good for N=32).
      max_abs_intermediate: only used if a safety bootstrap is ever needed.
      galois_key: unused here (kept for signature parity with silu_clenshaw).
    """
    import sys as _sys
    _sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")

    c = np.asarray(t_coeffs, dtype=np.float64)
    N = len(c) - 1
    if N < 2:
        raise ValueError("silu_cheb_bsgs: degree must be >= 2")
    # m must be a power of 2 and <= N for the giant doubling to land on N's
    # binary-aligned giants. Clamp to the largest power of 2 <= N if needed.
    while m > N:
        m //= 2
    user_scale = ct.scale()
    if max_abs_intermediate is None:
        max_abs_intermediate = float(np.abs(c).max() * 2.0)

    def _ctct(a, b):
        """One ct.ct multiply with chain/scale alignment + rescale. Returns
        a ciphertext at min(level)+1, scale == user_scale."""
        ca, cb = a.chain_index(), b.chain_index()
        if ca < cb:
            a = phantom.mod_switch_to(ctx, a, cb)
        elif cb < ca:
            b = phantom.mod_switch_to(ctx, b, ca)
        a.set_scale(user_scale)
        b.set_scale(user_scale)
        prod = phantom.multiply_and_relin(ctx, a, b, relin_key)
        prod = phantom.rescale_to_next(ctx, prod)
        prod.set_scale(user_scale)
        return prod

    def _sub_const(ct_in, val):
        """ct_in - val (scalar), level-free via sub_plain at ct_in's chain."""
        pt = encoder.encode_double_vector(
            ctx, [float(val)] * slot_count, ct_in.scale(), ct_in.chain_index())
        return phantom.sub_plain(ctx, ct_in, pt)

    def _sub_ct(a, b):
        """a - b with chain/scale alignment (deeper one wins, no extra level)."""
        ca, cb = a.chain_index(), b.chain_index()
        if ca < cb:
            a = phantom.mod_switch_to(ctx, a, cb)
        elif cb < ca:
            b = phantom.mod_switch_to(ctx, b, ca)
        a.set_scale(user_scale)
        b.set_scale(user_scale)
        return phantom.sub(ctx, a, b)

    # ---- z = x / D  (1 level) ----
    norm_pt = encoder.encode_double_vector(
        ctx, [1.0 / D] * slot_count, user_scale, ct.chain_index())
    z = phantom.multiply_plain(ctx, ct, norm_pt)
    z = phantom.rescale_to_next(ctx, z)
    z.set_scale(user_scale)

    # ---- Baby steps: T_0 .. T_m  (T_0 stays a scalar constant, never a ct) ----
    # T[k] holds the ciphertext for T_k(z); T[0] is represented implicitly (1).
    T = {1: z}
    # T_2 = 2 z^2 - 1
    if m >= 2:
        T[2] = _sub_const(phantom.add(ctx, (s := _ctct(T[1], T[1])), s), 1.0)
    for k in range(2, m + 1):
        if k in T:
            continue
        if k % 2 == 0:
            half = k // 2
            sq = _ctct(T[half], T[half])
            T[k] = _sub_const(phantom.add(ctx, sq, sq), 1.0)  # 2 T_h^2 - 1
        else:
            lo, hi = (k - 1) // 2, (k + 1) // 2
            pr = _ctct(T[lo], T[hi])
            dbl = phantom.add(ctx, pr, pr)                    # 2 T_lo T_hi
            T[k] = _sub_ct(dbl, z)                            # - T_1 = - z

    # ---- Giant steps: g = 2m, 4m, ... <= N  (T_{2g} = 2 T_g^2 - 1) ----
    g = m
    while g * 2 <= N:
        g2 = g * 2
        sq = _ctct(T[g], T[g])
        T[g2] = _sub_const(phantom.add(ctx, sq, sq), 1.0)
        g = g2
    # `g` is now the largest giant <= N (the top-level split index).

    def _add_ct(a, b):
        """a + b with chain/scale alignment (deeper one wins, no extra level)."""
        ca, cb = a.chain_index(), b.chain_index()
        if ca < cb:
            a = phantom.mod_switch_to(ctx, a, cb)
        elif cb < ca:
            b = phantom.mod_switch_to(ctx, b, ca)
        a.set_scale(user_scale)
        b.set_scale(user_scale)
        return phantom.add(ctx, a, b)

    # ---- Recursive combination p = p_lo + T_g * p_hi ----
    def _eval(coeffs):
        n = len(coeffs) - 1
        if n < m:
            # Base case: direct sum_{k=0}^{n} coeffs[k] T_k(z).
            # T_0 contributes the constant coeffs[0] (folded via add_plain at
            # the end). Each T_k (k>=1) is a ct.scalar multiply (1 level), all
            # at the baby-step chains; accumulate, then add the constant.
            acc = None
            for k in range(1, n + 1):
                ck = float(coeffs[k])
                if ck == 0.0:
                    continue
                tk = T[k]
                ck_pt = encoder.encode_double_vector(
                    ctx, [ck] * slot_count, tk.scale(), tk.chain_index())
                term = phantom.multiply_plain(ctx, tk, ck_pt)
                term = phantom.rescale_to_next(ctx, term)
                term.set_scale(user_scale)
                acc = term if acc is None else _add_ct(acc, term)
            const = float(coeffs[0])
            if acc is None:
                # Constant-only sub-series (no T_>=1 term). Lift the constant
                # onto T_1's chain via 0*T_1 + const so it stays a ciphertext
                # the caller can combine. Rare for silu fits; guard anyway.
                z_pt = encoder.encode_double_vector(
                    ctx, [0.0] * slot_count, T[1].scale(), T[1].chain_index())
                acc = phantom.multiply_plain(ctx, T[1], z_pt)
                acc = phantom.rescale_to_next(ctx, acc)
                acc.set_scale(user_scale)
            const_pt = encoder.encode_double_vector(
                ctx, [const] * slot_count, acc.scale(), acc.chain_index())
            return phantom.add_plain(ctx, acc, const_pt)
        # Split about the largest giant gg <= n.
        gg = m
        while gg * 2 <= n:
            gg *= 2
        c_lo, c_hi = _cheb_split_coeffs(coeffs, gg)
        lo = _eval(c_lo)
        hi = _eval(c_hi)
        prod = _ctct(T[gg], hi)          # T_gg * p_hi  (1 level)
        return _add_ct(lo, prod)

    return _eval(c)


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
