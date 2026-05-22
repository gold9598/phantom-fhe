"""SK-free CKKS bootstrap helpers.

CKKS bootstrap is built around an EvalMod sine polynomial that is only accurate
when slot values lie inside roughly [-0.5, 0.5]. Inputs outside that band wrap
modularly: a value of v becomes v - round(v). Both a non-zero mean and a large
per-slot magnitude break the bootstrap, so callers must scale inputs into the
safe domain before calling `engine.bootstrap_inplace`.

`bootstrap`: given a static caller-provided bound `max_abs`, pre-scales
`ct` by `target_mag / max_abs` into [-target_mag, target_mag], runs
`engine.bootstrap_inplace`, then unscales. Uses `target_mag=0.49` by default
(the best precision without crossing the polynomial's mod-1 wrap point).

`merge_bootstrap`: bootstraps two real ciphertexts as one complex pair — one
`bootstrap_inplace` call instead of two — by packing ct1 (real) and ct2 (imaginary)
into a single complex ciphertext, bootstrapping, then conjugate-splitting to
recover both. Pre- and post-scaling are folded into the pack/unpack multiplies.
"""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom


def bootstrap(engine, ctx, encoder, ct, max_abs, slot_count, target_mag=0.49):
    """SK-free bootstrap with static input range.

    Pre-scales `ct` by `target_mag / max_abs` so post-scale slots fit inside
    bootstrap's safe domain, runs `engine.bootstrap_inplace`, then unscales.
    Caller-provided `max_abs` is the static upper bound on |slot value|;
    the wrapper does not measure the input. If `max_abs <= target_mag`,
    no scaling is applied and the bootstrap runs directly (zero extra levels).

    Inputs are assumed approximately mean-zero. If a call site has a known
    non-zero mean, subtract it as a plaintext before calling this and add
    back after.

    target_mag=0.49: best precision without crossing the polynomial's mod-1
    wrap point (polynomial error is flat from |x|=0.1 to |x|=0.49; bigger
    target_mag shrinks the post-bootstrap scale-up factor linearly).
    """
    user_scale = engine.user_scale()
    needs_scale = max_abs > target_mag
    if needs_scale:
        if engine.user_level(ct) >= engine.max_user_level():
            raise ValueError(
                f"bootstrap: input at user_level {engine.user_level(ct)} "
                f"(== max_user_level {engine.max_user_level()}) requires scaling "
                f"(max_abs={max_abs} > target_mag={target_mag}); "
                "bootstrap one level earlier in the pipeline.")
        scale_down = target_mag / max_abs
        scale_up = 1.0 / scale_down
        scale_pt = encoder.encode_double_vector(
            ctx, [scale_down] * slot_count, ct.scale(), ct.chain_index())
        ct = phantom.multiply_plain(ctx, ct, scale_pt)
        ct = phantom.rescale_to_next(ctx, ct)
        ct.set_scale(user_scale)

    engine.bootstrap_inplace(ct)

    if needs_scale:
        unscale_pt = encoder.encode_double_vector(
            ctx, [scale_up] * slot_count, ct.scale(), ct.chain_index())
        ct = phantom.multiply_plain(ctx, ct, unscale_pt)
        ct = phantom.rescale_to_next(ctx, ct)
        ct.set_scale(user_scale)
    return ct


def _imag_unit_pt_cache():
    """Lazy cache of complex constant single_chain_plaintexts (i, 0.5, -0.5j).
    Keyed by (encoder_id, ctx_id, scale). Avoids re-encoding the constants
    on every merge_bootstrap call (small but adds up across 32 layers)."""
    if not hasattr(_imag_unit_pt_cache, "_c"):
        _imag_unit_pt_cache._c = {}
    return _imag_unit_pt_cache._c


def merge_bootstrap(engine, ctx, encoder, ct1, ct2, max_abs, slot_count,
                     galois_key, target_mag=0.49):
    """Bootstrap two real ciphertexts as one complex pair (one bootstrap_inplace
    instead of two). Saves ~163ms/pair on a 5090.

    Pre-scaling is absorbed into the merge multiplications: ct1 is multiplied
    by `sd` (real scale-down) and ct2 by `sd·i` (complex). Both consume
    exactly the 1 level a bootstrap(ct) would have used for its own
    scale-down. Summing the two scaled cts is free (same chain). The merged
    ct's slot magnitudes are ≤ target_mag, so bootstrap_inplace runs directly
    (no internal scaling). Conjugate, split, and fold the rescale-up
    (0.5/sd, -0.5i/sd) into the split multiplies — same cost as two
    separate post-bootstrap unscales. Net: one bootstrap call saved.

    `max_abs` bounds max(|ct1|, |ct2|); merged magnitude bounded by
    sqrt(2)·max_abs.
    """
    user_scale = engine.user_scale()
    merged_max = max_abs * 1.42
    sd = target_mag / merged_max if merged_max > target_mag else 1.0
    sd_inv = 1.0 / sd

    # Align ct1, ct2 to the same chain.
    target_chain = max(ct1.chain_index(), ct2.chain_index())
    if ct1.chain_index() < target_chain:
        ct1 = phantom.mod_switch_to(ctx, ct1, target_chain)
    if ct2.chain_index() < target_chain:
        ct2 = phantom.mod_switch_to(ctx, ct2, target_chain)
    ct1.set_scale(user_scale); ct2.set_scale(user_scale)

    # ct1 := ct1 · sd  (real plaintext, 1 level)
    if sd != 1.0:
        sd_pt = encoder.encode_double_vector(
            ctx, [sd] * slot_count, user_scale, ct1.chain_index())
        ct1 = phantom.multiply_plain(ctx, ct1, sd_pt)
        ct1 = phantom.rescale_to_next(ctx, ct1)
        ct1.set_scale(user_scale)

    # ct2 := ct2 · (sd · i)  (complex plaintext, 1 level)
    sdi_scpt = phantom.encode_single_chain_plaintext(
        ctx, encoder, [sd * 1j] * slot_count, user_scale)
    sdi_pt = phantom.expand_single_chain_to_full(
        ctx, sdi_scpt, ct2.chain_index())
    ct2 = phantom.multiply_plain(ctx, ct2, sdi_pt)
    ct2 = phantom.rescale_to_next(ctx, ct2)
    ct2.set_scale(user_scale)

    # Align after pre-scales.
    if ct1.chain_index() != ct2.chain_index():
        common = max(ct1.chain_index(), ct2.chain_index())
        if ct1.chain_index() < common:
            ct1 = phantom.mod_switch_to(ctx, ct1, common)
        if ct2.chain_index() < common:
            ct2 = phantom.mod_switch_to(ctx, ct2, common)
        ct1.set_scale(user_scale); ct2.set_scale(user_scale)

    # ct_merged = sd·ct1 + sd·i·ct2  (slots in [-target_mag, target_mag])
    ct_merged = phantom.add(ctx, ct1, ct2)

    # bootstrap_inplace (input already in safe domain).
    engine.bootstrap_inplace(ct_merged)

    # Conjugate (free, key-switch only) and split with rescale-up folded in.
    ct_conj = phantom.rotate(ctx, ct_merged, 0, galois_key)
    half_sd_inv = 0.5 * sd_inv
    half_scpt = phantom.encode_single_chain_plaintext(
        ctx, encoder, [half_sd_inv + 0j] * slot_count, user_scale)
    neg_half_i_sd_inv_scpt = phantom.encode_single_chain_plaintext(
        ctx, encoder, [-0.5j * sd_inv] * slot_count, user_scale)
    half_pt = phantom.expand_single_chain_to_full(
        ctx, half_scpt, ct_merged.chain_index())
    neg_half_i_pt = phantom.expand_single_chain_to_full(
        ctx, neg_half_i_sd_inv_scpt, ct_merged.chain_index())

    # ct1_new = (z + conj(z)) · (0.5 / sd)
    s = phantom.add(ctx, ct_merged, ct_conj)
    ct1_new = phantom.multiply_plain(ctx, s, half_pt)
    ct1_new = phantom.rescale_to_next(ctx, ct1_new)
    ct1_new.set_scale(user_scale)

    # ct2_new = (z - conj(z)) · (-0.5i / sd)
    d = phantom.sub(ctx, ct_merged, ct_conj)
    ct2_new = phantom.multiply_plain(ctx, d, neg_half_i_pt)
    ct2_new = phantom.rescale_to_next(ctx, ct2_new)
    ct2_new.set_scale(user_scale)

    return ct1_new, ct2_new
