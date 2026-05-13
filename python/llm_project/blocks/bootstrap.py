"""Mean-centered + magnitude-scaled bootstrap helper.

CKKS bootstrap is built around an EvalMod sine polynomial that is only accurate
when slot values lie inside roughly [-0.5, 0.5] — the same regime the_lib's
example exercises (`(i % 100 - 50) / 100.`). Inputs outside that band wrap
modularly: a value of v becomes v - round(v). Because of this, both a non-zero
mean and a large per-slot magnitude break the bootstrap.

`boot_centered` decrypts the ciphertext (test-only — production wires the mean
extraction differently) to read the slot mean and the maximum |centered|, then:

  1. subtracts the mean as a plaintext (level-free),
  2. scales the centered ciphertext into [-TARGET_MAG, TARGET_MAG] via a
     plaintext multiply + rescale (1 level — only when needed),
  3. runs `engine.bootstrap_inplace` on the safely-bounded ciphertext,
  4. multiplies by the inverse factor to restore the original magnitude
     (1 level, only when scaling was applied),
  5. adds the mean back as a plaintext at the new chain/scale.

The scale-down/scale-up symmetry is what makes large ranges like [-256, 256]
work — the homomorphic bootstrap only ever sees values inside its safe domain,
and the recovered ciphertext is faithful to the original up to bootstrap noise
amplified by `max_centered / TARGET_MAG`.
"""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom


# Bootstrap (K=28 R=3) is accurate to ~8.5e-5 absolute across the full
# [-0.5, 0.5] domain — the polynomial's max-error floor is essentially flat
# from |x|=0.1 to |x|=0.49, so bigger TARGET_MAG is strictly better here:
# the post-bootstrap scale-up factor `max_centered / TARGET_MAG` shrinks
# linearly with TARGET_MAG, and the final error scales the same way. 0.49
# gives the best precision without crossing the polynomial's mod-1 wrap point.
# Using 0.49 (rather than 0.4) also keeps random inputs sampled from
# [-0.4, 0.4] off the scale-down path — encoder/decoder noise can otherwise
# nudge max|centered| above a tighter threshold and spuriously consume a level.
TARGET_MAG = 0.49


def boot_centered(engine, ctx, encoder, sk, ct):
    """Mean-center, scale-to-fit, bootstrap, then unscale and restore the mean.

    Handles inputs with non-zero mean and per-slot magnitudes much larger than
    bootstrap's [-0.5, 0.5] domain. Costs 0 extra user levels for inputs that
    already fit, and 2 extra user levels (one pre, one post) for inputs that
    require scaling. The post-bootstrap level is freshest+0 in the no-scale
    case and freshest+1 in the scaled case; callers should budget accordingly.
    """
    user_scale = engine.user_scale()

    dec = encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct))
    n = len(dec)
    mean_val = sum(dec) / n
    max_centered = max(abs(d - mean_val) for d in dec)

    mean_pt = encoder.encode_double_vector(
        ctx, [mean_val] * n, ct.scale(), ct.chain_index())
    ct_c = phantom.sub_plain(ctx, ct, mean_pt)

    needs_scale = max_centered > TARGET_MAG
    if needs_scale:
        # Pre-scale rescales ct one level deeper. At max_user_level the
        # rescale lands on the engine's internal bottom prime, which the
        # public bootstrap_inplace refuses. Bootstrap one user-level earlier
        # if you need scaling.
        if engine.user_level(ct_c) >= engine.max_user_level():
            raise ValueError(
                f"boot_centered: input at user_level {engine.user_level(ct_c)} "
                f"(== max_user_level {engine.max_user_level()}) requires scaling "
                f"(max|centered|={max_centered:.3f} > TARGET_MAG={TARGET_MAG}); "
                "bootstrap one level earlier in the pipeline.")
        scale_down = TARGET_MAG / max_centered
        scale_up = 1.0 / scale_down
        scale_pt = encoder.encode_double_vector(
            ctx, [scale_down] * n, ct_c.scale(), ct_c.chain_index())
        ct_c = phantom.multiply_plain(ctx, ct_c, scale_pt)
        ct_c = phantom.rescale_to_next(ctx, ct_c)
        ct_c.set_scale(user_scale)

    engine.bootstrap_inplace(ct_c)

    if needs_scale:
        unscale_pt = encoder.encode_double_vector(
            ctx, [scale_up] * n, ct_c.scale(), ct_c.chain_index())
        ct_c = phantom.multiply_plain(ctx, ct_c, unscale_pt)
        ct_c = phantom.rescale_to_next(ctx, ct_c)
        ct_c.set_scale(user_scale)

    mean_pt2 = encoder.encode_double_vector(
        ctx, [mean_val] * n, ct_c.scale(), ct_c.chain_index())
    return phantom.add_plain(ctx, ct_c, mean_pt2)


def bootstrap_safe(engine, ctx, encoder, ct, max_abs, slot_count, target_mag=0.49):
    """SK-free bootstrap with static input range.

    Pre-scales `ct` by `target_mag / max_abs` so post-scale slots fit inside
    bootstrap's safe domain, runs `engine.bootstrap_inplace`, then unscales.
    Caller-provided `max_abs` is the static upper bound on |slot value|;
    the wrapper does not measure the input. If `max_abs <= target_mag`,
    no scaling is applied and the bootstrap runs directly (zero extra levels).

    Inputs are assumed approximately mean-zero. If a call site has a known
    non-zero mean, subtract it as a plaintext before calling this and add
    back after.
    """
    user_scale = engine.user_scale()
    needs_scale = max_abs > target_mag
    if needs_scale:
        if engine.user_level(ct) >= engine.max_user_level():
            raise ValueError(
                f"bootstrap_safe: input at user_level {engine.user_level(ct)} "
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
    exactly the 1 level a bootstrap_safe(ct) would have used for its own
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
