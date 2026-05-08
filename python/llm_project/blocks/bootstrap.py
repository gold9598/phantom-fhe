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
