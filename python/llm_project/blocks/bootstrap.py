"""Mean-centered bootstrap helper for the bootstrap-aware decoder pipeline.

CKKSEngine's EvalMod loses accuracy on inputs with non-zero mean; mean-centering
before bootstrap and adding the constant back after restores precision.
"""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom


def boot_centered(engine, ctx, encoder, sk, ct):
    """Mean-center, bootstrap, then restore the mean.

    Decrypts ct (in-test only — production wires this differently) to read the
    slot mean, subtracts the mean as a plaintext, runs engine.bootstrap_inplace
    on the centered ct, then adds the mean back as a plaintext at the bootstrap
    output's chain/scale.
    """
    dec = encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct))
    n = len(dec)
    mean_val = sum(dec) / n
    mean_pt = encoder.encode_double_vector(
        ctx, [mean_val] * n, ct.scale(), ct.chain_index())
    ct_c = phantom.sub_plain(ctx, ct, mean_pt)
    engine.bootstrap_inplace(ct_c)
    mean_pt2 = encoder.encode_double_vector(
        ctx, [mean_val] * n, ct_c.scale(), ct_c.chain_index())
    return phantom.add_plain(ctx, ct_c, mean_pt2)
