"""Chain-aligning residual connection for FHE pipelines."""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom


def residual(ctx, ct_x, ct_y):
    """Add ct_x and ct_y at the deeper of their two chain levels.

    Mod-switches whichever ct is at a shallower level down to the deeper one,
    snaps scales to match, and returns ct_x + ct_y.
    """
    target_chain = max(ct_x.chain_index(), ct_y.chain_index())

    if ct_x.chain_index() != target_chain:
        a = phantom.mod_switch_to(ctx, ct_x, target_chain)
    else:
        a = ct_x

    if ct_y.chain_index() != target_chain:
        b = phantom.mod_switch_to(ctx, ct_y, target_chain)
    else:
        b = ct_y
    a.set_scale(b.scale())
    return phantom.add(ctx, a, b)
