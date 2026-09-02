"""Correctness of the triple-line `junction` layer in tools/ranked_build_store.py.

The layer exists for one situation the main distance field cannot describe: where the
interface between two structures meets the surface against a third label. The discriminating
test is therefore a planar interface between A and B that runs up into a background cap - a
straight triple line - checked on both sides of the cap. Above it the background wins, and the
field must STILL be the signed distance to the A|B plane there: that continuation is what lets
a reader interpolate it at the surface, where half of every stencil is background.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools" / "ranked_build_store.py"
SP = [1.5, 1.5, 1.5]
CLIP = 8.0


def _load_tools():
    pytest.importorskip("zarr")
    spec = importlib.util.spec_from_file_location("ranked_build_store", TOOLS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _encode(logits):
    """Dense logits ``(K, Z, Y, X)`` -> ``ranks`` (K planes) and ``support``, as the store holds them."""
    order = np.argsort(-logits, axis=0, kind="stable")
    ranks = (order + 1).astype(np.uint8)
    top = np.take_along_axis(logits, order, axis=0)
    gaps = top[0:1] - top[1:]
    support = np.rint((1.0 - np.minimum(gaps, CLIP) / CLIP) * 255.0).astype(np.uint8)
    return ranks, support


def _decode(q, truncation, jmax):
    return (q.astype(np.float64) - 128.0) / jmax * truncation


def _triple_line(n=40, k=0.4):
    """A|B divided by the plane x = x0, both under a background cap above z = z0.

    l_bg sits `k * (z - z0)` above the better of A and B, so the background wins exactly
    above the cap whatever x is, and the A|B margin `l_A - l_B = -2k (x - x0)` does not depend
    on z at all - the virtual sheet is the plane x = x0 through the whole box.

    `k` is kept gentle so that no deficit in the tube reaches the clip: a deficit that clips
    saturates the field by design (see the builder), and this test is about the unsaturated
    part.
    """
    sp = SP[0]
    z, y, x = np.meshgrid(*[np.arange(n) * sp] * 3, indexing="ij")
    x0, z0 = 19.3 * sp, 24.6 * sp
    la = -(x - x0) * k
    lb = (x - x0) * k
    lbg = np.maximum(la, lb) + (z - z0) * k
    return _encode(np.stack([lbg, la, lb])), x0, z0, x, z


def test_signed_distance_holds_on_both_sides_of_the_cap():
    rbs = _load_tools()
    (ranks, support), x0, z0, x, z = _triple_line()
    T = 2.0 * min(SP)
    jn, pair = rbs.junction_field(ranks, support, CLIP, SP, T)
    present = jn > 0
    assert present.any()

    truth = x0 - x                                   # positive on A's side, A = class 1 < B
    band = present & (np.abs(truth) <= 0.9 * T)      # inside the truncation, off the clamp
    err = np.abs(_decode(jn[band], T, rbs.JUNCTION_MAX) - truth[band])
    # The store quantizes each gap to clip/255 logits; the difference of two gaps over a
    # gradient of 2k carries that into millimetres, and the gradient itself is differenced
    # from quantized gaps. Two tenths of a millimetre is a tenth of a voxel here, an order of
    # magnitude under the staircase the layer exists to replace.
    tol = 0.2
    assert err.max() <= tol, f"off by {1e3 * err.max():.0f} um"

    # The continuation through the third region is the whole point.
    above = band & (z > z0 + 0.5 * SP[0]) & (ranks[0] == 1)
    assert above.sum() > 100, "no background voxels in the tube - the cap was not crossed"
    err_above = np.abs(_decode(jn[above], T, rbs.JUNCTION_MAX) - truth[above])
    assert err_above.max() <= tol, (
        f"above the cap the field is off by {1e3 * err_above.max():.0f} um: the sheet does not "
        "continue into the background")


def test_pair_is_canonical_and_absent_with_the_byte():
    rbs = _load_tools()
    (ranks, support), *_ = _triple_line()
    T = 2.0 * min(SP)
    jn, pair = rbs.junction_field(ranks, support, CLIP, SP, T)
    present = jn > 0
    assert (pair[0][present] == 2).all() and (pair[1][present] == 3).all(), \
        "the pair must be (A, B) as class + 1, lower class first"
    assert (pair[0][~present] == 0).all() and (pair[1][~present] == 0).all(), \
        "where the byte is the sentinel the pair must be too"


def test_written_only_in_tubes_around_the_triple_line():
    """Sparse by construction: nothing on the A|B interface away from the cap, nothing on the
    cap away from the interface, and only a small fraction of the box in all."""
    rbs = _load_tools()
    (ranks, support), x0, z0, x, z = _triple_line()
    T = 2.0 * min(SP)
    sp = SP[0]
    jn, _ = rbs.junction_field(ranks, support, CLIP, SP, T)
    far_down = (np.abs(x - x0) < 0.6 * sp) & (z < z0 - 8 * sp)      # on the interface, deep
    far_along = (np.abs(z - z0) < 0.6 * sp) & (x > x0 + 8 * sp)     # on the cap, far from it
    assert (jn[far_down] == 0).all(), "written along a plain two-structure interface"
    assert (jn[far_along] == 0).all(), "written along a plain outer surface"
    near = (np.abs(x - x0) < 1.5 * sp) & (np.abs(z - z0) < 1.5 * sp)
    assert (jn[near] > 0).all(), "the triple line itself is not covered"
    frac = np.count_nonzero(jn) / jn.size
    assert frac < 0.12, f"{100 * frac:.1f} % of the box written - that is a slab, not a tube"


def test_no_triple_line_means_nothing_written():
    rbs = _load_tools()
    n, sp = 24, SP[0]
    x = np.broadcast_to((np.arange(n) * sp)[None, None, :], (n, n, n))
    la = -(x - 11.4 * sp)
    lb = -la
    lbg = np.full_like(la, -20.0)                    # never wins: a plain A|B interface
    ranks, support = _encode(np.stack([lbg, la, lb]))
    jn, pair = rbs.junction_field(ranks, support, CLIP, SP, 2.0 * sp)
    assert not jn.any() and not pair.any()
