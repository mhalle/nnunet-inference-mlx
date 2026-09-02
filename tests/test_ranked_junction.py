"""Correctness of the triple-line `junction` layer, for every implementation of it.

The layer exists for one situation the main distance field cannot describe: where the
interface between two structures meets the surface against a third label. The discriminating
test is therefore a planar interface between A and B that runs up into a background cap - a
straight triple line - checked on both sides of the cap. Above it the background wins, and the
field must STILL be the signed distance to the A|B plane there: that continuation is what lets
a reader interpolate it at the surface, where half of every stencil is background.

Two implementations answer one contract - the numpy reference in tools/ and the torch version
in nnseg.ranked - and every test runs against each, the way the distance tests do. Resolution
is lazy so that a missing dependency skips one parameter, not the suite.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools" / "ranked_build_store.py"
SP = [1.5, 1.5, 1.5]
CLIP = 8.0
JZERO, JSPAN = 128, 127


def _load_tools():
    pytest.importorskip("zarr")
    spec = importlib.util.spec_from_file_location("ranked_build_store", TOOLS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _torch_devices():
    try:
        import torch
    except ImportError:
        return []
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        out.append("mps")
    return out


IMPL_NAMES = ["numpy"] + [f"torch-{d}" for d in _torch_devices()]


def _resolve(name):
    if name == "numpy":
        ref = _load_tools().junction_field
        return lambda ranks, support, clip, sp, T: ref(ranks, support, clip, sp, T)
    dev = name.split("-", 1)[1]
    torch_field = pytest.importorskip("nnseg.ranked").junction_field

    def call(ranks, support, clip, sp, T, _dev=dev):
        return torch_field(ranks, support, clip=clip, spacing_zyx=sp, truncation=T, device=_dev)
    return call


@pytest.fixture(params=IMPL_NAMES)
def field(request):
    return _resolve(request.param)


def _encode(logits):
    """Dense logits ``(K, Z, Y, X)`` -> ``ranks`` (K planes) and ``support``, as the store holds them."""
    order = np.argsort(-logits, axis=0, kind="stable")
    ranks = (order + 1).astype(np.uint8)
    top = np.take_along_axis(logits, order, axis=0)
    gaps = top[0:1] - top[1:]
    support = np.rint((1.0 - np.minimum(gaps, CLIP) / CLIP) * 255.0).astype(np.uint8)
    return ranks, support


def _decode(q, truncation):
    return (q.astype(np.float64) - JZERO) / JSPAN * truncation


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


def test_signed_distance_holds_on_both_sides_of_the_cap(field):
    (ranks, support), x0, z0, x, z = _triple_line()
    T = 2.0 * min(SP)
    jn, pair = field(ranks, support, CLIP, SP, T)
    present = jn > 0
    assert present.any()

    truth = x0 - x                                   # positive on A's side, A = class 1 < B
    band = present & (np.abs(truth) <= 0.9 * T)      # inside the truncation, off the clamp
    err = np.abs(_decode(jn[band], T) - truth[band])
    # The store quantizes each gap to clip/255 logits; the difference of two gaps over a
    # gradient of 2k carries that into millimetres, and the gradient itself is differenced
    # from quantized gaps. Two tenths of a millimetre is a tenth of a voxel here, an order of
    # magnitude under the staircase the layer exists to replace.
    tol = 0.2
    assert err.max() <= tol, f"off by {1e3 * err.max():.0f} um"

    # The continuation through the third region is the whole point.
    above = band & (z > z0 + 0.5 * SP[0]) & (ranks[0] == 1)
    assert above.sum() > 100, "no background voxels in the tube - the cap was not crossed"
    err_above = np.abs(_decode(jn[above], T) - truth[above])
    assert err_above.max() <= tol, (
        f"above the cap the field is off by {1e3 * err_above.max():.0f} um: the sheet does not "
        "continue into the background")


def test_pair_is_canonical_and_absent_with_the_byte(field):
    (ranks, support), *_ = _triple_line()
    T = 2.0 * min(SP)
    jn, pair = field(ranks, support, CLIP, SP, T)
    present = jn > 0
    assert (pair[0][present] == 2).all() and (pair[1][present] == 3).all(), \
        "the pair must be (A, B) as class + 1, lower class first"
    assert (pair[0][~present] == 0).all() and (pair[1][~present] == 0).all(), \
        "where the byte is the sentinel the pair must be too"


def test_written_only_in_tubes_around_the_triple_line(field):
    """Sparse by construction: nothing on the A|B interface away from the cap, nothing on the
    cap away from the interface, and only a small fraction of the box in all."""
    (ranks, support), x0, z0, x, z = _triple_line()
    T = 2.0 * min(SP)
    sp = SP[0]
    jn, _ = field(ranks, support, CLIP, SP, T)
    far_down = (np.abs(x - x0) < 0.6 * sp) & (z < z0 - 8 * sp)      # on the interface, deep
    far_along = (np.abs(z - z0) < 0.6 * sp) & (x > x0 + 8 * sp)     # on the cap, far from it
    assert (jn[far_down] == 0).all(), "written along a plain two-structure interface"
    assert (jn[far_along] == 0).all(), "written along a plain outer surface"
    near = (np.abs(x - x0) < 1.5 * sp) & (np.abs(z - z0) < 1.5 * sp)
    assert (jn[near] > 0).all(), "the triple line itself is not covered"
    frac = np.count_nonzero(jn) / jn.size
    assert frac < 0.12, f"{100 * frac:.1f} % of the box written - that is a slab, not a tube"


def test_no_triple_line_means_nothing_written(field):
    n, sp = 24, SP[0]
    x = np.broadcast_to((np.arange(n) * sp)[None, None, :], (n, n, n))
    la = -(x - 11.4 * sp)
    lb = -la
    lbg = np.full_like(la, -20.0)                    # never wins: a plain A|B interface
    ranks, support = _encode(np.stack([lbg, la, lb]))
    jn, pair = field(ranks, support, CLIP, SP, 2.0 * sp)
    assert not jn.any() and not pair.any()


@pytest.mark.parametrize("name", [n for n in IMPL_NAMES if n != "numpy"])
def test_torch_matches_the_numpy_reference(name):
    """Two implementations of one algorithm: byte-identical pairs, and bytes within one quantum
    (GPU float reassociation in the gradient's sum of squares is the only latitude)."""
    ref = _resolve("numpy")
    other = _resolve(name)
    # A sphere of A inside B under a tilted cap of background: triple lines that are curves,
    # so every axis and every sign combination is exercised.
    n, sp = 36, SP[0]
    z, y, x = np.meshgrid(*[np.arange(n) * sp] * 3, indexing="ij")
    c = (n - 1) / 2 * sp
    rad = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    la = (12.0 - rad) * 0.4
    lb = -la
    lbg = np.maximum(la, lb) + ((z - c) * 0.7 + (x - c) * 0.3) * 0.3
    ranks, support = _encode(np.stack([lbg, la, lb]))
    T = 2.0 * sp
    jn_r, jp_r = ref(ranks, support, CLIP, SP, T)
    jn_o, jp_o = other(ranks, support, CLIP, SP, T)
    assert jn_r.any()
    assert (jp_r == jp_o).all(), "the pair planes differ"
    assert ((jn_r > 0) == (jn_o > 0)).all(), "the tubes differ"
    diff = np.abs(jn_r.astype(int) - jn_o.astype(int))
    assert diff.max() <= 1, f"bytes differ by up to {diff.max()} quanta"


def test_in_place_tool_matches_the_dense_function(tmp_path):
    """The slab-wise, zarr-backed path of tools/ranked_add_junction.py must produce exactly the
    arrays the dense function returns - the same tubes, bytes and pairs - and leave the store
    decodable. Peak memory is the point of that path, so it must not buy it with a different
    answer."""
    zarr = pytest.importorskip("zarr")
    rbs = _load_tools()
    spec = importlib.util.spec_from_file_location(
        "ranked_add_junction", TOOLS.parent / "ranked_add_junction.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    (ranks, support), *_ = _triple_line()
    T = 2.0 * min(SP)
    want_j, want_p = rbs.junction_field(ranks, support, CLIP, SP, T)

    store = tmp_path / "t.duckn"
    root = zarr.create_group(store=str(store))
    g = root.create_group("parts/0")
    axes = [{"kind": "space", "centering": "node", "unit": "mm",
             "space_direction": [SP[i] if j == i else 0.0 for j in range(3)]} for i in range(3)]
    attrs4 = {"duckn": {"version": "1.0", "space": "left-posterior-superior",
                        "space_origin": [0.0, 0.0, 0.0], "axes": [{"kind": "list"}] + axes}}
    for nm, arr in (("ranks", ranks), ("support", support)):
        z = g.create_array(nm, shape=arr.shape, dtype=arr.dtype, chunks=(1, 16, 16, 16),
                           attributes=attrs4)
        z[:] = arr
    g.attrs.update({"duckn": {"version": "1.0", "extensions": {"ranked": {
        "clip": CLIP, "distance_truncation": T, "support_max": 255}}}})

    tool.add(store)
    g = zarr.open_group(str(store), mode="r")["parts/0"]
    got_j = np.asarray(g["junction"][:])
    got_p = np.asarray(g["junction_pair"][:])
    assert (got_j == want_j).all(), "the slab-wise tool differs from the dense function"
    assert (got_p == want_p).all()
    m = g.attrs.asdict()["duckn"]["extensions"]["ranked"]
    assert m["junction_truncation"] == round(T, 6)
    assert m["junction_zero"] == rbs.JUNCTION_ZERO and m["junction_span"] == rbs.JUNCTION_SPAN
