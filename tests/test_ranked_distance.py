"""Correctness of the distance planes in tools/ranked_build_store.py.

These exist because the propagation was wrong once and two earlier tests failed to see it:

  - a PLANAR interface cannot separate a Euclidean distance from a taxicab one, because along
    its own normal the two agree exactly;
  - a curved interface can, but only if the statistic is the error against known truth.
    |grad d| looks fine for both once the field is clamped at the truncation, which is what a
    narrow band mostly is.

So the discriminating test is a sphere, measured against its analytic distance, at a truncation
wide enough that propagation actually runs.
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


# Every implementation answers the same contract, so every test runs against each. The numpy
# band version in tools/ is the reference; the torch version in nnseg.ranked is the product
# path (CUDA on the Modal worker, MPS locally). Parametrizing the suite is the point - two of
# three implementations of this algorithm were wrong in ways that rendered plausibly, and a
# test that runs against only one would certify the other by association. Resolution is LAZY:
# each param skips on its own missing dependency instead of a module-level importorskip
# skipping everything (zarr's absence must not silence the torch tests, or vice versa).
IMPL_NAMES = ["numpy"] + [f"torch-{d}" for d in _torch_devices()]


def _resolve(name):
    if name == "numpy":
        return _load_tools().distance_field
    dev = name.split("-", 1)[1]
    torch_field = pytest.importorskip("nnseg.ranked").distance_field

    def call(ranks, support, clip, sp, T, _dev=dev):
        return torch_field(ranks, support, clip=clip, spacing_zyx=sp,
                           truncation=T, device=_dev)
    return call


@pytest.fixture(params=IMPL_NAMES)
def field(request):
    return _resolve(request.param)


def _encode(signed):
    """A two-class ranked part whose winner margin is |signed|, as the store would hold it."""
    margin = np.minimum(np.abs(signed), CLIP)
    ranks = np.zeros((3,) + signed.shape, np.uint8)
    ranks[0] = np.where(signed > 0, 1, 2)
    ranks[1] = np.where(signed > 0, 2, 1)
    ranks[2] = 3
    support = np.zeros((2,) + signed.shape, np.uint8)
    support[0] = np.rint((1.0 - margin / CLIP) * 255.0)
    return ranks, support


def _decode(q, truncation):
    return (1.0 - q.astype(np.float64) / 255.0) * truncation


def test_planar_interface_is_exact(field):
    """A plane is where the encoding and the crossing interpolation can be checked exactly."""
    n = 64
    x = np.broadcast_to((np.arange(n) * SP[2])[None, None, :], (4, 4, n))
    x0 = 31.5 * SP[2]
    ranks, support = _encode(x - x0)
    truth = np.abs(x - x0)

    T = 2.0 * min(SP)
    q = field(ranks, support, CLIP, SP, T)
    band = q > 0
    err = np.abs(_decode(q[band], T) - truth[band])
    assert err.max() <= T / 255.0, f"planar decode off by {1e3 * err.max():.1f} um"
    assert (q[truth > T] == 0).all(), "beyond the truncation must be the zero sentinel"


@pytest.mark.parametrize("truncation_voxels", [2.0, 6.0])
def test_sphere_distance_is_euclidean_not_taxicab(field, truncation_voxels):
    """The case a plane cannot see: a taxicab propagation overshoots off-axis.

    Thresholds are set well clear of both implementations rather than tuned to one -- measured
    on this sphere, the taxicab error is 171 um at 2 voxels and 515 um at 6, against 94 and
    140 um for the Godunov update.
    """
    n, radius = 80, 18.0
    centre = (n - 1) / 2.0 * SP[0]
    g = np.meshgrid(*[np.arange(n) * SP[0] for _ in range(3)], indexing="ij")
    rad = np.sqrt(sum((c - centre) ** 2 for c in g))
    ranks, support = _encode(radius - rad)
    truth = np.abs(rad - radius)

    T = truncation_voxels * min(SP)
    q = field(ranks, support, CLIP, SP, T)
    band = q > 0
    err = np.abs(_decode(q[band], T) - truth[band])
    limit = 130e-3 if truncation_voxels <= 2.0 else 250e-3      # mm
    assert np.median(err) < limit, (
        f"median error {1e3 * np.median(err):.0f} um at T={truncation_voxels} voxels; a taxicab "
        f"propagation lands near {171 if truncation_voxels <= 2 else 515} um")


def test_zero_is_the_sentinel_everywhere(field):
    """`0` must keep meaning 'nothing here' so all-zero chunks elide, as in the other arrays."""
    n = 48
    x = np.broadcast_to((np.arange(n) * SP[2])[None, None, :], (4, 4, n))
    ranks, support = _encode(x - 23.5 * SP[2])
    T = 2.0 * min(SP)
    q = field(ranks, support, CLIP, SP, T)
    far = np.abs(x - 23.5 * SP[2]) > T
    assert (q[far] == 0).all()
    assert q.max() > 0, "the whole field cannot be sentinel"


def test_field_is_three_dimensional(field):
    """One 3-D field on the data grid, like `tail` - not a stack with a list axis.

    A second field keyed to the next logit rank was tried and dropped: it measures the
    l_winner = l_third level set, which generically sits buried under the runner-up and is
    visible nowhere. The nearest-surface field is found from the labelmap, so it already
    covers whichever pair forms the surface.
    """
    n = 24
    x = np.broadcast_to((np.arange(n) * SP[2])[None, None, :], (4, 4, n))
    ranks, support = _encode(x - 11.5 * SP[2])
    T = 2.0 * min(SP)
    out = field(ranks, support, CLIP, SP, T)
    assert out.shape == ranks.shape[1:]
    assert out.dtype == np.uint8


def test_seeding_follows_the_labelmap_not_the_runner_up(field):
    """An argmax change is found even when the overtaking class is not the runner-up.

    Watching the (winner, runner-up) pair misses this: at the left voxels A leads B leads D, at
    the right D leads A leads B. The winner changed from A to D, and the A-vs-B pair never
    crossed, so a pair-driven seeder finds no surface here at all.
    """
    n = 16
    shape = (4, 4, n)
    half = n // 2
    ranks = np.zeros((3,) + shape, np.uint8)
    support = np.zeros((2,) + shape, np.uint8)
    left = np.zeros(shape, bool)
    left[..., :half] = True

    ranks[0] = np.where(left, 1, 4)                 # winner  A | D
    ranks[1] = np.where(left, 2, 1)                 # runner  B | A
    ranks[2] = np.where(left, 4, 2)                 # third   D | B
    # left: B is 1 logit back, D is 3. right: A is 1 back, B is 3.
    support[0] = np.rint((1.0 - 1.0 / CLIP) * 255.0)
    support[1] = np.rint((1.0 - 3.0 / CLIP) * 255.0)

    T = 2.0 * min(SP)
    q = field(ranks, support, CLIP, SP, T)
    boundary = q[:, :, half - 1:half + 1]
    assert (boundary > 0).all(), (
        "no surface found at an argmax change whose overtaking class is not the runner-up")


@pytest.mark.parametrize("name", [n for n in IMPL_NAMES if n != "numpy"])
def test_torch_matches_the_numpy_reference_within_one_quantum(name):
    """GPU float reassociation rules out byte-identity; one uint8 quantum is the bound.

    Structured, not random: overlapping spheres of three classes give curved surfaces, a
    junction, and anisotropy-exercising geometry in one small volume.
    """
    ref = _resolve("numpy")
    impl = _resolve(name)
    n = 48
    sp = [2.0, 1.5, 1.5]
    g = np.meshgrid(*[np.arange(n) * s for s in sp], indexing="ij")
    c = [(n - 1) / 2.0 * s for s in sp]
    r1 = np.sqrt((g[0] - c[0]) ** 2 + (g[1] - c[1]) ** 2 + (g[2] - c[2]) ** 2)
    r2 = np.sqrt((g[0] - c[0]) ** 2 + (g[1] - c[1] * 0.6) ** 2 + (g[2] - c[2] * 1.3) ** 2)
    l1, l2 = 14.0 - r1, 11.0 - r2                     # two overlapping spheres vs background
    lg = np.stack([np.zeros_like(l1), l1, l2])
    order = np.argsort(-lg, axis=0)
    ranks = (order[:3] + 1).astype(np.uint8)
    top = np.take_along_axis(lg, order, axis=0)
    gaps = top[0:1] - top[1:3]
    support = np.rint((1.0 - np.clip(gaps, 0, CLIP) / CLIP) * 255.0).astype(np.uint8)

    T = 2.0 * min(sp)
    a = ref(ranks, support, CLIP, sp, T)
    b = impl(ranks, support, CLIP, sp, T)
    diff = np.abs(a.astype(int) - b.astype(int))
    assert diff.max() <= 1, (
        f"{name} differs from the numpy reference by up to {diff.max()} quanta "
        f"at {int((diff > 1).sum())} voxels")
    assert (a > 0).mean() > 0.01, "vacuous: the band is empty"
