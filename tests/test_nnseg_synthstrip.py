"""SynthStrip engine: geometry (SDT restore), conform shape, the vendored model,
and the weights identity - all with synthetic data, no real weights or GPU. The
model-dependent compute is validated by the live Modal smoke."""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from nnseg.engines import synthstrip as ss


def _img(arr, spacing, origin=(0., 0., 0.), direction=None):
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    im.SetSpacing(spacing); im.SetOrigin(origin)
    if direction is not None:
        im.SetDirection(direction)
    return im


def test_module_imports_without_model():
    # importing the engine must not require torch/surfa/the .pt (all lazy in functions)
    assert callable(ss.segment) and callable(ss.restore_sdt_gpu) and callable(ss.restore_sdt_cpu)


def test_weights_installed_is_the_cache_key_identity():
    wi = ss.weights_installed()
    assert wi == [{"id": "synthstrip", "version": ss.WEIGHTS_VERSION}]


def test_stripmodel_state_dict_roundtrips_and_forward_is_one_channel():
    torch = pytest.importorskip("torch")
    from nnseg.engines._synthstrip_model import StripModel
    m = StripModel()
    m2 = StripModel()
    missing, unexpected = m2.load_state_dict(m.state_dict(), strict=True)
    assert not missing and not unexpected           # arch is self-consistent
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 1, 64, 64, 64))       # 7 levels -> min side 64
    assert tuple(out.shape) == (1, 1, 64, 64, 64)    # single-channel SDT


def test_restore_sdt_places_zero_crossing_at_the_plane():
    """A signed distance to the plane x = c, restored to a finer grid and
    thresholded at 0, must put the brain/background boundary at x = c (sub-voxel),
    not snapped to the coarse grid."""
    Z = Y = X = 20
    xc = 11.3                                        # boundary off the coarse centers
    xphys = (np.arange(X))[None, None, :] * np.ones((Z, Y, X))
    sdt = (xphys - xc).astype(np.float32)           # <0 (brain) for x < xc
    source = _img(sdt * 0, (1., 1., 1.))
    f = 4
    target = _img(np.zeros((Z*f, Y*f, X*f)), (0.25, 0.25, 0.25))
    native = ss.restore_sdt_gpu(sdt, source, target, device="cpu")
    mid = native[native.shape[0]//2, native.shape[1]//2, :]
    cross = int(np.argmax(mid >= 0))                # first x where SDT >= 0 (background)
    assert abs(cross * 0.25 - xc) <= 0.25 + 1e-6, (cross * 0.25, xc)


def test_restore_sdt_gpu_matches_cpu_reference():
    """The GPU SDT restore (grid_sample) must match the SimpleITK CPU restore -
    same physical-space mapping and half-pixel convention, same outside fill.
    Run on the CPU torch device (no GPU needed). Flipped direction + anisotropy."""
    pytest.importorskip("torch")
    Z = Y = X = 16
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    sdt = ((xx + 0.5*yy + 0.25*zz) - 12.0).astype(np.float32)     # smooth signed field
    flip = (-1., 0., 0., 0., -1., 0., 0., 0., 1.)
    source = _img(np.zeros((Z, Y, X)), (1.5, 1.25, 1.0), origin=(10., -20., 5.), direction=flip)
    target = _img(np.zeros((Z*2, Y*2, X*2)), (0.75, 0.6, 0.5), origin=(9., -18., 6.), direction=flip)
    gpu = ss.restore_sdt_gpu(sdt, source, target, device="cpu")
    cpu = ss.restore_sdt_cpu(sdt, source, target)
    # Bit-identical in the interior; the two differ only in the FOV-boundary band
    # (grid_sample vs SimpleITK edge handling), which is far-field (>> border ->
    # background) and removed by the connected-component step - never the brain.
    interior = tuple(slice(6, -6) for _ in range(3))
    assert np.allclose(gpu[interior], cpu[interior], atol=1e-3), \
        float(np.abs(gpu[interior] - cpu[interior]).max())


# NOTE: conform is now SynthStrip's own surfa conform (the trained-input contract),
# exercised end-to-end by the live Modal smoke - surfa's reorient crashes on some
# local numpy builds, so it is not unit-tested here.
