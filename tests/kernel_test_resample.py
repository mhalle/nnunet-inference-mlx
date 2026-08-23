"""The forward resampler must reproduce its CPU references exactly, not approximately.

Absorbed from the nnU-Net fork (tag resample-gpu-v1) so nnseg does not pin a fork; these are
the checks that make that safe to rely on.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
ndimage = pytest.importorskip("scipy.ndimage")
from nnseg.resample import resample_data, scipy_axis_matrix, target_shape

try:
    from skimage.transform import resize as sk_resize
except Exception:                                                  # pragma: no cover
    sk_resize = None

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
          (["cuda"] if torch.cuda.is_available() else [])


def _vol(shape=(9, 11, 13), seed=0):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("factors", [(1.7, 2.3, 0.6), (0.45, 0.45, 3.1), (2.0, 0.5, 1.0)])
def test_corner_matches_scipy_zoom(device, order, factors):
    """convention="corner" == scipy.ndimage.zoom(grid_mode=False), which is what
    TotalSegmentator's change_spacing does."""
    vol = _vol()
    want = ndimage.zoom(vol, factors, order=order, mode="nearest", grid_mode=False)
    got = resample_data(vol, want.shape, convention="corner", order=order, mode="nearest", device=device)
    np.testing.assert_allclose(got, want, atol=2e-5 if device != "cpu" else 1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", [1, 3])
def test_center_matches_skimage_resize(device, order):
    """convention="center" == skimage.resize == nnU-Net's own resample_data_or_seg_to_shape."""
    if sk_resize is None:
        pytest.skip("scikit-image not installed")
    vol = _vol()
    out_shape = (12, 21, 7)
    want = sk_resize(vol, out_shape, order=order, mode="edge", anti_aliasing=False, preserve_range=True)
    # scipy's "nearest" is skimage's "edge" - same rule, different spelling; this takes scipy names
    got = resample_data(vol, out_shape, convention="center", order=order, mode="nearest", device=device)
    np.testing.assert_allclose(got, want, atol=2e-5 if device != "cpu" else 1e-9)


def test_center_clips_like_skimage_and_corner_does_not():
    """skimage.resize clips to the input range per call (nnU-Net inherits that);
    scipy.ndimage.zoom does not. Cubic overshoot is where it shows."""
    step = np.zeros((4, 4, 20), dtype=np.float64)
    step[..., 10:] = 1.0
    up = (4, 4, 60)
    center = resample_data(step, up, convention="center", order=3, mode="nearest", device="cpu")
    corner = resample_data(step, up, convention="corner", order=3, mode="nearest", device="cpu")
    assert center.min() >= -1e-12 and center.max() <= 1 + 1e-12, "center must clip"
    assert corner.min() < -1e-6 or corner.max() > 1 + 1e-6, "corner must ring like scipy"
    np.testing.assert_allclose(corner, ndimage.zoom(step, np.array(up) / np.array(step.shape),
                                                    order=3, mode="nearest", grid_mode=False), atol=1e-9)


def test_target_shape_is_the_change_spacing_rule():
    assert target_shape((709, 768, 768), (1.0, 0.651, 0.651), (3.0, 3.0, 3.0)) == (236, 167, 167)
    assert target_shape((10, 10, 10), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)) == (10, 10, 10)


def test_identity_is_exact():
    vol = _vol((5, 6, 7))
    np.testing.assert_array_equal(resample_data(vol, (5, 6, 7), device="cpu"), vol)


def test_axis_matrix_is_the_zoom_operator():
    """The identity-probe trick: zooming an identity matrix gives the operator itself."""
    w = scipy_axis_matrix(7, 11, 3, "nearest", False)
    assert w.shape == (11, 7)
    x = np.random.default_rng(1).normal(size=7)
    np.testing.assert_allclose(w @ x, ndimage.zoom(x, 11 / 7, order=3, mode="nearest", grid_mode=False), atol=1e-12)


def test_out_dtype_truncates_like_totalsegmentator():
    vol = np.full((4, 4, 4), 1.9)
    out = resample_data(vol, (4, 4, 4), out_dtype=np.int32, device="cpu")
    assert out.dtype == np.int32 and (out == 1).all()


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        resample_data(np.zeros((4, 4)), (2, 2, 2))
    with pytest.raises(ValueError):
        resample_data(np.zeros((4, 4, 4)), (2, 2))
    with pytest.raises(ValueError):
        resample_data(np.zeros((4, 4, 4)), (2, 2, 2), convention="node")
    with pytest.raises(ValueError):
        resample_data(np.zeros((4, 4, 4)))
