"""The convention constructors against the resamplers they claim to invert."""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates, zoom
from skimage.transform import resize

from nnseg import Mapping, build_tables, reference


def _vol(shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float64)


def _values(vol, out_shape, mapping, interp="linear", outside="background"):
    tables = build_tables(out_shape, vol.shape, mapping, interp=interp, outside=outside)
    v, valid = reference.interpolate(vol[None], tables)
    assert valid.all()
    return v[0]


@pytest.mark.parametrize("factors", [(1.7, 2.3, 0.6), (0.45, 0.45, 3.1), (2.0, 0.5, 1.0)])
def test_corner_matches_scipy_zoom(factors):
    vol = _vol((7, 9, 11))
    want = zoom(vol, factors, order=1, mode="nearest", grid_mode=False)
    out_shape = want.shape
    got = _values(vol, out_shape, Mapping.corner(out_shape, vol.shape))
    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("out_shape", [(12, 21, 7), (3, 4, 25), (7, 9, 11)])
def test_center_matches_skimage_resize(out_shape):
    vol = _vol((7, 9, 11))
    want = resize(vol, out_shape, order=1, mode="edge", anti_aliasing=False, preserve_range=True)
    got = _values(vol, out_shape, Mapping.center(out_shape, vol.shape))
    np.testing.assert_allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("factors", [(1.7, 2.3, 0.6), (0.45, 0.45, 3.1), (2.5, 1.0, 0.5)])
def test_nearest_matches_scipy_zoom_order0(factors):
    lab = np.random.default_rng(1).integers(0, 50, size=(7, 9, 11)).astype(np.float64)
    want = zoom(lab, factors, order=0, mode="nearest", grid_mode=False)
    got = _values(lab, want.shape, Mapping.corner(want.shape, lab.shape), interp="nearest")
    np.testing.assert_array_equal(got, want)


def test_separate_z_matches_nnunet_style_reference():
    """linear in-plane (skimage, center), nearest along z (scipy order 0)."""
    vol = _vol((6, 9, 11))
    out_shape = (15, 17, 21)
    planes = np.stack([resize(vol[z], out_shape[1:], order=1, mode="edge", anti_aliasing=False,
                              preserve_range=True) for z in range(vol.shape[0])])
    cz = (np.arange(out_shape[0]) + 0.5) * vol.shape[0] / out_shape[0] - 0.5
    iz = np.clip(np.floor(np.clip(cz, 0, vol.shape[0] - 1) + 0.5), 0, vol.shape[0] - 1).astype(int)
    want = planes[iz]
    got = _values(vol, out_shape, Mapping.center(out_shape, vol.shape), interp=("nearest", "linear", "linear"))
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_map_coordinates_with_explicit_affine():
    vol = _vol((8, 9, 10))
    m = Mapping((0.6, 1.3, 0.9), (0.4, -0.2, 1.1))
    out_shape = (9, 6, 8)
    tables = build_tables(out_shape, vol.shape, m, outside="clamp")
    got, _ = reference.interpolate(vol[None], tables)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in out_shape], indexing="ij"), -1)
    coords = np.clip(m.apply(grid), 0, np.array(vol.shape) - 1)
    want = map_coordinates(vol, coords.reshape(-1, 3).T, order=1, mode="nearest").reshape(out_shape)
    np.testing.assert_allclose(got[0], want, atol=1e-12)


def test_spacing_rule_is_the_mlx_kernels():
    m = Mapping.spacing((1.5, 1.5, 1.5), (3.0, 3.0, 3.0))
    np.testing.assert_allclose(m.apply([10, 20, 30]), [5, 10, 15])
    m = Mapping.spacing((1.0, 0.651, 0.651), (3.0, 3.0, 3.0), shift=(1, 2, 3))
    np.testing.assert_allclose(m.apply([0, 0, 0]), [1, 2, 3])


def test_compose_and_inverse():
    m1 = Mapping((0.5, 2.0, 1.5), (1.0, -3.0, 0.25))
    m2 = Mapping((3.0, 0.25, 1.0), (-1.0, 2.0, 0.0))
    x = np.array([[1.0, 2.0, 3.0], [7.0, 0.0, 5.5]])
    np.testing.assert_allclose((m1 >> m2).apply(x), m2.apply(m1.apply(x)))
    np.testing.assert_allclose((m1 >> m1.inverse()).apply(x), x, atol=1e-12)
    with pytest.raises(ValueError):
        Mapping((-1, 1, 1))
    with pytest.raises(ValueError):
        Mapping.corner((5, 5, 5), (1, 5, 5)).inverse()


def test_corner_single_sample_axis():
    m = Mapping.corner((1, 5, 5), (4, 5, 5))
    assert m.a[0] == 0.0
    assert m.apply([0, 0, 0])[0] == 0.0
