import numpy as np
import pytest

from nnseg import Grid, Mapping


def test_like_and_duck_typing():
    g = Grid((4, 5, 6), (1, 2, 3), (10, 20, 30))
    assert Grid.like(g) is g

    class Geo:
        shape = (4, 5, 6)
        spacing = (1.0, 2.0, 3.0)
        origin = (10.0, 20.0, 30.0)

    assert Grid.like(Geo()) == g


def test_index_mm_round_trip():
    g = Grid((4, 5, 6), (1.5, 2.0, 0.7), (-3.0, 4.0, 1.25))
    idx = np.array([[0, 0, 0], [3, 4, 5], [1.5, 2.25, 0.5]])
    mm = g.index_to_mm(idx)
    np.testing.assert_allclose(mm[0], g.origin)
    np.testing.assert_allclose(g.mm_to_index(mm), idx, atol=1e-12)


def test_resampled_edges_keeps_volume_extent():
    g = Grid((10, 20, 30), (3, 3, 3), (0, 0, 0))
    r = g.resampled(1.5, align="edges")
    assert r.shape == (20, 40, 60)
    np.testing.assert_allclose(r.origin, (-0.75, -0.75, -0.75))
    for a, b in zip(g.extent_mm, r.extent_mm):
        np.testing.assert_allclose(a, b)


def test_resampled_centers_keeps_center_extent():
    g = Grid((10, 20, 30), (3, 3, 3), (1, 2, 3))
    r = g.resampled(1.5, align="centers")
    assert r.shape == (19, 39, 59)
    assert r.origin == g.origin
    for a, b in zip(g.center_extent_mm, r.center_extent_mm):
        np.testing.assert_allclose(a, b)


def test_isotropic_is_resampled():
    g = Grid((10, 20, 30), (5, 0.7, 0.7), (0, 0, 0))
    assert Grid.isotropic(1.0, like=g) == g.resampled((1, 1, 1))
    assert Grid.isotropic(1.0, like=g).shape == (50, 14, 21)


def test_roi_on_lattice():
    g = Grid((10, 10, 10), (1, 1, 1), (0, 0, 0))
    r = g.roi((2.2, 0, 0), (5.1, 9, 9))
    assert r.shape == (4, 10, 10)
    assert r.origin == (2.0, 0.0, 0.0)
    assert r.spacing == g.spacing
    # clipped to the grid
    assert g.roi((-5, -5, -5), (50, 50, 50)) == g


def test_between_identity_and_physical():
    g = Grid((10, 20, 30), (3, 3, 3), (1, 2, 3))
    ident = Mapping.between(g, g)
    np.testing.assert_allclose(ident.a, (1, 1, 1))
    np.testing.assert_allclose(ident.b, (0, 0, 0))
    iso = Grid.isotropic(1.0, like=g)
    m = Mapping.between(iso, g)
    # index 0 of iso is at iso.origin; on g that is mm_to_index(iso.origin)
    np.testing.assert_allclose(m.apply([0, 0, 0]), g.mm_to_index(iso.origin))
    np.testing.assert_allclose(m.apply([3, 6, 9]), g.mm_to_index(iso.index_to_mm([3, 6, 9])))


def test_validation():
    with pytest.raises(ValueError):
        Grid((0, 1, 1))
    with pytest.raises(ValueError):
        Grid((1, 1, 1), (0, 1, 1))
    with pytest.raises(ValueError):
        Grid((1, 1), (1, 1))
    with pytest.raises(ValueError):
        Grid((2, 2, 2)).resampled(1.0, align="corners")
