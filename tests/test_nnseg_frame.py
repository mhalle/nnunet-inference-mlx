"""nnseg.Frame + SimpleITK IO: mapping composition and the orientation round trip."""
import numpy as np
import pytest

lg = pytest.importorskip("labelgrid")
sitk = pytest.importorskip("SimpleITK")
from nnunet_inference_mlx.values import Geometry
from nnseg import io as nio
from nnseg.frame import Frame


def _frame(direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    source = lg.Grid((40, 60, 50), (1.0, 0.7, 0.7), (0.0, 0.0, 0.0))       # zyx, local frame
    canonical = Geometry(spacing_zyx=(1.0, 0.7, 0.7), shape_zyx=(40, 60, 50),
                         origin_xyz=(-20.0, -30.0, 5.0), direction_xyz=direction)
    return Frame(source=source, model_shape=(13, 14, 12), model_spacing=(3.0, 3.0, 3.0),
                 convention="corner", canonical=canonical, original_orientation="LPS")


def test_input_grid_mapping_is_the_forward_rule():
    f = _frame()
    m = f.mapping(f.source)
    want = lg.Mapping.corner(f.source.shape, f.model_shape)
    np.testing.assert_allclose(m.a, want.a); np.testing.assert_allclose(m.b, want.b, atol=1e-12)
    np.testing.assert_allclose(m.apply(np.array(f.source.shape) - 1), np.array(f.model_shape) - 1)


def test_isotropic_grid_mapping_goes_through_mm():
    f = _frame()
    g = f.resolve_grid(1.0)
    assert g.spacing == (1.0, 1.0, 1.0)
    idx = np.array([3.0, 7.0, 11.0])
    np.testing.assert_allclose(f.mapping(g).apply(idx),
                               f.forward_rule.apply(f.source.mm_to_index(g.index_to_mm(idx))))


def test_output_geometry_matches_the_input_grid():
    f = _frame()
    geo = f.output_geometry(f.source)
    assert geo.spacing_zyx == f.canonical.spacing_zyx
    assert geo.shape_zyx == f.canonical.shape_zyx
    np.testing.assert_allclose(geo.origin_xyz, f.canonical.origin_xyz)
    np.testing.assert_allclose(geo.direction_xyz, f.canonical.direction_xyz)


def test_output_geometry_of_an_isotropic_grid_keeps_world_position():
    f = _frame()
    g = f.resolve_grid(1.0)
    geo = f.output_geometry(g)
    assert geo.spacing_zyx == (1.0, 1.0, 1.0) and geo.shape_zyx == g.shape
    offset_xyz = np.asarray(g.origin)[::-1] - np.asarray(f.source.origin)[::-1]
    np.testing.assert_allclose(geo.origin_xyz, np.asarray(f.canonical.origin_xyz) + offset_xyz)


def test_output_geometry_rotates_the_offset_for_an_oblique_image():
    """Direction cosines must be applied to the grid offset, or an oblique acquisition
    comes back shifted along the wrong axes."""
    d = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)                     # 90 deg in-plane
    f = _frame(direction=d)
    g = f.resolve_grid(1.0)
    geo = f.output_geometry(g)
    offset_xyz = np.asarray(g.origin)[::-1] - np.asarray(f.source.origin)[::-1]
    want = np.asarray(f.canonical.origin_xyz) + np.asarray(d).reshape(3, 3) @ offset_xyz
    np.testing.assert_allclose(geo.origin_xyz, want)
    np.testing.assert_allclose(geo.direction_xyz, d)


def test_sitk_round_trip_through_a_non_ras_image(tmp_path):
    """read() canonicalizes to RAS; restore_orientation() puts it back exactly."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 9, size=(9, 7, 5)).astype(np.uint8)             # (z, y, x)
    img = sitk.GetImageFromArray(data)
    img.SetSpacing((0.8, 0.9, 2.0))
    img.SetOrigin((10.0, 20.0, -5.0))
    # SimpleITK's world frame is LPS, so an identity direction means the image *is* LPS -
    # the orientation the networks must never see (it mirrors left and right).
    img.SetDirection((1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    path = tmp_path / "vol.nii.gz"
    sitk.WriteImage(img, str(path))
    original = nio.orientation_of(img)
    assert original == "LPS"

    arr, geo, orient = nio.read(path)
    assert orient == original
    assert nio.orientation_of(nio.to_image(arr, geo)) == "RAS"
    assert geo.shape_zyx == arr.shape

    back = nio.restore_orientation(nio.to_image(arr, geo), orient)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(back), data)
    np.testing.assert_allclose(back.GetOrigin(), img.GetOrigin())
    np.testing.assert_allclose(back.GetDirection(), img.GetDirection(), atol=1e-12)
    np.testing.assert_allclose(back.GetSpacing(), img.GetSpacing())


def test_read_rejects_non_3d(tmp_path):
    img = sitk.GetImageFromArray(np.zeros((4, 5), dtype=np.uint8))
    p = tmp_path / "flat.nii.gz"
    sitk.WriteImage(img, str(p))
    with pytest.raises(ValueError):
        nio.read(p)


def test_resolve_grid_variants():
    f = _frame()
    assert f.resolve_grid("input") == f.source and f.resolve_grid(None) == f.source
    g = lg.Grid((5, 5, 5), (2, 2, 2), (1, 1, 1))
    assert f.resolve_grid(g) is g
    with pytest.raises(ValueError):
        Frame(source=f.source, model_shape=(1, 1, 1), model_spacing=(1, 1, 1), convention="node",
              canonical=f.canonical)
