"""nnseg.Frame: mapping composition and the orientation round trip (fast, no models)."""
import numpy as np
import pytest

lg = pytest.importorskip("labelgrid")
nib = pytest.importorskip("nibabel")
from nnseg.frame import Frame
from nnseg.preprocess import undo_canonical


def _frame():
    source = lg.Grid((40, 60, 50), (1.0, 0.7, 0.7), (0.0, 0.0, 0.0))       # zyx, local frame
    affine = np.array([[0.7, 0, 0, -20.0], [0, 0.7, 0, -30.0], [0, 0, 1.0, 5.0], [0, 0, 0, 1]])
    return Frame(source=source, model_shape=(13, 14, 12), model_spacing=(3.0, 3.0, 3.0), convention="corner", affine_canonical=affine)


def test_input_grid_mapping_is_the_forward_rule():
    f = _frame()
    m = f.mapping(f.source)
    want = lg.Mapping.corner(f.source.shape, f.model_shape)
    np.testing.assert_allclose(m.a, want.a); np.testing.assert_allclose(m.b, want.b, atol=1e-12)
    # last source voxel lands on the last model voxel (corner rule)
    np.testing.assert_allclose(m.apply(np.array(f.source.shape) - 1), np.array(f.model_shape) - 1)


def test_isotropic_grid_mapping_goes_through_mm():
    f = _frame()
    g = f.resolve_grid(1.0)
    assert g.spacing == (1.0, 1.0, 1.0)
    m = f.mapping(g)
    idx = np.array([3.0, 7.0, 11.0])
    mm = g.index_to_mm(idx)
    src_idx = f.source.mm_to_index(mm)
    np.testing.assert_allclose(m.apply(idx), f.forward_rule.apply(src_idx))


def test_output_affine_identity_and_isotropic():
    f = _frame()
    np.testing.assert_allclose(f.output_affine(f.source), f.affine_canonical)
    g = f.resolve_grid(1.0)
    A = f.output_affine(g)
    # world position of the isotropic grid's voxel (0,0,0) == canonical world of its source index
    src_idx_zyx = f.source.mm_to_index(g.index_to_mm([0, 0, 0]))
    want = f.affine_canonical @ np.array([*src_idx_zyx[::-1], 1.0])
    np.testing.assert_allclose(A @ np.array([0, 0, 0, 1.0]), want)
    np.testing.assert_allclose(np.abs(np.diag(A)[:3]), 1.0)


def test_canonical_round_trip_is_exact():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 9, size=(9, 7, 5)).astype(np.uint8)
    affine = np.array([[0, 0, -1.0, 10], [0, -0.8, 0, 20], [2.0, 0, 0, -5], [0, 0, 0, 1]])   # an S-P-L-like image
    img = nib.Nifti1Image(data, affine)
    can = nib.as_closest_canonical(img)
    assert nib.aff2axcodes(can.affine) == ("R", "A", "S")
    back = undo_canonical(can, img)
    np.testing.assert_array_equal(np.asanyarray(back.dataobj), data)
    np.testing.assert_allclose(back.affine, affine)


def test_resolve_grid_variants():
    f = _frame()
    assert f.resolve_grid("input") == f.source and f.resolve_grid(None) == f.source
    g = lg.Grid((5, 5, 5), (2, 2, 2), (1, 1, 1))
    assert f.resolve_grid(g) is g
    with pytest.raises(ValueError):
        Frame(source=f.source, model_shape=(1, 1, 1), model_spacing=(1, 1, 1), convention="node", affine_canonical=np.eye(4))
