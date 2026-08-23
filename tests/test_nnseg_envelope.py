"""The body envelope: mask, box, margin, and the degenerate cases that must fail safe."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from nnseg.envelope import body_mask, envelope_of, margin_in_voxels


def _ct(shape=(20, 30, 40)):
    """air everywhere, a patient block, a thin disconnected 'table' and a speck"""
    hu = np.full(shape, -1000.0)
    hu[4:16, 8:24, 10:32] = 40.0             # patient
    hu[17, 2:28, 5:38] = 200.0               # table: thin, disconnected
    hu[1, 1, 1] = 500.0                      # a stray bright voxel
    return hu


def test_mask_keeps_the_patient_and_drops_the_rest():
    m = body_mask(_ct())
    assert m[10, 15, 20] and not m[17, 15, 20] and not m[1, 1, 1]
    assert m.sum() == 12 * 16 * 22


def test_mask_without_component_filter_keeps_everything_above_air():
    m = body_mask(_ct(), largest_component=False)
    assert m[17, 15, 20] and m[1, 1, 1]


def test_envelope_is_the_patient_box_plus_margin_clipped():
    env = envelope_of(body_mask(_ct()), margin_voxels=(2, 3, 100))
    assert env.lo == (2, 5, 0) and env.hi == (18, 27, 40)
    assert env.slices == (slice(2, 18), slice(5, 27), slice(0, 40))
    assert 0 < env.fraction < 1 and not env.is_whole()


def test_empty_mask_means_the_whole_volume_never_an_empty_slab():
    env = envelope_of(np.zeros((5, 6, 7), bool), margin_voxels=1)
    assert env.is_whole() and env.fraction == 1.0


def test_speck_does_not_shrink_the_fov():
    """a single false-positive voxel far from the body must not become the envelope"""
    hu = _ct()
    hu[4:16, 8:24, 10:32] = -1000.0          # no patient at all: only table + speck remain
    m = body_mask(hu)                        # largest component = the table line
    env = envelope_of(m, margin_voxels=0)
    assert env.lo[0] == 17 and env.hi[0] == 18


def test_margin_in_voxels_rounds_up():
    assert margin_in_voxels(20.0, (3.0, 3.0, 3.0)) == (7, 7, 7)
    assert margin_in_voxels(20.0, (1.5, 0.7, 0.7)) == (14, 29, 29)
    assert margin_in_voxels(0.0, (3.0, 3.0, 3.0)) == (0, 0, 0)
