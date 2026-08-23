"""The body envelope: mask, box, margin, and the degenerate cases that must fail safe."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from nnseg.envelope import Envelope, body_mask, envelope_of, margin_in_voxels


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


def test_label_roi_boxes_the_requested_classes():
    from nnseg.envelope import label_roi
    lab = np.zeros((20, 30, 40), np.uint8)
    lab[5:9, 10:14, 12:18] = 3          # class 3 here
    lab[15, 25, 35] = 7                 # class 7 elsewhere
    roi = label_roi(lab, [3], margin_voxels=(1, 1, 1))
    assert roi.lo == (4, 9, 11) and roi.hi == (10, 15, 19)   # box of class 3 only, +1
    both = label_roi(lab, [3, 7], margin_voxels=0)
    assert both.lo == (5, 10, 12) and both.hi == (16, 26, 36)  # spans both
    absent = label_roi(lab, [99], margin_voxels=0)
    assert absent.is_whole()            # missing class -> whole grid, never empty


def test_otsu_splits_a_bimodal_histogram_between_the_modes():
    from nnseg.envelope import otsu_threshold
    rng = np.random.default_rng(0)
    lo = rng.normal(-2.0, 0.2, 40000)          # "air" mode
    hi = rng.normal(+2.0, 0.3, 40000)          # "tissue" mode
    t = otsu_threshold(np.concatenate([lo, hi]))
    assert -2.0 < t < 2.0                                    # between the two modes
    assert (lo < t).mean() > 0.99 and (hi > t).mean() > 0.99  # and it separates them cleanly


def test_otsu_degenerate_inputs_fail_safe():
    from nnseg.envelope import otsu_threshold
    assert otsu_threshold(np.zeros(100)) == 0.0            # constant -> lo, no crash
    assert otsu_threshold(np.array([], dtype=float)) == 0.0


def test_body_threshold_ct_matches_the_hu_formula():
    from nnseg.envelope import body_threshold, AIR_HU
    props = {"percentile_00_5": -700.0, "mean": 100.0, "std": 250.0}
    # CT: dataset-derived HU cut in normalized units, exactly the old inline formula
    t = body_threshold(np.zeros((4, 4, 4)), normalization_schemes=("CTNormalization",),
                       intensity_properties=props)
    assert t == (max(AIR_HU, props["percentile_00_5"]) - props["mean"]) / props["std"]


def test_body_threshold_mr_is_data_driven_and_crops_the_blob():
    """A ZScore-normalized MR: background near -1, a centered blob near +2. The threshold must
    come from the image (Otsu), and the resulting box must exclude the surrounding air."""
    from nnseg.envelope import body_threshold, body_mask, envelope_of
    x = np.full((24, 40, 40), -1.0)
    x[6:18, 10:30, 12:28] = 2.0                # the "body"
    t = body_threshold(x, normalization_schemes=("ZScoreNormalization",), intensity_properties={})
    assert -1.0 < t < 2.0
    env = envelope_of(body_mask(x, threshold=t), margin_voxels=(1, 1, 1))
    assert not env.is_whole() and env.fraction < 0.5
    assert env.lo[0] <= 6 and env.hi[0] >= 18   # the blob is fully inside the box


def test_body_threshold_mr_ignores_stray_ct_properties():
    """Even if a ZScore model happens to carry foreground props, MR must not use the HU path."""
    from nnseg.envelope import body_threshold
    x = np.concatenate([np.full(20000, -1.0), np.full(20000, 2.0)])
    t = body_threshold(x, normalization_schemes=("ZScoreNormalization",),
                       intensity_properties={"percentile_00_5": -700.0, "mean": 100.0, "std": 250.0})
    assert -1.0 < t < 2.0                        # Otsu, not the (-700-100)/250 HU value


def test_worth_cropping_collapses_a_near_whole_box():
    from nnseg.envelope import worth_cropping
    shape = (100, 100, 100)
    near = Envelope((0, 0, 0), (100, 100, 98), shape)     # crops 2% -> below the 5% default
    assert 0.97 < near.fraction < 0.99
    assert worth_cropping(near).is_whole()


def test_worth_cropping_keeps_a_real_crop():
    from nnseg.envelope import worth_cropping
    shape = (100, 100, 100)
    real = Envelope((0, 0, 0), (100, 100, 70), shape)     # crops 30%
    out = worth_cropping(real)
    assert out is real and not out.is_whole()


def test_worth_cropping_passes_through_whole_and_respects_min_saving():
    from nnseg.envelope import worth_cropping
    shape = (10, 10, 10)
    assert worth_cropping(Envelope((0, 0, 0), shape, shape)).is_whole()   # already whole
    box = Envelope((0, 0, 0), (10, 10, 8), shape)                          # exactly 20% saving
    assert not worth_cropping(box, min_saving=0.05).is_whole()             # 20% >= 5% -> kept
    assert worth_cropping(box, min_saving=0.25).is_whole()                 # 20% < 25% -> collapsed
