"""Normalization is per model, and several models may share one resample.

nnU-Net's CT normalization clips to each dataset's own foreground percentiles and z-scores by
its own foreground mean and std. Two models at the same spacing therefore need different
normalization of the same resampled array. The parts of `total` differ sharply - the organs
model clips at +276 HU where the ribs model clips at +1302 - so sharing a *normalized* array
runs parts 2..N on part 1's statistics and flattens every bone above that first clip. These
tests pin the split that keeps the resample shared and the normalization not.
"""
import numpy as np
import pytest

pytest.importorskip("nnseg")
pytest.importorskip("nnunetv2")
from nnseg.preprocess import (normalization_fingerprint, normalize, normalize_for,
                              to_model_grid)
from nnseg.values import Geometry

SPACING = (1.5, 1.5, 1.5)


class _Model:
    """Only what resampling and normalization actually read off a model."""

    def __init__(self, props, scheme="CTNormalization", use_mask=False):
        self.spacing_zyx = SPACING
        self.normalization_schemes = (scheme,)
        self.use_mask_for_norm = (use_mask,)
        self._props = props

    def intensity_properties(self, channel):
        return dict(self._props)


def _ct_model(mean, std, lo, hi):
    return _Model({"mean": mean, "std": std, "percentile_00_5": lo, "percentile_99_5": hi})


# the real spread, read from the installed plans of total's five parts
ORGANS = _ct_model(-370.0, 436.6, -1024.0, 276.0)
RIBS = _ct_model(292.0, 261.5, -110.0, 1302.0)


def _ct(shape=(16, 20, 18)):
    hu = np.full(shape, -1000.0, dtype=np.float32)
    hu[3:13, 5:15, 4:14] = 40.0          # soft tissue
    hu[5:9, 7:11, 6:10] = 400.0          # trabecular bone / dense cartilage
    hu[6:8, 8:10, 7:9] = 900.0           # cortical bone
    return hu


def _grid(shape=(16, 20, 18)):
    hu = _ct(shape)
    geometry = Geometry(spacing_zyx=SPACING, shape_zyx=shape, origin_xyz=(0.0, 0.0, 0.0),
                        direction_xyz=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    return hu, to_model_grid(hu, geometry, SPACING, convention="corner", device="cpu")


def test_the_shared_grid_carries_no_normalization():
    """What is cached must be the resample and nothing else - a normalized array is the one
    thing two models cannot share."""
    hu, grid = _grid()
    assert grid.data_zyx.min() == pytest.approx(hu.min())
    assert grid.data_zyx.max() == pytest.approx(hu.max())


def test_each_model_normalizes_the_shared_grid_with_its_own_statistics():
    _, grid = _grid()
    organs = normalize_for(grid, ORGANS).numpy()[0]
    ribs = normalize_for(grid, RIBS).numpy()[0]
    assert not np.allclose(organs, ribs)
    np.testing.assert_allclose(organs, normalize(grid.data_zyx, ORGANS.normalization_schemes,
                                                 ORGANS.intensity_properties(0),
                                                 use_mask_for_norm=ORGANS.use_mask_for_norm))
    np.testing.assert_allclose(ribs, normalize(grid.data_zyx, RIBS.normalization_schemes,
                                               RIBS.intensity_properties(0),
                                               use_mask_for_norm=RIBS.use_mask_for_norm))


def test_bone_keeps_its_contrast_for_the_model_whose_clip_admits_it():
    """The defect in one assertion. The organs model clips at +276 HU, so every bone density
    above it collapses to a single value and cortical bone becomes indistinguishable from
    trabecular; the ribs model, clipping at +1302, keeps them apart. Running parts 2..N on
    organs statistics is exactly this loss, everywhere in the volume."""
    _, grid = _grid()
    cortical, trabecular = (6, 8, 7), (5, 7, 6)
    organs = normalize_for(grid, ORGANS).numpy()[0]
    ribs = normalize_for(grid, RIBS).numpy()[0]
    assert organs[cortical] == pytest.approx(organs[trabecular], abs=1e-6)   # both at the clip
    assert ribs[cortical] > ribs[trabecular] + 1.0                           # still separable


def test_normalizing_does_not_disturb_the_shared_grid():
    """Two models normalize the same grid in sequence; the second must see the first's input,
    not its output."""
    _, grid = _grid()
    before = grid.data_zyx.copy()
    first = normalize_for(grid, ORGANS).numpy()[0].copy()
    normalize_for(grid, RIBS)
    np.testing.assert_array_equal(grid.data_zyx, before)
    np.testing.assert_array_equal(normalize_for(grid, ORGANS).numpy()[0], first)


def test_the_input_is_stamped_with_the_normalization_it_received():
    """The stamp is what lets the consumer refuse an input meant for another model."""
    _, grid = _grid()
    assert normalize_for(grid, ORGANS)._nnseg_normalization == normalization_fingerprint(ORGANS)
    assert normalize_for(grid, RIBS)._nnseg_normalization != normalization_fingerprint(ORGANS)


def test_fingerprints_separate_models_that_normalize_differently():
    assert normalization_fingerprint(ORGANS) != normalization_fingerprint(RIBS)


def test_fingerprints_match_for_models_that_normalize_identically():
    """The tripwire must not fire on a task that legitimately runs one model twice."""
    a = _ct_model(-370.0, 436.6, -1024.0, 276.0)
    b = _ct_model(-370.0, 436.6, -1024.0, 276.0)
    assert normalization_fingerprint(a) == normalization_fingerprint(b)


def test_zscore_models_without_statistics_are_interchangeable():
    """Why total_mr was never harmed: ZScoreNormalization is a per-image transform reading no
    dataset statistics, so its parts genuinely do produce the same array."""
    a = _Model({}, scheme="ZScoreNormalization")
    b = _Model({}, scheme="ZScoreNormalization")
    assert normalization_fingerprint(a) == normalization_fingerprint(b)
    _, grid = _grid()
    np.testing.assert_array_equal(normalize_for(grid, a).numpy(), normalize_for(grid, b).numpy())
