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


def test_zscore_models_are_interchangeable_despite_differing_statistics():
    """Why total_mr was never harmed - and the fingerprint is a coarser thing than it looks.

    ZScoreNormalization is a per-image transform that reads no dataset statistics, so two such
    models produce the same array whatever their plans say. Real total_mr parts DO carry
    different foreground intensity properties (verified against the installed weights), so
    their fingerprints differ while their output does not. The fingerprint therefore answers
    "which model normalized this", not "what normalization was applied". That is deliberate:
    it can only over-trigger, and under-triggering is the direction that hurts."""
    a = _Model({"mean": -370.0, "std": 436.6}, scheme="ZScoreNormalization")
    b = _Model({"mean": 292.0, "std": 261.5}, scheme="ZScoreNormalization")
    assert normalization_fingerprint(a) != normalization_fingerprint(b)      # differ...
    _, grid = _grid()
    np.testing.assert_array_equal(normalize_for(grid, a).numpy(),
                                  normalize_for(grid, b).numpy())            # ...output does not


# --- the same property, through segment() ------------------------------------------------
#
# The tests above pin `normalize_for`. The defect was not there: it was in segment()'s cache,
# which shared an already-normalized tensor between the parts of a multi-model task. A test
# that never runs segment() would have passed throughout the bug's life, so this drives the
# real path with stub models - no weights, no network.

class _StubModel(_Model):
    """A model just real enough for segment(): it records the input it is asked to predict on."""

    def __init__(self, props, K=3):
        super().__init__(props)
        self.K = K
        self.transpose_forward = (0, 1, 2)
        self.accumulate_choice = {"on_device": False}
        self.received = None

    def predict_logits(self, crop, report=None):
        import torch
        self.received = crop.clone()
        logits = torch.zeros((self.K, *crop.shape[1:]), dtype=torch.float32)
        logits[0] = 1.0                       # everything background; labels are not the point
        return logits


def _two_part_task(tmp_path, parts):
    """A union task over two models at one spacing, with stub weights and model caches."""
    from nnseg.tasks import TaskSpec, UnionPart
    folder = tmp_path / "Dataset000_stub" / "trainer__plans__3d_fullres"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "fold_0").mkdir(exist_ok=True)      # provenance reads available_folds

    class _Store:
        root = folder.parent.parent

        def resolve(self, weights_id, *, configuration=None):
            return folder

        def describe(self, *a, **k):
            return {}

    class _Cache:
        def __init__(self):
            self.order = list(parts)

        def get(self, folder, **kw):
            return self.order.pop(0)

        def release(self, model):
            pass

    spec = TaskSpec(name="stub_union", shape="union",
                    union=(UnionPart(weights_id=1, label_remap={1: 1}, name="first"),
                           UnionPart(weights_id=2, label_remap={1: 2}, name="second")),
                    label_map={1: "a", 2: "b"})
    return spec, _Store(), _Cache()


def _write_ct(tmp_path, shape=(12, 14, 16)):
    import SimpleITK as sitk
    hu = _ct(shape)
    img = sitk.GetImageFromArray(hu.astype("int16"))
    img.SetSpacing((float(SPACING[2]), float(SPACING[1]), float(SPACING[0])))
    p = tmp_path / "ct.nii.gz"
    sitk.WriteImage(img, str(p))
    return p


def test_segment_gives_each_part_of_a_multi_model_task_its_own_normalization(tmp_path, monkeypatch):
    """The regression test for the defect itself.

    Two models at one spacing with different intensity statistics. Before the fix both parts
    were handed the tensor the first part normalized, so `received` would be identical.
    """
    pytest.importorskip("SimpleITK")
    from nnseg import pipeline

    organs, ribs = _StubModel(ORGANS._props), _StubModel(RIBS._props)
    spec, store, cache = _two_part_task(tmp_path, [organs, ribs])
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    pipeline.segment(str(_write_ct(tmp_path)), spec, models=cache, device="cpu",
                     envelope_mm=None, convention="corner", folds=(0,))

    assert organs.received is not None and ribs.received is not None
    a, b = organs.received.numpy()[0], ribs.received.numpy()[0]
    assert not np.allclose(a, b), "both parts received the same normalized volume"

    # Stronger than "they differ": the second part must have re-normalized the SAME underlying
    # volume with its own statistics. CT normalization is invertible wherever it did not clip,
    # so recover the HU from the first part's input and predict what the second should have got.
    # Comparing this way says nothing about orientation, which the pipeline is free to change.
    o, r = ORGANS.intensity_properties(0), RIBS.intensity_properties(0)
    hu = a * o["std"] + o["mean"]
    unclipped = (hu > o["percentile_00_5"] + 1) & (hu < o["percentile_99_5"] - 1)
    assert unclipped.any(), "nothing survived the first model's clip; the fixture proves nothing"
    expected = (np.clip(hu, r["percentile_00_5"], r["percentile_99_5"]) - r["mean"]) / r["std"]
    np.testing.assert_allclose(b[unclipped], expected[unclipped], atol=1e-4)


def test_segment_still_resamples_once_for_models_that_share_a_spacing(tmp_path, monkeypatch):
    """The cache has to keep earning its place: the fix splits what is cached, it does not
    resample per part."""
    pytest.importorskip("SimpleITK")
    from nnseg import pipeline

    calls = []
    real = pipeline.to_model_grid

    def counting(*a, **k):
        calls.append(a[2])                    # the target spacing
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "to_model_grid", counting)
    spec, store, cache = _two_part_task(tmp_path, [_StubModel(ORGANS._props), _StubModel(RIBS._props)])
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    pipeline.segment(str(_write_ct(tmp_path)), spec, models=cache, device="cpu",
                     envelope_mm=None, convention="corner", folds=(0,))
    assert len(calls) == 1, f"resampled {len(calls)} times for two models at one spacing"
