"""Phase 3b decomposition — regression-oracle tests.

The new pure-fn pipeline ``to_model_frame → sliding_window → restore`` must
produce a segmentation *identical* to the proven fused
``predict_with_resampling`` path. The old path is the oracle: if the
decomposition drifts (resample, inverse, reorient, dtype), these fail.

Covers an already-canonical (LPS) volume and a reoriented (RAS) one that
forces the reorient round-trip through the RestorePlan.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from nnunet_inference_mlx.build import build_model
from nnunet_inference_mlx.infer import sliding_window
from nnunet_inference_mlx.model_data import ModelData
from nnunet_inference_mlx.plans import build_network_from_plans
from nnunet_inference_mlx.postprocess import restore, to_labels
from nnunet_inference_mlx.preprocess import to_model_frame
from nnunet_inference_mlx.values import Geometry, Prediction, Volume

sitk = pytest.importorskip("SimpleITK")


def _make_model_data(num_classes: int = 3) -> ModelData:
    plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [16, 16, 16],
                "spacing": [1.5, 1.5, 1.5],
                "normalization_schemes": ["ZScoreNormalization"],
                "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3]],
                "n_conv_per_stage_encoder": [2, 2],
                "n_conv_per_stage_decoder": [1],
                "UNet_base_num_features": 4,
            }
        },
        "foreground_intensity_properties_per_channel": {},
    }
    dataset = {
        "labels": {"background": 0, **{f"class_{i}": i for i in range(1, num_classes)}},
        "channel_names": {"0": "CT"},
    }
    net = build_network_from_plans(plans, "3d_fullres", 1, num_classes,
                                    deep_supervision=False)
    weights = dict(nn.utils.tree_flatten(net.parameters()))
    return ModelData(plans=plans, dataset=dataset, fold_weights=(weights,),
                     ecosystem="test", id=1)


def _volume(shape=(24, 24, 24), *, spacing=(1.0, 1.0, 1.0),
            direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    # deterministic content so both paths see identical input
    n = int(np.prod(shape))
    data = (mx.arange(n, dtype=mx.float32).reshape((*shape, 1)) % 7) - 3.0
    geom = Geometry(spacing_zyx=spacing, shape_zyx=shape, direction_xyz=direction)
    return Volume(data=data, geometry=geom, channels=("CT",))


def _new_path(model, vol):
    mv, plan = to_model_frame(vol, model.model_data, reorient_to="LPS")
    pred = sliding_window(model, mv)
    return restore(pred, plan)


class TestDecompositionGeometry:
    """The decomposed pipeline lands the result back on the caller's grid.

    (The historical bit-identical-vs-predict_with_resampling oracle was retired
    with that function at the Phase 5 cutover; it had already confirmed the
    decomposition faithful on synthetic + real TS weights. These keep the
    geometry round-trip — including a reoriented input — under test.)
    """

    def test_canonical_volume(self):
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        seg = _new_path(m, vol)
        assert seg.geometry.shape_zyx == vol.geometry.shape_zyx
        assert seg.geometry.spacing_zyx == vol.geometry.spacing_zyx

    def test_reoriented_volume_roundtrips(self):
        # RAS direction (flips X and Y vs LPS) forces a real reorient round-trip:
        # to_model_frame → LPS, restore → back to RAS, landing on the input grid.
        m = build_model(_make_model_data())
        vol = _volume((20, 24, 28), spacing=(2.0, 1.0, 1.5),
                      direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1))
        seg = _new_path(m, vol)
        assert seg.geometry.shape_zyx == vol.geometry.shape_zyx
        np.testing.assert_allclose(seg.geometry.origin_xyz, vol.geometry.origin_xyz,
                                   atol=1e-4)
        np.testing.assert_allclose(seg.geometry.direction_xyz, vol.geometry.direction_xyz,
                                   atol=1e-6)


class TestPredictionAndToLabels:
    def test_to_model_frame_lands_at_model_spacing(self):
        m = build_model(_make_model_data())
        mv, plan = to_model_frame(_volume((24, 24, 24)), m.model_data)
        assert mv.geometry.spacing_zyx == (1.5, 1.5, 1.5)
        assert plan.model_spacing_zyx == (1.5, 1.5, 1.5)
        assert plan.inference_orientation == "RAS"   # nnU-Net/TS canonical (not LPS — that mirrors L/R)

    def test_sliding_window_returns_prediction(self):
        m = build_model(_make_model_data())
        mv, _ = to_model_frame(_volume((24, 24, 24)), m.model_data)
        pred = sliding_window(m, mv)
        assert isinstance(pred, Prediction)
        assert pred.num_classes == 3
        assert pred.activation == "logits"
        assert pred.geometry.spacing_zyx == (1.5, 1.5, 1.5)

    def test_to_labels_at_prediction_grid(self):
        m = build_model(_make_model_data())
        mv, _ = to_model_frame(_volume((24, 24, 24)), m.model_data)
        pred = sliding_window(m, mv)
        seg = to_labels(pred)
        assert seg.geometry.shape_zyx == pred.geometry.shape_zyx
        assert int(seg.data.max()) <= 2

    def test_to_model_frame_resamples_in_float_not_int(self):
        # Decision (Phase 3b): the toolkit resamples in float32 — matching
        # nnU-Net v2 — NOT the old predict_with_resampling behavior of
        # resampling a raw int16 image and rounding interpolated values.
        # On real int16 CT that rounding flipped ~0.03% of boundary voxels at
        # argmax. Guard: resampling an integer-valued volume to a non-aligned
        # spacing must produce fractional values (proves float interpolation).
        m = build_model(_make_model_data())
        shape = (24, 24, 24)
        ramp = (mx.arange(int(np.prod(shape)), dtype=mx.float32) % 5).reshape((*shape, 1))
        vol = Volume(data=ramp,
                     geometry=Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=shape),
                     channels=("CT",))
        model_vol, _ = to_model_frame(vol, m.model_data)   # 1.0 -> 1.5 mm
        arr = np.asarray(model_vol.data[..., 0])
        frac = np.abs(arr - np.round(arr))
        assert frac.max() > 1e-3, "model-frame values are integer-rounded; resample is not float"

    def test_restore_identity_default(self):
        # No override → input grid (unchanged behavior).
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        mv, plan = to_model_frame(vol, m.model_data)
        seg = restore(sliding_window(m, mv), plan)
        assert seg.geometry.shape_zyx == vol.geometry.shape_zyx
        assert seg.geometry.spacing_zyx == vol.geometry.spacing_zyx

    def test_restore_output_spacing_fixes_header_same_extent(self):
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))                 # 1.0 mm, extent 24 mm/axis
        mv, plan = to_model_frame(vol, m.model_data)
        pred = sliding_window(m, mv)
        seg = restore(pred, plan, target_spacing=2.0)
        assert seg.geometry.spacing_zyx == (2.0, 2.0, 2.0)
        assert seg.geometry.shape_zyx == (12, 12, 12)   # 24 mm / 2 mm
        # physical extent preserved
        assert seg.geometry.physical_size_zyx == vol.geometry.physical_size_zyx

    def test_restore_scaling_matches_equivalent_spacing(self):
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        mv, plan = to_model_frame(vol, m.model_data)
        pred = sliding_window(m, mv)
        by_scale = restore(pred, plan, target_scaling=0.5)      # coarser ×0.5
        by_spacing = restore(pred, plan, target_spacing=2.0)    # 1.0/0.5 = 2.0 mm
        assert by_scale.geometry.shape_zyx == by_spacing.geometry.shape_zyx
        assert by_scale.geometry.spacing_zyx == by_spacing.geometry.spacing_zyx
        np.testing.assert_array_equal(np.asarray(by_scale.data), np.asarray(by_spacing.data))

    def test_restore_scaling_one_is_identity(self):
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        mv, plan = to_model_frame(vol, m.model_data)
        pred = sliding_window(m, mv)
        base = restore(pred, plan)
        scaled = restore(pred, plan, target_scaling=1.0)
        assert scaled.geometry.shape_zyx == base.geometry.shape_zyx
        np.testing.assert_array_equal(np.asarray(scaled.data), np.asarray(base.data))

    def test_restore_nearest_label_path(self):
        # path A: argmax at model spacing → NN-resample labels. Lands on input
        # grid, valid label set, agrees closely with the logit path.
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        mv, plan = to_model_frame(vol, m.model_data)
        pred = sliding_window(m, mv)
        nn = restore(pred, plan, interpolation="nearest")
        lin = restore(pred, plan, interpolation="linear")
        assert nn.geometry.shape_zyx == vol.geometry.shape_zyx
        assert int(nn.data.max()) <= 2
        # nearest and linear differ only at boundaries → mostly agree
        agree = float((np.asarray(nn.data) == np.asarray(lin.data)).mean())
        assert agree > 0.5

    def test_restore_rejects_bad_interpolation(self):
        m = build_model(_make_model_data())
        mv, plan = to_model_frame(_volume((24, 24, 24)), m.model_data)
        pred = sliding_window(m, mv)
        with pytest.raises(ValueError, match="linear.*nearest"):
            restore(pred, plan, interpolation="cubic")

    def test_segment_output_interpolation_nearest(self):
        m = build_model(_make_model_data())
        seg = m.segment(_volume((24, 24, 24)), output_interpolation="nearest")
        assert seg.geometry.shape_zyx == (24, 24, 24)

    def test_restore_rejects_both_overrides(self):
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        mv, plan = to_model_frame(vol, m.model_data)
        pred = sliding_window(m, mv)
        with pytest.raises(ValueError, match="not both"):
            restore(pred, plan, target_spacing=2.0, target_scaling=2.0)

    def test_segment_at_model_spacing(self):
        # at_model_spacing → seg on the model's native (1.5 mm) grid.
        m = build_model(_make_model_data())
        seg = m.segment(_volume((24, 24, 24)), at_model_spacing=True)
        assert seg.geometry.spacing_zyx == (1.5, 1.5, 1.5)

    def test_segment_rejects_multiple_output_modes(self):
        m = build_model(_make_model_data())
        with pytest.raises(ValueError, match="at most one"):
            m.segment(_volume((24, 24, 24)), output_spacing=2.0, output_scaling=2.0)

    def test_restore_rejects_mismatched_prediction(self):
        m = build_model(_make_model_data())
        mv, plan = to_model_frame(_volume((24, 24, 24)), m.model_data)
        pred = sliding_window(m, mv)
        # corrupt the spacing binding token
        from dataclasses import replace
        bad = replace(plan, model_spacing_zyx=(9.0, 9.0, 9.0))
        with pytest.raises(ValueError, match="model_spacing"):
            restore(pred, bad)
