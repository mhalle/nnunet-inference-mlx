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


def _old_path(model, vol):
    from nnunet_inference_mlx.imageio import sitk_to_segmentation, volume_to_sitk
    from nnunet_inference_mlx.resampling import predict_with_resampling
    seg_sitk = predict_with_resampling(model._engine, volume_to_sitk(vol),
                                       reorient_to="LPS")
    return sitk_to_segmentation(seg_sitk, model.schema)


def _new_path(model, vol):
    mv, plan = to_model_frame(vol, model.model_data, reorient_to="LPS")
    pred = sliding_window(model, mv)
    return restore(pred, plan)


class TestDecompositionMatchesOracle:
    def test_canonical_volume(self):
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        seg_new = _new_path(m, vol)
        seg_old = _old_path(m, vol)
        np.testing.assert_array_equal(np.asarray(seg_new.data), np.asarray(seg_old.data))
        assert seg_new.geometry.shape_zyx == vol.geometry.shape_zyx
        assert seg_new.geometry.spacing_zyx == vol.geometry.spacing_zyx

    def test_reoriented_volume_roundtrips(self):
        # RAS direction (flips X and Y vs LPS) forces a real reorient round-trip.
        m = build_model(_make_model_data())
        vol = _volume((20, 24, 28), spacing=(2.0, 1.0, 1.5),
                      direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1))
        seg_new = _new_path(m, vol)
        seg_old = _old_path(m, vol)
        np.testing.assert_array_equal(np.asarray(seg_new.data), np.asarray(seg_old.data))
        assert seg_new.geometry.shape_zyx == vol.geometry.shape_zyx
        np.testing.assert_allclose(seg_new.geometry.origin_xyz, vol.geometry.origin_xyz,
                                   atol=1e-4)


class TestPredictionAndToLabels:
    def test_to_model_frame_lands_at_model_spacing(self):
        m = build_model(_make_model_data())
        mv, plan = to_model_frame(_volume((24, 24, 24)), m.model_data)
        assert mv.geometry.spacing_zyx == (1.5, 1.5, 1.5)
        assert plan.model_spacing_zyx == (1.5, 1.5, 1.5)
        assert plan.inference_orientation == "LPS"

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

    def test_restore_rejects_mismatched_prediction(self):
        m = build_model(_make_model_data())
        mv, plan = to_model_frame(_volume((24, 24, 24)), m.model_data)
        pred = sliding_window(m, mv)
        # corrupt the spacing binding token
        from dataclasses import replace
        bad = replace(plan, model_spacing_zyx=(9.0, 9.0, 9.0))
        with pytest.raises(ValueError, match="model_spacing"):
            restore(pred, bad)
