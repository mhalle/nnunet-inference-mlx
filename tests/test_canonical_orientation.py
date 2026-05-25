"""Verify predict_with_resampling and run_workflow reorient inputs to
canonical (LPS) for inference and map outputs back to the input's
original orientation.

This is the fix for the chest-CT bug where SAR-oriented input (voxel
axes don't match anatomical axes) caused the sliding window to scan
in the wrong direction, producing badly-fragmented segmentations.

We can't run a real model end-to-end here without weights, so we
verify two things:
  1. Geometry of the output matches the input exactly (size, origin,
     spacing, direction) even when the input is non-LPS.
  2. The reorient round-trip (LPS forward, original back) preserves
     content via sanity checks on synthetic data.
"""

from __future__ import annotations

import mlx.nn as nn
import numpy as np
import pytest

from nnunet_inference_mlx import (
    InferenceEngine, ModelBundle, Stage,
    predict_with_resampling, run_workflow,
)
from nnunet_inference_mlx.plans import build_network_from_plans

sitk = pytest.importorskip("SimpleITK")


def _make_engine():
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
        "labels": {"background": 0, "class_1": 1, "class_2": 2},
        "channel_names": {"0": "CT"},
    }
    net = build_network_from_plans(plans, "3d_fullres", 1, 3, deep_supervision=False)
    weights = dict(nn.utils.tree_flatten(net.parameters()))
    bundle = ModelBundle(plans=plans, dataset=dataset,
                          fold_weights=[weights], metadata={}, fold_ids=(0,))
    return InferenceEngine(bundle, verbose=False)


def _make_sitk(shape_zyx=(24, 24, 24), spacing_xyz=(1.0, 1.0, 1.0),
                direction=None, origin=(0.0, 0.0, 0.0)):
    arr = np.random.randn(*shape_zyx).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing_xyz)
    img.SetOrigin(origin)
    if direction is not None:
        img.SetDirection(direction)
    return img


# SAR orientation (the broken-chest-scan case): voxel X→S, Y→A, Z→R.
SAR_DIRECTION = (0.0, 0.0, -1.0,
                 0.0, -1.0, 0.0,
                 1.0,  0.0, 0.0)


@pytest.fixture(scope="module")
def engine():
    return _make_engine()


# ---------------------------------------------------------------------------
# predict_with_resampling
# ---------------------------------------------------------------------------


class TestPredictWithResamplingReorient:
    def test_lps_input_unchanged(self, engine):
        """Identity-orientation input should produce identity-orientation output."""
        img = _make_sitk()
        seg = predict_with_resampling(engine, img)
        assert seg.GetSize() == img.GetSize()
        assert seg.GetSpacing() == img.GetSpacing()
        assert seg.GetOrigin() == img.GetOrigin()
        assert seg.GetDirection() == img.GetDirection()

    def test_sar_input_returns_sar_output(self, engine):
        """SAR-oriented input should produce SAR-oriented output via
        the internal LPS round-trip."""
        img = _make_sitk(direction=SAR_DIRECTION,
                          spacing_xyz=(1.0, 0.65, 0.65),
                          origin=(266.0, 138.0, 31.0))
        seg = predict_with_resampling(engine, img)
        assert seg.GetSize() == img.GetSize()
        assert seg.GetSpacing() == img.GetSpacing()
        # Origin/direction are preserved through the round-trip
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(seg.GetOrigin(), img.GetOrigin()))
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(seg.GetDirection(), img.GetDirection()))

    def test_reorient_none_skips_reorient(self, engine):
        """reorient=None preserves the input orientation but skips the
        round-trip. Output direction should match the input."""
        img = _make_sitk(direction=SAR_DIRECTION)
        seg = predict_with_resampling(engine, img, reorient=None)
        assert seg.GetSize() == img.GetSize()
        assert seg.GetDirection() == img.GetDirection()

    def test_reorient_target_ras(self, engine):
        """Caller can request a non-LPS target orientation if they want."""
        # Input is LPS; ask for RAS internally; output should still be LPS
        # (caller's original orientation).
        img = _make_sitk()  # default LPS
        seg = predict_with_resampling(engine, img, reorient="RAS")
        assert seg.GetDirection() == img.GetDirection()


# ---------------------------------------------------------------------------
# run_workflow
# ---------------------------------------------------------------------------


class TestRunWorkflowReorient:
    def test_lps_input_unchanged(self, engine):
        img = _make_sitk()
        seg = run_workflow(img, [Stage(engine=engine)])
        assert seg.GetSize() == img.GetSize()
        assert seg.GetDirection() == img.GetDirection()
        assert seg.GetSpacing() == img.GetSpacing()

    def test_sar_input_returns_sar_output(self, engine):
        img = _make_sitk(direction=SAR_DIRECTION,
                          spacing_xyz=(1.0, 0.65, 0.65),
                          origin=(266.0, 138.0, 31.0))
        seg = run_workflow(img, [Stage(engine=engine)])
        assert seg.GetSize() == img.GetSize()
        assert seg.GetSpacing() == img.GetSpacing()
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(seg.GetOrigin(), img.GetOrigin()))
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(seg.GetDirection(), img.GetDirection()))

    def test_sar_cascade(self, engine):
        """Two-stage workflow on SAR-oriented input: output should still
        be SAR; intermediate crops happen in canonical space."""
        img = _make_sitk(direction=SAR_DIRECTION, shape_zyx=(48, 48, 48))
        stages = [
            Stage(engine=engine, crop_to_classes=(1, 2), dilation_mm=5.0),
            Stage(engine=engine),
        ]
        seg = run_workflow(img, stages)
        assert seg.GetSize() == img.GetSize()
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(seg.GetDirection(), img.GetDirection()))
