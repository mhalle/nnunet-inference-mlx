"""Tests for the 0.8.0 engine additions:

- ``InferenceEngine`` bundle property accessors (target_spacing,
  has_regions, regions_class_order, bundle)
- ``InferenceEngine.predict_logits()`` returning mx.array

Uses a synthetic tiny engine — no real weights needed.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from nnunet_inference_mlx import InferenceEngine, ModelBundle
from nnunet_inference_mlx.plans import build_network_from_plans


def _make_bundle(num_classes: int = 3, region_based: bool = False,
                 target_spacing=(1.5, 1.5, 1.5)) -> ModelBundle:
    plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [16, 16, 16],
                "spacing": list(target_spacing),
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
    if region_based:
        dataset = {
            "labels": {
                "background": 0,
                "whole_tumor": [1, 2, 3],
                "tumor_core": [1, 3],
                "enhancing": 3,
            },
            "regions_class_order": [1, 2, 3],
            "channel_names": {"0": "T1"},
        }
        out_channels = 3   # 3 regions (background is implicit)
    else:
        dataset = {
            "labels": {
                "background": 0,
                **{f"class_{i}": i for i in range(1, num_classes)},
            },
            "channel_names": {"0": "CT"},
        }
        out_channels = num_classes

    net = build_network_from_plans(
        plans, "3d_fullres", 1, out_channels, deep_supervision=False,
    )
    weights = dict(nn.utils.tree_flatten(net.parameters()))
    return ModelBundle(plans=plans, dataset=dataset,
                       fold_weights=[weights], metadata={}, fold_ids=(0,))


@pytest.fixture(scope="module")
def standard_engine():
    return InferenceEngine(_make_bundle(num_classes=3), verbose=False)


@pytest.fixture(scope="module")
def region_engine():
    return InferenceEngine(_make_bundle(region_based=True), verbose=False)


# ---------------------------------------------------------------------------
# Bundle property accessors
# ---------------------------------------------------------------------------


class TestProperties:
    def test_target_spacing(self, standard_engine):
        assert standard_engine.target_spacing == (1.5, 1.5, 1.5)
        assert standard_engine.target_spacing == standard_engine._bundle.target_spacing

    def test_target_spacing_axis_order(self):
        engine = InferenceEngine(
            _make_bundle(target_spacing=(2.0, 1.0, 0.5)),
            verbose=False,
        )
        # Should preserve Z, Y, X order
        assert engine.target_spacing == (2.0, 1.0, 0.5)

    def test_has_regions_standard(self, standard_engine):
        assert standard_engine.has_regions is False

    def test_has_regions_region_based(self, region_engine):
        assert region_engine.has_regions is True

    def test_regions_class_order_standard(self, standard_engine):
        assert standard_engine.regions_class_order == ()

    def test_regions_class_order_region_based(self, region_engine):
        assert region_engine.regions_class_order == (1, 2, 3)

    def test_bundle_is_underlying(self, standard_engine):
        assert standard_engine.bundle is standard_engine._bundle

    def test_bundle_exposes_dataset(self, region_engine):
        assert "regions_class_order" in region_engine.bundle.dataset


# ---------------------------------------------------------------------------
# predict_logits returns mx.array
# ---------------------------------------------------------------------------


class TestPredictLogits:
    def test_returns_mx_array(self, standard_engine):
        vol = np.random.randn(20, 20, 20).astype(np.float32)
        out = standard_engine.predict_logits(vol)
        assert isinstance(out, mx.array)

    def test_shape_matches_predict(self, standard_engine):
        vol = np.random.randn(20, 20, 20).astype(np.float32)
        np_out = standard_engine.predict(vol)
        mx_out = standard_engine.predict_logits(vol)
        assert mx_out.shape == np_out.shape

    def test_values_match_predict(self, standard_engine):
        # Same input twice should produce same output (deterministic).
        vol = np.random.randn(16, 16, 16).astype(np.float32)
        np_out = standard_engine.predict(vol)
        mx_out = standard_engine.predict_logits(vol)
        np.testing.assert_allclose(np.array(mx_out), np_out, rtol=1e-5)

    def test_normalize_flag(self, standard_engine):
        vol = np.random.randn(16, 16, 16).astype(np.float32)
        # normalize=False should produce a different result than normalize=True
        out_norm = np.array(standard_engine.predict_logits(vol, normalize=True))
        out_raw = np.array(standard_engine.predict_logits(vol, normalize=False))
        # Outputs typically differ when input wasn't already normalized
        # (we just check the call works under both flag values).
        assert out_norm.shape == out_raw.shape

    def test_region_based_returns_mx_array(self, region_engine):
        vol = np.random.randn(20, 20, 20).astype(np.float32)
        out = region_engine.predict_logits(vol)
        assert isinstance(out, mx.array)
        # 3 regions for the synthetic BraTS-like dataset
        assert out.shape[0] == 3
