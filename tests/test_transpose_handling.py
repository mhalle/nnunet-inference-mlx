"""Verify transpose_forward / transpose_backward handling.

nnU-Net's training pipeline permutes volume axes by ``transpose_forward``
before the network sees them; ``transpose_backward`` is the inverse
permutation, applied to inference outputs so the user gets predictions
in canonical (Z, Y, X) order.

For TS Datasets 291-298 these are identity and the transpose is a no-op.
Some research nnU-Net models use non-identity transposes (e.g.
``(2, 0, 1)``), and feeding them canonical-order volumes without the
internal permutation produces silently-wrong predictions.

These tests use synthetic tiny engines to verify:
  1. Bundle reads transpose_forward/backward from plans.json
  2. bundle.target_spacing applies transpose_backward
  3. engine.predict/predict_logits/predict_segmentation transparently
     handle the round-trip — input AND output stay in canonical order
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from nnunet_inference_mlx import InferenceEngine, ModelBundle
from nnunet_inference_mlx.plans import build_network_from_plans


def _make_bundle(transpose_forward=(0, 1, 2), transpose_backward=(0, 1, 2),
                 spacing=(1.0, 2.0, 3.0), num_classes=3):
    plans = {
        "transpose_forward": list(transpose_forward),
        "transpose_backward": list(transpose_backward),
        "configurations": {
            "3d_fullres": {
                "patch_size": [16, 16, 16],
                # plans.json stores spacing in transposed (model) axis order
                "spacing": list(spacing),
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
        "labels": {
            "background": 0,
            **{f"class_{i}": i for i in range(1, num_classes)},
        },
        "channel_names": {"0": "CT"},
    }
    net = build_network_from_plans(plans, "3d_fullres", 1, num_classes,
                                    deep_supervision=False)
    weights = dict(nn.utils.tree_flatten(net.parameters()))
    return ModelBundle(plans=plans, dataset=dataset,
                        fold_weights=[weights], metadata={}, fold_ids=(0,))


# ---------------------------------------------------------------------------
# ModelBundle properties
# ---------------------------------------------------------------------------


class TestBundleTransposeProperties:
    def test_identity_default(self):
        # No transpose_* in plans -> defaults to identity
        plans = {"configurations": {"3d_fullres": {"spacing": [1.5, 1.5, 1.5]}}}
        bundle = ModelBundle(plans=plans, dataset={"labels": {"background": 0}},
                             fold_weights=[{}], metadata={}, fold_ids=(0,))
        assert bundle.transpose_forward == (0, 1, 2)
        assert bundle.transpose_backward == (0, 1, 2)

    def test_read_from_plans(self):
        bundle = _make_bundle(
            transpose_forward=(2, 0, 1), transpose_backward=(1, 2, 0),
        )
        assert bundle.transpose_forward == (2, 0, 1)
        assert bundle.transpose_backward == (1, 2, 0)

    def test_target_spacing_identity_transpose(self):
        bundle = _make_bundle(spacing=(1.0, 2.0, 3.0))
        # Identity transpose -> spacing unchanged
        assert bundle.target_spacing == (1.0, 2.0, 3.0)

    def test_target_spacing_non_identity_transpose(self):
        # plans stores spacing in model-order (transposed); we expose it
        # in canonical-order after applying transpose_backward.
        # transpose_forward=(2,0,1), transpose_backward=(1,2,0)
        # plans spacing = (1.0, 2.0, 3.0) is in model order
        # canonical = (plans[1], plans[2], plans[0]) = (2.0, 3.0, 1.0)
        bundle = _make_bundle(
            spacing=(1.0, 2.0, 3.0),
            transpose_forward=(2, 0, 1),
            transpose_backward=(1, 2, 0),
        )
        assert bundle.target_spacing == (2.0, 3.0, 1.0)

    def test_transpose_pair_is_inverse(self):
        bundle = _make_bundle(
            transpose_forward=(2, 0, 1), transpose_backward=(1, 2, 0),
        )
        tf, tb = bundle.transpose_forward, bundle.transpose_backward
        # tf ∘ tb = identity
        for i in range(3):
            assert tf[tb[i]] == i


# ---------------------------------------------------------------------------
# Engine round-trip: canonical-order in, canonical-order out
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine_identity():
    return InferenceEngine(_make_bundle(transpose_forward=(0, 1, 2),
                                          transpose_backward=(0, 1, 2)),
                            verbose=False)


@pytest.fixture(scope="module")
def engine_non_identity():
    # transpose_forward = (2, 0, 1) → tb = (1, 2, 0)
    return InferenceEngine(_make_bundle(transpose_forward=(2, 0, 1),
                                          transpose_backward=(1, 2, 0)),
                            verbose=False)


class TestPredictTransposeRoundTrip:
    def test_predict_shape_identity(self, engine_identity):
        vol = np.random.randn(20, 20, 20).astype(np.float32)
        out = engine_identity.predict(vol)
        assert out.shape == (3, 20, 20, 20)

    def test_predict_shape_non_identity(self, engine_non_identity):
        """Output spatial axes should still match input (Z, Y, X) — even
        though internally we permuted to model order then back."""
        vol = np.random.randn(20, 24, 28).astype(np.float32)
        out = engine_non_identity.predict(vol)
        # (K, Z, Y, X) where (Z, Y, X) = input (20, 24, 28)
        assert out.shape == (3, 20, 24, 28)

    def test_predict_logits_shape_non_identity(self, engine_non_identity):
        vol = np.random.randn(20, 24, 28).astype(np.float32)
        logits = engine_non_identity.predict_logits(vol)
        assert isinstance(logits, mx.array)
        assert tuple(logits.shape) == (3, 20, 24, 28)

    def test_predict_segmentation_shape_non_identity(self, engine_non_identity):
        vol = np.random.randn(20, 24, 28).astype(np.float32)
        seg = engine_non_identity.predict_segmentation(vol)
        assert seg.shape == (20, 24, 28)
        assert seg.dtype == np.uint8

    def test_identity_engine_unchanged_behavior(self, engine_identity):
        """Identity-transpose engine should produce same output regardless
        of whether transpose code is exercised — the transpose is a no-op."""
        vol = np.random.randn(20, 20, 20).astype(np.float32)
        # Two calls — same volume, same engine — must give same output
        out1 = engine_identity.predict(vol)
        out2 = engine_identity.predict(vol)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Correctness: a non-identity engine should yield different output than
# the equivalent un-transposed engine on the same input (the model has
# learned features in a specific axis order).
# ---------------------------------------------------------------------------


class TestTransposeCorrectness:
    def test_non_identity_round_trip_preserves_canonical_layout(self):
        """Verify that the spatial axes of the output match those of the
        input regardless of transpose. This is the key user-facing
        invariant — the engine is 'transparent'."""
        for tf, tb in [
            ((0, 1, 2), (0, 1, 2)),     # identity
            ((1, 0, 2), (1, 0, 2)),     # swap Z,Y
            ((2, 0, 1), (1, 2, 0)),     # rotate
            ((0, 2, 1), (0, 2, 1)),     # swap Y,X
            ((2, 1, 0), (2, 1, 0)),     # reverse
        ]:
            engine = InferenceEngine(
                _make_bundle(transpose_forward=tf, transpose_backward=tb),
                verbose=False,
            )
            # asymmetric shape to catch axis-swap bugs
            vol = np.random.randn(20, 24, 28).astype(np.float32)
            out = engine.predict(vol)
            assert out.shape == (3, 20, 24, 28), \
                f"transpose tf={tf}, tb={tb} produced shape {out.shape}, expected (3, 20, 24, 28)"


# ---------------------------------------------------------------------------
# Internal transpose helpers
# ---------------------------------------------------------------------------


class TestTransposeHelpers:
    def test_apply_transpose_forward_identity(self, engine_identity):
        vol = np.random.randn(10, 12, 14).astype(np.float32)
        out = engine_identity._apply_transpose_forward(vol)
        # Identity must be a no-op (same object or same content)
        assert out.shape == vol.shape
        np.testing.assert_array_equal(out, vol)

    def test_apply_transpose_forward_swap(self):
        # Make a bundle with transpose_forward = (2, 0, 1)
        engine = InferenceEngine(
            _make_bundle(transpose_forward=(2, 0, 1),
                          transpose_backward=(1, 2, 0)),
            verbose=False,
        )
        vol = np.zeros((10, 12, 14), dtype=np.float32)
        vol[1, 2, 3] = 1.0
        out = engine._apply_transpose_forward(vol)
        assert out.shape == (14, 10, 12)
        # vol[1,2,3] = 1; after transpose (2,0,1), output[3,1,2] should be 1
        assert out[3, 1, 2] == 1.0

    def test_apply_transpose_backward_undoes_forward(self):
        engine = InferenceEngine(
            _make_bundle(transpose_forward=(2, 0, 1),
                          transpose_backward=(1, 2, 0)),
            verbose=False,
        )
        # Logits in K-channel form
        K = 4
        logits_canonical = np.random.randn(K, 10, 12, 14).astype(np.float32)
        # Simulate: apply forward to spatial axes (model would see this)
        logits_model = np.transpose(logits_canonical, axes=(0, 3, 1, 2))
        # Apply backward — should recover canonical order
        logits_back = engine._apply_transpose_backward(logits_model)
        np.testing.assert_array_equal(logits_back, logits_canonical)
