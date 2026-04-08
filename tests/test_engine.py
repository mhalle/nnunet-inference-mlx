"""
Test ModelBundle + InferenceEngine with synthetic weights.

    uv run python tests/test_engine.py
"""

import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from nnunet_inference_mlx import InferenceEngine, ModelBundle
from nnunet_inference_mlx.model import PlainConvUNet
from nnunet_inference_mlx.inference import predict_sliding_window


def make_synthetic_bundle(num_classes=4):
    """Create a ModelBundle with random weights for testing."""
    plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [32, 32, 32],
                "normalization_schemes": ["ZScoreNormalization"],
                "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "n_conv_per_stage_encoder": [2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2],
                "UNet_base_num_features": 8,
            }
        },
        "foreground_intensity_properties_per_channel": {},
    }
    dataset = {
        "labels": {str(i): f"class_{i}" for i in range(num_classes)},
        "channel_names": {"0": "CT"},
    }

    # Build the same network the plans parser would build,
    # then extract its random weights
    from nnunet_inference_mlx.plans import build_network_from_plans
    network = build_network_from_plans(
        plans, "3d_fullres", 1, num_classes, deep_supervision=False
    )

    weights = dict(nn.utils.tree_flatten(network.parameters()))
    return ModelBundle(plans=plans, dataset=dataset, weights=weights)


def test_predict():
    """Test that predict produces correct shape output."""
    print("=" * 60)
    print("Test 1: Predict shape and dtype")
    print("=" * 60)

    bundle = make_synthetic_bundle(num_classes=4)
    engine = InferenceEngine(bundle, verbose=True)

    vol = np.random.randn(40, 40, 40).astype(np.float32)
    logits = engine.predict(vol, normalize=True)

    print(f"  Output shape: {logits.shape} (expected (4, 40, 40, 40))")
    print(f"  Dtype: {logits.dtype}")
    assert logits.shape == (4, 40, 40, 40)
    assert logits.dtype == np.float32


def test_normalize_separate():
    """Test normalize + predict(normalize=False) matches predict()."""
    print("\n" + "=" * 60)
    print("Test 2: Separate normalize")
    print("=" * 60)

    bundle = make_synthetic_bundle(num_classes=4)
    engine = InferenceEngine(bundle)

    vol = np.random.randn(40, 40, 40).astype(np.float32)
    logits_a = engine.predict(vol, normalize=True)
    normalized = engine.normalize(vol)
    logits_b = engine.predict(normalized, normalize=False)

    print(f"  Exact match: {np.array_equal(logits_a, logits_b)}")
    assert np.array_equal(logits_a, logits_b)


def test_prepare_cache():
    """Test shape caching."""
    print("\n" + "=" * 60)
    print("Test 3: Shape caching")
    print("=" * 60)

    bundle = make_synthetic_bundle(num_classes=4)
    engine = InferenceEngine(bundle)

    ctx1 = engine.prepare((40, 40, 40))
    ctx2 = engine.prepare((40, 40, 40))
    print(f"  Same object: {ctx1 is ctx2}")
    print(f"  Patches: {ctx1.n_patches}")
    assert ctx1 is ctx2


def test_deterministic():
    """Test that two runs produce identical output."""
    print("\n" + "=" * 60)
    print("Test 4: Deterministic output")
    print("=" * 60)

    bundle = make_synthetic_bundle(num_classes=4)
    engine = InferenceEngine(bundle)

    vol = np.random.randn(40, 40, 40).astype(np.float32)
    logits_a = engine.predict(vol, normalize=False)
    logits_b = engine.predict(vol, normalize=False)

    print(f"  Exact match: {np.array_equal(logits_a, logits_b)}")
    assert np.array_equal(logits_a, logits_b)


def main():
    """Script entry point: run each test, catch AssertionError, summarize."""
    np.random.seed(42)
    tests = [test_predict, test_normalize_separate, test_prepare_cache, test_deterministic]
    failures = []
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, e))
            print(f"  FAIL: {e}")

    print("\n" + "=" * 60)
    if not failures:
        print("All tests PASSED")
    else:
        print(f"{len(failures)} tests FAILED: {[name for name, _ in failures]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
