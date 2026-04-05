"""
Test ModelBundle + InferenceEngine.

    uv run python tests/test_engine.py
"""

import json
import time

import mlx.core as mx
import numpy as np

from nnunet_mlx import InferenceEngine, ModelBundle
from nnunet_mlx.inference import predict_sliding_window
from nnunet_mlx.plans import build_network_from_plans
from nnunet_mlx.weights import load_model_weights, fuzzy_load_weights


def run_existing_pipeline(data_zyx, plans, dataset, weights):
    """Run the existing pipeline for comparison. Returns logits (K, Z, Y, X)."""
    from nnunet_mlx.preprocessing import preprocess_volume

    config = plans["configurations"]["3d_fullres"]
    patch_size = tuple(config["patch_size"])
    num_classes = len(dataset["labels"])
    num_channels = len(dataset.get("channel_names", dataset.get("modality", {})))

    network = build_network_from_plans(
        plans, "3d_fullres", num_channels, num_classes, deep_supervision=False
    )
    try:
        network.load_weights(list(weights.items()))
    except Exception:
        fuzzy_load_weights(network, weights)
    compiled = mx.compile(network)

    dummy = mx.random.normal((1, *patch_size, num_channels))
    mx.eval(compiled(dummy))
    del dummy

    # Existing pipeline: (Z,Y,X) → (X,Y,Z) for preprocess → (C,Z,Y,X)
    data_xyz = data_zyx.transpose(2, 1, 0)
    preprocessed = preprocess_volume(data_xyz, plans, "3d_fullres")
    preprocessed = preprocessed.transpose(0, 3, 2, 1).copy()

    return predict_sliding_window(
        network=compiled, input_image=preprocessed,
        patch_size=patch_size, num_classes=num_classes,
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        batch_size=1, use_fp16=False, verbose=False,
    )


def test_from_task():
    """Test ModelBundle.from_task() + InferenceEngine."""
    print("=" * 60)
    print("Test 1: ModelBundle.from_task() + InferenceEngine")
    print("=" * 60)

    np.random.seed(42)
    vol_zyx = np.random.uniform(-200, 200, (130, 120, 120)).astype(np.float32)

    # New API
    print("  Loading bundle...")
    t0 = time.perf_counter()
    bundle = ModelBundle.from_task(297)
    print(f"  Bundle loaded: {time.perf_counter() - t0:.2f}s")

    print("  Creating engine...")
    t0 = time.perf_counter()
    engine = InferenceEngine(bundle, verbose=True)
    print(f"  Engine ready: {time.perf_counter() - t0:.2f}s")

    print("  Running predict...")
    t0 = time.perf_counter()
    logits_engine = engine.predict(vol_zyx)
    print(f"  Predict: {time.perf_counter() - t0:.1f}s")

    # Existing pipeline (reuse the same weights)
    print("\n  Running existing pipeline...")
    logits_existing = run_existing_pipeline(
        vol_zyx, bundle.plans, bundle.dataset, bundle.weights
    )

    diff = np.abs(logits_engine - logits_existing)
    print(f"\n  Max diff: {diff.max():.8f}")
    print(f"  Exact match: {np.array_equal(logits_engine, logits_existing)}")
    ok = diff.max() < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_from_folder():
    """Test ModelBundle.from_folder()."""
    print("\n" + "=" * 60)
    print("Test 2: ModelBundle.from_folder()")
    print("=" * 60)

    bundle_task = ModelBundle.from_task(297)
    # Resolve the folder path manually
    from nnunet_mlx.engine import _default_weights_dir, _find_model_folder
    folder = _find_model_folder(297, _default_weights_dir())
    bundle_folder = ModelBundle.from_folder(folder)

    # Same plans and dataset
    ok = (bundle_task.plans == bundle_folder.plans
          and bundle_task.dataset == bundle_folder.dataset
          and set(bundle_task.weights.keys()) == set(bundle_folder.weights.keys()))
    print(f"  Plans match: {bundle_task.plans == bundle_folder.plans}")
    print(f"  Dataset match: {bundle_task.dataset == bundle_folder.dataset}")
    print(f"  Weight keys match: {set(bundle_task.weights.keys()) == set(bundle_folder.weights.keys())}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_normalize_separate():
    """Test normalize + predict(normalize=False)."""
    print("\n" + "=" * 60)
    print("Test 3: Separate normalize")
    print("=" * 60)

    np.random.seed(42)
    vol = np.random.uniform(-200, 200, (130, 120, 120)).astype(np.float32)

    bundle = ModelBundle.from_task(297)
    engine = InferenceEngine(bundle)

    logits_a = engine.predict(vol, normalize=True)
    normalized = engine.normalize(vol)
    logits_b = engine.predict(normalized, normalize=False)

    ok = np.array_equal(logits_a, logits_b)
    print(f"  Exact match: {ok}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_prepare_cache():
    """Test shape caching."""
    print("\n" + "=" * 60)
    print("Test 4: Shape caching")
    print("=" * 60)

    bundle = ModelBundle.from_task(297)
    engine = InferenceEngine(bundle)

    ctx1 = engine.prepare((130, 120, 120))
    ctx2 = engine.prepare((130, 120, 120))
    ok = ctx1 is ctx2
    print(f"  Same object: {ok}")
    print(f"  Patches: {ctx1.n_patches}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = [
        test_from_task(),
        test_from_folder(),
        test_normalize_separate(),
        test_prepare_cache(),
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
