"""
Test InferenceEngine against existing pipeline.

Verifies identical logits on synthetic and real volumes.

    uv run python tests/test_engine.py
"""

import json
import time

import mlx.core as mx
import numpy as np

from nnunet_mlx import InferenceEngine
from nnunet_mlx.inference import predict_sliding_window
from nnunet_mlx.plans import build_network_from_plans
from nnunet_mlx.predict import find_model_folder
from nnunet_mlx.preprocessing import preprocess_volume
from nnunet_mlx.weights import load_model_weights, fuzzy_load_weights


def run_existing_pipeline(data_zyx, task_id=297):
    """Run the existing file-based pipeline on a (Z, Y, X) volume.
    Returns logits (K, Z, Y, X)."""
    model_folder = find_model_folder(
        task_id, trainer="nnUNetTrainer_4000epochs_NoMirroring"
    )
    plans = json.loads((model_folder / "plans.json").read_text())
    dataset = json.loads((model_folder / "dataset.json").read_text())
    config = plans["configurations"]["3d_fullres"]
    patch_size = tuple(config["patch_size"])
    num_classes = len(dataset["labels"])
    num_channels = len(dataset.get("channel_names", dataset.get("modality", {})))

    network = build_network_from_plans(
        plans, "3d_fullres", num_channels, num_classes, deep_supervision=False
    )
    weights = load_model_weights(model_folder, fold=0)
    try:
        network.load_weights(list(weights.items()))
    except Exception:
        fuzzy_load_weights(network, weights)
    compiled = mx.compile(network)

    dummy = mx.random.normal((1, *patch_size, num_channels))
    mx.eval(compiled(dummy))
    del dummy

    # Preprocess: the existing pipeline expects (X, Y, Z) nibabel order,
    # then transposes to (C, Z, Y, X). Our data is already (Z, Y, X),
    # so we need to go (Z, Y, X) → (X, Y, Z) for preprocess_volume,
    # then it gets transposed back.
    data_xyz = data_zyx.transpose(2, 1, 0)
    preprocessed = preprocess_volume(data_xyz, plans, "3d_fullres")
    preprocessed = preprocessed.transpose(0, 3, 2, 1).copy()

    logits = predict_sliding_window(
        network=compiled,
        input_image=preprocessed,
        patch_size=patch_size,
        num_classes=num_classes,
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        batch_size=1,
        use_fp16=False,
        verbose=False,
    )
    return logits


def test_synthetic():
    """Test on synthetic volume."""
    print("=" * 60)
    print("Test 1: Synthetic volume")
    print("=" * 60)

    np.random.seed(42)
    # Create in (Z, Y, X) order — the engine's expected input
    vol_zyx = np.random.uniform(-200, 200, (130, 120, 120)).astype(np.float32)

    # Engine path
    print("  Creating InferenceEngine...")
    t0 = time.perf_counter()
    engine = InferenceEngine(task_id=297, verbose=True)
    print(f"  Engine init: {time.perf_counter() - t0:.1f}s")

    print("  Running engine.predict()...")
    t0 = time.perf_counter()
    logits_engine = engine.predict(vol_zyx)
    dt_engine = time.perf_counter() - t0
    print(f"  Engine predict: {dt_engine:.1f}s")
    print(f"  Output: {logits_engine.shape}, dtype={logits_engine.dtype}")

    # Existing pipeline path
    print("\n  Running existing pipeline...")
    t0 = time.perf_counter()
    logits_existing = run_existing_pipeline(vol_zyx)
    dt_existing = time.perf_counter() - t0
    print(f"  Existing pipeline: {dt_existing:.1f}s")

    # Compare
    diff = np.abs(logits_engine - logits_existing)
    seg_e = np.argmax(logits_engine, axis=0)
    seg_x = np.argmax(logits_existing, axis=0)
    match_pct = 100 * (seg_e == seg_x).mean()

    print(f"\n  Max diff: {diff.max():.8f}")
    print(f"  Mean diff: {diff.mean():.8f}")
    print(f"  Segmentation agreement: {match_pct:.2f}%")

    if diff.max() < 1e-3:
        print("  PASS")
    else:
        print(f"  FAIL — max diff {diff.max()}")

    return diff.max() < 1e-3


def test_api():
    """Test normalize + predict(normalize=False) matches predict()."""
    print("\n" + "=" * 60)
    print("Test 2: Separate normalize")
    print("=" * 60)

    np.random.seed(42)
    vol = np.random.uniform(-200, 200, (130, 120, 120)).astype(np.float32)

    engine = InferenceEngine(task_id=297)

    logits_a = engine.predict(vol, normalize=True)
    normalized = engine.normalize(vol)
    logits_b = engine.predict(normalized, normalize=False)

    diff = np.abs(logits_a - logits_b)
    print(f"  Max diff: {diff.max():.8f}")
    print(f"  Exact match: {np.array_equal(logits_a, logits_b)}")

    if np.array_equal(logits_a, logits_b):
        print("  PASS")
    else:
        print("  FAIL")

    return np.array_equal(logits_a, logits_b)


def test_prepare():
    """Test that prepare() caching works."""
    print("\n" + "=" * 60)
    print("Test 3: Shape caching with prepare()")
    print("=" * 60)

    engine = InferenceEngine(task_id=297)
    shape = (130, 120, 120)

    ctx1 = engine.prepare(shape)
    ctx2 = engine.prepare(shape)
    print(f"  Same object: {ctx1 is ctx2}")
    print(f"  Patches: {ctx1.n_patches}")
    print(f"  Padded shape: {ctx1.padded_shape}")
    print(f"  Needs padding: {ctx1.needs_padding}")

    if ctx1 is ctx2:
        print("  PASS")
    else:
        print("  FAIL")

    return ctx1 is ctx2


def main():
    results = []
    results.append(test_synthetic())
    results.append(test_api())
    results.append(test_prepare())

    print("\n" + "=" * 60)
    if all(results):
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
