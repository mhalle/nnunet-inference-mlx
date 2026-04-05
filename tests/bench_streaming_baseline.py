"""
Baseline benchmark for streaming accumulator work.

Measures predict_sliding_window with full logit accumulation on task 297
(117 classes, 128^3 patches) at batch=1. Records:
  - Wall time for sliding window
  - Peak memory (accumulator + model)
  - Patch count and volume shape

Run from nnunet-mlx/:
    python tests/bench_streaming_baseline.py

Requires Dataset297 weights in ~/.totalsegmentator/
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from nnunet_mlx.inference import (
    choose_batch_size,
    compute_gaussian,
    compute_sliding_window_steps,
    predict_sliding_window,
)
from nnunet_mlx.plans import build_network_from_plans
from nnunet_mlx.predict import find_model_folder
from nnunet_mlx.preprocessing import preprocess_volume
from nnunet_mlx.weights import load_model_weights, fuzzy_load_weights


def load_model(task_id=297, fold=0):
    """Load and compile the model, return (compiled_net, plans, metadata)."""
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

    weights = load_model_weights(model_folder, fold=fold)
    try:
        network.load_weights(list(weights.items()))
    except Exception:
        fuzzy_load_weights(network, weights)

    compiled = mx.compile(network)

    # Warmup
    dummy = mx.random.normal((1, *patch_size, num_channels))
    mx.eval(compiled(dummy))
    del dummy

    n_params = sum(v.size for _, v in nn.utils.tree_flatten(network.parameters()))
    param_mb = sum(v.nbytes for _, v in nn.utils.tree_flatten(network.parameters())) / 1e6

    return compiled, plans, {
        "patch_size": patch_size,
        "num_classes": num_classes,
        "num_channels": num_channels,
        "n_params": n_params,
        "param_mb": param_mb,
    }


def make_synthetic_volume(shape=(120, 120, 130)):
    """CT-like synthetic volume in (X, Y, Z) nibabel order."""
    data = np.random.uniform(-200, 200, shape).astype(np.float32)
    center = np.array(shape) // 2
    coords = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(sum((c - cn) ** 2 for c, cn in zip(coords, center)))
    data[dist < 20] = 400
    data[dist > 50] = -800
    return data


def benchmark_sliding_window(network, preprocessed, patch_size, num_classes,
                             batch_size, label=""):
    """Run predict_sliding_window and measure time + memory."""
    # Estimate memory usage
    spatial = preprocessed.shape[1:]
    accum_bytes = num_classes * np.prod(spatial) * 4  # float32
    weights_bytes = np.prod(spatial) * 4
    gaussian_bytes = np.prod(patch_size) * 4
    accum_mb = (accum_bytes + weights_bytes + gaussian_bytes) / 1e6

    steps = compute_sliding_window_steps(spatial, patch_size, 0.5)
    n_patches = len(steps[0]) * len(steps[1]) * len(steps[2])

    print(f"\n--- {label} ---")
    print(f"  Volume (after preprocess): {preprocessed.shape}")
    print(f"  Spatial: {spatial}, Patch: {patch_size}")
    print(f"  Patches: {n_patches}, Batch size: {batch_size}")
    print(f"  Accumulator: {accum_mb:.1f} MB")
    print(f"  Running...")

    t0 = time.perf_counter()
    logits = predict_sliding_window(
        network=network,
        input_image=preprocessed,
        patch_size=patch_size,
        num_classes=num_classes,
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        batch_size=batch_size,
        use_fp16=False,
        verbose=True,
    )
    dt = time.perf_counter() - t0

    print(f"  Time: {dt:.2f}s")
    print(f"  Output: {logits.shape}, dtype={logits.dtype}")
    print(f"  Logit range: [{logits.min():.2f}, {logits.max():.2f}]")
    seg = np.argmax(logits, axis=0)
    print(f"  Labels: {np.unique(seg).tolist()[:10]}{'...' if len(np.unique(seg)) > 10 else ''}")

    return {"time": dt, "accum_mb": accum_mb, "n_patches": n_patches,
            "logits_shape": logits.shape, "logits": logits}


def main():
    print("=" * 60)
    print("Baseline benchmark: predict_sliding_window (full accumulator)")
    print("=" * 60)

    # Load model
    print("\nLoading model (task 297, fast mode)...")
    t0 = time.perf_counter()
    network, plans, meta = load_model()
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Params: {meta['n_params']:,} ({meta['param_mb']:.0f} MB)")
    print(f"  Patch: {meta['patch_size']}, Classes: {meta['num_classes']}")

    auto_batch = choose_batch_size(meta["patch_size"], meta["num_classes"])
    print(f"  Auto batch size: {auto_batch}")

    # Metal info
    try:
        info = mx.device_info()
        max_buf_gb = info["max_buffer_length"] / 1e9
        print(f"  Metal max buffer: {max_buf_gb:.1f} GB")
    except Exception:
        pass

    # Synthetic volume — (X, Y, Z) nibabel order, then preprocess
    vol_shape = (120, 120, 130)
    print(f"\nSynthetic volume: {vol_shape} (X, Y, Z)")
    data = make_synthetic_volume(vol_shape)

    preprocessed = preprocess_volume(data, plans, "3d_fullres")
    # Transpose to nnU-Net order: (C, X, Y, Z) -> (C, Z, Y, X)
    preprocessed = preprocessed.transpose(0, 3, 2, 1).copy()
    print(f"  Preprocessed: {preprocessed.shape} (C, Z, Y, X)")

    # Benchmark batch=1
    r1 = benchmark_sliding_window(
        network, preprocessed, meta["patch_size"], meta["num_classes"],
        batch_size=1, label="batch=1"
    )

    # Try batch=2 if auto suggests it
    if auto_batch >= 2:
        r2 = benchmark_sliding_window(
            network, preprocessed, meta["patch_size"], meta["num_classes"],
            batch_size=2, label="batch=2"
        )
        print(f"\n  Speedup batch=2 vs batch=1: {r1['time']/r2['time']:.2f}x")

        # Verify logits match
        diff = np.abs(r1["logits"] - r2["logits"])
        print(f"  Max logit diff (batch=1 vs batch=2): {diff.max():.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Volume: {vol_shape}")
    print(f"  Patches: {r1['n_patches']}")
    print(f"  Accumulator: {r1['accum_mb']:.1f} MB")
    print(f"  batch=1 time: {r1['time']:.2f}s")
    if auto_batch >= 2:
        print(f"  batch=2 time: {r2['time']:.2f}s")


if __name__ == "__main__":
    main()
