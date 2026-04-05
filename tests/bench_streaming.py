"""
Benchmark: streaming accumulator vs full accumulator.

Compares predict_sliding_window (full) vs predict_sliding_window_streaming
(rolling Z buffer). Verifies logit correctness, then benchmarks batch sizes.

Run from nnunet-mlx/:
    uv run python tests/bench_streaming.py
"""

import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from nnunet_mlx.inference import (
    choose_batch_size,
    compute_sliding_window_steps,
    predict_sliding_window,
    predict_sliding_window_streaming,
)
from nnunet_mlx.plans import build_network_from_plans
from nnunet_mlx.predict import find_model_folder
from nnunet_mlx.preprocessing import preprocess_volume
from nnunet_mlx.weights import load_model_weights, fuzzy_load_weights


def load_model(task_id=297, fold=0):
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
    dummy = mx.random.normal((1, *patch_size, num_channels))
    mx.eval(compiled(dummy))
    del dummy

    return compiled, plans, patch_size, num_classes, num_channels


def make_volume(shape):
    np.random.seed(42)
    data = np.random.uniform(-200, 200, shape).astype(np.float32)
    center = np.array(shape) // 2
    coords = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(sum((c - cn) ** 2 for c, cn in zip(coords, center)))
    data[dist < 20] = 400
    data[dist > 50] = -800
    return data


def preprocess(data, plans):
    preprocessed = preprocess_volume(data, plans, "3d_fullres")
    return preprocessed.transpose(0, 3, 2, 1).copy()


def run_one(fn, network, preprocessed, patch_size, num_classes, batch_size, label):
    print(f"\n  {label}")
    t0 = time.perf_counter()
    logits = fn(
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
    print(f"  → {dt:.2f}s, shape={logits.shape}, dtype={logits.dtype}")
    return logits, dt


def check_correctness(logits_a, logits_b, label_a, label_b):
    diff = np.abs(logits_a - logits_b)
    seg_a = np.argmax(logits_a, axis=0)
    seg_b = np.argmax(logits_b, axis=0)
    match_pct = 100 * (seg_a == seg_b).mean()
    print(f"\n  {label_a} vs {label_b}:")
    print(f"    Max diff: {diff.max():.8f}")
    print(f"    Mean diff: {diff.mean():.8f}")
    print(f"    Segmentation agreement: {match_pct:.2f}%")
    if diff.max() < 1e-3:
        print("    PASS")
    else:
        print(f"    FAIL — max diff {diff.max():.6f}")


def main():
    print("Loading model...")
    network, plans, patch_size, num_classes, num_channels = load_model()
    pZ, pY, pX = patch_size
    print(f"  patch={patch_size}, classes={num_classes}")

    # --- Test 1: Small volume, correctness ---
    vol_shape = (120, 120, 130)
    print(f"\n{'='*60}")
    print(f"Test 1: Correctness on small volume {vol_shape}")
    print(f"{'='*60}")

    data = make_volume(vol_shape)
    preprocessed = preprocess(data, plans)
    spatial = preprocessed.shape[1:]
    steps = compute_sliding_window_steps(spatial, patch_size, 0.5)
    n_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
    print(f"  Preprocessed: {preprocessed.shape}, {n_patches} patches")
    print(f"  Z steps: {steps[0]}")

    logits_full, t_full = run_one(
        predict_sliding_window, network, preprocessed,
        patch_size, num_classes, 1, "full, batch=1")

    logits_stream, t_stream = run_one(
        predict_sliding_window_streaming, network, preprocessed,
        patch_size, num_classes, 1, "streaming, batch=1")

    check_correctness(logits_full, logits_stream, "full", "streaming")

    # --- Test 2: Batch size comparison ---
    print(f"\n{'='*60}")
    print(f"Test 2: Batch size comparison (streaming)")
    print(f"{'='*60}")

    results = {"full_b1": t_full}
    for bs in [1, 2]:
        try:
            _, dt = run_one(
                predict_sliding_window_streaming, network, preprocessed,
                patch_size, num_classes, bs, f"streaming, batch={bs}")
            results[f"stream_b{bs}"] = dt
        except Exception as e:
            print(f"  batch={bs} failed: {e}")
            break

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for label, dt in results.items():
        speedup = results["full_b1"] / dt
        print(f"  {label:20s}: {dt:.2f}s  ({speedup:.2f}x vs full_b1)")


if __name__ == "__main__":
    main()
