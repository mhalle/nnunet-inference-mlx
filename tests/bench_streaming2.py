"""
Focused A/B test: run ONE sliding window pass, either full or streaming.
Pass 'full' or 'stream' as arg. Compare times across separate invocations
to avoid thermal throttling.

    uv run python tests/bench_streaming2.py full
    uv run python tests/bench_streaming2.py stream
    uv run python tests/bench_streaming2.py stream2   # batch=2
"""

import json
import sys
import time

import mlx.core as mx
import numpy as np

from nnunet_mlx.inference import (
    predict_sliding_window,
    predict_sliding_window_streaming,
)
from nnunet_mlx.plans import build_network_from_plans
from nnunet_mlx.predict import find_model_folder
from nnunet_mlx.preprocessing import preprocess_volume
from nnunet_mlx.weights import load_model_weights, fuzzy_load_weights


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    # Load model
    model_folder = find_model_folder(
        297, trainer="nnUNetTrainer_4000epochs_NoMirroring")
    plans = json.loads((model_folder / "plans.json").read_text())
    dataset = json.loads((model_folder / "dataset.json").read_text())

    config = plans["configurations"]["3d_fullres"]
    patch_size = tuple(config["patch_size"])
    num_classes = len(dataset["labels"])
    num_channels = len(dataset.get("channel_names", dataset.get("modality", {})))

    network = build_network_from_plans(
        plans, "3d_fullres", num_channels, num_classes, deep_supervision=False)
    weights = load_model_weights(model_folder, fold=0)
    try:
        network.load_weights(list(weights.items()))
    except Exception:
        fuzzy_load_weights(network, weights)
    compiled = mx.compile(network)

    # Warmup
    print("Warmup...")
    dummy = mx.random.normal((1, *patch_size, num_channels))
    mx.eval(compiled(dummy))
    del dummy

    # Synthetic volume
    np.random.seed(42)
    data = np.random.uniform(-200, 200, (120, 120, 130)).astype(np.float32)
    preprocessed = preprocess_volume(data, plans, "3d_fullres")
    preprocessed = preprocessed.transpose(0, 3, 2, 1).copy()

    batch_size = 2 if mode == "stream2" else 1
    fn = predict_sliding_window_streaming if mode.startswith("stream") else predict_sliding_window

    print(f"\nMode: {mode}, batch_size={batch_size}")
    print(f"Volume: {preprocessed.shape}, patch={patch_size}, classes={num_classes}")

    t0 = time.perf_counter()
    logits = fn(
        network=compiled,
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

    seg = np.argmax(logits, axis=0)
    print(f"\nTotal: {dt:.2f}s")
    print(f"Output: {logits.shape}, labels={np.unique(seg).size}")

    # Save logits hash for correctness comparison
    h = hash(logits.tobytes())
    print(f"Logits hash: {h}")


if __name__ == "__main__":
    main()
