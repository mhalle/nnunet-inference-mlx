"""
Test fp16 + streaming + batch=2 on real chest CT.

    uv run python tests/bench_streaming_chest2.py stream_fp16_b1
    uv run python tests/bench_streaming_chest2.py stream_fp16_b2
"""

import json
import sys
import time

import mlx.core as mx
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from nnunet_mlx.inference import (
    compute_sliding_window_steps,
    predict_sliding_window,
    predict_sliding_window_streaming,
)
from nnunet_mlx.plans import build_network_from_plans
from nnunet_mlx.predict import find_model_folder
from nnunet_mlx.preprocessing import preprocess_volume
from nnunet_mlx.weights import load_model_weights, fuzzy_load_weights


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

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

    print("Warmup...")
    dummy = mx.random.normal((1, *patch_size, num_channels))
    mx.eval(compiled(dummy))
    del dummy

    print("Loading chest CT at 3mm...")
    img = nib.load("/tmp/chest.nii")
    data = np.asarray(img.dataobj, dtype=np.float32)
    spacing = np.array(img.header.get_zooms(), dtype=np.float32)
    data_3mm = zoom(data, spacing / 3.0, order=3).astype(np.float32)
    preprocessed = preprocess_volume(data_3mm, plans, "3d_fullres")
    preprocessed = preprocessed.transpose(0, 3, 2, 1).copy()
    print(f"  Shape: {preprocessed.shape}")

    spatial = preprocessed.shape[1:]
    steps = compute_sliding_window_steps(spatial, patch_size, 0.5)
    n_patches = len(steps[0]) * len(steps[1]) * len(steps[2])

    configs = {
        "full":             (predict_sliding_window,           1, False),
        "stream":           (predict_sliding_window_streaming, 1, False),
        "stream_fp16_b1":   (predict_sliding_window_streaming, 1, True),
        "stream_fp16_b2":   (predict_sliding_window_streaming, 2, True),
        "full_fp16_b1":     (predict_sliding_window,           1, True),
    }

    fn, batch_size, use_fp16 = configs[mode]
    label = f"{mode} (batch={batch_size}, fp16={use_fp16})"
    print(f"\nRunning: {label}, {n_patches} patches")

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
        use_fp16=use_fp16,
        verbose=True,
    )
    dt = time.perf_counter() - t0

    seg = np.argmax(logits, axis=0)
    print(f"\nTotal: {dt:.2f}s  ({dt/n_patches:.1f}s/patch)")
    print(f"Output: {logits.shape}, {logits.dtype}, {np.unique(seg).size} labels")


if __name__ == "__main__":
    main()
