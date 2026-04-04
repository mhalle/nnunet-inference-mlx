"""
Benchmark streaming vs full accumulator on real CT chest volume at 3mm.

    uv run python tests/bench_streaming_chest.py full
    uv run python tests/bench_streaming_chest.py stream
    uv run python tests/bench_streaming_chest.py stream2
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

    # Load model
    print("Loading model...")
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

    # Load and resample chest volume to 3mm
    print("Loading /tmp/chest.nii...")
    img = nib.load("/tmp/chest.nii")
    data = np.asarray(img.dataobj, dtype=np.float32)
    spacing = np.array(img.header.get_zooms(), dtype=np.float32)
    target_spacing = 3.0

    print(f"  Original: {data.shape}, spacing={spacing}, {data.nbytes/1e6:.0f}MB")

    scale = spacing / target_spacing
    print(f"  Resampling to 3mm (scale={scale})...")
    t0 = time.perf_counter()
    data_3mm = zoom(data, scale, order=3).astype(np.float32)
    print(f"  Resampled: {data_3mm.shape} in {time.perf_counter()-t0:.1f}s, "
          f"{data_3mm.nbytes/1e6:.0f}MB")

    # Preprocess (CT normalization)
    preprocessed = preprocess_volume(data_3mm, plans, "3d_fullres")
    # Transpose to nnU-Net order: (C, X, Y, Z) -> (C, Z, Y, X)
    preprocessed = preprocessed.transpose(0, 3, 2, 1).copy()
    spatial = preprocessed.shape[1:]
    print(f"  Preprocessed: {preprocessed.shape} (C, Z, Y, X)")

    steps = compute_sliding_window_steps(spatial, patch_size, 0.5)
    n_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
    accum_mb = num_classes * np.prod(spatial) * 4 / 1e6
    print(f"  Patches: {n_patches}, Z steps: {steps[0]}")
    print(f"  Full accumulator would be: {accum_mb:.0f}MB")

    # Select mode
    batch_size = 2 if mode == "stream2" else 1
    fn = predict_sliding_window_streaming if mode.startswith("stream") else predict_sliding_window

    print(f"\nRunning: {mode}, batch_size={batch_size}")
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
    n_labels = np.unique(seg).size
    print(f"\nTotal: {dt:.2f}s")
    print(f"Output: {logits.shape}, {logits.dtype}, {n_labels} labels")
    print(f"Logits: [{logits.min():.1f}, {logits.max():.1f}]")


if __name__ == "__main__":
    main()
