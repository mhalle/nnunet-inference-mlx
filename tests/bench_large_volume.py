"""
Benchmark: 256^3 volume with 105 classes using segmentation mode.

Run: python tests/bench_large_volume.py
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from nnunet_mlx.model import PlainConvUNet
from nnunet_mlx.inference import (
    predict_sliding_window_segmentation,
    compute_sliding_window_steps,
)

ARCH = dict(
    in_channels=1,
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    kernel_sizes=[3, 3, 3, 3, 3, 3],
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_conv_per_stage=[2, 2, 2, 2, 2, 2],
    num_classes=105,
    n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
    bias=True,
    norm_kwargs={"eps": 1e-5, "affine": True},
    nonlin_kwargs={"negative_slope": 0.01},
    deep_supervision=False,
)


def main():
    patch_size = (128, 128, 128)
    volume_shape = (1, 128, 128, 128)

    steps = compute_sliding_window_steps(volume_shape[1:], patch_size, 0.75)
    n_patches = len(steps[0]) * len(steps[1]) * len(steps[2])

    print(f"Volume:  128^3, 105 classes (segmentation mode)")
    print(f"Patches: {n_patches}")
    print(f"Accum:   ~50MB (labels + scores only)")

    model = PlainConvUNet(**ARCH)
    compiled = mx.compile(model)
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"Params:  {n_params:,}")

    volume = np.random.randn(*volume_shape).astype(np.float32)

    print(f"\nRunning...")
    t0 = time.perf_counter()
    top_labels, top_scores = predict_sliding_window_segmentation(
        network=compiled,
        input_image=volume,
        patch_size=patch_size,
        num_classes=105,
        tile_step_size=0.75,
        use_gaussian=True,
        use_fp16=False,
        batch_size=1,
        verbose=True,
    )
    dt = time.perf_counter() - t0

    margin = top_scores[..., 0] - top_scores[..., 1]

    print(f"\nResults:")
    print(f"  Labels:    {top_labels.shape} {top_labels.dtype}")
    print(f"  Scores:    {top_scores.shape} {top_scores.dtype}")
    print(f"  Unique:    {np.unique(top_labels[..., 0]).size} classes")
    print(f"  Margin:    median={np.median(margin):.2f} "
          f"p5={np.percentile(margin, 5):.2f}")
    print(f"  Total:     {dt:.1f}s ({dt/60:.1f} min)")
    print(f"  Per patch: {dt/n_patches:.2f}s")


if __name__ == "__main__":
    main()
