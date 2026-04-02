"""
Performance benchmarks for MLX nnU-Net inference.

Default tests use a small model and finish in seconds.
Run slow tests with: pytest -m slow -v -s

Run: pytest tests/test_performance.py -v -s
"""

import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from nnunet_mlx.model import PlainConvUNet
from nnunet_mlx.inference import (
    predict_sliding_window,
    compute_sliding_window_steps,
    choose_batch_size,
)

# Smaller config — fast to build and run
ARCH_SMALL = dict(
    in_channels=1,
    n_stages=4,
    features_per_stage=[16, 32, 64, 128],
    kernel_sizes=[3, 3, 3, 3],
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_conv_per_stage=[2, 2, 2, 2],
    num_classes=3,
    n_conv_per_stage_decoder=[2, 2, 2],
    bias=True,
    norm_kwargs={"eps": 1e-5, "affine": True},
    nonlin_kwargs={"negative_slope": 0.01},
    deep_supervision=False,
)

# Realistic TotalSegmentator-like config (6 stages, 105 classes)
ARCH_LARGE = dict(
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


def _count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


# ---- Fast tests (default) ----

def test_single_patch_forward():
    """Single 64^3 patch through small model."""
    model = PlainConvUNet(**ARCH_SMALL)
    patch = mx.random.normal((1, 64, 64, 64, 1))

    # Warmup
    out = model(patch)
    mx.eval(out)

    t0 = time.perf_counter()
    out = model(patch)
    mx.eval(out)
    dt = time.perf_counter() - t0

    print(f"\n  {_count_params(model):,} params, 64^3 fp32: {dt:.3f}s")
    assert out.shape == (1, 64, 64, 64, 3)


def test_single_patch_fp16():
    """Single 64^3 patch in fp16."""
    model = PlainConvUNet(**ARCH_SMALL)
    patch = mx.random.normal((1, 64, 64, 64, 1)).astype(mx.float16)

    out = model(patch)
    mx.eval(out)

    t0 = time.perf_counter()
    out = model(patch)
    mx.eval(out)
    dt = time.perf_counter() - t0

    print(f"\n  {_count_params(model):,} params, 64^3 fp16: {dt:.3f}s")
    assert out.shape == (1, 64, 64, 64, 3)


def test_sliding_window_small():
    """Sliding window over 128^3 volume, small model, 64^3 patches."""
    model = PlainConvUNet(**ARCH_SMALL)
    compiled = mx.compile(model)
    patch_size = (64, 64, 64)
    volume = np.random.randn(1, 128, 128, 128).astype(np.float32)

    steps = compute_sliding_window_steps(volume.shape[1:], patch_size, 0.5)
    n_patches = len(steps[0]) * len(steps[1]) * len(steps[2])

    # Baseline
    t0 = time.perf_counter()
    logits = predict_sliding_window(
        network=model,
        input_image=volume,
        patch_size=patch_size,
        num_classes=3,
        tile_step_size=0.5,
        use_gaussian=True,
        use_fp16=True,
        batch_size=2,
    )
    dt_base = time.perf_counter() - t0

    # Compiled
    t0 = time.perf_counter()
    logits = predict_sliding_window(
        network=compiled,
        input_image=volume,
        patch_size=patch_size,
        num_classes=3,
        tile_step_size=0.5,
        use_gaussian=True,
        use_fp16=True,
        batch_size=2,
    )
    dt_comp = time.perf_counter() - t0

    print(f"\n  128^3 volume, 64^3 patches, {n_patches} patches:")
    print(f"    baseline: {dt_base:.3f}s ({dt_base/n_patches:.3f}s/patch)")
    print(f"    compiled: {dt_comp:.3f}s ({dt_comp/n_patches:.3f}s/patch)")
    print(f"    speedup:  {dt_base/dt_comp:.2f}x")
    assert logits.shape == (3, 128, 128, 128)


# ---- Slow tests (pytest -m slow) ----

@pytest.mark.slow
def test_large_model_single_patch():
    """Single 128^3 patch through full-size TotalSegmentator model."""
    model = PlainConvUNet(**ARCH_LARGE)
    patch = mx.random.normal((1, 128, 128, 128, 1))

    out = model(patch)
    mx.eval(out)

    t0 = time.perf_counter()
    out = model(patch)
    mx.eval(out)
    dt = time.perf_counter() - t0

    print(f"\n  {_count_params(model):,} params, 128^3 fp32: {dt:.3f}s")
    assert out.shape == (1, 128, 128, 128, 105)


@pytest.mark.slow
def test_large_model_sliding_window():
    """Sliding window over 128x256x256 with full-size model."""
    model = PlainConvUNet(**ARCH_LARGE)
    patch_size = (128, 128, 128)
    volume = np.random.randn(1, 128, 256, 256).astype(np.float32)

    batch_size = choose_batch_size(patch_size, num_classes=105, dtype_bytes=2)
    steps = compute_sliding_window_steps(volume.shape[1:], patch_size, 0.5)
    n_patches = len(steps[0]) * len(steps[1]) * len(steps[2])

    t0 = time.perf_counter()
    logits = predict_sliding_window(
        network=model,
        input_image=volume,
        patch_size=patch_size,
        num_classes=105,
        tile_step_size=0.5,
        use_gaussian=True,
        use_fp16=True,
        batch_size=batch_size,
        verbose=True,
    )
    dt = time.perf_counter() - t0

    print(f"\n  128x256x256, patch 128^3, {n_patches} patches, batch={batch_size}")
    print(f"  Total: {dt:.3f}s ({dt/n_patches:.3f}s/patch)")
    assert logits.shape == (105, 128, 256, 256)
