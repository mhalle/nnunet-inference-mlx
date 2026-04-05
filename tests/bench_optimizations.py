"""
Benchmark: mx.compile impact on large model inference.

Run: python tests/bench_optimizations.py
"""

import time
import mlx.core as mx
import mlx.nn as nn
from nnunet_inference_mlx.model import PlainConvUNet

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

PATCH = (1, 128, 128, 128, 1)


def bench(label, fn, x, n=3):
    # Warmup
    out = fn(x)
    mx.eval(out)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(x)
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    best = min(times)
    print(f"  {label:25s}  avg={avg:.3f}s  best={best:.3f}s")
    return best


def main():
    model = PlainConvUNet(**ARCH_LARGE)
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    compiled = mx.compile(model)

    x32 = mx.random.normal(PATCH)
    x16 = x32.astype(mx.float16)

    print(f"=== PlainConvUNet 128^3 -> 105 classes ({n_params:,} params) ===\n")

    bench("baseline fp32", model, x32)
    bench("baseline fp16", model, x16)
    bench("compiled fp32", compiled, x32)
    bench("compiled fp16", compiled, x16)

    print()
    baseline = bench("baseline fp32 (ref)", model, x32)
    best = bench("compiled fp32 (ref)", compiled, x32)
    print(f"\n  Speedup: {baseline/best:.2f}x")


if __name__ == "__main__":
    main()
