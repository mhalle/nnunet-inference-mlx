"""Manual composition of the layered architecture for advanced callers.

Most users want the InferenceEngine facade — but the package is designed so
that callers needing fine-grained control can compose the individual
layers themselves. This example demonstrates three patterns:

1. Direct Predictor use (single-patch forward, no sliding window).
   The pattern an nnInteractive-style consumer would use: encode a click
   into a 192^3 patch, run one forward pass, return logits. Skips the
   sliding-window wrapper entirely.

2. Manual SlidingWindowEngine with custom step_size.
   Useful when you want a non-default overlap (e.g. step_size=0.25 for
   higher-quality boundaries at 4x the patch count).

3. FoldEnsemble with explicit per-fold averaging.
   Useful when you want to control which folds participate or how their
   outputs are combined.

Usage:
    python examples/05_layered_engine.py [TASK_ID]
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np

from nnunet_inference_mlx import (
    FoldEnsemble,
    ModelBundle,
    Predictor,
    SlidingWindowEngine,
)


def pattern_1_single_patch(bundle: ModelBundle) -> None:
    """Run one forward pass on a single patch — no sliding window.

    The nnInteractive-style pattern. After Predictor is built and warm,
    each forward is a single Metal kernel launch + readback.
    """
    print("\n=== Pattern 1: single-patch Predictor (nnInteractive-style) ===")
    predictor = Predictor(bundle, verbose=False)
    pz, py, px = predictor.patch_size
    print(f"  patch_size {predictor.patch_size}, K={predictor.num_classes}")

    # The Predictor's network expects channels-last input shape (1, Z, Y, X, C).
    rng = np.random.default_rng(0)
    patch = rng.standard_normal(
        (1, pz, py, px, predictor.num_input_channels)
    ).astype(np.float32)
    patch_mx = mx.array(patch)

    # Warmup
    _ = predictor.network(patch_mx)
    mx.eval(_)

    t0 = time.perf_counter()
    logits = predictor.network(patch_mx)
    mx.eval(logits)
    print(f"  forward: {(time.perf_counter() - t0) * 1000:.1f} ms  "
          f"output shape {logits.shape}")


def pattern_2_custom_sliding_window(bundle: ModelBundle) -> None:
    """SlidingWindowEngine with explicit step_size.

    step_size=0.25 means 75% overlap between adjacent patches — 4x the
    number of patches vs the 0.5 default, but smoother boundaries.
    """
    print("\n=== Pattern 2: SlidingWindowEngine with step_size=0.25 ===")
    predictor = Predictor(bundle, verbose=False)
    engine = SlidingWindowEngine(
        predictor,
        step_size=0.25,
        use_mirroring=False,
        progress=False,
    )

    pz, py, px = predictor.patch_size
    volume = np.random.randn(pz, py, px).astype(np.float32)
    t0 = time.perf_counter()
    logits = engine.predict(volume)
    print(f"  predict: {time.perf_counter() - t0:.2f}s  "
          f"logits shape {logits.shape}")


def pattern_3_explicit_fold_ensemble(bundle: ModelBundle) -> None:
    """FoldEnsemble averaging across folds.

    Bundle must have multiple folds for ensembling to matter. We
    demonstrate the assembly even on a single-fold bundle here.
    """
    if len(bundle.fold_weights) < 2:
        print("\n=== Pattern 3: FoldEnsemble (skipped — bundle has 1 fold) ===")
        return
    print(f"\n=== Pattern 3: FoldEnsemble across {len(bundle.fold_weights)} folds ===")
    predictor = Predictor(bundle, verbose=False)
    sliding = SlidingWindowEngine(predictor)
    ensemble = FoldEnsemble(
        sliding, bundle.fold_weights, region_based=bundle.has_regions,
    )
    pz, py, px = predictor.patch_size
    volume = np.random.randn(pz, py, px).astype(np.float32)
    t0 = time.perf_counter()
    averaged = ensemble.predict(volume)
    print(f"  predict: {time.perf_counter() - t0:.2f}s  "
          f"output shape {averaged.shape} (post-{'sigmoid' if bundle.has_regions else 'softmax'} mean)")


def main() -> int:
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 297
    print(f"Loading bundle for task {task_id}...")
    bundle = ModelBundle.from_task(task_id, folds="all")
    print(f"  {len(bundle.fold_weights)} folds, K={len(bundle.dataset['labels'])}")

    pattern_1_single_patch(bundle)
    pattern_2_custom_sliding_window(bundle)
    pattern_3_explicit_fold_ensemble(bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
