"""Single-volume inference: NIfTI in, NIfTI out.

The minimum viable usage. Loads a model, runs inference, writes the
segmentation to disk. Output dtype is auto-picked (uint8 for K <= 255).

Usage:
    python examples/01_single_volume.py scan.nii.gz seg.nii.gz [TASK_ID]

If TASK_ID is omitted, defaults to 297 (TotalSegmentator's "fast" 3 mm task).

Prerequisites:
    - macOS with Apple Silicon, mlx >= 0.25
    - TotalSegmentator weights for the requested task downloaded into
      ~/.totalsegmentator/nnunet/results/ (the default TS install path)
      OR set $nnUNet_results / $TOTALSEG_WEIGHTS_PATH to your weights dir.
"""

from __future__ import annotations

import sys
import time

from nnunet_inference_mlx import (
    InferenceEngine,
    ModelBundle,
    predict_nifti,
)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    task_id = int(sys.argv[3]) if len(sys.argv) > 3 else 297

    print(f"Loading task {task_id}...")
    t0 = time.perf_counter()
    bundle = ModelBundle.from_task(task_id, folds=0)
    engine = InferenceEngine(bundle, verbose=False, progress=True)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  patch_size={engine.patch_size}, num_classes={engine.num_classes}")

    print(f"Inferring {in_path} -> {out_path} ...")
    t0 = time.perf_counter()
    predict_nifti(engine, in_path, out_path)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
