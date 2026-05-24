"""Full path-B pipeline via predict_with_resampling.

Path B = trilinear interpolation on the K-channel logit volume followed by
argmax at the end, instead of nearest-neighbor on post-argmax labels. The
inverse step is slab-streamed in unified memory so it fits even at K~117.

This example also enables cc3d-based dust cleanup as a single arg.

Usage:
    python examples/03_path_b_resampling.py scan.nii.gz seg.nii.gz [TASK_ID]

Extras required:
    pip install 'nnunet-inference-mlx[preprocessing,postprocessing]'
"""

from __future__ import annotations

import sys
import time

import SimpleITK as sitk

from nnunet_inference_mlx import (
    cached_engine_from_task,
    predict_with_resampling,
)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    task_id = int(sys.argv[3]) if len(sys.argv) > 3 else 297

    print(f"Loading task {task_id}...")
    engine = cached_engine_from_task(task_id, folds=0)

    print(f"Reading {in_path}...")
    img = sitk.ReadImage(in_path)
    print(f"  size XYZ {img.GetSize()}, spacing {img.GetSpacing()}")

    print("Running predict_with_resampling (path B + dust)...")
    t0 = time.perf_counter()
    seg = predict_with_resampling(
        engine,
        img,
        interpolation="linear",          # forward resample interpolation
        remove_small_components_mm3=200, # TS-equivalent dust threshold
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    sitk.WriteImage(seg, out_path)
    print(f"Wrote {out_path}")
    print(f"  output size XYZ {seg.GetSize()} (matches input)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
