"""Two-stage cascade: low-res localizer → high-res organ-specific model.

The first stage runs over the full volume at coarse spacing. Its output is
used to compute a bounding box around the target class; the second stage
runs only inside that box at finer spacing. Final output is pasted back
into the original image's geometry.

This is the MOOSE-style cascade pattern. For abdomen CTs it typically
gives 1.5-3x speedup at equal quality vs running the high-res model on
the full volume.

Usage:
    python examples/04_two_stage_cascade.py scan.nii.gz seg.nii.gz

The defaults use TotalSegmentator's Dataset298 (6 mm coarse) as the
localizer and Dataset297 (3 mm fast) as the second stage, cropping to
the body bbox between them. Adjust to your own model pair as needed.

Extras required:
    pip install 'nnunet-inference-mlx[preprocessing]'
"""

from __future__ import annotations

import sys
import time

import SimpleITK as sitk

from nnunet_inference_mlx import (
    Stage,
    cached_engine_from_task,
    run_workflow,
)


# Class IDs used as the cropping target after stage 1.
#
# For Dataset298 (6 mm fast), class 1 is typically a body-region label
# in TotalSegmentator's label scheme. Adjust if your localizer uses a
# different class numbering — Bundle.dataset["labels"] lists them.
LOCALIZER_CROP_CLASSES: tuple[int, ...] = tuple(range(1, 50))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    print("Loading stage-1 (6 mm localizer) and stage-2 (3 mm fast) engines...")
    t0 = time.perf_counter()
    stage1_engine = cached_engine_from_task(298, folds=0)
    stage2_engine = cached_engine_from_task(297, folds=0)
    print(f"  both engines ready in {time.perf_counter() - t0:.1f}s")

    print(f"Reading {in_path}...")
    img = sitk.ReadImage(in_path)
    print(f"  size XYZ {img.GetSize()}")

    stages = [
        # Stage 1: coarse localizer over the full volume.
        # crop_to_classes uses every foreground class so we get a bbox
        # around any anatomy the localizer detects, with a 10 mm margin.
        Stage(
            engine=stage1_engine,
            crop_to_classes=LOCALIZER_CROP_CLASSES,
            dilation_mm=10.0,
        ),
        # Stage 2: high-res organ model, runs only inside the bbox.
        # Optional dust cleanup on the final result.
        Stage(
            engine=stage2_engine,
            remove_small_components_mm3=200.0,
        ),
    ]

    print("Running 2-stage workflow...")
    t0 = time.perf_counter()
    seg = run_workflow(img, stages, verbose=True)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    sitk.WriteImage(seg, out_path)
    print(f"Wrote {out_path}")
    print(f"  output size XYZ {seg.GetSize()} (matches input)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
