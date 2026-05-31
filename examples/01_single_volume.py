"""Single-volume inference with the toolkit API: NIfTI in, NIfTI out.

The minimal usage: a ``ModelStore`` (id → model; read-through + bounded), a
reader, the ``segment`` one-liner, a writer.

    uv run python examples/01_single_volume.py scan.nii.gz seg.nii.gz [TASK]

``TASK`` is a task name (default ``"total_fast"`` — TS 3 mm). Weights are read
from ``~/.totalsegmentator/nnunet/results`` (or ``$TOTALSEG_WEIGHTS_PATH``).

Equivalent CLIs:
    uv run nnmlx segment total_fast scan.nii.gz seg.nii.gz
    uv run TotalSegmentator -i scan.nii.gz -o seg.nii.gz --ml --fast
"""

from __future__ import annotations

import sys

import numpy as np

from nnunet_inference_mlx import (
    ModelStore, NiftiReader, NiftiWriter, TaskCatalog, segment,
)


def main(inp: str, out: str, task: str = "total_fast") -> None:
    store = ModelStore("totalsegmentator")           # bounded, read-through, owned
    catalog = TaskCatalog("totalsegmentator")         # name → recipe (no global)

    image = NiftiReader().read(inp)                   # → Volume (channels-last + geometry)
    seg = segment(task, image, store=store, catalog=catalog)   # → Segmentation
    NiftiWriter().write(out, seg)

    labels = sorted(int(v) for v in np.unique(seg.data) if v)
    print(f"wrote {out}: {seg.geometry.shape_zyx}, {len(labels)} structures")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
