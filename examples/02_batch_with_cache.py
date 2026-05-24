"""Process a folder of CT volumes reusing one cached engine.

The engine cache holds the built+compiled+warmed network across calls.
For batches of N scans this saves ~3-5 s per scan on cold-start cost.

Cache is auto-enabled on >= 32 GB Macs and auto-disabled below; override
via NNUNET_MLX_CACHE_ENGINES=1 (force on) or =0 (force off).

Usage:
    python examples/02_batch_with_cache.py /path/to/ct/dir /path/to/out/dir [TASK_ID]

Looks for *.nii.gz in the input directory, writes <stem>_seg.nii.gz files
to the output directory.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from nnunet_inference_mlx import cached_engine_from_task, predict_nifti


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    task_id = int(sys.argv[3]) if len(sys.argv) > 3 else 297

    out_dir.mkdir(parents=True, exist_ok=True)

    # On 16 GB Macs the cache is off by default; opt in here so this script
    # demonstrates the warm path even on small machines.
    if "NNUNET_MLX_CACHE_ENGINES" not in os.environ:
        os.environ["NNUNET_MLX_CACHE_ENGINES"] = "1"

    scans = sorted(in_dir.glob("*.nii.gz"))
    if not scans:
        print(f"No .nii.gz files in {in_dir}")
        return 1
    print(f"Found {len(scans)} scans.")

    # First call builds the engine; subsequent calls return the cached one.
    print(f"Loading task {task_id} (first call builds, ~3-5 s)...")
    t0 = time.perf_counter()
    engine = cached_engine_from_task(task_id, folds=0)
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    times = []
    for i, scan in enumerate(scans, 1):
        out = out_dir / f"{scan.name.replace('.nii.gz', '_seg.nii.gz')}"
        t0 = time.perf_counter()
        predict_nifti(engine, scan, out)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  [{i}/{len(scans)}] {scan.name}: {dt:.1f}s -> {out.name}")

    if len(times) >= 2:
        first, rest = times[0], times[1:]
        print(f"\nTimings: first {first:.1f}s, "
              f"subsequent mean {sum(rest)/len(rest):.1f}s "
              f"({len(rest)} scans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
