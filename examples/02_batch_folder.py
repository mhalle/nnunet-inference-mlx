"""Segment a folder of CT volumes, reusing one resident model.

The ``ModelStore`` *is* the cache now: ``store.load(id)`` builds + compiles the
model once and keeps it resident (bounded by ``max_memory_mb``, LRU-evicted),
so files 2..N skip the cold-start cost. No hidden global cache, no env-var
toggles — the store is explicit and owned.

    uv run python examples/02_batch_folder.py /ct/dir /out/dir [TASK]

Reads ``*.nii.gz`` from the input dir, writes ``<stem>_seg.nii.gz`` to the output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from nnunet_inference_mlx import (
    ModelStore, NiftiReader, NiftiWriter, TaskCatalog, segment,
)


def main(in_dir: str, out_dir: str, task: str = "total_fast") -> None:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scans = sorted(in_dir.glob("*.nii.gz"))
    if not scans:
        print(f"no .nii.gz in {in_dir}")
        return

    store = ModelStore("totalsegmentator", max_memory_mb=8000)   # holds the model across files
    catalog = TaskCatalog("totalsegmentator")
    reader, writer = NiftiReader(), NiftiWriter()

    times = []
    for i, scan in enumerate(scans, 1):
        t0 = time.perf_counter()
        seg = segment(task, reader.read(scan), store=store, catalog=catalog)
        writer.write(out_dir / f"{scan.name.replace('.nii.gz', '_seg.nii.gz')}", seg)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  [{i}/{len(scans)}] {scan.name}: {dt:.1f}s")

    if len(times) >= 2:
        print(f"\nfirst {times[0]:.1f}s (includes model build); "
              f"subsequent mean {sum(times[1:]) / len(times[1:]):.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
