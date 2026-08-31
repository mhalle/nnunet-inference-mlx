"""Run a task through the PRODUCT path and keep its ranked output distribution.

Supersedes `encode_inline.py`, which hand-rolled the codec in this script. That copy predates
the tie-breaking rule, so its ranks can differ from the shipped encoder wherever two logits are
exactly equal - and fp16 logits tie often. A demo store that disagrees with `nnseg.ranked` at
ties is a store nothing else reproduces, so this drives `segment(probabilities=RankedSpec(...))`
instead and touches no codec of its own.

The pipeline attaches the geometry itself (envelope start/stop, model grid, the channel -> label
lut, convention, orientation, frame), so each part lands self-describing and the duckn build
needs no second derivation.

`envelope_mm` is passed through: "none" runs the full model grid, which makes a store whose
array IS the model grid - no crop offset to apply, so the origin is the frame's own and a reader
needs no envelope arithmetic to place it. It costs inference time and some size (the extra voxels
are air, which compresses well but not to nothing).

usage: uv run python tools/ranked_emit.py IMAGE TASK OUTDIR [depth] [clip] [envelope_mm|none]
"""
import json
import time
from pathlib import Path

import numpy as np

from nnseg.pipeline import segment
from nnseg.ranked import RankedSpec


def main(image, task, outdir, depth=6, clip=8.0, envelope_mm=20.0):
    depth, clip = int(depth), float(clip)
    envelope_mm = (None if str(envelope_mm).lower() in ("none", "null", "")
                   else float(envelope_mm))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    metas, t0 = {}, time.perf_counter()

    def sink(part, code):
        for name, arr in (("ranks", code.ranks), ("support", code.support), ("tail", code.tail)):
            if arr is not None:
                np.save(out / f"{part}_{name}.npy", arr)
        metas[part] = code.meta
        print(f"  {part:<12} {code!r}  ->  {out.name}/{part}_*.npy", flush=True)

    seg = segment(image, task, probabilities=RankedSpec(sink=sink, depth=depth, clip=clip),
                  envelope_mm=envelope_mm,
                  progress=lambda p: print(f"    {p}", flush=True))

    (out / "meta.json").write_text(json.dumps(
        {"image": str(image), "task": task, "depth": depth, "clip": clip,
         "envelope_mm": envelope_mm,
         "parts": metas, "provenance": seg.provenance, "timings": seg.timings},
        indent=1, default=str))
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    # argparse, not a positional slice. `main(*sys.argv[1:6])` silently dropped the sixth
    # argument once, so `envelope_mm` kept its default and the run was quietly not the one
    # asked for - visible only because meta.json records what was actually used.
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("task")
    ap.add_argument("outdir")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--clip", type=float, default=8.0)
    ap.add_argument("--envelope-mm", default="20.0",
                    help='margin in mm, or "none" to run the full model grid')
    a = ap.parse_args()
    main(a.image, a.task, a.outdir, a.depth, a.clip, a.envelope_mm)
