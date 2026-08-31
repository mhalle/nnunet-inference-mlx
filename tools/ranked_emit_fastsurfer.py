"""FastSurfer brain: run the engine, keep the 79-class logit field, and write the 1 mm image
the field actually lives on.

The counterpart to `emit_ranked.py` for the other engine. FastSurfer conforms its input to
FreeSurfer's 1 mm 256^3 LIA space and the VINN logits are native to THAT grid, so "resample to
1 mm" is not an extra step here - it is what the engine already does, and the store lands on the
conformed grid with no crop and no offset.

The conformed image is not re-derived - it is *captured*. The engine builds it with FastSurfer's
own name-mangled conform knobs (`r._RunModelOnData__conform_kwargs()`), so any reimplementation
here could drift from what the network was actually fed. Instead the driver wraps
`emit_probabilities`, which already receives that image as its `source_ref`, stashes it, and
delegates. The image written is therefore the one the logits came from, not a lookalike; the
script still asserts its geometry equals the `source_grid` recorded in the store.

Wrapping a module function is a scratch-driver move, not something product code should do - it
is here because the alternative is duplicating the engine's conform contract.

usage: .venvs/fastsurfer/bin/python tools/ranked_emit_fastsurfer.py T1 OUTDIR [depth] [clip] [device]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main(t1, outdir, depth=6, clip=8.0, device="mps"):
    from nnseg import io
    from nnseg.engines import fastsurfer as fs
    from nnseg.engines.geometry import grid_record
    from nnseg.ranked import RankedSpec

    depth, clip = int(depth), float(clip)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    metas, t0 = {}, time.perf_counter()

    def sink(part, code):
        for name, arr in (("ranks", code.ranks), ("support", code.support), ("tail", code.tail)):
            if arr is not None:
                np.save(out / f"{part}_{name}.npy", arr)
        metas[part] = code.meta
        print(f"  {part:<8} {code!r}", flush=True)

    # capture the conformed image the engine actually fed the network (see module docstring)
    captured = {}
    _real = fs.emit_probabilities

    def _capture(spec, logits, source_ref, target_ref, class_labels):
        captured["conf"] = source_ref
        return _real(spec, logits, source_ref, target_ref, class_labels)

    fs.emit_probabilities = _capture
    try:
        t1_img = io.read_image(str(t1))
        seg = fs.segment(t1_img, device=device, restore="cpu",
                         probabilities=RankedSpec(sink=sink, depth=depth, clip=clip))
    finally:
        fs.emit_probabilities = _real

    conf = captured["conf"]
    got, want = grid_record(conf), metas["brain"]["source_grid"]
    same = all(np.allclose(got[k], want[k], atol=1e-4) for k in got)
    print(f"\nconformed grid vs the store's source_grid: {'MATCH' if same else 'MISMATCH'}")
    print(f"  {got['shape_zyx']} @ {[round(v, 6) for v in got['spacing_zyx']]}")
    if not same:
        raise SystemExit(f"conform disagrees with the engine\n  got  {got}\n  want {want}")
    sitk.WriteImage(conf, str(out / "conformed_1mm.nii.gz"), useCompression=True)

    (out / "meta.json").write_text(json.dumps(
        {"image": str(t1), "task": "fastsurfer:brain", "engine": "fastsurfer",
         "depth": depth, "clip": clip, "device": device,
         "parts": metas, "provenance": seg.provenance, "timings": seg.timings},
        indent=1, default=str))
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
