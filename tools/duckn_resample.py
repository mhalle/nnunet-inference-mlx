# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "zarr>=3", "scipy"]
# ///
"""Resample a duckn array onto a new grid, with the geometry and the metadata done by duckn.

This replaces a hand-written resampler that carried metadata by transcribing the fields its
author happened to be thinking about. That approach failed three separate ways in one file:

  - `value_transforms` was dropped, so the output declared no meaning for its own numbers and
    air read as 0 HU instead of -1024. The values were right -- an affine transform commutes
    with linear interpolation -- which is exactly why nothing downstream complained.
  - the true corner-rule spacing was computed and then discarded in favour of the nominal
    target, declaring 1.5 mm for a grid that is actually 1.504063 mm on two axes.
  - `centering: cell` was declared while the sampler used the node rule.

None of those are reachable here: `duckn.resample` copies the source metadata and drops only
what provably cannot survive a derivation, derives the spacing from the shape it actually
produced, and records the centering it applied.

Target grid
-----------
`--spacing/--shape/--factor` are duckn's own, and `--like` matches another array's sample
counts, which is how you land on a model's grid (nnseg/TotalSegmentator resample by
`round(n * s / target)`, so ask for that shape rather than that spacing).

Centering is the sample-count-to-extent relationship and it decides where the grid lands:
`cell` holds the field of view and shifts sample centres by half the spacing change; `node`
holds the first and last sample centres. It is read from the source unless `--centering`
overrides it. nnseg's pipeline is node/corner, so that is what matching its grid wants.

`--no-anti-alias` turns off the downsampling pre-blur, which is what a consumer validated on
unfiltered resampling needs: see docs/resampler-parity-finding.md, where the blur costs
sub-centimetre structures 30-66 % of their contrast and TotalSegmentator then misses them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr


def _duckn():
    """duckn is a sibling repo, not a PyPI package, so say where it comes from."""
    try:
        from duckn.models import Centering
        from duckn.resample import resample
        from duckn.volume import Volume
        from duckn.zarr_io import read_duckn_metadata
    except ImportError as e:
        raise SystemExit(
            "duckn is required and is not installed. It is a sibling repo, not on PyPI:\n"
            "  uv pip install -e ../duckn        (path relative to the medseg workspace)\n"
            f"underlying error: {e}"
        ) from e
    return Centering, resample, Volume, read_duckn_metadata


def open_store(path: Path, mode: str = "r"):
    p = str(path)
    return zarr.storage.ZipStore(p, mode=mode) if p.endswith(".zip") else zarr.storage.LocalStore(p)


def target_shape_like(src_shape, src_spacing, ref_shape) -> tuple[int, ...]:
    if len(ref_shape) != len(src_shape):
        raise SystemExit(f"--like has {len(ref_shape)} axes, source has {len(src_shape)}")
    return tuple(int(n) for n in ref_shape)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spacing", type=float, help="isotropic target spacing")
    g.add_argument("--shape", type=int, nargs="+", help="explicit target sample counts")
    g.add_argument("--factor", type=float, nargs="+", help="relative zoom, scalar or per-axis")
    g.add_argument("--like", type=Path, help="match this array's sample counts")
    g.add_argument("--model-spacing", type=float,
                   help="nnseg/TotalSegmentator's rule: shape = round(n * s / this)")
    ap.add_argument("--centering", choices=["cell", "node"],
                    help="override the source's declared convention")
    ap.add_argument("--order", type=int, default=1, choices=[0, 1, 3],
                    help="0 nearest (labels), 1 linear (default), 3 cubic")
    ap.add_argument("--no-anti-alias", action="store_true",
                    help="skip the downsampling pre-blur (see docs/resampler-parity-finding.md)")
    ap.add_argument("--dtype", help="output dtype (default: the source's)")
    ap.add_argument("--chunks", type=int, nargs="+", help="output chunks (default: the source's)")
    args = ap.parse_args(argv)

    Centering, resample, Volume, read_duckn_metadata = _duckn()

    store = open_store(args.source)
    src = zarr.open(store=store, mode="r")
    meta = read_duckn_metadata(store)
    spacing_in = [float(np.linalg.norm(ax.space_direction))
                  if ax.space_direction is not None else None for ax in meta.axes]
    print(f"source {args.source.name}: shape {src.shape} dtype {src.dtype}")
    print(f"  spacing   {[round(v, 6) if v else None for v in spacing_in]}")
    print(f"  origin    {meta.space_origin}")
    print(f"  centering {[ax.centering.value if ax.centering else None for ax in meta.axes]}")
    print(f"  transform {meta.value_transforms}")

    kw: dict = {}
    if args.spacing is not None:
        kw["spacing"] = args.spacing
    elif args.shape is not None:
        kw["shape"] = tuple(args.shape)
    elif args.factor is not None:
        kw["factor"] = args.factor[0] if len(args.factor) == 1 else list(args.factor)
    elif args.like is not None:
        ref = zarr.open(store=open_store(args.like), mode="r")
        kw["shape"] = target_shape_like(src.shape, spacing_in, ref.shape)
        print(f"  matching  {args.like.name} shape {ref.shape}")
    else:
        kw["shape"] = tuple(round(n * s / args.model_spacing)
                            for n, s in zip(src.shape, spacing_in))
        print(f"  model rule round(n * s / {args.model_spacing}) -> {kw['shape']}")

    if args.centering:
        kw["centering"] = Centering(args.centering)

    print(f"\nreading {np.prod(src.shape) * src.dtype.itemsize / 1e9:.2f} GB ...", flush=True)
    vol = Volume(raw=np.asarray(src[:]), metadata=meta)

    print("resampling ...", flush=True)
    out = resample(vol, order=args.order, anti_alias=not args.no_anti_alias, **kw)

    # Round back into an integral dtype; duckn returns float for order > 0, and the source's
    # value_transform is defined against the stored integers.
    dtype = np.dtype(args.dtype) if args.dtype else src.dtype
    data = np.asarray(out.raw)
    if dtype.kind in "iu" and data.dtype.kind == "f":
        info = np.iinfo(dtype)
        data = np.rint(data).clip(info.min, info.max).astype(dtype)
    elif data.dtype != dtype:
        data = data.astype(dtype)

    sp_out = [float(np.linalg.norm(ax.space_direction))
              if ax.space_direction is not None else None for ax in out.metadata.axes]
    print(f"\noutput: shape {data.shape} dtype {data.dtype}")
    print(f"  spacing   {[round(v, 9) if v else None for v in sp_out]}  (realized, not nominal)")
    print(f"  origin    {out.metadata.space_origin}")
    print(f"  centering {[ax.centering.value if ax.centering else None for ax in out.metadata.axes]}")
    print(f"  transform {out.metadata.value_transforms}")

    record_processing(out.metadata, args, src.shape, spacing_in)

    chunks = tuple(args.chunks) if args.chunks else tuple(
        min(c, n) for c, n in zip(src.chunks, data.shape))
    args.dest.unlink(missing_ok=True)
    dst = zarr.create_array(store=open_store(args.dest, "w"), name=None,
                            shape=data.shape, dtype=data.dtype, chunks=chunks,
                            compressors=zarr.codecs.ZstdCodec(level=3))
    dst[:] = data
    dst.attrs.update({"duckn": out.metadata.model_dump(exclude_none=True, mode="json")})
    del dst
    print(f"\nwrote {args.dest} ({args.dest.stat().st_size / 1e6:.1f} MB, chunks {chunks})")
    return 0


def record_processing(metadata, args, src_shape, src_spacing) -> None:
    """Append a duckn `processing` step, keeping whatever provenance is already there.

    duckn documents the provenance extension but does not yet model it, so this writes the
    documented shape directly. Switch to the model when duckn implements one.
    """
    ext = metadata.extensions if isinstance(metadata.extensions, dict) else {}
    ext = dict(ext or {})
    prov = dict(ext.get("provenance") or {})
    steps = list(prov.get("processing") or [])
    steps.append({
        "operation": "resample",
        "software": {"name": "duckn.resample", "parameters": {
            "order": args.order,
            "anti_alias": not args.no_anti_alias,
            "centering": args.centering or "from source",
        }},
        "source_shape": [int(n) for n in src_shape],
        "source_spacing": [round(v, 9) if v else None for v in src_spacing],
    })
    prov["processing"] = steps
    ext["provenance"] = prov
    metadata.extensions = ext


if __name__ == "__main__":
    sys.exit(main())
