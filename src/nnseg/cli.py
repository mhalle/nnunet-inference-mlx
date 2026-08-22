"""nnseg segment IN --task total_fast -o OUT [--spacing 1.0] [--interp nearest|linear]"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nnseg", description="nnU-Net-family segmentation on torch with labelgrid restore")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segment", help="segment one NIfTI")
    s.add_argument("input")
    s.add_argument("--task", required=True, help="task name from the catalog, e.g. total_fast, total")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--spacing", type=float, default=None, help="isotropic output spacing in mm (default: the input grid)")
    s.add_argument("--interp", choices=("linear", "nearest"), default="linear",
                   help="logit interpolation for the restore: linear = sub-voxel boundaries; nearest = TotalSegmentator semantics")
    s.add_argument("--device", default="mps")
    s.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    s.add_argument("--accumulate", choices=("auto", "device", "host"), default="auto",
                   help="sliding-window accumulator placement: auto (from free device memory), device (fastest, needs headroom), host")
    s.add_argument("--model-root", default=None)
    s.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "segment":
        from .pipeline import segment
        import nibabel as nib
        img, schema, T = segment(args.input, args.task, model_root=args.model_root, device=args.device, dtype=args.dtype,
                                 grid=args.spacing if args.spacing else "input", interp=args.interp, accumulate=args.accumulate,
                                 progress=None if args.quiet else (lambda m: print(f"  {m}", file=sys.stderr, flush=True)))
        nib.save(img, args.output)
        if not args.quiet:
            for k, v in T.items():
                print(f"  {v:7.2f} s  {k}", file=sys.stderr)
            print(f"wrote {args.output}: {img.shape}, {len(schema.names)} labels", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
