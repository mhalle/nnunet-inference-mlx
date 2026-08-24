"""nnseg segment IN --task total_fast -o OUT [--spacing 1.0] [--interp nearest|linear]"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nnseg", description="nnU-Net-family segmentation on torch: fused logit restore onto any grid")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segment", help="segment one NIfTI")
    s.add_argument("input", help="NIfTI / NRRD / MetaImage file, or a DICOM series directory")
    s.add_argument("--task", required=True, help="task name from the catalog, e.g. total_fast, total")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--spacing", type=float, default=None, help="isotropic output spacing in mm (default: the input grid)")
    s.add_argument("--interp", choices=("linear", "nearest"), default="linear",
                   help="logit interpolation for the restore: linear = sub-voxel boundaries; nearest = TotalSegmentator semantics")
    s.add_argument("--device", default="auto", help="auto (default), cuda, mps or cpu")
    s.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    s.add_argument("--accumulate", choices=("auto", "device", "host"), default="auto",
                   help="sliding-window accumulator placement: auto (from free device memory), device (fastest, needs headroom), host")
    s.add_argument("--batch-size", default="auto", help="patches per forward pass: auto (default), or an int")
    s.add_argument("--envelope", type=float, default=20.0,
                   help="restrict inference to the body's bounding box plus this margin in mm; 0 or negative = whole volume")
    s.add_argument("--model-root", default=None)
    s.add_argument("--quiet", action="store_true")

    w = sub.add_parser("weights", help="provision and maintain model weights")
    wsub = w.add_subparsers(dest="wcmd", required=True)
    wf = wsub.add_parser("fetch", help="download everything a task needs")
    wf.add_argument("task")
    wf.add_argument("--root", default=None, help="weights root (default: the ecosystem's location)")
    wsub.add_parser("coverage", help="which catalog tasks the manifest can provision")
    wr = wsub.add_parser("refresh", help="merge newly published weights into the manifest")
    wr.add_argument("--repo", default=None, help="GitHub repo to read releases from")
    wr.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    wr.add_argument("--update-existing", action="store_true",
                    help="also repoint datasets at newer releases (changes which weights download)")

    args = ap.parse_args(argv)
    if args.cmd == "segment":
        from .pipeline import segment
        r = segment(args.input, args.task, model_root=args.model_root, device=args.device, dtype=args.dtype,
                    grid=args.spacing if args.spacing else "input", interp=args.interp, accumulate=args.accumulate,
                    batch_size=args.batch_size if args.batch_size == "auto" else int(args.batch_size),
                    envelope_mm=args.envelope if args.envelope > 0 else None,
                    progress=None if args.quiet else (lambda m: print(f"  {m}", file=sys.stderr, flush=True)))
        r.save(args.output)
        if not args.quiet:
            for k, v in r.timings.items():
                print(f"  {v:7.2f} s  {k}", file=sys.stderr)
            print(f"wrote {args.output}: {tuple(r.grid.shape)}, "
                  f"{len(r.present())}/{len(r.schema.names)} structures present", file=sys.stderr)
    if args.cmd == "weights":
        from . import weights_fetch as wfm
        say = lambda m: print(m, file=sys.stderr, flush=True)
        if args.wcmd == "fetch":
            from .tasks import weights_root
            root = args.root or weights_root("totalsegmentator")
            paths = wfm.ensure_task_weights(args.task, root, progress=lambda m: say(f"  {m}"))
            print(f"{len(paths)} model(s) under {root}")
        elif args.wcmd == "coverage":
            c = wfm.coverage()
            print(f"{len(c['covered'])}/{c['n_tasks']} tasks provisionable from {c['n_weights']} manifest entries")
            for name, ids in sorted(c["license_required"].items()):
                print(f"  LICENSE  {name:32s} {','.join(ids)}  (TotalSegmentator licensed backend)")
            for name, ids in sorted(c["missing"].items()):
                print(f"  MISSING  {name:32s} {','.join(ids)}")
            return 1 if c["missing"] else 0
        elif args.wcmd == "refresh":
            kw = {"write": not args.dry_run, "update_existing": args.update_existing, "progress": say}
            if args.repo:
                kw["repo"] = args.repo
            r = wfm.refresh_manifest(**kw)
            for wid, e in sorted(r["added"].items(), key=lambda kv: int(kv[0])):
                print(f"  + {wid:5s} new dataset, current {e['current']}")
            for wid, tags in sorted(r["new_versions"].items(), key=lambda kv: int(kv[0])):
                print(f"  v {wid:5s} versions recorded: {', '.join(tags)}")
            for wid, (ours, theirs) in sorted(r["behind_upstream"].items(), key=lambda kv: int(kv[0])):
                print(f"  ~ {wid:5s} current {ours}, upstream newest {theirs}"
                      + ("" if args.update_existing else "   [not repointed]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
