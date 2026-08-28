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

    sv = sub.add_parser("serve", help="run the REST job server (needs the serve extra)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8790)
    sv.add_argument("--device", default="auto")
    sv.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    sv.add_argument("--cache-models", type=int, default=5,
                    help="models kept warm across jobs (5 covers a total union)")
    sv.add_argument("--model-root", default=None)
    sv.add_argument("--max-pending", type=int, default=16, help="queue bound; past it POST returns 429")
    sv.add_argument("--keep-finished", type=int, default=50, help="finished jobs (and files) retained")
    sv.add_argument("--jobs-ttl-hours", type=float, default=24.0,
                    help="how long a job RECORD lasts (keep-finished bounds memory "
                         "and files; this bounds the durable record)")
    sv.add_argument("--workdir", default=None, help="job storage (default: a temp directory)")
    sv.add_argument("--cache-dir", default=None,
                    help="result cache (default: ~/.cache/nnseg/results; durable, unlike the workdir)")
    sv.add_argument("--no-result-cache", action="store_true")
    sv.add_argument("--token", default=None,
                    help="bearer token; without it a request gets health/tasks/cached reads only")

    mo = sub.add_parser("modal", help="deploy the server to Modal (needs the modal extra)")
    mosub = mo.add_subparsers(dest="mcmd", required=True)
    md = mosub.add_parser("deploy", help="deploy the packaged app to your Modal account")
    md.add_argument("--gpu", default=None, help="worker GPU (default L40S; A10 is the economical fast-mode choice)")
    md.add_argument("--app-name", default=None)
    md.add_argument("--scaledown", type=int, default=None,
                    help="seconds a warm worker lingers after its last job (Modal caps at 1200)")
    md.add_argument("--no-proxy-auth", action="store_true",
                    help="deploy WITHOUT auth - smoke tests only; anyone with the URL can spend your GPU credit")
    mosub.add_parser("app-path", help="print the deployable app file's path")

    rc = sub.add_parser("remote", help="talk to an nnseg server (needs the remote extra)")
    rc.add_argument("--server", default=None,
                    help="server URL, e.g. http://gpu-box:8790 (or set NNSEG_SERVER)")
    rc.add_argument("--token", default=None, help="bearer token, if the server wants one")
    rsub = rc.add_subparsers(dest="rcmd", required=True)
    rs = rsub.add_parser("submit", help="upload, wait with progress, download the labels")
    rs.add_argument("input", help="a local image file, or idc:<crdc_series_uuid> to segment straight from the Imaging Data Commons")
    rs.add_argument("--task", required=True)
    rs.add_argument("-o", "--output", default=None, help="where to save the labels (default: <input>_<task>.nii.gz)")
    rs.add_argument("--no-wait", action="store_true", help="print the job id and return")
    rst = rsub.add_parser("status", help="one job's status, as JSON")
    rst.add_argument("job_id")
    rf = rsub.add_parser("fetch", help="download a finished job's labels")
    rf.add_argument("job_id")
    rf.add_argument("-o", "--output", required=True)
    rx = rsub.add_parser("cancel", help="cancel an active job / delete a finished one")
    rx.add_argument("job_id")
    rsub.add_parser("tasks", help="what the server can segment")

    args = ap.parse_args(argv)
    if args.cmd == "modal":
        from importlib.resources import files
        apppath = str(files("nnseg").joinpath("modal_app.py"))
        if args.mcmd == "app-path":
            print(apppath)
            return 0
        try:
            import modal  # noqa: F401
        except ImportError:
            print("needs the modal extra: uv sync --extra modal "
                  "(or pip install 'nnseg[modal]')", file=sys.stderr)
            return 2
        import os
        import subprocess
        env = dict(os.environ)
        if args.gpu:
            env["NNSEG_GPU"] = args.gpu
        if args.app_name:
            env["NNSEG_APP_NAME"] = args.app_name
        if args.scaledown:
            env["NNSEG_SCALEDOWN"] = str(args.scaledown)
        if args.no_proxy_auth:
            env["NNSEG_PROXY_AUTH"] = "0"
        return subprocess.call([sys.executable, "-m", "modal", "deploy", apppath], env=env)
    if args.cmd == "serve":
        from .serve import main_serve
        return main_serve(args)
    if args.cmd == "remote":
        import json
        import os
        from .client import RemoteClient
        server = args.server or os.environ.get("NNSEG_SERVER")
        if not server:
            print("no server: pass --server or set NNSEG_SERVER", file=sys.stderr)
            return 2
        c = RemoteClient(server, token=args.token)
        if args.rcmd == "tasks":
            for t in c.tasks():
                print(t)
        elif args.rcmd == "status":
            print(json.dumps(c.status(args.job_id), indent=2))
        elif args.rcmd == "fetch":
            print(c.fetch(args.job_id, args.output))
        elif args.rcmd == "cancel":
            print(json.dumps(c.cancel(args.job_id)))
        elif args.rcmd == "submit":
            if args.no_wait:
                print(c.submit(args.input, args.task))
                return 0
            stem = args.input[4:16] if args.input.startswith("idc:") else args.input.rsplit(".nii", 1)[0].rstrip("/")
            out = args.output or f"{stem}_{args.task}.seg.nrrd"
            last = {}
            def show(s, _last=last):
                p = s.get("progress") or {}
                line = (f"  {s['state']:9s} " + (f"[queue {s['queue_position']}] " if s.get("queue_position") is not None else "")
                        + f"{p.get('stage', '')} {p.get('detail', '')} "
                        + (f"{p.get('fraction', 0) * 100:3.0f}%" if p else ""))
                if line != _last.get("line"):
                    print(line, file=sys.stderr, flush=True)
                    _last["line"] = line
            final = c.run(args.input, args.task, out, on_status=show)
            if final["state"] == "done":
                print(out)
            else:
                print(f"job ended {final['state']}", file=sys.stderr)
                return 1
        return 0
    if args.cmd == "segment":
        from .pipeline import segment
        r = segment(args.input, args.task, weights=args.model_root, device=args.device, dtype=args.dtype,
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
            root = args.root or weights_root("ts")
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
                print(f"  + {wid:5s} new dataset, default {e['default']}")
            for wid, tags in sorted(r["new_versions"].items(), key=lambda kv: int(kv[0])):
                print(f"  v {wid:5s} versions recorded: {', '.join(tags)}")
            for wid, (ours, theirs) in sorted(r["behind_upstream"].items(), key=lambda kv: int(kv[0])):
                print(f"  ~ {wid:5s} default {ours}, TotalSegmentator pins {theirs}"
                      + ("" if args.update_existing else "   [not repointed]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
