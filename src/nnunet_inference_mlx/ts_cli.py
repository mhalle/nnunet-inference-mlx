"""``totalseg-mlx`` — a TotalSegmentator-compatible CLI on the MLX backend.

A drop-in front end: it accepts the *same* arguments as TotalSegmentator's
``TotalSegmentator`` command (mirrored from its argparse), so existing command
lines and scripts run unchanged on the Apple-Silicon/MLX backend. Flags the MLX
backend doesn't implement are accepted and ignored with a warning (so scripts
don't break), not rejected.

    totalseg-mlx -i ct.nii.gz -o segmentations
    totalseg-mlx -i ct.nii.gz -o seg.nii.gz --ml
    totalseg-mlx -i ct.nii.gz -o seg --fast -rs liver spleen

Supported: -i/-o, -ot nifti, -ml, -f/--fast, -ff/--fastest, -ta/--task,
-rs/--roi_subset, -rmb/--remove_small_blobs, -s/--statistics, -ss/--skip_saving,
-q/--quiet, -v/--verbose, --version. Everything else parses but is ignored
(warned unless --quiet). For the native, non-TS interface use ``nnmlx``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# TS task name -> our registry name (identical except where TS differs).
# Our catalog was generated from TS, so most names already match 1:1.
_TASK_ALIASES = {
    # add remaps here if any TS task name diverges from our registry
}

# Flags TS exposes that the MLX backend does not implement. dest -> human note.
# Accepted (so command lines parse) and ignored, with a warning unless --quiet.
_UNSUPPORTED = {
    "nr_thr_resamp": "threading is fixed (MLX/Metal)",
    "nr_thr_saving": "threading is fixed (MLX/Metal)",
    "nora_tag": "Nora integration not supported",
    "preview": "PNG preview not supported",
    "roi_subset_robust": "robust-crop ROI selection not supported (use --roi_subset)",
    "robust_crop": "robust cropping not supported",
    "higher_order_resampling": "higher-order resampling not supported",
    "radiomics": "radiomics not supported",
    "stats_include_incomplete": "incomplete-ROI stats flag ignored",
    "crop_path": "custom crop path not supported",
    "body_seg": "body cropping not supported",
    "force_split": "chunked processing not supported (MLX streams internally)",
    "no_derived_masks": "derived masks not generated",
    "v1_order": "v1 label ordering not supported",
    "save_probabilities": "probability saving not supported (use the nnmlx/predict API)",
    "license_number": "no license needed for the MLX backend",
}


def build_parser() -> argparse.ArgumentParser:
    """Mirror TotalSegmentator's argparse so identical command lines parse."""
    p = argparse.ArgumentParser(
        prog="totalseg-mlx",
        description="Segment anatomical structures in CT/MR images (MLX backend; "
                    "TotalSegmentator-compatible CLI).",
    )
    p.add_argument("-i", metavar="filepath", dest="input", required=True,
                   type=lambda x: Path(x).absolute(),
                   help="CT/MR nifti image or folder of dicom slices.")
    p.add_argument("-o", metavar="directory", dest="output", required=True,
                   type=lambda x: Path(x).absolute(),
                   help="Output directory for per-class masks, or output nifti path with --ml.")
    p.add_argument("-ot", "--output_type", type=str, nargs="+", default=None,
                   help="Output type(s): nifti (dicom_seg/dicom_rtstruct not supported).")
    p.add_argument("-ml", "--ml", action="store_true", default=False,
                   help="Save one multilabel image for all classes.")
    p.add_argument("-nr", "--nr_thr_resamp", type=int, default=1)
    p.add_argument("-ns", "--nr_thr_saving", type=int, default=6)
    p.add_argument("-f", "--fast", action="store_true", default=False,
                   help="Run faster lower resolution model (3mm).")
    p.add_argument("-ff", "--fastest", action="store_true", default=False,
                   help="Run even faster lower resolution model (6mm).")
    p.add_argument("-t", "--nora_tag", type=str, default="None")
    p.add_argument("-p", "--preview", action="store_true", default=False)
    p.add_argument("-ta", "--task", type=str, default="total",
                   help="Which model/task to run (default: total).")
    p.add_argument("-rs", "--roi_subset", type=str, nargs="+", default=None,
                   help="Subset of class names to save (space separated).")
    p.add_argument("-rsr", "--roi_subset_robust", type=str, nargs="+", default=None)
    p.add_argument("-rc", "--robust_crop", action="store_true", default=False)
    p.add_argument("-ho", "--higher_order_resampling", action="store_true", default=False)
    p.add_argument("-s", "--statistics", nargs="?", const=True, default=False, metavar="filepath",
                   help="Calc volume (mm3) and mean intensity -> statistics.json.")
    p.add_argument("-r", "--radiomics", action="store_true", default=False)
    p.add_argument("-sii", "--stats_include_incomplete", action="store_true", default=False)
    p.add_argument("-sa", "--stats_aggregation", type=str, choices=["mean", "median"], default="mean")
    p.add_argument("-cp", "--crop_path", type=lambda x: Path(x).absolute(), default=None)
    p.add_argument("-bs", "--body_seg", action="store_true", default=False)
    p.add_argument("-fs", "--force_split", action="store_true", default=False)
    p.add_argument("-ss", "--skip_saving", action="store_true", default=False,
                   help="Skip saving masks (e.g. when only statistics are wanted).")
    p.add_argument("-ndm", "--no_derived_masks", action="store_true", default=False)
    p.add_argument("-v1o", "--v1_order", action="store_true", default=False)
    p.add_argument("-rmb", "--remove_small_blobs", action="store_true", default=False,
                   help="Remove small connected components (<0.2 ml) from the output.")
    p.add_argument("-d", "--device", type=str, default="gpu",
                   help="Accepted for compatibility; the MLX backend always runs on Metal.")
    p.add_argument("-q", "--quiet", action="store_true", default=False)
    p.add_argument("-sp", "--save_probabilities", type=lambda x: Path(x).absolute(), default=None)
    p.add_argument("-v", "--verbose", action="store_true", default=False)
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("-l", "--license_number", type=str, default=None)
    p.add_argument("--test", type=int, choices=[0, 1, 3], default=0)
    try:
        import importlib.metadata as _im
        ver = _im.version("nnunet-inference-mlx")
    except Exception:
        ver = "unknown"
    p.add_argument("--version", action="version",
                   version=f"totalseg-mlx (nnunet-inference-mlx {ver}; TotalSegmentator-compatible CLI)")
    return p


def _warn_unsupported(args, log) -> None:
    parser_defaults = build_parser().parse_args(["-i", "x", "-o", "y"])
    hit = []
    for dest, note in _UNSUPPORTED.items():
        if getattr(args, dest) != getattr(parser_defaults, dest):
            hit.append(f"  --{dest}: {note}")
    if hit:
        log(f"warning: {len(hit)} option(s) ignored (not supported by the MLX backend):")
        for h in hit:
            log(h)


def _resolve_task(catalog, task, fast, fastest):
    from .catalog import AmbiguousTaskError
    name = _TASK_ALIASES.get(task, task)
    candidate = f"{name}_fastest" if fastest else f"{name}_fast" if fast else name
    for cand in (candidate, name):           # fall back if no fast variant exists
        try:
            return catalog.get(cand)
        except (KeyError, AmbiguousTaskError):
            continue
    raise KeyError(task)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    # output type: nifti only
    if args.output_type:
        types = [t for chunk in args.output_type for t in str(chunk).split(",")]
        if any(t not in ("nifti",) for t in types):
            print(f"error: output_type {types} not supported (nifti only on the MLX backend)",
                  file=sys.stderr)
            return 2

    _warn_unsupported(args, log)

    import numpy as np
    import SimpleITK as sitk

    from .catalog import AmbiguousTaskError, TaskCatalog
    from .imageio import DicomReader, NiftiReader, array_to_sitk
    from .postprocess import drop_small_components
    from .segment import segment
    from .store import ModelStore

    store = ModelStore("totalsegmentator")
    catalog = TaskCatalog("totalsegmentator")

    try:
        spec = _resolve_task(catalog, args.task, args.fast, args.fastest)
    except (KeyError, AmbiguousTaskError) as e:
        print(f"error: task {args.task!r} not available on the MLX backend ({e}). "
              f"Try `nnmlx tasks list`.", file=sys.stderr)
        return 2

    import time

    # TS-style console output (stdout, unless --quiet).
    def out(msg):
        if not args.quiet:
            print(msg, flush=True)

    out("\nIf you use this tool please cite: https://pubs.rsna.org/doi/10.1148/ryai.230024\n")
    if args.fastest:
        out("Using 'fastest' option: resampling to lower resolution (6mm)")
    elif args.fast:
        out("Using 'fast' option: resampling to lower resolution (3mm)")

    reader = DicomReader() if args.input.is_dir() else NiftiReader()
    image = reader.read(args.input)

    st = time.time()
    seg = segment(spec, image, store=store, catalog=catalog, progress=out)
    out(f"  Predicted in {time.time() - st:.2f}s")

    if args.remove_small_blobs:
        out("Removing small blobs...")
        st = time.time()
        seg = drop_small_components(seg, min_volume_mm3=200.0, in_place=True)
        out(f"  Removed in {time.time() - st:.2f}s")

    data = np.asarray(seg.data)
    names = dict(seg.schema.names)                 # {label_id: roi_name}
    wanted = set(args.roi_subset) if args.roi_subset else None

    # ----- save segmentations -----
    if not args.skip_saving:
        out("Saving segmentations...")
        st = time.time()
        if args.ml:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(array_to_sitk(data, seg.geometry), str(args.output))
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            classes = [(lid, nm) for lid, nm in sorted(names.items())
                       if lid != 0 and (not wanted or nm in wanted)]
            iterator = classes
            if not args.quiet:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(classes, desc="  saving", unit="roi", leave=False)
                except Exception:
                    pass
            for lid, nm in iterator:
                mask = (data == lid).astype(np.uint8)
                sitk.WriteImage(array_to_sitk(mask, seg.geometry), str(args.output / f"{nm}.nii.gz"))
        out(f"  Saved in {time.time() - st:.2f}s")

    # ----- statistics -----
    if args.statistics:
        ct = np.asarray(image.data[..., 0]).astype(np.float32)
        vox_mm3 = float(np.prod(seg.geometry.spacing_zyx))
        stats = {}
        for lid, nm in names.items():
            if lid == 0 or (wanted and nm not in wanted):
                continue
            m = data == lid
            cnt = int(m.sum())
            stats[nm] = {
                "volume_mm3": cnt * vox_mm3,
                "intensity_mean": float(ct[m].mean()) if cnt else 0.0,
            }
        if args.statistics is True:
            stats_path = (args.output.parent if args.ml else args.output) / "statistics.json"
        else:
            stats_path = Path(args.statistics).absolute()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))
        log(f"[totalseg-mlx] wrote statistics {stats_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
