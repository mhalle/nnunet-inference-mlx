"""MONAI model-zoo bundles as an nnseg engine.

A bundle is not just weights: ``configs/inference.json`` is an executable
description of the whole chain - network, preprocessing, sliding-window inferer,
postprocessing - and every bundle writes its own. That is why MONAI needs its own
engine rather than riding the nnU-Net pipeline the way MOOSE's checkpoints do,
and why this module **runs the bundle's own config** instead of reimplementing
the chain. Bundles are too heterogeneous for a reimplementation to hold, and the
config is the contract their authors test against.

The cost of that choice is a file round-trip: bundle inference is dataset- and
writer-shaped, so the input goes to a temp NIfTI and the prediction is read back
from the bundle's output directory. FastSurfer avoided a round-trip because it
had one known conform to drive; that does not generalize to a whole zoo.

Labels come from the installed bundle's own ``metadata.json``
(``network_data_format.outputs.pred.channel_def``) - the bundle is the spec.

**Restore fidelity is the bundle's choice, not ours, and it varies.** A bundle's
postprocessing inverts its own spacing transform, and the order matters: spleen,
swin_unetr_btcv and wholeBrainSeg run ``Invertd(nearest_interp=False)`` BEFORE
``AsDiscreted(argmax=True)`` - resampling probabilities and arguing after, which is
the graded restore this project implements everywhere else. But
``wholeBody_ct_segmentation`` argmaxes FIRST and inverts the labelmap with
``nearest_interp=True``, so its output is a nearest-neighbour label resample, with
the blocky boundaries that implies on small structures. We do not override it: the
whole point of running the bundle's own config is that its authors test that chain,
and rewriting the postprocessing per bundle is the fragility this engine exists to
avoid. It is recorded here because it explains real quality differences between
bundles, and because the alternative - silently "fixing" someone else's model - is
worse than a documented limitation.
"""
from __future__ import annotations

import numpy as np

ENGINE = "monai"


def weights_installed(bundle: str, version: str) -> list[dict]:
    """This bundle's identity for the result-cache key. Per bundle+version, not a
    per-engine constant - the ecosystem owns it (see
    :meth:`nnseg.ecosystems.MonaiEcosystem.weights_identity`); this mirror exists
    so the worker can key a run it was handed directly."""
    return [{"id": bundle, "version": version}]


def label_names(bundle_dir) -> dict[int, str]:
    """``{label value: name}`` from the bundle's own metadata, background dropped."""
    import json
    from pathlib import Path

    meta = json.loads((Path(bundle_dir) / "configs" / "metadata.json").read_text())
    fmt = (meta.get("network_data_format") or {})
    channel_def = ((fmt.get("outputs") or {}).get("pred") or {}).get("channel_def") or {}
    return {int(k): str(v) for k, v in channel_def.items()
            if str(v).lower() != "background"}


def inference_config(bundle_dir):
    """The bundle's inference config. Not every bundle spells it the same way -
    pancreas_ct_dints ships `inference.yaml` where spleen ships `inference.json` -
    and hardcoding one extension is a failure that only shows up on the bundle you
    have not run yet."""
    from pathlib import Path

    for name in ("inference.json", "inference.yaml", "inference.yml"):
        p = Path(bundle_dir) / "configs" / name
        if p.is_file():
            return p
    have = sorted(x.name for x in (Path(bundle_dir) / "configs").glob("*"))
    raise FileNotFoundError(
        f"no inference config in {bundle_dir}/configs (have: {', '.join(have)})")


def _run_bundle(bundle_dir, image_path, out_dir, device: str, progress=None,
                timings: dict | None = None):
    """Drive the bundle's own inference workflow over exactly one image.

    Overrides only the three things that are ours to decide - which image, where
    the output goes, and which device - and leaves the bundle's transforms,
    network and inferer untouched.

    **Built per job, deliberately for now.** Every other engine caches its model
    across jobs (nnU-Net's ModelCache, FastSurfer's _RUNNERS, SynthStrip's
    _MODELS, VoxTell's _PREDICTORS), and this one does not, because a
    ``ConfigWorkflow`` resolves the network *and* the datalist into one cached
    parser: re-running it would re-run the same image, so reuse means reaching
    into the parser's resolved content, which is exactly the kind of thing that
    breaks across heterogeneous bundles. The build is timed separately so the
    decision is made on a number rather than a guess - if ``build`` turns out to
    be a large share of a warm run, caching earns its fragility; if it is a
    second, it does not.
    """
    import time

    from monai.bundle import create_workflow

    overrides = {
        "bundle_root": str(bundle_dir),
        "dataset_dir": str(image_path.parent),
        "datalist": [str(image_path)],
        "output_dir": str(out_dir),
        "device": device,
    }
    t0 = time.perf_counter()
    workflow = create_workflow(
        workflow_name=None, config_file=str(inference_config(bundle_dir)),
        workflow_type="inference", **overrides)
    workflow.initialize()
    if timings is not None:
        timings["build"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    workflow.run()
    workflow.finalize()
    if timings is not None:
        timings["infer"] = time.perf_counter() - t0


def segment(image_input, bundle: str, *, root, version: str | None = None,
            out_dir=None, device: str = "cuda", progress=None, cancel=None):
    """Run a MONAI bundle and return a :class:`nnseg.result.Segmentation` on the
    input's grid.

    ``bundle`` is the bundle name (``spleen_ct_segmentation``); ``root`` is the
    weights root the ecosystem installed it under. The bundle's own preprocessing
    decides spacing and orientation, and its postprocessing writes a labelmap,
    which is then resampled back onto the caller's grid if the bundle's writer
    moved it.
    """
    import tempfile
    import time
    from pathlib import Path

    import SimpleITK as sitk

    from .. import io as nio
    from ..ecosystems import MonaiEcosystem
    from ..errors import InputError
    from ..grid import Grid
    from ..progress import Reporter
    from ..result import Segmentation
    from ..values import LabelSchema

    report = Reporter.of(progress, cancel=cancel)
    eco = MonaiEcosystem()
    eco.ensure(bundle, root, progress=progress, version=version)
    bundle_dir = eco.bundle_root(bundle, root)

    img = image_input if isinstance(image_input, sitk.Image) else \
        nio.read_image(str(image_input))

    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_path, out_path = td / "input.nii.gz", td / "out"
        out_path.mkdir()
        _t = time.perf_counter()
        sitk.WriteImage(img, str(in_path))
        timings["stage"] = time.perf_counter() - _t

        report.tick(0, 1)
        _t = time.perf_counter()
        _run_bundle(bundle_dir, in_path, out_path, device, progress=progress,
                    timings=timings)
        timings["network"] = time.perf_counter() - _t
        report.tick(1, 1)

        # The bundle's writer decides the filename and nesting; take the only
        # volume it produced rather than guessing at its naming convention.
        written = sorted(p for p in out_path.rglob("*")
                         if p.suffix in (".gz", ".nii", ".nrrd") and p.is_file())
        if not written:
            raise InputError(
                f"monai bundle {bundle!r} produced no output under {out_path}; "
                "its inference config may write somewhere this engine does not look")
        pred = sitk.ReadImage(str(written[-1]))

    _t = time.perf_counter()
    if pred.GetSize() != img.GetSize():
        # a bundle whose postprocessing did not invert its own spacing transform
        pred = sitk.Resample(pred, img, sitk.Transform(), sitk.sitkNearestNeighbor,
                             0, pred.GetPixelID())
    out = sitk.Cast(pred, sitk.sitkUInt16)
    out.CopyInformation(img)
    timings["restore"] = time.perf_counter() - _t

    arr = sitk.GetArrayFromImage(out)
    names = label_names(bundle_dir)
    present = sorted({int(v) for v in np.unique(arr) if v != 0})
    print(f"[monai] bundle={bundle} labels_present={len(present)}/{len(names)} "
          f"grid={out.GetSize()}", flush=True)

    grid = Grid(shape=tuple(int(s) for s in arr.shape),
                spacing=tuple(float(s) for s in reversed(out.GetSpacing())),
                origin=tuple(float(o) for o in reversed(out.GetOrigin())))
    prov = {"engine": ENGINE, "bundle": bundle,
            "bundle_version": eco._entry(bundle)["version"],
            "network": f"MONAI bundle {bundle}",
            "labels_declared": len(names), "labels_present": len(present),
            "device": device}
    return Segmentation(labels=out, schema=LabelSchema(names=names), grid=grid,
                        spec=None, timings=timings, provenance=prov)
