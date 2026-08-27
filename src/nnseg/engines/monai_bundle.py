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


def input_roles(bundle_dir) -> list:
    """The bundle's declared INPUT channel names, in channel order.

    The same ``channel_def`` idea as the label map, on the other side of the
    network. This is the order the stacked tensor must be in, and it is the
    bundle's to declare - MONAI's BraTS bundle wants T1c first where nnU-Net's
    own BraTS convention puts FLAIR there, which is exactly why the wire binds
    by name and the ORDER is read from here rather than from the request.
    """
    import json
    from pathlib import Path

    meta = json.loads((Path(bundle_dir) / "configs" / "metadata.json").read_text())
    inp = (((meta.get("network_data_format") or {}).get("inputs") or {})
           .get("image") or {})
    cd = inp.get("channel_def") or {}
    try:
        return [str(v) for _, v in sorted(cd.items(), key=lambda kv: int(kv[0]))]
    except (TypeError, ValueError):
        return []


def _same_grid(a, b, tol: float = 1e-4) -> bool:
    return (a.GetSize() == b.GetSize()
            and all(abs(x - y) < tol for x, y in zip(a.GetSpacing(), b.GetSpacing()))
            and all(abs(x - y) < tol for x, y in zip(a.GetOrigin(), b.GetOrigin()))
            and all(abs(x - y) < tol for x, y in zip(a.GetDirection(), b.GetDirection())))


def _stack_inputs(image_input, bundle_dir, bundle: str, read):
    """Order a ``{role: image}`` mapping into the bundle's channel order.

    **Co-registration is assumed, not performed.** A multi-channel network
    consumes one tensor, so its channels must already share a grid; producing
    that is a registration step, and registration belongs upstream - Slicer does
    it before it ever calls us, and doing it silently inside an inference call
    would be a geometry decision made on the caller's behalf, which is the
    failure mode this project has been bitten by three times. What we DO owe the
    caller is to check the assumption and say so plainly when it does not hold,
    rather than letting a shape mismatch surface from inside someone else's
    transform chain.
    """
    from ..errors import InputError

    roles = input_roles(bundle_dir)
    if len(roles) != len(image_input):
        raise InputError(
            f"monai bundle {bundle!r} declares {len(roles)} input channels "
            f"({', '.join(roles) or 'unnamed'}) but {len(image_input)} were "
            "supplied")
    missing = [r for r in roles if r not in image_input]
    if missing:
        raise InputError(f"monai bundle {bundle!r} is missing input(s): "
                         + ", ".join(missing))
    images = [(role, read(image_input[role])) for role in roles]
    ref_role, ref = images[0]
    for role, img in images[1:]:
        if not _same_grid(ref, img):
            raise InputError(
                f"input {role!r} is not on the same grid as {ref_role!r} "
                f"({img.GetSize()} @ {tuple(round(s, 4) for s in img.GetSpacing())} "
                f"vs {ref.GetSize()} @ {tuple(round(s, 4) for s in ref.GetSpacing())}). "
                "A multi-channel model stacks its inputs into one tensor, so they "
                "must be co-registered and resampled to a common grid before "
                "submission; nnseg does not register images.")
    return images


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


def _datalist_overrides(bundle_dir, paths: list) -> dict:
    """Point the bundle's own case list at the file(s) we staged.

    Bundles do not agree on how a case gets in. ``spleen_ct_segmentation``
    resolves a ``datalist`` list directly; ``brats_mri_segmentation`` loads a
    Decathlon JSON named by ``data_list_file_path`` and joined against
    ``dataset_dir``. Overriding the wrong one is worse than an error - the
    bundle quietly keeps its author's hardcoded path
    (``/workspace/data/medical/brats2018challenge``) and fails somewhere far from
    the cause, which is exactly how this surfaced the first time.

    So read which mechanism the config actually exposes and drive that one. This
    is the same kind of adaptation as finding ``inference.json`` vs
    ``inference.yaml``: a knob the bundle publishes, not surgery on its chain.
    """
    import json
    from pathlib import Path

    from monai.bundle import ConfigParser

    cfg = ConfigParser.load_config_file(str(inference_config(bundle_dir)))
    # a list of channel files is ONE case; MONAI's loader stacks it
    item = [str(p) for p in paths] if len(paths) > 1 else str(paths[0])
    parent = Path(paths[0]).parent
    if "data_list_file_path" in cfg:
        dl = parent / "datalist.json"
        dl.write_text(json.dumps({"testing": [{"image": item}]}))
        return {"data_list_file_path": str(dl), "dataset_dir": str(parent)}
    return {"datalist": [item], "dataset_dir": str(parent)}


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

    # One case either way: a single path, or the ordered list of channel files
    # that MONAI's loader stacks into multi-channel data.
    paths = image_path if isinstance(image_path, list) else [image_path]
    overrides = {
        "bundle_root": str(bundle_dir),
        "output_dir": str(out_dir),
        "device": device,
        **_datalist_overrides(bundle_dir, paths),
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

    def _read(x):
        return x if isinstance(x, sitk.Image) else nio.read_image(str(x))

    if isinstance(image_input, dict):
        channels = _stack_inputs(image_input, bundle_dir, bundle, _read)
        img = channels[0][1]              # the reference grid: channel 0
    else:
        channels, img = None, _read(image_input)

    timings: dict[str, float] = {}
    # Staging is a real cost, so pay as little of it as possible. A bundle's own
    # config loads from a FILE (datalist -> LoadImaged), and there is no supported
    # way to hand it the array we already decoded - injecting an in-memory
    # MetaTensor means replacing the loader in each bundle's chain, which is the
    # per-bundle surgery this engine exists to avoid. So: write UNCOMPRESSED (.nii,
    # not .nii.gz) into tmpfs when there is one. gzip is the dominant IO cost in
    # this pipeline - a warm total_fast case measured 82 % decompression - and it
    # buys nothing for a file that exists for one read, seconds from now.
    shm = Path("/dev/shm")
    parent = str(shm) if shm.is_dir() else None
    with tempfile.TemporaryDirectory(dir=parent) as td:
        td = Path(td)
        out_path = td / "out"
        out_path.mkdir()
        _t = time.perf_counter()
        if channels is None:
            in_path = td / "input.nii"
            sitk.WriteImage(img, str(in_path), False)   # useCompression=False
        else:
            # One file per channel, IN THE BUNDLE'S ORDER. MONAI's LoadImage
            # stacks a list of filenames into multi-channel data, so the datalist
            # item is the list - which keeps the bundle's own loader in charge,
            # the whole reason this engine runs their config instead of ours.
            in_path = []
            for i, (role, image) in enumerate(channels):
                p = td / f"input_{i}_{role}.nii"
                sitk.WriteImage(image, str(p), False)
                in_path.append(p)
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
