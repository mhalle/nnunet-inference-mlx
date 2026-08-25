"""segment(): the torch pipeline - task -> parts -> logits -> labels on the chosen grid."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from .envelope import (Envelope, body_mask, body_threshold, envelope_of, label_roi,
                       margin_in_voxels, worth_cropping)
from .frame import Frame
from .mapping import Mapping
from .restore import to_labels
from .network import TorchModel, available_folds
from .tasks import ModelNotFound, TaskCatalog, TaskSpec, resolve_model_folder
from .cache import ModelCache
from .result import Segmentation
from .progress import Reporter
from .weights import as_store
from .cache import ModelCache
from .values import LabelSchema
from .preprocess import to_model_frame


def _version() -> str:
    from . import __version__                    # deferred: __init__ imports this module
    return __version__


def _warm_restore_kernel(device: str) -> None:
    """Compile the fused CUDA kernel while the network runs, instead of inside the first restore."""
    if not str(device).startswith("cuda"):
        return
    from .backends import triton_gpu
    if not triton_gpu.available():
        return
    threading.Thread(target=triton_gpu.warmup, daemon=True).start()


def _resolve_spec(task, catalog) -> TaskSpec:
    """A TaskSpec, a catalog name, or a path to a stock nnU-Net model folder."""
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, Path) or (isinstance(task, str) and Path(task).expanduser().is_dir()):
        return TaskSpec.from_model_folder(task)
    return catalog.get(task)


def _is_native(spec) -> bool:
    return spec.source == "nnunet"


def _lut(K: int, remap: dict | None) -> np.ndarray:
    lut = np.arange(K, dtype=np.int64)
    if remap:
        lut[:] = 0
        for local, global_ in remap.items():
            if not 0 <= int(local) < K:
                raise ModelNotFound(
                    f"catalog remap references local label {local} but the model "
                    f"emits {K} channels - the task catalog does not match the "
                    "installed weights (stale class map?)")
            lut[int(local)] = int(global_)
    return lut


def segment(image, task: str, *, catalog=None, weights=None, device: str = "auto", dtype: str = "fp16",
            grid="input", interp="linear", outside: str = "background", convention: str = "auto",
            folds=(0,), accumulate: str = "auto", resampling_order: int = 3, batch_size="auto",
            envelope_mm: float | None = 20.0, configuration: str | None = None,
            models=None, cancel=None, progress=None):
    """Segment an image with a task from the toolkit's catalog.

    ``image`` is a path to anything SimpleITK reads - NIfTI, NRRD, MetaImage, a DICOM series
    directory - or a SimpleITK image the caller already holds.

    ``task`` is a catalog name, a ``TaskSpec``, or a path to a stock nnU-Net result folder
    (``.../Dataset<id>_<name>/<trainer>__<plans>__<config>/``, or the dataset folder - then
    ``configuration`` picks among 2d / 3d_lowres / 3d_fullres, preferring 3d_fullres).

    ``convention`` defaults to ``"auto"``: ``"center"`` (skimage half-pixel, plus crop-to-nonzero)
    for nnU-Net-native models, ``"corner"`` (TotalSegmentator's ``change_spacing``, no crop) for
    the TS catalog - each model's own training-time preprocessing.

    ``device`` defaults to ``"auto"``: CUDA, then MPS, then CPU.

    ``resampling_order`` is the spline order of the forward resample; 3 (cubic) matches what
    nnU-Net trained the models with - TotalSegmentator v2.18 defaults to 1 for speed, which is
    a mild train/test mismatch that grows with the downsampling factor.

    ``envelope_mm`` restricts inference to the patient's bounding box (air removed, largest
    connected component, plus this margin in mm) - on a chest CT that is a third of the
    volume and ~3x fewer patches per model. The air cut is the CT -500 HU threshold for CT
    models and a data-driven (Otsu) split for per-image-normalized MRI. ``None`` runs the
    full volume.

    ``batch_size`` is patches per forward pass: an int, or ``"auto"`` - 1 on Apple silicon
    (measured fastest), 4 on CUDA when the measured working set says it fits (18 % faster
    steady-state on an A10); it only applies when the accumulator is on the device.

    ``cancel`` is a :class:`~nnseg.progress.CancelToken`; the run stops at the next patch
    boundary. ``progress`` is called with a :class:`~nnseg.progress.Progress` snapshot (which
    prints readably, so a ``lambda p: print(p)`` callback works).

    ``models`` is a :class:`~nnseg.cache.ModelCache`; pass one with ``capacity>=1`` (or use
    :class:`~nnseg.segmenter.Segmenter`) to keep models warm between calls instead of rebuilding
    them every time - the difference between a script and a server.

    ``accumulate`` picks where the sliding-window accumulator lives: ``"auto"`` (from the
    device's free memory), ``"device"`` (fastest, needs headroom), ``"host"``.

    Returns a :class:`~nnseg.result.Segmentation`: ``.labels`` is the label volume as a SimpleITK
    image in the *input's* orientation on the requested grid (``"input"`` = the input grid, a number =
    isotropic at that spacing, a ``Grid`` = as given). It also carries ``.array``, ``.mask(name)``,
    ``.present()``, ``.volumes_ml()``, ``.save(path)``, ``.timings`` and ``.provenance`` - what
    models, folds, device and preprocessing policy actually ran.
    Multi-model tasks composite at the label level in part order (later parts win).
    """
    from . import io as nio
    from .job import device_lock

    from .resample import resolve_device
    device = str(resolve_device(device))                  # "auto" -> cuda / mps / cpu, once
    report = Reporter.of(progress, cancel=cancel)
    _warm_restore_kernel(device)
    T: dict[str, float] = {}
    t0 = time.perf_counter()
    lock = device_lock(device)                            # reentrant: a Job already holds it
    catalog = catalog or TaskCatalog("totalsegmentator")
    models = models if models is not None else ModelCache()   # no caching unless asked
    spec = _resolve_spec(task, catalog)
    # nnU-Net-native models were trained on their own preprocessing: skimage's half-pixel
    # ("center") resample and crop-to-nonzero. TS bypasses both (corner-aligned change_spacing,
    # no crop). Getting this backwards is a silent geometry error, so it follows the task's
    # ecosystem unless the caller is explicit. See docs/resampler-parity-finding.md.
    native = _is_native(spec)
    store = as_store(weights, ecosystem="nnunet" if native else "totalsegmentator")
    if convention == "auto":
        convention = "center" if native else "corner"
    crop_nonzero = native
    # Orientation follows the model's own reader. TS canonicalizes to RAS; nnU-Net's default
    # SimpleITKIO/NibabelIO do NOT - only the *WithReorient variants do - so a native model
    # expects its acquisition orientation. Reorienting it anyway mirrors left/right.
    reorient = True
    if native and spec.single is not None:
        reorient = nio.reader_reorients(store.resolve(spec.single, configuration=configuration))
    schema = LabelSchema(names={int(k): str(v) for k, v in spec.label_map.items()})
    prov = {"task": spec.name, "source": spec.source, "device": device, "dtype": dtype,
            "convention": convention, "reoriented_to_ras": reorient, "interp": interp,
            "envelope_mm": envelope_mm, "resampling_order": resampling_order, "models": [],
            "weights_store": store.describe(),
            "nnseg": _version()}

    if isinstance(image, (str, Path)):
        data_zyx, geometry, orientation = nio.read(image, reorient=reorient)
    else:                                        # a SimpleITK image the caller already holds
        import SimpleITK as sitk
        orientation = nio.orientation_of(image)
        if reorient:
            image = sitk.DICOMOrient(image, nio.CANONICAL)
        data_zyx, geometry = sitk.GetArrayFromImage(image), nio.geometry_of(image)
    T["read+canonical"] = time.perf_counter() - t0

    labels = None
    out_grid = None
    frame: Frame | None = None
    cached = {}                                   # model spacing -> (x, frame)

    def load(wid):
        folder = store.resolve(wid, configuration=configuration)
        m = models.get(folder, folds=folds, device=device, dtype=dtype,
                       accumulate=accumulate, batch_size=batch_size)
        # the folder name does NOT identify the weights version - Dataset297 ships as both
        # v2.0.0 and v2.0.4 and both unpack to the same name - so read what fetch_one recorded
        from .weights_fetch import installed_version
        rec = installed_version(folder) or {}
        prov["models"].append({"weights": str(wid), "folder": folder.name,
                               "version": rec.get("tag", "unknown"), "sha256": rec.get("sha256"),
                               "folds": list(available_folds(folder, folds)), "K": m.K,
                               "spacing": tuple(round(v, 4) for v in m.spacing_zyx)})
        return m

    def model_frame(model):
        key = model.spacing_zyx
        if key not in cached:
            cached[key] = to_model_frame(data_zyx, geometry, model, convention=convention, device=device,
                                         order=resampling_order, original_orientation=orientation,
                                         crop_to_nonzero=crop_nonzero)
        return cached[key]

    def crop_on_model_grid(model, x, frame, *, use_body, roi_mm):
        """A voxel box on this model's grid: the body envelope (optional) intersected with a
        physical ROI (optional, from a coarse cascade stage). None means run the whole grid."""
        shape = tuple(int(s) for s in x.shape[1:])
        lo = [0, 0, 0]
        hi = list(shape)
        if use_body and envelope_mm is not None:
            xnp = x[0].numpy()
            thr = body_threshold(xnp, normalization_schemes=model.normalization_schemes,
                                 intensity_properties=model.intensity_properties(0))
            e = envelope_of(body_mask(xnp, threshold=thr),
                            margin_voxels=margin_in_voxels(envelope_mm, model.spacing_zyx))
            lo = [max(a, b) for a, b in zip(lo, e.lo)]
            hi = [min(a, b) for a, b in zip(hi, e.hi)]
        if roi_mm is not None:
            (lo_mm, hi_mm), dil = roi_mm
            # mm -> index on the grid the resampler actually consumed (== source unless
            # crop-to-nonzero moved its origin), then that grid's index -> model coordinate
            rf = frame.resampled_from
            fr = frame.forward_rule
            c0 = fr.apply(rf.mm_to_index(lo_mm))
            c1 = fr.apply(rf.mm_to_index(hi_mm))
            dv = margin_in_voxels(dil, model.spacing_zyx)
            for ax in range(3):
                a, b = sorted((c0[ax], c1[ax]))
                lo[ax] = max(lo[ax], int(np.floor(a)) - dv[ax])
                hi[ax] = min(hi[ax], int(np.ceil(b)) + 1 + dv[ax])
        lo = [max(0, v) for v in lo]
        hi = [min(n, v) for n, v in zip(shape, hi)]
        if any(h <= l for l, h in zip(lo, hi)):               # empty -> fall back to whole grid
            return Envelope((0, 0, 0), shape, shape)
        # a box that barely crops only re-tiles the window (churn, ~no speedup): run whole instead
        return worth_cropping(Envelope(tuple(lo), tuple(hi), shape))

    def predict_into(model, x, frame, ogrid, env, *, lut, paint, out):
        crop = x[(slice(None), *env.slices)] if not env.is_whole() else x
        logits = model.predict_logits(crop, report=report).to(device)
        mapping = frame.mapping(ogrid)
        if not env.is_whole():
            mapping = mapping >> Mapping((1.0, 1.0, 1.0), tuple(-float(v) for v in env.lo))
        to_labels(logits, ogrid, mapping, interp=interp, outside="background", lut=lut, paint=paint,
                  out=out, backend="auto")
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        del logits
        if device in ("cuda", "mps"):
            (torch.cuda if device == "cuda" else torch.mps).empty_cache()

    def run_single_or_union(spc, tag):
        parts = spc.parts
        report.n_parts = max(report.n_parts, len(parts))
        og = None
        out = None
        fr = None
        for i, (wid, remap, pname) in enumerate(parts):
            t = time.perf_counter()
            report.enter_part(i, f"{pname} ({wid})")
            model = load(wid)
            T[f"load:{tag}:{pname}"] = time.perf_counter() - t
            t = time.perf_counter()
            x, fr = model_frame(model)
            env = crop_on_model_grid(model, x, fr, use_body=True, roi_mm=None)
            if not env.is_whole():
                report.stage("preprocess", f"envelope {env.fraction * 100:.0f} % of the model grid")
            T[f"preprocess:{tag}:{pname}"] = time.perf_counter() - t
            if og is None:
                og = fr.resolve_grid(grid)
                max_label = max((int(v) for v in spc.label_map), default=255)
                out = torch.zeros(og.shape, dtype=torch.uint8 if max_label <= 255 else torch.uint16, device=device)
            t = time.perf_counter()
            report.stage("predict", pname)
            predict_into(model, x, fr, og, env, lut=_lut(model.K, remap), paint=len(parts) > 1, out=out)
            report.stage("restore", f"{'device' if model.accumulate_choice['on_device'] else 'host'} accumulator")
            T[f"network:{tag}:{pname}"] = time.perf_counter() - t
            models.release(model)
        return out, fr, og

    def run_cascade(spc, tag):
        roi_mm = None
        out = fr = og = None
        stages = spc.cascade
        report.n_parts = max(report.n_parts, len(stages))
        for i, step in enumerate(stages):
            last = i == len(stages) - 1
            if step.crop_from_task is not None:
                report.stage("cascade", f"{tag} stage {i + 1}/{len(stages)}: crop from {step.crop_from_task!r}")
                sub_labels, sub_fr, sub_og = run_task_canonical(catalog.get(step.crop_from_task), f"{tag}:{step.crop_from_task}")
                e = label_roi(sub_labels.cpu().numpy(), step.crop_to_classes,
                              margin_voxels=margin_in_voxels(step.dilation_mm, sub_og.spacing))
                roi_mm = None if e.is_whole() else ((tuple(float(v) for v in sub_og.index_to_mm(e.lo)),
                                                     tuple(float(v) for v in sub_og.index_to_mm([h - 1 for h in e.hi]))), step.dilation_mm)
                report.stage("cascade", f"ROI from {step.crop_from_task!r}: "
                             + ("whole volume" if roi_mm is None else f"{e.fraction * 100:.0f} % of the grid"))
                continue
            t = time.perf_counter()
            report.enter_part(i, f"{tag} stage {i + 1}/{len(stages)}: model {step.weights_id}"
                              + ("" if roi_mm is None else " (cropped)"))
            model = load(step.weights_id)
            x, fr = model_frame(model)
            T[f"load:{tag}:s{i}"] = time.perf_counter() - t
            env = crop_on_model_grid(model, x, fr, use_body=not last, roi_mm=roi_mm)
            og = fr.resolve_grid(grid)
            out = torch.zeros(og.shape, dtype=torch.uint8, device=device)
            t = time.perf_counter()
            predict_into(model, x, fr, og, env, lut=np.arange(model.K, dtype=np.int32), paint=False, out=out)
            T[f"network:{tag}:s{i}"] = time.perf_counter() - t
            if not last:
                e = label_roi(out.cpu().numpy(), step.crop_to_classes,
                              margin_voxels=margin_in_voxels(step.dilation_mm, og.spacing))
                roi_mm = None if e.is_whole() else ((tuple(float(v) for v in og.index_to_mm(e.lo)),
                                                     tuple(float(v) for v in og.index_to_mm([h - 1 for h in e.hi]))), step.dilation_mm)
                report.stage("cascade", "ROI: " + ("absent -> whole volume next" if roi_mm is None
                             else f"{e.fraction * 100:.0f} % of the grid, +{step.dilation_mm} mm"))
            models.release(model)
        return out, fr, og

    def run_task_canonical(spc, tag=""):
        return run_cascade(spc, tag or spc.name) if spc.shape == "cascade" else run_single_or_union(spc, tag or spc.name)

    with lock:
        labels, frame, out_grid = run_task_canonical(spec)
    t = time.perf_counter()
    # back to the input's own orientation: a permute + flip where the labels already live,
    # not a single-threaded DICOMOrient over the host copy
    arr, geo = nio.reorient(labels, frame.output_geometry(out_grid), frame.original_orientation)
    out_img = nio.to_image(arr, geo)
    T["to input orientation"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    prov.update(input_orientation=orientation, output_grid=tuple(out_grid.shape),
                cropped_to_nonzero=frame.model_source is not None)
    return Segmentation(labels=out_img, schema=schema, grid=out_grid, spec=spec,
                        timings=T, provenance=prov)
