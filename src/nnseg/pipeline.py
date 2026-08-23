"""segment(): the torch pipeline - task -> parts -> logits -> labels on the chosen grid."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from .envelope import AIR_HU, Envelope, body_mask, envelope_of, margin_in_voxels
from .frame import Frame
from .mapping import Mapping
from .restore import to_labels
from .network import TorchModel
from .tasks import TaskCatalog, resolve_model_folder
from .values import LabelSchema
from .preprocess import to_model_frame


def _run_cascade(spec, load, model_frame, crop_on_model_grid, predict_into, *, frame_box, device, say, T):
    """Run a cascade: each stage but the last crops the next to the bounding box of its
    ``crop_to_classes`` (dilated). Only the last stage's labels are the result; earlier stages
    are discarded once their box is taken. Returns ``(array_zyx, geometry)`` in the input frame.

    A stage that only reuses another task's output (``crop_from_task``) is not supported yet and
    raises - none of the fetchable cascades need it.
    """
    import numpy as np
    import torch
    from .envelope import label_roi, margin_in_voxels
    roi_mm = None
    out_img = None
    frame = None
    for i, step in enumerate(spec.cascade):
        last = i == len(spec.cascade) - 1
        if step.weights_id is None:
            raise NotImplementedError(f"cascade stage {i} of {spec.name!r} crops from a task "
                                      f"({step.crop_from_task!r}); not supported yet")
        t = time.perf_counter()
        say(f"cascade stage {i + 1}/{len(spec.cascade)}: model {step.weights_id}"
            + ("" if roi_mm is None else " (cropped to the previous stage)"))
        model = load(step.weights_id)
        x, frame = model_frame(model)
        T[f"load:stage{i}"] = time.perf_counter() - t
        # coarse stages: run inside the body; the target stage: run inside the ROI only
        env = crop_on_model_grid(model, x, frame, use_body=not last, roi_mm=roi_mm)
        ogrid = frame.resolve_grid(frame_box)
        out = torch.zeros(ogrid.shape, dtype=torch.uint8, device=device)
        t = time.perf_counter()
        predict_into(model, x, frame, ogrid, env, lut=np.arange(model.K, dtype=np.int32), paint=False, out=out)
        T[f"network:stage{i}"] = time.perf_counter() - t
        if not last:
            arr = out.cpu().numpy()
            e = label_roi(arr, step.crop_to_classes, margin_voxels=margin_in_voxels(step.dilation_mm, ogrid.spacing))
            if e.is_whole():
                say(f"  crop classes {list(step.crop_to_classes)} absent/empty in the coarse output "
                    f"-> next stage runs the whole volume")
                roi_mm = None
            else:
                lo_mm = tuple(float(v) for v in ogrid.index_to_mm(e.lo))
                hi_mm = tuple(float(v) for v in ogrid.index_to_mm([h - 1 for h in e.hi]))
                roi_mm = ((lo_mm, hi_mm), step.dilation_mm)
                say(f"  ROI from classes {list(step.crop_to_classes)}: {e.fraction * 100:.0f} % of the grid, "
                    f"+{step.dilation_mm} mm for the next stage")
            del arr
        else:
            from . import io as nio
            out_img = nio.reorient(out, frame.output_geometry(ogrid), frame.original_orientation)
        del model, out
        if device in ("cuda", "mps"):
            (torch.cuda if device == "cuda" else torch.mps).empty_cache()
    return out_img


def _warm_restore_kernel(device: str) -> None:
    """Compile the fused CUDA kernel while the network runs, instead of inside the first restore."""
    if not str(device).startswith("cuda"):
        return
    from .backends import triton_gpu
    if not triton_gpu.available():
        return
    threading.Thread(target=triton_gpu.warmup, daemon=True).start()


def _lut(K: int, remap: dict | None) -> np.ndarray:
    lut = np.arange(K, dtype=np.int64)
    if remap:
        lut[:] = 0
        for local, global_ in remap.items():
            lut[int(local)] = int(global_)
    return lut


def segment(image, task: str, *, catalog=None, model_root=None, device: str = "auto", dtype: str = "fp16",
            grid="input", interp="linear", outside: str = "background", convention: str = "corner",
            folds=(0,), accumulate: str = "auto", resampling_order: int = 3, batch_size="auto",
            envelope_mm: float | None = 20.0, progress=None):
    """Segment an image with a task from the toolkit's catalog.

    ``image`` is a path to anything SimpleITK reads - NIfTI, NRRD, MetaImage, a DICOM series
    directory - or a SimpleITK image the caller already holds.

    ``device`` defaults to ``"auto"``: CUDA, then MPS, then CPU.

    ``resampling_order`` is the spline order of the forward resample; 3 (cubic) matches what
    nnU-Net trained the models with - TotalSegmentator v2.18 defaults to 1 for speed, which is
    a mild train/test mismatch that grows with the downsampling factor.

    ``envelope_mm`` restricts inference to the patient's bounding box (HU > -500, largest
    connected component, plus this margin in mm) - on a chest CT that is a third of the
    volume and ~3x fewer patches per model. ``None`` runs the full volume.

    ``batch_size`` is patches per forward pass: an int, or ``"auto"`` - 1 on Apple silicon
    (measured fastest), 4 on CUDA when the measured working set says it fits (18 % faster
    steady-state on an A10); it only applies when the accumulator is on the device.

    ``accumulate`` picks where the sliding-window accumulator lives: ``"auto"`` (from the
    device's free memory), ``"device"`` (fastest, needs headroom), ``"host"``.

    Returns ``(labels_img, schema, timings)``: a SimpleITK image of the labels in the *input's*
    orientation on the requested grid (``"input"`` = the input grid, a number = isotropic at
    that spacing, a ``Grid`` = as given), the label schema, and per-stage seconds.
    Multi-model tasks composite at the label level in part order (later parts win).
    """
    from . import io as nio

    from .resample import resolve_device
    device = str(resolve_device(device))                  # "auto" -> cuda / mps / cpu, once
    say = progress or (lambda s: None)
    _warm_restore_kernel(device)
    T: dict[str, float] = {}
    t0 = time.perf_counter()
    catalog = catalog or TaskCatalog("totalsegmentator")
    spec = catalog.get(task)
    schema = LabelSchema(names={int(k): str(v) for k, v in spec.label_map.items()})

    if isinstance(image, (str, Path)):
        data_zyx, geometry, orientation = nio.read(image)
    else:                                        # a SimpleITK image the caller already holds
        import SimpleITK as sitk
        orientation = nio.orientation_of(image)
        image = sitk.DICOMOrient(image, nio.CANONICAL)
        data_zyx, geometry = sitk.GetArrayFromImage(image), nio.geometry_of(image)
    T["read+canonical"] = time.perf_counter() - t0

    labels = None
    out_grid = None
    frame: Frame | None = None
    cached = {}                                   # model spacing -> (x, frame)

    def load(wid):
        return TorchModel(resolve_model_folder(wid, model_root=model_root), folds=folds, device=device,
                          dtype=dtype, accumulate=accumulate, batch_size=batch_size).to_device()

    def model_frame(model):
        key = model.spacing_zyx
        if key not in cached:
            cached[key] = to_model_frame(data_zyx, geometry, model, convention=convention, device=device,
                                         order=resampling_order, original_orientation=orientation)
        return cached[key]

    def crop_on_model_grid(model, x, frame, *, use_body, roi_mm):
        """A voxel box on this model's grid: the body envelope (optional) intersected with a
        physical ROI (optional, from a coarse cascade stage). None means run the whole grid."""
        shape = tuple(int(s) for s in x.shape[1:])
        lo = [0, 0, 0]
        hi = list(shape)
        if use_body and envelope_mm is not None:
            props = model.intensity_properties(0)
            z_thr = (max(AIR_HU, props["percentile_00_5"]) - props["mean"]) / max(props["std"], 1e-8)
            e = envelope_of(body_mask(x[0].numpy(), threshold=z_thr),
                            margin_voxels=margin_in_voxels(envelope_mm, model.spacing_zyx))
            lo = [max(a, b) for a, b in zip(lo, e.lo)]
            hi = [min(a, b) for a, b in zip(hi, e.hi)]
        if roi_mm is not None:
            (lo_mm, hi_mm), dil = roi_mm
            src_sp = np.asarray(frame.source.spacing)
            fr = frame.forward_rule
            c0 = fr.apply(np.asarray(lo_mm) / src_sp)
            c1 = fr.apply(np.asarray(hi_mm) / src_sp)
            dv = margin_in_voxels(dil, model.spacing_zyx)
            for ax in range(3):
                a, b = sorted((c0[ax], c1[ax]))
                lo[ax] = max(lo[ax], int(np.floor(a)) - dv[ax])
                hi[ax] = min(hi[ax], int(np.ceil(b)) + 1 + dv[ax])
        lo = [max(0, v) for v in lo]
        hi = [min(n, v) for n, v in zip(shape, hi)]
        if any(h <= l for l, h in zip(lo, hi)):               # empty -> fall back to whole grid
            return Envelope((0, 0, 0), shape, shape)
        return Envelope(tuple(lo), tuple(hi), shape)

    def predict_into(model, x, frame, ogrid, env, *, lut, paint, out):
        crop = x[(slice(None), *env.slices)] if not env.is_whole() else x
        logits = model.predict_logits(crop).to(device)
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

    if spec.shape == "cascade":
        result = _run_cascade(spec, load, model_frame, crop_on_model_grid, predict_into,
                              frame_box=grid, device=device, say=say, T=T)
        out_img = nio.to_image(*result)
        T["total"] = time.perf_counter() - t0
        return out_img, schema, T

    parts = spec.parts

    # single or union: every part paints into one shared output (union) or is the whole output
    # (single). Cascade returned earlier.
    for i, (wid, remap, pname) in enumerate(parts):
        t = time.perf_counter()
        say(f"loading {pname} ({wid})")
        model = load(wid)
        T[f"load:{pname}"] = time.perf_counter() - t
        t = time.perf_counter()
        x, frame = model_frame(model)
        env = crop_on_model_grid(model, x, frame, use_body=True, roi_mm=None)
        if not env.is_whole():
            say(f"envelope: {env.fraction * 100:.0f} % of the model grid ({env.lo} .. {env.hi})")
        T[f"preprocess:{pname}"] = time.perf_counter() - t
        if out_grid is None:
            out_grid = frame.resolve_grid(grid)
            max_label = max((int(v) for v in spec.label_map), default=255)
            labels = torch.zeros(out_grid.shape, dtype=torch.uint8 if max_label <= 255 else torch.uint16,
                                 device=device)
        t = time.perf_counter()
        say(f"predicting {pname} ({i + 1}/{len(parts)})")
        lut = _lut(model.K, remap)
        predict_into(model, x, frame, out_grid, env, lut=lut, paint=len(parts) > 1, out=labels)
        say(f"accumulator: {'device' if model.accumulate_choice['on_device'] else 'host'} - {model.accumulate_choice['why']}; {model.batch_choice['why']}")
        T[f"network:{pname}"] = time.perf_counter() - t
        del model
        if device in ("cuda", "mps"):
            (torch.cuda if device == "cuda" else torch.mps).empty_cache()

    t = time.perf_counter()
    # back to the input's own orientation: a permute + flip where the labels already live,
    # not a 4 s single-threaded DICOMOrient over the host copy
    arr, geo = nio.reorient(labels, frame.output_geometry(out_grid), frame.original_orientation)
    out_img = nio.to_image(arr, geo)
    T["to input orientation"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    return out_img, schema, T
