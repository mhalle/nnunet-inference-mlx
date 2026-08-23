"""segment(): the torch pipeline - task -> parts -> logits -> labels on the chosen grid."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .frame import Frame
from .restore import to_labels
from .network import TorchModel
from .tasks import TaskCatalog, resolve_model_folder
from .values import LabelSchema
from .preprocess import to_model_frame


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
            folds=(0,), accumulate: str = "auto", resampling_order: int = 3, batch_size: int = 1,
            prefetch: bool = True, progress=None):
    """Segment an image with a task from the toolkit's catalog.

    ``image`` is a path to anything SimpleITK reads - NIfTI, NRRD, MetaImage, a DICOM series
    directory - or a SimpleITK image the caller already holds.

    ``device`` defaults to ``"auto"``: CUDA, then MPS, then CPU.

    ``resampling_order`` is the spline order of the forward resample; 3 (cubic) matches what
    nnU-Net trained the models with - TotalSegmentator v2.18 defaults to 1 for speed, which is
    a mild train/test mismatch that grows with the downsampling factor.

    ``prefetch`` loads each part's model while the previous one predicts.

    ``batch_size`` is patches per forward pass - 1 on Apple silicon (measured fastest), a
    lever on CUDA cards with headroom; it only applies when the accumulator is on the device.

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
    parts = spec.parts
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
                          dtype=dtype, accumulate=accumulate, batch_size=batch_size)

    # Model loads are disk + CPU work that the GPU does not need: read the next part's
    # checkpoint while the current part predicts. Measured on an A10, five cold loads were
    # 11 % of a whole-body run with nothing else to do.
    pool = ThreadPoolExecutor(max_workers=1) if prefetch and len(parts) > 1 else None
    pending = pool.submit(load, parts[0][0]) if pool else None
    for i, (wid, remap, pname) in enumerate(parts):
        t = time.perf_counter()
        say(f"loading {pname} ({wid})")
        model = pending.result() if pending is not None else load(wid)
        if pool and i + 1 < len(parts):
            pending = pool.submit(load, parts[i + 1][0])
        T[f"load:{pname}"] = time.perf_counter() - t
        t = time.perf_counter()
        key = model.spacing_zyx
        if key not in cached:
            cached[key] = to_model_frame(data_zyx, geometry, model, convention=convention, device=device,
                                         order=resampling_order, original_orientation=orientation)
        x, frame = cached[key]
        T[f"preprocess:{pname}"] = time.perf_counter() - t
        if out_grid is None:
            out_grid = frame.resolve_grid(grid)
            max_label = max(int(v) for v in spec.label_map) if spec.label_map else 255
            labels = torch.zeros(out_grid.shape, dtype=torch.uint8 if max_label <= 255 else torch.uint16, device=device)
        t = time.perf_counter()
        say(f"predicting {pname} ({i + 1}/{len(parts)})")
        logits = model.predict_logits(x)
        say(f"accumulator: {'device' if model.accumulate_choice['on_device'] else 'host'} - {model.accumulate_choice['why']}")
        T[f"network:{pname}"] = time.perf_counter() - t
        t = time.perf_counter()
        # free the network before the restore: on a memory-tight device the weights and the
        # cached activations are dead weight while the logits need to be resident
        lut = _lut(model.K, remap)
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        logits = logits.to(device)
        to_labels(logits, out_grid, frame.mapping(out_grid), interp=interp, outside=outside,
                     lut=lut, paint=len(parts) > 1, out=labels, backend="auto")
        if device == "mps":
            torch.mps.synchronize()
        T[f"restore:{pname}"] = time.perf_counter() - t
        del logits
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    if pool:
        pool.shutdown(wait=False)
    t = time.perf_counter()
    # back to the input's own orientation: a permute + flip where the labels already live,
    # not a 4 s single-threaded DICOMOrient over the host copy
    arr, geo = nio.reorient(labels, frame.output_geometry(out_grid), frame.original_orientation)
    out_img = nio.to_image(arr, geo)
    T["to input orientation"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    return out_img, schema, T
