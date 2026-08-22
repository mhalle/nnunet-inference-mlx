"""segment(): the torch pipeline - task -> parts -> logits -> labels on the chosen grid."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import labelgrid as lg

from .frame import Frame
from .network import TorchModel, resolve_model_folder
from .preprocess import load_canonical, to_model_frame, undo_canonical


def _parts(spec):
    if spec.single is not None:
        return [(spec.single, None, spec.name)]
    if spec.union:
        return [(p.weights_id, dict(p.label_remap), p.name or str(p.weights_id)) for p in spec.union]
    raise NotImplementedError(f"task {spec.name!r}: cascades are not implemented in nnseg yet")


def _lut(K: int, remap: dict | None) -> np.ndarray:
    lut = np.arange(K, dtype=np.int64)
    if remap:
        lut[:] = 0
        for local, global_ in remap.items():
            lut[int(local)] = int(global_)
    return lut


def segment(image, task: str, *, catalog=None, model_root=None, device: str = "mps", dtype: str = "fp16",
            grid="input", interp="linear", outside: str = "background", convention: str = "corner",
            folds=(0,), accumulate: str = "auto", progress=None):
    """Segment a NIfTI path (or nibabel image) with a task from the toolkit's catalog.

    ``accumulate`` picks where the sliding-window accumulator lives: ``"auto"`` (from the
    device's free memory), ``"device"`` (fastest, needs headroom), ``"host"``.

    Returns ``(labels_img, schema, timings)``: a nibabel image of the labels in the *input's*
    orientation on the requested grid (``"input"`` = the input grid, a number = isotropic at
    that spacing, a ``labelgrid.Grid`` = as given), the label schema, and per-stage seconds.
    Multi-model tasks composite at the label level in part order (later parts win).
    """
    from nnunet_inference_mlx.catalog import TaskCatalog
    from nnunet_inference_mlx.values import LabelSchema
    import nibabel as nib

    say = progress or (lambda s: None)
    T: dict[str, float] = {}
    t0 = time.perf_counter()
    catalog = catalog or TaskCatalog("totalsegmentator")
    spec = catalog.get(task)
    parts = _parts(spec)
    schema = LabelSchema(names={int(k): str(v) for k, v in spec.label_map.items()})

    if isinstance(image, (str, Path)):
        img_can, img_orig = load_canonical(image)
    else:
        img_orig = image
        img_can = nib.as_closest_canonical(image)
    T["read+canonical"] = time.perf_counter() - t0

    labels = None
    out_grid = None
    frame: Frame | None = None
    cached = {}                                   # model spacing -> (x, frame)
    for i, (wid, remap, pname) in enumerate(parts):
        t = time.perf_counter()
        say(f"loading {pname} ({wid})")
        model = TorchModel(resolve_model_folder(wid, model_root=model_root), folds=folds, device=device,
                           dtype=dtype, accumulate=accumulate)
        T[f"load:{pname}"] = time.perf_counter() - t
        t = time.perf_counter()
        key = model.spacing_zyx
        if key not in cached:
            cached[key] = to_model_frame(img_can, model, convention=convention, device=device)
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
        logits = logits.to(device)
        lg.to_labels(logits, out_grid, frame.mapping(out_grid), interp=interp, outside=outside,
                     lut=_lut(model.K, remap), paint=len(parts) > 1, out=labels, backend="auto")
        if device == "mps":
            torch.mps.synchronize()
        T[f"restore:{pname}"] = time.perf_counter() - t
        del logits, model
        if device == "mps":
            torch.mps.empty_cache()

    t = time.perf_counter()
    arr_xyz = np.ascontiguousarray(labels.cpu().numpy().T)
    out_can = nib.Nifti1Image(arr_xyz, frame.output_affine(out_grid))
    out_img = undo_canonical(out_can, img_orig)
    T["to input orientation"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    return out_img, schema, T
