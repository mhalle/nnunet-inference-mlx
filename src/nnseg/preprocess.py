"""Source image -> the model's input frame: canonical orientation, forward resample
(the frozen fork resampler, scipy-exact), nnU-Net normalization."""
from __future__ import annotations

import numpy as np
import torch
from labelgrid import Grid

from .frame import Frame


def load_canonical(path):
    """nibabel image in closest-canonical (RAS) orientation, plus the original for the undo."""
    import nibabel as nib
    img_orig = nib.load(str(path))
    img_can = nib.as_closest_canonical(img_orig)
    return img_can, img_orig


def undo_canonical(img_can, img_orig):
    """Inverse of ``as_closest_canonical`` (TotalSegmentator's ``undo_canonical``)."""
    from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
    return img_can.as_reoriented(ornt_transform(axcodes2ornt("RAS"), io_orientation(img_orig.affine)))


def forward_resample(data_xyz: np.ndarray, spacing_xyz, new_spacing_xyz, *, convention: str = "corner",
                     order: int = 3, device="mps", out_dtype=np.int32):
    """TotalSegmentator's ``change_spacing`` on the fork's GPU resampler: zoom = spacing /
    new_spacing, new shape = round(shape * zoom), scipy.zoom corner rule, order 3, edge
    mode, no anti-aliasing, then the dtype conversion TS applies (``astype`` = truncation)."""
    from nnunetv2.preprocessing.resampling.resample_gpu import resample_data_or_seg_to_shape_gpu
    zoom = np.asarray(spacing_xyz, dtype=np.float64) / np.asarray(new_spacing_xyz, dtype=np.float64)
    new_shape = tuple(int(round(o * z)) for o, z in zip(data_xyz.shape, zoom))
    # exactly TS's resample_img_torch: float32 in, shape-based corner mapping, spline prefilter, edge mode
    arr = np.ascontiguousarray(data_xyz).astype(np.float32)[None]
    out = resample_data_or_seg_to_shape_gpu(arr, new_shape, is_seg=False, device=device,
                                            convention=convention, order=order, mode="nearest", anti_alias=False)
    out = np.asarray(out)[0]
    return out.astype(out_dtype) if out_dtype is not None else out, new_shape


def normalize(data: np.ndarray, schemes, props: dict) -> np.ndarray:
    """nnU-Net's per-channel normalization for a single-channel input."""
    if tuple(schemes) != ("CTNormalization",):
        raise NotImplementedError(f"normalization {schemes!r}: only CTNormalization is implemented so far")
    x = np.asarray(data, dtype=np.float32)
    x = np.clip(x, props["percentile_00_5"], props["percentile_99_5"])
    return (x - props["mean"]) / max(props["std"], 1e-8)


def to_model_frame(img_can, model, *, convention: str = "corner", device="mps") -> tuple[torch.Tensor, Frame]:
    """Canonical nibabel image -> ``(x (1, Z, Y, X) float32 CPU, Frame)`` for ``model``.

    Mirrors TotalSegmentator's preprocessing for its models: float64 voxel values, resample
    to the model spacing with the corner rule, truncate to int32, then nnU-Net's normalization
    (CT: clip to the 0.5 / 99.5 percentiles, z-score). No crop-to-nonzero (TS-style).
    """
    spacing_xyz = tuple(float(z) for z in img_can.header.get_zooms()[:3])
    data_xyz = img_can.get_fdata(dtype=np.float64)
    model_spacing_xyz = tuple(model.spacing_zyx[::-1])
    res_xyz, new_shape_xyz = forward_resample(data_xyz, spacing_xyz, model_spacing_xyz, convention=convention, device=device)
    x_zyx = normalize(np.ascontiguousarray(res_xyz.T), model.normalization_schemes, model.intensity_properties(0))
    source = Grid(tuple(int(s) for s in data_xyz.shape[::-1]), spacing_xyz[::-1], (0.0, 0.0, 0.0))
    frame = Frame(source=source, model_shape=tuple(x_zyx.shape), model_spacing=tuple(model.spacing_zyx),
                  convention=convention, affine_canonical=np.asarray(img_can.affine))
    return torch.from_numpy(x_zyx)[None], frame
