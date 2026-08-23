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


# nnU-Net's preprocessor resamples training data with a cubic spline, and TotalSegmentator
# matched that at inference until v2.18 dropped the default to linear for speed. Cubic is what
# the models were trained on, and the mismatch grows with the downsampling factor (0.65 mm ->
# 3 mm is 4.6x, where linear reads only the 2 nearest samples per axis). On the GPU resampler
# cubic costs ~2.5-3.4 s instead of scipy's 16.6 s, so the speed argument does not apply here.
# Measured: order 1 vs 3 on the same weights moves small structures by ~1.4 % Dice
# (gallbladder 0.783). See docs/resampler-parity-finding.md.
DEFAULT_RESAMPLING_ORDER = 3


def forward_resample(data_xyz: np.ndarray, spacing_xyz, new_spacing_xyz, *, convention: str = "corner",
                     order: int = DEFAULT_RESAMPLING_ORDER, device="mps", out_dtype=np.int32):
    """TotalSegmentator's ``change_spacing`` on the fork's GPU resampler: zoom = spacing /
    new_spacing, new shape = round(shape * zoom), scipy.zoom corner rule, order 3, edge
    mode, no anti-aliasing, then the dtype conversion TS applies (``astype`` = truncation)."""
    from nnunetv2.preprocessing.resampling.resample_gpu import resample_data_or_seg_to_shape_gpu
    zoom = np.asarray(spacing_xyz, dtype=np.float64) / np.asarray(new_spacing_xyz, dtype=np.float64)
    new_shape = tuple(int(round(o * z)) for o, z in zip(data_xyz.shape, zoom))
    # exactly TS's resample_img_torch: float32 in, shape-based corner mapping, spline prefilter, edge mode
    arr = np.ascontiguousarray(data_xyz, dtype=np.float32)[None]        # no copy when already float32 C-order
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


def to_model_frame(img_can, model, *, convention: str = "corner", device="mps",
                   order: int = DEFAULT_RESAMPLING_ORDER) -> tuple[torch.Tensor, Frame]:
    """Canonical nibabel image -> ``(x (1, Z, Y, X) float32 CPU, Frame)`` for ``model``.

    Mirrors TotalSegmentator's preprocessing for its models: float64 voxel values, resample
    to the model spacing with the corner rule, truncate to int32, then nnU-Net's normalization
    (CT: clip to the 0.5 / 99.5 percentiles, z-score). No crop-to-nonzero (TS-style).
    """
    spacing_xyz = tuple(float(z) for z in img_can.header.get_zooms()[:3])
    # float32, not TotalSegmentator's float64 get_fdata: the resampler casts to float32 anyway,
    # and on a whole-body native grid the float64 copy alone is 3.3 GB of host memory.
    data_xyz = np.asanyarray(img_can.dataobj).astype(np.float32, copy=False)
    source_shape_xyz = data_xyz.shape
    model_spacing_xyz = tuple(model.spacing_zyx[::-1])
    res_xyz, new_shape_xyz = forward_resample(data_xyz, spacing_xyz, model_spacing_xyz, convention=convention,
                                              device=device, order=order)
    del data_xyz
    x_zyx = normalize(np.ascontiguousarray(res_xyz.T), model.normalization_schemes, model.intensity_properties(0))
    del res_xyz
    source = Grid(tuple(int(s) for s in source_shape_xyz[::-1]), spacing_xyz[::-1], (0.0, 0.0, 0.0))
    frame = Frame(source=source, model_shape=tuple(x_zyx.shape), model_spacing=tuple(model.spacing_zyx),
                  convention=convention, affine_canonical=np.asarray(img_can.affine))
    return torch.from_numpy(x_zyx)[None], frame
