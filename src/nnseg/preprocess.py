"""Source image -> the model's input frame: canonical orientation, forward resample
(the frozen fork resampler, scipy-exact), nnU-Net normalization."""
from __future__ import annotations

import numpy as np
import torch
from .grid import Grid

from .frame import Frame


def load_canonical(path):
    """Read through SimpleITK, reoriented to RAS: ``(array (Z, Y, X), geometry, orientation)``."""
    from .io import read
    return read(path)


# nnU-Net's preprocessor resamples training data with a cubic spline, and TotalSegmentator
# matched that at inference until v2.18 dropped the default to linear for speed. Cubic is what
# the models were trained on, and the mismatch grows with the downsampling factor (0.65 mm ->
# 3 mm is 4.6x, where linear reads only the 2 nearest samples per axis). On the GPU resampler
# cubic costs ~2.5-3.4 s instead of scipy's 16.6 s, so the speed argument does not apply here.
# Measured: order 1 vs 3 on the same weights moves small structures by ~1.4 % Dice
# (gallbladder 0.783). See docs/resampler-parity-finding.md.
DEFAULT_RESAMPLING_ORDER = 3


def forward_resample(data_zyx: np.ndarray, spacing_zyx, new_spacing_zyx, *, convention: str = "corner",
                     order: int = DEFAULT_RESAMPLING_ORDER, device="auto", out_dtype=np.int32):
    """TotalSegmentator's ``change_spacing`` semantics on :mod:`nnseg.resample`: new shape =
    round(shape * spacing / new_spacing), scipy.zoom corner rule, edge mode, no anti-aliasing,
    then the dtype conversion TS applies (``astype`` = truncation, not rounding)."""
    from .resample import resample_data, target_shape
    new_shape = target_shape(data_zyx.shape, spacing_zyx, new_spacing_zyx)
    out = resample_data(data_zyx, new_shape, convention=convention, order=order, mode="nearest",
                        device=device, out_dtype=out_dtype)
    return out, new_shape


def normalize(data: np.ndarray, schemes, props: dict) -> np.ndarray:
    """nnU-Net's per-channel normalization for a single-channel input."""
    if tuple(schemes) != ("CTNormalization",):
        raise NotImplementedError(f"normalization {schemes!r}: only CTNormalization is implemented so far")
    x = np.asarray(data, dtype=np.float32)
    x = np.clip(x, props["percentile_00_5"], props["percentile_99_5"])
    return (x - props["mean"]) / max(props["std"], 1e-8)


def to_model_frame(data_zyx, geometry, model, *, convention: str = "corner", device="auto",
                   order: int = DEFAULT_RESAMPLING_ORDER, original_orientation: str = "RAS") -> tuple[torch.Tensor, Frame]:
    """Canonical (RAS) array + geometry -> ``(x (1, Z, Y, X) float32 CPU, Frame)`` for ``model``.

    Resamples to the model spacing with the caller's convention (corner = TotalSegmentator's
    ``change_spacing``), then applies nnU-Net's normalization (CT: clip to the 0.5 / 99.5
    percentiles, z-score). No crop-to-nonzero (TS-style). Arrays stay in (Z, Y, X) throughout -
    SimpleITK hands them over that way, so there are no transposes.
    """
    spacing_zyx = tuple(float(s) for s in geometry.spacing_zyx)
    source_shape_zyx = tuple(int(s) for s in data_zyx.shape)
    res_zyx, _ = forward_resample(np.asarray(data_zyx, dtype=np.float32), spacing_zyx,
                                  tuple(model.spacing_zyx), convention=convention, device=device, order=order)
    x_zyx = normalize(res_zyx, model.normalization_schemes, model.intensity_properties(0))
    del res_zyx
    source = Grid(source_shape_zyx, spacing_zyx, (0.0, 0.0, 0.0))
    frame = Frame(source=source, model_shape=tuple(x_zyx.shape), model_spacing=tuple(model.spacing_zyx),
                  convention=convention, canonical=geometry, original_orientation=original_orientation)
    return torch.from_numpy(np.ascontiguousarray(x_zyx))[None], frame
