"""Source image -> the model's input frame: canonical orientation, forward resample
(the frozen fork resampler, scipy-exact), nnU-Net normalization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from .grid import Grid

from .errors import UnsupportedModel
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


def nonzero_box(data_zyx: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """nnU-Net's crop-to-nonzero box: ``(lo, hi)`` half-open, (Z, Y, X).

    nnU-Net crops every case to the bounding box of its nonzero voxels *before* resampling
    (``crop_to_nonzero``: ``data != 0``, holes filled, then the bbox). On CT this is usually
    the whole image; on MRI, where the background outside the acquired volume is exactly 0,
    it removes a real border. An all-zero image yields the whole grid rather than an empty
    box - the same fail-safe direction as the body envelope.
    """
    mask = np.asarray(data_zyx) != 0
    shape = tuple(int(s) for s in mask.shape)
    if not mask.any():
        return (0, 0, 0), shape
    try:                                    # nnU-Net fills holes so interior zeros do not matter;
        from scipy import ndimage           # for a bounding box it changes nothing, so it is optional
        mask = ndimage.binary_fill_holes(mask)
    except ImportError:
        pass
    idx = np.nonzero(mask)
    lo = tuple(int(i.min()) for i in idx)
    hi = tuple(int(i.max()) + 1 for i in idx)
    return lo, hi


def normalize(data: np.ndarray, schemes, props, *, use_mask_for_norm=None, seg=None) -> np.ndarray:
    """Normalize a single-channel image exactly as nnU-Net would, by delegating to nnU-Net's own
    normalization classes (named in ``plans``) - CTNormalization, ZScoreNormalization, etc.

    ``schemes`` is the plans' ``normalization_schemes`` (class names, e.g. "ZScoreNormalization");
    ``props`` the channel's ``foreground_intensity_properties`` (CT needs it, ZScore does not).
    ``use_mask_for_norm`` (from plans) selects masked ZScore for e.g. brain MRI; without a
    provided ``seg`` the nonzero region is used, matching nnU-Net's mask definition.

    Single channel only (Tier A); multi-channel MRI is a later step.
    """
    from nnunetv2.preprocessing.normalization import default_normalization_schemes as N
    names = list(schemes)
    if len(names) != 1:
        raise UnsupportedModel(f"multi-channel normalization {names!r} not supported yet (single channel only)")
    cls = getattr(N, names[0], None)
    if cls is None:
        raise UnsupportedModel(f"unknown normalization scheme {names[0]!r}")
    umn = bool(use_mask_for_norm[0]) if isinstance(use_mask_for_norm, (list, tuple)) else bool(use_mask_for_norm)
    # a fresh C-contiguous copy: run() mutates in place (must not touch the caller's array), and
    # mean/std on a non-contiguous view can round differently. np.array(copy=True) guarantees it.
    x = np.array(data, dtype=np.float32, order="C")[None]                # (C=1, Z, Y, X)
    if umn and seg is None:
        seg = np.where(x != 0, 0, -1).astype(np.int8)     # nnU-Net's nonzero mask: >= 0 is inside
    norm = cls(use_mask_for_norm=umn, intensityproperties=dict(props or {}), target_dtype=np.float32)
    return norm.run(x, seg)[0]


@dataclass(frozen=True)
class ResampledGrid:
    """A source volume cropped and resampled to a model spacing, with NO normalization applied.

    Deliberately a different type from a network input. nnU-Net's normalization is PER MODEL -
    CT clips to that dataset's foreground percentiles and z-scores by its foreground mean/std,
    all read from its own plans - so several models at one spacing can share this resample but
    must not share a normalized array. Keeping the shareable thing normalization-free, and
    making it un-feedable to a network, is what stops that mix-up being expressible.
    """

    data_zyx: np.ndarray
    frame: Frame


def normalization_fingerprint(model) -> tuple:
    """What distinguishes one model's normalization from another's.

    The five parts of TotalSegmentator's ``total`` illustrate the spread: the organs model
    clips CT to [-1024, 276] and z-scores by mean -370 / std 437, while the ribs model clips to
    [-110, 1302] around mean 292 / std 262. Feeding one model's normalization to another
    flattens everything above the first model's upper clip, which is silent and severe.
    """
    props = model.intensity_properties(0) or {}
    numeric = tuple(sorted((str(k), float(v)) for k, v in props.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)))
    umn = model.use_mask_for_norm
    umn = bool(umn[0]) if isinstance(umn, (list, tuple)) else bool(umn)
    return (tuple(model.normalization_schemes), umn, numeric)


def to_model_grid(data_zyx, geometry, spacing_zyx, *, convention: str = "corner", device="auto",
                  order: int = DEFAULT_RESAMPLING_ORDER, original_orientation: str = "RAS",
                  crop_to_nonzero: bool = False) -> ResampledGrid:
    """Canonical (RAS) array + geometry -> a ``ResampledGrid`` at ``spacing_zyx``.

    Optionally crops to the nonzero box first (``crop_to_nonzero=True``, nnU-Net-native), then
    resamples with the caller's convention (corner = TotalSegmentator's ``change_spacing``,
    center = skimage / nnU-Net's own resampler). Arrays stay in (Z, Y, X) throughout - SimpleITK
    hands them over that way, so there are no transposes.

    Everything here depends only on the geometry and the target spacing, never on which model
    asked, which is what makes the result safe to share between models at one spacing.

    The crop is recorded as ``Frame.model_source`` (a sub-grid with the crop offset as its
    origin), so output grids keep referring to the full source and the un-crop is implicit in
    the mapping - nothing has to be pasted back afterwards.
    """
    spacing_src_zyx = tuple(float(s) for s in geometry.spacing_zyx)
    source_shape_zyx = tuple(int(s) for s in data_zyx.shape)
    source = Grid(source_shape_zyx, spacing_src_zyx, (0.0, 0.0, 0.0))

    data_zyx = np.asarray(data_zyx, dtype=np.float32)
    model_source = None
    if crop_to_nonzero:
        lo, hi = nonzero_box(data_zyx)
        if (lo, hi) != ((0, 0, 0), source_shape_zyx):
            data_zyx = data_zyx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            model_source = Grid(tuple(h - l for l, h in zip(lo, hi)), spacing_src_zyx,
                                tuple(float(source.index_to_mm(lo)[a]) for a in range(3)))

    res_zyx, _ = forward_resample(data_zyx, spacing_src_zyx, tuple(spacing_zyx),
                                  convention=convention, device=device, order=order)
    frame = Frame(source=source, model_shape=tuple(res_zyx.shape), model_spacing=tuple(spacing_zyx),
                  convention=convention, canonical=geometry, original_orientation=original_orientation,
                  model_source=model_source)
    return ResampledGrid(res_zyx, frame)


def normalize_for(grid: ResampledGrid, model) -> torch.Tensor:
    """This model's network input: ``grid`` normalized with THIS model's statistics.

    Returns a fresh tensor - ``normalize`` copies, so the grid stays unnormalized and reusable -
    carrying a fingerprint of the normalization applied, which the consumer checks.
    """
    x_zyx = normalize(grid.data_zyx, model.normalization_schemes, model.intensity_properties(0),
                      use_mask_for_norm=model.use_mask_for_norm)
    x = torch.from_numpy(np.ascontiguousarray(x_zyx))[None]
    x._nnseg_normalization = normalization_fingerprint(model)
    return x


def to_model_frame(data_zyx, geometry, model, *, convention: str = "corner", device="auto",
                   order: int = DEFAULT_RESAMPLING_ORDER, original_orientation: str = "RAS",
                   crop_to_nonzero: bool = False) -> tuple[torch.Tensor, Frame]:
    """Canonical (RAS) array + geometry -> ``(x (1, Z, Y, X) float32 CPU, Frame)`` for ``model``.

    ``to_model_grid`` then ``normalize_for``, for a single model. Callers that run several
    models at one spacing should use those two directly, sharing the grid and normalizing per
    model; see ``pipeline.segment``.
    """
    grid = to_model_grid(data_zyx, geometry, model.spacing_zyx, convention=convention, device=device,
                         order=order, original_orientation=original_orientation,
                         crop_to_nonzero=crop_to_nonzero)
    return normalize_for(grid, model), grid.frame
