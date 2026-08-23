"""Forward resampling: image intensities from the acquisition grid onto the model grid.

The other leg from :mod:`nnseg.restore`. Exactness is the whole point - a network sees what
its training preprocessing produced, so this reproduces the CPU resamplers rather than
approximating them, and does it on the GPU:

* ``convention="corner"`` - ``scipy.ndimage.zoom(grid_mode=False)``, the voxel-corner point
  grid, which is what TotalSegmentator's ``change_spacing`` uses.
* ``convention="center"`` - ``scipy.ndimage.zoom(grid_mode=True)`` == ``skimage.resize``, the
  voxel-center (half-pixel) grid, which is what nnU-Net's own
  ``resample_data_or_seg_to_shape`` uses.

Neither is re-implemented. ``zoom`` is linear and separable, so zooming an identity matrix
along one axis *is* the operator for that axis - spline prefilter, boundary mode and
coordinate convention included, for any order. We build those matrices with scipy once (they
are tiny and cached) and apply them on the GPU, which is exact by construction: there is no
boundary handling here to get subtly wrong.

Ported from the nnU-Net fork's ``resample_data_or_seg_to_shape_gpu`` (tag ``resample-gpu-v1``,
draft PR mhalle/nnUNet#1), keeping only the intensity path so nnseg does not pin a fork. The
fork's label / one-hot / separate-z / anti-aliased paths stay there; anti-aliasing in
particular is a distribution shift for models trained on the scipy pipeline
(``docs/resampler-parity-finding.md``) and nnseg does not want it.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

CONVENTIONS = ("corner", "center")


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(spec="auto") -> torch.device:
    """``"auto"`` picks the best available accelerator; anything else is taken literally.

    The default must not name a vendor: nnseg exists to be portable, and a hard-coded "mps"
    made it fail on the first CUDA machine it ever met.
    """
    if spec is None or spec == "auto":
        return best_device()
    return torch.device(spec)


@functools.lru_cache(maxsize=128)
def scipy_axis_matrix(n_in: int, n_out: int, order: int, mode: str, grid_mode: bool) -> np.ndarray:
    """The exact 1-D operator of ``scipy.ndimage.zoom`` along one axis, ``(n_out, n_in)`` float64."""
    from scipy import ndimage
    if n_in == n_out:
        return np.eye(n_in)
    probe = ndimage.zoom(np.eye(n_in, dtype=np.float64), (1.0, n_out / n_in),
                         order=order, mode=mode, grid_mode=grid_mode)
    if probe.shape != (n_in, n_out):
        raise RuntimeError(f"scipy zoom probe produced {probe.shape}, expected {(n_in, n_out)}")
    return np.ascontiguousarray(probe.T)


def _apply_axis(x: torch.Tensor, axis: int, w: torch.Tensor) -> torch.Tensor:
    """Apply an ``(n_out, n_in)`` operator along ``axis`` via matmul."""
    x = x.movedim(axis, -1)
    shp = x.shape
    out = (x.reshape(-1, shp[-1]) @ w.t()).reshape(*shp[:-1], w.shape[0])
    return out.movedim(-1, axis)


def target_shape(shape_zyx, spacing_zyx, new_spacing_zyx) -> tuple[int, int, int]:
    """``round(shape * spacing / new_spacing)`` - TotalSegmentator's ``change_spacing`` rule."""
    zoom = np.asarray(spacing_zyx, dtype=np.float64) / np.asarray(new_spacing_zyx, dtype=np.float64)
    return tuple(max(1, int(round(float(s) * float(z)))) for s, z in zip(shape_zyx, zoom))


@torch.no_grad()
def resample_data(data_zyx, new_shape=None, *, spacing_zyx=None, new_spacing_zyx=None,
                  convention: str = "corner", order: int = 3, mode: str = "nearest",
                  device=None, out_dtype=None, clip: bool | None = None) -> np.ndarray:
    """Resample a 3-D intensity volume to ``new_shape`` (or to ``new_spacing_zyx``).

    ``mode`` is a **scipy** boundary name (``"nearest"`` is what skimage spells ``"edge"``).

    ``clip`` bounds the output to the input's value range, which is what ``skimage.resize``
    does per call and nnU-Net inherits; ``scipy.ndimage.zoom`` does not clip. The default
    follows the convention being reproduced (on for ``center``, off for ``corner``) so that
    each mode matches its reference exactly. ``out_dtype`` applies the caller's cast at the
    end - note that TotalSegmentator uses ``astype``, i.e. truncation, not rounding.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}; got {convention!r}")
    arr = np.asarray(data_zyx)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D volume; got shape {arr.shape}")
    if new_shape is None:
        if spacing_zyx is None or new_spacing_zyx is None:
            raise ValueError("pass new_shape, or both spacing_zyx and new_spacing_zyx")
        new_shape = target_shape(arr.shape, spacing_zyx, new_spacing_zyx)
    new_shape = tuple(int(s) for s in new_shape)
    if len(new_shape) != 3:
        raise ValueError(f"new_shape must be (Z, Y, X); got {new_shape}")
    if clip is None:
        clip = convention == "center"

    dev = torch.device(device) if device is not None else best_device()
    # float64 is unsupported on MPS and unnecessary here: the operators are float32-exact to
    # ~1e-7 relative, far below the intensity quantization these volumes carry.
    work = torch.float32 if dev.type != "cpu" else (torch.float64 if arr.dtype == np.float64 else torch.float32)
    t = torch.as_tensor(np.ascontiguousarray(arr), device=dev, dtype=work)
    lo, hi = (t.amin(), t.amax()) if clip else (None, None)
    grid_mode = convention == "center"
    did_spline = False
    for axis in range(3):
        n_in, n_out = t.shape[axis], new_shape[axis]
        if n_in == n_out:
            continue
        w = torch.as_tensor(scipy_axis_matrix(int(n_in), int(n_out), int(order), str(mode), grid_mode),
                            device=dev, dtype=work)
        t = _apply_axis(t, axis, w)
        did_spline = did_spline or order >= 2
    if clip and did_spline:
        t = torch.clamp(t, lo, hi)
    out = t.cpu().numpy()
    del t
    return out.astype(out_dtype) if out_dtype is not None else out
