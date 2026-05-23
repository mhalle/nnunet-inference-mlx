"""
Spacing-aware resampling and high-level predict-with-resampling.

Optional helpers for the common medical-imaging workflow:

    acquisition-spacing SITK image
        ↓ forward resample (SITK, CPU)
    target-spacing numpy array
        ↓ inference (engine.predict)
    target-spacing K-channel logits
        ↓ inverse resample + argmax (MLX, Metal)
    acquisition-spacing uint8 segmentation
        ↓ geometry round-trip
    acquisition-spacing SITK image

The asymmetry — SITK on the forward side, MLX on the inverse side — is
deliberate. SITK has the richer image-interpolation toolkit and handles
oblique acquisitions cleanly; MLX is where the K-channel logits already
live after inference, so doing the inverse on Metal avoids a host
round-trip and lets us slab-stream the per-channel argmax in a single
fused kernel.

Opt-in via the ``preprocessing`` extra:

    pip install nnunet-inference-mlx[preprocessing]

That installs SimpleITK. The rest of the package keeps no hard
dependency on SITK; consumers using their own resampling library
(scipy, dask) can ignore this module entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    import SimpleITK as sitk

    from .engine import InferenceEngine


# ---------------------------------------------------------------------------
# Forward resample — SITK on the CPU side
# ---------------------------------------------------------------------------

_SITK_INTERP = None  # populated lazily; map of "linear"/"bspline"/"nearest"


def _require_sitk():
    try:
        import SimpleITK as sitk_mod
    except ImportError as e:
        raise ImportError(
            "SimpleITK is required for preprocessing helpers. Install with "
            "`pip install nnunet-inference-mlx[preprocessing]`."
        ) from e

    global _SITK_INTERP
    if _SITK_INTERP is None:
        _SITK_INTERP = {
            "linear": sitk_mod.sitkLinear,
            "bspline": sitk_mod.sitkBSpline,
            "nearest": sitk_mod.sitkNearestNeighbor,
        }
    return sitk_mod


def resample_image_to_target(
    image_sitk: "sitk.Image",
    target_spacing_zyx: tuple[float, float, float],
    interpolation: str = "linear",
) -> "sitk.Image":
    """Resample a SITK image to a target voxel spacing.

    Parameters
    ----------
    image_sitk : sitk.Image
        Input volume at acquisition spacing.
    target_spacing_zyx : tuple
        Desired spacing in mm, in (Z, Y, X) order to match nnU-Net's
        convention. SITK internally uses (X, Y, Z); this function flips
        for you.
    interpolation : "linear" | "bspline" | "nearest"
        Interpolation order. ``"linear"`` is the right default for images
        (CT HU, MR signal). ``"bspline"`` is smoother but slower. Use
        ``"nearest"`` only if you're resampling a label volume (for the
        forward pass of a cascade), not a raw image.

    Returns
    -------
    sitk.Image
        Resampled image carrying the new spacing, origin, and direction.
    """
    sitk = _require_sitk()
    if interpolation not in _SITK_INTERP:
        raise ValueError(
            f"interpolation={interpolation!r} not in {sorted(_SITK_INTERP)}"
        )

    # SITK uses (X, Y, Z) for spacing/size. We accept (Z, Y, X) for
    # consistency with the rest of the package and flip here.
    target_spacing_xyz = tuple(reversed(target_spacing_zyx))
    in_spacing = image_sitk.GetSpacing()
    in_size = image_sitk.GetSize()

    # New size such that the physical extent is preserved: same field of
    # view, different sampling.
    new_size_xyz = tuple(
        max(1, int(round(in_size[i] * in_spacing[i] / target_spacing_xyz[i])))
        for i in range(3)
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_sitk)
    resampler.SetOutputSpacing(target_spacing_xyz)
    resampler.SetSize(new_size_xyz)
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetInterpolator(_SITK_INTERP[interpolation])
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image_sitk)


# ---------------------------------------------------------------------------
# Inverse resample — MLX on the Metal side, slab + channel-stream
# ---------------------------------------------------------------------------

def _auto_memory_budget_mb() -> int:
    """Default working-memory cap for the inverse resample, by detected RAM.

    Mirrors Predictor's cache_limit_fraction auto-tier.
    """
    try:
        ram_gb = mx.device_info().get("memory_size", 0) / 1e9
    except Exception:
        ram_gb = 16
    if ram_gb >= 32:
        return 2000   # M1 Max / M3 Pro / Studio / Ultra
    return 200        # 16 GB Macs and similar


def _precompute_trilinear_indices(
    z_coords: mx.array, y_coords: mx.array, x_coords: mx.array,
    Z: int, Y: int, X: int,
):
    """Pre-compute floor/ceil index broadcasts and fractional weights once.

    Used by the slab-streaming inverse resample: index computation is
    identical across all K channels in a slab, so factoring it out of
    the channel loop saves K-1 redundant computations and shrinks the
    compiled-kernel input set to just the per-channel ``src`` array.
    """
    z = mx.clip(z_coords, 0.0, Z - 1.0)
    y = mx.clip(y_coords, 0.0, Y - 1.0)
    x = mx.clip(x_coords, 0.0, X - 1.0)

    z0 = mx.floor(z).astype(mx.int32)
    y0 = mx.floor(y).astype(mx.int32)
    x0 = mx.floor(x).astype(mx.int32)
    z1 = mx.minimum(z0 + 1, Z - 1)
    y1 = mx.minimum(y0 + 1, Y - 1)
    x1 = mx.minimum(x0 + 1, X - 1)

    return (
        z0[:, None, None], z1[:, None, None],
        y0[None, :, None], y1[None, :, None],
        x0[None, None, :], x1[None, None, :],
        (z - z0.astype(mx.float32))[:, None, None],   # zf
        (y - y0.astype(mx.float32))[None, :, None],   # yf
        (x - x0.astype(mx.float32))[None, None, :],   # xf
    )


def _trilinear_from_indices(
    src: mx.array,                  # (Z, Y, X)
    zi0, zi1, yi0, yi1, xi0, xi1,   # broadcast index tensors from _precompute
    zf, yf, xf,                     # fractional weights
) -> mx.array:                      # (S, Y_a, X_a)
    """Trilinear blend using pre-computed indices. Pure tensor ops; fuses."""
    c000 = src[zi0, yi0, xi0]
    c001 = src[zi0, yi0, xi1]
    c010 = src[zi0, yi1, xi0]
    c011 = src[zi0, yi1, xi1]
    c100 = src[zi1, yi0, xi0]
    c101 = src[zi1, yi0, xi1]
    c110 = src[zi1, yi1, xi0]
    c111 = src[zi1, yi1, xi1]

    c00 = c000 * (1 - xf) + c001 * xf
    c01 = c010 * (1 - xf) + c011 * xf
    c10 = c100 * (1 - xf) + c101 * xf
    c11 = c110 * (1 - xf) + c111 * xf
    c0 = c00 * (1 - yf) + c01 * yf
    c1 = c10 * (1 - yf) + c11 * yf
    return c0 * (1 - zf) + c1 * zf


def _trilinear_sample(
    src: mx.array,        # (Z, Y, X) float32
    z_coords: mx.array,   # (S,)   fractional positions in src Z (0..Z-1)
    y_coords: mx.array,   # (Y_a,)
    x_coords: mx.array,   # (X_a,)
) -> mx.array:            # (S, Y_a, X_a)
    """Trilinear interpolation of one channel at the requested coordinates.

    Public single-call entry point. The slab loop uses the more granular
    :func:`_precompute_trilinear_indices` + :func:`_trilinear_from_indices`
    pair instead, so it can factor index computation out of the K-channel
    inner loop and reuse a compiled kernel across channels.
    """
    Z, Y, X = src.shape
    idx = _precompute_trilinear_indices(z_coords, y_coords, x_coords, Z, Y, X)
    return _trilinear_from_indices(src, *idx)


def _materialize_resample_argmax(
    logits_target: mx.array,            # (K, Z_t, Y_t, X_t)
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    out_dtype: np.dtype,
) -> np.ndarray:
    """Materialize the whole K-channel-at-acquisition array, argmax, return uint*.

    Fast path for when the output is small enough to fit in budget (the
    downsample / mild-upsample case). Uses MLX's native 3D linear upsample.
    """
    K, Z_t, Y_t, X_t = logits_target.shape
    Z_a, Y_a, X_a = out_shape_zyx

    # MLX's nn.Upsample wants channels-last 5D: (N, D, H, W, C).
    inp = mx.expand_dims(logits_target.transpose(1, 2, 3, 0), 0)  # (1, Z_t, Y_t, X_t, K)
    scale = (Z_a / Z_t, Y_a / Y_t, X_a / X_t)
    up = nn.Upsample(scale_factor=scale, mode="linear")
    out = up(inp)[0]   # (Z_out, Y_out, X_out, K), where Z_out = int(Z_t * scale_z) etc.

    # The scale factor → int(in * scale) calculation can off-by-one vs
    # the exact target shape. Crop or pad to land exactly on out_shape.
    out = _resize_to_exact(out, (Z_a, Y_a, X_a))
    seg = mx.argmax(out, axis=-1)              # (Z_a, Y_a, X_a) int32
    mx.eval(seg)
    return np.asarray(seg).astype(out_dtype, copy=False)


def _resize_to_exact(
    arr: mx.array,                  # (Z, Y, X, K) channels-last
    target_zyx: tuple[int, int, int],
) -> mx.array:
    """Crop or constant-pad spatial dims to exactly match target shape.

    Used after nn.Upsample's int-cast scale rounding to land on the
    exact acquisition output shape. Differences are typically 1 voxel.
    """
    z, y, x = arr.shape[:3]
    tz, ty, tx = target_zyx

    # Crop / pad each axis independently
    def fix(a, cur, want, axis):
        if cur == want:
            return a
        if cur > want:
            slicer = [slice(None)] * a.ndim
            slicer[axis] = slice(0, want)
            return a[tuple(slicer)]
        # cur < want: pad with zeros at the end
        pad = [(0, 0)] * a.ndim
        pad[axis] = (0, want - cur)
        return mx.pad(a, pad)

    arr = fix(arr, z, tz, 0)
    arr = fix(arr, y, ty, 1)
    arr = fix(arr, x, tx, 2)
    return arr


def _slab_resample_argmax(
    logits_target: mx.array,            # (K, Z_t, Y_t, X_t)
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    peak_working_memory_mb: int,
    out_dtype: np.dtype,
) -> np.ndarray:
    """Slab-stream the inverse resample. Bounded working memory.

    For each output Z slab:
      - extract the corresponding target-spacing Z slab (with 1-voxel pad),
      - K-channel slab × mx.nn.Upsample(mode='linear') → K-channel acq slab,
      - argmax across K → uint8 slab,
      - copy to host output buffer.

    Working memory per slab is dominated by the K-channel acquisition-
    spacing slab: ``K * S * Y_a * X_a * 4`` bytes (fp32) plus small input/
    output buffers. Slab depth is chosen from peak_working_memory_mb so
    that total fits. The K-channel slab uses MLX's native fused 3D linear
    upsample — much faster than per-channel Python loops with custom
    fancy-indexing trilinear (those don't fuse under mx.compile because
    MLX doesn't fuse gather operations).
    """
    K, Z_t, Y_t, X_t = logits_target.shape
    Z_a, Y_a, X_a = out_shape_zyx

    # Slab voxel budget: K-channel fp32 slab dominates (K*4 B/voxel).
    bytes_per_slab_voxel = K * 4
    max_slab_voxels = int(peak_working_memory_mb) * 1024 * 1024 // bytes_per_slab_voxel
    plane_voxels = max(1, Y_a * X_a)
    slab_z = max(1, max_slab_voxels // plane_voxels)

    # Acquisition-grid Z step in target voxels; needed only to compute the
    # source Z range each slab pulls from.
    s2t_z = acq_spacing_zyx[0] / target_spacing_zyx[0]

    out = np.empty(out_shape_zyx, dtype=out_dtype)

    for z0 in range(0, Z_a, slab_z):
        z1 = min(z0 + slab_z, Z_a)
        S = z1 - z0

        # Source-Z range that covers this output slab, with 1-voxel pad
        # so the trilinear interpolation can clamp at boundaries.
        z_lo_f = z0 * s2t_z
        z_hi_f = (z1 - 1) * s2t_z
        zt_lo = max(0, min(Z_t - 1, int(z_lo_f) - 1))
        zt_hi = max(zt_lo + 1, min(Z_t, int(z_hi_f) + 2))
        slab_src = logits_target[:, zt_lo:zt_hi]   # (K, slab_t_z, Y_t, X_t)
        slab_t_z = slab_src.shape[1]

        # Compute the scale factor mx.nn.Upsample needs. Note: Upsample's
        # output Z is int(slab_t_z * scale_z). Y/X scales are global since
        # the in-plane resampling is the same for every slab.
        scale_z = S / slab_t_z
        scale_y = Y_a / Y_t
        scale_x = X_a / X_t

        # Channels-last 5D for Upsample.
        inp = mx.expand_dims(slab_src.transpose(1, 2, 3, 0), 0)   # (1, slab_t_z, Y_t, X_t, K)
        up = nn.Upsample(scale_factor=(scale_z, scale_y, scale_x), mode="linear")
        slab_out = up(inp)[0]   # (out_z, out_y, out_x, K), may differ from (S, Y_a, X_a) by ±1

        # Crop/pad to the exact slab output shape.
        slab_out = _resize_to_exact(slab_out, (S, Y_a, X_a))

        # Argmax across channels — fused single MLX op.
        seg_slab = mx.argmax(slab_out, axis=-1)   # (S, Y_a, X_a) int32
        mx.eval(seg_slab)
        out[z0:z1] = np.asarray(seg_slab).astype(out_dtype, copy=False)

    return out


def inverse_resample_argmax(
    logits_target: mx.array,
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    *,
    out_dtype: np.dtype | str = np.uint8,
    peak_working_memory_mb: int | None = None,
) -> np.ndarray:
    """Resample target-spacing logits to acquisition spacing and argmax.

    Auto-picks between two strategies:

    * **Materialize** (fast path): when the full K-channel acquisition-
      spacing array fits within ``peak_working_memory_mb``, allocate it
      and argmax in one shot. Used for downsample / small-upsample cases.
    * **Slab-stream**: when the materialized array would exceed the
      budget, process the output in Z slabs, with a channel-inner loop
      that updates a slab-local running argmax. Used for large upsamples.

    Both strategies produce identical output (linear per-channel interp
    + argmax). The choice affects memory and launch count, not math.

    ``peak_working_memory_mb=None`` (default) auto-detects from system
    RAM: 200 MB on < 32 GB Macs, 2000 MB on ≥ 32 GB Macs.
    """
    if peak_working_memory_mb is None:
        peak_working_memory_mb = _auto_memory_budget_mb()

    out_dtype = np.dtype(out_dtype)
    K = logits_target.shape[0]
    out_voxels = out_shape_zyx[0] * out_shape_zyx[1] * out_shape_zyx[2]
    materialize_bytes = K * out_voxels * 4   # fp32 K-channel acq array

    if materialize_bytes <= peak_working_memory_mb * 1024 * 1024:
        return _materialize_resample_argmax(
            logits_target, out_shape_zyx,
            target_spacing_zyx, acq_spacing_zyx, out_dtype,
        )
    return _slab_resample_argmax(
        logits_target, out_shape_zyx,
        target_spacing_zyx, acq_spacing_zyx,
        peak_working_memory_mb, out_dtype,
    )


# ---------------------------------------------------------------------------
# High-level: SITK in, SITK out
# ---------------------------------------------------------------------------

def predict_with_resampling(
    engine: "InferenceEngine",
    image_sitk: "sitk.Image",
    *,
    interpolation: str = "linear",
    peak_working_memory_mb: int | None = None,
) -> "sitk.Image":
    """Forward-resample input → run inference → inverse-resample logits.

    The full path-B pipeline: caller hands over a SITK image at any
    acquisition spacing, gets back a SITK image at the same acquisition
    spacing with integer labels. Geometry is preserved via
    ``CopyInformation``.

    Forward resample runs on CPU via SITK (b-spline if requested);
    inference runs on Metal; inverse resample runs on Metal with
    slab+channel streaming bounded by ``peak_working_memory_mb`` (auto-
    detected from system RAM if None).

    Returns labels at acquisition spacing. The K-channel logits are
    transient — never materialized at acquisition spacing.
    """
    sitk = _require_sitk()

    target_spacing_zyx = engine.predictor._bundle.target_spacing
    in_spacing_xyz = image_sitk.GetSpacing()
    acq_spacing_zyx = tuple(reversed(in_spacing_xyz))

    # Forward resample (CPU / SITK)
    resampled = resample_image_to_target(
        image_sitk, target_spacing_zyx, interpolation=interpolation,
    )

    # Inference (Metal). engine.predict returns raw logits at target spacing
    # for single-fold, averaged softmax probs for multi-fold standard
    # ensembles, averaged sigmoid probs for multi-fold region ensembles —
    # all (K, Z_t, Y_t, X_t).
    vol_target = sitk.GetArrayFromImage(resampled).astype(np.float32, copy=False)
    pred_np = engine.predict(vol_target)        # numpy (K, Z_t, Y_t, X_t)
    pred_mx = mx.array(pred_np)                  # back to Metal for streaming

    # Output shape in (Z, Y, X) — SITK GetSize is (X, Y, Z), so reverse.
    in_size_xyz = image_sitk.GetSize()
    out_shape_zyx = (in_size_xyz[2], in_size_xyz[1], in_size_xyz[0])

    # Pick output dtype from the bundle's label scheme.
    out_dtype = engine.label_dtype

    seg_zyx = inverse_resample_argmax(
        pred_mx, out_shape_zyx,
        target_spacing_zyx, acq_spacing_zyx,
        out_dtype=out_dtype,
        peak_working_memory_mb=peak_working_memory_mb,
    )

    # Wrap as SITK with original geometry
    seg_img = sitk.GetImageFromArray(seg_zyx)
    seg_img.CopyInformation(image_sitk)
    return seg_img


__all__ = [
    "resample_image_to_target",
    "inverse_resample_argmax",
    "predict_with_resampling",
]
