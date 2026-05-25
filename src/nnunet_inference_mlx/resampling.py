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
    """Trilinear interpolation of one channel at the requested coordinates."""
    Z, Y, X = src.shape
    idx = _precompute_trilinear_indices(z_coords, y_coords, x_coords, Z, Y, X)
    return _trilinear_from_indices(src, *idx)


def _trilinear_from_indices_K(
    src: mx.array,                  # (K, Z, Y, X)
    zi0, zi1, yi0, yi1, xi0, xi1,
    zf, yf, xf,
) -> mx.array:                      # (K, S, Y_a, X_a)
    """K-channel trilinear blend using pre-computed indices.

    Same math as :func:`_trilinear_from_indices` but broadcasts the K
    axis from ``src`` in a single gather per corner. Used by the slab
    loop so all K channels' samples land at exactly the same source
    positions — eliminating the slab-boundary discontinuities that
    arise with per-slab ``mx.nn.Upsample(scale_factor=...)``: that path
    distributes output uniformly across ``[0, slab_source_size - 1]``,
    which doesn't match the global coordinate system across slabs.
    """
    # Broadcast index tensors from (S,1,1) / (1,Y_a,1) / (1,1,X_a) shapes
    # are gathered with a leading K from src via standard fancy indexing.
    c000 = src[:, zi0, yi0, xi0]
    c001 = src[:, zi0, yi0, xi1]
    c010 = src[:, zi0, yi1, xi0]
    c011 = src[:, zi0, yi1, xi1]
    c100 = src[:, zi1, yi0, xi0]
    c101 = src[:, zi1, yi0, xi1]
    c110 = src[:, zi1, yi1, xi0]
    c111 = src[:, zi1, yi1, xi1]

    one_minus_xf = 1.0 - xf
    one_minus_yf = 1.0 - yf
    one_minus_zf = 1.0 - zf

    c00 = c000 * one_minus_xf + c001 * xf
    c01 = c010 * one_minus_xf + c011 * xf
    c10 = c100 * one_minus_xf + c101 * xf
    c11 = c110 * one_minus_xf + c111 * xf
    c0 = c00 * one_minus_yf + c01 * yf
    c1 = c10 * one_minus_yf + c11 * yf
    return c0 * one_minus_zf + c1 * zf


def _kchannel_trilinear_full(
    src: mx.array,            # (K, Z, Y, X) at src_spacing
    out_shape: tuple[int, int, int],
    src_spacing_zyx: tuple[float, float, float],
    out_spacing_zyx: tuple[float, float, float],
) -> mx.array:
    """K-channel trilinear at a coarser output grid, full materialize.

    Used for the intermediate steps in :func:`_cascade_then_slab` — each
    intermediate is small enough (~K × 1/8 voxels per step) that
    materializing the K-channel coarser-spacing array is cheap. No
    argmax: the K dimension carries through as the continuous logit
    field for the next cascade step.
    """
    K, Z, Y, X = src.shape
    Z_o, Y_o, X_o = out_shape
    z_ratio = out_spacing_zyx[0] / src_spacing_zyx[0]
    y_ratio = out_spacing_zyx[1] / src_spacing_zyx[1]
    x_ratio = out_spacing_zyx[2] / src_spacing_zyx[2]
    z_coords = mx.arange(Z_o, dtype=mx.float32) * z_ratio
    y_coords = mx.arange(Y_o, dtype=mx.float32) * y_ratio
    x_coords = mx.arange(X_o, dtype=mx.float32) * x_ratio
    idx = _precompute_trilinear_indices(z_coords, y_coords, x_coords, Z, Y, X)
    out = _trilinear_from_indices_K(src, *idx)
    mx.eval(out)
    return out


def _cascade_then_slab(
    logits_target: mx.array,
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    peak_working_memory_mb: int,
    out_dtype: np.dtype,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Cascade 2× K-channel downsamples until source-ratio ≤ 2×; then slab+argmax.

    For aggressive downsampling (source-ratio > 2× in any axis), the
    standard single-step trilinear is undersampled — kernel support
    (2³ source voxels = 2 × src_spacing per axis) is smaller than the
    output voxel's physical footprint. Result: thin structures alias out.

    Cascade fix: build a sequence of intermediate K-channel arrays at
    progressively coarser spacings, each step a 2× downsample (kernel
    exactly matches footprint). Final step runs the standard slab argmax
    from the last intermediate to the requested output grid, where the
    source-ratio is ≤ 2× by construction.

    The K-channel logits carry the continuous decision surface through
    the cascade; argmax happens only at the very end.

    Memory at each intermediate is ~1/8 the previous (one 2× downsample
    per axis), so cascading is cheap — the cascade total stays bounded
    even for very aggressive downsample ratios.
    """
    cur = logits_target
    cur_spacing = tuple(target_spacing_zyx)
    cur_shape = cur.shape[1:]
    step = 0
    if verbose:
        print(f"  [cascade] start: K={cur.shape[0]} shape={cur_shape} "
              f"spacing={tuple(round(s,2) for s in cur_spacing)}")

    while any(o > 2.001 * c for c, o in zip(cur_spacing, acq_spacing_zyx)):
        next_spacing = tuple(
            min(2.0 * c, o) for c, o in zip(cur_spacing, acq_spacing_zyx)
        )
        next_shape = tuple(
            max(1, int(round(s * c / n)))
            for s, c, n in zip(cur_shape, cur_spacing, next_spacing)
        )
        step += 1
        if verbose:
            print(f"  [cascade] step {step}: shape {next_shape} "
                  f"spacing {tuple(round(s,2) for s in next_spacing)}")
        cur = _kchannel_trilinear_full(cur, next_shape, cur_spacing, next_spacing)
        cur_spacing = next_spacing
        cur_shape = next_shape

    if verbose:
        max_ratio = max(o / c for c, o in zip(cur_spacing, acq_spacing_zyx))
        print(f"  [cascade] final slab+argmax from {cur_shape} "
              f"@{tuple(round(s,2) for s in cur_spacing)} → "
              f"{out_shape_zyx} @{tuple(round(s,2) for s in acq_spacing_zyx)} "
              f"(source-ratio {max_ratio:.2f}×)")

    return _slab_resample_argmax(
        cur, out_shape_zyx, cur_spacing, acq_spacing_zyx,
        peak_working_memory_mb, out_dtype,
    )


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


def _slab_resample_paint(
    logits_target: mx.array,            # (K, Z_t, Y_t, X_t)
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    peak_working_memory_mb: int,
    regions_class_order: tuple[int, ...],
    threshold: float,
    out_dtype: np.dtype,
) -> np.ndarray:
    """Slab-streamed inverse resample for region-based (BraTS-style) models.

    Identical slab-streaming shape and trilinear-interpolation step as
    :func:`_slab_resample_argmax`; differs only in the per-slab finishing:
    instead of argmax across K, do per-region threshold and paint in
    ``regions_class_order`` order.

    Painting in order means later (higher-priority) regions overwrite
    earlier ones at overlapping voxels — the standard nnU-Net region
    semantics. This matches what :func:`convert_logits_to_segmentation`
    does at target spacing, just applied per-slab at acquisition spacing.

    The slab's K channels are read into MLX once via trilinear gather,
    then a numpy host loop paints each region (cheap: only K passes over
    the slab, each a thresholded mask write).
    """
    K, Z_t, Y_t, X_t = logits_target.shape
    Z_a, Y_a, X_a = out_shape_zyx

    if K != len(regions_class_order):
        raise ValueError(
            f"Region prediction has {K} channels but regions_class_order "
            f"has {len(regions_class_order)} entries."
        )

    bytes_per_slab_voxel = K * 4
    max_slab_voxels = int(peak_working_memory_mb) * 1024 * 1024 // bytes_per_slab_voxel
    plane_voxels = max(1, Y_a * X_a)
    slab_z = max(1, max_slab_voxels // plane_voxels)

    s2t_z = acq_spacing_zyx[0] / target_spacing_zyx[0]
    s2t_y = acq_spacing_zyx[1] / target_spacing_zyx[1]
    s2t_x = acq_spacing_zyx[2] / target_spacing_zyx[2]

    y_coords = mx.arange(Y_a, dtype=mx.float32) * s2t_y
    x_coords = mx.arange(X_a, dtype=mx.float32) * s2t_x

    out = np.zeros(out_shape_zyx, dtype=out_dtype)

    for z0 in range(0, Z_a, slab_z):
        z1 = min(z0 + slab_z, Z_a)

        z_global = mx.arange(z0, z1, dtype=mx.float32) * s2t_z
        z_lo_f = z0 * s2t_z
        z_hi_f = (z1 - 1) * s2t_z
        zt_lo = max(0, min(Z_t - 1, int(z_lo_f) - 1))
        zt_hi = max(zt_lo + 1, min(Z_t, int(z_hi_f) + 2))
        slab_src = logits_target[:, zt_lo:zt_hi]
        slab_t_z = slab_src.shape[1]
        z_local = z_global - zt_lo

        idx = _precompute_trilinear_indices(
            z_local, y_coords, x_coords, slab_t_z, Y_t, X_t,
        )
        slab_K = _trilinear_from_indices_K(slab_src, *idx)   # (K, S, Y_a, X_a)
        mx.eval(slab_K)
        slab_K_np = np.asarray(slab_K)

        # Paint in regions_class_order: later regions overwrite earlier ones.
        slab_seg = np.zeros((z1 - z0, Y_a, X_a), dtype=out_dtype)
        for region_idx, label_value in enumerate(regions_class_order):
            slab_seg[slab_K_np[region_idx] > threshold] = label_value
        out[z0:z1] = slab_seg

    return out


def _slab_resample_argmax(
    logits_target: mx.array,            # (K, Z_t, Y_t, X_t)
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    peak_working_memory_mb: int,
    out_dtype: np.dtype,
) -> np.ndarray:
    """Slab-stream the inverse resample with explicit-coordinate trilinear.

    For each output Z slab:
      - compute the *global* (z, y, x) coordinates in source space — these
        are the exact positions the K-channel logits should be sampled at,
      - slice the target-spacing source to just the Z range needed,
      - run the 8-corner trilinear gather across all K channels in one
        sweep (broadcasting K from src), producing a (K, S, Y_a, X_a) slab,
      - argmax across K → int slab,
      - copy uint8 to host output buffer.

    Coordinates are computed globally, so adjacent slabs sample at
    consistent positions — no boundary discontinuities. (This is the bug
    that the earlier ``mx.nn.Upsample(scale_factor=...)``-per-slab
    implementation had: each slab's Upsample distributes output uniformly
    across its own source range, which doesn't honor the offset between
    the slab's source-start and the desired-output-start position.)

    Working memory per slab is dominated by the K-channel acquisition-
    spacing slab: ``K * S * Y_a * X_a * 4`` bytes (fp32). Slab depth is
    chosen from peak_working_memory_mb so that total fits.
    """
    K, Z_t, Y_t, X_t = logits_target.shape
    Z_a, Y_a, X_a = out_shape_zyx

    # Slab voxel budget: K-channel fp32 slab + 8 corner gathers held
    # transiently in the trilinear blend. Conservative: count 12 bytes
    # per voxel-K (the K-channel slab + ~2 corners in flight + margin).
    bytes_per_slab_voxel = K * 4
    max_slab_voxels = int(peak_working_memory_mb) * 1024 * 1024 // bytes_per_slab_voxel
    plane_voxels = max(1, Y_a * X_a)
    slab_z = max(1, max_slab_voxels // plane_voxels)

    # Global coordinate-space scale factors: acquisition voxel → target voxel.
    s2t_z = acq_spacing_zyx[0] / target_spacing_zyx[0]
    s2t_y = acq_spacing_zyx[1] / target_spacing_zyx[1]
    s2t_x = acq_spacing_zyx[2] / target_spacing_zyx[2]

    # Y / X coords are the same for every slab — compute once.
    y_coords = mx.arange(Y_a, dtype=mx.float32) * s2t_y
    x_coords = mx.arange(X_a, dtype=mx.float32) * s2t_x

    out = np.empty(out_shape_zyx, dtype=out_dtype)

    for z0 in range(0, Z_a, slab_z):
        z1 = min(z0 + slab_z, Z_a)
        S = z1 - z0

        # Output Z positions for this slab — in global target voxel coords.
        z_global = mx.arange(z0, z1, dtype=mx.float32) * s2t_z

        # Target-spacing Z slice needed for this slab (with ±1 voxel pad
        # so the trilinear interpolation can clamp at boundaries). Clamp
        # to [0, Z_t]; guarantee zt_hi > zt_lo.
        z_lo_f = z0 * s2t_z
        z_hi_f = (z1 - 1) * s2t_z
        zt_lo = max(0, min(Z_t - 1, int(z_lo_f) - 1))
        zt_hi = max(zt_lo + 1, min(Z_t, int(z_hi_f) + 2))
        slab_src = logits_target[:, zt_lo:zt_hi]    # (K, slab_t_z, Y_t, X_t)
        slab_t_z = slab_src.shape[1]
        z_local = z_global - zt_lo                  # source-local Z coords

        # Pre-compute indices once for the (z_local, y_coords, x_coords)
        # grid — these are shared across all K channels.
        idx = _precompute_trilinear_indices(
            z_local, y_coords, x_coords, slab_t_z, Y_t, X_t,
        )

        # All K channels at once via broadcasted gather.
        slab_K = _trilinear_from_indices_K(slab_src, *idx)   # (K, S, Y_a, X_a)

        # Argmax across channels.
        seg_slab = mx.argmax(slab_K, axis=0)        # (S, Y_a, X_a) int32
        mx.eval(seg_slab)
        out[z0:z1] = np.asarray(seg_slab).astype(out_dtype, copy=False)

    return out


def inverse_resample_argmax(
    logits_target: "mx.array | np.ndarray",
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    *,
    out_dtype: np.dtype | str = np.uint8,
    peak_working_memory_mb: int | None = None,
    cascade_downsample: bool | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Resample target-spacing logits to acquisition spacing and argmax.

    Strategy: a single Z-slab loop with explicit-coordinate trilinear
    over all K channels. Slab depth auto-sized from
    ``peak_working_memory_mb`` so the K-channel acquisition-spacing slab
    fits the budget. When the whole output fits in one slab, the loop
    runs once — equivalent to a materialize pass.

    For aggressive downsampling (source-ratio > 2× in any axis), the
    single-step trilinear undersamples — its 2³ kernel covers less than
    the output voxel's physical footprint. ``cascade_downsample``
    enables a multi-step path that builds intermediate K-channel arrays
    at progressively coarser spacings (each step a 2× downsample, the
    sweet spot for trilinear), preserving thin structures that single-
    step would alias out.

    ``cascade_downsample``:
      * ``False`` (default) — single-step trilinear regardless of source ratio.
      * ``True`` — cascade 2× downsample steps until source-ratio ≤ 2×,
        then a final argmax pass. Trades more smoothing for less
        boundary aliasing. *Not* a strict correctness win on
        aggressive downsamples — the cascade dilutes thin-structure
        logit peaks more than single-step does, so very thin structures
        (small vessels, sub-voxel anatomy) can vanish at the cascade
        stage even though single-step preserved them. Tighter
        large-structure boundaries can also overshoot the reference.
      * ``None`` — alias for ``False``. Reserved for future auto-tuning.

    Aggressive downsampling (source-ratio > 2× in any axis) is
    inherently lossy for thin structures regardless of method. Single-
    step gives marginally better preservation of small structures;
    cascade gives marginally smoother large-structure boundaries.

    Note that at large source-ratios many small structures are simply
    sub-voxel at the output spacing — e.g. an 8× decimate of ~1.5 mm CT
    yields ~12 mm voxels, which is larger than common-carotid diameter
    (6–8 mm). No resampling method recovers structures below Nyquist;
    pick the output spacing first, then expect anatomy thinner than
    a voxel to disappear regardless of cascade setting.

    Throughput is dominated by MLX's native trilinear+gather kernels
    (~11 ns / voxel-K on M2 base). ``peak_working_memory_mb`` is a
    memory-bound knob, not a perf knob.

    ``peak_working_memory_mb=None`` (default) auto-detects from system
    RAM: 200 MB on < 32 GB Macs, 2000 MB on ≥ 32 GB Macs.
    """
    if peak_working_memory_mb is None:
        peak_working_memory_mb = _auto_memory_budget_mb()
    out_dtype = np.dtype(out_dtype)

    # Accept either mx.array (preferred — already in unified memory) or
    # any numpy-convertible. The bare `mx.array(...)` wrap at every call
    # site is API friction; this internalizes the same one-time copy.
    if not isinstance(logits_target, mx.array):
        logits_target = mx.array(logits_target)

    max_ratio = max(
        o / t for t, o in zip(target_spacing_zyx, acq_spacing_zyx)
    )
    if cascade_downsample is None:
        cascade_downsample = False
    if verbose:
        print(f"  [inverse_resample_argmax] source-ratio max={max_ratio:.3f}, "
              f"cascade={'on' if cascade_downsample else 'off'}")

    if cascade_downsample and max_ratio > 2.001:
        return _cascade_then_slab(
            logits_target, out_shape_zyx,
            target_spacing_zyx, acq_spacing_zyx,
            peak_working_memory_mb, out_dtype,
            verbose=verbose,
        )

    return _slab_resample_argmax(
        logits_target, out_shape_zyx,
        target_spacing_zyx, acq_spacing_zyx,
        peak_working_memory_mb, out_dtype,
    )


def inverse_resample_paint(
    logits_target: "mx.array | np.ndarray",
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    regions_class_order: tuple[int, ...],
    *,
    threshold: float = 0.0,
    out_dtype: np.dtype | str | None = None,
    peak_working_memory_mb: int | None = None,
) -> np.ndarray:
    """Resample region-based logits to acquisition spacing, threshold + paint.

    The region-based (BraTS-style) sibling of :func:`inverse_resample_argmax`.
    For models whose K output channels are *independent sigmoid heads*
    (not a softmax distribution), the correct conversion to integer labels
    is per-region thresholding followed by paint-priority overwrite —
    *not* argmax (which would silently pick "the region with the highest
    sigmoid" instead of "all regions above threshold, painted by priority").

    Strategy: identical slab-streaming + K-channel trilinear-gather pass
    as :func:`inverse_resample_argmax`. The per-slab finishing step
    paints each region's label value (from ``regions_class_order``) at
    voxels where that region's interpolated logit exceeds ``threshold``.
    Higher-index regions overwrite lower-index ones at overlaps — the
    standard nnU-Net region semantics.

    Parameters
    ----------
    logits_target :
        Per-region logits or post-sigmoid probabilities at the model's
        target spacing, shape ``(K, Z_t, Y_t, X_t)``. Accepts either
        ``mx.array`` (preferred — already in unified memory) or a numpy
        array (converted internally).
    out_shape_zyx :
        Shape of the output segmentation at acquisition spacing.
    target_spacing_zyx, acq_spacing_zyx :
        Spacings, in millimeters, of the input logit volume and the
        desired output respectively. Axis order ``(Z, Y, X)``.
    regions_class_order :
        Tuple of K label values, one per region channel, in paint-priority
        order. Region 0 paints first; region K-1 paints last and wins at
        overlaps. From ``engine.regions_class_order`` for region-based bundles.
    threshold :
        Cut for region membership. ``0.0`` (default) matches raw-logit
        output (``sigmoid > 0.5`` ↔ ``logit > 0``). Pass ``0.5`` if the
        input has already been sigmoid'd (e.g., a multi-fold ensemble
        which averages post-sigmoid probabilities).
    out_dtype :
        Output integer dtype. ``None`` (default) picks the smallest
        unsigned dtype that fits ``max(regions_class_order)`` — typically
        ``uint8``. Pass an explicit dtype to override.
    peak_working_memory_mb :
        Same slab budget as :func:`inverse_resample_argmax`; auto-detects
        from system RAM when ``None``.

    Returns
    -------
    np.ndarray
        Integer label volume at acquisition spacing, shape ``out_shape_zyx``.
    """
    if peak_working_memory_mb is None:
        peak_working_memory_mb = _auto_memory_budget_mb()

    if not isinstance(logits_target, mx.array):
        logits_target = mx.array(logits_target)

    if out_dtype is None:
        max_label = max(regions_class_order) if regions_class_order else 0
        if max_label < 256:
            out_dtype = np.uint8
        elif max_label < 65536:
            out_dtype = np.uint16
        else:
            out_dtype = np.uint32
    out_dtype = np.dtype(out_dtype)

    return _slab_resample_paint(
        logits_target, out_shape_zyx,
        target_spacing_zyx, acq_spacing_zyx,
        peak_working_memory_mb,
        tuple(int(v) for v in regions_class_order),
        float(threshold),
        out_dtype,
    )


# ---------------------------------------------------------------------------
# High-level: SITK in, SITK out
# ---------------------------------------------------------------------------

def _orientation_code(sitk_module, image: "sitk.Image") -> str:
    """Return the 3-letter DICOM-style orientation code for a SITK image.

    e.g. ``"LPS"`` (canonical: Left-Posterior-Superior voxel-axis directions),
    ``"RAS"``, or for oblique scans something like ``"SAR"``.
    """
    return sitk_module.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


def predict_with_resampling(
    engine: "InferenceEngine",
    image_sitk: "sitk.Image",
    *,
    interpolation: str = "linear",
    peak_working_memory_mb: int | None = None,
    remove_small_components_mm3: float = 0.0,
    reorient: str | None = "LPS",
) -> "sitk.Image":
    """Forward-resample input → run inference → inverse-resample logits.

    The full path-B pipeline: caller hands over a SITK image at any
    acquisition spacing, gets back a SITK image at the same acquisition
    spacing with integer labels. Geometry is preserved via
    ``CopyInformation``, including the original orientation.

    Forward resample runs on CPU via SITK (b-spline if requested);
    inference runs on Metal; inverse resample runs on Metal with
    slab+channel streaming bounded by ``peak_working_memory_mb`` (auto-
    detected from system RAM if None).

    Orientation handling
    --------------------
    nnU-Net models are trained with a consistent voxel-axis ↔ anatomical-
    axis mapping (cranio-caudal as the slowest axis, etc.). For
    axis-aligned scans this is implicit, but oblique acquisitions or
    re-formatted volumes have a non-identity direction matrix — feeding
    them to the model raw causes the sliding window to scan in the
    wrong anatomical direction, severely degrading segmentation quality
    (vertebrae fragment, cardiac chambers slice in the wrong plane,
    etc.).

    ``reorient`` (default ``"LPS"``) reorients the input to canonical
    LPS orientation before forward resample / inference, then maps the
    output back to the input's original orientation. Set to ``None`` to
    skip the reorient (useful only when you know the input is already
    in canonical orientation and want to save the few seconds of
    reorient cost on huge volumes).

    Parameters
    ----------
    remove_small_components_mm3 :
        If > 0, drop connected components smaller than this physical
        volume (mm³) from the output, using multi-label-aware CC. ``0``
        (default) disables the cleanup. ``200.0`` matches
        TotalSegmentator's ``--remove_small_blobs`` flag. Requires the
        ``[postprocessing]`` optional extra (cc3d).
    reorient :
        Canonical orientation target. ``"LPS"`` (default) is what most
        nnU-Net models were trained with. Pass ``None`` to skip the
        reorient + back-reorient round-trip (only safe when the input is
        already in canonical orientation; otherwise expect badly-axis-
        aligned segmentations).
    """
    sitk = _require_sitk()

    # --- Reorient input to canonical before inference -----------------
    # Save the original orientation so we can map the segmentation back
    # to the caller's coordinate system at the end.
    if reorient is not None:
        original_orient = _orientation_code(sitk, image_sitk)
        if original_orient != reorient:
            image_for_infer = sitk.DICOMOrient(image_sitk, reorient)
        else:
            image_for_infer = image_sitk
    else:
        original_orient = None
        image_for_infer = image_sitk

    target_spacing_zyx = engine.target_spacing
    in_spacing_xyz = image_for_infer.GetSpacing()
    acq_spacing_zyx = tuple(reversed(in_spacing_xyz))

    # Forward resample (CPU / SITK), at canonical orientation
    resampled = resample_image_to_target(
        image_for_infer, target_spacing_zyx, interpolation=interpolation,
    )

    # Inference (Metal). predict_logits returns mx.array at target spacing
    # for single-fold, averaged softmax probs for multi-fold standard
    # ensembles, averaged sigmoid probs for multi-fold region ensembles —
    # all (K, Z_t, Y_t, X_t).
    vol_target = sitk.GetArrayFromImage(resampled).astype(np.float32, copy=False)
    pred_mx = engine.predict_logits(vol_target)

    # Output shape in (Z, Y, X) — SITK GetSize is (X, Y, Z), so reverse.
    in_size_xyz = image_for_infer.GetSize()
    out_shape_zyx = (in_size_xyz[2], in_size_xyz[1], in_size_xyz[0])

    # Pick output dtype from the bundle's label scheme.
    out_dtype = engine.label_dtype

    # Scheme-aware inverse resample: argmax for standard, threshold+paint
    # for region-based. Prior to 0.8.0 this always called argmax, which
    # silently mis-converts region-based (BraTS-style) outputs.
    if engine.has_regions:
        # Multi-fold ensembles average post-sigmoid probabilities (threshold
        # at 0.5); single-fold returns raw logits (threshold at 0 ↔ sigmoid>0.5).
        threshold = 0.5 if len(engine.bundle.fold_weights) > 1 else 0.0
        seg_zyx = inverse_resample_paint(
            pred_mx, out_shape_zyx,
            target_spacing_zyx, acq_spacing_zyx,
            regions_class_order=engine.regions_class_order,
            threshold=threshold,
            out_dtype=out_dtype,
            peak_working_memory_mb=peak_working_memory_mb,
        )
    else:
        seg_zyx = inverse_resample_argmax(
            pred_mx, out_shape_zyx,
            target_spacing_zyx, acq_spacing_zyx,
            out_dtype=out_dtype,
            peak_working_memory_mb=peak_working_memory_mb,
        )

    if remove_small_components_mm3 > 0:
        from .postprocessing import remove_small_components
        seg_zyx = remove_small_components(
            seg_zyx, acq_spacing_zyx,
            min_volume_mm3=remove_small_components_mm3,
            in_place=True,
        )

    # Wrap as SITK in the canonical-orientation grid
    seg_img = sitk.GetImageFromArray(seg_zyx)
    seg_img.CopyInformation(image_for_infer)

    # --- Reorient seg back to caller's original orientation -----------
    if original_orient is not None and original_orient != reorient:
        seg_img = sitk.DICOMOrient(seg_img, original_orient)
        # After DICOMOrient, geometry attributes reflect the new (original)
        # orientation. Sanity: size/spacing/origin/direction should now
        # match the input.

    return seg_img


__all__ = [
    "resample_image_to_target",
    "inverse_resample_argmax",
    "inverse_resample_paint",
    "predict_with_resampling",
]
