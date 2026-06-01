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

Backed by SimpleITK, a core dependency (segmenting means reading/resampling
images). Consumers using their own resampling library (scipy, dask) can
ignore this module.
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
            "SimpleITK is required for resampling helpers (it is a core "
            "dependency; `uv run` installs it). Install with `pip install SimpleITK`."
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
# Forward resample — MLX per-axis (anti-aliased cubic down / linear up)
# ---------------------------------------------------------------------------
#
# Axis-aligned spacing change (the forward image resample, after reorient to a
# canonical orientation) done on Metal. Per-axis policy keyed on the resample
# factor ``f = out_spacing/in_spacing``:
#   * f > 1 (downsampling): factor-scaled Catmull-Rom cubic — the kernel
#     support stretches with f so it averages the whole output-voxel footprint,
#     i.e. genuine anti-aliasing. Plain linear at large factors undersamples
#     (2 taps vs an f-voxel footprint) and aliases thin/high-contrast structure.
#   * f <= 1 (upsampling / near-identity): linear. Anti-aliasing doesn't apply,
#     and cubic's negative lobes ring/overshoot across high-contrast edges —
#     e.g. inventing ±140 HU haloes between the sparse slices of thick-slice CT
#     when the through-plane axis is upsampled to the model grid. Linear is
#     monotone → never invents values outside the data. This is nnU-Net's
#     separate-z idea, generalized to a per-axis decision.
# Corner-aligned (c = j * f), matching SITK's resample and the inverse restore,
# so forward/inverse round-trips. One Metal kernel pass per axis (separable):
# ~4f taps/axis, vs (4f)^3 for a fused 3D kernel — sub-second on 418 M voxels.

_RESAMPLE_1D_SRC = r"""
  uint elem = thread_position_in_grid.x;
  int N_in=ip[0], N_out=ip[1], M=ip[2], n_taps=ip[3], mode=ip[4];
  if ((int)elem >= N_out*M) return;
  int j=(int)elem/M, m=(int)elem%M;
  float f=fp[0], scale=fp[1], support=fp[2];
  float c=(float)j*f;
  int base=(int)metal::floor(c-support+1.0f);
  float acc=0.0f, wsum=0.0f;
  for (int t=0;t<n_taps;t++){
    int k=base+t; float x=((float)k-c)/scale, ax=metal::fabs(x), w;
    if (mode==1){ // Catmull-Rom cubic (a=-0.5)
      if (ax<1.0f) w=1.5f*ax*ax*ax-2.5f*ax*ax+1.0f;
      else if (ax<2.0f) w=-0.5f*ax*ax*ax+2.5f*ax*ax-4.0f*ax+2.0f;
      else w=0.0f;
    } else { // triangle / linear
      w = ax<1.0f ? (1.0f-ax) : 0.0f;
    }
    int kc = k<0 ? 0 : (k>=N_in ? N_in-1 : k);
    acc += w*arr[(long)kc*M+m]; wsum += w;
  }
  out[elem]=acc/(wsum+1e-8f);
"""

_RESAMPLE_1D_KERNEL = None


def _get_resample_1d_kernel():
    global _RESAMPLE_1D_KERNEL
    if _RESAMPLE_1D_KERNEL is None:
        _RESAMPLE_1D_KERNEL = mx.fast.metal_kernel(
            name="resample1d_axis",
            input_names=["arr", "ip", "fp"],
            output_names=["out"],
            source=_RESAMPLE_1D_SRC,
        )
    return _RESAMPLE_1D_KERNEL


def _resample_axis_mlx(a: mx.array, axis: int, n_out: int, f: float,
                       aa_threshold: float) -> mx.array:
    cubic = f > aa_threshold
    scale = f if cubic else 1.0
    support = (2.0 if cubic else 1.0) * scale
    n_taps = int(np.floor(2.0 * support)) + 2
    a = mx.moveaxis(a, axis, 0)
    shp = a.shape
    n_in = shp[0]
    m = 1
    for s in shp[1:]:
        m *= int(s)
    a2 = mx.reshape(a, (n_in, m))
    ip = mx.array([n_in, n_out, m, n_taps, 1 if cubic else 0], dtype=mx.int32)
    fp = mx.array([float(f), float(scale), float(support)], dtype=mx.float32)
    (o,) = _get_resample_1d_kernel()(
        inputs=[a2, ip, fp],
        grid=(n_out * m, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_out, m)],
        output_dtypes=[mx.float32],
    )
    return mx.moveaxis(mx.reshape(o, (n_out, *shp[1:])), 0, axis)


def resample_volume_mlx(
    volume_zyx: "mx.array | np.ndarray",
    out_shape_zyx: tuple[int, int, int],
    src_spacing_zyx: tuple[float, float, float],
    tgt_spacing_zyx: tuple[float, float, float],
    *,
    aa_threshold: float = 1.05,
) -> mx.array:
    """Per-axis Metal resample: anti-aliased cubic on downsampling axes, linear
    on upsampling/near-identity axes (see module notes above).

    ``aa_threshold`` is the factor above which an axis switches to anti-aliased
    cubic (default 1.05 — anything meaningfully downsampling). Returns a float32
    ``mx.array`` of shape ``out_shape_zyx``.
    """
    if not isinstance(volume_zyx, mx.array):
        volume_zyx = mx.array(np.asarray(volume_zyx, dtype=np.float32))
    a = volume_zyx.astype(mx.float32)
    lo, hi = mx.min(a), mx.max(a)
    for ax in range(3):
        f = float(tgt_spacing_zyx[ax]) / float(src_spacing_zyx[ax])
        a = _resample_axis_mlx(a, ax, int(out_shape_zyx[ax]), f, aa_threshold)
    # Clamped cubic: Catmull-Rom's negative lobes can ring slightly past the
    # data range at sharp edges (e.g. ~26 HU below the air floor on CT). Clip
    # to the input's value range — removes out-of-range overshoot/undershoot
    # while keeping cubic's interior sharpness. (Linear axes never ring; this
    # only ever clips the cubic ones.)
    a = mx.clip(a, lo, hi)
    mx.eval(a)
    return a


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

    The 8 corner fetches dominate this (memory-gather-bound) op. Gathering
    via a flattened 1-D ``mx.take`` along a single (Z·Y·X) axis is measurably
    faster than 3-axis advanced indexing (``src[:, zi, yi, xi]``) — ~28% on a
    512×512 output, 117-channel job — for identical results.
    """
    K, Z, Y, X = src.shape
    S, Y_a, X_a = zf.shape[0], yf.shape[1], xf.shape[2]
    src_flat = src.reshape(K, Z * Y * X)

    def gather(zi, yi, xi):  # broadcast (S,1,1)+(1,Y_a,1)+(1,1,X_a) → flat (S·Y_a·X_a,)
        flat = (zi * (Y * X) + yi * X + xi).reshape(-1)
        return mx.take(src_flat, flat, axis=1).reshape(K, S, Y_a, X_a)

    c000 = gather(zi0, yi0, xi0)
    c001 = gather(zi0, yi0, xi1)
    c010 = gather(zi0, yi1, xi0)
    c011 = gather(zi0, yi1, xi1)
    c100 = gather(zi1, yi0, xi0)
    c101 = gather(zi1, yi0, xi1)
    c110 = gather(zi1, yi1, xi0)
    c111 = gather(zi1, yi1, xi1)

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


# ---------------------------------------------------------------------------
# Fused Metal kernel — trilinear + argmax/paint inline, no K-channel materialize
# ---------------------------------------------------------------------------
#
# The slab paths above gather all 8 trilinear corners into full
# (K, S, Y_a, X_a) arrays, blend them, then run a separate argmax/paint pass.
# That materializes ~8× the K-channel output in transient memory and is
# memory-gather-bound (8 fancy-index fetches + a final argmax read).
#
# These kernels do the whole inverse resample with one thread per *output*
# voxel: each thread trilinear-interpolates all K channels inline (reading
# only the 8 source corners per channel) and reduces to a single integer
# label on the fly. The only large buffer is the input logits (fixed); the
# output is the small int label volume — so there is nothing to slab. ~100×
# faster than the slab path on a 512² / 117-channel job, bit-identical on
# random and real data.

# Separable trilinear in the same evaluation order as the MLX slab path, so
# float rounding (hence argmax ties) match. Helper emitted into both kernels.
_TRILINEAR_HEADER = r"""
inline void corner_setup(
    int z, int y, int x,
    float s2t_z, float s2t_y, float s2t_x,
    int Z_t, int Y_t, int X_t,
    thread int* b, thread float* zf, thread float* yf, thread float* xf)
{
    float zc = metal::clamp((float)z * s2t_z, 0.0f, (float)(Z_t - 1));
    float yc = metal::clamp((float)y * s2t_y, 0.0f, (float)(Y_t - 1));
    float xc = metal::clamp((float)x * s2t_x, 0.0f, (float)(X_t - 1));
    int z0 = (int)metal::floor(zc); int z1 = metal::min(z0 + 1, Z_t - 1);
    int y0 = (int)metal::floor(yc); int y1 = metal::min(y0 + 1, Y_t - 1);
    int x0 = (int)metal::floor(xc); int x1 = metal::min(x0 + 1, X_t - 1);
    *zf = zc - (float)z0; *yf = yc - (float)y0; *xf = xc - (float)x0;
    int plane = Y_t * X_t;
    b[0] = z0*plane + y0*X_t + x0;  // 000
    b[1] = z0*plane + y0*X_t + x1;  // 001
    b[2] = z0*plane + y1*X_t + x0;  // 010
    b[3] = z0*plane + y1*X_t + x1;  // 011
    b[4] = z1*plane + y0*X_t + x0;  // 100
    b[5] = z1*plane + y0*X_t + x1;  // 101
    b[6] = z1*plane + y1*X_t + x0;  // 110
    b[7] = z1*plane + y1*X_t + x1;  // 111
}

inline float sample_channel(
    const device float* logits, long off, thread const int* b,
    float zf, float yf, float xf)
{
    // Separable blend, matching the host MLX path's op order.
    float c00 = logits[off+b[0]] * (1.0f-xf) + logits[off+b[1]] * xf;
    float c01 = logits[off+b[2]] * (1.0f-xf) + logits[off+b[3]] * xf;
    float c10 = logits[off+b[4]] * (1.0f-xf) + logits[off+b[5]] * xf;
    float c11 = logits[off+b[6]] * (1.0f-xf) + logits[off+b[7]] * xf;
    float c0 = c00 * (1.0f-yf) + c01 * yf;
    float c1 = c10 * (1.0f-yf) + c11 * yf;
    return c0 * (1.0f-zf) + c1 * zf;
}
"""

_FUSED_ARGMAX_SRC = r"""
    uint elem = thread_position_in_grid.x;
    int n_out = iparams[7];
    if ((int)elem >= n_out) return;

    int K = iparams[0], Z_t = iparams[1], Y_t = iparams[2], X_t = iparams[3];
    int Y_a = iparams[5], X_a = iparams[6];
    float s2t_z = fparams[0], s2t_y = fparams[1], s2t_x = fparams[2];

    int x = (int)elem % X_a;
    int y = ((int)elem / X_a) % Y_a;
    int z = (int)elem / (X_a * Y_a);

    int b[8]; float zf, yf, xf;
    corner_setup(z, y, x, s2t_z, s2t_y, s2t_x, Z_t, Y_t, X_t, b, &zf, &yf, &xf);

    long chan_stride = (long)Z_t * (long)Y_t * (long)X_t;
    float best = -INFINITY; int best_k = 0;
    for (int k = 0; k < K; k++) {
        float v = sample_channel(logits, (long)k * chan_stride, b, zf, yf, xf);
        if (v > best) { best = v; best_k = k; }
    }
    out[elem] = (uint32_t)best_k;
"""

_FUSED_PAINT_SRC = r"""
    uint elem = thread_position_in_grid.x;
    int n_out = iparams[7];
    if ((int)elem >= n_out) return;

    int K = iparams[0], Z_t = iparams[1], Y_t = iparams[2], X_t = iparams[3];
    int Y_a = iparams[5], X_a = iparams[6];
    float s2t_z = fparams[0], s2t_y = fparams[1], s2t_x = fparams[2];
    float threshold = fparams[3];

    int x = (int)elem % X_a;
    int y = ((int)elem / X_a) % Y_a;
    int z = (int)elem / (X_a * Y_a);

    int b[8]; float zf, yf, xf;
    corner_setup(z, y, x, s2t_z, s2t_y, s2t_x, Z_t, Y_t, X_t, b, &zf, &yf, &xf);

    long chan_stride = (long)Z_t * (long)Y_t * (long)X_t;
    // Paint in channel order: later regions overwrite earlier ones at overlaps.
    uint32_t label = 0;
    for (int k = 0; k < K; k++) {
        float v = sample_channel(logits, (long)k * chan_stride, b, zf, yf, xf);
        if (v > threshold) label = (uint32_t)region_labels[k];
    }
    out[elem] = label;
"""

_FUSED_ARGMAX_KERNEL = None
_FUSED_PAINT_KERNEL = None
_FUSED_TG = 256


def _get_fused_argmax_kernel():
    global _FUSED_ARGMAX_KERNEL
    if _FUSED_ARGMAX_KERNEL is None:
        _FUSED_ARGMAX_KERNEL = mx.fast.metal_kernel(
            name="fused_resample_argmax",
            input_names=["logits", "iparams", "fparams"],
            output_names=["out"],
            header=_TRILINEAR_HEADER,
            source=_FUSED_ARGMAX_SRC,
        )
    return _FUSED_ARGMAX_KERNEL


def _get_fused_paint_kernel():
    global _FUSED_PAINT_KERNEL
    if _FUSED_PAINT_KERNEL is None:
        _FUSED_PAINT_KERNEL = mx.fast.metal_kernel(
            name="fused_resample_paint",
            input_names=["logits", "iparams", "fparams", "region_labels"],
            output_names=["out"],
            header=_TRILINEAR_HEADER,
            source=_FUSED_PAINT_SRC,
        )
    return _FUSED_PAINT_KERNEL


def _fused_resample_argmax(
    logits_target: mx.array,            # (K, Z_t, Y_t, X_t)
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    out_dtype: np.dtype,
) -> np.ndarray:
    """One-launch fused trilinear+argmax — the fast inverse-resample path."""
    K, Z_t, Y_t, X_t = logits_target.shape
    Z_a, Y_a, X_a = out_shape_zyx
    n_out = Z_a * Y_a * X_a
    iparams = mx.array([K, Z_t, Y_t, X_t, Z_a, Y_a, X_a, n_out], dtype=mx.int32)
    s2t = [acq_spacing_zyx[i] / target_spacing_zyx[i] for i in range(3)]
    fparams = mx.array(s2t, dtype=mx.float32)
    (out,) = _get_fused_argmax_kernel()(
        inputs=[logits_target, iparams, fparams],
        grid=(n_out, 1, 1),
        threadgroup=(_FUSED_TG, 1, 1),
        output_shapes=[(Z_a, Y_a, X_a)],
        output_dtypes=[mx.uint32],
    )
    mx.eval(out)
    return np.asarray(out).astype(out_dtype, copy=False)


def _fused_resample_paint(
    logits_target: mx.array,            # (K, Z_t, Y_t, X_t)
    out_shape_zyx: tuple[int, int, int],
    target_spacing_zyx: tuple[float, float, float],
    acq_spacing_zyx: tuple[float, float, float],
    regions_class_order: tuple[int, ...],
    threshold: float,
    out_dtype: np.dtype,
) -> np.ndarray:
    """One-launch fused trilinear + threshold-paint — region-model fast path."""
    K, Z_t, Y_t, X_t = logits_target.shape
    Z_a, Y_a, X_a = out_shape_zyx
    n_out = Z_a * Y_a * X_a
    iparams = mx.array([K, Z_t, Y_t, X_t, Z_a, Y_a, X_a, n_out], dtype=mx.int32)
    s2t = [acq_spacing_zyx[i] / target_spacing_zyx[i] for i in range(3)]
    fparams = mx.array([*s2t, float(threshold)], dtype=mx.float32)
    region_labels = mx.array(list(regions_class_order), dtype=mx.int32)
    (out,) = _get_fused_paint_kernel()(
        inputs=[logits_target, iparams, fparams, region_labels],
        grid=(n_out, 1, 1),
        threadgroup=(_FUSED_TG, 1, 1),
        output_shapes=[(Z_a, Y_a, X_a)],
        output_dtypes=[mx.uint32],
    )
    mx.eval(out)
    return np.asarray(out).astype(out_dtype, copy=False)


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
    use_fused_kernel: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Resample target-spacing logits to acquisition spacing and argmax.

    Strategy (default, ``use_fused_kernel=True``): a single fused Metal
    kernel with one thread per *output* voxel. Each thread trilinear-
    interpolates all K channels inline — reading only the 8 source corners
    per channel — and tracks the argmax on the fly, writing one integer
    label. Nothing K-channel-sized is materialized, so there is no slab
    budget to tune and ``peak_working_memory_mb`` is ignored on this path.
    ~100× the slab path on a 512²/117-channel job (24.7 s → 0.25 s).

    Fallback (``use_fused_kernel=False``, or if the kernel errors): a Z-slab
    loop with explicit-coordinate trilinear over all K channels, gathering
    the 8 corners into full ``(K, S, Y_a, X_a)`` arrays, blending, then a
    separate argmax. Slab depth auto-sized from ``peak_working_memory_mb``
    so the K-channel acquisition-spacing slab fits the budget.

    The two paths agree bit-for-bit on synthetic logits; on real smooth
    logit fields a handful of boundary voxels (≈4e-5) can flip where the
    Metal compiler's FMA contraction rounds differently than the host MLX
    mul/add — negligible next to MLX↔PyTorch numeric divergence.

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

    ``peak_working_memory_mb`` applies only to the slab fallback (the fused
    kernel needs no transient K-channel buffer). ``None`` (default) auto-
    detects from system RAM: 200 MB on < 32 GB Macs, 2000 MB on ≥ 32 GB.

    ``use_fused_kernel`` (default ``True``) selects the fused Metal path;
    set ``False`` to force the pure-MLX slab loop (e.g. for debugging or on
    a backend without the custom-kernel runtime).
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

    # Fast path: a single fused Metal kernel (trilinear + argmax inline, no
    # K-channel materialization, ~100× the slab path). Falls back to the
    # pure-MLX slab loop if the kernel is unavailable or errors.
    if use_fused_kernel:
        try:
            return _fused_resample_argmax(
                logits_target, out_shape_zyx,
                target_spacing_zyx, acq_spacing_zyx, out_dtype,
            )
        except Exception as e:  # pragma: no cover - depends on Metal runtime
            if verbose:
                print(f"  [inverse_resample_argmax] fused kernel failed "
                      f"({e!r}); falling back to slab path")

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
    use_fused_kernel: bool = True,
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
    rco = tuple(int(v) for v in regions_class_order)

    if use_fused_kernel:
        try:
            return _fused_resample_paint(
                logits_target, out_shape_zyx,
                target_spacing_zyx, acq_spacing_zyx,
                rco, float(threshold), out_dtype,
            )
        except Exception:  # pragma: no cover - depends on Metal runtime
            pass

    return _slab_resample_paint(
        logits_target, out_shape_zyx,
        target_spacing_zyx, acq_spacing_zyx,
        peak_working_memory_mb,
        rco,
        float(threshold),
        out_dtype,
    )


# ---------------------------------------------------------------------------
# High-level: SITK in, SITK out
# ---------------------------------------------------------------------------

def get_orientation(image: "sitk.Image") -> str:
    """Return the 3-letter DICOM-style orientation code for a SITK image.

    e.g. ``"LPS"`` (canonical: Left-Posterior-Superior voxel-axis directions),
    ``"RAS"``, or for oblique scans something like ``"SAR"``.

    The code describes the *anatomical* direction each voxel axis points
    toward — first letter for the +X voxel axis, second for +Y, third for +Z.
    """
    sitk = _require_sitk()
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


# Per-letter voxel-axis direction in SITK's LPS world frame.
_LETTER_DIR = {
    "L": (1.0, 0.0, 0.0), "R": (-1.0, 0.0, 0.0),
    "P": (0.0, 1.0, 0.0), "A": (0.0, -1.0, 0.0),
    "S": (0.0, 0.0, 1.0), "I": (0.0, 0.0, -1.0),
}


def _code_direction(code: str) -> np.ndarray:
    """3x3 direction matrix (columns = voxel-axis world dirs) for a DICOM code."""
    code = code.upper()
    if len(code) != 3 or any(c not in _LETTER_DIR for c in code):
        raise ValueError(f"invalid orientation code {code!r}")
    return np.array([_LETTER_DIR[c] for c in code], dtype=np.float64).T


def reorient_array_mlx(
    arr_zyx: "mx.array | np.ndarray",
    *,
    direction_xyz: tuple[float, ...],
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    target_code: str,
):
    """Reorient a (Z, Y, X) array to a DICOM-style ``target_code`` on the GPU.

    Pure axis permutation + flips (no interpolation), derived from the direction
    cosines — bit-identical to ``sitk.DICOMOrient`` but done as ``mx.transpose`` /
    ``mx.flip`` on Metal (~0.46 s vs ~3 s on a 418 M-voxel CPU shuffle, since
    reorientation is memory-bandwidth-bound and the GPU has ~10× the bandwidth).

    Returns ``(out_zyx: mx.array, new_geometry: Geometry)``. Handles any
    axis-aligned input/target orientation (RAS/LPS and the rest).
    """
    from .values import Geometry

    if not isinstance(arr_zyx, mx.array):
        arr_zyx = mx.array(np.asarray(arr_zyx))
    D = np.array(direction_xyz, dtype=np.float64).reshape(3, 3)
    sp_xyz = np.array(tuple(reversed(spacing_zyx)), dtype=np.float64)
    org = np.array(origin_xyz, dtype=np.float64)
    size_xyz = np.array((arr_zyx.shape[2], arr_zyx.shape[1], arr_zyx.shape[0]), dtype=np.float64)

    Tt = _code_direction(target_code)
    M = D.T @ Tt                                  # M[i,j] = D[:,i]·Tt[:,j]
    perm = np.argmax(np.abs(M), axis=0)           # output voxel axis j <- input axis perm[j]
    signs = np.sign(M[perm, np.arange(3)])

    # SITK axis a corresponds to numpy axis (2-a). Build the numpy transpose order.
    np_perm = [0, 0, 0]
    for j in range(3):
        np_perm[2 - j] = int(2 - perm[j])
    out = mx.transpose(arr_zyx, np_perm)
    for j in range(3):
        if signs[j] < 0:
            ax = 2 - j
            out = out[tuple(slice(None, None, -1) if k == ax else slice(None)
                            for k in range(out.ndim))]
    out = mx.contiguous(out)

    new_sp_xyz = sp_xyz[perm]
    in_idx = np.zeros(3)
    for j in range(3):
        if signs[j] < 0:
            in_idx[perm[j]] = size_xyz[perm[j]] - 1
    new_org = org + D @ (sp_xyz * in_idx)

    new_geom = Geometry(
        spacing_zyx=tuple(reversed(new_sp_xyz.tolist())),
        shape_zyx=(out.shape[0], out.shape[1], out.shape[2]),
        origin_xyz=tuple(new_org.tolist()),
        direction_xyz=tuple(Tt.reshape(-1).tolist()),
    )
    return out, new_geom


def reorient(image: "sitk.Image", code: str) -> "sitk.Image":
    """Reorient ``image`` so its voxel axes map to the given DICOM code.

    Thin wrapper over ``sitk.DICOMOrient`` that no-ops when the image is
    already in the requested orientation. Exists at top level so callers
    composing custom pipelines (e.g. a multi-task union workflow) can
    invoke the same reorient logic ``predict_with_resampling`` uses.

    Parameters
    ----------
    code :
        Three-letter DICOM-style code, e.g. ``"RAS"`` (the nnU-Net /
        TotalSegmentator / nibabel canonical — the orientation the models were
        trained on; the inference default), ``"LPS"`` (DICOM/SITK world
        convention — **mirrors L↔R vs RAS, so do not feed it to the model**),
        ``"SAR"`` (some oblique CT scans).

    Returns
    -------
    sitk.Image
        Either ``image`` unchanged (if it's already in ``code``) or a new
        SITK image with the requested orientation. Geometry attributes
        (spacing, origin, direction, size) reflect the new orientation.

    Round-trip pattern::

        orig = get_orientation(image)
        canonical = reorient(image, "RAS")
        # ... inference / pipeline at canonical orientation ...
        result = reorient(seg, orig)
    """
    sitk = _require_sitk()
    if get_orientation(image) == code:
        return image
    return sitk.DICOMOrient(image, code)


__all__ = [
    "resample_image_to_target",
    "resample_volume_mlx",
    "reorient_array_mlx",
    "inverse_resample_argmax",
    "inverse_resample_paint",
    "get_orientation",
    "reorient",
]
