"""postprocess — turn a :class:`Prediction` into a :class:`Segmentation`.

Two conversions, both scheme-aware (argmax for standard models, threshold +
paint-priority for region models):

* ``to_labels(prediction)`` — labels at the prediction's *own* grid (no
  resample). The cheap "what did the model say here" view.
* ``restore(prediction, plan)`` — the full inverse: resample the per-class
  logits back to the caller's grid (via the proven slab-streaming
  ``inverse_resample_*``), then reorient to the original orientation. This
  resamples *logits then argmaxes* (higher quality than argmax-then-resample-
  labels), which is why ``segment`` is ``predict → restore`` rather than
  ``predict → to_labels → resample``.

Plus ``drop_small_components`` for connected-component cleanup.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .values import LabelSchema, Prediction, RestorePlan, Segmentation


def _label_dtype(schema: LabelSchema) -> np.dtype:
    """Smallest unsigned int dtype that fits every label value in ``schema``."""
    values = list(schema.names.keys())
    values += list(schema.paint_priority)
    for r in schema.regions:
        values.append(r.label_value)
    max_label = max((int(v) for v in values), default=0)
    if max_label <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_label <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _paint_threshold(prediction: Prediction) -> float:
    """Region membership cut: 0 for raw logits, 0.5 for sigmoid'd probs."""
    return 0.5 if prediction.activation == "sigmoid" else 0.0


def to_labels(prediction: Prediction) -> Segmentation:
    """Convert per-class output to integer labels at the prediction's grid.

    Standard schema → ``argmax`` over channels. Region schema → per-region
    threshold + paint in ``paint_priority`` order (later regions win overlaps).
    No resampling: the result shares ``prediction.geometry``.
    """
    schema = prediction.schema
    out_dtype = _label_dtype(schema)

    if schema.is_region_model:
        threshold = _paint_threshold(prediction)
        logits_np = np.asarray(prediction.data)            # (K, Z, Y, X)
        out = np.zeros(prediction.geometry.shape_zyx, dtype=out_dtype)
        for i, label_value in enumerate(schema.paint_priority):
            out[logits_np[i] > threshold] = label_value
        data = mx.array(out)
    else:
        labels = np.asarray(mx.argmax(prediction.data, axis=0)).astype(out_dtype, copy=False)
        data = mx.array(labels)

    return Segmentation(data=data, geometry=prediction.geometry, schema=schema)


def _resolve_output_grid(grid, target_spacing, target_scaling):
    """The output grid for restore: same FOV/origin/orientation as ``grid``,
    at a possibly different spacing.

    ``grid`` is the caller's (canonical, source-spacing) grid. With neither
    override, returns ``grid`` unchanged (identity). ``target_spacing`` is
    absolute mm (scalar → isotropic, or a (Z,Y,X) tuple); ``target_scaling`` is
    a resolution multiplier (2 → finer/half-spacing, 0.5 → coarser). Shape is
    recomputed as ``round(N · spacing / new_spacing)`` so the physical extent is
    preserved (origin fixed → voxel-(0,0,0) centers aligned, so the extent holds
    to within sub-voxel rounding). Mutually exclusive overrides.
    """
    if target_spacing is not None and target_scaling is not None:
        raise ValueError("pass target_spacing or target_scaling, not both")

    src_spacing = tuple(float(s) for s in grid.spacing_zyx)
    if target_spacing is None and target_scaling is None:
        return grid
    if target_scaling is not None:
        s = float(target_scaling)
        if s <= 0:
            raise ValueError(f"target_scaling must be > 0; got {target_scaling}")
        new_spacing = tuple(v / s for v in src_spacing)
    else:
        if isinstance(target_spacing, (int, float)):
            new_spacing = (float(target_spacing),) * 3
        else:
            new_spacing = tuple(float(v) for v in target_spacing)
            if len(new_spacing) != 3:
                raise ValueError(f"target_spacing must be scalar or 3 values (Z,Y,X); got {target_spacing}")
        if any(v <= 0 for v in new_spacing):
            raise ValueError(f"target_spacing values must be > 0; got {new_spacing}")

    new_shape = tuple(
        max(1, int(round(n * s / t)))
        for n, s, t in zip(grid.shape_zyx, src_spacing, new_spacing)
    )
    from .values import Geometry
    return Geometry(spacing_zyx=new_spacing, shape_zyx=new_shape,
                    origin_xyz=grid.origin_xyz, direction_xyz=grid.direction_xyz)


def restore(
    prediction: Prediction,
    plan: RestorePlan,
    *,
    target_spacing: "float | tuple[float, float, float] | None" = None,
    target_scaling: float | None = None,
    interpolation: str = "linear",
    peak_working_memory_mb: int | None = None,
) -> Segmentation:
    """Inverse-resample model-frame logits back onto the caller's grid.

    ``interpolation="linear"`` (default, path B) resamples the per-class *logits*
    to the output grid (scheme-aware, argmax/paint fused into the slab loop),
    then reorients back — higher fidelity than resampling a finished label map,
    matching vanilla nnU-Net. ``interpolation="nearest"`` (path A) is the fast
    TS-style alternative: argmax at the model spacing, then nearest-neighbour
    resample the *integer label map* — single-channel, ~no 117-channel gather, so
    much faster on large grids, at the cost of stair-stepped boundaries.

    With no spacing override the result lands on the caller's input grid; pass
    ``target_spacing`` (absolute mm; scalar or (Z,Y,X)) or ``target_scaling``
    (multiplier; 2 = finer, 0.5 = coarser) to render at a different resolution
    (header recomputed over the same physical extent). Downsampling is
    Nyquist-limited regardless of method.
    """
    if tuple(prediction.geometry.spacing_zyx) != tuple(plan.model_spacing_zyx):
        raise ValueError(
            f"prediction spacing {prediction.geometry.spacing_zyx} does not match "
            f"plan.model_spacing_zyx {plan.model_spacing_zyx}; prediction was not "
            "produced by this plan's to_model_frame step."
        )
    if interpolation not in ("linear", "nearest"):
        raise ValueError(f"interpolation must be 'linear' or 'nearest'; got {interpolation!r}")
    from .imageio import array_to_sitk, geometry_from_sitk, sitk_to_segmentation
    from .resampling import (
        inverse_resample_argmax,
        inverse_resample_paint,
        reorient_array_mlx,
    )

    schema = prediction.schema
    grid = _resolve_output_grid(plan.inference_geometry, target_spacing, target_scaling)

    # Path A (fast, TS-style): argmax at model spacing, then NN-resample the
    # single-channel label map onto the output grid. Same orientation/origin as
    # the prediction, so SITK's world-coordinate resample handles the spacing change.
    if interpolation == "nearest":
        import SimpleITK as sitk
        labels_model = to_labels(prediction)                 # (Z,Y,X) at model spacing
        src = array_to_sitk(np.asarray(labels_model.data), prediction.geometry)
        ref = array_to_sitk(np.zeros(grid.shape_zyx, dtype=np.asarray(labels_model.data).dtype), grid)
        out = sitk.Resample(src, ref, sitk.Transform(), sitk.sitkNearestNeighbor,
                            0, src.GetPixelID())
        if plan.source_orientation != plan.inference_orientation:
            og = geometry_from_sitk(out)
            oa, ogeom = reorient_array_mlx(
                sitk.GetArrayFromImage(out), direction_xyz=og.direction_xyz,
                spacing_zyx=og.spacing_zyx, origin_xyz=og.origin_xyz,
                target_code=plan.source_orientation)
            out = array_to_sitk(np.asarray(oa), ogeom)
        return sitk_to_segmentation(out, schema)

    out_shape = grid.shape_zyx
    target_spacing_zyx = tuple(plan.model_spacing_zyx)
    acq_spacing = tuple(grid.spacing_zyx)
    out_dtype = _label_dtype(schema)

    if schema.is_region_model:
        seg_zyx = inverse_resample_paint(
            prediction.data, out_shape, target_spacing_zyx, acq_spacing,
            regions_class_order=schema.paint_priority,
            threshold=_paint_threshold(prediction),
            out_dtype=out_dtype,
            peak_working_memory_mb=peak_working_memory_mb,
        )
    else:
        seg_zyx = inverse_resample_argmax(
            prediction.data, out_shape, target_spacing_zyx, acq_spacing,
            out_dtype=out_dtype,
            peak_working_memory_mb=peak_working_memory_mb,
        )

    if plan.source_orientation != plan.inference_orientation:
        # GPU reorient back to the input orientation (transpose+flip, ~6x faster
        # than SITK DICOMOrient on a large label map).
        oa, ogeom = reorient_array_mlx(
            seg_zyx, direction_xyz=grid.direction_xyz,
            spacing_zyx=grid.spacing_zyx, origin_xyz=grid.origin_xyz,
            target_code=plan.source_orientation)
        seg_img = array_to_sitk(np.asarray(oa), ogeom)
    else:
        seg_img = array_to_sitk(seg_zyx, grid)
    return sitk_to_segmentation(seg_img, schema)


def drop_small_components(
    segmentation: Segmentation,
    *,
    min_volume_mm3: float,
    in_place: bool = False,
) -> Segmentation:
    """Drop connected components smaller than ``min_volume_mm3`` (multi-label).

    Requires the ``[postprocessing]`` extra (cc3d). ``200.0`` matches
    TotalSegmentator's ``--remove_small_blobs``.
    """
    from .postprocessing import remove_small_components

    arr = np.asarray(segmentation.data)
    cleaned = remove_small_components(
        arr, segmentation.geometry.spacing_zyx,
        min_volume_mm3=min_volume_mm3, in_place=in_place,
    )
    return segmentation.with_data(mx.array(cleaned))


__all__ = ["to_labels", "restore", "drop_small_components"]
