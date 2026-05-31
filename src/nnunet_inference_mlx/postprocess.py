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
    peak_working_memory_mb: int | None = None,
) -> Segmentation:
    """Inverse-resample model-frame logits back onto the caller's grid.

    Resamples the per-class logits from the model spacing to
    ``plan.inference_geometry`` (the canonical-orientation source-spacing
    grid) — scheme-aware, argmax/paint fused into the slab loop — then
    reorients from ``plan.inference_orientation`` to ``plan.source_orientation``.

    With no override the result lands on the caller's input grid. Pass
    ``target_spacing`` (absolute mm; scalar isotropic or (Z,Y,X)) or
    ``target_scaling`` (resolution multiplier; 2 = finer, 0.5 = coarser) to
    render the labels at a different resolution — *from the logits*, then argmax,
    so it's higher quality than resampling a finished label map. The output
    header is recomputed to the new spacing/shape over the same physical extent,
    so it still overlays the input. (Aggressive downsampling is Nyquist-limited:
    structures thinner than the new voxel are lost regardless.)
    """
    if tuple(prediction.geometry.spacing_zyx) != tuple(plan.model_spacing_zyx):
        raise ValueError(
            f"prediction spacing {prediction.geometry.spacing_zyx} does not match "
            f"plan.model_spacing_zyx {plan.model_spacing_zyx}; prediction was not "
            "produced by this plan's to_model_frame step."
        )
    from .imageio import array_to_sitk, sitk_to_segmentation
    from .resampling import (
        inverse_resample_argmax,
        inverse_resample_paint,
        reorient as _reorient,
    )

    schema = prediction.schema
    grid = _resolve_output_grid(plan.inference_geometry, target_spacing, target_scaling)
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

    seg_img = array_to_sitk(seg_zyx, grid)
    if plan.source_orientation != plan.inference_orientation:
        seg_img = _reorient(seg_img, plan.source_orientation)
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
