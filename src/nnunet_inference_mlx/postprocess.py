"""postprocess — turn a :class:`Prediction` into a :class:`Segmentation` or a :class:`Mesh`.

Conversions of the model's per-class output:

* ``to_labels(prediction)`` — labels at the prediction's *own* grid (no
  resample). Scheme-aware (argmax for standard models, threshold +
  paint-priority for region models). The cheap "what did the model say
  here" view.
* ``restore(prediction, plan)`` — the full inverse: resample the per-class
  logits back to the caller's grid (via the proven slab-streaming
  ``inverse_resample_*``), then reorient to the original orientation. This
  resamples *logits then argmaxes* (higher quality than argmax-then-
  resample-labels), which is why ``segment`` is ``predict → restore``
  rather than ``predict → to_labels → resample``.
* ``to_mesh(prediction)`` — surface mesh at the prediction's *own* grid,
  via SurfaceNets-from-logits. Sibling of ``to_labels`` (same input, same
  grid; just produces a surface instead of a labelmap).
* ``resample_prediction(prediction, ...)`` — trilinear K-channel resample
  of the logit volume to a target grid. Lets the user choose the
  segmentation/mesh resolution independently of the network's training
  spacing. Composes naturally with ``to_labels`` / ``to_mesh``.

Plus ``drop_small_components`` for connected-component cleanup.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np

from .values import Geometry, LabelSchema, Mesh, Prediction, RestorePlan, Segmentation


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


def to_mesh(
    prediction: Prediction,
    *,
    project_to_surface: bool = False,
    emit_normals: bool = False,
    confidence_margin: float = 0.0,
    confidence_threshold: float = 0.0,
    drop_components_below_mm3: float = 0.0,
) -> Mesh:
    """Extract a SurfaceNets dual mesh from the prediction's logits.

    Sibling of :func:`to_labels` — same input (a :class:`Prediction` at
    the network's training spacing), same grid (no resample), produces a
    surface mesh instead of a label map. Vertex positions come from
    sub-voxel edge-crossing interpolation in the continuous logit field;
    quads carry the VTK ``(Label0, Label1)`` convention so the mesh drops
    straight into VTK-side pipelines (see :func:`meshio.mesh_to_vtk_polydata`).

    Standard schema only — region-model meshing is not yet supported (it
    needs a different "is this label dominant here" rule and is gated on
    a region-model port; raises ``NotImplementedError`` until then).

    Parameters
    ----------
    project_to_surface :
        If True, do one Newton step toward ``logit_i = logit_j`` per
        binary-cell vertex — places it on the actual decision surface
        instead of the centroid of its edge crossings.
    emit_normals :
        If True, attach per-vertex normals computed from the logit
        gradient ``∇(logit_i − logit_j)``. Independent of mesh
        discretization; usually visibly smoother than VTK's
        averaged-face-normal computation.
    confidence_margin :
        Margin threshold for the spike-voxel edge filter. A voxel is
        a spike if (top1−top2 logit margin < confidence_margin) AND
        (no 6-connected same-label neighbor) — the same dual condition
        as ``confidence_threshold``, but applied as a *non-destructive*
        edge filter (the voxel keeps its argmax; only its outgoing
        crossings are dropped, killing the octahedron spike topology).
        Composes cleanly with the geometric refinements above. Values
        0.5–2.0 typically clean the noise floor on TS-fast logits;
        0.0 (default) leaves the topology criterion at plain argmax.
        Recommended over ``confidence_threshold`` because it does
        not perturb the label volume.
    confidence_threshold :
        Logit-margin floor for treating an argmax decision as confident.
        Voxels whose top-1 vs top-2 margin falls below this AND whose
        6-connected neighbors unanimously carry the same other label
        are relabeled to that neighbor. Targeted fix for sub-Nyquist
        single-voxel "blob" octahedron artifacts. 0.0 (default) leaves
        labels untouched; values in 0.5–1.5 typically clean the noise
        floor without suppressing real thin features.
    drop_components_below_mm3 :
        Drop connected components of any label whose physical volume is
        below this threshold (26-conn, multi-label aware). Catches
        noise clusters too large for ``confidence_threshold`` to reach.
        ``200.0`` matches TS's ``--remove_small_blobs``. Requires
        ``cc3d``; 0.0 (default) is a no-op.

    See :func:`mesh.surfacenets_logits` for algorithm details, the
    triple-junction rule, and volume-boundary closure caveats.
    """
    if prediction.schema.is_region_model:
        raise NotImplementedError(
            "to_mesh does not yet support region-based label schemas; "
            "only standard argmax schemas are handled."
        )
    from .mesh import surfacenets_logits

    # Pass the mx.array through directly — surfacenets_logits runs
    # the K-channel dense ops (argmax, top-2 margin, edge crossings,
    # etc.) on the GPU. Wrapping in np.asarray here would round-trip
    # a 3 GB copy back to MLX inside surfacenets_logits.
    return surfacenets_logits(
        prediction.data, prediction.geometry, prediction.schema,
        project_to_surface=project_to_surface,
        emit_normals=emit_normals,
        confidence_margin=confidence_margin,
        confidence_threshold=confidence_threshold,
        drop_components_below_mm3=drop_components_below_mm3,
    )


def resample_prediction(
    prediction: Prediction,
    *,
    output_spacing_zyx: tuple[float, float, float] | None = None,
    output_shape_zyx: tuple[int, int, int] | None = None,
    scale: float | tuple[float, float, float] | None = None,
    cascade_for_aliasing: bool = True,
    memory_ceiling_gb: float = 12.0,
) -> Prediction:
    """Trilinear-resample the K-channel logit volume to a target grid.

    Specify exactly one target form:

    * ``output_spacing_zyx`` — target voxel spacing in mm (Z, Y, X).
      Output shape derived to preserve the prediction's physical extent.
    * ``output_shape_zyx`` — exact target voxel counts (Z, Y, X). Output
      spacing derived from the input's physical extent.
    * ``scale`` — multiplier on the input shape. ``scale > 1`` upsamples
      (finer, denser); ``scale < 1`` downsamples (coarser, smaller).
      Scalar (uniform) or 3-tuple (per-axis).

    For aggressive downsamples (target spacing more than 2× the source
    per axis) the resample cascades through intermediate 2× steps to
    avoid trilinear undersampling — the same pattern
    :func:`restore` uses for logits-to-acquisition-spacing. Set
    ``cascade_for_aliasing=False`` to skip (faster, but aliases on
    heavy downsamples).

    Composes naturally:

        # Mesh at finer-than-training resolution
        hi = resample_prediction(prediction, scale=2.0)
        mesh = to_mesh(hi, confidence_margin=1.0, ...)

        # Quick coarse preview
        lo = resample_prediction(prediction, scale=0.5)
        seg = to_labels(lo)

        # Or aim at a specific clinical voxel size
        clin = resample_prediction(prediction, output_spacing_zyx=(1.5, 1.5, 1.5))

    Parameters
    ----------
    memory_ceiling_gb :
        Refuse to run if the **peak** in-flight memory would exceed
        this. The trilinear gather materializes 8 K-channel intermediate
        arrays (one per cube corner) during compute, so peak memory is
        about **9× the output prediction size**. For chest TS-fast at
        ``scale=(1.5, 1, 1)`` (output 4.6 GB) peak is ~37 GB — well
        beyond M2 17 GB unified memory, and the OS gets killed before
        Python sees a clean exception. Default ceiling 12 GB
        corresponds to ~1.3 GB output, conservative for M2 17 GB.
        Bump if you have more RAM, but be careful: the failure mode
        is a *machine crash*, not a Python error. The path forward
        for legitimately-large outputs is the slab-streaming variant
        (not yet implemented).

    Returns
    -------
    Prediction
        A new prediction at the target grid. Schema, activation, and
        geometry origin/direction are unchanged; only ``spacing_zyx``,
        ``shape_zyx``, and ``data`` change.
    """
    n_set = sum(x is not None for x in (output_spacing_zyx, output_shape_zyx, scale))
    if n_set != 1:
        raise ValueError(
            "specify exactly one of output_spacing_zyx, output_shape_zyx, or "
            f"scale; got {n_set}"
        )

    src_spacing = tuple(float(s) for s in prediction.geometry.spacing_zyx)
    src_shape = tuple(int(n) for n in prediction.geometry.shape_zyx)

    if output_spacing_zyx is not None:
        tgt_spacing = tuple(float(s) for s in output_spacing_zyx)
        if len(tgt_spacing) != 3 or any(s <= 0 for s in tgt_spacing):
            raise ValueError(f"output_spacing_zyx must be 3 positive floats; got {output_spacing_zyx!r}")
        tgt_shape = tuple(
            max(1, int(round(n * s_in / s_out)))
            for n, s_in, s_out in zip(src_shape, src_spacing, tgt_spacing)
        )
    elif output_shape_zyx is not None:
        tgt_shape = tuple(int(n) for n in output_shape_zyx)
        if len(tgt_shape) != 3 or any(n < 1 for n in tgt_shape):
            raise ValueError(f"output_shape_zyx must be 3 positive ints; got {output_shape_zyx!r}")
        tgt_spacing = tuple(
            s_in * n_in / n_out
            for s_in, n_in, n_out in zip(src_spacing, src_shape, tgt_shape)
        )
    else:
        if isinstance(scale, (int, float)):
            sc = (float(scale), float(scale), float(scale))
        else:
            sc = tuple(float(s) for s in scale)  # type: ignore[arg-type]
        if len(sc) != 3 or any(s <= 0 for s in sc):
            raise ValueError(f"scale must be a positive scalar or 3-tuple; got {scale!r}")
        tgt_shape = tuple(max(1, int(round(n * s))) for n, s in zip(src_shape, sc))
        tgt_spacing = tuple(s_in / s for s_in, s in zip(src_spacing, sc))

    # No-op fast path.
    if tgt_shape == src_shape and all(
        abs(a - b) < 1e-6 for a, b in zip(tgt_spacing, src_spacing)
    ):
        return prediction

    K = int(prediction.data.shape[0])
    output_bytes = K * tgt_shape[0] * tgt_shape[1] * tgt_shape[2] * 4
    # Trilinear gather materializes 8 K-channel corner arrays + the final
    # blend during compute. Conservative peak factor of 9 — empirically
    # an output of 4.6 GB peaks well above 30 GB and kernel-panics M2
    # 17 GB. Refusing here is the only safe behavior because the failure
    # mode below the limit is a system-wide OOM, not a Python exception.
    PEAK_FACTOR = 9
    peak_bytes = output_bytes * PEAK_FACTOR
    ceiling_bytes = int(memory_ceiling_gb * (1024 ** 3))
    if peak_bytes > ceiling_bytes:
        raise MemoryError(
            f"Resampled prediction is {output_bytes / (1024**3):.2f} GB "
            f"(K={K}, shape={tgt_shape}, fp32); the trilinear gather "
            f"peaks at ~{peak_bytes / (1024**3):.1f} GB during compute "
            f"(8 K-channel intermediate arrays + final blend), exceeding "
            f"the {memory_ceiling_gb:g} GB peak ceiling. Pick a smaller "
            f"scale, use the per-axis scale form to upsample only the "
            f"coarsest axis, or downsample. The streaming variant "
            f"(slab-by-slab; see roadmap) is the path for large outputs."
        )

    from .resampling import _cascade_kchannel_to_target, _kchannel_trilinear_full

    needs_cascade = cascade_for_aliasing and any(
        s_out > 2.001 * s_in for s_in, s_out in zip(src_spacing, tgt_spacing)
    )
    if needs_cascade:
        new_data = _cascade_kchannel_to_target(
            prediction.data, tgt_shape, src_spacing, tgt_spacing,
        )
    else:
        new_data = _kchannel_trilinear_full(
            prediction.data, tgt_shape, src_spacing, tgt_spacing,
        )

    new_geometry = replace(
        prediction.geometry,
        spacing_zyx=tgt_spacing,
        shape_zyx=tgt_shape,
    )
    return replace(prediction, data=new_data, geometry=new_geometry)


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


__all__ = ["to_labels", "restore", "to_mesh", "drop_small_components"]
