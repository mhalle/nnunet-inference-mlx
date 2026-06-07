"""
Label-scheme post-processing: logits → integer segmentation.

Two schemes are supported, dispatched by inspecting ``dataset.json``:

* **Standard N-class softmax.** Each label is a single int. The model has
  one softmax output per class; conversion is plain ``argmax(axis=0)``.
* **Region-based.** Each "label" is a union of underlying classes (a list
  of ints). The model has one *independent sigmoid* head per region — they
  are not softmax-related. Conversion is "paint each region in priority
  order from ``regions_class_order``." Used by BraTS-style brain-tumor
  models and several medical-imaging-challenge datasets.

Standard:                  Region-based:
  "labels": {                "labels": {
    "background": 0,           "background": 0,
    "spleen":     1,           "WT":  [1, 2, 3],
    "kidney":     2            "TC":  [1, 3],
    ...                        "ET":  3
  }                          },
                             "regions_class_order": [2, 1, 3]

Region-based dispatch is automatic — callers don't need to inspect the
dataset.json themselves.
"""

from __future__ import annotations

import numpy as np


def has_regions(dataset_json: dict) -> bool:
    """True iff ``dataset_json["labels"]`` defines any label as a union
    of base classes (a list/tuple of ints) — the region-based scheme.

    A single int value is a plain class; a list/tuple of ints is a region.
    Mixed dicts (e.g. ``background=0`` plus regions) count as region-based.
    """
    labels = dataset_json.get("labels", {})
    return any(isinstance(v, (list, tuple)) for v in labels.values())


def regions_class_order(dataset_json: dict) -> tuple[int, ...]:
    """Region paint-priority order. Empty tuple for non-region datasets.

    Required for region-based datasets; raises ``ValueError`` if missing.
    """
    if not has_regions(dataset_json):
        return ()
    order = dataset_json.get("regions_class_order")
    if order is None:
        raise ValueError(
            "dataset_json has region-based labels but no regions_class_order; "
            "cannot deterministically convert region predictions to labels."
        )
    return tuple(int(c) for c in order)


def label_dtype(dataset_json: dict) -> np.dtype:
    """Smallest unsigned integer dtype that fits every label value the
    dataset can produce.

    Inspects both ``labels`` (region member classes and standard labels)
    and ``regions_class_order`` (the integer values actually painted at
    inference for region-based models). Returns ``uint8`` / ``uint16`` /
    ``uint32`` per nnUNetv2's convention — small enough to be efficient
    on disk and in memory, large enough to hold every label without
    silent truncation.
    """
    labels = dataset_json.get("labels", {})
    max_val = 0
    for v in labels.values():
        if isinstance(v, (list, tuple)):
            if v:
                max_val = max(max_val, max(int(x) for x in v))
        else:
            max_val = max(max_val, int(v))
    for v in dataset_json.get("regions_class_order", ()):
        max_val = max(max_val, int(v))

    if max_val <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_val <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def sigmoid_inplace(arr: np.ndarray) -> np.ndarray:
    """Apply the logistic sigmoid in place to a float32 array.

    Numerically safe for any logit range. Inputs are clipped to ``±88``
    before ``exp`` to keep float32 ``exp(-x)`` in finite range; outside
    that band the sigmoid is already ``0`` or ``1`` to float32 precision,
    so the clip is exact for the values we care about.
    """
    np.clip(arr, -88.0, 88.0, out=arr)
    np.negative(arr, out=arr)
    np.exp(arr, out=arr)
    arr += 1.0
    np.reciprocal(arr, out=arr)
    return arr


def convert_logits_to_segmentation(
    pred: np.ndarray,
    dataset_json: dict,
    threshold: float = 0.0,
    dtype: str | np.dtype | None = None,
) -> np.ndarray:
    """Convert per-channel predictions into an integer segmentation map.

    Parameters
    ----------
    pred : np.ndarray
        Shape ``(K, ...)``, float32. For standard models, ``K`` is the
        class count and these are logits or softmax probabilities (the
        result is the same after ``argmax``). For region-based models,
        ``K`` is the number of regions and these are *per-region* sigmoid
        logits or post-sigmoid probabilities.
    dataset_json : dict
        Parsed dataset.json; determines the label scheme.
    threshold : float, default 0.0
        Cut for region-based predictions. ``0.0`` matches "logit > 0",
        i.e. "sigmoid > 0.5"; pass ``0.5`` if you've already applied
        sigmoid and have probabilities. Ignored for standard datasets.
    dtype : str, np.dtype, or None
        Output integer dtype. ``None`` (default) auto-picks the smallest
        unsigned integer that fits every label value (``uint8`` / ``uint16``
        / ``uint32``) via :func:`label_dtype`. Pass an explicit dtype
        (``"uint16"``, ``np.int32``, etc.) to override — useful when a
        downstream tool requires a specific integer width regardless of
        what the dataset actually needs.

    Returns
    -------
    np.ndarray
        Shape ``pred.shape[1:]``, dtype as resolved above.
    """
    out_dtype = np.dtype(dtype) if dtype is not None else label_dtype(dataset_json)

    if not has_regions(dataset_json):
        return np.argmax(pred, axis=0).astype(out_dtype)

    order = regions_class_order(dataset_json)
    if pred.shape[0] != len(order):
        raise ValueError(
            f"Region prediction has {pred.shape[0]} channels but "
            f"regions_class_order has {len(order)} entries."
        )
    seg = np.zeros(pred.shape[1:], dtype=out_dtype)
    for region_idx, label in enumerate(order):
        seg[pred[region_idx] > threshold] = label
    return seg


def remap_labels(
    seg: np.ndarray,
    mapping: dict[int, int],
    *,
    out_dtype: np.dtype | str | None = None,
    background: int = 0,
) -> np.ndarray:
    """Remap integer labels in ``seg`` via a lookup table.

    Designed for the multi-task union pattern: each task predicts labels
    in its own local ID space (TS lung_vessels uses 1=artery, 2=vein...)
    and the caller wants to fold them into a shared unified space
    (1=lung_artery, 2=lung_vein in the TS full-mode label scheme). A
    plain ``dict`` is the cleanest spelling of that mapping.

    Implementation is a vectorized LUT — a single ``arr[lut]`` indexed
    lookup, no Python loop over voxels. The LUT is sized to the maximum
    source value, so very sparse mappings on very large source values
    cost RAM proportional to the max — fine for ``< 10000`` (typical
    class counts) but a poor fit for arbitrary remapping.

    Parameters
    ----------
    seg :
        Integer label volume of any shape. Must be a NumPy array.
    mapping :
        ``{source_id: target_id}``. Source IDs not listed are remapped
        to ``background`` — explicit drop. You can put ``background``
        on the source side to remap it to a foreground value, though
        it's unusual.
    out_dtype :
        Output dtype. ``None`` (default) picks the smallest unsigned
        integer that fits every target value in ``mapping`` and
        ``background``. Pass an explicit dtype to override.
    background :
        The fill value for source IDs not in ``mapping``. Default ``0``
        (the standard nnU-Net background).

    Returns
    -------
    np.ndarray
        Same shape as ``seg``, with values from the target side of
        ``mapping``.

    Raises
    ------
    ValueError
        If ``mapping`` contains negative IDs (LUTs are non-negative).
    """
    if not isinstance(seg, np.ndarray):
        raise TypeError(f"seg must be a numpy array, got {type(seg).__name__}")

    for k, v in mapping.items():
        if int(k) < 0 or int(v) < 0:
            raise ValueError(
                f"remap_labels does not support negative IDs (got {k} -> {v})"
            )

    # Resolve target dtype from the values we'll write.
    if out_dtype is None:
        max_target = max([int(background), *map(int, mapping.values())], default=0)
        if max_target <= np.iinfo(np.uint8).max:
            out_dtype = np.dtype(np.uint8)
        elif max_target <= np.iinfo(np.uint16).max:
            out_dtype = np.dtype(np.uint16)
        else:
            out_dtype = np.dtype(np.uint32)
    else:
        out_dtype = np.dtype(out_dtype)

    # Build the LUT. Size it to cover every possible source value we'd
    # see in ``seg`` (i.e. seg.max()) and every key in ``mapping``.
    # Out-of-mapping source IDs land on ``background``.
    max_src = int(seg.max()) if seg.size else 0
    if mapping:
        max_src = max(max_src, max(int(k) for k in mapping.keys()))

    lut = np.full(max_src + 1, background, dtype=out_dtype)
    for src, dst in mapping.items():
        lut[int(src)] = dst

    return lut[seg]


def paint_union(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Overwrite ``target`` with ``source`` wherever ``source != 0``.

    The "union by paint priority" primitive for cross-task label merging.
    Treats source-background (``0``) as transparent: only non-zero
    source voxels overwrite the target. Same convention used inside
    region-based painting (``_slab_resample_paint``): list order is
    priority, later overwrites earlier.

    The operation is in-place on ``target`` and returns it for chaining.

    Parameters
    ----------
    target :
        Integer label volume. Modified in place.
    source :
        Integer label volume of the same shape. Read-only.

    Returns
    -------
    np.ndarray
        ``target`` (the same object passed in), with ``source``'s
        non-zero voxels written into it.

    Raises
    ------
    ValueError
        On shape mismatch.

    Examples
    --------
    Multi-task union by paint order::

        unified = np.zeros_like(template, dtype=np.uint16)
        for task_seg in segs_in_low_to_high_priority_order:
            paint_union(unified, task_seg)
    """
    if target.shape != source.shape:
        raise ValueError(
            f"paint_union shape mismatch: target {target.shape} "
            f"vs source {source.shape}"
        )
    mask = source != 0
    target[mask] = source[mask]
    return target


def mesh_concat(target: "Mesh", source: "Mesh") -> "Mesh":
    """Concatenate two surface meshes that share a geometry and a schema.

    The mesh analog of :func:`paint_union` — the composite primitive for
    cross-task surface assembly. Both meshes must already be in the same
    *global* label namespace (``boundary_labels`` carry global label values)
    and share the same training-grid geometry; ``mesh_concat`` is then a
    clean ``vstack`` of points plus an index-offset on the source's quads
    so they refer to the concatenated point buffer.

    Unlike ``paint_union``, this is *not* in-place. ``Mesh`` is frozen; a new
    ``Mesh`` is returned. The expected accumulator pattern is::

        mesh = Mesh.empty(geometry, schema)
        for task in tasks:
            prediction = run_inference(task)
            mesh = mesh_concat(mesh, to_mesh(prediction))

    Parameters
    ----------
    target :
        The accumulator mesh. Often ``Mesh.empty(...)`` on the first call.
    source :
        The mesh to append. Its ``quads`` indices are offset by
        ``target.num_points`` so they point into the concatenated points.

    Returns
    -------
    Mesh
        A new mesh with concatenated points, quads, and boundary_labels.

    Raises
    ------
    ValueError
        If geometries differ, schemas differ by identity, or one mesh has
        normals/stencils while the other does not.

    Notes
    -----
    No vertex deduplication is performed at the seam. Two tasks that
    produce surfaces along the same physical region will produce two
    parallel sub-voxel-spaced sheets; deduplication is a downstream concern
    if/when it matters for the use case.
    """
    if target.geometry != source.geometry:
        raise ValueError(
            f"mesh_concat geometry mismatch: target {target.geometry} "
            f"vs source {source.geometry}"
        )
    if target.schema is not source.schema:
        # Identity check: per the multi-task pattern, every per-task mesh is
        # produced with the same global schema object. Distinct schemas mean
        # the caller hasn't unified the label namespace and concatenation
        # would silently mix label IDs.
        raise ValueError(
            "mesh_concat schemas differ by identity; both meshes must carry "
            "the same global LabelSchema. Apply class_map at to_mesh time."
        )
    if target.has_normals != source.has_normals:
        raise ValueError(
            "mesh_concat normals mismatch: one mesh has normals, the other "
            "does not. Pass emit_normals consistently."
        )
    if target.has_stencils != source.has_stencils:
        raise ValueError(
            "mesh_concat stencils mismatch: one mesh has stencils, the other "
            "does not. Pass emit_stencils consistently."
        )

    # Fast path: one side empty → return the other (preserve identity).
    if source.is_empty:
        return target
    if target.is_empty:
        return source

    n_target_pts = target.num_points
    points = np.concatenate([target.points, source.points], axis=0)
    quads = np.concatenate(
        [target.quads, source.quads + np.int32(n_target_pts)], axis=0
    )
    boundary_labels = np.concatenate(
        [target.boundary_labels, source.boundary_labels], axis=0
    )

    normals = None
    if target.has_normals:
        normals = np.concatenate([target.normals, source.normals], axis=0)

    stencils = None
    if target.has_stencils:
        t_off, t_conn = target.stencils
        s_off, s_conn = source.stencils
        # CSR concat: source's offsets shift by len(t_conn); skip source's
        # leading 0 since target's tail value already covers it.
        new_off = np.concatenate([t_off, s_off[1:] + t_off[-1]])
        new_conn = np.concatenate([t_conn, s_conn + np.int32(n_target_pts)])
        stencils = (new_off, new_conn)

    from .values import Mesh
    return Mesh(
        points=points,
        quads=quads,
        boundary_labels=boundary_labels,
        geometry=target.geometry,
        schema=target.schema,
        normals=normals,
        stencils=stencils,
    )


__all__ = [
    "has_regions",
    "regions_class_order",
    "label_dtype",
    "convert_logits_to_segmentation",
    "sigmoid_inplace",
    "remap_labels",
    "paint_union",
    "mesh_concat",
]
