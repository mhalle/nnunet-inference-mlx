"""segment — the named-task / pipeline dispatcher on the new (Volume + store) path.

``segment(task, image, store=…)`` resolves a task to its recipe and dispatches
by shape, returning a :class:`Segmentation` in the input's geometry. It is the
top-level verb for "produce a segmentation"; ``LoadedModel.segment`` is the
same verb for a single already-loaded model.

* **single**   → ``store.load(id).segment(image)`` (the native path)
* **cascade**  → coarse → crop FOV → fine → paste
* **label_union** → run parts → remap → paint by priority

All three shapes are expressed over the toolkit stages — ``LoadedModel.segment``
(itself ``to_model_frame → sliding_window → restore``), the Volume-native
``geometry`` ops (``bbox_of_labels``/``crop``/``paste``), and the ``labels``
primitives (``remap_labels``/``paint_union``). No bridge to the old SITK
``run_workflow``/``run_label_union_workflow``.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np

from .tasks import TaskSpec
from .values import LabelSchema, Segmentation, Volume

if TYPE_CHECKING:
    from .catalog import TaskCatalog
    from .store import ModelStore


def segment(
    task: str | TaskSpec,
    image: Volume,
    *,
    store: "ModelStore",
    catalog: "TaskCatalog | None" = None,
    reorient_to: str | None = "RAS",
    peak_working_memory_mb: int | None = None,
    output_spacing: "float | tuple[float, float, float] | None" = None,
    output_scaling: float | None = None,
    at_model_spacing: bool = False,
    output_interpolation: str = "linear",
    progress: "Callable[[str], None] | None" = None,
) -> Segmentation:
    """Segment a :class:`Volume` with a named task (or recipe) → :class:`Segmentation`.

    ``task`` may be a recipe (:class:`TaskSpec`) or a name resolved via
    ``catalog`` (default: a catalog for the store's ecosystem). Models are
    fetched/built/cached through ``store``.

    Output-resolution knobs (``output_spacing`` / ``output_scaling`` /
    ``at_model_spacing``, mutually exclusive) are currently supported for
    ``single`` tasks only; on cascade/union they raise (the output is assembled
    from integer label maps, so high-quality logit-render at a new grid needs
    more plumbing — tracked for a later step).

    ``progress`` is an optional callback invoked with short human-readable phase
    strings (e.g. ``"Predicting..."``, ``"Predicting part 2 of 5 ..."``) — used
    by CLIs to report progress without the toolkit owning any console output.
    """
    spec = task if isinstance(task, TaskSpec) else _resolve(task, catalog, store)
    _resample = output_spacing is not None or output_scaling is not None or at_model_spacing

    if spec.shape == "single":
        return _segment_single(spec, image, store,
                              reorient_to=reorient_to,
                              peak_working_memory_mb=peak_working_memory_mb,
                              output_spacing=output_spacing,
                              output_scaling=output_scaling,
                              at_model_spacing=at_model_spacing,
                              output_interpolation=output_interpolation,
                              progress=progress)
    if _resample:
        raise NotImplementedError(
            f"output resampling (output_spacing/output_scaling/at_model_spacing) is not "
            f"yet supported for {spec.shape!r} tasks — only single-model tasks."
        )
    if spec.shape == "cascade":
        return _segment_cascade(spec, image, store, catalog,
                              reorient_to=reorient_to,
                              peak_working_memory_mb=peak_working_memory_mb,
                              output_interpolation=output_interpolation,
                              progress=progress)
    if spec.shape == "label_union":
        return _segment_union(spec, image, store,
                            reorient_to=reorient_to,
                            peak_working_memory_mb=peak_working_memory_mb,
                            output_interpolation=output_interpolation,
                            progress=progress)
    raise ValueError(f"unhandled task shape: {spec.shape!r}")


# ---------------------------------------------------------------------------
# resolution / helpers
# ---------------------------------------------------------------------------


def _resolve(name: str, catalog, store) -> TaskSpec:
    if catalog is None:
        from .catalog import TaskCatalog
        catalog = TaskCatalog(store.ecosystem)
    return catalog.get(name)


def _schema(spec: TaskSpec) -> LabelSchema:
    return LabelSchema(names={int(k): str(v) for k, v in spec.label_map.items()})


def required_weights_ids(spec: TaskSpec, *, store=None, catalog=None) -> list:
    """The weights ids a task needs — single id, cascade stages, or union parts.

    For pre-download (e.g. CLI auto-download): ``store.download(required_weights_ids(spec))``.
    """
    if spec.shape == "single":
        return [spec.single]
    if spec.shape == "cascade":
        return [wid for (wid, _crop, _dil) in _flatten_cascade(spec, catalog, store)]
    if spec.shape == "label_union":
        return [p.weights_id for p in spec.union]
    return []


def _flatten_cascade(spec: TaskSpec, catalog, store, _depth: int = 0):
    """Flatten a (possibly nested-task) cascade to ``[(id, crop, dilation)]``."""
    if _depth > 8:
        raise ValueError(f"cascade nesting too deep at {spec.qualified_name!r}")
    if spec.shape == "single":
        return [(spec.single, None, 10.0)]
    if spec.shape != "cascade":
        raise ValueError(f"{spec.qualified_name!r} (shape={spec.shape!r}) is not a crop source")
    out = []
    for step in spec.cascade:
        if step.crop_from_task is not None:
            if catalog is None:
                from .catalog import TaskCatalog
                catalog = TaskCatalog(store.ecosystem)
            ref = catalog.get(f"{spec.source}:{step.crop_from_task}")
            sub = _flatten_cascade(ref, catalog, store, _depth + 1)
            wid, _, _ = sub[-1]
            sub[-1] = (wid, step.crop_to_classes, step.dilation_mm)
            out.extend(sub)
        else:
            out.append((step.weights_id, step.crop_to_classes, step.dilation_mm))
    return out


# ---------------------------------------------------------------------------
# per-shape runners
# ---------------------------------------------------------------------------


def _segment_single(spec, image, store, *, reorient_to, peak_working_memory_mb,
                    output_spacing=None, output_scaling=None, at_model_spacing=False,
                    output_interpolation="linear", progress=None) -> Segmentation:
    model = store.load(spec.single)
    if progress:
        progress("Predicting...")
    return model.segment(image, reorient_to=reorient_to,
                         peak_working_memory_mb=peak_working_memory_mb,
                         output_spacing=output_spacing,
                         output_scaling=output_scaling,
                         at_model_spacing=at_model_spacing,
                         output_interpolation=output_interpolation)


def _segment_cascade(spec, image, store, catalog, *, reorient_to,
                     peak_working_memory_mb, output_interpolation="linear",
                     progress=None) -> Segmentation:
    """coarse → crop FOV around target classes → fine → paste into full grid.

    Each stage runs the proven single-model pipeline (``model.segment``, which
    reorients to canonical internally), so its output is in the *current*
    input's grid. The foreground bbox of ``crop_to_classes`` in that output
    crops the next stage's input; the final stage's output is pasted back into
    the original grid. Crop/paste stay in the caller's orientation throughout.
    """
    from .geometry import Box, bbox_of_labels, crop, paste

    descriptors = _flatten_cascade(spec, catalog, store)
    schema = _schema(spec)

    current = image
    cumulative = Box.full(image.geometry.shape_zyx)
    final_seg: Segmentation | None = None

    for i, (wid, crop_classes, dilation) in enumerate(descriptors):
        model = store.load(wid)
        if progress:
            progress(f"Predicting stage {i + 1} of {len(descriptors)} ...")
        seg = model.segment(current, reorient_to=reorient_to,
                            peak_working_memory_mb=peak_working_memory_mb,
                            output_interpolation=output_interpolation)
        if i == len(descriptors) - 1:
            final_seg = seg
            break
        if crop_classes is None:
            continue
        box = bbox_of_labels(seg, classes=tuple(crop_classes), dilation_mm=dilation)
        if box is None:
            continue   # target class absent — leave FOV unchanged
        current = crop(current, box)
        cumulative = cumulative.compose(box)

    assert final_seg is not None
    # Re-label the final output with the cascade's unified schema, then place
    # it back in the original grid if any stage cropped.
    final_seg = Segmentation(data=final_seg.data, geometry=final_seg.geometry,
                             schema=schema)
    if cumulative.shape_zyx == image.geometry.shape_zyx:
        return final_seg   # no crop happened — already in the original grid
    return paste(final_seg, image.geometry, cumulative)


def _segment_union(spec, image, store, *, reorient_to, peak_working_memory_mb,
                   output_interpolation="linear", progress=None) -> Segmentation:
    """Run each part independently, remap into the unified space, paint by priority.

    Parts share the same input; later parts overwrite earlier ones at
    overlapping voxels (``paint_union``). Each part's ``model.segment`` returns
    a segmentation in the input's grid, so the remapped arrays line up for the
    paint without any further resampling.
    """
    import mlx.core as mx

    from .labels import paint_union, remap_labels

    schema = _schema(spec)

    # Unified dtype: smallest uint fitting every remap target across parts.
    max_target = 0
    for part in spec.union:
        if part.label_remap:
            max_target = max(max_target, max(int(v) for v in part.label_remap.values()))
    if max_target <= np.iinfo(np.uint8).max:
        out_dtype = np.dtype(np.uint8)
    elif max_target <= np.iinfo(np.uint16).max:
        out_dtype = np.dtype(np.uint16)
    else:
        out_dtype = np.dtype(np.uint32)

    unified = np.zeros(image.geometry.shape_zyx, dtype=out_dtype)
    for i, part in enumerate(spec.union):
        model = store.load(part.weights_id)
        if progress:
            progress(f"Predicting part {i + 1} of {len(spec.union)} ...")
        seg = model.segment(image, reorient_to=reorient_to,
                           peak_working_memory_mb=peak_working_memory_mb,
                           output_interpolation=output_interpolation)
        remapped = remap_labels(np.asarray(seg.data), dict(part.label_remap),
                                out_dtype=out_dtype)
        paint_union(unified, remapped)

    return Segmentation(data=mx.array(unified), geometry=image.geometry, schema=schema)


__all__ = ["segment", "required_weights_ids"]
