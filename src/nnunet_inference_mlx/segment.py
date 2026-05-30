"""segment — the named-task / pipeline dispatcher on the new (Volume + store) path.

``segment(task, image, store=…)`` resolves a task to its recipe and dispatches
by shape, returning a :class:`Segmentation` in the input's geometry. It is the
top-level verb for "produce a segmentation"; ``LoadedModel.segment`` is the
same verb for a single already-loaded model.

* **single**   → ``store.load(id).segment(image)`` (the native path)
* **cascade**  → coarse → crop FOV → fine → paste
* **label_union** → run parts → remap → paint by priority

The recipe layer (``TaskSpec``/``TaskCatalog``) and the cascade/union
*orchestration* (``run_workflow`` / ``run_label_union_workflow``) are reused
from the proven implementation during migration — bridged here via the store's
loaded models and ``Volume``↔SITK conversion. At cutover the orchestration is
re-expressed over the decomposed ``Volume``-native stage namespaces (phase 3b)
and the old workflow module is folded in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    reorient_to: str | None = "LPS",
    peak_working_memory_mb: int | None = None,
) -> Segmentation:
    """Segment a :class:`Volume` with a named task (or recipe) → :class:`Segmentation`.

    ``task`` may be a recipe (:class:`TaskSpec`) or a name resolved via
    ``catalog`` (default: a catalog for the store's ecosystem). Models are
    fetched/built/cached through ``store``.
    """
    spec = task if isinstance(task, TaskSpec) else _resolve(task, catalog, store)

    if spec.shape == "single":
        return _segment_single(spec, image, store,
                              reorient_to=reorient_to,
                              peak_working_memory_mb=peak_working_memory_mb)
    if spec.shape == "cascade":
        return _segment_cascade(spec, image, store, catalog,
                              peak_working_memory_mb=peak_working_memory_mb)
    if spec.shape == "label_union":
        return _segment_union(spec, image, store,
                            reorient_to=reorient_to,
                            peak_working_memory_mb=peak_working_memory_mb)
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


def _segment_single(spec, image, store, *, reorient_to, peak_working_memory_mb) -> Segmentation:
    model = store.load(spec.single)
    return model.segment(image, reorient_to=reorient_to,
                         peak_working_memory_mb=peak_working_memory_mb)


def _segment_cascade(spec, image, store, catalog, *, peak_working_memory_mb) -> Segmentation:
    from .imageio import sitk_to_segmentation, volume_to_sitk
    from .workflow import Stage, run_workflow

    descriptors = _flatten_cascade(spec, catalog, store)
    stages = [
        Stage(
            engine=store.load(wid)._engine,
            crop_to_classes=crop,
            dilation_mm=dil,
            peak_working_memory_mb=peak_working_memory_mb,
        )
        for (wid, crop, dil) in descriptors
    ]
    seg_sitk = run_workflow(volume_to_sitk(image), stages)
    return sitk_to_segmentation(seg_sitk, _schema(spec))


def _segment_union(spec, image, store, *, reorient_to, peak_working_memory_mb) -> Segmentation:
    from .imageio import sitk_to_segmentation, volume_to_sitk
    from .workflow import ParallelStage, run_label_union_workflow

    stages = [
        ParallelStage(
            engine=store.load(part.weights_id)._engine,
            label_remap=dict(part.label_remap),
            part_name=part.name,
        )
        for part in spec.union
    ]
    seg_sitk = run_label_union_workflow(
        volume_to_sitk(image), stages,
        reorient_to=reorient_to, peak_working_memory_mb=peak_working_memory_mb,
    )
    return sitk_to_segmentation(seg_sitk, _schema(spec))


__all__ = ["segment"]
