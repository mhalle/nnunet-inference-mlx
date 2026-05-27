"""Declarative task registry for nnU-Net-based segmentation pipelines.

A *task* is a named, structured description of a segmentation pipeline:
which model(s), which cascade or union shape, what label remapping, what
modality. The registry is a dict of :class:`TaskSpec` entries — TS tasks
shipped via ``data/ts_tasks.json``, extensible for MOOSE and user-defined
entries via :func:`register_task`.

The dispatcher :func:`run_named_task` walks the registry, loads (cached)
engines, and routes to :func:`predict_with_resampling`,
:func:`run_workflow`, or :func:`run_label_union_workflow` based on the
task's :class:`TaskSpec.shape`.

The schema is informed by both TotalSegmentator's catalog (which has all
three shapes: single, cascade, label-union) and MOOSE's
``moosez/constants.py`` (which contributed modality-as-first-class-field,
body coverage hints, and weight provisioning slots). See
``docs/post-0.8.2-roadmap.md`` for the design analysis.

Schema overview
---------------
.. code-block:: python

    @dataclass(frozen=True)
    class TaskSpec:
        name: str                                # unique key
        source: Literal["ts", "moose", "user"]
        modality: Literal["CT", "MR", "PET"]
        shape: Literal["single", "cascade", "label_union"]

        # Exactly one of these is set, matching shape:
        single: int | None = None                # weights_id
        cascade: tuple[CascadeStep, ...] | None = None
        union: tuple[UnionPart, ...] | None = None

        # Output label naming
        label_map: dict[int, str] = {}

        # Informational (not used at dispatch)
        expected_coverage: str = "any"
        weights_url: str | None = None
        weights_sha256: str | None = None

Storage
-------
The shipped registry lives at ``data/ts_tasks.json`` inside the package
wheel. It's loaded lazily on first use. The JSON file is machine-readable
and machine-generated (see ``scripts/refresh_ts_registry.py``); humans
review diffs in PRs rather than hand-editing.

Users can extend the registry with :func:`register_task` at runtime —
useful for custom-trained models or MOOSE entries before native MOOSE
support lands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    import SimpleITK as sitk

    from .engine import InferenceEngine


# ---------------------------------------------------------------------------
# Shape-specific data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeStep:
    """One stage of a ``shape="cascade"`` task.

    Mirrors :class:`workflow.Stage` but stores ``weights_id`` instead of a
    built engine — engines are constructed lazily by the dispatcher via
    the engine cache.

    Parameters
    ----------
    weights_id :
        nnU-Net dataset ID (integer) used by ``cached_engine_from_task``.
    crop_to_classes :
        If set, after this stage runs, the foreground bbox of these class
        IDs (in this stage's output) is used to crop the *next* stage's
        input. ``None`` for the final stage of the cascade.
    dilation_mm :
        Safety margin added to the bbox in physical units (mm).
    """

    weights_id: int
    crop_to_classes: tuple[int, ...] | None = None
    dilation_mm: float = 10.0


@dataclass(frozen=True)
class UnionPart:
    """One part of a ``shape="label_union"`` task.

    Each part runs independently against the same input volume; its
    task-local labels are remapped into the unified output space via
    ``label_remap``; later parts overwrite earlier parts at overlapping
    voxels (list-order = paint priority).

    Parameters
    ----------
    weights_id :
        nnU-Net dataset ID for this part's model.
    label_remap :
        ``{task_local_id: unified_id}`` — exactly the same shape as the
        argument to :func:`remap_labels`.
    name :
        Human-readable name (``"organs"``, ``"vertebrae"``, …) for logs.
    """

    weights_id: int
    label_remap: Mapping[int, int]
    name: str = ""


# ---------------------------------------------------------------------------
# Main TaskSpec
# ---------------------------------------------------------------------------


_VALID_SHAPES = ("single", "cascade", "label_union")
_VALID_MODALITIES = ("CT", "MR", "PET")
_VALID_SOURCES = ("ts", "moose", "user")


@dataclass(frozen=True)
class TaskSpec:
    """Declarative description of a named segmentation task.

    Exactly one of ``single`` / ``cascade`` / ``union`` is populated,
    matching ``shape``. Validated in ``__post_init__``.

    Parameters
    ----------
    name :
        Unique key in the registry, e.g. ``"total_fast"``, ``"lung_vessels"``.
    source :
        Provenance — ``"ts"``, ``"moose"``, or ``"user"``. Used for
        display + collision resolution when registries are merged.
    modality :
        ``"CT"``, ``"MR"``, or ``"PET"``. Drives expected normalization
        scheme; surfaced to CLI for filtering.
    shape :
        Dispatch shape: ``"single"`` → :func:`predict_with_resampling`;
        ``"cascade"`` → :func:`run_workflow`; ``"label_union"`` →
        :func:`run_label_union_workflow`.
    single :
        Weights ID for ``shape="single"``.
    cascade :
        Tuple of :class:`CascadeStep` (≥ 2) for ``shape="cascade"``.
    union :
        Tuple of :class:`UnionPart` (≥ 1) for ``shape="label_union"``.
    label_map :
        ``{unified_id: human_name}`` for the task's output classes.
        Informational; mostly used by CLI and stats reporting.
    expected_coverage :
        Body region hint: ``"whole_body"`` / ``"trunk"`` / ``"thorax"``
        / ``"abdomen"`` / ``"head_neck"`` / ``"extremity"`` / ``"any"``.
        Informational. Surfaced via ``--list-tasks`` and runtime warnings.
        MOOSE-inspired.
    weights_url, weights_sha256 :
        Slots for future weight-download support (the ``[remote]`` extra,
        roadmap 0.11.0+). Populated when known, ignored at dispatch.
    """

    name: str
    source: str
    modality: str
    shape: str

    # Shape-specific fields (exactly one populated)
    single: int | None = None
    cascade: tuple[CascadeStep, ...] | None = None
    union: tuple[UnionPart, ...] | None = None

    # Informational
    label_map: Mapping[int, str] = field(default_factory=dict)
    expected_coverage: str = "any"
    weights_url: str | None = None
    weights_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TaskSpec.name must be non-empty")
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"TaskSpec.source must be one of {_VALID_SOURCES}, "
                f"got {self.source!r}"
            )
        if self.modality not in _VALID_MODALITIES:
            raise ValueError(
                f"TaskSpec.modality must be one of {_VALID_MODALITIES}, "
                f"got {self.modality!r}"
            )
        if self.shape not in _VALID_SHAPES:
            raise ValueError(
                f"TaskSpec.shape must be one of {_VALID_SHAPES}, "
                f"got {self.shape!r}"
            )

        # Exactly one of the shape-fields is set, and it matches `shape`.
        # `shape` names ↔ field names: label_union ↔ union; the others
        # share the name.
        shape_to_field = {
            "single": "single",
            "cascade": "cascade",
            "label_union": "union",
        }
        populated = {
            "single": self.single is not None,
            "cascade": self.cascade is not None,
            "union": self.union is not None,
        }
        n_set = sum(populated.values())
        if n_set != 1:
            set_names = [k for k, v in populated.items() if v]
            raise ValueError(
                f"Exactly one of (single, cascade, union) must be set; "
                f"got {n_set}: {set_names}"
            )
        expected_field = shape_to_field[self.shape]
        if not populated[expected_field]:
            raise ValueError(
                f"shape={self.shape!r} requires the {expected_field!r} field "
                f"to be populated"
            )

        if self.shape == "cascade":
            assert self.cascade is not None  # for type narrowing
            if len(self.cascade) < 2:
                raise ValueError(
                    f"cascade must have at least 2 steps, got {len(self.cascade)}"
                )
        elif self.shape == "label_union":
            assert self.union is not None
            if len(self.union) < 1:
                raise ValueError(
                    f"label_union must have at least 1 part, got {len(self.union)}"
                )


# ---------------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------------


def _taskspec_from_dict(d: Mapping) -> TaskSpec:
    """Construct a TaskSpec from its JSON-friendly dict form.

    JSON only supports string keys; this restores integer keys in
    ``label_remap`` and ``label_map``.
    """
    shape = d["shape"]

    single = int(d["single"]) if shape == "single" else None

    cascade: tuple[CascadeStep, ...] | None = None
    if shape == "cascade":
        cascade = tuple(
            CascadeStep(
                weights_id=int(step["weights_id"]),
                crop_to_classes=(
                    tuple(int(c) for c in step["crop_to_classes"])
                    if step.get("crop_to_classes")
                    else None
                ),
                dilation_mm=float(step.get("dilation_mm", 10.0)),
            )
            for step in d["cascade"]
        )

    union: tuple[UnionPart, ...] | None = None
    if shape == "label_union":
        union = tuple(
            UnionPart(
                weights_id=int(part["weights_id"]),
                label_remap={int(k): int(v) for k, v in part["label_remap"].items()},
                name=part.get("name", ""),
            )
            for part in d["union"]
        )

    label_map = {int(k): str(v) for k, v in d.get("label_map", {}).items()}

    return TaskSpec(
        name=d["name"],
        source=d["source"],
        modality=d["modality"],
        shape=shape,
        single=single,
        cascade=cascade,
        union=union,
        label_map=label_map,
        expected_coverage=d.get("expected_coverage", "any"),
        weights_url=d.get("weights_url"),
        weights_sha256=d.get("weights_sha256"),
    )


def _taskspec_to_dict(spec: TaskSpec) -> dict:
    """Serialize a TaskSpec to its JSON-friendly dict form.

    Inverse of :func:`_taskspec_from_dict`. Omits default-valued
    informational fields for a cleaner on-disk artifact.
    """
    d: dict = {
        "name": spec.name,
        "source": spec.source,
        "modality": spec.modality,
        "shape": spec.shape,
    }
    if spec.shape == "single":
        d["single"] = spec.single
    elif spec.shape == "cascade":
        assert spec.cascade is not None
        d["cascade"] = [
            {
                "weights_id": step.weights_id,
                "crop_to_classes": (
                    list(step.crop_to_classes) if step.crop_to_classes else None
                ),
                "dilation_mm": step.dilation_mm,
            }
            for step in spec.cascade
        ]
    elif spec.shape == "label_union":
        assert spec.union is not None
        d["union"] = [
            {
                "weights_id": part.weights_id,
                "label_remap": {str(k): v for k, v in part.label_remap.items()},
                "name": part.name,
            }
            for part in spec.union
        ]

    if spec.label_map:
        d["label_map"] = {str(k): v for k, v in spec.label_map.items()}
    if spec.expected_coverage != "any":
        d["expected_coverage"] = spec.expected_coverage
    if spec.weights_url is not None:
        d["weights_url"] = spec.weights_url
    if spec.weights_sha256 is not None:
        d["weights_sha256"] = spec.weights_sha256
    return d


# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, TaskSpec] = {}
_BUILTIN_LOADED = False


def _builtin_registry_path() -> Path:
    """Path to the shipped TS registry JSON inside the package."""
    return Path(__file__).parent / "data" / "ts_tasks.json"


def _load_builtin_registry() -> None:
    """Load the shipped ``ts_tasks.json`` (lazy, one-shot).

    Called automatically on first registry access. Idempotent — only the
    first call does I/O.
    """
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    _BUILTIN_LOADED = True
    path = _builtin_registry_path()
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    for entry in payload.get("tasks", []):
        spec = _taskspec_from_dict(entry)
        _REGISTRY[spec.name] = spec


def register_task(spec: TaskSpec, *, overwrite: bool = False) -> None:
    """Register a :class:`TaskSpec` under ``spec.name``.

    Raises ``ValueError`` if a task is already registered under the same
    name unless ``overwrite=True``. Use this for user-defined tasks or to
    add MOOSE entries before the native MOOSE-compat release (0.9.2).
    """
    _load_builtin_registry()
    if spec.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"task {spec.name!r} already registered "
            f"(source={_REGISTRY[spec.name].source!r}); "
            f"pass overwrite=True to replace."
        )
    _REGISTRY[spec.name] = spec


def unregister_task(name: str) -> None:
    """Remove a task from the registry. Raises ``KeyError`` if absent.

    Useful in tests and when temporarily overriding a builtin task.
    """
    _load_builtin_registry()
    if name not in _REGISTRY:
        raise KeyError(f"unknown task: {name!r}")
    del _REGISTRY[name]


def get_task(name: str) -> TaskSpec:
    """Look up a :class:`TaskSpec` by name.

    Raises ``KeyError`` with a list of available task names if not found.
    """
    _load_builtin_registry()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown task: {name!r}. "
            f"Available: {sorted(_REGISTRY) or '(empty)'}"
        )
    return _REGISTRY[name]


def list_registered_tasks() -> list[str]:
    """Return the sorted list of registered task names."""
    _load_builtin_registry()
    return sorted(_REGISTRY)


def list_tasks_by_modality(modality: str) -> list[str]:
    """Return registered task names matching ``modality`` (``CT`` / ``MR`` / ``PET``)."""
    _load_builtin_registry()
    return sorted(
        name for name, spec in _REGISTRY.items() if spec.modality == modality
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run_named_task(
    name: str,
    image_sitk: "sitk.Image",
    *,
    folds: int | Iterable[int] | str | None = None,
    reorient_to: str | None = "LPS",
    peak_working_memory_mb: int | None = None,
    verbose: bool = False,
    engine_factory: "Callable[[int], InferenceEngine] | None" = None,
) -> "sitk.Image":
    """Run a named segmentation task end-to-end on a SITK image.

    Looks up ``name`` in the registry, builds the necessary engines via
    the engine cache, and dispatches to the appropriate backend:

    * ``shape="single"`` → :func:`predict_with_resampling`
    * ``shape="cascade"`` → :func:`run_workflow`
    * ``shape="label_union"`` → :func:`run_label_union_workflow`

    Engines are cached across calls — the second call with the same
    task name doesn't reload weights.

    Parameters
    ----------
    name :
        Registered task name. See :func:`list_registered_tasks`.
    image_sitk :
        Input SITK image.
    folds :
        Forwarded to ``cached_engine_from_task`` for each underlying
        weights ID. ``None`` (default) uses the layout's default fold(s).
    reorient_to :
        Canonical orientation target. ``"LPS"`` (default) for nnU-Net /
        TS / MOOSE. Pass ``None`` to skip the reorient round-trip.
    peak_working_memory_mb :
        Inverse-resample memory budget passed to each stage. ``None``
        (default) auto-tiers from system RAM.
    verbose :
        Print per-stage progress.
    engine_factory :
        Override engine construction. Signature: ``(weights_id: int)
        -> InferenceEngine``. Used by tests with synthetic engines and
        by callers with non-standard weight locations. ``None``
        (default) uses :func:`cached_engine_from_task`.

    Returns
    -------
    sitk.Image
        Segmentation in the original input's geometry.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    from .resampling import predict_with_resampling
    from .workflow import (
        ParallelStage, Stage, run_label_union_workflow, run_workflow,
    )

    spec = get_task(name)

    if engine_factory is None:
        from .engine_cache import cached_engine_from_task

        def engine_factory(wid: int):  # type: ignore[misc]
            return cached_engine_from_task(wid, folds=folds, verbose=verbose)

    if verbose:
        print(
            f"[run_named_task] {name} "
            f"(source={spec.source}, modality={spec.modality}, "
            f"shape={spec.shape})"
        )

    if spec.shape == "single":
        assert spec.single is not None
        engine = engine_factory(spec.single)
        return predict_with_resampling(
            engine, image_sitk,
            reorient_to=reorient_to,
            peak_working_memory_mb=peak_working_memory_mb,
        )

    if spec.shape == "cascade":
        assert spec.cascade is not None
        stages = [
            Stage(
                engine=engine_factory(step.weights_id),
                crop_to_classes=step.crop_to_classes,
                dilation_mm=step.dilation_mm,
                peak_working_memory_mb=peak_working_memory_mb,
            )
            for step in spec.cascade
        ]
        return run_workflow(image_sitk, stages, verbose=verbose)

    if spec.shape == "label_union":
        assert spec.union is not None
        union_stages = [
            ParallelStage(
                engine=engine_factory(part.weights_id),
                label_remap=dict(part.label_remap),
                part_name=part.name,
            )
            for part in spec.union
        ]
        return run_label_union_workflow(
            image_sitk, union_stages,
            reorient_to=reorient_to,
            peak_working_memory_mb=peak_working_memory_mb,
            verbose=verbose,
        )

    raise ValueError(f"unhandled shape: {spec.shape!r}")


__all__ = [
    "CascadeStep",
    "TaskSpec",
    "UnionPart",
    "get_task",
    "list_registered_tasks",
    "list_tasks_by_modality",
    "register_task",
    "run_named_task",
    "unregister_task",
]
