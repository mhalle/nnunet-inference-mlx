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

from dataclasses import dataclass, field
from typing import Mapping


# A weights identifier is either an integer nnU-Net dataset ID (nnU-Net /
# TotalSegmentator convention, resolved by ``cached_engine_from_task``) or
# a string model/folder identifier (MOOSE convention, e.g. "clin_ct_organs"
# or "Dataset123_Organs"). The dispatcher's ``engine_factory`` is what
# turns either form into an engine; the default factory handles only the
# integer form (string resolution is source-aware and provided by the
# caller's factory).
WeightsId = int | str


# ---------------------------------------------------------------------------
# Shape-specific data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeStep:
    """One stage of a ``shape="cascade"`` task.

    Mirrors :class:`workflow.Stage` but stores a *reference* to a model
    instead of a built engine — engines are constructed lazily by the
    dispatcher via the engine cache.

    Exactly one of ``weights_id`` / ``crop_from_task`` is set:

    * ``weights_id`` — run this model directly (the common case).
    * ``crop_from_task`` — run another *registered task* by name as this
      step. This is the "nested-task cascade": the referenced task may
      itself be single or cascade and is expanded inline by the
      dispatcher. Used by TS ``teeth`` (crops from
      ``craniofacial_structures``) and MOOSE's FOV-limited models.

    Parameters
    ----------
    weights_id :
        Weights identifier (see :data:`WeightsId`): an integer nnU-Net
        dataset ID (TS) or a string model/folder identifier (MOOSE).
        Mutually exclusive with ``crop_from_task``.
    crop_from_task :
        Bare name of another registered task to run as this step,
        resolved within the *same source* as the enclosing task.
        Mutually exclusive with ``weights_id``.
    crop_to_classes :
        If set, after this step runs, the foreground bbox of these class
        IDs (in this step's output) is used to crop the *next* step's
        input. ``None`` for the final step of the cascade.
    dilation_mm :
        Safety margin added to the bbox in physical units (mm).
    """

    weights_id: WeightsId | None = None
    crop_from_task: str | None = None
    crop_to_classes: tuple[int, ...] | None = None
    dilation_mm: float = 10.0

    def __post_init__(self) -> None:
        has_wid = self.weights_id is not None
        has_ref = self.crop_from_task is not None
        if has_wid == has_ref:
            raise ValueError(
                "CascadeStep requires exactly one of weights_id / "
                f"crop_from_task (got weights_id={self.weights_id!r}, "
                f"crop_from_task={self.crop_from_task!r})"
            )


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
        Weights identifier (see :data:`WeightsId`): an integer nnU-Net
        dataset ID (TS) or a string model/folder identifier (MOOSE).
    label_remap :
        ``{task_local_id: unified_id}`` — exactly the same shape as the
        argument to :func:`remap_labels`.
    name :
        Human-readable name (``"organs"``, ``"vertebrae"``, …) for logs.
    """

    weights_id: WeightsId
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
        Weights identifier (see :data:`WeightsId`) for ``shape="single"``.
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
    single: WeightsId | None = None
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
        if ":" in self.name:
            raise ValueError(
                f"TaskSpec.name must not contain ':' (reserved as the "
                f"source qualifier separator, e.g. 'ts:total'); got {self.name!r}"
            )
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

    @property
    def qualified_name(self) -> str:
        """The unambiguous ``"source:name"`` registry key for this task.

        Two model systems can ship a task with the same bare ``name``
        (e.g. TS and MOOSE both having ``"total"``); the qualifier keeps
        them distinct. ``run_named_task`` / ``get_task`` accept either the
        bare name (when unambiguous) or this qualified form.
        """
        return f"{self.source}:{self.name}"


# ---------------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------------


def _coerce_weights_id(v: object) -> WeightsId:
    """Normalize a JSON weights-id value to ``int`` or ``str``.

    JSON already distinguishes ``123`` (int) from ``"Dataset123_Organs"``
    (str), so we preserve the incoming type rather than forcing one — an
    integer stays an nnU-Net dataset ID, a string stays a MOOSE
    model/folder identifier. ``bool`` (a subclass of ``int``) is rejected
    to catch malformed input.
    """
    if isinstance(v, bool):
        raise TypeError(f"weights_id must be int or str, got bool: {v!r}")
    if isinstance(v, int):
        return v
    return str(v)


def _taskspec_from_dict(d: Mapping) -> TaskSpec:
    """Construct a TaskSpec from its JSON-friendly dict form.

    JSON only supports string keys; this restores integer keys in
    ``label_remap`` and ``label_map``. Weights identifiers keep their
    JSON type (int dataset ID or str model identifier).
    """
    shape = d["shape"]

    single = _coerce_weights_id(d["single"]) if shape == "single" else None

    cascade: tuple[CascadeStep, ...] | None = None
    if shape == "cascade":
        cascade = tuple(
            CascadeStep(
                weights_id=(
                    _coerce_weights_id(step["weights_id"])
                    if step.get("weights_id") is not None
                    else None
                ),
                crop_from_task=step.get("crop_from_task"),
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
                weights_id=_coerce_weights_id(part["weights_id"]),
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
        cascade_list = []
        for step in spec.cascade:
            step_d: dict = {}
            if step.weights_id is not None:
                step_d["weights_id"] = step.weights_id
            if step.crop_from_task is not None:
                step_d["crop_from_task"] = step.crop_from_task
            step_d["crop_to_classes"] = (
                list(step.crop_to_classes) if step.crop_to_classes else None
            )
            step_d["dilation_mm"] = step.dilation_mm
            cascade_list.append(step_d)
        d["cascade"] = cascade_list
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
# Errors
# ---------------------------------------------------------------------------


class AmbiguousTaskError(LookupError):
    """Raised when a bare task name matches entries from multiple sources.

    The caller must disambiguate with a ``"source:name"`` qualifier (e.g.
    ``"ts:total"``). Reused by :class:`~nnunet_inference_mlx.catalog.TaskCatalog`,
    which is the owned (no-global) replacement for the old module-level registry.
    """


__all__ = [
    "AmbiguousTaskError",
    "CascadeStep",
    "TaskSpec",
    "UnionPart",
    "WeightsId",
]
