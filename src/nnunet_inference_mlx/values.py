"""Core frozen value types for the toolkit API.

These are the *currency* every stage passes around — geometry, images,
segmentations, probabilities, the inverse-transform recipe, the label
schema, and the build options. They are pure data: no GPU allocation, no
IO, no hidden state. A stage takes one and returns a new one; nothing is
mutated in place.

Conventions
-----------
* Array axis order is **(Z, Y, X)** to match the rest of the package.
* Images carry an explicit trailing **channel** axis: ``(Z, Y, X, C)``
  (channels-last — the MLX port is channels-last end-to-end). Single-channel
  data is ``C == 1``, not a missing axis.
* Geometry is recorded in SITK conventions (origin/direction in world XYZ,
  spacing mirrored to ZYX array order) so a round-trip to/from a SITK image
  is lossless. SITK-dependent construction lives in the IO layer, not here —
  these types stay backend-free.

Array-carrying types use ``eq=False``: MLX arrays don't define scalar
equality, and we never compare images by value. ``EngineOptions`` and
``Geometry`` hold only scalars/tuples, so they are fully frozen-hashable and
usable as cache keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping, Sequence

import mlx.core as mx


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Voxel-grid placement in physical (patient) space.

    Hashable: holds only tuples. Two geometries are equal iff every field
    matches — used to assert that an inverse-resample lands back on the
    caller's grid.

    Parameters
    ----------
    spacing_zyx :
        mm per voxel, in array (Z, Y, X) order.
    shape_zyx :
        voxel counts, (Z, Y, X).
    origin_xyz :
        world coordinate of voxel (0, 0, 0), SITK XYZ order.
    direction_xyz :
        9-tuple row-major direction cosines, SITK XYZ axes. Identity is
        ``(1, 0, 0, 0, 1, 0, 0, 0, 1)``.
    """

    spacing_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction_xyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.spacing_zyx) != 3:
            raise ValueError(f"spacing_zyx must have 3 elements, got {self.spacing_zyx!r}")
        if len(self.shape_zyx) != 3:
            raise ValueError(f"shape_zyx must have 3 elements, got {self.shape_zyx!r}")
        if len(self.direction_xyz) != 9:
            raise ValueError(
                f"direction_xyz must have 9 elements, got {len(self.direction_xyz)}"
            )
        # Normalize to plain tuples of the right scalar type.
        object.__setattr__(self, "spacing_zyx", tuple(float(s) for s in self.spacing_zyx))
        object.__setattr__(self, "shape_zyx", tuple(int(s) for s in self.shape_zyx))
        object.__setattr__(self, "origin_xyz", tuple(float(o) for o in self.origin_xyz))
        object.__setattr__(self, "direction_xyz", tuple(float(d) for d in self.direction_xyz))

    @property
    def is_axis_aligned(self) -> bool:
        """True if the direction matrix is the identity (no oblique axes)."""
        return self.direction_xyz == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    @property
    def physical_size_zyx(self) -> tuple[float, float, float]:
        """Extent in mm along each array axis (spacing * shape)."""
        return tuple(s * n for s, n in zip(self.spacing_zyx, self.shape_zyx))

    def with_spacing(self, spacing_zyx: Sequence[float]) -> "Geometry":
        """A copy at a different voxel spacing (shape unchanged)."""
        return replace(self, spacing_zyx=tuple(float(s) for s in spacing_zyx))

    def with_shape(self, shape_zyx: Sequence[int]) -> "Geometry":
        """A copy at a different shape (spacing/origin/direction unchanged)."""
        return replace(self, shape_zyx=tuple(int(s) for s in shape_zyx))


# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A region-model output: one sigmoid head whose positive voxels paint
    ``label_value``, semantically the union of ``member_classes``."""

    label_value: int
    member_classes: tuple[int, ...] = ()


@dataclass(frozen=True, eq=False)
class LabelSchema:
    """Integer label ↔ name, plus region semantics if any.

    A *standard* schema has empty ``regions`` and is converted by argmax. A
    *region* schema (BraTS-style) has one sigmoid head per region and is
    converted by threshold + paint in ``paint_priority`` order.
    """

    names: Mapping[int, str]
    regions: tuple[Region, ...] = ()
    paint_priority: tuple[int, ...] = ()

    @property
    def is_region_model(self) -> bool:
        return bool(self.regions)

    @property
    def num_outputs(self) -> int:
        """Channel count of the network head (K)."""
        if self.is_region_model:
            return len(self.regions)
        return len(self.names)

    def name_of(self, value: int) -> str:
        return self.names.get(int(value), f"label_{int(value)}")

    @staticmethod
    def from_dataset_json(dataset_json: Mapping) -> "LabelSchema":
        """Build a schema from an nnU-Net ``dataset.json`` ``labels`` block.

        Standard: ``{"background": 0, "liver": 1, ...}`` → name↔id.
        Region:   values that are lists of base classes, plus a top-level
        ``regions_class_order`` giving the paint order.
        """
        labels = dataset_json.get("labels", {})
        is_region = any(isinstance(v, (list, tuple)) for v in labels.values())

        if not is_region:
            names = {int(v): str(k) for k, v in labels.items()}
            return LabelSchema(names=names)

        order = dataset_json.get("regions_class_order")
        if order is None:
            raise ValueError(
                "region-based dataset.json has no regions_class_order; cannot "
                "determine paint priority."
            )
        paint_priority = tuple(int(c) for c in order)
        regions = tuple(
            Region(
                label_value=int(v[0]) if isinstance(v, (list, tuple)) and v else int(v),
                member_classes=tuple(int(x) for x in v) if isinstance(v, (list, tuple)) else (int(v),),
            )
            for v in labels.values()
            if isinstance(v, (list, tuple))
        )
        # Names for the painted label values (best-effort from the keys).
        names = {}
        for k, v in labels.items():
            if isinstance(v, (list, tuple)) and v:
                names[int(v[0])] = str(k)
            elif isinstance(v, int):
                names[int(v)] = str(k)
        return LabelSchema(names=names, regions=regions, paint_priority=paint_priority)


# ---------------------------------------------------------------------------
# Image / segmentation / probability volumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Volume:
    """A 3-D image: channels-last array + geometry + channel identity.

    Pure in-memory value. ``data`` is ``(Z, Y, X, C)`` float32.
    """

    data: mx.array
    geometry: Geometry
    channels: tuple[str, ...] = ("CT",)

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(
                f"Volume.data must be 4-D (Z, Y, X, C); got ndim={self.data.ndim}"
            )
        z, y, x, c = self.data.shape
        if (int(z), int(y), int(x)) != self.geometry.shape_zyx:
            raise ValueError(
                f"Volume data spatial shape {(z, y, x)} != geometry.shape_zyx "
                f"{self.geometry.shape_zyx}"
            )
        if len(self.channels) != int(c):
            raise ValueError(
                f"Volume has {c} channels but {len(self.channels)} channel names "
                f"{self.channels}"
            )

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def is_multichannel(self) -> bool:
        return self.num_channels > 1

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.geometry.shape_zyx

    def select_channels(self, names: Sequence[str]) -> "Volume":
        """A view with only the named channels, in the given order."""
        idx = []
        for n in names:
            if n not in self.channels:
                raise KeyError(f"channel {n!r} not in {self.channels}")
            idx.append(self.channels.index(n))
        return Volume(
            data=self.data[..., idx],
            geometry=self.geometry,
            channels=tuple(names),
        )

    def with_data(self, data: mx.array, *, geometry: Geometry | None = None,
                  channels: Sequence[str] | None = None) -> "Volume":
        """A new Volume with replaced array (and optionally geometry/channels).

        The vehicle for pure preprocess steps: resample/normalize return a new
        Volume rather than mutating.
        """
        return Volume(
            data=data,
            geometry=geometry if geometry is not None else self.geometry,
            channels=tuple(channels) if channels is not None else self.channels,
        )


@dataclass(frozen=True, eq=False)
class Segmentation:
    """An integer label map in a geometry, with the schema that names it.

    ``data`` is ``(Z, Y, X)`` integer.
    """

    data: mx.array
    geometry: Geometry
    schema: LabelSchema

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(
                f"Segmentation.data must be 3-D (Z, Y, X); got ndim={self.data.ndim}"
            )
        if tuple(int(s) for s in self.data.shape) != self.geometry.shape_zyx:
            raise ValueError(
                f"Segmentation shape {tuple(self.data.shape)} != geometry.shape_zyx "
                f"{self.geometry.shape_zyx}"
            )

    def with_data(self, data: mx.array, *, geometry: Geometry | None = None) -> "Segmentation":
        return Segmentation(
            data=data,
            geometry=geometry if geometry is not None else self.geometry,
            schema=self.schema,
        )


@dataclass(frozen=True, eq=False)
class Probabilities:
    """Per-class continuous output at a geometry.

    ``data`` is ``(K, Z, Y, X)`` float32. ``activation`` records what the
    values are so downstream code doesn't have to guess.
    """

    data: mx.array
    geometry: Geometry
    schema: LabelSchema
    activation: Literal["logits", "softmax", "sigmoid"] = "logits"

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(
                f"Probabilities.data must be 4-D (K, Z, Y, X); got ndim={self.data.ndim}"
            )
        k, z, y, x = self.data.shape
        if (int(z), int(y), int(x)) != self.geometry.shape_zyx:
            raise ValueError(
                f"Probabilities spatial shape {(z, y, x)} != geometry.shape_zyx "
                f"{self.geometry.shape_zyx}"
            )

    @property
    def num_classes(self) -> int:
        return int(self.data.shape[0])


# ---------------------------------------------------------------------------
# RestorePlan — the inverse-transform recipe (returned, never hidden)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestorePlan:
    """Everything needed to map a model-frame result back to the caller's grid.

    Returned by the preprocess step that moved a Volume into model frame
    (reorient → permute → resample). ``postprocess.restore`` consumes it.
    This is how the pipeline avoids hidden state: the inverse instructions are
    a value the caller holds, not state stashed on an object or a temp file.

    ``source_geometry`` doubles as a binding token — restore checks that the
    result it's handed actually came from this plan's input grid.
    """

    source_geometry: Geometry
    source_orientation: str
    target_orientation: str
    axis_permutation: tuple[int, int, int]
    model_spacing_zyx: tuple[float, float, float]


# ---------------------------------------------------------------------------
# BuildOptions — what determines the built model (frozen, hashable → cache key)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildOptions:
    """Knobs that determine *what model gets built* — and thus its cache identity.

    Hashable so it keys the model store's loaded-model cache. Holds only the
    build-identity fields: a different value here means a genuinely different
    compiled model. Deliberately **excludes** run-behavior knobs (``step_size``,
    ``use_mirroring``) — those change how you *run* a model, not what's built,
    so they're per-call arguments to ``segment`` / the infer step, not part of
    the cache key. (Also excludes ``verbose`` / ``progress`` — display flags.)
    """

    configuration: str | None = None
    folds: tuple[int, ...] | Literal["all"] = "all"
    batch_size: int | None = None
    compile: bool = True
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.folds != "all":
            object.__setattr__(self, "folds", tuple(int(f) for f in self.folds))


__all__ = [
    "Geometry",
    "Region",
    "LabelSchema",
    "Volume",
    "Segmentation",
    "Probabilities",
    "RestorePlan",
    "BuildOptions",
]
