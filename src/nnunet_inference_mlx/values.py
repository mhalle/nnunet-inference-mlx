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
equality, and we never compare images by value. ``BuildOptions`` and
``Geometry`` hold only scalars/tuples, so they are fully frozen-hashable and
usable as cache keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping, Sequence

import mlx.core as mx
import numpy as np


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
class Prediction:
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
                f"Prediction.data must be 4-D (K, Z, Y, X); got ndim={self.data.ndim}"
            )
        k, z, y, x = self.data.shape
        if (int(z), int(y), int(x)) != self.geometry.shape_zyx:
            raise ValueError(
                f"Prediction spatial shape {(z, y, x)} != geometry.shape_zyx "
                f"{self.geometry.shape_zyx}"
            )

    @property
    def num_classes(self) -> int:
        return int(self.data.shape[0])


# ---------------------------------------------------------------------------
# Mesh — surface geometry extracted from a Prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Mesh:
    """A multi-material surface mesh: vertices, quad faces, per-face label pairs.

    Sibling of :class:`Segmentation` — both are produced from a
    :class:`Prediction`; ``Segmentation`` is a voxel labelmap, ``Mesh`` is the
    surface separating the labels. The surface is the SurfaceNets dual: one
    vertex per boundary cell, one quad per crossed cell-edge.

    Conventions
    -----------
    * ``points`` live in the *training-grid index coordinate system* of
      ``geometry`` — fractional indices, sub-voxel precise. World-mm
      conversion is ``apply_geometry(points, geometry)``; kept in grid
      coords so cross-task ``mesh_concat`` is a clean vstack without affine
      reconciliation (every task shares the same grid for its source).
    * Component order is **(Z, Y, X)** to match the rest of the toolkit —
      ``points[i] = (Z_i, Y_i, X_i)``. This is a left-handed permutation
      of standard ``(X, Y, Z)``; downstream consumers expecting Cartesian
      conventions (VTK, glTF, PLY, ``np.cross`` for geometric normals)
      must swap to ``(X, Y, Z)`` first. The ``meshio`` exporters do this
      transparently at write time.
    * ``boundary_labels`` per face follows the vtkSurfaceNets3D convention:
      a 2-component (Label0, Label1) tuple where the quad normal points
      Label0 → Label1 *after* the ``(Z, Y, X) → (X, Y, Z)`` swap. If
      background (label 0) is involved it goes in slot 1; otherwise the
      pair is sorted ascending.
    * Quads, not triangles. With smoothing disabled (the smooth-field case)
      faces are planar and quads are the natural primary form. Callers that
      need triangles can ``triangulate()`` at the boundary.

    All arrays are numpy on host — surface meshes are sparse, downstream
    consumers (VTK, Slicer, glTF) live host-side.

    Parameters
    ----------
    points :
        ``(N, 3) float32`` vertex positions in training-grid index coords.
    quads :
        ``(M, 4) int32`` face vertex indices into ``points``.
    boundary_labels :
        ``(M, 2) int32`` label pair per face, VTK convention.
    geometry :
        The grid the points are indexed against. Carries world-mm placement.
    schema :
        Label name lookup. For multi-task meshes this is the *global* schema;
        ``boundary_labels`` are in the same global namespace.
    normals :
        Optional ``(N, 3) float32`` per-vertex normals. ``None`` unless the
        caller requested field-gradient normals at extraction time.
    stencils :
        Optional ``(offsets, connectivity)`` CSR pair — per-vertex adjacency
        for downstream constrained smoothing. ``None`` unless requested.
    """

    points: np.ndarray
    quads: np.ndarray
    boundary_labels: np.ndarray
    geometry: Geometry
    schema: LabelSchema
    normals: np.ndarray | None = None
    stencils: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"Mesh.points must be (N, 3); got shape {self.points.shape}"
            )
        if self.quads.ndim != 2 or self.quads.shape[1] != 4:
            raise ValueError(
                f"Mesh.quads must be (M, 4); got shape {self.quads.shape}"
            )
        if self.boundary_labels.ndim != 2 or self.boundary_labels.shape[1] != 2:
            raise ValueError(
                f"Mesh.boundary_labels must be (M, 2); got shape "
                f"{self.boundary_labels.shape}"
            )
        if self.quads.shape[0] != self.boundary_labels.shape[0]:
            raise ValueError(
                f"Mesh has {self.quads.shape[0]} quads but "
                f"{self.boundary_labels.shape[0]} boundary_labels rows"
            )
        if self.normals is not None:
            if self.normals.shape != self.points.shape:
                raise ValueError(
                    f"Mesh.normals shape {self.normals.shape} != points shape "
                    f"{self.points.shape}"
                )
        if self.stencils is not None:
            offsets, connectivity = self.stencils
            if offsets.ndim != 1 or connectivity.ndim != 1:
                raise ValueError("Mesh.stencils offsets and connectivity must be 1-D")
            if offsets.shape[0] != self.points.shape[0] + 1:
                raise ValueError(
                    f"Mesh.stencils offsets length {offsets.shape[0]} != "
                    f"num_points + 1 = {self.points.shape[0] + 1}"
                )

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_quads(self) -> int:
        return int(self.quads.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.num_points == 0 and self.num_quads == 0

    @property
    def has_normals(self) -> bool:
        return self.normals is not None

    @property
    def has_stencils(self) -> bool:
        return self.stencils is not None

    @classmethod
    def empty(
        cls,
        geometry: Geometry,
        schema: LabelSchema,
        *,
        with_normals: bool = False,
        with_stencils: bool = False,
    ) -> "Mesh":
        """Construct an empty mesh — useful as a ``mesh_concat`` accumulator seed."""
        return cls(
            points=np.zeros((0, 3), dtype=np.float32),
            quads=np.zeros((0, 4), dtype=np.int32),
            boundary_labels=np.zeros((0, 2), dtype=np.int32),
            geometry=geometry,
            schema=schema,
            normals=(np.zeros((0, 3), dtype=np.float32) if with_normals else None),
            stencils=(
                (np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32))
                if with_stencils else None
            ),
        )


# ---------------------------------------------------------------------------
# RestorePlan — the inverse-transform recipe (returned, never hidden)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestorePlan:
    """Everything needed to map a model-frame prediction back to the caller's grid.

    Returned by ``preprocess.to_model_frame`` (which reoriented the input to a
    canonical orientation and resampled it to the model's spacing).
    ``postprocess.restore`` consumes it: it inverse-resamples the model-frame
    logits onto ``inference_geometry`` (the canonical-orientation grid at the
    source spacing), then reorients from ``inference_orientation`` back to
    ``source_orientation`` — landing on ``source_geometry``.

    This is how the pipeline avoids hidden state: the inverse instructions are
    a value the caller holds, not state stashed on an object or a temp file.
    The axis transpose (``transpose_forward``/``backward``) is *not* recorded
    here — the engine round-trips it internally, so the inverse never sees it.

    ``model_spacing_zyx`` doubles as a binding token — restore checks the
    prediction it's handed was computed at this plan's model spacing.
    """

    source_geometry: Geometry
    source_orientation: str
    inference_geometry: Geometry
    inference_orientation: str
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
    "Prediction",
    "Mesh",
    "RestorePlan",
    "BuildOptions",
]
