"""Axis-aligned sampling grids in a canonical frame."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Vec3 = tuple[float, float, float]
Shape3 = tuple[int, int, int]


def _vec3(v, name: str) -> Vec3:
    if np.isscalar(v):
        v = (v, v, v)
    t = tuple(float(x) for x in v)
    if len(t) != 3:
        raise ValueError(f"{name} must have 3 entries (Z, Y, X); got {v!r}")
    return t


def _shape3(v, name: str) -> Shape3:
    t = tuple(int(x) for x in v)
    if len(t) != 3 or any(x < 1 for x in t):
        raise ValueError(f"{name} must be 3 positive ints (Z, Y, X); got {v!r}")
    return t


@dataclass(frozen=True)
class Grid:
    """A regular, axis-aligned sampling grid.

    ``shape`` voxels per axis at ``spacing`` mm; the *center* of voxel (0, 0, 0)
    sits at ``origin`` mm. Axis order is (Z, Y, X) everywhere. A Grid lives in
    whatever canonical frame the caller uses (nnU-Net: RAS) and carries no
    direction cosines - orientation is the caller's ``Frame``, not the grid's.
    """

    shape: Shape3
    spacing: Vec3 = (1.0, 1.0, 1.0)
    origin: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "shape", _shape3(self.shape, "shape"))
        object.__setattr__(self, "spacing", _vec3(self.spacing, "spacing"))
        object.__setattr__(self, "origin", _vec3(self.origin, "origin"))
        if any(s <= 0 for s in self.spacing):
            raise ValueError(f"spacing must be positive; got {self.spacing}")

    # -- geometry ---------------------------------------------------------
    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    def index_to_mm(self, index) -> np.ndarray:
        """Continuous voxel index (..., 3) -> physical position (..., 3) in mm."""
        return np.asarray(self.origin) + np.asarray(index, dtype=np.float64) * np.asarray(self.spacing)

    def mm_to_index(self, mm) -> np.ndarray:
        """Physical position (..., 3) in mm -> continuous voxel index (..., 3)."""
        return (np.asarray(mm, dtype=np.float64) - np.asarray(self.origin)) / np.asarray(self.spacing)

    @property
    def extent_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Outer edges of the voxel volumes: (lo, hi) in mm."""
        sp = np.asarray(self.spacing)
        lo = np.asarray(self.origin) - sp / 2
        hi = np.asarray(self.origin) + (np.asarray(self.shape) - 1) * sp + sp / 2
        return lo, hi

    @property
    def center_extent_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """First and last voxel centers: (lo, hi) in mm."""
        lo = np.asarray(self.origin, dtype=np.float64)
        return lo, lo + (np.asarray(self.shape) - 1) * np.asarray(self.spacing)

    # -- constructors -----------------------------------------------------
    @classmethod
    def like(cls, other) -> "Grid":
        """A Grid with the geometry of ``other`` (a Grid or anything with
        ``shape``, ``spacing`` and ``origin`` attributes in (Z, Y, X) order)."""
        if isinstance(other, Grid):
            return other
        return cls(other.shape, getattr(other, "spacing", (1.0, 1.0, 1.0)), getattr(other, "origin", (0.0, 0.0, 0.0)))

    def resampled(self, spacing, *, align: str = "edges") -> "Grid":
        """The same field of view at a new spacing.

        ``align="edges"`` keeps the outer edges of the voxel volumes in place
        (``n_out = round(n * s_in / s_out)``, origin shifted by half a voxel of
        each - the voxel-center / half-pixel convention in physical terms).
        ``align="centers"`` keeps the first and last voxel centers in place
        (``n_out = round((n - 1) * s_in / s_out) + 1``, origin unchanged - the
        voxel-corner convention).
        """
        s_out = _vec3(spacing, "spacing")
        n = np.asarray(self.shape)
        s_in = np.asarray(self.spacing)
        s_o = np.asarray(s_out)
        if align == "edges":
            n_out = np.maximum(1, np.rint(n * s_in / s_o)).astype(int)
            origin = np.asarray(self.origin) - s_in / 2 + s_o / 2
        elif align == "centers":
            n_out = np.maximum(1, np.rint((n - 1) * s_in / s_o) + 1).astype(int)
            origin = np.asarray(self.origin)
        else:
            raise ValueError(f"align must be 'edges' or 'centers'; got {align!r}")
        return Grid(tuple(int(x) for x in n_out), s_out, tuple(float(x) for x in origin))

    @classmethod
    def isotropic(cls, spacing: float, *, like: "Grid", align: str = "edges") -> "Grid":
        """An isotropic grid covering ``like``'s field of view."""
        return Grid.like(like).resampled((float(spacing),) * 3, align=align)

    def roi(self, lo_mm, hi_mm) -> "Grid":
        """The sub-grid of this lattice whose voxel volumes intersect the box
        ``[lo_mm, hi_mm]`` (clipped to this grid)."""
        lo = np.asarray(_vec3(lo_mm, "lo_mm"))
        hi = np.asarray(_vec3(hi_mm, "hi_mm"))
        if np.any(hi < lo):
            raise ValueError("hi_mm must be >= lo_mm on every axis")
        n = np.asarray(self.shape)
        i0 = np.clip(np.floor(self.mm_to_index(lo) + 0.5), 0, n - 1).astype(int)
        i1 = np.clip(np.ceil(self.mm_to_index(hi) - 0.5), i0, n - 1).astype(int)
        shape = tuple(int(x) for x in (i1 - i0 + 1))
        origin = tuple(float(x) for x in self.index_to_mm(i0))
        return Grid(shape, self.spacing, origin)

    def __repr__(self) -> str:
        sp = ", ".join(f"{s:g}" for s in self.spacing)
        og = ", ".join(f"{o:g}" for o in self.origin)
        return f"Grid(shape={self.shape}, spacing=({sp}), origin=({og}))"
