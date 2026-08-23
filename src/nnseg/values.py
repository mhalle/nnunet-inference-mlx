"""Small value types nnseg needs, free of any framework.

These mirror the MLX toolkit's ``Geometry`` and ``LabelSchema`` deliberately: that package
imports ``mlx.core`` at module level, which does not exist off Apple silicon, so depending on
it would make nnseg unimportable on exactly the machines the torch path exists for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Geometry:
    """Voxel-grid placement in physical space, SimpleITK's conventions.

    ``spacing_zyx`` / ``shape_zyx`` are in array order; ``origin_xyz`` and the 9-tuple
    row-major ``direction_xyz`` are in SimpleITK's X, Y, Z order.
    """

    spacing_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction_xyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "spacing_zyx", tuple(float(s) for s in self.spacing_zyx))
        object.__setattr__(self, "shape_zyx", tuple(int(s) for s in self.shape_zyx))
        object.__setattr__(self, "origin_xyz", tuple(float(o) for o in self.origin_xyz))
        object.__setattr__(self, "direction_xyz", tuple(float(d) for d in self.direction_xyz))
        if len(self.direction_xyz) != 9:
            raise ValueError(f"direction_xyz must have 9 entries; got {len(self.direction_xyz)}")


@dataclass(frozen=True)
class LabelSchema:
    """Integer label -> name. Region (sigmoid-head) models additionally carry a paint order."""

    names: Mapping[int, str]
    paint_priority: tuple[int, ...] = ()

    @property
    def is_region_model(self) -> bool:
        return bool(self.paint_priority)

    def name_of(self, value: int) -> str:
        return self.names.get(int(value), f"label_{int(value)}")

    def __len__(self) -> int:
        return len(self.names)
