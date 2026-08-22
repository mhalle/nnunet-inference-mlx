"""The geometry record: how the model grid relates to the source image, and how any
output grid maps into the model grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from labelgrid import Grid, Mapping

CONVENTIONS = ("corner", "center")


@dataclass(frozen=True)
class Frame:
    """Everything needed to put labels computed on the model grid onto a caller-chosen grid.

    ``source`` is the canonical-orientation source image grid in a *local* frame
    (voxel (0, 0, 0) at 0 mm, spacing = the image's zooms, (Z, Y, X) order);
    ``affine_canonical`` places that local frame in world space (the canonical
    image's affine, x/y/z voxel order as nibabel has it). ``model_shape`` is the
    model-grid shape the forward resampler produced from ``source`` under
    ``convention`` ("corner" = scipy.zoom / TotalSegmentator, "center" =
    skimage / nnU-Net-native). ``model_spacing`` is informational.
    """

    source: Grid
    model_shape: tuple[int, int, int]
    model_spacing: tuple[float, float, float]
    convention: str
    affine_canonical: np.ndarray

    def __post_init__(self):
        if self.convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}; got {self.convention!r}")
        object.__setattr__(self, "model_shape", tuple(int(s) for s in self.model_shape))
        object.__setattr__(self, "affine_canonical", np.asarray(self.affine_canonical, dtype=np.float64))

    @property
    def forward_rule(self) -> Mapping:
        """source index -> model-grid coordinate: the forward resampler's rule, inverted exactly."""
        if self.convention == "corner":
            return Mapping.corner(self.source.shape, self.model_shape)
        return Mapping.center(self.source.shape, self.model_shape)

    def mapping(self, grid: Grid) -> Mapping:
        """Output-grid index -> model-grid coordinate, for any grid sharing the source axes."""
        return Mapping.between(grid, self.source) >> self.forward_rule

    def resolve_grid(self, grid) -> Grid:
        """``"input"`` -> the source grid; a number -> isotropic at that spacing (same field of
        view); a Grid -> itself."""
        if grid is None or grid == "input":
            return self.source
        if isinstance(grid, Grid):
            return grid
        return Grid.isotropic(float(grid), like=self.source)

    def output_affine(self, grid: Grid) -> np.ndarray:
        """World placement (nibabel x/y/z voxel order) of labels on ``grid``."""
        s_g = np.asarray(grid.spacing)[::-1]          # zyx -> xyz
        s_s = np.asarray(self.source.spacing)[::-1]
        o_g = np.asarray(grid.origin)[::-1]
        o_s = np.asarray(self.source.origin)[::-1]
        t = np.eye(4)
        t[:3, :3] = np.diag(s_g / s_s)
        t[:3, 3] = (o_g - o_s) / s_s
        return self.affine_canonical @ t
