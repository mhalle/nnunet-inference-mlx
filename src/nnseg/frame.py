"""The geometry record: how the model grid relates to the source image, and how any
output grid maps into the model grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .grid import Grid
from .mapping import Mapping

CONVENTIONS = ("corner", "center")


@dataclass(frozen=True)
class Frame:
    """Everything needed to put labels computed on the model grid onto a caller-chosen grid.

    ``source`` is the canonical-orientation source image grid in a *local* frame
    (voxel (0, 0, 0) at 0 mm, spacing = the image's spacing, (Z, Y, X) order);
    ``canonical`` places that local frame in world space (the SimpleITK geometry of
    the RAS-reoriented image: origin and direction cosines), and
    ``original_orientation`` is the input's own orientation code, so the output can
    be written back in the frame the caller supplied. ``model_shape`` is the
    model-grid shape the forward resampler produced from ``source`` under
    ``convention`` ("corner" = scipy.zoom / TotalSegmentator, "center" =
    skimage / nnU-Net-native). ``model_spacing`` is informational.
    """

    source: Grid
    model_shape: tuple[int, int, int]
    model_spacing: tuple[float, float, float]
    convention: str
    canonical: object                      # nnseg.values.Geometry
    original_orientation: str = "RAS"
    model_source: Grid | None = None

    def __post_init__(self):
        if self.convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}; got {self.convention!r}")
        object.__setattr__(self, "model_shape", tuple(int(s) for s in self.model_shape))

    @property
    def resampled_from(self) -> Grid:
        """The grid actually handed to the forward resampler.

        Equal to ``source`` unless the source was cropped first (nnU-Net's crop-to-nonzero,
        which happens in source space *before* resampling); then it is the cropped sub-grid,
        whose ``origin`` carries the crop offset. Output grids still refer to ``source``, so
        ``mapping`` composes the offset in automatically and voxels outside the crop map
        outside the model grid - where ``outside="background"`` handles them.
        """
        return self.model_source or self.source

    @property
    def forward_rule(self) -> Mapping:
        """source index -> model-grid coordinate: the forward resampler's rule, inverted exactly."""
        if self.convention == "corner":
            return Mapping.corner(self.resampled_from.shape, self.model_shape)
        return Mapping.center(self.resampled_from.shape, self.model_shape)

    def mapping(self, grid: Grid) -> Mapping:
        """Output-grid index -> model-grid coordinate, for any grid sharing the source axes."""
        return Mapping.between(grid, self.resampled_from) >> self.forward_rule

    def resolve_grid(self, grid) -> Grid:
        """``"input"`` -> the source grid; a number -> isotropic at that spacing (same field of
        view); a Grid -> itself."""
        if grid is None or grid == "input":
            return self.source
        if isinstance(grid, Grid):
            return grid
        return Grid.isotropic(float(grid), like=self.source)

    def output_geometry(self, grid: Grid):
        """SimpleITK geometry for labels on ``grid``, in the canonical frame.

        The grid's origin is an offset in the source's local (Z, Y, X) millimetre frame;
        world position is the canonical origin plus that offset rotated by the direction
        cosines, so an oblique acquisition keeps its orientation.
        """
        from .values import Geometry
        d = np.asarray(self.canonical.direction_xyz, dtype=np.float64).reshape(3, 3)
        offset_xyz = np.asarray(grid.origin, dtype=np.float64)[::-1] - np.asarray(self.source.origin)[::-1]
        origin = np.asarray(self.canonical.origin_xyz, dtype=np.float64) + d @ offset_xyz
        return Geometry(spacing_zyx=tuple(float(x) for x in grid.spacing),
                        shape_zyx=tuple(int(x) for x in grid.shape),
                        origin_xyz=tuple(float(x) for x in origin),
                        direction_xyz=tuple(float(x) for x in self.canonical.direction_xyz))
