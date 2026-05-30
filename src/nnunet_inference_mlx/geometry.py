"""Volume/Segmentation-native geometry ops: bounding box, crop, paste.

The value-typed siblings of the SITK/numpy primitives in ``workflow.py``
(``Bbox``/``compute_fg_bbox``/``crop_image``/``paste_segmentation``). These
operate directly on :class:`Volume` / :class:`Segmentation` values and keep
world geometry correct: a crop shifts the origin to the world coordinate of
the cropped corner (matching SITK's ``RegionOfInterest``), so a cropped
sub-volume stays registered to the same physical space.

Used by the cascade re-expression in ``segment.py`` (coarse → crop FOV →
fine → paste) so the cascade no longer bridges to ``run_workflow``.

Pure values in, pure values out — no IO, no GPU allocation, no hidden state.
Axis order is ``(Z, Y, X)`` throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mlx.core as mx
import numpy as np

from .values import Geometry, LabelSchema, Segmentation, Volume


# ---------------------------------------------------------------------------
# Box — voxel-coordinate bounding box (Z, Y, X), inclusive-start exclusive-end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """Inclusive-start, exclusive-end voxel bounding box in ``(Z, Y, X)``.

    Indices are voxel positions in whichever grid the box refers to; the
    caller tracks which grid (``compose`` chains a sub-box back into an
    outer grid's coordinates).
    """

    z_start: int
    z_end: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.z_end - self.z_start,
                self.y_end - self.y_start,
                self.x_end - self.x_start)

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (slice(self.z_start, self.z_end),
                slice(self.y_start, self.y_end),
                slice(self.x_start, self.x_end))

    def clamped(self, max_shape_zyx: tuple[int, int, int]) -> "Box":
        return Box(
            z_start=max(0, self.z_start), z_end=min(max_shape_zyx[0], self.z_end),
            y_start=max(0, self.y_start), y_end=min(max_shape_zyx[1], self.y_end),
            x_start=max(0, self.x_start), x_end=min(max_shape_zyx[2], self.x_end),
        )

    def dilated(self, voxels_zyx: int | tuple[int, int, int], *,
                max_shape_zyx: tuple[int, int, int] | None = None) -> "Box":
        dz, dy, dx = (voxels_zyx, voxels_zyx, voxels_zyx) \
            if isinstance(voxels_zyx, int) else voxels_zyx
        new = Box(
            z_start=self.z_start - dz, z_end=self.z_end + dz,
            y_start=self.y_start - dy, y_end=self.y_end + dy,
            x_start=self.x_start - dx, x_end=self.x_end + dx,
        )
        return new.clamped(max_shape_zyx) if max_shape_zyx is not None else new

    def compose(self, sub: "Box") -> "Box":
        """Express ``sub`` (in this box's coords) in the outer coord system."""
        return Box(
            z_start=self.z_start + sub.z_start, z_end=self.z_start + sub.z_end,
            y_start=self.y_start + sub.y_start, y_end=self.y_start + sub.y_end,
            x_start=self.x_start + sub.x_start, x_end=self.x_start + sub.x_end,
        )

    @classmethod
    def full(cls, shape_zyx: tuple[int, int, int]) -> "Box":
        Z, Y, X = shape_zyx
        return cls(0, Z, 0, Y, 0, X)


# ---------------------------------------------------------------------------
# bbox of labels
# ---------------------------------------------------------------------------


def bbox_of_labels(
    seg: Segmentation,
    *,
    classes: Iterable[int] | None = None,
    dilation_mm: float = 0.0,
) -> Box | None:
    """Foreground bounding box of a :class:`Segmentation`.

    ``classes=None`` treats any nonzero label as foreground; pass a tuple to
    crop around specific structures. ``dilation_mm`` expands the box outward
    by that physical distance per axis (converted to voxels via the
    segmentation's spacing) and clamps to the volume. Returns ``None`` when
    no foreground voxel is present — the signal to skip the downstream crop.
    """
    labels = np.asarray(seg.data)
    spacing_zyx = seg.geometry.spacing_zyx

    if classes is None:
        fg = labels > 0
    else:
        info = np.iinfo(labels.dtype) if np.issubdtype(labels.dtype, np.integer) else None
        if info is not None:
            in_range = [c for c in classes if info.min <= c <= info.max]
        else:
            in_range = list(classes)
        if not in_range:
            return None
        fg = np.isin(labels, np.asarray(in_range, dtype=labels.dtype))

    if not fg.any():
        return None

    z_idx = np.where(fg.any(axis=(1, 2)))[0]
    y_idx = np.where(fg.any(axis=(0, 2)))[0]
    x_idx = np.where(fg.any(axis=(0, 1)))[0]
    box = Box(
        z_start=int(z_idx[0]), z_end=int(z_idx[-1]) + 1,
        y_start=int(y_idx[0]), y_end=int(y_idx[-1]) + 1,
        x_start=int(x_idx[0]), x_end=int(x_idx[-1]) + 1,
    )

    if dilation_mm > 0:
        dvox = tuple(max(1, int(round(dilation_mm / s))) for s in spacing_zyx)
        box = box.dilated(dvox, max_shape_zyx=labels.shape)

    return box


# ---------------------------------------------------------------------------
# crop / paste (world-geometry preserving)
# ---------------------------------------------------------------------------


def _shifted_origin(geometry: Geometry, box: Box) -> tuple[float, float, float]:
    """World coordinate of the box's start corner — the new origin after crop.

    Mirrors SITK ``RegionOfInterest``: ``origin + D · (spacing ⊙ index_xyz)``
    where ``index_xyz = (x_start, y_start, z_start)`` and ``D`` is the
    row-major direction matrix. Correct for oblique (non-identity) directions.
    """
    sz, sy, sx = geometry.spacing_zyx
    d = geometry.direction_xyz
    # physical step along each voxel axis, in XYZ index order
    step_xyz = (sx * box.x_start, sy * box.y_start, sz * box.z_start)
    ox, oy, oz = geometry.origin_xyz
    return (
        ox + d[0] * step_xyz[0] + d[1] * step_xyz[1] + d[2] * step_xyz[2],
        oy + d[3] * step_xyz[0] + d[4] * step_xyz[1] + d[5] * step_xyz[2],
        oz + d[6] * step_xyz[0] + d[7] * step_xyz[1] + d[8] * step_xyz[2],
    )


def crop(volume: Volume, box: Box) -> Volume:
    """Extract the sub-volume in ``box``, preserving world geometry.

    The returned volume's voxel (0,0,0) maps to the same physical point as
    ``box``'s start corner in the input; spacing and direction are unchanged.
    """
    box = box.clamped(volume.geometry.shape_zyx)
    data = volume.data[box.slices[0], box.slices[1], box.slices[2], :]
    geom = Geometry(
        spacing_zyx=volume.geometry.spacing_zyx,
        shape_zyx=box.shape_zyx,
        origin_xyz=_shifted_origin(volume.geometry, box),
        direction_xyz=volume.geometry.direction_xyz,
    )
    return Volume(data=data, geometry=geom, channels=volume.channels)


def paste(
    patch: Segmentation,
    canvas_geometry: Geometry,
    box: Box,
    *,
    fill: int = 0,
) -> Segmentation:
    """Paste a cropped-space :class:`Segmentation` into a full-grid canvas.

    The canvas (shape ``canvas_geometry.shape_zyx``) is filled with ``fill``
    (default background), then ``patch`` is written into ``box``. ``patch``'s
    spatial shape must equal ``box.shape_zyx``. The result carries
    ``canvas_geometry`` and ``patch``'s schema.
    """
    if patch.data.shape != box.shape_zyx:
        raise ValueError(
            f"patch shape {tuple(patch.data.shape)} != box shape {box.shape_zyx}"
        )
    patch_np = np.asarray(patch.data)
    out = np.full(canvas_geometry.shape_zyx, fill, dtype=patch_np.dtype)
    out[box.slices] = patch_np
    return Segmentation(data=mx.array(out), geometry=canvas_geometry,
                        schema=patch.schema)


__all__ = ["Box", "bbox_of_labels", "crop", "paste"]
