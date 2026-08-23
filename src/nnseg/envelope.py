"""Restrict inference to the body: the largest single speedup available for CT.

On a chest CT the labelled anatomy occupies about a third of the volume; the rest is air and
table, and nnU-Net's sliding window tiles all of it. A body envelope - the bounding box of the
patient with a margin - cuts the 1.5 mm patch count from 175 to 42-63 per model on the chest
measured in docs/backend-decision.md.

The envelope comes from a HU threshold, not from a model: air is below -500 HU in any CT, the
largest connected component above it is the patient, and that outline includes skin and fat,
which is the context the fine model's boundary patches need. A coarse *model* could do this
too, but it inherits that model's blind spots; the threshold cannot miss a body.

The margin is the one real parameter. Too small and the patches at the envelope edge lose
context they had in the full volume; the test sweeps it and asserts the labels inside the
body are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AIR_HU = -500.0


@dataclass(frozen=True)
class Envelope:
    """Half-open voxel bounds ``[lo, hi)`` on the grid the mask was computed on, (Z, Y, X)."""

    lo: tuple[int, int, int]
    hi: tuple[int, int, int]
    shape: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(int(a), int(b)) for a, b in zip(self.lo, self.hi))

    @property
    def fraction(self) -> float:
        return float(np.prod(np.array(self.hi) - np.array(self.lo)) / np.prod(self.shape))

    def is_whole(self) -> bool:
        return all(a == 0 for a in self.lo) and tuple(self.hi) == tuple(self.shape)


def body_mask(hu_zyx: np.ndarray, *, threshold: float = AIR_HU, largest_component: bool = True) -> np.ndarray:
    """Voxels that are not air, restricted to the largest connected component (the patient).

    Works on the coarse grid the fine model will run on, so the mask is cheap (the 3 mm grid
    of a chest is 6.6 M voxels). Table and cables are usually thin or disconnected from the
    body and drop out with the component filter; if not, they only enlarge the box slightly.
    """
    mask = np.asarray(hu_zyx) > threshold
    if not mask.any():
        return mask
    if largest_component:
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        if n > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            mask = labels == int(sizes.argmax())
    return mask


def envelope_of(mask_zyx: np.ndarray, *, margin_voxels) -> Envelope:
    """Bounding box of the mask, padded by ``margin_voxels`` per axis, clipped to the grid.

    An empty mask yields the whole grid: the safe direction is to run the full volume, never
    an empty slab.
    """
    shape = tuple(int(s) for s in mask_zyx.shape)
    if not mask_zyx.any():
        return Envelope((0, 0, 0), shape, shape)
    m = np.broadcast_to(np.asarray(margin_voxels, dtype=np.int64), (3,))
    idx = np.nonzero(mask_zyx)
    lo = tuple(int(max(0, i.min() - mm)) for i, mm in zip(idx, m))
    hi = tuple(int(min(n, i.max() + 1 + mm)) for i, n, mm in zip(idx, shape, m))
    return Envelope(lo, hi, shape)


def margin_in_voxels(margin_mm: float, spacing_zyx) -> tuple[int, int, int]:
    return tuple(int(np.ceil(float(margin_mm) / float(s))) for s in spacing_zyx)
