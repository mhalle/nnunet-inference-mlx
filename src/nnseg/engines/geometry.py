"""Grid geometry shared by the engines' restore paths.

Every engine computes its own conformed field (logits, a signed distance
transform, ...) on its own grid and then has to put that field back on the
input's grid. The mapping is the same derivation for all of them, so it lives
here once rather than being re-derived per engine.

numpy only - no torch, no SimpleITK import (the refs are duck-typed on the
SimpleITK image interface), so this stays importable anywhere.
"""
from __future__ import annotations

import numpy as np


def grid_record(ref) -> dict:
    """Plain JSON for one grid: what a reader needs to redo a restore without the image.

    The counterpart to :func:`resample_affine` - that function consumes two live refs, this
    one writes either of them down. An engine emitting a stored field has to carry both its
    own grid and the target's, or the arrays are only a picture of the grid they happened to
    be computed on and the restore can never be re-decided.

    Duck-typed on the SimpleITK image interface, like the rest of this module.
    """
    return {"size_xyz": [int(v) for v in ref.GetSize()],
            "spacing_xyz": [float(v) for v in ref.GetSpacing()],
            "origin_xyz": [float(v) for v in ref.GetOrigin()],
            "direction_xyz": [float(v) for v in ref.GetDirection()]}


def resample_affine(source_ref, target_ref):
    """``(A, t)`` mapping a TARGET voxel index ``(x, y, z)`` to the continuous
    SOURCE voxel index ``(x, y, z)``, composing both grids' physical transforms.

    ``source_ref`` / ``target_ref`` are SimpleITK images (only their
    origin/direction/spacing are read). The source direction is assumed
    orthonormal, so its inverse is the transpose scaled by the spacing. Feed the
    result to ``grid_sample`` (or an equivalent) to resample a field between the
    two grids in physical space - which is what makes oblique acquisitions work.
    """
    O_s = np.asarray(source_ref.GetOrigin(), dtype=np.float64)
    O_t = np.asarray(target_ref.GetOrigin(), dtype=np.float64)
    D_s = np.asarray(source_ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    D_t = np.asarray(target_ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    S_s = np.asarray(source_ref.GetSpacing(), dtype=np.float64)
    S_t = np.asarray(target_ref.GetSpacing(), dtype=np.float64)
    inv_s = np.diag(1.0 / S_s) @ D_s.T
    return inv_s @ D_t @ np.diag(S_t), inv_s @ (O_t - O_s)
