"""Post-inference cleanup helpers.

Currently exposes a single op: connected-component "dust" — drop
small isolated label islands while preserving the original class IDs.

cc3d is the only backend. It handles multi-label volumes in a single
C++ pass (two same-class neighbors are connected, different-class
neighbors are not) and is ~10× faster than the equivalent SITK
ScalarConnectedComponent + RelabelComponent chain, ~90× faster than
scipy's per-label loop. Imported lazily so the optional
``[postprocessing]`` extra (``pip install nnunet-inference-mlx[postprocessing]``)
is only required when these helpers are actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def _require_cc3d():
    try:
        import cc3d
    except ImportError as e:
        raise ImportError(
            "connected-components-3d is required for component cleanup. "
            "Install with: pip install 'nnunet-inference-mlx[postprocessing]'"
        ) from e
    return cc3d


def remove_small_components(
    labels: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    min_volume_mm3: float = 200.0,
    connectivity: int = 26,
    in_place: bool = False,
) -> np.ndarray:
    """Drop label-island components smaller than ``min_volume_mm3``.

    Multi-label aware: two neighboring voxels count as connected only if
    they share the same nonzero label, so disconnected pieces of the same
    class are filtered independently. The original label IDs are
    preserved — only background (0) replaces removed components.

    Matches TotalSegmentator's ``remove_small_blobs_multilabel`` default
    (``size_thr_mm3=200``) but applies it across every nonzero class in
    one pass instead of looping per ROI.

    Parameters
    ----------
    labels :
        Integer label volume in (Z, Y, X) order.
    spacing_zyx :
        Voxel spacing in mm, in the same axis order as ``labels``. Used
        to convert ``min_volume_mm3`` into a voxel count.
    min_volume_mm3 :
        Physical-volume threshold in mm³. Components below this size are
        zeroed. Defaults to ``200.0`` to match TS's ``--remove_small_blobs``
        flag. Pass ``0`` for a no-op.
    connectivity :
        Voxel adjacency for the CC pass. ``26`` (face + edge + corner) is
        the default; ``18`` (face + edge) and ``6`` (face only) are also
        accepted by cc3d.
    in_place :
        If True, mutate ``labels`` directly. Default False returns a copy.

    Returns
    -------
    np.ndarray
        Cleaned label volume in (Z, Y, X) order, same dtype as input.
    """
    if min_volume_mm3 <= 0:
        return labels if in_place else labels.copy()

    cc3d = _require_cc3d()
    vox_mm3 = float(np.prod(spacing_zyx))
    threshold_vox = max(1, int(round(min_volume_mm3 / vox_mm3)))
    return cc3d.dust(
        labels,
        threshold=threshold_vox,
        connectivity=connectivity,
        in_place=in_place,
    )


__all__ = ["remove_small_components"]
