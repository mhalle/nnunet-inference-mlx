"""Real-weights integration tests (@slow; skipped when weights/CT absent).

These exercise what synthetic models and old-vs-new parity structurally cannot:
anatomical correctness on a real CT. The left/right handedness check below
fails on the LPS-default mirror bug and passes once the canonical is RAS — the
class of bug only real data surfaces.

Run with: uv run pytest -m slow
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow
sitk = pytest.importorskip("SimpleITK")

_RESULTS = Path("~/.totalsegmentator/nnunet/results").expanduser()
_CT = Path("~/tmp/data/CT_Abdo.nii.gz").expanduser()


def _weights_present() -> bool:
    return _CT.exists() and bool(list(_RESULTS.glob("Dataset297_*")))


requires_weights = pytest.mark.skipif(
    not _weights_present(),
    reason="TS Dataset297 weights or the test CT (~/tmp/data/CT_Abdo.nii.gz) not present",
)


@requires_weights
def test_total_fast_left_right_anatomy():
    """Liver (patient-right) must sit on the opposite world-X side from spleen
    (patient-left). SITK world space is LPS (+X = patient LEFT), so a correct
    segmentation has spleen.X > liver.X. The LPS-canonical bug mirrored this.
    """
    from nnunet_inference_mlx import ModelStore, NiftiReader
    from nnunet_inference_mlx.imageio import array_to_sitk

    store = ModelStore("totalsegmentator", max_memory_mb=8000)
    seg = store.load(297).segment(NiftiReader().read(_CT))
    arr = np.asarray(seg.data)
    img = array_to_sitk(arr, seg.geometry)

    def world_x(label: int) -> float:
        idx = np.argwhere(arr == label)
        assert len(idx), f"label {label} not segmented"
        c = idx.mean(0)  # (z, y, x)
        return img.TransformContinuousIndexToPhysicalPoint(
            [float(c[2]), float(c[1]), float(c[0])]
        )[0]

    liver, spleen = world_x(5), world_x(1)
    assert spleen > liver, (
        f"left/right mirrored: spleen world-X ({spleen:.1f}) should exceed "
        f"liver world-X ({liver:.1f}) — liver is a patient-right organ."
    )
