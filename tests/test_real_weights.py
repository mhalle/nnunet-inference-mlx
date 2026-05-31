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

import json
import os

_RESULTS = Path("~/.totalsegmentator/nnunet/results").expanduser()
_CT = Path("~/tmp/data/CT_Abdo.nii.gz").expanduser()


def _weights_present() -> bool:
    return _CT.exists() and bool(list(_RESULTS.glob("Dataset297_*")))


def _dataset_present(prefix: str) -> bool:
    return _CT.exists() and bool(list(_RESULTS.glob(f"{prefix}*")))


def _find_region_dataset() -> int | None:
    """First region-based (sigmoid/paint) dataset id under the TS results, if any."""
    for ds in sorted(_RESULTS.glob("Dataset*")):
        for j in ds.glob("*/dataset.json"):
            try:
                labels = json.loads(j.read_text()).get("labels", {})
            except Exception:
                continue
            if any(isinstance(v, (list, tuple)) for v in labels.values()):
                stem = ds.name[len("Dataset"):].split("_", 1)[0]
                return int(stem) if stem.isdigit() else None
    return None


def _moose_root() -> Path | None:
    for var in ("NNUNET_MLX_MOOSE_MODELS", "MOOSE_MODELS"):
        v = os.environ.get(var)
        if v and Path(v).expanduser().is_dir():
            return Path(v).expanduser()
    return None


requires_weights = pytest.mark.skipif(
    not _weights_present(),
    reason="TS Dataset297 weights or the test CT (~/tmp/data/CT_Abdo.nii.gz) not present",
)


def _world_x(arr, geometry, label):
    import numpy as np
    from nnunet_inference_mlx.imageio import array_to_sitk
    idx = np.argwhere(arr == label)
    if not len(idx):
        return None
    c = idx.mean(0)
    img = array_to_sitk(arr, geometry)
    return img.TransformContinuousIndexToPhysicalPoint([float(c[2]), float(c[1]), float(c[0])])[0]


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


@requires_weights
def test_total_fast_organ_volumes_sane():
    """On an abdominal CT the segmentation should be anatomically plausible:
    many structures found, and liver (largest solid organ) bigger than spleen."""
    from nnunet_inference_mlx import ModelStore, NiftiReader

    store = ModelStore("totalsegmentator", max_memory_mb=8000)
    seg = store.load(297).segment(NiftiReader().read(_CT))
    arr = np.asarray(seg.data)
    present = {int(v) for v in np.unique(arr)} - {0}
    assert len(present) >= 20, f"only {len(present)} structures found"
    assert int((arr == 5).sum()) > int((arr == 1).sum()), "liver should exceed spleen volume"


@pytest.mark.skipif(not _dataset_present("Dataset117_"),
                    reason="Dataset117 (lung airways/vessels) weights not present")
def test_nondefault_trainer_model_runs():
    """Dataset117 uses a non-default trainer (nnUNetTrainerSkeletonRecall) — exercises
    config/trainer resolution + a standard model outside the TS 'total' family."""
    from nnunet_inference_mlx import ModelStore, NiftiReader

    store = ModelStore("totalsegmentator", max_memory_mb=8000)
    model = store.load(117)
    assert model.config.config_name == "3d_fullres"
    seg = model.segment(NiftiReader().read(_CT))
    assert seg.geometry.shape_zyx == NiftiReader().read(_CT).geometry.shape_zyx
    assert int(np.asarray(seg.data).max()) <= 4   # bg + 4 airway/vessel classes


@pytest.mark.skipif(_find_region_dataset() is None,
                    reason="no region-based (sigmoid/paint) weights available to test")
def test_region_model_paint_path():
    """Region (BraTS-style) models exercise the sigmoid + paint-priority restore
    path. Runs automatically if any region-based dataset is downloaded."""
    from nnunet_inference_mlx import ModelStore, NiftiReader

    ds = _find_region_dataset()
    store = ModelStore("totalsegmentator", max_memory_mb=8000)
    model = store.load(ds)
    assert model.schema.is_region_model
    seg = model.segment(NiftiReader().read(_CT))
    assert seg.geometry.shape_zyx == NiftiReader().read(_CT).geometry.shape_zyx


@pytest.mark.skipif(_moose_root() is None,
                    reason="no MOOSE models dir (set NNUNET_MLX_MOOSE_MODELS / MOOSE_MODELS)")
def test_moose_string_id_model_runs():
    """MOOSE identifies models by string folder name — exercises the moose
    ecosystem store path. Runs automatically if a MOOSE models dir is configured."""
    from nnunet_inference_mlx import ModelStore, NiftiReader

    store = ModelStore("moose")
    ids = store.downloaded()
    if not ids:
        pytest.skip("MOOSE models dir present but empty")
    seg = store.load(ids[0]).segment(NiftiReader().read(_CT))
    assert seg.geometry.shape_zyx == NiftiReader().read(_CT).geometry.shape_zyx
