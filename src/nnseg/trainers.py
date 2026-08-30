"""Make models trained with custom nnU-Net trainers loadable for inference.

Many TotalSegmentator models were trained with custom trainer classes - the fine model behind
lung_vessels uses ``nnUNetTrainerSkeletonRecall``, others use NoMirroring / different-epoch /
region-loss variants. nnU-Net's predictor recovers the trainer by name to build the network,
and stock nnU-Net does not contain those classes, so a bare install cannot load them.

But those trainers differ from the base only in *training* - loss, augmentation, schedule,
epoch count. The inference network architecture comes from ``plans.json`` via the base
``build_network_architecture``, so for inference a trivial subclass of ``nnUNetTrainer`` is
behaviorally identical. This writes such a shim for any unresolved trainer name into a cache
directory and points ``nnUNet_extTrainer`` at it - generic, so it needs no per-model manifest
and keeps working as TS adds trainers.
"""
from __future__ import annotations

import os
from pathlib import Path

_SHIM_DIR = Path(os.environ.get("NNSEG_TRAINER_SHIMS", Path.home() / ".cache" / "nnseg" / "trainer_shims"))


def trainer_name_of(model_folder) -> str:
    """The trainer name from a ``{trainer}__{plans}__{config}`` folder name."""
    return Path(model_folder).name.split("__")[0]


def _resolvable(trainer_name: str) -> bool:
    try:
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
        recursive_find_trainer_class_by_name(trainer_name)
        return True
    except Exception:
        return False


def ensure_trainer(model_folder, *, shim_dir: Path | None = None) -> str | None:
    """Guarantee the model's trainer resolves; shim it if not. Returns the shim path if written.

    Idempotent. The shim is a subclass of ``nnUNetTrainer`` with no overrides - correct for
    inference because the architecture is defined by the plans, not the trainer. If a future
    trainer genuinely changed the architecture, this would load the default one; no TS model
    does, and a mismatch would surface immediately as a weight-shape error at load.
    """
    name = trainer_name_of(model_folder)
    if _resolvable(name):
        return None
    shim_dir = Path(shim_dir or _SHIM_DIR)
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / f"{name}.py"
    if not shim.exists():
        shim.write_text(
            "# Auto-generated inference shim: a custom trainer's architecture comes from plans.json,\n"
            "# so a bare subclass loads and predicts identically. See nnseg/trainers.py.\n"
            "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n\n\n"
            f"class {name}(nnUNetTrainer):\n    pass\n"
        )
    existing = os.environ.get("nnUNet_extTrainer", "")
    paths = existing.split(os.pathsep) if existing else []
    if str(shim_dir) not in paths:
        os.environ["nnUNet_extTrainer"] = os.pathsep.join([*paths, str(shim_dir)])
    return str(shim)
