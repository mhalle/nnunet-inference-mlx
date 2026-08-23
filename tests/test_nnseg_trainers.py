"""Custom-trainer shimming (no nnU-Net needed for the shim-writing logic)."""
import os
from nnseg.trainers import ensure_trainer, trainer_name_of


def test_trainer_name_from_folder():
    assert trainer_name_of("/w/Dataset117_x/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres") == "nnUNetTrainerSkeletonRecall"


def test_shim_written_and_env_set_for_unknown_trainer(tmp_path, monkeypatch):
    monkeypatch.delenv("nnUNet_extTrainer", raising=False)
    folder = tmp_path / "Dataset999_x" / "nnUNetTrainerMadeUpNonexistent__nnUNetPlans__3d_fullres"
    folder.mkdir(parents=True)
    shims = tmp_path / "shims"
    p = ensure_trainer(folder, shim_dir=shims)
    assert p and (shims / "nnUNetTrainerMadeUpNonexistent.py").exists()
    body = (shims / "nnUNetTrainerMadeUpNonexistent.py").read_text()
    assert "class nnUNetTrainerMadeUpNonexistent(nnUNetTrainer)" in body
    assert str(shims) in os.environ["nnUNet_extTrainer"]
    # idempotent: second call adds the path once, does not duplicate
    ensure_trainer(folder, shim_dir=shims)
    assert os.environ["nnUNet_extTrainer"].count(str(shims)) == 1
