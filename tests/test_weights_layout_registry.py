"""Tests for the WeightsLayout + registry machinery in engine.py.

These cover the pure-logic discovery and resolution paths; they don't
load any actual weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nnunet_inference_mlx import (
    WeightsLayout,
    discover_weights,
    list_weights_layouts,
    register_weights_layout,
)
from nnunet_inference_mlx.engine import (
    _WEIGHTS_LAYOUTS,
    _find_model_folder,
)


@pytest.fixture
def isolated_layouts(monkeypatch):
    """Snapshot + restore the module-level layouts list around each test."""
    original = list(_WEIGHTS_LAYOUTS)
    _WEIGHTS_LAYOUTS.clear()
    yield _WEIGHTS_LAYOUTS
    _WEIGHTS_LAYOUTS.clear()
    _WEIGHTS_LAYOUTS.extend(original)


def test_builtin_layouts_present():
    layouts = list_weights_layouts()
    names = [L.name for L in layouts]
    assert "nnUNet" in names
    assert "TotalSegmentator" in names


def test_weights_layout_resolves_from_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_WEIGHTS", str(tmp_path))
    L = WeightsLayout(name="X", env_var="MY_WEIGHTS")
    assert L.resolve_weights_dir() == tmp_path


def test_weights_layout_resolves_from_default_path(monkeypatch, tmp_path):
    monkeypatch.delenv("MY_WEIGHTS", raising=False)
    L = WeightsLayout(name="X", env_var="MY_WEIGHTS", default_path=tmp_path)
    assert L.resolve_weights_dir() == tmp_path


def test_weights_layout_returns_none_when_nothing_resolves(monkeypatch, tmp_path):
    monkeypatch.delenv("MY_WEIGHTS", raising=False)
    L = WeightsLayout(
        name="X", env_var="MY_WEIGHTS",
        default_path=tmp_path / "definitely-does-not-exist",
    )
    assert L.resolve_weights_dir() is None


def test_weights_layout_env_var_takes_precedence_over_default(monkeypatch, tmp_path):
    env_dir = tmp_path / "from_env"
    env_dir.mkdir()
    default_dir = tmp_path / "from_default"
    default_dir.mkdir()
    monkeypatch.setenv("MY_WEIGHTS", str(env_dir))
    L = WeightsLayout(name="X", env_var="MY_WEIGHTS", default_path=default_dir)
    assert L.resolve_weights_dir() == env_dir


def test_register_layout_appends(isolated_layouts, tmp_path, monkeypatch):
    monkeypatch.setenv("A_VAR", str(tmp_path))
    monkeypatch.setenv("B_VAR", str(tmp_path))
    register_weights_layout(WeightsLayout(name="A", env_var="A_VAR"))
    register_weights_layout(WeightsLayout(name="B", env_var="B_VAR"))
    names = [L.name for L in list_weights_layouts()]
    assert names == ["A", "B"]


def test_register_layout_prepend(isolated_layouts, tmp_path, monkeypatch):
    monkeypatch.setenv("FIRST", str(tmp_path))
    monkeypatch.setenv("SECOND", str(tmp_path))
    register_weights_layout(WeightsLayout(name="First", env_var="FIRST"))
    register_weights_layout(
        WeightsLayout(name="Second", env_var="SECOND"), prepend=True,
    )
    names = [L.name for L in list_weights_layouts()]
    assert names == ["Second", "First"]


def test_discover_weights_picks_first_matching(isolated_layouts, tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    monkeypatch.setenv("A_ENV", str(a))
    monkeypatch.setenv("B_ENV", str(b))
    register_weights_layout(WeightsLayout(name="A", env_var="A_ENV"))
    register_weights_layout(WeightsLayout(name="B", env_var="B_ENV"))

    path, layout = discover_weights()
    assert path == a
    assert layout.name == "A"


def test_discover_weights_raises_when_nothing_found(isolated_layouts, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    register_weights_layout(
        WeightsLayout(name="N", env_var="NOPE",
                      default_path=Path("/this/should/not/exist/anywhere")),
    )
    with pytest.raises(FileNotFoundError, match="No weights directory found"):
        discover_weights()


def test_find_model_folder_auto_picks_first_trainer(tmp_path):
    """When trainer/plans/model are all None, picks the alphabetically-first trainer subdir."""
    dataset_dir = tmp_path / "Dataset123_Test"
    dataset_dir.mkdir()
    (dataset_dir / "nnUNetTrainerB__nnUNetPlans__3d_fullres").mkdir()
    (dataset_dir / "nnUNetTrainerA__nnUNetPlans__3d_fullres").mkdir()

    found = _find_model_folder(123, tmp_path)
    assert found.name == "nnUNetTrainerA__nnUNetPlans__3d_fullres"


def test_find_model_folder_explicit_trainer(tmp_path):
    """Explicit trainer/plans/model builds the exact name and finds it."""
    dataset_dir = tmp_path / "Dataset123_Test"
    dataset_dir.mkdir()
    target = dataset_dir / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
    target.mkdir()
    (dataset_dir / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir()

    found = _find_model_folder(
        123, tmp_path,
        trainer="nnUNetTrainerNoMirroring",
    )
    assert found == target


def test_find_model_folder_missing_trainer_lists_available(tmp_path):
    dataset_dir = tmp_path / "Dataset123_Test"
    dataset_dir.mkdir()
    (dataset_dir / "nnUNetTrainerB__nnUNetPlans__3d_fullres").mkdir()
    (dataset_dir / "nnUNetTrainerA__nnUNetPlans__3d_fullres").mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        _find_model_folder(123, tmp_path, trainer="DoesNotExist")
    msg = str(exc_info.value)
    assert "DoesNotExist" in msg
    # The error message lists what's available
    assert "nnUNetTrainerA" in msg
    assert "nnUNetTrainerB" in msg


def test_find_model_folder_no_dataset_for_task(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model found for task 999"):
        _find_model_folder(999, tmp_path)


def test_find_model_folder_dataset_with_no_trainer(tmp_path):
    (tmp_path / "Dataset123_Empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No trainer folder"):
        _find_model_folder(123, tmp_path)
