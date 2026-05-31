"""Tests for the nnmlx CLI — command structure, catalog/store wiring, error paths.

No real weights: exercises the light commands (tasks/models/help) and the
segment error path. The real-weights segment happy-path is covered by the
@slow integration tests.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from nnunet_inference_mlx.cli import app

runner = CliRunner()


class TestHelpAndStructure:
    def test_root_help(self):
        r = runner.invoke(app, ["--help"])
        assert r.exit_code == 0
        for cmd in ("segment", "tasks", "models"):
            assert cmd in r.stdout

    def test_subcommand_help(self):
        assert runner.invoke(app, ["tasks", "--help"]).exit_code == 0
        assert runner.invoke(app, ["models", "--help"]).exit_code == 0
        assert runner.invoke(app, ["segment", "--help"]).exit_code == 0


class TestTasks:
    def test_list_totalsegmentator(self):
        r = runner.invoke(app, ["tasks", "list"])
        assert r.exit_code == 0
        assert "ts:total" in r.stdout

    def test_list_by_modality(self):
        r = runner.invoke(app, ["tasks", "list", "--modality", "CT"])
        assert r.exit_code == 0
        assert "ts:total" in r.stdout

    def test_show(self):
        r = runner.invoke(app, ["tasks", "show", "total"])
        assert r.exit_code == 0
        assert "label_union" in r.stdout
        assert "spleen" in r.stdout

    def test_show_unknown_exits_2(self):
        r = runner.invoke(app, ["tasks", "show", "no_such_task_xyz"])
        assert r.exit_code == 2


class TestModels:
    def test_list_empty_root(self, tmp_path):
        r = runner.invoke(app, ["--model-root", str(tmp_path), "models", "list"])
        assert r.exit_code == 0
        assert "none downloaded" in r.stdout

    def test_loaded_none(self, tmp_path):
        r = runner.invoke(app, ["--model-root", str(tmp_path), "models", "loaded"])
        assert r.exit_code == 0
        assert "none loaded" in r.stdout


class TestSegmentErrors:
    def test_missing_input_exits_nonzero(self, tmp_path):
        r = runner.invoke(app, ["segment", "total_fast",
                                str(tmp_path / "nope.nii.gz"), str(tmp_path / "out.nii.gz")])
        assert r.exit_code != 0   # Typer's exists=True rejects the missing input

    def test_unknown_task_exits_2(self, tmp_path):
        sitk = pytest.importorskip("SimpleITK")
        import numpy as np
        inp = tmp_path / "ct.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16)), str(inp))
        r = runner.invoke(app, ["segment", "no_such_task_xyz", str(inp),
                                str(tmp_path / "out.nii.gz")])
        assert r.exit_code == 2
