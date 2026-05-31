"""Tests for the TotalSegmentator-compatible CLI (totalseg-mlx).

Parsing, task resolution, and error paths — no real weights. The full
per-class/--ml output path is exercised by the real-weights runs.
"""

from __future__ import annotations

import pytest

from nnunet_inference_mlx.catalog import TaskCatalog
from nnunet_inference_mlx.ts_cli import _resolve_task, build_parser, main


class TestParserMirrorsTS:
    def test_full_ts_command_line_parses(self):
        args = build_parser().parse_args([
            "-i", "ct.nii.gz", "-o", "out", "--fast", "-rs", "liver", "spleen",
            "-rmb", "-s", "--device", "gpu", "--radiomics", "-bs", "-q",
        ])
        assert args.input.name == "ct.nii.gz"
        assert args.fast and args.remove_small_blobs and args.quiet
        assert args.roi_subset == ["liver", "spleen"]
        assert args.statistics is True          # -s with no value
        assert args.radiomics and args.body_seg  # unsupported, still parse

    def test_statistics_with_path(self):
        args = build_parser().parse_args(["-i", "a", "-o", "b", "-s", "stats.json"])
        assert str(args.statistics) == "stats.json"

    def test_io_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_version_exits_zero(self):
        with pytest.raises(SystemExit) as e:
            build_parser().parse_args(["--version"])
        assert e.value.code == 0


class TestTaskResolution:
    def test_total_default(self):
        assert _resolve_task(TaskCatalog("totalsegmentator"), "total", False, False).name == "total"

    def test_fast_maps_to_fast_variant(self):
        assert _resolve_task(TaskCatalog("totalsegmentator"), "total", True, False).name == "total_fast"

    def test_fastest_maps_to_fastest_variant(self):
        assert _resolve_task(TaskCatalog("totalsegmentator"), "total", False, True).name == "total_fastest"

    def test_fast_falls_back_when_no_variant(self):
        cat = TaskCatalog("totalsegmentator")
        names = {n.split(":")[-1] for n in cat.names()}
        base = "cerebral_bleed"
        if base not in names or f"{base}_fast" in names:
            pytest.skip("no suitable no-fast-variant task in registry")
        assert _resolve_task(cat, base, True, False).name == base


class TestMainErrorPaths:
    def test_dicom_output_type_rejected(self, tmp_path):
        rc = main(["-i", str(tmp_path / "x.nii.gz"), "-o", str(tmp_path / "o"),
                   "-ot", "dicom_seg", "-q"])
        assert rc == 2

    def test_unknown_task_exits_2(self, tmp_path):
        rc = main(["-i", str(tmp_path / "x.nii.gz"), "-o", str(tmp_path / "o"),
                   "-ta", "no_such_task_xyz", "-q"])
        assert rc == 2
