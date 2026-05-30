"""Tests for TaskCatalog (Phase 4): the explicit, owned task catalog."""

from __future__ import annotations

import pytest

from nnunet_inference_mlx.catalog import AmbiguousTaskError, TaskCatalog
from nnunet_inference_mlx.tasks import TaskSpec


class TestTaskCatalog:
    def test_loads_builtin_ts(self):
        cat = TaskCatalog("totalsegmentator")
        assert len(cat) >= 40           # 51 as of TS 2.13
        assert cat["total_fast"].shape == "single"
        assert cat["total"].shape == "label_union"

    def test_bare_and_qualified_lookup(self):
        cat = TaskCatalog("totalsegmentator")
        assert cat.get("lung_vessels").shape == "cascade"
        assert cat.get("ts:lung_vessels").shape == "cascade"

    def test_unknown_raises(self):
        cat = TaskCatalog("totalsegmentator")
        with pytest.raises(KeyError, match="unknown task"):
            cat.get("not_a_task")

    def test_names_and_modality(self):
        cat = TaskCatalog("totalsegmentator")
        names = cat.names()
        assert all(n.startswith("ts:") for n in names)
        assert "ts:total_mr" in cat.by_modality("MR")
        assert "ts:total" in cat.by_modality("CT")

    def test_no_global_two_catalogs_independent(self):
        a = TaskCatalog("totalsegmentator")
        b = TaskCatalog()                      # empty
        assert len(b) == 0 and len(a) > 0      # constructing a didn't populate b

    def test_register_and_contains(self):
        cat = TaskCatalog()
        spec = TaskSpec(name="mine", source="user", modality="CT",
                        shape="single", single=5)
        cat.register(spec)
        assert "mine" in cat
        assert cat["mine"].single == 5

    def test_unknown_ecosystem_raises(self):
        with pytest.raises(ValueError, match="no built-in catalog"):
            TaskCatalog("bogus")


class TestMergeAndConflicts:
    def _user(self, name, single):
        return TaskSpec(name=name, source="user", modality="CT",
                        shape="single", single=single)

    def test_merge(self):
        ts = TaskCatalog("totalsegmentator")
        mine = TaskCatalog()
        mine.register(self._user("total", 999))   # collides with ts:total by bare name
        merged = ts | mine
        assert "ts:total" in merged and "user:total" in merged

    def test_ambiguous_bare_name_after_merge(self):
        ts = TaskCatalog("totalsegmentator")
        mine = TaskCatalog()
        mine.register(self._user("total", 999))
        merged = ts.merged_with(mine)
        with pytest.raises(AmbiguousTaskError, match="multiple sources"):
            merged.get("total")                     # ts:total vs user:total
        assert merged.get("user:total").single == 999   # qualify resolves
