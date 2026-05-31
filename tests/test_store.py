"""Tests for ModelStore (Phase 2): the explicit read-through store.

The store's two transforms (read: folder→artifact, build: artifact→engine)
are injected as fakes, so we exercise resolution, the memory-bounded LRU,
the readiness verbs, freeing, and the absence of any global — all without
real weights or a GPU.
"""

from __future__ import annotations

import pytest

from nnunet_inference_mlx.store import ModelStore, _resolve_model_root_dir


# ---------------------------------------------------------------------------
# Fakes + synthetic trees
# ---------------------------------------------------------------------------


class FakeLoadedModel:
    def __init__(self, id, memory_mb):
        self.id = id
        self.memory_mb = memory_mb
        self.closed = False

    def close(self):
        self.closed = True


def _nnunet_tree(tmp_path, ids):
    root = tmp_path / "models"
    for i in ids:
        (root / f"Dataset{i}_X" / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir(parents=True)
    return root


def _moose_tree(tmp_path, names):
    root = tmp_path / "moose"
    for n in names:
        (root / n / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir(parents=True)
    return root


def _fake_read_nnunet(folder, **kw):
    # folder = .../Dataset{ID}_X/config  → recover the int id
    ds = folder.parent.name
    return int(ds[len("Dataset"):].split("_", 1)[0])


def _store(root, sizes, *, max_memory_mb=10_000, build_calls=None):
    def fake_build(artifact, opts):
        if build_calls is not None:
            build_calls.append(artifact)
        return FakeLoadedModel(artifact, sizes.get(artifact, 100.0))
    return ModelStore("nnunet", model_root_dir=root, max_memory_mb=max_memory_mb,
                      read=_fake_read_nnunet, build=fake_build)


# ---------------------------------------------------------------------------
# Construction / resolution / precedence
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_unknown_ecosystem_raises(self):
        with pytest.raises(ValueError, match="unknown ecosystem"):
            ModelStore("bogus")

    def test_explicit_root_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("nnUNet_results", "/env/path")
        assert _resolve_model_root_dir("nnunet", str(tmp_path)) == tmp_path

    def test_env_var_used_when_no_explicit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("nnUNet_results", str(tmp_path))
        assert _resolve_model_root_dir("nnunet", None) == tmp_path

    def test_totalsegmentator_default(self, monkeypatch):
        monkeypatch.delenv("TOTALSEG_WEIGHTS_PATH", raising=False)
        got = _resolve_model_root_dir("totalsegmentator", None)
        assert got is not None and got.name == "results"

    def test_missing_root_raises_on_use(self, monkeypatch):
        monkeypatch.delenv("nnUNet_results", raising=False)
        store = ModelStore("nnunet", read=_fake_read_nnunet, build=lambda a, o: None)
        store.model_root_dir = None
        with pytest.raises(FileNotFoundError, match="model_root_dir"):
            store.get(297)


# ---------------------------------------------------------------------------
# Cold layer: get / downloaded / delete_downloads
# ---------------------------------------------------------------------------


class TestColdLayer:
    def test_get_resolves_and_reads(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        store = _store(root, {})
        assert store.get(297) == 297  # fake read returns the id marker

    def test_get_missing_raises(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        store = _store(root, {})
        with pytest.raises(FileNotFoundError, match="Dataset999"):
            store.get(999)

    def test_downloaded_lists_present_ids(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297, 117])
        store = _store(root, {})
        assert store.downloaded() == [117, 297]

    def test_delete_downloads_removes_dir(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297, 117])
        store = _store(root, {})
        store.delete_downloads(297)
        assert store.downloaded() == [117]

    def test_delete_downloads_unloads_first(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        store = _store(root, {297: 100.0})
        eng = store.load(297)
        store.delete_downloads(297)
        assert eng.closed
        assert len(store) == 0

    def test_download_local_present_ok(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        store = _store(root, {})
        store.download(297)  # present → no error

    def test_download_missing_raises(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        store = _store(root, {})
        with pytest.raises(FileNotFoundError, match="no fetch is configured"):
            store.download(999)


# ---------------------------------------------------------------------------
# Hot layer: load / cache / LRU / unload
# ---------------------------------------------------------------------------


class TestHotLayer:
    def test_load_builds_and_caches(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        calls = []
        store = _store(root, {297: 100.0}, build_calls=calls)
        a = store.load(297)
        b = store.load(297)
        assert a is b                # cache hit
        assert len(calls) == 1       # built once
        assert len(store) == 1

    def test_load_list(self, tmp_path):
        root = _nnunet_tree(tmp_path, [291, 292, 293])
        store = _store(root, {291: 100, 292: 100, 293: 100})
        engs = store.load([291, 292, 293])
        assert [e.id for e in engs] == [291, 292, 293]
        assert len(store) == 3

    def test_lru_evicts_to_fit_budget(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297, 117, 298])
        store = _store(root, {297: 1000, 117: 1000, 298: 1000}, max_memory_mb=2500)
        e297 = store.load(297)
        store.load(117)
        assert store.loaded_mb == 2000
        store.load(298)              # would be 3000 > 2500 → evict oldest (297)
        assert store.loaded_mb == 2000
        ids = [i for i, _ in store.loaded()]
        assert ids == [117, 298]
        assert e297.closed           # evicted engine freed

    def test_single_engine_over_budget_degrades(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        store = _store(root, {297: 1000}, max_memory_mb=500)
        eng = store.load(297)        # 1000 > 500 but it's the only one → kept
        assert not eng.closed
        assert len(store) == 1

    def test_loaded_and_loaded_mb(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297, 117])
        store = _store(root, {297: 600, 117: 400})
        store.load([297, 117])
        assert dict(store.loaded()) == {297: 600.0, 117: 400.0}
        assert store.loaded_mb == 1000.0

    def test_unload_one(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297, 117])
        store = _store(root, {297: 100, 117: 100})
        e297 = store.load(297)
        store.load(117)
        store.unload(297)
        assert e297.closed
        assert [i for i, _ in store.loaded()] == [117]

    def test_unload_absent_is_noop(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        store = _store(root, {297: 100})
        store.unload(999)  # no error

    def test_unload_all(self, tmp_path):
        root = _nnunet_tree(tmp_path, [291, 292])
        store = _store(root, {291: 100, 292: 100})
        engs = store.load([291, 292])
        store.unload_all()
        assert all(e.closed for e in engs)
        assert len(store) == 0


# ---------------------------------------------------------------------------
# Lifecycle: context manager frees; no global
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_context_manager_frees_on_exit(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        with _store(root, {297: 100}) as store:
            eng = store.load(297)
            assert not eng.closed
        assert eng.closed
        assert len(store) == 0

    def test_two_stores_are_independent(self, tmp_path):
        root = _nnunet_tree(tmp_path, [297])
        a = _store(root, {297: 100})
        b = _store(root, {297: 100})
        a.load(297)
        assert len(a) == 1 and len(b) == 0   # no shared global


# ---------------------------------------------------------------------------
# MOOSE ecosystem (string ids)
# ---------------------------------------------------------------------------


class TestMooseEcosystem:
    def _moose_store(self, root, sizes):
        def fake_read(folder, **kw):
            return folder.parent.name   # the MOOSE folder name
        def fake_build(artifact, opts):
            return FakeLoadedModel(artifact, sizes.get(artifact, 100.0))
        return ModelStore("moose", model_root_dir=root,
                          read=fake_read, build=fake_build)

    def test_moose_get_by_folder_name(self, tmp_path):
        root = _moose_tree(tmp_path, ["Dataset123_Organs"])
        store = self._moose_store(root, {})
        assert store.get("Dataset123_Organs") == "Dataset123_Organs"

    def test_moose_downloaded(self, tmp_path):
        root = _moose_tree(tmp_path, ["Dataset123_Organs", "Dataset111_Vertebrae"])
        store = self._moose_store(root, {})
        assert store.downloaded() == ["Dataset111_Vertebrae", "Dataset123_Organs"]

    def test_moose_load(self, tmp_path):
        root = _moose_tree(tmp_path, ["Dataset123_Organs"])
        store = self._moose_store(root, {"Dataset123_Organs": 100})
        eng = store.load("Dataset123_Organs")
        assert eng.id == "Dataset123_Organs"


# ---------------------------------------------------------------------------
# download(): idempotent ensure-present + force; verify_and_unpack integrity
# ---------------------------------------------------------------------------


def _fake_fetch_factory(calls):
    def fetch(i, root):
        calls.append(i)
        (root / f"Dataset{i}_X" / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir(parents=True)
    return fetch


class TestDownload:
    def test_idempotent_skips_present(self, tmp_path):
        root = _nnunet_tree(tmp_path, [1])
        calls = []
        store = ModelStore("nnunet", model_root_dir=root, fetch=_fake_fetch_factory(calls))
        assert store.download(1) == []          # already present → no-op
        assert calls == []

    def test_fetches_only_missing(self, tmp_path):
        root = _nnunet_tree(tmp_path, [1])
        calls = []
        store = ModelStore("nnunet", model_root_dir=root, fetch=_fake_fetch_factory(calls))
        assert store.download([1, 2]) == [2]    # 1 present, 2 fetched
        assert calls == [2]
        assert 2 in store.downloaded()

    def test_force_refetches_present(self, tmp_path):
        root = _nnunet_tree(tmp_path, [1])
        calls = []
        store = ModelStore("nnunet", model_root_dir=root, fetch=_fake_fetch_factory(calls))
        assert store.download(1, force=True) == [1]
        assert calls == [1]

    def test_missing_without_fetch_raises_but_present_is_noop(self, tmp_path):
        root = _nnunet_tree(tmp_path, [1])
        store = ModelStore("nnunet", model_root_dir=root)   # no fetch configured
        assert store.download(1) == []                      # present → fine
        with pytest.raises(FileNotFoundError, match="no fetch is configured"):
            store.download(2)


class TestVerifyAndUnpack:
    def _zip(self, path, name="Dataset9_X/plans.json", data="{}"):
        import zipfile
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(name, data)
        return path

    def test_sha_match_unpacks_and_marks(self, tmp_path):
        from nnunet_inference_mlx.store import sha256_file, verify_and_unpack
        arc = self._zip(tmp_path / "m.zip")
        sha = sha256_file(arc)
        dest = tmp_path / "out"
        verify_and_unpack(arc, sha, dest)
        assert (dest / "Dataset9_X" / "plans.json").exists()
        assert (dest / ".verified").read_text() == sha

    def test_sha_mismatch_raises_and_unpacks_nothing(self, tmp_path):
        from nnunet_inference_mlx.store import verify_and_unpack
        arc = self._zip(tmp_path / "m.zip")
        dest = tmp_path / "out"
        with pytest.raises(ValueError, match="sha256 mismatch"):
            verify_and_unpack(arc, "0" * 64, dest)
        assert not dest.exists()           # refused before creating anything

    def test_no_sha_skips_verification(self, tmp_path):
        from nnunet_inference_mlx.store import verify_and_unpack
        arc = self._zip(tmp_path / "m.zip", name="f.txt", data="x")
        dest = tmp_path / "out"
        verify_and_unpack(arc, None, dest)
        assert (dest / "f.txt").exists()
        assert (dest / ".verified").read_text() == "unverified"
