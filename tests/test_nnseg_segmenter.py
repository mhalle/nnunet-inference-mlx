"""Segmenter policy + the warm model cache.

Both are exercised against a stub loader rather than real weights: what matters here is the
caching and policy logic, not the network, and these must stay in the fast suite.
"""
import pytest

from nnseg.segmenter import POLICY, Segmenter
from nnseg.cache import ModelCache


class _FakeModel:
    """Stands in for a TorchModel so the REAL ModelCache.get logic is what gets tested."""

    built = 0

    def __init__(self, folder, **policy):
        _FakeModel.built += 1
        self.folder, self.policy = folder, policy
        self.device = type("d", (), {"type": "cpu"})()

    def to_device(self):
        return self


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    _FakeModel.built = 0
    monkeypatch.setattr("nnseg.network.TorchModel", _FakeModel)
    yield


# -- the store --------------------------------------------------------------------------
def test_capacity_zero_never_caches():
    s = ModelCache()                       # the default: segment()'s historical behavior
    a, b = s.get("/m", folds=(0,)), s.get("/m", folds=(0,))
    assert a is not b and len(s) == 0
    assert s.hits == 0 and s.misses == 2


def test_a_warm_model_is_reused():
    s = ModelCache(capacity=2)
    a, b = s.get("/m", folds=(0,)), s.get("/m", folds=(0,))
    assert a is b and s.hits == 1 and s.misses == 1
    assert _FakeModel.built == 1           # built once, not twice


def test_policy_is_part_of_the_key_so_a_different_policy_is_a_different_model():
    """Reusing a model built under another policy would silently ignore the new one."""
    s = ModelCache(capacity=8)
    base = dict(folds=(0,), device="cpu", dtype="fp16", accumulate="auto", batch_size="auto")
    s.get("/m", **base)
    for changed in ({"dtype": "fp32"}, {"device": "cuda"}, {"folds": (1,)},
                    {"batch_size": 4}, {"accumulate": "host"}):
        s.get("/m", **{**base, **changed})
    assert s.hits == 0 and s.misses == 6   # every one is a distinct model


def test_a_different_folder_is_a_different_model():
    s = ModelCache(capacity=4)
    s.get("/a", folds=(0,)); s.get("/b", folds=(0,)); s.get("/a", folds=(0,))
    assert s.hits == 1 and len(s) == 2


def test_lru_evicts_the_least_recently_used():
    s = ModelCache(capacity=2)
    s.get("/a", folds=(0,)); s.get("/b", folds=(0,))
    s.get("/a", folds=(0,))                # /a is now the most recent
    s.get("/c", folds=(0,))                # evicts /b, not /a
    assert len(s) == 2
    s.get("/a", folds=(0,))                # still warm
    assert s.hits == 2
    s.get("/b", folds=(0,))                # was evicted, rebuilt
    assert s.misses == 4


def test_release_keeps_a_cached_model_and_drops_an_uncached_one():
    warm = ModelCache(capacity=1)
    m = warm.get("/m", folds=(0,))
    warm.release(m)                        # cached: still there
    assert len(warm) == 1 and warm.get("/m", folds=(0,)) is m
    cold = ModelCache()
    cold.release(cold.get("/m", folds=(0,)))   # uncached: no error, nothing retained
    assert len(cold) == 0


def test_clear_drops_everything():
    s = ModelCache(capacity=4)
    s.get("/a", folds=(0,)); s.get("/b", folds=(0,))
    s.clear()
    assert len(s) == 0


# -- the Segmenter ----------------------------------------------------------------------
def test_policy_is_held_and_passed_through(monkeypatch):
    seen = {}

    def fake_segment(image, task, **kw):
        seen.update(kw); return "result"

    monkeypatch.setattr("nnseg.pipeline.segment", fake_segment)
    seg = Segmenter(device="cuda", dtype="fp32", envelope_mm=None, cache_models=2)
    assert seg.segment("img.nii.gz", "total_fast") == "result"
    assert seen["device"] == "cuda" and seen["dtype"] == "fp32" and seen["envelope_mm"] is None
    assert seen["models"] is seg.models and seen["catalog"] is seg.catalog


def test_per_call_arguments_override_the_policy(monkeypatch):
    seen = {}
    monkeypatch.setattr("nnseg.pipeline.segment", lambda image, task, **kw: seen.update(kw))
    seg = Segmenter(device="cpu", interp="linear")
    seg.segment("img.nii.gz", "total_fast", interp="nearest", grid=1.5)
    assert seen["interp"] == "nearest" and seen["grid"] == 1.5
    assert seen["device"] == "cpu"                     # unchanged policy still applies


def test_an_unknown_argument_is_rejected_rather_than_silently_ignored():
    seg = Segmenter()
    with pytest.raises(TypeError, match="unknown argument"):
        seg.segment("img.nii.gz", "total_fast", devcie="cuda")   # typo


def test_callable_shorthand_is_the_same_operation(monkeypatch):
    monkeypatch.setattr("nnseg.pipeline.segment", lambda image, task, **kw: ("ran", task))
    assert Segmenter()("img.nii.gz", "total_fast") == ("ran", "total_fast")


def test_every_policy_key_is_accepted_as_an_override(monkeypatch):
    """POLICY and the constructor must not drift apart."""
    monkeypatch.setattr("nnseg.pipeline.segment", lambda image, task, **kw: kw)
    seg = Segmenter()
    assert set(POLICY) <= set(seg.policy)
    for key in POLICY:
        seg.segment("i", "t", **{key: seg.policy[key]})          # no TypeError


# -- introspection ----------------------------------------------------------------------
def test_describe_reports_a_task_without_running_it():
    seg = Segmenter()
    d = seg.describe("total_fast")
    assert d["source"] == "ts" and d["modality"] == "CT"
    assert d["n_structures"] == len(d["structures"]) > 100
    assert "liver" in d["structures"] and d["weights"] == ["297"]


def test_structures_are_in_label_order():
    seg = Segmenter()
    names = seg.structures("total_fast")
    spec = seg.catalog.get("total_fast")
    assert names == [spec.label_map[k] for k in sorted(spec.label_map)]


def test_tasks_lists_the_catalog():
    seg = Segmenter()
    assert "ts:total" in seg.tasks() and len(seg.tasks()) == len(seg.catalog)
    assert seg.resolve_task("total") == "ts:total"          # short form resolves
    assert seg.resolve_task("ts:total@v2.0.0") == "ts:total"


def test_describe_works_for_a_union_task_and_lists_every_model():
    seg = Segmenter()
    d = seg.describe("total")
    assert d["shape"] == "label_union" and len(d["weights"]) == 5


# -- the weights store -------------------------------------------------------------------
def test_a_path_is_coerced_into_a_store(tmp_path):
    from nnseg.weights import WeightsStore, as_store
    s = as_store(tmp_path)
    assert isinstance(s, WeightsStore) and s.root == tmp_path
    assert as_store(s) is s                       # a store passes through unchanged


def test_default_store_resolves_its_root_lazily_from_the_ecosystem(monkeypatch, tmp_path):
    from nnseg.weights import WeightsStore
    monkeypatch.setenv("TOTALSEG_WEIGHTS_PATH", str(tmp_path))
    s = WeightsStore()
    assert s.describe()["root"] == "(default)"    # nothing touched yet
    assert s.root == tmp_path                     # resolved on demand


def test_have_is_offline_and_resolve_finds_the_model(tmp_path):
    from nnseg.weights import WeightsStore
    d = tmp_path / "Dataset297_Total" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    d.mkdir(parents=True)
    s = WeightsStore(tmp_path, fetch=False)
    assert s.have(297) and not s.have(999)
    assert s.resolve(297) == d


def test_fetch_disabled_raises_modelnotfound_instead_of_reaching_for_the_network(tmp_path):
    from nnseg import errors
    from nnseg.weights import WeightsStore
    s = WeightsStore(tmp_path, fetch=False)
    with pytest.raises(errors.ModelNotFound, match="fetching disabled"):
        s.resolve(297)


def test_a_model_folder_path_bypasses_the_store_layout(tmp_path):
    """A caller with a model on disk should never have to arrange it into a store's layout."""
    from nnseg.weights import WeightsStore
    d = tmp_path / "anywhere" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    d.mkdir(parents=True)
    s = WeightsStore(tmp_path / "elsewhere", fetch=False)
    assert s.resolve(d) == d


def test_no_download_source_for_nnunet_says_so_clearly(tmp_path):
    from nnseg import errors
    from nnseg.weights import WeightsStore
    s = WeightsStore(tmp_path, ecosystem="nnunet")
    with pytest.raises(errors.ModelNotFound, match="no download source|has no download source"):
        s.resolve(500)


def test_a_subclass_can_fetch_from_somewhere_else(tmp_path):
    """The override point: where weights come from is the store's business."""
    from nnseg.weights import WeightsStore

    class Pretend(WeightsStore):
        def fetch(self, weights_id):
            d = self.root / f"Dataset{weights_id}_Made" / "nnUNetTrainer__p__3d_fullres"
            d.mkdir(parents=True)
            return d

    s = Pretend(tmp_path)
    assert not s.have(42)
    assert s.resolve(42).name == "nnUNetTrainer__p__3d_fullres"
    assert s.have(42)                             # now cached locally


def test_segmenter_holds_a_store_and_passes_it_through(monkeypatch, tmp_path):
    from nnseg.weights import WeightsStore
    seen = {}
    monkeypatch.setattr("nnseg.pipeline.segment", lambda image, task, **kw: seen.update(kw))
    seg = Segmenter(weights=tmp_path)
    assert isinstance(seg.weights, WeightsStore)
    seg.segment("i", "total_fast")
    assert seen["weights"] is seg.weights
