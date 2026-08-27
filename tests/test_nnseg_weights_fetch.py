"""Weight provisioning: manifest coverage and the skip-if-present contract (no network)."""
import pytest

from nnseg.weights_fetch import dataset_key, selected, _manifest, ensure_task_weights, fetch_one, is_present
from nnseg.tasks import TaskCatalog


def test_manifest_covers_the_core_tasks():
    """The manifest is a partial snapshot - newer datasets may need a refresh - but the tasks
    people actually run must resolve to fetchable weights."""
    m = _manifest()
    cat = TaskCatalog("ts")
    for name in ("total", "total_fast", "total_fastest", "body", "lung_vessels"):
        spec = cat.get(name)
        ids = ([spec.single] if spec.single is not None else []) + [p.weights_id for p in spec.union]
        # cascade crop-from parts count too
        for i in ids:
            k = dataset_key(i)
            assert k in m, f"{name}: weights id {i} not in the manifest"
            assert selected(m[k]).get("url"), f"{name}: weights id {i} has no URL to fetch"


def test_every_url_present_points_at_the_official_source():
    for wid, entry in _manifest().items():
        url = entry.get("url")
        if url is not None:
            assert url.startswith("https://github.com/wasserth/TotalSegmentator/releases/"), f"{wid}: {url}"


def test_present_and_skip(tmp_path):
    assert not is_present(297, tmp_path)
    d = tmp_path / "Dataset297_TotalSegmentator_total_3mm_1559subj"
    d.mkdir()
    assert is_present(297, tmp_path)
    assert fetch_one(297, tmp_path) == d          # present -> no network


def test_unknown_id_is_a_clear_error(tmp_path):
    """ModelNotFound, not a KeyError leaking out of the manifest dict - and it says what to do."""
    from nnseg.errors import ModelNotFound
    with pytest.raises(ModelNotFound, match="weights refresh"):
        fetch_one(999999, tmp_path)


def test_total_resolves_to_five_ids(tmp_path):
    for did in (291, 292, 293, 294, 295):
        (tmp_path / f"Dataset{did}_part").mkdir()
    assert len(ensure_task_weights("total", tmp_path)) == 5
