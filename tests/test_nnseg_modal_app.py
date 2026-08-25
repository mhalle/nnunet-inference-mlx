"""The Modal deployment module is import-safe and shaped as create_app expects."""
import pytest

modal = pytest.importorskip("modal")


def test_modal_app_imports_and_executor_matches_protocol():
    from nnseg import modal_app
    assert modal_app.app.name == modal_app.APP_NAME
    ex = modal_app.ModalExecutor
    for attr in ("new_job_dir", "submit", "status_of", "statuses", "cancel", "result_file"):
        assert callable(getattr(ex, attr)), attr
    assert ex.supports_push is False
    assert modal_app.Worker is not None


def test_worker_uses_series_cache():
    from nnseg import modal_app
    import inspect
    src = inspect.getsource(modal_app)
    assert "SeriesCache" in src
    assert "series_cache.get_or_fetch" in src


def test_purgeable_policy():
    from nnseg.modal_app import _purgeable
    now = 1000000.0
    ttl = 3600.0
    assert _purgeable({"state": "done", "finished": now - 7200}, now, ttl)
    assert _purgeable({"state": "failed", "finished": now - 7200}, now, ttl)
    assert not _purgeable({"state": "done", "finished": now - 60}, now, ttl)
    # active records are never purged by age
    assert not _purgeable({"state": "queued", "created": now - 10 ** 6}, now, ttl)
    assert not _purgeable({"state": "running", "started": now - 10 ** 6}, now, ttl)
    # garbage records are purgeable
    assert _purgeable(None, now, ttl)


def test_emit_is_terminal_wins():
    """A worker progress emit racing an API cancel must not resurrect a
    terminal record to running - the lost-cancel wedge."""
    from nnseg import modal_app

    class FakeDict(dict):
        pass

    orig = modal_app.jobs_dict
    modal_app.jobs_dict = fake = FakeDict()
    try:
        modal_app._emit("j1", {"state": "running", "started": 1.0})
        modal_app._emit("j1", {"state": "cancelled", "finished": 2.0})
        modal_app._emit("j1", {"state": "running", "progress": {"stage": "x"}})
        assert fake["j1"]["state"] == "cancelled"          # cancel survives
        modal_app._emit("j1", {"progress": {"stage": "y"}})
        assert fake["j1"]["state"] == "cancelled"          # stateless merge too
        modal_app._emit("j1", {"state": "done", "finished": 3.0})
        assert fake["j1"]["state"] == "done"               # terminal->terminal ok
    finally:
        modal_app.jobs_dict = orig
