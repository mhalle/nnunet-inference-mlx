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
