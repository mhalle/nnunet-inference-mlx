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


class _Rep:
    def check(self):
        pass


def test_prefetch_marker_protocol(tmp_path, monkeypatch):
    """A prefetch directory is only readable once its .done marker lands; a
    removed directory (failed writer) ends the wait immediately."""
    import threading
    import time
    from nnseg import modal_app

    monkeypatch.setattr(modal_app, "_prefetch_base", lambda s: tmp_path / f"prefetch_{s}")

    # nothing prefetched
    assert modal_app._prefetched_or_none("u1", _Rep()) is None

    # writer mid-flight: wait resolves when the marker lands
    base = tmp_path / "prefetch_u2"
    (base / "series").mkdir(parents=True)
    threading.Timer(0.1, (base / ".done").touch).start()
    got = modal_app._prefetched_or_none("u2", _Rep())
    assert got == base / "series"

    # failed writer removed its directory: wait ends, caller fetches itself
    base3 = tmp_path / "prefetch_u3"
    base3.mkdir()

    def fail():
        time.sleep(0.1)
        base3.rmdir()

    threading.Thread(target=fail).start()
    assert modal_app._prefetched_or_none("u3", _Rep()) is None
