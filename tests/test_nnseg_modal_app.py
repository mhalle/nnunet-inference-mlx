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


def _swap_dict(monkeypatch):
    from nnseg import modal_app
    fake = {}
    monkeypatch.setattr(modal_app, "jobs_dict", fake)
    return modal_app, fake


def test_inflight_marker_ownership(monkeypatch):
    """Opus verification round: the marker operations, unit-reachable at
    module level. Under duplicate flights the marker names the latest job;
    installs never stomp, releases are compare-and-delete."""
    m, fake = _swap_dict(monkeypatch)
    m._install_inflight("K", "A")
    m._install_inflight("K", "B")          # refused: A owns
    assert fake["inflight:K"] == "A"
    m._release_inflight("K", "B")          # not the owner: no-op
    assert fake["inflight:K"] == "A"
    m._release_inflight("K", "A")
    assert "inflight:K" not in fake
    fake["inflight:K"] = "B"               # a NEWER submit installed directly
    m._release_inflight("K", "A")          # the older flight's finally
    assert fake["inflight:K"] == "B", "survivor's marker was clobbered"


def test_pending_marker_ownership(monkeypatch):
    m, fake = _swap_dict(monkeypatch)
    m._set_pending_marker("K", "A")
    m._set_pending_marker("K", "B")        # refuse-if-present: A keeps owning
    assert fake["artifacts:K"]["job"] == "A"
    m._clear_pending_marker("K", "B")      # not the owner: no-op
    assert "artifacts:K" in fake
    m._clear_pending_marker("K", "A")
    assert "artifacts:K" not in fake
    # legacy marker without a job field: unowned, clearable by anyone
    fake["artifacts:K"] = {"state": "pending", "t": 0}
    m._clear_pending_marker("K", "B")
    assert "artifacts:K" not in fake
    # the failure-path wrapper respects ownership too
    m._set_pending_marker("K", "A")
    m._clear_own_artifacts_marker("B", {"cache_key": "K"})
    assert "artifacts:K" in fake
    m._clear_own_artifacts_marker("A", {"cache_key": "K"})
    assert "artifacts:K" not in fake


def test_fresh_weights_versions_reloads_once(monkeypatch):
    """Opus verification round: the freshness reload converges after ONE
    volume reload and is throttled - 'unknown' is also the permanent honest
    answer for weights nnseg did not install, and the unthrottled version
    reloaded a multi-GB volume on every HEAD probe forever."""
    import time as _t

    from nnseg import modal_app

    class Vol:
        n = 0

        def reload(self):
            Vol.n += 1

    class Seg:
        def describe(self, task):
            if Vol.n:
                return {"weights_installed": [{"id": "297", "version": "v2"}]}
            return {"weights_installed": [{"id": "297"}]}   # -> unknown

    monkeypatch.setattr(modal_app, "weights_vol", Vol())
    ex = modal_app.ModalExecutor()
    ex.segmenter = Seg()
    type(ex)._weights_reloaded_at = 0.0
    type(ex)._wv_cache = {}
    wv = ex._fresh_weights_versions("t")
    assert Vol.n == 1 and not any("unknown" in v for v in wv), wv
    # within the cache window the segmenter is not even consulted (the
    # listing derives per entry - this is what keeps it cheap)
    class Boom(Seg):
        def describe(self, task):
            raise AssertionError("described during the cache window")
    ex.segmenter = Boom()
    assert ex._fresh_weights_versions("t") == wv
    # cache aged out but reload throttled: stays unknown without reloading
    class Never(Seg):
        def describe(self, task):
            return {"weights_installed": [{"id": "297"}]}
    ex.segmenter = Never()
    type(ex)._wv_cache = {}
    wv = ex._fresh_weights_versions("t")
    assert Vol.n == 1                       # throttled
    type(ex)._weights_reloaded_at = _t.time() - 60
    type(ex)._wv_cache = {}
    ex._fresh_weights_versions("t")
    assert Vol.n == 2                       # window elapsed: one more


def test_spawn_worker_rejects_fastsurfer_when_disabled(monkeypatch):
    """Dispatch: a fastsurfer: task on a deployment without the engine enabled
    fails loudly rather than routing to a nonexistent worker."""
    from nnseg import modal_app
    monkeypatch.setattr(modal_app, "FASTSURFER", False)
    with pytest.raises(RuntimeError, match="fastsurfer engine is not enabled"):
        modal_app._spawn_worker("fastsurfer:brain", "j1")


def test_spawn_worker_rejects_synthstrip_when_disabled(monkeypatch):
    """Same dispatch guard for the synthstrip: engine."""
    from nnseg import modal_app
    monkeypatch.setattr(modal_app, "SYNTHSTRIP", False)
    with pytest.raises(RuntimeError, match="synthstrip engine is not enabled"):
        modal_app._spawn_worker("synthstrip:mask", "j1")


def test_execute_job_and_hooks_exist():
    """The engine seam: a shared _execute_job + per-worker _ensure/_compute
    hooks (so adding an engine is a compute, not a run_job copy)."""
    from nnseg import modal_app
    assert callable(modal_app._execute_job) and callable(modal_app._spawn_worker)
    for h in ("_ensure", "_compute", "_prepare"):
        assert callable(getattr(modal_app.Worker, h))


def test_all_engine_workers_build_with_the_shared_base():
    """The three @app.cls workers inherit setup/_artifact_worker/run_job from
    _WorkerBase, so Modal must collect those across the MRO. CI never sets the
    engine env vars and never constructs a worker, so this runs the decorated
    class bodies in a subprocess with both engines enabled - the only local
    coverage that the inheritance actually holds together."""
    import subprocess
    import sys
    code = (
        "import nnseg.modal_app as M\n"
        "ws = M._worker_classes()\n"
        "assert sorted(ws) == ['fastsurfer', 'nnunetv2', 'synthstrip'], sorted(ws)\n"
        # every worker exposes the full hook set, inherited or not
        "for name, cls in ws.items():\n"
        "    for m in ('setup', 'preload', 'run_job', '_compute', '_prepare', '_ensure'):\n"
        "        assert hasattr(cls, m), (name, m)\n"
        # the engine shim reports the registry's identity, which is what keys the cache
        "from nnseg.engines import registry as R\n"
        "for e in ('fastsurfer', 'synthstrip'):\n"
        "    got = M._EngineShim(e).describe('x')['weights_installed']\n"
        "    assert got == R.ENGINES[e].weights_identity(), (e, got)\n"
    )
    env = {"NNSEG_FASTSURFER": "1", "NNSEG_SYNTHSTRIP": "1", "NNSEG_PROXY_AUTH": "0"}
    import os
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, **env})
    assert r.returncode == 0, r.stderr[-1500:]


def test_spawn_worker_routes_by_engine_not_by_task_prefix():
    """Dispatch is a registry lookup: an nnU-Net ecosystem (any of ts/moose/custom)
    routes to the default engine without naming any of them."""
    from nnseg import modal_app
    from nnseg.engines import registry as R
    assert R.engine_for_task("ts:total_fast").name == R.NNUNETV2
    assert R.engine_for_task("custom:mine").name == R.NNUNETV2
    assert R.engine_for_task("fastsurfer:brain").name == "fastsurfer"
    assert modal_app._WORKER_CLASSES.keys() == R.ENGINES.keys()


def test_the_two_executors_speak_the_same_submit_signature():
    """One protocol, two implementations - and the wire calls whichever it has.
    A parameter added to one and not the other is a 500 that only appears on the
    deployed side, which is exactly how `inputs` first shipped broken."""
    import inspect

    from nnseg.modal_app import ModalExecutor
    from nnseg.serve import LocalExecutor
    local = inspect.signature(LocalExecutor.submit).parameters
    modal_ = inspect.signature(ModalExecutor.submit).parameters
    assert set(local) - {"self"} <= set(modal_), \
        f"ModalExecutor.submit is missing {sorted(set(local) - set(modal_))}"
