"""The nnseg REST protocol deployed on Modal - same contract, platform underneath.

    NNSEG_PROXY_AUTH=0 NNSEG_GPU=A10 nnseg modal deploy     # or: modal deploy <this file>

One `modal deploy` of this file gives a scale-to-zero deployment of the exact
contract `nnseg serve` speaks locally, so the client and the Slicer module cannot
tell the difference:

- an ASGI function on a cheap CPU container runs :func:`nnseg.serve.create_app` over
  a :class:`ModalExecutor` - stateless by construction, since `.spawn()` detaches the
  GPU work and all state lives in a `modal.Dict` (job metadata + progress snapshots)
  and a jobs Volume (uploads, label outputs);
- a GPU `Worker` class holds a warm ``Segmenter(cache_models=5)`` across jobs
  (``scaledown_window`` keeps it alive between a session's runs); weights
  self-provision into the shared ``nnseg-weights`` Volume on first use;
- where the LocalExecutor supplies a bounded FIFO, here **Modal is the queue**:
  spawn enqueues and the autoscaler drains up to ``NNSEG_MAX_CONTAINERS``
  (default 1: parallel requests queue and run serially on one warm worker - the
  economical posture, and every job after the first is warm; raise it at deploy
  time for cohort fan-out, up to the plan's GPU cap). ``accepting`` is always
  true - the backlog has no bound to enforce;
- progress writes are rate-limited (~4/s) because each Dict write is an RPC; the
  server's SSE endpoint reads them through its poll branch (``supports_push=False``);
- cancel is `FunctionCall.cancel()` - the container stops, billing stops;
- auth is Modal proxy auth (on by default; ``NNSEG_PROXY_AUTH=0`` to disable for a
  smoke test) - per-person tokens minted and revoked in the Modal dashboard, zero
  auth code here. The client sends them as Modal-Key / Modal-Secret headers.

Deploy-time configuration is by environment variable because Modal resolves
decorators at import: NNSEG_GPU (default L40S - wins `total` outright; A10 is the
economical fast-mode choice), NNSEG_APP_NAME, NNSEG_SCALEDOWN (seconds, default 600;
Modal caps warmth at 20 min - longer means min_containers money), NNSEG_PROXY_AUTH,
NNSEG_SNAPSHOT (memory snapshots, default ON - measured 2026-08-24: cold spawn->start
10-14 s -> 6.4-6.7 s, one 35 s snapshot-creation run per deploy).

The image mounts the *running* nnseg package (works from an editable checkout or an
installed wheel alike). TODO(release): switch to ``uv_pip_install("nnseg==<ver>")``
once published, so a deploy is pinned to a version instead of a working tree.
"""
import os
import sys
import threading
import time
from pathlib import Path

import modal

APP_NAME = os.environ.get("NNSEG_APP_NAME", "nnseg-serve")
GPU = os.environ.get("NNSEG_GPU", "L40S")
PROXY_AUTH = os.environ.get("NNSEG_PROXY_AUTH", "1") not in ("0", "false", "no")
SCALEDOWN = int(os.environ.get("NNSEG_SCALEDOWN", "600"))
GPU_SNAPSHOT = os.environ.get("NNSEG_GPU_SNAPSHOT", "0") not in ("0", "false", "no", "")
SNAPSHOT = (os.environ.get("NNSEG_SNAPSHOT", "1") not in ("0", "false", "no")) or GPU_SNAPSHOT
WARM_TASK = os.environ.get("NNSEG_WARM_TASK", "total_fast")
MAX_CONTAINERS = int(os.environ.get("NNSEG_MAX_CONTAINERS", "1"))
SHM_CACHE_GB = float(os.environ.get("NNSEG_SHM_CACHE_GB", "8"))
WEIGHTS_ROOT, JOBS_ROOT, CACHE_ROOT = "/weights", "/jobs", "/cache"
PUBLIC = os.environ.get("NNSEG_PUBLIC", "0") not in ("0", "false", "no", "")


def _pkg_dir() -> Path:
    try:
        import nnseg
    except ImportError:                      # inside the container: the mounted copy
        sys.path.insert(0, "/root/pkg")
        import nnseg
    return Path(nnseg.__file__).parent


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.7", "numpy>=1.24", "triton", "nnunetv2>=2.5",
                    "SimpleITK>=2.3", "obstore", "fastapi", "python-multipart")
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/nnseg")
)

app = modal.App(APP_NAME, image=image)
weights_vol = modal.Volume.from_name("nnseg-weights", create_if_missing=True)
jobs_vol = modal.Volume.from_name(f"{APP_NAME}-jobs", create_if_missing=True)
jobs_dict = modal.Dict.from_name(f"{APP_NAME}-jobs", create_if_missing=True)
cache_vol = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)


def _prefetch_next(current_jid: str, stop, cache, read_ahead) -> None:
    """Best-effort CPU downloader, parallel to this GPU job: watch the shared
    jobs Dict for the oldest OTHER queued idc job and stage its series into the
    /dev/shm series cache. Scans every 2 s for the length of the run - a single
    scan at job start misses jobs whose submit lands moments later (the
    warm-chain case). With max_containers=1 the same container serves the next
    input, so the staging is always warm-handed. One-ahead only; the cache owns
    claims (atomic mkdir), commits (``.done`` marker) and LRU eviction, and a
    failed staging leaves nothing behind."""
    import threading

    def scan_once():
        cands = []
        for k in jobs_dict.keys():
            if str(k).startswith("inflight:") or k == current_jid:
                continue
            m = jobs_dict.get(k) or {}
            if m.get("state") != "queued":
                continue
            src = (m.get("source") or [{}])[0]
            if src.get("kind") == "idc" and src.get("crdc_series_uuid"):
                cands.append((m.get("created", 0), src["crdc_series_uuid"]))
        return min(cands)[1] if cands else None

    def work():
        try:
            while not stop.is_set():
                series = scan_once()
                if series is None or cache.staging(series):
                    stop.wait(2.0)
                    continue
                if not cache.has(series):
                    t_f = time.time()
                    if not cache.prefetch(series):
                        stop.wait(2.0)
                        continue
                    print(f"[prefetch] {series[:13]} staged in {time.time() - t_f:.1f}s "
                          f"(parallel to {current_jid})", flush=True)
                t_r = time.time()
                if read_ahead.fill(series, cache.path(series)):
                    print(f"[read-ahead] {series[:13]} read in {time.time() - t_r:.1f}s "
                          f"(parallel to {current_jid})", flush=True)
                return                         # one-ahead only
        except Exception as e:
            print(f"[prefetch] failed: {e}", flush=True)

    threading.Thread(target=work, name="nnseg-prefetch", daemon=True).start()


def _emit(jid: str, update: dict) -> None:
    meta = jobs_dict.get(jid) or {}
    meta.update(update)
    jobs_dict[jid] = meta


_cls_extra = {"experimental_options": {"enable_gpu_snapshot": True}} if GPU_SNAPSHOT else {}


@app.cls(gpu=GPU, timeout=3600, memory=32768, scaledown_window=SCALEDOWN,
         max_containers=MAX_CONTAINERS,
         volumes={WEIGHTS_ROOT: weights_vol, JOBS_ROOT: jobs_vol, CACHE_ROOT: cache_vol},
         enable_memory_snapshot=SNAPSHOT, **_cls_extra)
class Worker:
    def _gpu_setup(self):
        os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
        from nnseg import Segmenter
        self.seg = Segmenter(device="cuda", weights=WEIGHTS_ROOT, cache_models=5)
        self._ensured = set()            # tasks whose weights this container verified

    @modal.enter(snap=SNAPSHOT)
    def preload(self):
        """The import bill, paid once per deploy: Modal snapshots memory after this
        and boots later cold containers from the snapshot. With classic snapshots
        CUDA must not be touched here - plain imports only. With the experimental
        GPU snapshot (NNSEG_GPU_SNAPSHOT=1) the CUDA state itself is captured, so
        the Segmenter is built and WARM_TASK's model loaded ONTO the GPU before the
        snapshot - a restored cold container then starts with a loaded model."""
        _pkg_dir()
        import nnunetv2  # noqa: F401
        import torch     # noqa: F401
        import nnseg     # noqa: F401 - pulls the pipeline import chain
        if GPU_SNAPSHOT:
            from nnseg.weights_fetch import ensure_task_weights
            self._gpu_setup()
            ensure_task_weights(WARM_TASK, WEIGHTS_ROOT, progress=None)
            self.seg.warm(WARM_TASK)
            self._ensured.add(WARM_TASK)

    @modal.enter()
    def setup(self):
        """Post-restore, GPU attached: everything CUDA-adjacent lives here (unless
        the GPU snapshot already carries it)."""
        _pkg_dir()
        from nnseg.serve import ReadAhead, SeriesCache, _fetch_idc_series
        self.series_cache = SeriesCache(Path("/dev/shm/series_cache"), _fetch_idc_series,
                                        budget_bytes=int(SHM_CACHE_GB * (1 << 30)))
        self.read_ahead = ReadAhead()
        if not hasattr(self, "seg"):
            self._gpu_setup()

    @modal.method()
    def run_job(self, jid: str) -> None:
        from dataclasses import asdict

        from nnseg.errors import Cancelled
        from nnseg.progress import CancelToken, Reporter
        from nnseg.serve import _fetch_idc_series
        from nnseg.weights_fetch import ensure_task_weights

        meta = jobs_dict.get(jid)
        if meta is None or meta.get("state") == "cancelled":
            return
        jdir = Path(JOBS_ROOT) / jid
        last = {"t": 0.0}

        def on_progress(p):
            now = time.time()
            if now - last["t"] >= 0.25 or p.stage in ("restore", "finalize"):
                last["t"] = now
                _emit(jid, {"progress": asdict(p)})

        token = CancelToken()
        _emit(jid, {"state": "running", "started": time.time()})
        prefetch_stop = threading.Event()
        _prefetch_next(jid, prefetch_stop, self.series_cache,
                       self.read_ahead)   # CPU downloader + pre-reader
        try:
            if meta["task"] not in self._ensured:
                # a Volume.commit scans the whole multi-GB weights tree - measured as
                # a suspect in the 41.8 s warm gap - so ensure+commit once per
                # container, not once per job
                ensure_task_weights(meta["task"], WEIGHTS_ROOT, progress=None)
                weights_vol.commit()
                self._ensured.add(meta["task"])
            src = (meta.get("source") or [{"kind": "upload"}])[0]
            if src.get("kind") == "idc":
                rep = Reporter.of(on_progress, cancel=token)
                series = src["crdc_series_uuid"]
                if self.series_cache.has(series):
                    how = "cached"
                elif self.series_cache.staging(series):
                    how = "prefetched"
                else:
                    how = "inline"
                rep.stage("fetch", series[:13] if how == "inline" else how)
                t_f = time.time()
                input_path = self.series_cache.get_or_fetch(series, check=rep.check)
                print(f"[fetch] {series[:13]} {how} {time.time() - t_f:.1f}s", flush=True)
                rep.check()
                preread = self.read_ahead.pop(series)
                if preread is not None:
                    rep.stage("read", "preread")
                    print(f"[read] {series[:13]} preread", flush=True)
                    input_path = preread
            else:
                jobs_vol.reload()
                input_path = next(jdir.glob("input_*"))
            from nnseg.serve import RESULT_NAME, ResultCache
            s = self.seg.segment(input_path, meta["task"], progress=on_progress,
                                 cancel=token, **(meta.get("options") or {}))
            s.save(jdir / RESULT_NAME)
            jobs_vol.commit()
            result = {"names": {int(k): v for k, v in s.schema.names.items()},
                      "volumes_ml": {k: round(float(v), 2)
                                     for k, v in s.volumes_ml().items()},
                      "provenance": s.provenance}
            if meta.get("cache_key"):
                ResultCache(CACHE_ROOT, keep=10 ** 6).put(
                    meta["cache_key"], jdir / RESULT_NAME, result,
                    {"identity": meta.get("input_identity"), "task": meta["task"],
                     "options": meta.get("options"), "job": jid,
                     "computed": meta.get("started")})
                cache_vol.commit()
            _emit(jid, {"state": "done", "finished": time.time(), "result": result})
        except Cancelled:
            _emit(jid, {"state": "cancelled", "finished": time.time()})
        except Exception as e:               # noqa: BLE001 - reported to the client
            _emit(jid, {"state": "failed", "finished": time.time(),
                        "error": f"{type(e).__name__}: {e}"})
        finally:
            prefetch_stop.set()            # end the scan loop with the run
            if meta.get("cache_key"):
                try:
                    del jobs_dict[f"inflight:{meta['cache_key']}"]
                except Exception:
                    pass


class ModalExecutor:
    """The :func:`nnseg.serve.create_app` executor protocol over Modal primitives."""

    supports_push = False                    # SSE uses the server's poll branch
    accepting = True                         # Modal's backlog is the queue

    def new_job_dir(self):
        import uuid
        jid = uuid.uuid4().hex[:12]
        d = Path(JOBS_ROOT) / jid
        d.mkdir(parents=True, exist_ok=True)
        return jid, d

    def submit(self, jid, jdir, input_path, task, options, *, source=None,
               identity=(), no_cache: bool = False):
        from nnseg.serve import result_key, weights_versions_of
        jobs_vol.commit()                    # make any upload visible to the worker
        key = None
        if identity:
            key = result_key(identity, task, options,
                             weights_versions_of(self.segmenter, task))
            if not no_cache:
                hit = self.cache_get(key)
                if hit is not None:
                    meta = {"id": jid, "task": task, "options": options,
                            "input_identity": list(identity), "state": "done",
                            "cached": True, "created": time.time(),
                            "started": time.time(), "finished": time.time(),
                            "result": hit[1], "cache_path": str(hit[0])}
                    jobs_dict[jid] = meta
                    return meta
        meta = {"id": jid, "task": task, "options": options,
                "source": list(source or [{"kind": "upload"}]),
                "input_identity": list(identity), "cache_key": key,
                "state": "queued", "created": time.time()}
        jobs_dict[jid] = meta
        if key:
            jobs_dict[f"inflight:{key}"] = jid
        call = Worker().run_job.spawn(jid)
        meta["call_id"] = call.object_id
        jobs_dict[jid] = meta
        return meta

    def cache_get(self, key):
        from nnseg.serve import ResultCache
        try:
            cache_vol.reload()
        except Exception:
            pass
        return ResultCache(CACHE_ROOT, keep=10 ** 6).get(key)

    def find_inflight(self, key):
        jid = jobs_dict.get(f"inflight:{key}")
        if not jid:
            return None
        meta = jobs_dict.get(jid) or {}
        return jid if meta.get("state") in ("queued", "running") else None

    def cache_delete(self, key):
        from nnseg.serve import ResultCache
        try:
            cache_vol.reload()
        except Exception:
            pass
        deleted = ResultCache(CACHE_ROOT, keep=10 ** 6).delete(key)
        if deleted:
            cache_vol.commit()
        return deleted

    def status_of(self, jid):
        meta = jobs_dict.get(jid)
        if meta is None:
            return None
        keys = ("id", "task", "state", "created", "started", "finished",
                "progress", "error", "input_identity", "cached")
        d = {k: meta.get(k) for k in keys if meta.get(k) is not None}
        if meta.get("state") == "done" and meta.get("result") is not None:
            d["result"] = meta["result"]
        return d

    def statuses(self):
        out = []
        try:
            for jid in jobs_dict.keys():
                s = self.status_of(jid)
                if s:
                    out.append({k: s.get(k) for k in
                                ("id", "task", "state", "created", "started", "finished")})
        except Exception:                    # keys() availability varies by client version
            pass
        return out

    def cancel(self, jid):
        meta = jobs_dict.get(jid)
        if meta is None:
            return None, False
        state = meta.get("state")
        if state in ("queued", "running"):
            call_id = meta.get("call_id")
            if call_id:
                try:
                    modal.FunctionCall.from_id(call_id).cancel()
                except Exception:
                    pass
            _emit(jid, {"state": "cancelled", "finished": time.time()})
            return "cancelled", False
        import shutil
        try:
            del jobs_dict[jid]
        except Exception:
            pass
        shutil.rmtree(Path(JOBS_ROOT) / jid, ignore_errors=True)
        jobs_vol.commit()
        return state, True

    def result_file(self, jid):
        from nnseg.serve import RESULT_NAME
        meta = jobs_dict.get(jid)
        if meta is None:
            return None, None
        if meta.get("cache_path"):
            p = Path(meta["cache_path"])
            try:
                cache_vol.reload()
            except Exception:
                pass
            return meta["state"], (p if p.exists() else None)
        jobs_vol.reload()
        p = Path(JOBS_ROOT) / jid / RESULT_NAME
        return meta["state"], (p if p.exists() else None)


@app.function(cpu=2.0, memory=2048, scaledown_window=300,
              volumes={JOBS_ROOT: jobs_vol, WEIGHTS_ROOT: weights_vol,
                       CACHE_ROOT: cache_vol})
@modal.concurrent(max_inputs=100)
@modal.asgi_app(requires_proxy_auth=PROXY_AUTH)
def api():
    _pkg_dir()
    os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
    from nnseg import Segmenter
    from nnseg.serve import create_app

    ex = ModalExecutor()
    # catalog/describe only - jobs run on the Worker; device string is cosmetic here
    ex.segmenter = Segmenter(device="cpu", weights=WEIGHTS_ROOT)
    return create_app(ex)


if PUBLIC:
    @app.function(cpu=1.0, memory=1024, scaledown_window=300,
                  volumes={CACHE_ROOT: cache_vol, WEIGHTS_ROOT: weights_vol})
    @modal.concurrent(max_inputs=100)
    @modal.asgi_app(requires_proxy_auth=False)
    def public():
        """The anonymous read-only twin (NNSEG_PUBLIC=1): cache hits only, no
        compute path in the function at all - it cannot spend GPU by
        construction. Shares the cache volume with the authed api."""
        _pkg_dir()
        os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
        from nnseg import Segmenter
        from nnseg.serve import (ResultCache, create_public_app, result_key,
                                 weights_versions_of)
        seg = Segmenter(device="cpu", weights=WEIGHTS_ROOT)
        cache = ResultCache(CACHE_ROOT, keep=10 ** 6)

        def key_fn(uuid, task):
            return result_key((f"idc:{uuid}",), task, {},
                              weights_versions_of(seg, task))

        def get(key):
            try:
                cache_vol.reload()
            except Exception:
                pass
            return cache.get(key)

        def inflight(key):
            jid = jobs_dict.get(f"inflight:{key}")
            if not jid:
                return None
            meta = jobs_dict.get(jid) or {}
            if meta.get("state") not in ("queued", "running"):
                return None
            return {"progress": meta.get("progress")}

        return create_public_app(key_fn, get, seg.tasks, inflight=inflight)
