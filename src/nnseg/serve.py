"""The REST job protocol over a Segmenter - run segmentations for remote callers.

This is the seam the toolkit's remote consumers share (docs/slicer-modal-design.md in
the medseg workspace): a 3D Slicer panel, a CLI on another machine, a notebook. The
contract is deliberately small:

    GET    /v1/health              who am I - version, device, task count
    GET    /v1/tasks               task names
    GET    /v1/tasks/{task}        describe: structures, modality, weights
    POST   /v1/jobs                task + input (+ options JSON) -> {id}; the input is a
                                   one-element `source` list - {"kind": "upload"} (the
                                   default; multipart part "file") or {"kind": "idc",
                                   "crdc_series_uuid": ...} (the server fetches from
                                   the public IDC buckets; needs the idc extra). A bare
                                   multipart file remains valid shorthand forever. Kind
                                   "url" is reserved for the authenticated tier.
    GET    /v1/jobs                brief status of every known job
    GET    /v1/jobs/{id}           full status; includes result metadata once done
    GET    /v1/jobs/{id}/events    Server-Sent Events: status snapshots until terminal
    GET    /v1/jobs/{id}/result    the label volume (.nii.gz)
    DELETE /v1/jobs/{id}           cancel an active job / delete a finished one

Jobs run through :class:`LocalExecutor`: a **bounded FIFO** ahead of one dispatcher
thread per server. The per-device lock in :mod:`nnseg.job` already makes concurrent
runs safe; what it does not give is order (lock wake-ups are not FIFO), bounds,
introspection, or instant cancel-while-queued - the queue exists for those four
things and nothing more. It is in-memory by design: a restart drops queued jobs, and
re-submission is cheap for the caller. Progress events are idempotent state
snapshots, which is what makes the SSE stream robust - a dropped connection needs no
replay, just a resubscribe or a fall back to polling the same JSON.

Task names at this boundary are catalog names only - the in-process API's freedom to
take a model-folder path does not cross the wire, so a served deployment never treats
server-readable directories as runnable tasks. ``POST /v1/tasks`` is reserved for a
future privileged install/register endpoint (weights by URL or upload into a
content-addressed store); nothing else may squat on that route, and no weight fetching
by URL happens anywhere in serve until it lands with its own auth.

Every input resolves to a content identity - an upload to its sha256, an IDC series to
its immutable crdc UUID (the reserved ``series_instance_uid`` form, with an optional
``idc_version`` defaulting to the latest data release, will resolve to one at submit) -
and the job's ``input_identity`` is the ordered tuple of them:
the key a future result cache will use. IDC fetches happen at dispatch, as a visible
"fetch" progress stage, so submits stay tiny and a 429 never wastes an upload.

FastAPI and uvicorn live behind the ``serve`` extra; nothing here imports them until
:func:`create_app` runs. No authentication: this server is for localhost and trusted
LANs. The authenticated deployment is the Modal one, where the platform supplies
both the queue (spawn + autoscaler) and the auth (proxy tokens).
"""
import json
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import Cancelled, InputError, NnsegError
from .progress import CancelToken, Reporter

ACTIVE = ("queued", "running")
TERMINAL = ("done", "failed", "cancelled")
RESULT_NAME = "labels.seg.nrrd"          # the information-preserving default artifact
CRDC_RE = r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}"


def result_key(identity, task, options, weights_versions, version=None) -> str:
    """The result-cache key: everything that determines the output bytes.

    (input identity) x (task + options) x (weights versions) x (nnseg version) -
    the design's cache contract. Over-keying on an option that turns out inert
    only costs hits; under-keying would serve wrong bytes, so all options count.
    """
    import hashlib
    payload = json.dumps({"identity": list(identity), "task": str(task),
                          "options": {k: options[k] for k in sorted(options)},
                          "weights": list(weights_versions),
                          "nnseg": version or _version()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def weights_versions_of(segmenter, task) -> list:
    """The key's model component, from the install sidecars via describe().
    "unknown" when nothing better exists - documented, never guessed."""
    try:
        entries = segmenter.describe(task).get("weights_installed") or []
        out = [f"{e.get('id')}={e.get('version') or e.get('sha256') or 'unknown'}"
               for e in entries]
        return out or ["unknown"]
    except Exception:
        return ["unknown"]


class ResultCache:
    """Content-keyed store of finished results: <root>/<key>/labels.seg.nrrd +
    result.json + meta.json (the readable key components - a cache you can ls).
    LRU by directory mtime, count-bounded; results are ~MBs so the bound is
    generous by default."""

    def __init__(self, root, *, keep: int = 500):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = int(keep)

    def get(self, key: str):
        d = self.root / key
        labels = d / RESULT_NAME
        if not labels.exists():
            return None
        try:
            import os as _os
            _os.utime(d)                       # LRU touch
            result = json.loads((d / "result.json").read_text())
        except (OSError, json.JSONDecodeError):
            result = {}
        return labels, result

    def put(self, key: str, labels_path, result: dict, meta: dict) -> None:
        import shutil
        d = self.root / key
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(labels_path, d / RESULT_NAME)
        (d / "result.json").write_text(json.dumps(result))
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        self.evict()

    def delete(self, key: str) -> bool:
        """Remove one entry; True if it existed."""
        import shutil
        d = self.root / key
        if not d.is_dir():
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True

    def evict(self) -> None:
        import shutil
        dirs = [d for d in self.root.iterdir() if d.is_dir()]
        if len(dirs) <= self.keep:
            return
        for d in sorted(dirs, key=lambda x: x.stat().st_mtime)[: len(dirs) - self.keep]:
            shutil.rmtree(d, ignore_errors=True)


class QueueFull(NnsegError):
    """The pending queue is at its bound; the caller should retry later."""


@dataclass
class JobRecord:
    """Everything the server knows about one job. Mutated only under the executor's
    condition variable, except ``progress`` which is an atomic rebind (same contract
    as :class:`nnseg.job.Job`)."""

    id: str
    task: str
    options: dict
    dir: Path
    input_path: Path
    state: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict | None = None
    error: str | None = None
    result: dict | None = None                    # names / volumes_ml / provenance
    labels_path: Path | None = None
    source: list = field(default_factory=list)
    input_identity: tuple = ()
    cached: bool = False
    cache_key: str | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)
    subscribers: list = field(default_factory=list)   # (event_loop, asyncio.Queue)


class LocalExecutor:
    """A bounded FIFO of jobs ahead of one dispatcher thread.

    ``segmenter`` supplies the task catalog and, by default, the work itself;
    ``segment_fn`` overrides the work for tests (same signature as
    ``Segmenter.segment``). ``max_pending`` bounds the queue - submit past it raises
    :class:`QueueFull` (HTTP 429). Finished jobs and their files are kept until
    ``keep_finished`` more finish after them.
    """

    def __init__(self, segmenter, *, workdir, max_pending: int = 16,
                 keep_finished: int = 50, segment_fn=None, fetch_idc_fn=None,
                 cache_dir=None, keep_cached: int = 500):
        self.segmenter = segmenter
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.max_pending = int(max_pending)
        self.keep_finished = int(keep_finished)
        self._segment = segment_fn or segmenter.segment
        self._fetch_idc = fetch_idc_fn or _fetch_idc_series
        self.cache = ResultCache(cache_dir, keep=keep_cached) if cache_dir else None
        self._inflight: dict[str, str] = {}      # cache key -> active job id
        self._cv = threading.Condition()
        self._pending: deque[str] = deque()
        self._jobs: dict[str, JobRecord] = {}
        self._done_order: deque[str] = deque()
        self._stop = False
        self._thread = threading.Thread(target=self._dispatch, name="nnseg-serve", daemon=True)
        self._thread.start()

    # -- intake --------------------------------------------------------------
    def new_job_dir(self) -> tuple[str, Path]:
        jid = uuid.uuid4().hex[:12]
        d = self.workdir / jid
        d.mkdir(parents=True, exist_ok=True)
        return jid, d

    @property
    def accepting(self) -> bool:
        with self._cv:
            return len(self._pending) < self.max_pending

    def submit(self, jid: str, jdir: Path, input_path, task: str, options: dict,
               *, source=None, identity: tuple = (), no_cache: bool = False) -> JobRecord:
        rec = JobRecord(id=jid, task=task, options=options, dir=jdir, input_path=input_path,
                        source=list(source or [{"kind": "upload"}]), input_identity=tuple(identity))
        if self.cache is not None and identity:
            rec.cache_key = result_key(identity, task, options,
                                       weights_versions_of(self.segmenter, task))
            if not no_cache:
                hit = self.cache.get(rec.cache_key)
                if hit is not None:
                    rec.labels_path, rec.result = Path(hit[0]), hit[1]
                    rec.state, rec.cached = "done", True
                    rec.started = rec.finished = time.time()
                    with self._cv:
                        self._jobs[jid] = rec
                        self._done_order.append(jid)
                    self._emit(rec)
                    self._evict()
                    return rec
        with self._cv:
            if len(self._pending) >= self.max_pending:
                raise QueueFull(f"queue is full ({self.max_pending} pending)")
            self._jobs[jid] = rec
            self._pending.append(jid)
            if rec.cache_key:
                self._inflight[rec.cache_key] = jid
            self._cv.notify()
        self._emit(rec)
        return rec

    # -- cache face (shared by the path surface and the public tier) ---------
    def cache_get(self, key: str):
        return self.cache.get(key) if self.cache is not None else None

    def find_inflight(self, key: str):
        with self._cv:
            jid = self._inflight.get(key)
            rec = self._jobs.get(jid) if jid else None
        return jid if rec is not None and rec.state in ACTIVE else None

    def cache_delete(self, key: str) -> bool:
        return self.cache.delete(key) if self.cache is not None else False

    # -- introspection -------------------------------------------------------
    def get(self, jid: str) -> JobRecord | None:
        with self._cv:
            return self._jobs.get(jid)

    def jobs(self) -> list[JobRecord]:
        with self._cv:
            return list(self._jobs.values())

    def position(self, jid: str) -> int | None:
        """0 = next to run; None once no longer queued."""
        with self._cv:
            try:
                return list(self._pending).index(jid)
            except ValueError:
                return None

    # -- control -------------------------------------------------------------
    def cancel(self, jid: str):
        """Cancel an active job; delete a finished one.

        Returns ``(state, deleted)`` - ``deleted`` is True only when a finished
        job's record and files were actually removed, never for a transition into
        ``cancelled``. ``(None, False)`` for an unknown job."""
        with self._cv:
            rec = self._jobs.get(jid)
            if rec is None:
                return None, False
            if rec.state == "queued":
                self._pending.remove(jid)
                rec.state, rec.finished = "cancelled", time.time()
                self._done_order.append(jid)
                if rec.cache_key:
                    self._inflight.pop(rec.cache_key, None)
                state = rec.state
            elif rec.state == "running":
                rec.cancel_token.cancel()      # honored at the next patch boundary
                state = rec.state
            else:
                self._jobs.pop(jid)
                self._rm(rec)
                return rec.state, True
        if state == "cancelled":
            self._emit(rec)
            self._requeue_positions()
        return state, False

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # -- SSE plumbing --------------------------------------------------------
    def subscribe(self, jid: str, loop, q) -> bool:
        with self._cv:
            rec = self._jobs.get(jid)
            if rec is None:
                return False
            rec.subscribers.append((loop, q))
            return True

    def unsubscribe(self, jid: str, loop, q) -> None:
        with self._cv:
            rec = self._jobs.get(jid)
            if rec is not None and (loop, q) in rec.subscribers:
                rec.subscribers.remove((loop, q))

    def _emit(self, rec: JobRecord) -> None:
        snap = self.status(rec)
        for loop, q in list(rec.subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, snap)
            except RuntimeError:               # loop already closed
                pass

    def _requeue_positions(self) -> None:
        with self._cv:
            queued = [self._jobs[j] for j in self._pending]
        for rec in queued:
            self._emit(rec)

    # -- the dispatcher ------------------------------------------------------
    def _dispatch(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                jid = self._pending.popleft()
                rec = self._jobs[jid]
                rec.state, rec.started = "running", time.time()
            self._emit(rec)
            self._requeue_positions()
            try:
                reporter = Reporter.of(progress=lambda p, r=rec: self._on_progress(r, p),
                                       cancel=rec.cancel_token)
                src = rec.source[0] if rec.source else {"kind": "upload"}
                if src.get("kind") == "idc":
                    reporter.stage("fetch", str(src.get("crdc_series_uuid", ""))[:13])
                    rec.input_path = Path(self._fetch_idc(src["crdc_series_uuid"], rec.dir))
                    reporter.check()
                seg = self._segment(rec.input_path, rec.task, progress=reporter,
                                    cancel=rec.cancel_token, **rec.options)
                rec.labels_path = Path(seg.save(rec.dir / RESULT_NAME))
                rec.result = {
                    "names": {int(k): v for k, v in seg.schema.names.items()},
                    "volumes_ml": {k: round(float(v), 2) for k, v in seg.volumes_ml().items()},
                    "provenance": seg.provenance,
                }
                (rec.dir / "result.json").write_text(json.dumps(rec.result))
                rec.state = "done"
                if self.cache is not None and rec.cache_key:
                    self.cache.put(rec.cache_key, rec.labels_path, rec.result,
                                   {"identity": list(rec.input_identity), "task": rec.task,
                                    "options": rec.options, "computed": rec.started,
                                    "job": rec.id})
            except Cancelled:
                rec.state = "cancelled"
            except Exception as e:             # noqa: BLE001 - reported to the client
                rec.state = "failed"
                rec.error = f"{type(e).__name__}: {e}"
            finally:
                rec.finished = time.time()
                with self._cv:
                    self._done_order.append(rec.id)
                    if rec.cache_key:
                        self._inflight.pop(rec.cache_key, None)
                self._emit(rec)
                self._evict()

    def _on_progress(self, rec: JobRecord, p) -> None:
        rec.progress = asdict(p)
        self._emit(rec)

    # -- housekeeping --------------------------------------------------------
    def _evict(self) -> None:
        with self._cv:
            victims = []
            while len(self._done_order) > self.keep_finished:
                jid = self._done_order.popleft()
                rec = self._jobs.pop(jid, None)
                if rec is not None:
                    victims.append(rec)
        for rec in victims:
            self._rm(rec)

    @staticmethod
    def _rm(rec: JobRecord) -> None:
        import shutil
        shutil.rmtree(rec.dir, ignore_errors=True)

    # -- the executor protocol create_app depends on -------------------------
    # (ModalExecutor in modal_app.py implements the same surface over Modal
    # primitives: new_job_dir, accepting, submit, status_of, statuses, cancel,
    # result_file, and supports_push=False for the SSE poll branch.)
    supports_push = True

    def status_of(self, jid: str) -> dict | None:
        rec = self.get(jid)
        return None if rec is None else self.status(rec)

    def statuses(self) -> list[dict]:
        return [self.status(r, brief=True) for r in self.jobs()]

    def result_file(self, jid: str):
        """(state, labels path or None); (None, None) for an unknown job."""
        rec = self.get(jid)
        if rec is None:
            return None, None
        return rec.state, rec.labels_path

    # -- serialization -------------------------------------------------------
    def status(self, rec: JobRecord, *, brief: bool = False) -> dict:
        d = {"id": rec.id, "task": rec.task, "state": rec.state,
             "created": rec.created, "started": rec.started, "finished": rec.finished}
        if rec.state == "queued":
            d["queue_position"] = self.position(rec.id)
        if rec.progress is not None:
            d["progress"] = rec.progress
        if rec.error is not None:
            d["error"] = rec.error
        if not brief and rec.input_identity:
            d["input_identity"] = list(rec.input_identity)
        if rec.cached:
            d["cached"] = True
        if not brief and rec.state == "done" and rec.result is not None:
            d["result"] = rec.result
        return d


# The three public IDC buckets, probed in order (idc-open-data holds 99.5 % of
# series). If IDC ever adds a bucket, series in it will fail loudly below; the
# upgrade path is resolving per series via idc-index (`series_aws_url`), which we
# deliberately keep OUT of the server tier to stay dependency-light - see the
# 2026-08-24 three-bucket finding in the medseg design doc before re-deriving this.
IDC_BUCKETS = ("idc-open-data", "idc-open-data-two", "idc-open-data-cr")


def _idc_enabled() -> bool:
    try:
        import obstore  # noqa: F401
        return True
    except ImportError:
        return False


def _fetch_idc_series(series: str, jobdir: Path) -> Path:
    """Fetch one IDC series into the job directory, anonymously, 32 threads.

    IDC spreads series across three public buckets (found the hard way 2026-08-24),
    so the prefix is probed in order rather than assumed. No client-supplied URLs -
    the series UUID is the whole reference, which is what keeps this SSRF-free.
    """
    from concurrent.futures import ThreadPoolExecutor

    from obstore.store import S3Store
    keys, store = [], None
    for bucket in IDC_BUCKETS:
        store = S3Store.from_url(f"s3://{bucket}", config={"aws_skip_signature": "true"})
        keys = [(o.get("path") if isinstance(o, dict) else str(o))
                for b in store.list(prefix=f"{series}/") for o in b]
        if keys:
            break
    if not keys:
        raise InputError(f"no objects under {series!r}/ in any probed IDC bucket "
                         f"({', '.join(IDC_BUCKETS)}); if the series exists, IDC may "
                         "have added a bucket this server does not know")
    dest = jobdir / "series"
    dest.mkdir(exist_ok=True)

    def one(k):
        with open(dest / k.rsplit("/", 1)[-1], "wb") as f:
            f.write(bytes(store.get(k).bytes()))

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, keys))
    return dest


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def _prefer_wait(request, default: float, maximum: float) -> float:
    """RFC 7240: Prefer: wait=N / respond-async. Never in the query string -
    the URL stays the pure cache key, and 200s never Vary on Prefer."""
    for raw in request.headers.getlist("prefer"):
        for token in raw.split(","):
            token = token.strip().lower()
            if token == "respond-async":
                return 0.0
            if token.startswith("wait="):
                try:
                    return max(0.0, min(float(token[5:]), maximum))
                except ValueError:
                    pass
    return default


def create_app(executor: LocalExecutor, *, token: str | None = None,
               wait_default: float = 30.0, wait_max: float = 110.0):
    """The FastAPI app over an executor. Import cost lives here, behind the
    ``serve`` extra.

    With ``token`` set, unauthenticated requests get the public subset only:
    health, tasks, and cache *hits* on the /v1/idc path surface - never a
    compute. That is the local single-server form of the tiering the Modal
    deployment expresses as two functions (proxy-auth'd api + read-only twin).
    """
    import asyncio

    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

    seg = executor.segmenter
    app = FastAPI(title="nnseg", version=_version())

    def authed(request) -> bool:
        if token is None:
            return True
        return request.headers.get("authorization", "") == f"Bearer {token}"

    def require_auth(request) -> None:
        if not authed(request):
            raise HTTPException(401, "this server requires a bearer token for "
                                     "anything beyond cached reads")

    @app.get("/v1/health")
    def health():
        policy = getattr(seg, "policy", {})
        return {"name": "nnseg", "version": _version(),
                "device": str(policy.get("device", "?")),
                "n_tasks": len(seg.tasks()), "accepting": executor.accepting,
                "sources": ["upload"] + (["idc"] if _idc_enabled() else [])}

    @app.get("/v1/tasks")
    def tasks():
        return {"tasks": seg.tasks()}

    @app.get("/v1/tasks/{task}")
    def describe(task: str):
        try:
            return seg.describe(task)
        except Exception as e:
            raise HTTPException(404, f"unknown task {task!r}: {e}") from e

    @app.post("/v1/jobs", status_code=202)
    async def submit(request: Request, file: UploadFile | None = File(None),
                     task: str = Form(...), options: str = Form("{}"),
                     source: str = Form(None)):
        require_auth(request)
        try:
            opts = json.loads(options)
            if not isinstance(opts, dict):
                raise ValueError("options must be a JSON object")
            src = json.loads(source) if source else [{"kind": "upload"}]
            if not (isinstance(src, list) and all(isinstance(x, dict) for x in src)):
                raise ValueError("source must be a JSON list of objects")
        except ValueError as e:
            raise HTTPException(422, f"bad request: {e}") from e
        no_cache = bool(opts.pop("no_cache", False))
        if len(src) != 1:
            raise HTTPException(422, "multi-channel input is not yet supported over the "
                                     f"wire (got {len(src)} sources)")
        kind = src[0].get("kind", "upload")
        if kind == "url":
            raise HTTPException(422, "source kind 'url' is reserved for the "
                                     "authenticated tier")
        if kind not in ("upload", "idc"):
            raise HTTPException(422, f"unknown source kind {kind!r}")
        if kind == "idc" and not _idc_enabled():
            raise HTTPException(422, "source kind 'idc' is not enabled on this server "
                                     "(install the idc extra)")
        names = seg.tasks()
        if task not in names:              # catalog names only at the wire boundary
            raise HTTPException(404, f"unknown task {task!r}; this server offers "
                                     f"{len(names)} catalog tasks, e.g. "
                                     + ", ".join(names[:4]))
        try:                               # fail at submit where the spec already knows
            chan = (seg.describe(task) or {}).get("channel_names")
        except Exception:
            chan = None
        if chan and len(chan) > 1:
            raise HTTPException(422, f"{task!r} needs {len(chan)} input channels; "
                                     "multi-channel is not yet supported over the wire")
        if not executor.accepting:
            raise HTTPException(429, "queue is full, retry later",
                                headers={"Retry-After": "30"})
        jid, jdir = executor.new_job_dir()
        if kind == "upload":
            if file is None:
                raise HTTPException(422, "source kind 'upload' needs a multipart file "
                                         "part named 'file'")
            if src[0].get("part", "file") != "file":
                raise HTTPException(422, "only the multipart part name 'file' is "
                                         "supported for now")
            import hashlib
            h = hashlib.sha256()
            name = Path(file.filename or "input.nii.gz").name
            input_path = jdir / f"input_{name}"
            with open(input_path, "wb") as f:
                while chunk := await file.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
            identity = (f"sha256:{h.hexdigest()}",)
        else:
            if file is not None:
                raise HTTPException(422, "unexpected file upload with an 'idc' source")
            # Explicit identifier fields, no format sniffing: the two IDC
            # identifiers MEAN different things. crdc_series_uuid is the storage
            # identity - version-pinned, immutable, cache-key-grade. A DICOM
            # SeriesInstanceUID names the series in DICOM space and can resolve to
            # different crdc uuids across IDC data releases; accepting it is a
            # resolution step, which is /v1/resolve's job when it lands.
            if "series_instance_uid" in src[0]:
                raise HTTPException(422, "series_instance_uid (+ optional idc_version, "
                                         "default latest) is not supported yet; "
                                         "resolution arrives with /v1/resolve - "
                                         "today pass crdc_series_uuid")
            if "series" in src[0]:
                raise HTTPException(422, "ambiguous field 'series': be explicit - "
                                         "crdc_series_uuid (IDC storage id, "
                                         "8-4-4-4-12 hex), or series_instance_uid "
                                         "(+ optional idc_version) once /v1/resolve "
                                         "lands")
            if "idc_version" in src[0]:
                raise HTTPException(422, "idc_version goes with series_instance_uid; "
                                         "a crdc_series_uuid is already pinned to one "
                                         "IDC data release")
            series = str(src[0].get("crdc_series_uuid") or "").strip().lower()
            if not series:
                raise HTTPException(422, "an 'idc' source needs crdc_series_uuid")
            if not re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", series):
                raise HTTPException(422, f"{series!r} is not a crdc_series_uuid "
                                         "(expected 8-4-4-4-12 hex; a dotted value "
                                         "would be a DICOM SeriesInstanceUID, which "
                                         "needs /v1/resolve)")
            input_path, identity = None, (f"idc:{series}",)
        try:
            executor.submit(jid, jdir, input_path, task, opts,
                            source=src, identity=identity, no_cache=no_cache)
        except QueueFull as e:
            import shutil
            shutil.rmtree(jdir, ignore_errors=True)
            raise HTTPException(429, str(e), headers={"Retry-After": "30"}) from e
        return executor.status_of(jid)

    @app.get("/v1/jobs")
    def jobs(request: Request):
        require_auth(request)
        return {"jobs": executor.statuses()}

    def _status_or_404(jid: str) -> dict:
        s = executor.status_of(jid)
        if s is None:
            raise HTTPException(404, f"no job {jid!r}")
        return s

    @app.get("/v1/jobs/{jid}")
    def status(request: Request, jid: str):
        require_auth(request)
        return _status_or_404(jid)

    @app.get("/v1/jobs/{jid}/events")
    async def events(request: Request, jid: str):
        require_auth(request)
        first = _status_or_404(jid)
        push = bool(getattr(executor, "supports_push", False))
        loop = q = None
        if push:
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()
            executor.subscribe(jid, loop, q)

        def sse(payload: dict) -> str:
            return f"event: status\ndata: {json.dumps(payload)}\n\n"

        async def stream():
            snap = first
            quiet = time.time()
            try:
                yield sse(snap)
                while snap["state"] not in TERMINAL:
                    if push:
                        try:
                            snap = await asyncio.wait_for(q.get(), timeout=15.0)
                            yield sse(snap)
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                    else:                       # poll branch: Modal, or any pushless executor
                        await asyncio.sleep(0.7)
                        nxt = executor.status_of(jid)
                        if nxt is None:
                            break
                        if nxt != snap:
                            snap = nxt
                            quiet = time.time()
                            yield sse(snap)
                        elif time.time() - quiet > 15:
                            quiet = time.time()
                            yield ": keepalive\n\n"
            finally:
                if push:
                    executor.unsubscribe(jid, loop, q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/v1/jobs/{jid}/result")
    def result(request: Request, jid: str, format: str = None):
        require_auth(request)
        state, path = executor.result_file(jid)
        if state is None:
            raise HTTPException(404, f"no job {jid!r}")
        if state != "done" or path is None:
            raise HTTPException(409, f"job is {state}, not done")
        task_name = (executor.status_of(jid) or {}).get("task", "labels")
        if format in ("nii.gz", "nii"):        # the LOSSY conversion, by request only
            import tempfile

            import SimpleITK as sitk
            out = Path(tempfile.mkdtemp(prefix="nnseg-conv-")) / f"{task_name}_{jid}.nii.gz"
            sitk.WriteImage(sitk.ReadImage(str(path)), str(out), True)
            return FileResponse(out, media_type="application/gzip", filename=out.name)
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"{task_name}_{jid}.seg.nrrd")

    @app.delete("/v1/jobs/{jid}")
    def cancel(request: Request, jid: str):
        require_auth(request)
        state, deleted = executor.cancel(jid)
        if state is None:
            raise HTTPException(404, f"no job {jid!r}")
        if deleted:
            return {"id": jid, "deleted": True, "state": state}
        return {"id": jid, "cancelling": state not in TERMINAL, "state": state}


    # -- the IDC path surface: results addressed like the source data ---------
    def _resource_headers(key: str) -> dict:
        return {"Cache-Control": "public, max-age=3600", "ETag": f'"{key[:32]}"'}

    # Registered BEFORE the GET: Starlette auto-adds HEAD to GET routes, which
    # would run the full handler - a `curl -I` on a miss would start a GPU job.
    # HEAD is the compute-free probe, and the one place status codes distinguish
    # in-flight: 200 materialized / 202 computing (authorized view) / 404 absent.
    @app.head("/v1/idc/{uuid}/{task}/labels.seg.nrrd")
    def idc_probe(request: Request, uuid: str, task: str):
        from fastapi import Response
        uuid = uuid.strip().lower()
        if not re.fullmatch(CRDC_RE, uuid) or task not in seg.tasks():
            raise HTTPException(404, "unknown resource")
        key = result_key((f"idc:{uuid}",), task, {}, weights_versions_of(seg, task))
        hit = executor.cache_get(key)
        if hit is not None:
            return Response(status_code=200, headers=_resource_headers(key))
        if executor.find_inflight(key) is not None:
            # anonymous callers see this too (user decision): watching a flight
            # for public data is harmless and tells them to check back
            return Response(status_code=202,
                            headers={"Retry-After": "10",
                                     "Cache-Control": "no-store"})
        raise HTTPException(404, "not materialized")

    @app.get("/v1/idc/{uuid}/{task}/labels.seg.nrrd")
    async def idc_resource(request: Request, uuid: str, task: str):
        uuid = uuid.strip().lower()
        if not re.fullmatch(CRDC_RE, uuid):
            raise HTTPException(422, f"{uuid!r} is not a crdc_series_uuid")
        if task not in seg.tasks():
            raise HTTPException(404, f"unknown task {task!r}")
        key = result_key((f"idc:{uuid}",), task, {}, weights_versions_of(seg, task))
        hit = executor.cache_get(key)
        if hit is not None:
            return FileResponse(hit[0], media_type="application/octet-stream",
                                filename=f"{task}_{uuid[:8]}.seg.nrrd",
                                headers=_resource_headers(key))
        jid = executor.find_inflight(key)      # single flight: ride an existing run
        is_authed = authed(request)
        if not is_authed and jid is None:
            raise HTTPException(404, "not materialized; authenticated access can "
                                     "compute it")
        initiated = False
        if jid is None:                        # authed, nothing running: initiate
            if not _idc_enabled():
                raise HTTPException(404, "not materialized, and this server cannot "
                                         "fetch IDC series (idc extra not installed)")
            initiated = True
            jid, jdir = executor.new_job_dir()
            try:
                executor.submit(jid, jdir, None, task, {},
                                source=[{"kind": "idc", "crdc_series_uuid": uuid}],
                                identity=(f"idc:{uuid}",))
            except QueueFull as e:
                raise HTTPException(429, str(e), headers={"Retry-After": "30"}) from e
        wait = _prefer_wait(request, wait_default, wait_max)
        deadline = time.time() + wait
        snap = executor.status_of(jid) or {}
        while snap.get("state") not in TERMINAL and time.time() < deadline:
            await asyncio.sleep(0.5)
            snap = executor.status_of(jid) or {}
        if snap.get("state") == "done":
            hit = executor.cache_get(key)
            headers = _resource_headers(key)
            headers["Preference-Applied"] = f"wait={int(wait)}"
            src_path = hit[0] if hit else executor.result_file(jid)[1]
            return FileResponse(src_path, media_type="application/octet-stream",
                                filename=f"{task}_{uuid[:8]}.seg.nrrd", headers=headers)
        if snap.get("state") == "failed":
            if not is_authed:                  # for anonymous the resource just is not there
                raise HTTPException(404, "not materialized")
            raise HTTPException(502, f"segmentation failed: {snap.get('error')}")
        if not is_authed:                      # watcher's view: no job vocabulary
            p = snap.get("progress") or {}
            return JSONResponse({"state": "materializing",
                                 "progress": {"stage": p.get("stage"),
                                              "fraction": p.get("fraction")}},
                                status_code=202,
                                headers={"Retry-After": "10",
                                         "Cache-Control": "no-store"})
        return JSONResponse({"state": snap.get("state", "queued"), "job": jid,
                             "initiated": initiated,       # did THIS request start it?
                             "progress": snap.get("progress")},
                            status_code=202,
                            headers={"Retry-After": "10", "Cache-Control": "no-store"})

    @app.delete("/v1/idc/{uuid}/{task}")
    @app.delete("/v1/idc/{uuid}/{task}/labels.seg.nrrd")
    def idc_delete(request: Request, uuid: str, task: str):
        """Evict the cached entry (authorized only). Also cancels any in-flight
        single-flight compute for the same key - otherwise the entry would
        repopulate moments after being cleared. DELETE + GET = recompute with
        whatever is installed now; the jobs API's no_cache is the per-job form."""
        require_auth(request)
        uuid = uuid.strip().lower()
        if not re.fullmatch(CRDC_RE, uuid) or task not in seg.tasks():
            raise HTTPException(404, "unknown resource")
        key = result_key((f"idc:{uuid}",), task, {}, weights_versions_of(seg, task))
        jid = executor.find_inflight(key)
        if jid is not None:
            executor.cancel(jid)
        deleted = executor.cache_delete(key)
        if not deleted and jid is None:
            raise HTTPException(404, "not materialized")
        out = {"deleted": deleted}
        if jid is not None:
            out["cancelled_job"] = jid
        return out

    @app.get("/v1/idc/{uuid}/{task}/meta.json")
    def idc_meta(uuid: str, task: str):
        uuid = uuid.strip().lower()
        if not re.fullmatch(CRDC_RE, uuid) or task not in seg.tasks():
            raise HTTPException(404, "unknown resource")
        key = result_key((f"idc:{uuid}",), task, {}, weights_versions_of(seg, task))
        hit = executor.cache_get(key)
        if hit is None:
            raise HTTPException(404, "not materialized")
        return JSONResponse(hit[1], headers=_resource_headers(key))

    return app


def create_public_app(key_fn, cache_get, tasks_fn, inflight=None):
    """The anonymous read-only twin: cache hits and nothing else.

    Contains no compute path at all - it cannot spend a GPU cent by
    construction, which is the right worst case for a public face. Misses are
    404 (the resource genuinely does not exist yet) with a hint body - unless
    ``inflight(key)`` reports an authorized computation in progress, in which
    case the caller gets 202 + Retry-After (and may long-poll it with
    Prefer: wait): watching a flight is read-only; only starting one needs auth.
    """
    import asyncio

    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="nnseg public cache", version=_version())

    @app.get("/v1/health")
    def health():
        return {"name": "nnseg", "version": _version(), "mode": "public-cache",
                "n_tasks": len(tasks_fn())}

    @app.get("/v1/tasks")
    def tasks():
        return {"tasks": tasks_fn()}

    def _key_or_404(uuid: str, task: str):
        uuid = uuid.strip().lower()
        if not re.fullmatch(CRDC_RE, uuid) or task not in tasks_fn():
            raise HTTPException(404, "unknown resource")
        return uuid, key_fn(uuid, task)

    def _serve(hit, uuid, task, key):
        return FileResponse(hit[0], media_type="application/octet-stream",
                            filename=f"{task}_{uuid[:8]}.seg.nrrd",
                            headers={"Cache-Control": "public, max-age=3600",
                                     "ETag": f'"{key[:32]}"'})

    @app.head("/v1/idc/{uuid}/{task}/labels.seg.nrrd")
    def probe(uuid: str, task: str):
        uuid, key = _key_or_404(uuid, task)
        if cache_get(key) is not None:
            return Response(status_code=200)
        if inflight is not None and inflight(key):
            return Response(status_code=202, headers={"Retry-After": "10"})
        raise HTTPException(404, "not materialized")

    @app.get("/v1/idc/{uuid}/{task}/labels.seg.nrrd")
    async def resource(request: Request, uuid: str, task: str):
        uuid, key = _key_or_404(uuid, task)
        hit = cache_get(key)
        if hit is not None:
            return _serve(hit, uuid, task, key)
        state = inflight(key) if inflight is not None else None
        if not state:
            raise HTTPException(404, "not materialized; authenticated access can "
                                     "compute it")
        deadline = time.time() + _prefer_wait(request, 30.0, 110.0)
        while time.time() < deadline:          # watch only - nothing here computes
            await asyncio.sleep(0.7)
            hit = cache_get(key)
            if hit is not None:
                return _serve(hit, uuid, task, key)
            state = inflight(key)
            if not state:
                break
        hit = cache_get(key)
        if hit is not None:
            return _serve(hit, uuid, task, key)
        if state:
            p = state.get("progress") or {}
            return JSONResponse({"state": "materializing",
                                 "progress": {"stage": p.get("stage"),
                                              "fraction": p.get("fraction")}},
                                status_code=202,
                                headers={"Retry-After": "10",
                                         "Cache-Control": "no-store"})
        raise HTTPException(404, "not materialized; authenticated access can "
                                 "compute it")

    @app.get("/v1/idc/{uuid}/{task}/meta.json")
    def meta(uuid: str, task: str):
        uuid = uuid.strip().lower()
        if not re.fullmatch(CRDC_RE, uuid) or task not in tasks_fn():
            raise HTTPException(404, "unknown resource")
        hit = cache_get(key_fn(uuid, task))
        if hit is None:
            raise HTTPException(404, "not materialized")
        return JSONResponse(hit[1])

    return app


def main_serve(args) -> int:
    """`nnseg serve` - build a Segmenter from the CLI arguments and run uvicorn."""
    try:
        import uvicorn
    except ImportError as e:
        raise InputError("the server needs the serve extra: uv sync --extra serve "
                         "(or pip install 'nnseg[serve]')") from e
    import tempfile

    from .segmenter import Segmenter

    import os
    seg = Segmenter(device=args.device, dtype=args.dtype, weights=args.model_root,
                    cache_models=args.cache_models)
    workdir = args.workdir or Path(tempfile.gettempdir()) / "nnseg-serve"
    cache_dir = None
    if not getattr(args, "no_result_cache", False):
        cache_dir = (getattr(args, "cache_dir", None)
                     or os.environ.get("NNSEG_CACHE_DIR")
                     or Path(os.environ.get("XDG_CACHE_HOME",
                                            Path.home() / ".cache")) / "nnseg" / "results")
    ex = LocalExecutor(seg, workdir=workdir, max_pending=args.max_pending,
                       keep_finished=args.keep_finished, cache_dir=cache_dir)
    app = create_app(ex, token=getattr(args, "token", None))
    print(f"nnseg {_version()} serving on http://{args.host}:{args.port} "
          f"(device={args.device}, workdir={workdir})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
