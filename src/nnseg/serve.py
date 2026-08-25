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
                 keep_finished: int = 50, segment_fn=None, fetch_idc_fn=None):
        self.segmenter = segmenter
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.max_pending = int(max_pending)
        self.keep_finished = int(keep_finished)
        self._segment = segment_fn or segmenter.segment
        self._fetch_idc = fetch_idc_fn or _fetch_idc_series
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
               *, source=None, identity: tuple = ()) -> JobRecord:
        rec = JobRecord(id=jid, task=task, options=options, dir=jdir, input_path=input_path,
                        source=list(source or [{"kind": "upload"}]), input_identity=tuple(identity))
        with self._cv:
            if len(self._pending) >= self.max_pending:
                raise QueueFull(f"queue is full ({self.max_pending} pending)")
            self._jobs[jid] = rec
            self._pending.append(jid)
            self._cv.notify()
        self._emit(rec)
        return rec

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
    def cancel(self, jid: str) -> str | None:
        """Cancel an active job; delete a finished one. Returns the state seen."""
        with self._cv:
            rec = self._jobs.get(jid)
            if rec is None:
                return None
            if rec.state == "queued":
                self._pending.remove(jid)
                rec.state, rec.finished = "cancelled", time.time()
                self._done_order.append(jid)
                state = rec.state
            elif rec.state == "running":
                rec.cancel_token.cancel()      # honored at the next patch boundary
                state = rec.state
            else:
                self._jobs.pop(jid)
                self._rm(rec)
                return rec.state
        if state == "cancelled":
            self._emit(rec)
            self._requeue_positions()
        return state

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
                rec.labels_path = Path(seg.save(rec.dir / "labels.nii.gz"))
                rec.result = {
                    "names": {int(k): v for k, v in seg.schema.names.items()},
                    "volumes_ml": {k: round(float(v), 2) for k, v in seg.volumes_ml().items()},
                    "provenance": seg.provenance,
                }
                (rec.dir / "result.json").write_text(json.dumps(rec.result))
                rec.state = "done"
            except Cancelled:
                rec.state = "cancelled"
            except Exception as e:             # noqa: BLE001 - reported to the client
                rec.state = "failed"
                rec.error = f"{type(e).__name__}: {e}"
            finally:
                rec.finished = time.time()
                with self._cv:
                    self._done_order.append(rec.id)
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


def create_app(executor: LocalExecutor):
    """The FastAPI app over an executor. Import cost lives here, behind the
    ``serve`` extra."""
    import asyncio

    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, StreamingResponse

    seg = executor.segmenter
    app = FastAPI(title="nnseg", version=_version())

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
    async def submit(file: UploadFile | None = File(None), task: str = Form(...),
                     options: str = Form("{}"), source: str = Form(None)):
        try:
            opts = json.loads(options)
            if not isinstance(opts, dict):
                raise ValueError("options must be a JSON object")
            src = json.loads(source) if source else [{"kind": "upload"}]
            if not (isinstance(src, list) and all(isinstance(x, dict) for x in src)):
                raise ValueError("source must be a JSON list of objects")
        except ValueError as e:
            raise HTTPException(422, f"bad request: {e}") from e
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
            rec = executor.submit(jid, jdir, input_path, task, opts,
                                  source=src, identity=identity)
        except QueueFull as e:
            import shutil
            shutil.rmtree(jdir, ignore_errors=True)
            raise HTTPException(429, str(e), headers={"Retry-After": "30"}) from e
        return executor.status(rec)

    @app.get("/v1/jobs")
    def jobs():
        return {"jobs": [executor.status(r, brief=True) for r in executor.jobs()]}

    def _rec_or_404(jid: str) -> JobRecord:
        rec = executor.get(jid)
        if rec is None:
            raise HTTPException(404, f"no job {jid!r}")
        return rec

    @app.get("/v1/jobs/{jid}")
    def status(jid: str):
        return executor.status(_rec_or_404(jid))

    @app.get("/v1/jobs/{jid}/events")
    async def events(jid: str):
        rec = _rec_or_404(jid)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        executor.subscribe(jid, loop, q)

        def sse(payload: dict) -> str:
            return f"event: status\ndata: {json.dumps(payload)}\n\n"

        async def stream():
            try:
                snap = executor.status(rec)
                yield sse(snap)
                while snap["state"] not in TERMINAL:
                    try:
                        snap = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield sse(snap)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                executor.unsubscribe(jid, loop, q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/v1/jobs/{jid}/result")
    def result(jid: str):
        rec = _rec_or_404(jid)
        if rec.state != "done":
            raise HTTPException(409, f"job is {rec.state}, not done")
        return FileResponse(rec.labels_path, media_type="application/gzip",
                            filename=f"{rec.task}_{rec.id}.nii.gz")

    @app.delete("/v1/jobs/{jid}")
    def cancel(jid: str):
        state = executor.cancel(jid)
        if state is None:
            raise HTTPException(404, f"no job {jid!r}")
        if state in TERMINAL:
            return {"id": jid, "deleted": True, "state": state}
        return {"id": jid, "cancelling": True, "state": state}

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

    seg = Segmenter(device=args.device, dtype=args.dtype, weights=args.model_root,
                    cache_models=args.cache_models)
    workdir = args.workdir or Path(tempfile.gettempdir()) / "nnseg-serve"
    ex = LocalExecutor(seg, workdir=workdir, max_pending=args.max_pending,
                       keep_finished=args.keep_finished)
    app = create_app(ex)
    print(f"nnseg {_version()} serving on http://{args.host}:{args.port} "
          f"(device={args.device}, workdir={workdir})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
