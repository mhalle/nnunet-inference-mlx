"""The REST job protocol over a Segmenter - run segmentations for remote callers.

This is the seam the toolkit's remote consumers share (docs/slicer-modal-design.md in
the medseg workspace): a 3D Slicer panel, a CLI on another machine, a notebook. The
contract is deliberately small:

    GET    /v1/health              who am I - version, device, task count
    GET    /v1/version             what's deployed - contract, package + weights versions
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
    GET    /v1/inputs/{digest}     whether this server already holds that content
    POST   /v1/inputs             store a multi-file input (a DICOM series) as
                                   one tree - several parts, or a zip; returns
                                   its digest. `from_job=<id>` instead promotes
                                   that job's result into the store, so one job's
                                   output can be another's input without the
                                   bytes routing through the client
    A finished job reports `outputs` - the content digest of what it produced -
    and that digest is the ETag on the result, so If-None-Match gets a 304.
    PUT    /v1/inputs/{digest}     store it (preload). The digest in the URL is
                                   CHECKED against the bytes, never trusted; a
                                   source may then be {"kind":"input","sha256":...}
                                   instead of re-sending. No route hands input
                                   bytes back.
    Cache-Control: no-cache on a submit (or an authorized labels GET) means "do not
    serve me a stored result" - RFC 9111, so the recompute is still published. An
    engine may decline to be served from cache at all; VoxTell does, because its
    free-text prompts make the key space unbounded.

    GET    /v1/jobs/{id}           full status; includes result metadata once done,
                                   its `key` (the handle for the stored result) and a
                                   `links` object - follow those rather than building
                                   path-surface URLs client-side
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
import hashlib
import os
import re
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import content
from .content import ContentStore, is_digest
from .errors import Cancelled, InputError, NnsegError, ResourceError
from .progress import CancelToken, Reporter

ACTIVE = ("queued", "running")
ARTIFACT_PENDING_TTL = 900.0   # a pending marker older than this is a dead
                               # overlap thread's leavings (mirrors Modal's sweep)
TERMINAL = ("done", "failed", "cancelled")
RESULT_NAME = "labels.seg.nrrd"          # the information-preserving default artifact
from .sources import CRDC_RE, IDC_BUCKETS, registry as _source_registry  # noqa: E402


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


def wants_no_cache(request) -> bool:
    """Whether the request asked not to be served a stored result.

    RFC 9111 `Cache-Control: no-cache` - "do not reuse a stored response" - which
    is exactly the lever a caller needs after the model behind a task changes
    without the cache key changing. Note it does NOT mean "do not store": the
    recomputed result is published as usual, so artifacts, `key` and `links`
    survive. `no-store` is a separate directive we deliberately do not implement.
    """
    raw = ""
    try:
        raw = request.headers.get("cache-control") or ""
    except Exception:
        return False
    return any(d.strip().lower() == "no-cache" for d in raw.split(","))


def _validate_request(seg, task, sources: list, options: dict) -> list:
    """Check a submit against what the task itself declares.

    Two things, both answerable before any work starts: the options are validated
    against the task's published parameter schema (the same declaration that
    generates the schema clients read, so the advertised contract and the
    enforced one cannot drift), and the sources are bound to the task's declared
    inputs **by name**.

    Returns the ``[(role, source), ...]`` binding in the model's channel order.
    Nothing consumes the ordering yet - no engine accepts more than one input -
    but the refusal a multi-input task gets is now the accurate one, naming the
    roles it wanted.

    A task whose spec cannot be resolved at all is left alone deliberately: that
    is the worker's error to raise, with the worker's context, and guessing at it
    here would turn a clear downstream failure into a confusing 422.
    """
    from fastapi import HTTPException      # an extra, so imported at call time

    from .engines.registry import ENGINES, NNUNETV2
    from .errors import RequestError
    from .schemas import (bind_sources, declared_inputs, validate_options,
                          wire_params)
    try:
        desc = seg.describe(task) or {}
    except Exception:
        # A task whose own metadata is unreadable - a truncated dataset.json, a
        # half-written bundle. Fall through to the pre-existing contract (one
        # image, no option schema to check against) so the job dispatches and the
        # WORKER reports the real error with its context, which is what this
        # function's docstring promises. Returning an empty binding instead made
        # submit die on idents[0] with an opaque 500, and leaked the job dir.
        desc, unknown = {}, True
    else:
        unknown = False
    eng = ENGINES.get(desc.get("engine") or NNUNETV2) or ENGINES[NNUNETV2]
    try:
        if not unknown:
            validate_options(wire_params(eng.parameters, eng.processing_knobs), options)
        return bind_sources(sources, declared_inputs(desc),
                            multi_input=eng.multi_input, task=task)
    except RequestError as e:
        raise HTTPException(422, e.detail) from e


def engine_serves_from_cache(task) -> bool:
    """Whether this task's ENGINE allows a stored result to be served.

    An engine whose request space is unbounded (VoxTell's free text) sets this
    False: the lookup would almost never hit, and the entries only evict results
    that do get re-read. Routed by the task's grammar, the same way the worker
    dispatch resolves an engine, so no catalog is needed here.
    """
    try:
        from .engines.registry import engine_for_task
        return bool(engine_for_task(task).serve_from_cache)
    except Exception:
        return True                    # unknown engine: behave as before


def resource_links(task, identity, options, *, preview=False, statistics=False) -> dict:
    """The path-surface URLs for a finished result - ``{}`` when it is not
    path-addressable.

    The one place this URL grammar is written. Clients should follow these
    rather than assembling paths from a task name and an identity: the infix
    for grid variants, which identities are addressable at all (a single
    source identity, not an upload's sha256), and which options keep a result
    addressable are all decisions the server owns and can change.
    """
    ident = list(identity or [])
    if (len(ident) != 1 or ":" not in str(ident[0])
            or str(ident[0]).startswith("sha256:")):
        return {}                       # uploads and multi-source: job-scoped only
    opts = options or {}
    tok = next((t for t, o in GRID_TOKENS.items() if o == opts), None)
    if opts and tok is None:
        return {}                       # non-menu options have no path form
    infix = f"_{tok}" if tok else ""
    prefix, one = str(ident[0]).split(":", 1)
    base = f"/v1/{prefix}/{one}/{task}"
    links = {"labels": f"{base}/labels{infix}.seg.nrrd",
             "meta": f"{base}/meta{infix}.json"}
    if preview:
        links["preview"] = f"{base}/preview{infix}.png"
    if statistics:
        links["statistics"] = f"{base}/statistics{infix}.tsv"
    return links


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


class SeriesCache:
    """Series-keyed staging for fetched inputs: one directory per series under
    ``root``, claimed by atomic mkdir, committed by a ``.done`` marker holding
    the entry's byte count, evicted least-recently-used past ``budget_bytes``.
    One writer per series ever; readers wait on the marker. A directory without
    the marker is a writer mid-flight and is never read; a claim whose writer
    died (directory removed) ends the wait immediately; a claim stuck past
    ``claim_timeout`` is torn down and re-fetched. This is what lets several
    tasks on the same image download the series once."""

    MARKER = ".done"

    def __init__(self, root, fetch_fn, *, budget_bytes: int = 8 << 30,
                 claim_timeout: float = 180.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fetch = fetch_fn                  # fetch_fn(series, entry_dir) -> content dir
        self.budget = int(budget_bytes)
        self.claim_timeout = float(claim_timeout)
        self._lock = threading.Lock()          # eviction + pin bookkeeping
        self._pins: dict = {}                  # entry name -> refcount

    def _entry(self, series: str) -> Path:
        # Keys become directory names. Filesystem-safe keys keep their readable
        # verbatim name (a cache you can ls); anything else - separators, dot
        # names, absurd lengths - maps to a deterministic hash, so identifiers
        # with slashes (DOIs, org/name ids) are safe by construction rather
        # than forbidden.
        safe = ("/" not in series and "\\" not in series and "\x00" not in series
                and series not in (".", "..") and 0 < len(series) <= 200)
        if safe:
            return self.root / series
        import hashlib
        d = self.root / ("h_" + hashlib.sha256(series.encode()).hexdigest()[:32])
        return d

    def has(self, series: str) -> bool:
        return (self._entry(series) / self.MARKER).exists()

    def path(self, series: str) -> Path:
        """Content directory of a committed entry (valid only when has())."""
        return self._entry(series) / "series"

    def pin(self, series: str) -> None:
        """Protect an in-use entry from LRU eviction - the marker mtime ages
        for a whole run, so without a pin the CURRENT job's input can be
        evicted mid-read under budget pressure. Refcounted; pair with unpin
        in a finally."""
        name = self._entry(series).name
        with self._lock:
            self._pins[name] = self._pins.get(name, 0) + 1

    def unpin(self, series: str) -> None:
        name = self._entry(series).name
        with self._lock:
            n = self._pins.get(name, 0) - 1
            if n <= 0:
                self._pins.pop(name, None)
            else:
                self._pins[name] = n

    def _claim_owner(self, entry: Path) -> str:
        """Stamp a fresh claim with an owner token. A writer may only tear
        down a claim it still owns: after a false-dead reclaim (graveyard
        rename + re-claim), the ORIGINAL writer's path names the successor's
        claim, and deleting by bare path destroyed the successor's staging
        (adversarial round, 2026-08-25)."""
        token = uuid.uuid4().hex
        try:
            (entry / ".owner").write_text(token)
        except OSError:
            pass
        return token

    def _teardown_claim(self, entry: Path, token: str) -> None:
        """Remove a failed claim - only while this writer PROVABLY owns it.
        Unprovable (unreadable/absent .owner) means refuse: the successor's
        claim exists ownerless between its mkdir and its .owner write, and a
        writer whose own .owner write failed must not stay licensed to
        delete whatever later occupies the path. An abandoned claim costs
        one claim_timeout - the reclaim path collects it."""
        try:
            if (entry / ".owner").read_text() != token:
                return                     # reclaimed by a successor: not ours
        except OSError:
            return                         # cannot prove ownership: never delete
        shutil.rmtree(entry, ignore_errors=True)

    def _hb_start(self, entry: Path):
        """Tick the claim's mtime while the fetch runs. _writer_alive reads
        file mtimes as the heartbeat, but a fetch that buffers in memory
        (IDC reads whole objects before writing) is disk-silent for minutes -
        long enough to be declared dead while alive. The ticker makes
        liveness mean process-liveness, which is what reclaim must key on."""
        stop = threading.Event()
        beat = entry / ".owner"            # a FILE: _writer_alive scans files
                                           # (rglob), a directory utime is
                                           # invisible to it
        def tick():
            while not stop.wait(min(10.0, self.claim_timeout / 6)):
                try:
                    os.utime(beat if beat.exists() else entry)
                except OSError:
                    return

        threading.Thread(target=tick, name="nnseg-claim-hb", daemon=True).start()
        return stop

    def _writer_alive(self, entry: Path) -> bool:
        """A live writer keeps producing files - the newest mtime under the
        entry is its heartbeat. A timeout alone proves nothing about death."""
        try:
            newest = max((f.stat().st_mtime for f in entry.rglob("*")),
                         default=entry.stat().st_mtime)
        except OSError:
            return False
        return (time.time() - newest) < self.claim_timeout

    def staging(self, series: str) -> bool:
        e = self._entry(series)
        return e.exists() and not (e / self.MARKER).exists()

    def get_or_fetch(self, series: str, *, check=None, credentials=None,
                     fetch=None) -> Path:
        """Return the series content directory, fetching if needed (blocking).
        ``check`` is called while waiting on another writer; raise from it to
        cancel the wait.

        ``fetch`` overrides how the content is produced for this call only, so a
        caller that ALREADY has the bytes - an upload being adopted into the
        content store - reuses this method's claim, wait-on-another-writer and
        commit rather than growing a second copy of them. Two callers storing the
        same content at once behave exactly like two jobs fetching one series:
        the first writes, the second waits and then finds it there."""
        while True:
            entry = self._entry(series)
            marker = entry / self.MARKER
            if marker.exists():
                try:
                    os.utime(marker)           # LRU touch
                except OSError:                # evicted between check and touch
                    continue                   # reclaim on the next pass
                return entry / "series"
            try:
                entry.mkdir(parents=True)      # atomic claim: one writer per series
            except FileExistsError:
                deadline = time.time() + self.claim_timeout
                extensions = 0
                while not marker.exists():
                    if not entry.exists():
                        break                  # writer failed and cleaned up; reclaim
                    if time.time() > deadline:
                        if self._writer_alive(entry):
                            if extensions < 3:
                                # slow but alive: extend rather than destroy -
                                # a timeout teardown of a LIVE writer aliases
                                # two writers onto one path and they destroy
                                # each other's work
                                extensions += 1
                                deadline = time.time() + self.claim_timeout
                            else:
                                # STILL alive after 4x claim_timeout: give up
                                # loudly. A live writer is NEVER reclaimed -
                                # the owner token guards teardown, not
                                # _commit, and a round-4 probe showed the
                                # reclaim committing a MIXED series (two
                                # writers' slices in one entry) as complete.
                                # One writer per series is the invariant; the
                                # waiter's job fails retryably instead.
                                raise ResourceError(
                                    f"staging of {series!r} has been held for "
                                    f"{4 * self.claim_timeout:.0f}s by a writer "
                                    "that is still alive; giving up rather "
                                    "than risk a mixed series - retry later")
                        else:
                            # dead claim: rename to a graveyard name first so
                            # the old writer's paths stop aliasing the reclaim
                            grave = entry.parent / f"{entry.name}.stale{int(time.time())}"
                            try:
                                entry.rename(grave)
                            except OSError:
                                pass
                            shutil.rmtree(grave, ignore_errors=True)
                            break
                    if check is not None:
                        check()
                    time.sleep(0.2)
                continue
            token = self._claim_owner(entry)
            hb = self._hb_start(entry)
            try:
                fn = fetch or self.fetch
                dest = Path(fn(series, entry, credentials=credentials)
                            if credentials is not None else fn(series, entry))
                self._commit(entry, key=series, token=token)
                return dest
            except BaseException:
                self._teardown_claim(entry, token)
                raise
            finally:
                hb.set()

    def prefetch(self, series: str) -> bool:
        """Claim + fetch + commit without blocking on other writers. False if
        already present, claimed elsewhere, or the fetch failed."""
        entry = self._entry(series)
        if entry.exists():
            return False
        try:
            entry.mkdir(parents=True)
        except FileExistsError:
            return False
        token = self._claim_owner(entry)
        hb = self._hb_start(entry)
        try:
            self.fetch(series, entry)
            self._commit(entry, key=series, token=token)
            return True
        except Exception:
            self._teardown_claim(entry, token)
            return False
        finally:
            hb.set()

    def _commit(self, entry: Path, key: str | None = None,
                token: str | None = None) -> None:
        if token is not None:
            try:
                if (entry / ".owner").read_text() != token:
                    raise ResourceError(
                        f"claim for {key or entry.name!r} was reclaimed while "
                        "fetching; discarding this writer's result")
            except OSError:
                pass                       # unprovable = our own failed write
        # budget = content bytes; bookkeeping dotfiles (.owner/.key/.done)
        # stay out of the arithmetic
        size = sum(f.stat().st_size for f in entry.rglob("*")
                   if f.is_file() and not f.name.startswith("."))
        if key is not None and entry.name != key:
            (entry / ".key").write_text(key)   # readable name for hashed entries
        (entry / self.MARKER).write_text(str(size))
        self._evict(keep={entry.name})

    def _evict(self, keep) -> None:
        with self._lock:
            protected = set(keep) | set(self._pins)
            entries, kept_bytes = [], 0
            for e in self.root.iterdir():
                m = e / self.MARKER
                if not m.exists():
                    continue                   # a writer mid-flight: never touch
                try:
                    mtime, size = m.stat().st_mtime, int(m.read_text() or 0)
                except (OSError, ValueError):
                    continue
                if e.name in protected:
                    kept_bytes += size
                else:
                    entries.append((mtime, size, e))
            total = kept_bytes + sum(sz for _, sz, _ in entries)
            for _, size, e in sorted(entries, key=lambda t: t[0]):
                if total <= self.budget:
                    break
                shutil.rmtree(e, ignore_errors=True)
                total -= size


def _read_image(path):
    from . import io as nio
    return nio.read_image(path)


class ReadAhead:
    """At most one pre-read image, keyed by series: the prefetch thread fills
    it after staging bytes, the next run pops it. Capacity 1 by design - a
    full CT is ~1 GB of RAM and one-ahead is the pipeline's depth. The image
    is task-independent (stored orientation, IPP-corrected geometry via
    :func:`nnseg.io.read_image`); the pipeline applies each task's own
    reorientation exactly as it does for any caller-held image."""

    def __init__(self, read_fn=None):
        self.read = read_fn or _read_image
        self._lock = threading.Lock()
        self._key = None
        self._image = None

    def fill(self, series: str, path) -> bool:
        try:
            img = self.read(path)
        except Exception:
            return False
        with self._lock:
            self._key, self._image = series, img
        return True

    def has(self, key: str) -> bool:
        with self._lock:
            return self._key == key

    def pop(self, key: str):
        with self._lock:
            if self._key == key:
                img, self._key, self._image = self._image, None, None
                return img
            return None


class ResultCache:
    """Content-keyed store of finished results: <root>/<key>/labels.seg.nrrd +
    result.json + meta.json (the readable key components - a cache you can ls).
    LRU by directory mtime, count-bounded; results are ~MBs so the bound is
    generous by default."""

    def __init__(self, root, *, keep: int = 500):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = int(keep)

    def list(self, limit: int = 500) -> list:
        """Completed segmentations, newest first: the readable meta of every
        cached entry plus size and, when the entry is path-addressable (single
        source identity, default options), its path-surface URL."""
        out = []
        for d in self.root.iterdir():
            labels, meta_p = d / RESULT_NAME, d / "meta.json"
            if not (d.is_dir() and labels.exists()):
                continue
            try:
                meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
                st = labels.stat()
            except (OSError, json.JSONDecodeError):
                continue
            entry = {"key": d.name, "task": meta.get("task"),
                     "identity": meta.get("identity"),
                     "options": meta.get("options"),
                     "computed": meta.get("computed"), "bytes": st.st_size}
            links = resource_links(meta.get("task"), meta.get("identity"),
                                   meta.get("options"),
                                   preview=(d / "preview.png").exists(),
                                   statistics=(d / "statistics.json").exists())
            if links:
                entry["links"] = links
            out.append(entry)
        out.sort(key=lambda e: e.get("computed") or 0, reverse=True)
        return out[:limit]

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

    def put(self, key: str, labels_path, result: dict, meta: dict,
            preview_path=None, statistics_path=None) -> None:
        import os
        import shutil
        d = self.root / key
        d.mkdir(parents=True, exist_ok=True)

        def _place(name: str, write) -> None:
            # temp + rename, every file: an overwriting put (no_cache
            # recompute) must never truncate a file an open reader is
            # streaming, and a crash mid-copy must not leave a partial
            # gate file that get() would serve. The temp name is unique
            # per writer - concurrent same-key writers must not share it.
            tmp = d / f"{name}.{uuid.uuid4().hex[:8]}.tmp"
            write(tmp)
            os.replace(tmp, d / name)

        # sidecars first, labels LAST: get() and list() gate on the labels
        # file's existence, so an entry appears atomically complete
        _place("result.json", lambda t: t.write_text(json.dumps(result)))
        _place("meta.json", lambda t: t.write_text(json.dumps(meta, indent=2)))
        if preview_path and Path(preview_path).exists():
            _place("preview.png", lambda t: shutil.copy2(preview_path, t))
        if statistics_path and Path(statistics_path).exists():
            _place("statistics.json", lambda t: shutil.copy2(statistics_path, t))
        _place(RESULT_NAME, lambda t: shutil.copy2(labels_path, t))
        self.evict()

    def add_artifact(self, key: str, name: str, src_path) -> bool:
        """Place an eventually-consistent artifact (preview.png,
        statistics.json) into an existing entry, atomically (temp + rename) -
        the overlap thread calls this after "done" has already been served.
        False if the entry no longer exists (evicted meanwhile) - skip, never
        recreate."""
        import os
        import shutil
        d = self.root / key
        if not (d / RESULT_NAME).exists():
            return False
        tmp = d / f"{name}.{uuid.uuid4().hex[:8]}.tmp"   # unique per writer
        shutil.copy2(src_path, tmp)
        os.replace(tmp, d / name)
        return True

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

        def _mtime(d):                         # entries can vanish between
            try:                               # iterdir and stat (concurrent
                return d.stat().st_mtime       # delete/evict); a raise here
            except OSError:                    # would fail the putting job
                return 0.0

        try:
            dirs = [d for d in self.root.iterdir() if d.is_dir()]
        except OSError:
            return
        if len(dirs) <= self.keep:
            return
        for d in sorted(dirs, key=_mtime)[: len(dirs) - self.keep]:
            shutil.rmtree(d, ignore_errors=True)


class QueueFull(NnsegError):
    """The pending queue is at its bound; the caller should retry later."""


def _write_guard(executor):
    """The guard a write into the job directory must hold.

    An executor backed by a snapshot-consistent volume exposes one: a concurrent
    reload elsewhere discards uncommitted writes, so a partially written file can
    be truncated under us - and this store then hashes the truncation. Hoisted
    into one helper because it was documented as mandatory on the upload path and
    then quietly skipped by the two new ingest routes.
    """
    import contextlib
    return getattr(executor, "volume_guard", None) or contextlib.nullcontext()


def _discard(jdir) -> None:
    """Drop a job directory nothing owns yet."""
    import shutil
    shutil.rmtree(jdir, ignore_errors=True)


def reference_input(staged):
    """The one image a multi-input job's artifacts render against.

    A preview and a statistics table need ONE image, and the sensible choice is
    the task's first declared channel. Shared with the Modal worker rather than
    written twice: the two sides had already picked differently (declared order
    here, client source order there), so the same request previewed a different
    channel depending on where it ran.
    """
    if isinstance(staged, dict):
        return next(iter(staged.values()), None)
    return staged


def result_payload(seg, labels_path) -> dict:
    """A finished job's public result, built ONCE for both executors.

    They had drifted already - the Modal side reported `timings` and the local
    side did not - which is the same shape as the bug where an engine worker and
    the API disagreed about weights identity: two hand-written copies of one
    thing.

    `outputs` is new: the content digest of what the job produced. It is what
    lets a client skip re-downloading a result it already holds (it IS the
    ETag), and what lets one job's output become another's input without the
    bytes taking a round trip out through the client and back.
    """
    from .content import digest_file
    out = {"names": {int(k): v for k, v in seg.schema.names.items()},
           "volumes_ml": {k: round(float(v), 2) for k, v in seg.volumes_ml().items()},
           "provenance": seg.provenance}
    timings = getattr(seg, "timings", None)
    if timings:
        out["timings"] = {k: round(float(v), 3) for k, v in timings.items()}
    p = Path(labels_path)
    if p.exists():
        out["outputs"] = [{"name": "labels", "sha256": digest_file(p),
                           "bytes": p.stat().st_size}]
    return out


def etag_of(key: str, result=None) -> str:
    """The validator for a stored result.

    The CONTENT digest when we know it, so a client holding those exact bytes
    gets a 304 - and so two runs that produce identical bytes are identical to a
    cache. Falls back to the request key, which is what this always used to be:
    a fine validator for "the same request", but it changes when a weights
    version bumps even if the output does not.
    """
    outs = (result or {}).get("outputs") or []
    digest = outs[0].get("sha256") if outs else None
    return f'"{digest}"' if digest else f'"{key[:32]}"'


def not_modified(request, etag: str):
    """A 304 when the client already holds this exact content, else None.

    RFC 9110 conditional GET - the other half of an ETag we have been sending
    but never acting on. A label volume is megabytes and a Slicer client asks
    for the same one repeatedly.
    """
    from fastapi.responses import Response
    raw = ""
    try:
        raw = request.headers.get("if-none-match") or ""
    except Exception:
        return None
    if not raw:
        return None
    tags = {t.strip() for t in raw.split(",")}
    if etag in tags or "*" in tags:
        return Response(status_code=304, headers={"ETag": etag})
    return None


def publish_completion(*, segmenter, task, identity, options, cache_key,
                       labels_path, input_image, artifacts, cache_enabled,
                       migrate_key, set_pending, clear_pending, put,
                       mark_done, start_worker):
    """The one correct publication order for a finished segmentation, shared
    by both executors so ordering fixes cannot drift apart (in the 2026-08-25
    review, C7 and C8 each had to be written twice):

      1. re-key: the job may have been keyed while weights versions were
         still "unknown" (first install); they are knowable now, so migrate
         the key and the single-flight marker together - otherwise the entry
         is orphaned and every later probe misses forever.
      2. load the artifact pair while the input is still staged - the only
         inline artifact cost (~0.3 s).
      3. pending marker BEFORE the entry is visible: a probe that finds the
         fresh entry with no artifact must read "pending", never a
         definitive absence it would treat as a 404.
      4. the cache put (the entry's labels file is its commit marker).
      5. done becomes observable.
      6. the artifact worker starts; a start failure clears pending and must
         never disturb the already-observable done.

    The environment differences (in-process set vs Dict markers, volume
    commits, _emit vs a state field) ride in the callables; the ORDER lives
    here. Returns (cache_key, pair)."""
    if cache_key:
        try:
            fresh = result_key(identity, task, options,
                               weights_versions_of(segmenter, task))
            if fresh != cache_key:
                migrate_key(cache_key, fresh)
                cache_key = fresh
        except Exception:
            pass
    pair = None
    if artifacts and cache_enabled and cache_key:
        try:
            from .preview import load_oriented_pair
            pair = load_oriented_pair(input_image, labels_path)
        except Exception:
            pair = None
    if pair is not None:
        set_pending(cache_key)
    if cache_enabled and cache_key:
        put(cache_key)
    mark_done()
    if pair is not None:
        try:
            start_worker(pair, cache_key)
        except Exception:
            clear_pending(cache_key)
    return cache_key, pair


def artifact_overlap(pair, task, artifacts, *, preview_out, statistics_out,
                     place, finish):
    """The overlap thread's body, shared by both executors: render and
    compute from the in-RAM pair (no input or disk dependence), hand each
    file to ``place(name, path) -> bool`` (atomic within the cache; False
    when the entry was evicted meanwhile - dropped, never recreated), and
    ALWAYS run ``finish(placed)`` - the pending-marker clear, plus any
    commit/logging the environment wants. ``placed`` is [(name, seconds)]
    for what actually landed."""
    placed = []
    try:
        from .preview import render_preview
        from .statistics import compute_statistics
        if "preview" in artifacts:
            t0 = time.time()
            png = render_preview(None, None, preview_out, title=task, pair=pair)
            if png and place("preview.png", png):
                placed.append(("preview", time.time() - t0))
        if "statistics" in artifacts:
            t0 = time.time()
            sj = compute_statistics(None, None, statistics_out, pair=pair)
            if sj and place("statistics.json", sj):
                placed.append(("statistics", time.time() - t0))
    except Exception:
        pass
    finally:
        finish(placed)


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
    kind: str = "segment"                         # or "prepare": install weights only
    state: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict | None = None
    error: str | None = None
    result: dict | None = None                    # names / volumes_ml / provenance
    labels_path: Path | None = None
    source: list = field(default_factory=list)
    #: ``((role, path_or_None), ...)`` in the model's channel order, for a task
    #: that takes more than one input. Empty for the single-input case, which
    #: keeps using ``input_path``. Paths live here rather than on the ``source``
    #: entries because those are serialized to clients, and a server filesystem
    #: path is not theirs to see.
    input_paths: tuple = ()
    input_identity: tuple = ()
    cached: bool = False
    cache_key: str | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)
    subscribers: list = field(default_factory=list)   # (event_loop, asyncio.Queue)
    source_tokens: dict | None = None             # credentials in transit: never serialized


class _PrepareDone(Exception):
    """Control-flow escape: a prepare job finished; skip the segment path."""


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
                 cache_dir=None, keep_cached: int = 500,
                 input_cache_bytes: int = 8 << 30, read_fn=None, sources=None,
                 artifacts=("preview", "statistics")):
        self.segmenter = segmenter
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.max_pending = int(max_pending)
        self.keep_finished = int(keep_finished)
        self._segment = segment_fn or segmenter.segment
        self._fetch_idc = fetch_idc_fn or _fetch_idc_series
        self.sources = _source_registry(sources)
        self.series_cache = SeriesCache(self.workdir / "series_cache", self._fetch_source,
                                        budget_bytes=input_cache_bytes)
        self.read_ahead = ReadAhead(read_fn)
        # Uploads addressed by their own bytes, sharing the series cache's root,
        # budget and pin discipline - an entry is an entry whether a client sent
        # it or the server fetched it.
        self.content = ContentStore(self.series_cache)
        self.artifacts = set(artifacts or ())
        self._artifacts_pending: dict = {}   # cache_key -> (owner jid, set at)
        self.cache = ResultCache(cache_dir, keep=keep_cached) if cache_dir else None
        self._inflight: dict[str, str] = {}      # cache key -> active job id
        self._cv = threading.Condition()
        self._pending: deque[str] = deque()
        self._jobs: dict[str, JobRecord] = {}
        self._done_order: deque[str] = deque()
        self._stop = False
        self._thread = threading.Thread(target=self._dispatch, name="nnseg-serve", daemon=True)
        self._thread.start()

    def _fetch_source(self, key: str, entry, credentials=None):
        """Series-cache fetch dispatcher: keys are ``<prefix>:<identifier>``.
        The idc prefix routes through ``self._fetch_idc`` so the historical
        ``fetch_idc_fn`` injection seam keeps working. ``credentials`` is a
        per-request secret, forwarded and never stored."""
        prefix, ident = key.split(":", 1)
        if prefix == "idc":
            return self._fetch_idc(ident, entry)
        if credentials is not None:
            return self.sources[prefix].fetch(ident, entry, credentials=credentials)
        return self.sources[prefix].fetch(ident, entry)

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
               *, source=None, identity: tuple = (), no_cache: bool = False,
               source_tokens: dict | None = None, inputs: tuple = ()) -> JobRecord:
        rec = JobRecord(id=jid, task=task, options=options, dir=jdir, input_path=input_path,
                        source=list(source or [{"kind": "upload"}]), input_identity=tuple(identity),
                        input_paths=tuple(inputs), source_tokens=source_tokens or None)
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
            busy = any(r.state == "running" for r in self._jobs.values())
            self._cv.notify()
        self._emit(rec)
        if busy:
            self._prefetch_next()          # overlap this fetch with the running job
        return rec

    def submit_prepare(self, jid: str, jdir: Path, task: str) -> JobRecord:
        """Queue a weights-install job: same queue, no input, no cache."""
        rec = JobRecord(id=jid, task=task, options={}, dir=jdir, input_path=None,
                        kind="prepare", source=[])
        with self._cv:
            if len(self._pending) >= self.max_pending:
                raise QueueFull(f"queue is full ({self.max_pending} pending)")
            self._jobs[jid] = rec
            self._pending.append(jid)
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

    def cache_list(self) -> list:
        return self.cache.list() if self.cache is not None else []

    def artifact_state(self, key: str) -> str:
        """"pending" while the overlap thread is still placing this entry's
        artifacts; "absent" once it finished (or never ran) - at which point a
        missing artifact file is definitive."""
        cur = self._artifacts_pending.get(key)
        if cur is None:
            return "absent"
        if time.time() - cur[1] > ARTIFACT_PENDING_TTL:
            # a killed/hung overlap thread left its marker; without this the
            # refuse-if-present set makes it immortal and every artifact GET
            # 202s forever (Modal has the same rule as a 900s Dict sweep)
            self._artifacts_pending.pop(key, None)
            return "absent"
        return "pending"

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
                # compare-and-pop, same rule as the dispatcher's finally:
                # under duplicate flights the marker names the LATEST job
                if rec.cache_key and self._inflight.get(rec.cache_key) == jid:
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
    def _from_store(self, entry, pinned: list):
        """Resolve an ``input`` source: content this server already holds.

        Nothing is fetched or copied - the bytes are in the store already, which
        is the whole point of referring by digest. Pinned like any other input,
        because the store is LRU and the entry could otherwise be evicted out
        from under a running job.
        """
        digest = str(entry.get("id") or entry.get("sha256") or "")
        self.content.pin(digest)
        pinned.append(digest)
        return self.content.resolve(digest)

    def _stage_many(self, rec, entries, reporter, pinned: list) -> dict:
        """Stage every input of a multi-input job; returns ``{role: path}``.

        Fetches run sequentially inside this one dispatcher thread, each
        reporting its own ``fetch`` stage so a client can see which sequence is
        being pulled. The read-ahead cache stays out of this path deliberately:
        it holds one pre-read volume aimed at the NEXT job, and a multi-input job
        would evict it for a fraction of its own inputs.

        Uploads were already written at submit and arrive on ``rec.input_paths``;
        remote sources are fetched here, the same way and through the same
        pinned series cache as a single-input job.
        """
        staged = {}
        by_role = {e.get("role"): e for e in entries}
        for role, path in rec.input_paths:
            entry = by_role.get(role) or {}
            kind = entry.get("kind", "upload")
            if kind == "upload":
                staged[role] = path
                continue
            if kind == "input":
                staged[role] = self._from_store(entry, pinned)
                continue
            ident = str(entry.get("id") or entry.get("crdc_series_uuid") or "")
            key = f"{kind}:{ident}"
            self.series_cache.pin(key)
            pinned.append(key)
            reporter.stage("fetch", f"{role} {ident[:8]}")
            staged[role] = self.series_cache.get_or_fetch(
                key, check=reporter.check,
                credentials=(rec.source_tokens or {}).get(kind))
            reporter.check()
        # the reference input: what previews and statistics render against
        rec.input_path = staged[rec.input_paths[0][0]]
        return staged

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
            try:                           # nothing before the handler may kill
                self._emit(rec)            # the ONLY dispatcher thread
                self._requeue_positions()
                self._prefetch_next()      # the CPU downloader, parallel to this GPU job
            except Exception:
                pass
            pinned = []
            try:
                reporter = Reporter.of(progress=lambda p, r=rec: self._on_progress(r, p),
                                       cancel=rec.cancel_token)
                if rec.kind == "prepare":
                    reporter.stage("weights", rec.task)
                    rec.result = self.segmenter.prepare(rec.task)
                    reporter.check()
                    rec.state = "done"
                    raise _PrepareDone
                entries = rec.source or [{"kind": "upload"}]
                if len(entries) > 1:
                    inp = self._stage_many(rec, entries, reporter, pinned)
                else:
                    src = entries[0]
                    kind = src.get("kind", "upload")
                    if kind == "input":
                        rec.input_path = inp = self._from_store(src, pinned)
                    elif kind != "upload":
                        ident = src.get("id") or src.get("crdc_series_uuid")
                        key = f"{kind}:{ident}"
                        self.series_cache.pin(key)
                        pinned.append(key)
                        if self.series_cache.has(key):
                            reporter.stage("fetch", "cached")
                        elif self.series_cache.staging(key):
                            reporter.stage("fetch", "prefetched")
                        else:
                            reporter.stage("fetch", ident[:13])
                        rec.input_path = self.series_cache.get_or_fetch(
                            key, check=reporter.check,
                            credentials=(rec.source_tokens or {}).get(kind))
                        reporter.check()
                        preread = self.read_ahead.pop(key)
                        if preread is not None:
                            reporter.stage("read", "preread")
                            inp = preread
                        else:
                            inp = rec.input_path
                    else:
                        preread = self.read_ahead.pop(rec.id)
                        if preread is not None:
                            reporter.stage("read", "preread")
                            inp = preread
                        else:
                            inp = rec.input_path
                seg = self._segment(inp, rec.task, progress=reporter,
                                    cancel=rec.cancel_token, **rec.options)
                rec.labels_path = Path(seg.save(rec.dir / RESULT_NAME))
                rec.result = result_payload(seg, rec.labels_path)
                (rec.dir / "result.json").write_text(json.dumps(rec.result))

                def _migrate(old_key: str, new_key: str) -> None:
                    with self._cv:
                        if self._inflight.get(old_key) == rec.id:
                            self._inflight.pop(old_key, None)
                        if self._inflight.get(new_key) in (None, rec.id):
                            self._inflight[new_key] = rec.id   # never stomp a newer flight
                        rec.cache_key = new_key

                def _mark_done() -> None:
                    rec.state = "done"

                def _set_pending(key: str) -> None:
                    # refuse-if-present: a duplicate flight must not ACQUIRE
                    # ownership by stomping - it would then legally clear the
                    # marker while the sibling's overlap still renders, and
                    # probes would read a definitive 404 for artifacts that
                    # land seconds later. The owner's finish clears; then a
                    # later flight may set again.
                    cur = self._artifacts_pending.get(key)
                    if cur is None or time.time() - cur[1] > ARTIFACT_PENDING_TTL:
                        self._artifacts_pending[key] = (rec.id, time.time())

                def _clear_pending(key: str) -> None:
                    # only the owner clears: a duplicate flight's finish must
                    # not turn our still-pending artifacts into definitive 404s
                    if self._artifacts_pending.get(key, (None,))[0] == rec.id:
                        self._artifacts_pending.pop(key, None)

                def _start(pair, key: str) -> None:
                    threading.Thread(target=self._artifact_worker,
                                     args=(pair, key, rec.dir, rec.task, rec.id),
                                     name="nnseg-artifacts", daemon=True).start()

                rec.cache_key, _ = publish_completion(
                    segmenter=self.segmenter, task=rec.task,
                    identity=rec.input_identity, options=rec.options,
                    cache_key=rec.cache_key, labels_path=rec.labels_path,
                    input_image=reference_input(inp), artifacts=self.artifacts,
                    cache_enabled=self.cache is not None,
                    migrate_key=_migrate,
                    set_pending=_set_pending,
                    clear_pending=_clear_pending,
                    put=lambda key: self.cache.put(
                        key, rec.labels_path, rec.result,
                        {"identity": list(rec.input_identity), "task": rec.task,
                         "options": rec.options, "computed": rec.started,
                         "job": rec.id}),
                    mark_done=_mark_done, start_worker=_start)
            except _PrepareDone:
                pass
            except Cancelled:
                rec.state = "cancelled"
                if (rec.cache_key and self._artifacts_pending.get(
                        rec.cache_key, (None,))[0] == rec.id):
                    self._artifacts_pending.pop(rec.cache_key, None)
            except Exception as e:             # noqa: BLE001 - reported to the client
                rec.state = "failed"
                rec.error = f"{type(e).__name__}: {e}"
                if (rec.cache_key and self._artifacts_pending.get(
                        rec.cache_key, (None,))[0] == rec.id):
                    self._artifacts_pending.pop(rec.cache_key, None)   # failed after pending add
            finally:
                for key in pinned:
                    self.series_cache.unpin(key)
                rec.finished = time.time()
                with self._cv:
                    self._done_order.append(rec.id)
                    # compare-and-pop: under duplicate flights for one key the
                    # marker names the LATEST job - popping unconditionally
                    # made probes 404 while the survivor still ran and left
                    # DELETE unable to cancel it
                    if rec.cache_key and self._inflight.get(rec.cache_key) == rec.id:
                        self._inflight.pop(rec.cache_key, None)
                self._emit(rec)
                self._evict()

    def _artifact_worker(self, pair, cache_key: str, jdir: Path, task: str,
                         owner: str) -> None:
        """Post-completion artifacts via the shared overlap body; placement
        is the cache's atomic add_artifact, finish clears the pending marker
        only when this job still owns it."""

        def _finish(placed) -> None:
            if self._artifacts_pending.get(cache_key, (None,))[0] == owner:
                self._artifacts_pending.pop(cache_key, None)

        artifact_overlap(
            pair, task, self.artifacts,
            preview_out=jdir / "preview.png",
            statistics_out=jdir / "statistics.json",
            place=lambda name, path: self.cache.add_artifact(cache_key, name, path),
            finish=_finish)

    def _prefetch_next(self) -> None:
        """Best-effort: download the HEAD queued idc job's series into the
        series cache on a background thread while the GPU runs the current one.
        The fetch releases the GIL (Rust + file IO), so it costs the GPU loop
        nothing. The cache's atomic directory claim guarantees one writer per
        series; the dispatcher waits out an in-flight staging instead of racing
        it, and a failed staging leaves nothing behind, so the dispatcher falls
        back to its own fetch. This is the pre-loading half of the IO-prefetch
        pipeline - it hides the fetch; hiding the read is the follow-on."""
        with self._cv:
            nxt = self._jobs.get(self._pending[0]) if self._pending else None
            if nxt is None or nxt.state != "queued" or nxt.kind == "prepare":
                return                         # prepare has no input to stage
        if len(nxt.input_paths) > 1:
            # A multi-input job. The read-ahead holds ONE volume, so pre-reading
            # a fraction of this job's inputs would evict a useful entry to save
            # a fraction of one job's read; _stage_many deliberately never pops
            # it, and a pre-read nobody claims is pure waste.
            return
        src = nxt.source[0] if nxt.source else {"kind": "upload"}
        kind = src.get("kind", "upload")
        ident = src.get("id") or src.get("crdc_series_uuid")
        if kind != "upload" and ident:
            key = f"{kind}:{ident}"
            if self.series_cache.staging(key) or self.read_ahead.has(key):
                return                         # another writer, or already read

            def work():
                if self.series_cache.has(key) or self.series_cache.prefetch(key):
                    self.read_ahead.fill(key, self.series_cache.path(key))
        else:                                  # upload: bytes are local, hide the read
            key, path = nxt.id, nxt.input_path
            if self.read_ahead.has(key):
                return

            def work():
                self.read_ahead.fill(key, path)

        threading.Thread(target=work, name="nnseg-prefetch", daemon=True).start()

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
        """(state, labels path or None); (None, None) for an unknown job. The
        path is None when the bytes are gone (job dir purged under a kept
        record) - callers must not assume done implies readable."""
        rec = self.get(jid)
        if rec is None:
            return None, None
        p = rec.labels_path
        return rec.state, (p if p is not None and Path(p).exists() else None)

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
        if not brief:
            # the handle for this result, and the options its URL form depends on
            if rec.cache_key:
                d["key"] = rec.cache_key
            if rec.options:
                d["options"] = dict(rec.options)
        if rec.cached:
            d["cached"] = True
        if not brief and rec.state == "done" and rec.result is not None:
            d["result"] = rec.result
        return d


# The IDC fetch itself lives in nnseg.sources.IDCSource; these two shims keep
# the historical seams - tests monkeypatch _idc_enabled, and _fetch_idc_series
# is the injectable default for LocalExecutor(fetch_idc_fn=...).
# (kept text below for context)
# The three public IDC buckets, probed in order (idc-open-data holds 99.5 % of
# series). If IDC ever adds a bucket, series in it will fail loudly below; the
# upgrade path is resolving per series via idc-index (`series_aws_url`), which we
# deliberately keep OUT of the server tier to stay dependency-light - see the
# 2026-08-24 three-bucket finding in the medseg design doc before re-deriving this.
def _idc_enabled() -> bool:
    try:
        import obstore  # noqa: F401
        return True
    except ImportError:
        return False


def _fetch_idc_series(series: str, jobdir: Path) -> Path:
    """Fetch one IDC series (see nnseg.sources.IDCSource for the mechanics)."""
    from .sources import IDCSource
    return IDCSource().fetch(series, jobdir)

def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


CONTRACT_VERSION = "1"          # the /v1 wire contract; bump only on a breaking change


def _pkg_info(name: str) -> dict | None:
    """Best-effort ``{version, commit?}`` for an installed distribution. The commit
    (from a VCS install's ``direct_url.json``) is what actually pins an engine
    package's rev - the version string alone doesn't. Returns None if not installed
    (engine packages live only in their own images)."""
    import importlib.metadata as im
    try:
        info = {"version": im.version(name)}
    except Exception:
        return None
    try:
        import json
        raw = im.distribution(name).read_text("direct_url.json")
        commit = ((json.loads(raw).get("vcs_info") or {}).get("commit_id")
                  if raw else None)
        if commit:
            info["commit"] = commit
    except Exception:
        pass
    return info


def _progress_headers(progress: dict | None, extra: dict | None = None) -> dict:
    """202 progress as headers, so HEAD probes and header-only clients see how
    far along a flight is (a HEAD response cannot carry a body)."""
    h = {"Retry-After": "10", "Cache-Control": "no-store"}
    if extra:
        h.update(extra)
    p = progress or {}
    if p.get("stage"):
        h["NNSeg-Stage"] = str(p["stage"])
    if p.get("fraction") is not None:
        h["NNSeg-Fraction"] = f"{float(p['fraction']):.3f}"
    return h


# The path surface's grid-variant menu: entity token -> the canonical options
# it keys and computes under. Naming is BIDS-inspired (key-value entities in
# the stem: labels_res-1mm.seg.nrrd), NOT BIDS compliance - the entity form
# namespaces the token grammar so future dimensions compose instead of
# colliding. Absolute tokens only (a menu entry means the same thing for
# every task, case, and weights version); grow it when a real user asks.
GRID_TOKENS = {"res-1mm": {"grid": 1.0}}


def _task_stem(task: str) -> str:
    """The short task name for filenames - canonical names carry ':'."""
    return task.rpartition(":")[2]


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
                    w = float(token[5:])
                except ValueError:
                    continue
                if w != w:                 # NaN: min/max would pass it through
                    return 0.0             # (order-dependent); pin it explicitly
                return max(0.0, min(w, maximum))
    return default


def _prefer_wait_raw(request, maximum: float) -> float | None:
    """Like :func:`_prefer_wait`, but None when no Prefer token was sent -
    header PRESENCE expresses intent, the duration expresses patience:
    wait=0 means "materialize this, but do not hold my connection"
    (respond-async likewise)."""
    for raw in request.headers.getlist("prefer"):
        for token in raw.split(","):
            token = token.strip().lower()
            if token == "respond-async":
                return 0.0
            if token.startswith("wait="):
                try:
                    w = float(token[5:])
                except ValueError:
                    continue
                if w != w:                 # NaN: min/max would pass it through
                    return 0.0             # (order-dependent); pin it explicitly
                return max(0.0, min(w, maximum))
    return None


class CacheOnlyExecutor:
    """The executor protocol over a result cache and nothing else: no queue,
    no dispatcher, no compute path at all. ``create_app(..., read_only=True)``
    over one of these IS the public twin (review R4) - the same route code as
    the main app, so filenames, headers, grid variants, artifact routes and
    the listing cannot drift from it.

    ``key_fn(identity, task, opts)`` pins key derivation - the twin must key
    exactly as the writer did, including the weights-version component it has
    no segmenter to recompute. ``inflight_fn(key)`` optionally exposes an
    authorized flight's progress ({"progress": {...}} or None) so anonymous
    callers can watch it; the KEY doubles as the watch handle."""

    supports_push = False
    accepting = False

    class _TaskView:
        def __init__(self, tasks_fn, resolve_fn):
            self._tasks, self._resolve = tasks_fn, resolve_fn

        def tasks(self):
            return self._tasks()

        def resolve_task(self, t):
            if self._resolve is not None:
                return self._resolve(t)
            if t in self._tasks():
                return t
            raise LookupError(f"unknown task {t!r}")

    def __init__(self, cache_get, key_fn, tasks_fn, *, inflight_fn=None,
                 resolve_fn=None, list_fn=None, sources=None):
        self._get, self._key, self._inflight = cache_get, key_fn, inflight_fn
        self.segmenter = self._TaskView(tasks_fn, resolve_fn)
        self.sources = _source_registry(sources)
        if list_fn is not None:
            self.cache_list = list_fn

    def resource_key(self, identity: str, task: str, opts=None) -> str:
        return self._key(identity, task, opts or {})

    def cache_get(self, key: str):
        return self._get(key)

    def find_inflight(self, key: str):
        st = self._inflight(key) if self._inflight is not None else None
        return key if st else None

    def status_of(self, key: str) -> dict:
        if self._get(key) is not None:
            return {"state": "done"}
        st = self._inflight(key) if self._inflight is not None else None
        if st:
            return {"state": "running", "progress": st.get("progress")}
        if self._get(key) is not None:
            # completion window: the writer fills the cache BEFORE clearing
            # its marker, so miss-then-no-marker must recheck before
            # declaring failure
            return {"state": "done"}
        return {"state": "failed"}     # flight gone, nothing cached: stop watching

    def result_file(self, jid):
        return None, None


def create_app(executor: LocalExecutor, *, token: str | None = None,
               wait_default: float = 30.0, wait_max: float = 110.0,
               read_only: bool = False):
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
    sources = getattr(executor, "sources", None) or _source_registry(None)
    _key_override = getattr(executor, "resource_key", None)

    def derive_key(identity: str, task: str, opts: dict) -> str:
        """One key derivation for every route. An executor may pin it
        (CacheOnlyExecutor must: the twin has no segmenter to read weights
        versions from); otherwise keys come from the segmenter's view."""
        if _key_override is not None:
            return _key_override(identity, task, opts)
        return result_key((identity,), task, opts,
                          weights_versions_of(seg, task))

    def _source_enabled(srcobj) -> bool:
        # _idc_enabled stays the patchable seam for the idc source
        return _idc_enabled() if srcobj.prefix == "idc" else srcobj.enabled()

    def source_tokens_of(request) -> dict | None:
        """Per-source credentials from the NNSeg-Source-Token header
        (``prefix=token[,prefix=token]``). Headers, never query strings -
        query strings cache-bust and land in logs. The parsed dict lives only
        on the in-memory job record; the status whitelist cannot leak it."""
        raw = request.headers.get("nnseg-source-token")
        if not raw:
            return None
        out = {}
        for part in raw.split(","):
            prefix, _, tok = part.strip().partition("=")
            if prefix and tok:
                out[prefix] = tok
        return out or None

    def canon_task(t: str):
        """Canonical task name for any accepted form (short, eco:name,
        eco:name@version) - None when unknown/ambiguous. All forms converge to
        one canonical name and therefore one result-cache key.

        Path-bearing names are refused BEFORE resolution: the in-process
        API's freedom to run a model-folder path must not cross the wire,
        and a catalog without resolve() would pass the string through
        (Segmenter.resolve_task falls back to identity)."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@-]*", t or ""):
            return None
        if hasattr(seg, "resolve_task"):
            try:
                return seg.resolve_task(t)
            except LookupError:
                return None
        return t if t in seg.tasks() else None

    app = FastAPI(title="nnseg", version=_version())

    def authed(request) -> bool:
        if read_only:
            return False               # the twin can never compute
        if token is None:
            return True
        return request.headers.get("authorization", "") == f"Bearer {token}"

    def require_auth(request) -> None:
        if not authed(request):
            raise HTTPException(401, "this server requires a bearer token for "
                                     "anything beyond cached reads")

    import datetime as _dt
    _started_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    # -- the input store: send once, refer by digest ------------------------
    # Deliberately no route that hands input BYTES back. A client that referred
    # to a digest already has them, and serving them would turn "knows a hash"
    # into "can read what someone uploaded". Presence and storage only.

    def _store_or_404():
        store = getattr(executor, "content", None)
        if store is None:
            raise HTTPException(404, "this server has no input store")
        return store

    @app.get("/v1/inputs/{digest}")
    def input_status(digest: str, request: Request):
        """Whether this server already holds that content - the call a client
        makes before deciding whether an upload is needed at all."""
        store = _store_or_404()
        if not is_digest(digest):
            raise HTTPException(422, {
                "code": "bad_digest",
                "message": f"expected {content.BLOB}<hex> or {content.TREE}<hex>"})
        if not store.has(digest):
            raise HTTPException(404, {"code": "input_gone", "digest": digest,
                                      "message": "not held by this server"})
        where = store.resolve(digest)
        files = sorted(p for p in where.rglob("*") if p.is_file()) \
            if where.is_dir() else [where]
        return {"digest": digest,
                "kind": "tree" if digest.startswith(content.TREE) else "blob",
                "members": len(files),
                "bytes": sum(p.stat().st_size for p in files)}

    @app.put("/v1/inputs/{digest}")
    async def put_input(digest: str, request: Request,
                        filename: str | None = None):
        """Store content under the digest of its own bytes - the preload call.

        The digest in the URL is what the caller CLAIMS. It is checked against
        the bytes as they land and never taken on trust: accepting it would let
        one caller store bytes Y under digest X and silently poison every later
        job - and every memoized result - that refers to X.
        """
        require_auth(request)              # anonymous never computes, and never stores
        store = _store_or_404()
        if not is_digest(digest):
            raise HTTPException(422, {
                "code": "bad_digest",
                "message": f"expected {content.BLOB}<hex> or {content.TREE}<hex>"})
        if store.has(digest):
            return {"digest": digest, "stored": False, "reason": "already held"}
        if digest.startswith(content.TREE):
            raise HTTPException(422, {
                "code": "unsupported",
                "message": "a tree is stored by POSTing its members (or a zip) "
                           "to /v1/inputs; this route takes single blobs, whose "
                           "digest a client can compute before sending"})
        import tempfile
        jid, jdir = executor.new_job_dir()  # a scratch dir on the right filesystem
        try:
            fd, tmp = tempfile.mkstemp(dir=jdir)
            os.close(fd)
            tmp = Path(tmp)
            h = hashlib.sha256()
            with _write_guard(executor), open(tmp, "wb") as f:
                async for chunk in request.stream():
                    h.update(chunk)
                    f.write(chunk)
            actual = f"sha256:{h.hexdigest()}"
            if actual != digest:
                raise HTTPException(422, {
                    "code": "digest_mismatch", "declared": digest, "actual": actual,
                    "message": "the bytes are not the ones you said they were; "
                               "nothing was stored"})
            # The store keeps bytes, but SimpleITK chooses its reader from the
            # EXTENSION - so an entry has to land under a name that names its
            # format, or it is unreadable however correct its digest is. The
            # caller may say; otherwise it is read out of the content.
            name = filename or content.guess_name(tmp)
            if not name:
                raise HTTPException(422, {
                    "code": "unknown_format",
                    "message": "could not identify this content (expected NIfTI, "
                               "NRRD, MetaImage or DICOM); pass ?filename= to "
                               "name it explicitly"})
            store.put_file(tmp, computed=actual, name=Path(name).name)
            return {"digest": actual, "stored": True, "stored_as": Path(name).name,
                    "bytes": tmp.stat().st_size}
        finally:
            import shutil
            shutil.rmtree(jdir, ignore_errors=True)

    @app.post("/v1/inputs")
    async def post_input(request: Request, kind: str | None = None,
                         expect: str | None = None):
        """Store a multi-file input - a DICOM series - as one tree.

        Transport is the client's convenience: several multipart parts, or one
        zip. Identity is neither. The tree's digest is taken over the sorted
        digests of its MEMBERS, so it survives the order they arrived in, the
        names they were given, and the zip metadata of whatever carried them -
        the same series zipped twice differs in timestamps and member order, and
        keying on those bytes would make dedupe stop working while appearing to.

        Returns the digest, because a client cannot be expected to have computed
        our Merkle root before sending; one that HAS may pass ``expect`` and have
        it checked, or GET it first and skip the upload entirely.
        """
        require_auth(request)              # anonymous never computes, and never stores
        store = _store_or_404()
        form = await request.form()
        from_job = form.get("from_job")
        if from_job:
            # One job's output becomes another's input, without the bytes taking
            # a round trip out through the client and back. The digest returned
            # is the one the producing job already published in `outputs` - same
            # bytes, same address - so a client can predict it.
            #
            # Copied in ON REQUEST rather than for every result: outputs are
            # re-derivable and inputs are not, so filling the input store with
            # results nobody chains would evict the irreplaceable to keep the
            # reproducible.
            #
            # What a downstream task DOES with a labelmap is a separate question:
            # `inputs[].kind` is "image" everywhere today, and a task that
            # consumes a mask would have to declare so before this composes into
            # anything meaningful.
            state, path = executor.result_file(str(from_job))
            if state is None:
                raise HTTPException(404, {"code": "no_job",
                                          "message": f"no job {from_job!r}"})
            if state != "done" or path is None:
                raise HTTPException(409, {
                    "code": "not_done",
                    "message": f"job {from_job!r} is {state}; nothing to promote"})
            name = content.guess_name(path) or Path(path).name
            digest = store.put_file(path, expect=expect, name=Path(name).name)
            return {"digest": digest, "kind": "blob", "members": 1, "stored": True,
                    "bytes": Path(path).stat().st_size, "from_job": str(from_job)}
        # multi_items(), not values(): a form is a MULTIdict, and several
        # parts sharing one name - which is how a client sends a series -
        # collapse to a single value through the dict-shaped accessor.
        uploads = [v for _, v in form.multi_items() if hasattr(v, "read")]
        if not uploads:
            raise HTTPException(422, {
                "code": "no_content",
                "message": "POST /v1/inputs takes one or more file parts, or a "
                           "single zip"})
        if kind not in (None, "tree", "blob"):
            raise HTTPException(422, {"code": "bad_kind",
                                      "message": "kind must be 'tree' or 'blob'"})
        import shutil
        jid, work = executor.new_job_dir()
        try:
            staged = work / "in"
            staged.mkdir()
            for i, up in enumerate(uploads):
                name = Path(getattr(up, "filename", None) or f"part{i}").name
                with _write_guard(executor), open(staged / f"{i}_{name}", "wb") as f:
                    while chunk := await up.read(1 << 20):
                        f.write(chunk)
            files = sorted(staged.iterdir())
            # One zip is an ARCHIVE of members, not a member itself - unpack it
            # unless the caller explicitly asked for a blob.
            def _is_zip(path):
                # four bytes, not the whole upload: this runs on every
                # single-part POST, on an async route, against a file that may
                # sit on a network volume
                with open(path, "rb") as f:
                    return f.read(4) == b"PK\x03\x04"

            if len(files) == 1 and kind != "blob" and _is_zip(files[0]):
                unpacked = work / "unpacked"
                try:
                    with _write_guard(executor):
                        files = sorted(content.extract_zip(files[0], unpacked))
                except content.ArchiveError as e:
                    raise HTTPException(422, {"code": "bad_archive",
                                              "message": str(e)}) from e
                staged = unpacked
            as_tree = kind == "tree" or (kind is None and len(files) > 1) or \
                staged.name == "unpacked"
            if not as_tree:
                name = Path(getattr(uploads[0], "filename", "") or "").name \
                    or content.guess_name(files[0])
                if not name:
                    raise HTTPException(422, {
                        "code": "unknown_format",
                        "message": "could not identify this content; name the "
                                   "part or pass kind=tree"})
                digest = store.put_file(files[0], expect=expect,
                                        name=Path(name).name)
                return {"digest": digest, "kind": "blob", "members": 1,
                        "stored": True, "bytes": files[0].stat().st_size}
            # Refuse a mixed folder rather than pick a series out of it: reading
            # "the" series when there are two means choosing one, and choosing
            # silently is how a plausible, wrong segmentation gets produced.
            from nnseg.io import dicom_series_ids
            series = dicom_series_ids(staged)
            if len(series) > 1:
                raise HTTPException(422, {
                    "code": "multiple_series",
                    "message": f"these {len(files)} files hold {len(series)} DICOM "
                               "series; submit one series per input",
                    "series_instance_uids": sorted(series)})
            digest = store.put_dir(staged, expect=expect)
            return {"digest": digest, "kind": "tree", "members": len(files),
                    "stored": True,
                    "bytes": sum(p.stat().st_size for p in files),
                    "series_instance_uid": series[0] if series else None}
        except content.DigestMismatch as e:
            raise HTTPException(422, {"code": "digest_mismatch",
                                      "declared": e.expected, "actual": e.actual,
                                      "message": str(e)}) from e
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @app.get("/v1/health")
    def health():
        policy = getattr(seg, "policy", {})
        return {"name": "nnseg", "version": _version(),
                "mode": "public-cache" if read_only else "serve",
                "device": str(policy.get("device", "?")),
                "n_tasks": len(seg.tasks()), "accepting": executor.accepting,
                "sources": ["upload"] + [k for k, v in sources.items()
                                         if _source_enabled(v)]}

    @app.get("/v1/version")
    def version():
        """Self-report - what this deployment is running. Anonymous, no external
        calls: the wire contract version, the nnseg version, when this instance
        started, the installed package versions (with git commit for the engine
        packages, which is what pins their rev), and the weights identity per task.
        For clients (compat), operators (is it current?), and debugging (which rev
        produced this seg?)."""
        # nnseg itself is reported by the top-level "version" (its __version__): it's
        # mounted/source-on-path in the deployed images and CI, so it has no dist
        # metadata to read here. `packages` reports the installed DEPENDENCIES.
        pkgs = {}
        for name in ("nnunetv2", "torch", "fastsurfer-lean", "synthstrip-torch", "surfa"):
            info = _pkg_info(name)
            if info is not None:
                pkgs[name] = info
        weights: dict = {}
        cat = getattr(seg, "catalog", None)
        if hasattr(cat, "info"):
            for t in seg.tasks():
                try:
                    wi = dict(cat.info(t)).get("weights_installed")
                    if wi:
                        weights[t] = wi
                except Exception:
                    pass
        from .engines import registry as _engines
        engines = {n: {"enabled": _engines.enabled(n),
                       "description": _engines.ENGINES[n].description}
                   for n in _engines.ENGINES}
        return {"name": "nnseg", "contract": CONTRACT_VERSION, "version": _version(),
                "started_at": _started_at, "engines": engines,
                "packages": pkgs, "weights": weights}

    @app.get("/v1/tasks")
    def tasks():
        out = {"tasks": seg.tasks()}
        cat = getattr(seg, "catalog", None)
        if hasattr(cat, "info"):
            detail = {}
            for t in out["tasks"]:
                try:
                    i = dict(cat.info(t))
                    i.pop("structures", None)   # describe() carries the full list
                    detail[t] = i
                except Exception:
                    pass
            out["detail"] = detail
        return out

    @app.post("/v1/tasks/{task}/prepare", status_code=202)
    def prepare_task(request: Request, task: str):
        """Install a task's weights now (authorized): the deliberate form of
        what first use does implicitly. Returns a job to watch; its result is
        the task's full description once materialized."""
        require_auth(request)
        canonical = canon_task(task)
        if canonical is None:
            raise HTTPException(404, f"unknown task {task!r}")
        # canon_task drops @version by design (all wire forms converge to ONE
        # cache key). For prepare the version IS the request, so pass the
        # requested form through - dropping it silently installed the default
        # and answered 202, the "silent wrong version" bug this grammar exists
        # to prevent (found by the live integration pass).
        requested = task if "@" in task else canonical
        task = canonical
        if not executor.accepting:
            raise HTTPException(429, "queue is full, retry later",
                                headers={"Retry-After": "30"})
        jid, jdir = executor.new_job_dir()
        try:
            executor.submit_prepare(jid, jdir, requested)
        except QueueFull as e:
            import shutil
            shutil.rmtree(jdir, ignore_errors=True)
            raise HTTPException(429, str(e), headers={"Retry-After": "30"}) from e
        return {"id": jid, "kind": "prepare", "task": task,
                **({"version": requested.rpartition("@")[2]}
                   if requested != task else {})}

    @app.get("/v1/sources")
    def list_sources():
        return {"sources": [dict(v.describe(), enabled=_source_enabled(v))
                            for v in sources.values()]}

    @app.get("/v1/segmentations")
    def list_segmentations(request: Request):
        """Authorized only (review finding): the listing enumerates every
        cached result including sha256 identities of authenticated users'
        uploads - not part of the anonymous tier. The public twin's listing
        stays an explicit operator opt-in (list_fn): read_only serves it
        anonymously when a lister exists and drops the route otherwise."""
        if not read_only:
            require_auth(request)
        lister = getattr(executor, "cache_list", None)
        entries = lister() if lister else []
        keep = []                          # advertise only links that resolve
        wv_memo: dict = {}                 # weights per task, once per request
        for e in entries:                  # on THIS app's mounted sources
            p = e.get("path") or ""
            parts = p.split("/", 3)
            pfx = parts[2] if len(parts) > 2 else None
            if pfx is not None and pfx not in sources:
                continue
            t = e.get("task")
            canonical = canon_task(str(t)) if t is not None else None
            if t is not None and canonical is None:
                continue                   # task this catalog cannot serve
            ident = (e.get("identity") or [None])[0]
            key = e.get("key")
            if canonical and ident and key:
                # key round-trip: a stale-keyed entry's link 404s WITH a
                # recompute hint - following the listing's own remedy would
                # duplicate GPU work for bytes already cached
                try:
                    if _key_override is not None:
                        fresh = _key_override(ident, canonical,
                                              e.get("options") or {})
                    else:
                        if canonical not in wv_memo:
                            wv_memo[canonical] = weights_versions_of(seg, canonical)
                        fresh = result_key((ident,), canonical,
                                           e.get("options") or {},
                                           wv_memo[canonical])
                    if fresh != key:
                        continue
                except Exception:
                    pass
            keep.append(e)
        return {"segmentations": keep}

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
            if isinstance(opts.get("grid"), (int, float)):
                opts["grid"] = float(opts["grid"])   # {"grid": 1} == {"grid": 1.0} == labels.1mm
            src = json.loads(source) if source else [{"kind": "upload"}]
            if not (isinstance(src, list) and all(isinstance(x, dict) for x in src)):
                raise ValueError("source must be a JSON list of objects")
        except ValueError as e:
            raise HTTPException(422, f"bad request: {e}") from e
        # RFC 9111 semantics, deliberately: `no-cache` means "do not SERVE a stored
        # result", not "do not store one" - that is `no-store`, which we do not offer.
        # So this skips the lookup and still publishes (see the overwriting put in
        # ResultCache), which is what keeps the artifacts, `key` and `links` intact on
        # a forced recompute. Popped before keying: a cache directive is not part of
        # what identifies a result, or every forced run would land in its own entry.
        no_cache = bool(opts.pop("no_cache", False)) or wants_no_cache(request)
        for entry in src:                  # every source, not just the first
            kind = entry.get("kind", "upload")
            if kind == "input":
                continue                   # a digest, resolved against the store
            if kind == "url":
                raise HTTPException(422, "source kind 'url' is reserved for the "
                                         "authenticated tier")
            if kind != "upload" and kind not in sources:
                raise HTTPException(422, f"unknown source kind {kind!r}; this server "
                                         f"offers upload, {', '.join(sources)}")
            if kind != "upload" and not _source_enabled(sources[kind]):
                raise HTTPException(422, f"source kind {kind!r} is not enabled on this "
                                         "server (missing dependency)")
        kind = src[0].get("kind", "upload") if src else "upload"
        canonical = canon_task(task)
        if canonical is None:              # catalog names only at the wire boundary
            names = seg.tasks()
            raise HTTPException(404, f"unknown task {task!r}; this server offers "
                                     f"{len(names)} catalog tasks, e.g. "
                                     + ", ".join(names[:4]))
        task = canonical
        # ...and the engine may decline to be served from cache at all (see
        # engine_serves_from_cache): a per-engine default, overridable per request
        # only in the stricter direction - a caller cannot ask to be memoized.
        no_cache = no_cache or not engine_serves_from_cache(task)
        # Everything the task itself knows about the request: which images it
        # takes, and which options it accepts. Refused HERE, at submit, where the
        # answer is already in hand - not deep inside a worker minutes later, and
        # never in silence (an option we do not know is a typo or a stale client,
        # and accepting it leaves the caller believing a knob was turned).
        binding = _validate_request(seg, task, src, opts)
        if not executor.accepting:
            raise HTTPException(429, "queue is full, retry later",
                                headers={"Retry-After": "30"})
        jid, jdir = executor.new_job_dir()
        # From here the directory has an owner: this handler, until submit()
        # hands it to a JobRecord. Every refusal below - a missing part, a bad
        # digest, an evicted input, an invalid identifier - used to abandon it,
        # and a multi-input refusal abandoned up to N-1 already-streamed volumes
        # with it. Nothing reaps a directory that no record knows about, and on
        # Modal /scratch is a persistent Volume, so those were permanent.
        try:
            return await _accept(request, jid, jdir, binding, task, opts, src,
                                 file, no_cache, executor, seg)
        except QueueFull as e:
            _discard(jdir)
            raise HTTPException(429, str(e), headers={"Retry-After": "30"}) from e
        except BaseException:
            _discard(jdir)                 # 422/410, a client disconnect, a bug
            raise

    async def _accept(request, jid, jdir, binding, task, opts, src, file,
                      no_cache, executor, seg):
        multi = len(binding) > 1
        # Only a multi-input job needs to look past the declared `file` part;
        # re-parsing the form for the single case would change nothing and
        # re-read the body.
        form = await request.form() if multi else None

        async def _save(upload, role: str) -> tuple:
            """Stream one upload to the job dir, hashing as it goes."""
            import contextlib
            import hashlib
            h = hashlib.sha256()
            name = Path(getattr(upload, "filename", None) or "input.nii.gz").name
            dest = jdir / (f"input_{role}_{name}" if multi else f"input_{name}")
            # executors backed by a snapshot-consistent volume expose a guard:
            # a concurrent reload elsewhere would discard this uncommitted write
            with _write_guard(executor), open(dest, "wb") as f:
                while chunk := await upload.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
            digest = f"sha256:{h.hexdigest()}"
            store = getattr(executor, "content", None)
            if store is not None:
                # Addressable from now on: the same bytes submitted again - for
                # another task, or as another channel - can be referred to by
                # this digest instead of re-sent. `computed`, not `expect`: the
                # digest was taken here, from these bytes, as they streamed.
                store.put_file(dest, computed=digest)
            return dest, digest

        staged, idents = [], []
        for role, entry in binding:
            sent_as = str(entry.get("role") or "")   # the client's own spelling
            # the CANONICAL role - the model's own spelling, whatever the client
            # sent - so dispatch and the cache key agree on one name
            if multi:
                entry["role"] = role
            kind = entry.get("kind", "upload")
            if kind == "upload":
                # The client named its PART in its own vocabulary. Binding
                # accepts an alias (t1ce for T1c), so looking the part up under
                # the canonical name alone would let a request bind and then fail
                # to find its own upload - the alias table would work for
                # matching and break the transport it exists to enable.
                declared = entry.get("part")
                names = ([declared] if declared
                         else ["file"] if not multi
                         else [n for n in (sent_as, role) if n])
                upload = None
                for cand in names:
                    upload = file if (cand == "file" and file is not None) else (
                        form.get(cand) if form is not None else None)
                    if upload is not None and hasattr(upload, "read"):
                        part = cand
                        break
                    upload = None
                if upload is None:
                    raise HTTPException(422, {
                        "code": "missing_part",
                        "message": "source kind 'upload' needs a multipart part "
                                   f"named {' or '.join(repr(n) for n in names)}",
                        "part": names[0], "accepted": names})
                dest, digest = await _save(upload, role)
                staged.append((role, dest))
                idents.append(digest)
                continue
            if kind == "input":
                # A reference to content this server already holds. Its identity
                # IS the digest, which is exactly what an upload of the same
                # bytes produces - so referring and re-sending land on the same
                # result-cache key, as they must.
                digest = str(entry.get("sha256") or entry.get("digest")
                             or entry.get("id") or "").strip()
                if not is_digest(digest):
                    raise HTTPException(422, {
                        "code": "bad_digest",
                        "message": f"{digest!r} is not a content digest; expected "
                                   f"{content.BLOB}<hex> or {content.TREE}<hex>"})
                store = getattr(executor, "content", None)
                if store is None or not store.has(digest):
                    raise HTTPException(410, {
                        "code": "input_gone",
                        "message": f"{digest} is not held by this server (never "
                                   "stored, or evicted); send the bytes again",
                        "digest": digest})
                entry["id"] = digest
                staged.append((role, None))
                idents.append(digest)
                continue
            if file is not None and not any(
                    e.get("kind", "upload") == "upload" for _, e in binding):
                raise HTTPException(422, f"unexpected file upload with a {kind!r} source")
            src_entry = entry
            if kind == "idc":
                # Explicit identifier fields, no format sniffing: the two IDC
                # identifiers MEAN different things. crdc_series_uuid is the storage
                # identity - version-pinned, immutable, cache-key-grade. A DICOM
                # SeriesInstanceUID names the series in DICOM space and can resolve to
                # different crdc uuids across IDC data releases; accepting it is a
                # resolution step, which is /v1/resolve's job when it lands.
                if "series_instance_uid" in src_entry:
                    raise HTTPException(422, "series_instance_uid (+ optional idc_version, "
                                             "default latest) is not supported yet; "
                                             "resolution arrives with /v1/resolve - "
                                             "today pass crdc_series_uuid")
                if "series" in src_entry:
                    raise HTTPException(422, "ambiguous field 'series': be explicit - "
                                             "crdc_series_uuid (IDC storage id, "
                                             "8-4-4-4-12 hex), or series_instance_uid "
                                             "(+ optional idc_version) once /v1/resolve "
                                             "lands")
                if "idc_version" in src_entry:
                    raise HTTPException(422, "idc_version goes with series_instance_uid; "
                                             "a crdc_series_uuid is already pinned to one "
                                             "IDC data release")
            ident = str(src_entry.get("id") or src_entry.get("crdc_series_uuid") or "").strip()
            ident = ident.lower() if kind == "idc" else ident
            if not ident:
                need = "crdc_series_uuid" if kind == "idc" else "id"
                raise HTTPException(422, f"a {kind!r} source needs {need}")
            if not re.fullmatch(sources[kind].id_pattern, ident):
                hint = (" (expected 8-4-4-4-12 hex; a dotted value would be a DICOM "
                        "SeriesInstanceUID, which needs /v1/resolve)") if kind == "idc" else ""
                raise HTTPException(422, f"{ident!r} is not a valid {kind} identifier" + hint)
            src_entry["id"] = ident
            staged.append((role, None))
            idents.append(sources[kind].identity(ident))
        # Canonical identity. For one input it stays the bare identity it has
        # always been, so every cached result keeps its key. For several, each is
        # tagged with its role and the pairs are SORTED, which makes the key a
        # pure function of the request: permuting the source list cannot split a
        # key, and rebinding the same files to different roles must not merge
        # two - the output bytes differ.
        identity = (tuple(sorted(f"{role}={i}" for (role, _), i in zip(staged, idents)))
                    if multi else (idents[0],))
        if True:
            executor.submit(jid, jdir, staged[0][1], task, opts,
                            # in the model's declared channel order, not the
                            # order the client happened to list them: both
                            # executors take "the first source" as the reference
                            # image, and that must mean channel 0 either way
                            source=[e for _, e in binding],
                            identity=identity, no_cache=no_cache,
                            source_tokens=source_tokens_of(request),
                            inputs=tuple(staged) if multi else ())
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

    def _with_links(d: dict) -> dict:
        """Add the job's key and the URLs for everything it produced.

        Built here rather than in either executor: both LocalExecutor and
        ModalExecutor assemble their own status dict, and a thing written twice
        drifts (this codebase has paid for that already). Clients follow these
        instead of reconstructing paths - the URL grammar is the server's.
        """
        if not isinstance(d, dict):
            return d
        out = dict(d)
        key = out.pop("cache_key", None) or out.get("key")
        if key:
            out["key"] = key
        jid = out.get("id")
        links = {"self": f"/v1/jobs/{jid}", "events": f"/v1/jobs/{jid}/events"}
        if out.get("state") == "done":
            links["result"] = f"/v1/jobs/{jid}/result"
            kinds = set(getattr(executor, "artifacts", ()) or ())
            links.update(resource_links(
                out.get("task"), out.get("input_identity"), out.get("options"),
                # the artifact overlap runs after done, so advertise what this
                # deployment renders rather than probing the cache per request
                preview="preview" in kinds, statistics="statistics" in kinds))
        out["links"] = links
        return out

    @app.get("/v1/jobs/{jid}")
    def status(request: Request, jid: str):
        require_auth(request)
        return _with_links(_status_or_404(jid))

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
            # a transition between the first fetch and the subscribe was never
            # pushed; re-fetch so the stream can't idle on a stale snapshot
            first = executor.status_of(jid) or first

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
        if state != "done":
            raise HTTPException(409, f"job is {state}, not done")
        if path is None:                       # done, but the bytes were purged
            raise HTTPException(410, "result no longer on the server; recompute it")
        task_name = (executor.status_of(jid) or {}).get("task", "labels")
        stem = _task_stem(task_name)           # canonical eco:name is not filename-safe
        if format in ("nii.gz", "nii"):        # the LOSSY conversion, by request only
            import shutil
            import tempfile

            import SimpleITK as sitk
            from starlette.background import BackgroundTask
            tmpd = tempfile.mkdtemp(prefix="nnseg-conv-")
            out = Path(tmpd) / f"{stem}_{jid}.nii.gz"
            sitk.WriteImage(sitk.ReadImage(str(path)), str(out), True)
            return FileResponse(out, media_type="application/gzip", filename=out.name,
                                background=BackgroundTask(shutil.rmtree, tmpd,
                                                          ignore_errors=True))
        # The route a client uses for results that are NOT path-addressable -
        # uploads, and every multi-input job - so this is where revalidation
        # actually saves a download. The validator is the content digest the job
        # already published in `outputs`; no re-hashing per request.
        status = executor.status_of(jid) or {}
        etag = etag_of(jid, status.get("result"))
        fresh = not_modified(request, etag)
        if fresh is not None:
            return fresh
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"{stem}_{jid}.seg.nrrd",
                            headers={"ETag": etag})

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
    def _resource_headers(key: str, result=None) -> dict:
        # Vary: the 200s differ (Preference-Applied) by the Prefer header,
        # and these are the public-cacheable responses
        return {"Cache-Control": "public, max-age=3600",
                "ETag": etag_of(key, result), "Vary": "Prefer"}

    def _pref_headers(request, key: str, result=None) -> dict:
        """Resource headers + the RFC 7240 echo of the token the client
        SENT (respond-async is echoed as itself, never rewritten to
        wait=0); applied uniformly - labels and artifacts, waits and
        cache hits alike (a hit trivially satisfies wait=N)."""
        h = _resource_headers(key, result)
        for raw in request.headers.getlist("prefer"):
            for token in raw.split(","):
                token = token.strip().lower()
                if token == "respond-async":
                    h["Preference-Applied"] = "respond-async"
                    return h
                if token.startswith("wait="):
                    w = _prefer_wait_raw(request, wait_max)
                    if w is not None:
                        h["Preference-Applied"] = f"wait={int(w)}"
                    return h
        return h

    # The explicit HEAD registration is the compute-free probe, and the one
    # place status codes distinguish in-flight: 200 materialized / 202
    # computing / 404 absent. (FastAPI's APIRoute does NOT auto-add HEAD to
    # GETs - registering probes first is belt-and-braces, not load-bearing.)
    def _mount_source(prefix: str, srcobj) -> None:
        """The path surface for one data source: probe / blocking GET / evict /
        meta / preview, all parameterized by the source's prefix, identifier
        pattern, and identity. Routes are greedy catch-alls parsed
        RIGHT-TO-LEFT: the artifact name is fixed and the task is exactly one
        segment, so everything between the prefix and them is the identifier -
        which is what lets multi-segment identifiers (hf repo@sha paths,
        openneuro file paths, zenodo recid/file!member) live in ordinary URLs
        with no percent-encoded slashes for proxies to mangle. The source's
        id_pattern remains the validation boundary either way."""
        pat = srcobj.id_pattern
        base = f"/v1/{prefix}/{{ident:path}}/{{task}}"

        def norm(ident: str) -> str:
            ident = ident.strip()
            return ident.lower() if prefix == "idc" else ident

        def keyed(ident: str, task: str, opts: dict | None = None):
            canonical = canon_task(task)
            if not re.fullmatch(pat, ident) or canonical is None:
                return None
            return derive_key(srcobj.identity(ident), canonical, opts or {})

        def _grid_routes(register):
            """Register an artifact route for the default grid and each token:
            labels.seg.nrrd plus labels.1mm.seg.nrrd and friends - the token
            keys (and computes) under its canonical options."""
            register("", {})
            for tok, opts in GRID_TOKENS.items():
                register("_" + tok, dict(opts))

        def _register_probe(tok: str, gopts: dict):
            @app.head(base + f"/labels{tok}.seg.nrrd")
            def probe(request: Request, ident: str, task: str):
                from fastapi import Response
                key = keyed(norm(ident), task, gopts)
                if key is None:
                    raise HTTPException(404, "unknown resource")
                hit = executor.cache_get(key)
                if hit is not None:
                    return Response(status_code=200, headers=_resource_headers(key))
                jid = executor.find_inflight(key)
                if jid is not None:
                    # anonymous callers see this too (user decision): watching a
                    # flight for public data is harmless - check back later
                    snap = executor.status_of(jid) or {}
                    return Response(status_code=202,
                                    headers=_progress_headers(snap.get("progress")))
                raise HTTPException(404, "not materialized")

        _grid_routes(_register_probe)

        def _register_resource(tok: str, gopts: dict):
            @app.get(base + f"/labels{tok}.seg.nrrd")
            async def resource(request: Request, ident: str, task: str):
                ident = norm(ident)
                if not re.fullmatch(pat, ident):
                    raise HTTPException(422, f"{ident!r} is not a valid {prefix} identifier")
                canonical = canon_task(task)
                if canonical is None:
                    raise HTTPException(404, f"unknown task {task!r}")
                task = canonical
                key = derive_key(srcobj.identity(ident), task, gopts)
                fname = f"{_task_stem(task)}_{ident[:8]}{tok}.seg.nrrd"
                # Cache-Control: no-cache from an authorized caller skips the
                # lookup; everything below is the ordinary miss path, so the
                # anonymous rule and "a plain GET is a read" still hold - forcing
                # a recompute from this URL also needs Prefer: wait.
                skip = (wants_no_cache(request) and authed(request)) or \
                    not engine_serves_from_cache(task)
                hit = None if skip else executor.cache_get(key)
                if hit is not None:
                    headers = _pref_headers(request, key, hit[1])
                    # the client may already hold these exact bytes
                    fresh = not_modified(request, headers["ETag"])
                    if fresh is not None:
                        return fresh
                    return FileResponse(hit[0], media_type="application/octet-stream",
                                        filename=fname, headers=headers)
                jid = executor.find_inflight(key)  # single flight: ride an existing run
                is_authed = authed(request)
                if not is_authed and jid is None:
                    raise HTTPException(404, "not materialized; authenticated access can "
                                             "compute it")
                initiated = False
                if jid is None:                # authed, nothing running
                    # Labels follow the same rule as every derived artifact
                    # (user decision 2026-08-25, closing review F8): the
                    # Prefer header expresses the intent to compute, its
                    # duration the patience. A plain GET is a read.
                    if _prefer_wait_raw(request, wait_max) is None:
                        raise HTTPException(404, "not materialized; send Prefer: wait to "
                                                 "compute it from this URL (wait=0 fires "
                                                 "the job and returns 202 immediately)")
                    if not _source_enabled(srcobj):
                        raise HTTPException(404, "not materialized, and this server cannot "
                                                 f"fetch {prefix} data (missing dependency)")
                    initiated = True
                    srcdict = {"kind": prefix, "id": ident}
                    if prefix == "idc":
                        srcdict["crdc_series_uuid"] = ident
                    # The same door check submit does. This surface initiates a
                    # compute too, so validating on only one of them lets the two
                    # disagree about what is acceptable - a task refused at
                    # POST /v1/jobs would still run from here. Before
                    # new_job_dir(), so a refusal allocates nothing.
                    _validate_request(seg, task, [srcdict], dict(gopts))
                    jid, jdir = executor.new_job_dir()
                    try:
                        executor.submit(jid, jdir, None, task, dict(gopts),
                                        source=[srcdict],
                                        identity=(srcobj.identity(ident),),
                                        source_tokens=source_tokens_of(request))
                    except QueueFull as e:
                        import shutil
                        shutil.rmtree(jdir, ignore_errors=True)
                        raise HTTPException(429, str(e),
                                            headers={"Retry-After": "30"}) from e
                wait = _prefer_wait(request, wait_default, wait_max)
                deadline = time.time() + wait
                snap = executor.status_of(jid) or {}
                while snap.get("state") not in TERMINAL and time.time() < deadline:
                    await asyncio.sleep(0.5)
                    nxt = executor.status_of(jid)
                    if nxt is None:        # record purged mid-watch: the cache
                        if executor.cache_get(key) is not None:
                            nxt = {"state": "done"}
                        else:              # job gone, bytes gone - the honest
                                           # answer is absence, NOT a
                                           # synthesized failure (the job may
                                           # have succeeded under no cache)
                            raise HTTPException(404, "not materialized")
                    snap = nxt
                hit = executor.cache_get(key)
                if snap.get("state") == "done" or hit is not None:
                    # a hit wins even against a failed/cancelled marker:
                    # under duplicate flights the sibling may have published
                    headers = _pref_headers(request, key, hit[1] if hit else None)
                    src_path = hit[0] if hit else executor.result_file(jid)[1]
                    if src_path is None:       # evicted between done and read
                        raise HTTPException(404, "not materialized")
                    fresh = not_modified(request, headers["ETag"])
                    if fresh is not None:
                        return fresh
                    return FileResponse(src_path, media_type="application/octet-stream",
                                        filename=fname, headers=headers)
                if snap.get("state") == "failed":
                    if not is_authed:          # for anonymous the resource just is not there
                        raise HTTPException(404, "not materialized")
                    raise HTTPException(502, f"segmentation failed: {snap.get('error')}")
                if not is_authed:              # watcher's view: no job vocabulary
                    p = snap.get("progress") or {}
                    return JSONResponse({"state": "materializing",
                                         "progress": {"stage": p.get("stage"),
                                                      "fraction": p.get("fraction")}},
                                        status_code=202,
                                        headers=_progress_headers(snap.get("progress")))
                return JSONResponse({"state": snap.get("state", "queued"), "job": jid,
                                     "initiated": initiated,  # did THIS request start it?
                                     "progress": snap.get("progress")},
                                    status_code=202,
                                    headers=_progress_headers(snap.get("progress")))

        _grid_routes(_register_resource)

        def _register_evict(tok: str, gopts: dict):
            paths = [base + f"/labels{tok}.seg.nrrd"]

            def evict(request: Request, ident: str, task: str):
                """Evict the cached entry (authorized only). Also cancels any
                in-flight single-flight compute for the same key - otherwise
                the entry would repopulate moments after being cleared.
                DELETE + GET = recompute with whatever is installed now; the
                jobs API's no_cache is the per-job form."""
                require_auth(request)
                key = keyed(norm(ident), task, gopts)
                if key is None:
                    raise HTTPException(404, "unknown resource")
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

            for pth in paths:
                app.delete(pth)(evict)

        _grid_routes(_register_evict)

        # The bare-task DELETE alias registers LAST: its greedy ident would
        # otherwise swallow the variant URLs, capturing labels.1mm.seg.nrrd
        # as the task segment.
        @app.delete(base)
        def evict_bare(request: Request, ident: str, task: str):
            require_auth(request)
            key = keyed(norm(ident), task, {})
            if key is None:
                raise HTTPException(404, "unknown resource")
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

        def _register_meta(tok: str, gopts: dict):
            @app.get(base + f"/meta{tok}.json")
            def meta(ident: str, task: str):
                key = keyed(norm(ident), task, gopts)
                if key is None:
                    raise HTTPException(404, "unknown resource")
                hit = executor.cache_get(key)
                if hit is None:
                    raise HTTPException(404, "not materialized")
                return JSONResponse(hit[1], headers=_resource_headers(key))

        _grid_routes(_register_meta)

        def _register_preview(tok: str, gopts: dict):
            @app.get(base + f"/preview{tok}.png")
            async def preview(request: Request, ident: str, task: str):
                key = keyed(norm(ident), task, gopts)
                if key is None:
                    raise HTTPException(404, "unknown resource")
                w = _prefer_wait_raw(request, wait_max)
                deadline = None if w is None else time.time() + w
                hit = executor.cache_get(key)
                if hit is None:
                    hit = await _materialize_entry(request, norm(ident),
                                                   canon_task(task), gopts, key,
                                                   deadline)
                png = await _await_artifact(request, key, hit,
                                            "preview.png", "preview",
                                            deadline=deadline)
                return FileResponse(png, media_type="image/png",
                                    headers=_pref_headers(request, key))

        _grid_routes(_register_preview)

        async def _materialize_entry(request, ident: str, task: str, opts: dict,
                                     key: str, deadline: float):
            """Initiate the segmentation chain from an artifact GET - the
            refined rule (user decision): authorized callers with explicit
            intent (Prefer: wait) may materialize the whole dependency chain
            from any derived resource; implicit reads (no Prefer - e.g. bulk
            <img> preview embeds) never compute, and anonymous never computes
            anywhere. WATCHING an already-running flight is read-only and
            open to anyone, as on the labels URL - the gates guard initiation
            only. (Deliberate asymmetry: without Prefer this returns
            immediately - implicit reads never hold a connection - while the
            labels URL holds wait_default for watchers.) Returns the cache
            hit, or raises 202/404/429."""
            jid = executor.find_inflight(key)
            if jid is None:
                if not authed(request):
                    raise HTTPException(404, "not materialized; authenticated "
                                             "access can compute it")
                if deadline is None:
                    raise HTTPException(404, "not materialized; send Prefer: wait "
                                             "to compute it from this URL (wait=0 "
                                             "fires the job and returns 202 "
                                             "immediately)")
                if not _source_enabled(srcobj):
                    raise HTTPException(404, "not materialized, and this server "
                                             f"cannot fetch {prefix} data")
                srcdict = {"kind": prefix, "id": ident}
                if prefix == "idc":
                    srcdict["crdc_series_uuid"] = ident
                # the same door check submit does - this surface initiates a
                # compute too, and validating on only one door lets them disagree
                _validate_request(seg, task, [srcdict], dict(opts))
                jid, jdir = executor.new_job_dir()
                try:
                    executor.submit(jid, jdir, None, task, dict(opts),
                                    source=[srcdict],
                                    identity=(srcobj.identity(ident),),
                                    source_tokens=source_tokens_of(request))
                except QueueFull as e:
                    import shutil
                    shutil.rmtree(jdir, ignore_errors=True)
                    raise HTTPException(429, str(e),
                                        headers={"Retry-After": "30"}) from e
            snap = executor.status_of(jid) or {}
            wait_until = time.time() if deadline is None else deadline
            while snap.get("state") not in TERMINAL and time.time() < wait_until:
                await asyncio.sleep(0.5)
                nxt = executor.status_of(jid)
                if nxt is None:            # record purged mid-watch
                    if executor.cache_get(key) is not None:
                        nxt = {"state": "done"}
                    else:                  # job gone, bytes gone: absence,
                                           # never a synthesized failure
                        raise HTTPException(404, "not materialized")
                snap = nxt
            hit = executor.cache_get(key)
            if hit is not None:            # a hit wins even against a failed/
                return hit                 # cancelled marker (duplicate flights)
            st = snap.get("state")
            if st == "cancelled":          # terminal without a result: absence
                raise HTTPException(404, "not materialized")   # (202 here looped
            if st == "failed":                                 # a poller forever)
                if not authed(request):    # watcher's view: it just is not there
                    raise HTTPException(404, "not materialized")
                raise HTTPException(502, f"segmentation failed: {snap.get('error')}")
            raise HTTPException(202, detail="materializing",
                                headers=_progress_headers(snap.get("progress"),
                                                          {"Retry-After": "5"}))

        async def _await_artifact(request, key, hit, filename: str, what: str,
                                  deadline: float | None = None):
            """The artifact file, waiting out a pending overlap thread. 202 +
            Retry-After while pending (or Prefer: wait exhausted); 404 only
            when its absence is definitive. ``deadline`` is the REQUEST's one
            Prefer budget, shared with the materialize leg - two legs must
            not each spend wait_max."""
            path = Path(hit[0]).parent / filename
            if path.exists():
                return path
            state_fn = getattr(executor, "artifact_state", None)
            pending = state_fn is not None and state_fn(key) == "pending"
            if pending:
                if deadline is None:
                    deadline = time.time()
                while time.time() < deadline:
                    await asyncio.sleep(0.2)
                    if path.exists():
                        return path
                    if state_fn(key) != "pending":
                        break
                if path.exists():
                    return path
                if state_fn(key) == "pending":
                    raise HTTPException(202,
                                        headers=_progress_headers(
                                            None, {"Retry-After": "2"}),
                                        detail=f"{what} still materializing")
            if path.exists():                  # placed between our probe and the
                return path                    # pending flag clearing
            raise HTTPException(404, f"no {what} for this result")

        def _register_statistics(tok: str, gopts: dict):
            async def _stats_key(request, ident, task, opts):
                key = keyed(norm(ident), task, opts)
                if key is None:
                    raise HTTPException(404, "unknown resource")
                w = _prefer_wait_raw(request, wait_max)
                deadline = None if w is None else time.time() + w
                hit = executor.cache_get(key)
                if hit is None:
                    hit = await _materialize_entry(request, norm(ident),
                                                   canon_task(task), opts, key,
                                                   deadline)
                return key, hit, deadline

            @app.get(base + f"/statistics{tok}.json")
            async def statistics_json(request: Request, ident: str, task: str):
                key, hit, deadline = await _stats_key(request, ident, task, gopts)
                sj = await _await_artifact(request, key, hit,
                                           "statistics.json", "statistics",
                                           deadline=deadline)
                return JSONResponse(json.loads(sj.read_text()),
                                    headers=_pref_headers(request, key))

            @app.get(base + f"/statistics{tok}.tsv")
            async def statistics_tsv_view(request: Request, ident: str, task: str):
                from fastapi import Response
                from .statistics import statistics_tsv
                key, hit, deadline = await _stats_key(request, ident, task, gopts)
                sj = await _await_artifact(request, key, hit,
                                           "statistics.json", "statistics",
                                           deadline=deadline)
                return Response(statistics_tsv(json.loads(sj.read_text())),
                                media_type="text/tab-separated-values",
                                headers=_pref_headers(request, key))

        _grid_routes(_register_statistics)

    for _prefix, _srcobj in sources.items():
        _mount_source(_prefix, _srcobj)

    if read_only:
        # The read-only surface is the SAME routes minus everything that
        # writes: jobs, prepare, describe (needs a real segmenter), every
        # DELETE (path-surface evicts + job cancel), and the listing when
        # no lister was provided. Removing them from the router is exactly
        # never-registering them - a POST /v1/jobs 404s, it does not 401.
        def _keep(r):
            path = getattr(r, "path", "")
            if path.startswith("/v1/jobs") or path in ("/v1/tasks/{task}",
                                                       "/v1/tasks/{task}/prepare"):
                return False
            if "DELETE" in (getattr(r, "methods", None) or ()):
                return False
            if path == "/v1/segmentations" and getattr(executor, "cache_list",
                                                       None) is None:
                return False
            return True

        app.router.routes = [r for r in app.router.routes if _keep(r)]

    return app


def create_public_app(key_fn, cache_get, tasks_fn, inflight=None, sources=None,
                      list_fn=None, resolve_fn=None):
    """The anonymous read-only twin: cache hits and nothing else.

    Since review R4 this is create_app itself over a CacheOnlyExecutor with
    ``read_only=True`` - the same route code as the main app, so filenames,
    headers, grid variants, meta/preview/statistics routes and the listing
    hold parity by construction instead of by copy. Read-only means: authed()
    is constant-False (nothing can initiate) and every mutating route (jobs,
    prepare, DELETEs) is absent from the router - it cannot spend a GPU cent
    by construction, the right worst case for a public face. Misses are 404;
    ``inflight(key)`` (optional) lets anonymous callers watch an authorized
    flight with 202 + progress and long-poll it with Prefer: wait - watching
    is read-only, only starting needs auth.

    ``key_fn(identity, task, opts)`` maps a source identity string
    ("idc:<uuid>") and grid options to the result-cache key - the twin must
    key exactly as the writer did. ``list_fn`` (optional) backs
    /v1/segmentations; omit it to keep the twin listing-free.
    """
    ex = CacheOnlyExecutor(cache_get, key_fn, tasks_fn, inflight_fn=inflight,
                           resolve_fn=resolve_fn, list_fn=list_fn,
                           sources=sources)
    return create_app(ex, read_only=True)


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
