"""The job record store: sqlite, because the queue was already a database.

`LocalExecutor` kept five hand-maintained indexes over one set of records -
``_jobs`` (id -> record), ``_pending`` (FIFO of queued ids), ``_done_order``
(FIFO for eviction), ``_inflight`` (cache key -> active job, for single-flight
dedup) and ``_artifacts_pending`` - held consistent by hand across eighteen
``with self._cv`` blocks. Every one of those is a query:

    SELECT id FROM jobs WHERE state='queued' ORDER BY created           -- pending
    SELECT id FROM jobs WHERE cache_key=? AND state IN (...)            -- inflight
    DELETE FROM jobs WHERE state IN (terminal) AND finished < ?         -- the reaper

So this is not persistence bolted onto a queue; it is the queue's own
bookkeeping, written down once instead of maintained five times. sqlite is
stdlib, gives real transactions for state transitions, and - the part that
matters later - lets a second process read job state under WAL instead of
needing an IPC protocol invented for it.

**Records only, never bytes.** Inputs live in the content store and results in
the result cache, both content- or request-addressed DIRECTORIES that stay
inspectable and mountable. This holds the bookkeeping, which is the part that
wants transactions.

**Why durability is worth having here.** Jobs are content-keyed and idempotent -
identity x task x options x weights determines the result - so a job interrupted
by a restart can simply be run again. That is a retry, not a zombie, and it is
the same property that makes the result cache safe. Losing them is not free
either: `total` on an M2 was measured at 25.8 minutes.

Three fields never persist. ``cancel_token`` and ``subscribers`` are live
objects, and ``source_tokens`` are per-request credentials the wire marks
"never serialized" - so a job fetching from an authenticated source cannot be
resumed, and says so rather than retrying into a 401.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import warnings
import time
from pathlib import Path

TERMINAL = ("done", "failed", "cancelled")
LIVE = ("queued", "running")

#: Columns that are queried or ordered on live in their own fields; everything
#: else rides in one JSON blob, so adding a record field needs no migration.
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    task       TEXT NOT NULL,
    state      TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'segment',
    cache_key  TEXT,
    created    REAL NOT NULL,
    started    REAL,
    finished   REAL,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, created);
CREATE INDEX IF NOT EXISTS jobs_key   ON jobs(cache_key, state);
"""


class JobStore:
    """Durable job records. Thread-safe; safe for a second process to read."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        try:
            self._open()
        except sqlite3.DatabaseError as e:
            # A corrupt store is a LOST store, not a dead server. Records are
            # semi-transient - losing them costs resubmits, and a resubmit is a
            # digest now that inputs are content-addressed - so a file we cannot
            # open is moved aside and a fresh one started. Before durability a
            # corrupt file was impossible; adding it must not make a crash
            # mid-write able to brick the server it was meant to protect.
            spoiled = self.path.with_suffix(self.path.suffix + ".corrupt")
            try:
                self.path.replace(spoiled)
            except OSError:
                self.path.unlink(missing_ok=True)
                spoiled = None
            warnings.warn(
                f"job store at {self.path} could not be opened ({e}); starting a "
                f"new one" + (f" - the old file is at {spoiled}" if spoiled else ""),
                stacklevel=2)
            self._open()

    def _open(self) -> None:
        # WAL: a reader (another process, the CLI, a future sidecar) never
        # blocks the dispatcher, and a crash leaves a consistent database rather
        # than one this module would have to repair.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    # -- writing ------------------------------------------------------------
    def put(self, rec: dict) -> None:
        """Insert or replace one record. ``rec`` is the serializable view."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO jobs"
                " (id, task, state, kind, cache_key, created, started, finished, payload)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (rec["id"], rec.get("task") or "", rec.get("state") or "queued",
                 rec.get("kind") or "segment", rec.get("cache_key"),
                 float(rec.get("created") or time.time()),
                 rec.get("started"), rec.get("finished"), json.dumps(rec)))
            self._db.commit()

    def drop(self, jid: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id = ?", (jid,))
            self._db.commit()

    # -- reading ------------------------------------------------------------
    # All of these take the same lock the writers do. Python's sqlite3 is
    # usually built SERIALIZED (threadsafety 3), which would make sharing one
    # connection between the dispatcher thread and the ASGI threads safe - but
    # that is a property of how the interpreter was compiled, not of this code,
    # and at a few operations per second the lock costs nothing.
    def get(self, jid: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM jobs WHERE id = ?", (jid,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def all(self, limit: int = 500) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM jobs ORDER BY created DESC LIMIT ?",
                (limit,)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def in_state(self, states) -> list:
        q = ",".join("?" * len(states))
        with self._lock:
            rows = self._db.execute(
                f"SELECT payload FROM jobs WHERE state IN ({q}) ORDER BY created",
                tuple(states)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def inflight(self, cache_key: str) -> str | None:
        """The live job already computing this key - single-flight dedup, as a
        query rather than a dict kept in step with three others."""
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM jobs WHERE cache_key = ? AND state IN (?,?)"
                " ORDER BY created LIMIT 1", (cache_key, *LIVE)).fetchone()
        return row["id"] if row else None

    def counts(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT state, COUNT(*) n FROM jobs GROUP BY state").fetchall()
        return {r["state"]: r["n"] for r in rows}

    # -- lifecycle ----------------------------------------------------------
    def reap(self, ttl_s: float, now: float | None = None) -> list:
        """Drop terminal records older than the TTL; returns their ids so the
        caller can remove the directories they own. The same policy Modal's
        jobs store already runs, so the two substrates agree."""
        now = now if now is not None else time.time()
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM jobs WHERE state IN ({','.join('?' * len(TERMINAL))})"
                " AND COALESCE(finished, created) < ?", (*TERMINAL, now - ttl_s)
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._db.execute(
                    f"DELETE FROM jobs WHERE id IN ({','.join('?' * len(ids))})", ids)
                self._db.commit()
        return ids

    def reconcile(self) -> list:
        """Startup: decide what each surviving record means now.

        A job that was RUNNING when the process died cannot be resumed - the GPU
        work is gone - but it can be re-run, because the request is idempotent
        and content-keyed. So it goes back to queued rather than being marked
        failed, EXCEPT when it needed credentials, which are deliberately not
        persisted: that one fails with a message telling the caller to resubmit.

        Returns the records that are runnable again, oldest first.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM jobs WHERE state IN (?,?) ORDER BY created", LIVE
            ).fetchall()
            resumable, cursor = [], self._db
            for r in rows:
                rec = json.loads(r["payload"])
                if rec.get("needed_credentials"):
                    rec["state"], rec["error"] = "failed", (
                        "interrupted by a server restart; its source needed "
                        "credentials, which are never persisted - resubmit it")
                    rec["finished"] = time.time()
                else:
                    rec["state"], rec["started"], rec["progress"] = "queued", None, None
                    resumable.append(rec)
                cursor.execute(
                    "UPDATE jobs SET state=?, started=?, finished=?, payload=?"
                    " WHERE id=?",
                    (rec["state"], rec.get("started"), rec.get("finished"),
                     json.dumps(rec), rec["id"]))
            self._db.commit()
        return resumable

    def close(self) -> None:
        with self._lock:
            self._db.close()
