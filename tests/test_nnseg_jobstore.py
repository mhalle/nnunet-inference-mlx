"""The job record store: the queue's bookkeeping, written down once.

What is tested here is the behaviour the five hand-maintained indexes used to
provide - FIFO order, single-flight dedup, eviction - now that each is a query,
plus the one thing they could never provide: surviving the process.
"""
import time

import pytest

from nnseg.jobstore import JobStore


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


def _rec(jid, state="queued", *, key=None, created=None, task="ts:total_fast", **kw):
    return {"id": jid, "task": task, "state": state, "cache_key": key,
            "created": created if created is not None else time.time(), **kw}


def test_pending_comes_back_in_submission_order(store):
    """FIFO was a deque kept in step with a dict and another deque; it is now
    ORDER BY created, which cannot disagree with itself."""
    now = time.time()
    for i, jid in enumerate(("c", "a", "b")):
        store.put(_rec(jid, created=now + i))
    assert [r["id"] for r in store.in_state(["queued"])] == ["c", "a", "b"]


def test_single_flight_is_a_query(store):
    """Two requests for one key ride one job. This was a dict that had to be
    updated in step with the record's state; now the state IS the answer."""
    store.put(_rec("j1", "running", key="K"))
    assert store.inflight("K") == "j1"
    store.put(_rec("j1", "done", key="K"))
    assert store.inflight("K") is None          # terminal is not in flight
    assert store.inflight("nosuch") is None


def test_a_restart_re_queues_work_rather_than_losing_it(store):
    """A job that was RUNNING cannot be resumed - the GPU work is gone - but it
    can be re-run: the request is idempotent and content-keyed, which is the
    same property that makes the result cache safe."""
    store.put(_rec("was_running", "running", key="K"))
    store.put(_rec("was_queued", "queued", key="L"))
    store.put(_rec("was_done", "done", key="M"))
    resumable = store.reconcile()
    assert sorted(r["id"] for r in resumable) == ["was_queued", "was_running"]
    assert store.get("was_running")["state"] == "queued"
    assert store.get("was_running")["started"] is None      # and its clock reset
    assert store.get("was_done")["state"] == "done"         # terminal is untouched


def test_a_credentialed_job_fails_loudly_instead_of_retrying_into_a_401(store):
    """source_tokens are deliberately never persisted, so a job that needed
    them cannot be replayed. Saying so beats retrying and failing obscurely."""
    store.put(_rec("cred", "running", key="K", needed_credentials=True))
    assert store.reconcile() == []
    rec = store.get("cred")
    assert rec["state"] == "failed" and "resubmit" in rec["error"]


def test_the_reaper_drops_terminal_records_past_the_ttl_and_nothing_else(store):
    now = time.time()
    store.put(_rec("old_done", "done", created=now - 90000, finished=now - 89000))
    store.put(_rec("new_done", "done", created=now - 10, finished=now - 5))
    store.put(_rec("old_queued", "queued", created=now - 90000))   # live: never reaped
    assert store.reap(24 * 3600, now=now) == ["old_done"]
    assert {r["id"] for r in store.all()} == {"new_done", "old_queued"}


def test_records_survive_reopening_the_database(tmp_path):
    p = tmp_path / "jobs.db"
    JobStore(p).put(_rec("j", "queued", key="K"))
    assert JobStore(p).get("j")["cache_key"] == "K"


def test_a_new_record_field_needs_no_migration(store):
    """Queried columns are columns; everything else rides in one JSON payload,
    so JobRecord can grow without a schema change."""
    store.put(_rec("j", "done", something_new={"a": 1}))
    assert store.get("j")["something_new"] == {"a": 1}


def test_counts_by_state(store):
    store.put(_rec("a", "queued")); store.put(_rec("b", "queued"))
    store.put(_rec("c", "done"))
    assert store.counts() == {"queued": 2, "done": 1}


def test_a_corrupt_store_is_a_lost_store_not_a_dead_server(tmp_path):
    """Durability must not make a crash mid-write able to brick the server it
    was added to protect. Records are semi-transient - losing them costs
    resubmits, and a resubmit is a digest now that inputs are content-addressed
    - so an unopenable file is moved aside and a fresh one started."""
    import warnings
    p = tmp_path / "jobs.db"
    p.write_bytes(b"not a database, but what a crash mid-write leaves behind")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = JobStore(p)
    store.put(_rec("j"))
    assert store.get("j") is not None                     # it works
    assert caught and "could not be opened" in str(caught[0].message)
    assert p.with_suffix(".db.corrupt").exists()          # and the evidence is kept


def test_the_corrupt_file_is_preserved_rather_than_deleted(tmp_path):
    """Whatever went wrong is worth more as a file than as free disk."""
    p = tmp_path / "jobs.db"
    p.write_bytes(b"CORRUPT-MARKER")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        JobStore(p)
    assert p.with_suffix(".db.corrupt").read_bytes() == b"CORRUPT-MARKER"
