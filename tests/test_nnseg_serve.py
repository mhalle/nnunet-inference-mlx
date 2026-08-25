"""The REST job protocol end to end - queue semantics, SSE, and the HTTP surface.

A fake segmenter stands in for the GPU: it ticks progress through a Reporter, honors
the cancel token, and returns a Segmentation-shaped result, so the whole contract runs
in milliseconds. The real pipeline behind the same seam is exercised by the CUDA
harness, not here.
"""
import json
import threading
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nnseg.errors import Cancelled
from nnseg.progress import Reporter
from nnseg.serve import LocalExecutor, QueueFull, create_app


class FakeSeg:
    """Duck-typed Segmentation: enough for the dispatcher's result handling."""

    class _Schema:
        names = {0: "background", 1: "spleen", 2: "kidney_right"}

    schema = _Schema()
    provenance = {"device": "fake"}

    def volumes_ml(self):
        return {"spleen": 210.0}

    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"\x1f\x8b" + b"fake-nifti")
        return path


class FakeSegmenter:
    """Catalog + a controllable segment(): waits on `gate` if set, ticks n_steps."""

    def __init__(self, gate=None, steps=3, fail=False):
        self.gate, self.steps, self.fail = gate, steps, fail
        self.calls = []
        self.policy = {"device": "fake"}

    def tasks(self):
        return ["total_fast", "total"]

    def describe(self, task):
        if task not in self.tasks():
            raise KeyError(task)
        return {"name": task, "structures": ["spleen", "kidney_right"]}

    def segment(self, image, task, *, progress=None, cancel=None, **options):
        self.calls.append((str(image), task, options))
        rep = Reporter.of(progress, cancel=cancel)
        if self.gate is not None:
            self.gate.wait(timeout=5)
        if self.fail:
            raise RuntimeError("synthetic failure")
        for i in range(self.steps):
            rep.tick(i + 1, self.steps)          # checks the cancel token
            time.sleep(0.01)
        return FakeSeg()


def make(tmp_path, **kw):
    seg = FakeSegmenter(**{k: v for k, v in kw.items() if k in ("gate", "steps", "fail")})
    ex = LocalExecutor(seg, workdir=tmp_path,
                       max_pending=kw.get("max_pending", 4),
                       keep_finished=kw.get("keep_finished", 50))
    return seg, ex, TestClient(create_app(ex))


def submit(client, task="total_fast", options=None):
    r = client.post("/v1/jobs", files={"file": ("scan.nii.gz", b"\x1f\x8bdata")},
                    data={"task": task, "options": json.dumps(options or {})})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def wait_artifact(client, url, timeout=5.0):
    """Artifacts are eventually consistent (overlap thread): poll briefly."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(url)
        if r.status_code == 200:
            return r
        time.sleep(0.02)
    return r


def wait_state(client, jid, want=("done", "failed", "cancelled"), timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/v1/jobs/{jid}").json()
        if s["state"] in want:
            return s
        time.sleep(0.01)
    raise AssertionError(f"timeout waiting for {want}; last {s}")


# -- happy path --------------------------------------------------------------

def test_submit_run_result_roundtrip(tmp_path):
    seg, ex, client = make(tmp_path)
    jid = submit(client, options={"interp": "nearest"})
    s = wait_state(client, jid, ("done",))
    assert s["result"]["volumes_ml"] == {"spleen": 210.0}
    assert s["result"]["names"]["1"] == "spleen"          # JSON object keys are strings
    assert s["progress"]["fraction"] > 0
    assert seg.calls[0][1] == "total_fast"
    assert seg.calls[0][2] == {"interp": "nearest"}
    r = client.get(f"/v1/jobs/{jid}/result")
    assert r.status_code == 200 and r.content.startswith(b"\x1f\x8b")


def test_health_and_tasks(tmp_path):
    _, _, client = make(tmp_path)
    h = client.get("/v1/health").json()
    assert h["name"] == "nnseg" and h["accepting"] is True
    assert client.get("/v1/tasks").json()["tasks"] == ["total_fast", "total"]
    assert client.get("/v1/tasks/total").status_code == 200
    assert client.get("/v1/tasks/nope").status_code == 404


# -- queue semantics ---------------------------------------------------------

def test_fifo_order_and_positions(tmp_path):
    gate = threading.Event()
    seg, ex, client = make(tmp_path, gate=gate)
    a, b, c = (submit(client) for _ in range(3))
    time.sleep(0.05)                                       # a is running, gated
    sb = client.get(f"/v1/jobs/{b}").json()
    sc = client.get(f"/v1/jobs/{c}").json()
    assert sb["state"] == "queued" and sb["queue_position"] == 0
    assert sc["state"] == "queued" and sc["queue_position"] == 1
    gate.set()
    for jid in (a, b, c):
        wait_state(client, jid, ("done",))
    order = [call[0] for call in seg.calls]
    assert [a in p for p in order] == [True, False, False]  # a ran first...
    assert [c in p for p in order] == [False, False, True]  # ...and c last


def test_queue_bound_gives_429(tmp_path):
    gate = threading.Event()
    _, ex, client = make(tmp_path, gate=gate, max_pending=2)
    submit(client)                                         # running (gated)
    time.sleep(0.05)
    submit(client); submit(client)                         # fills the 2-slot queue
    r = client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                    data={"task": "total_fast"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "30"
    gate.set()


def test_cancel_queued_is_instant_and_shifts_positions(tmp_path):
    gate = threading.Event()
    _, ex, client = make(tmp_path, gate=gate)
    a, b, c = (submit(client) for _ in range(3))
    time.sleep(0.05)
    assert client.delete(f"/v1/jobs/{b}").json()["state"] == "cancelled"
    assert client.get(f"/v1/jobs/{b}").json()["state"] == "cancelled"
    assert client.get(f"/v1/jobs/{c}").json()["queue_position"] == 0
    gate.set()
    wait_state(client, a, ("done",)); wait_state(client, c, ("done",))


def test_cancel_running_lands_cancelled(tmp_path):
    seg, ex, client = make(tmp_path, steps=200)
    jid = submit(client)
    wait_state(client, jid, ("running",))
    client.delete(f"/v1/jobs/{jid}")
    s = wait_state(client, jid, ("cancelled",))
    assert "error" not in s


def test_failure_is_reported(tmp_path):
    _, _, client = make(tmp_path, fail=True)
    jid = submit(client)
    s = wait_state(client, jid, ("failed",))
    assert "synthetic failure" in s["error"]
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 409


def test_delete_finished_removes_job_and_files(tmp_path):
    _, ex, client = make(tmp_path)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    jdir = ex.get(jid).dir
    assert jdir.exists()
    assert client.delete(f"/v1/jobs/{jid}").json()["deleted"] is True
    assert client.get(f"/v1/jobs/{jid}").status_code == 404
    assert not jdir.exists()


def test_eviction_bounds_finished_jobs(tmp_path):
    _, ex, client = make(tmp_path, keep_finished=2)
    ids = [submit(client) for _ in range(4)]
    for jid in ids:
        try:
            wait_state(client, jid, ("done",))
        except AssertionError:
            pass                                           # early ones may evict mid-wait
    time.sleep(0.1)
    alive = [j["id"] for j in client.get("/v1/jobs").json()["jobs"]]
    assert len(alive) == 2 and ids[-1] in alive


def test_bad_options_and_unknown_job(tmp_path):
    _, _, client = make(tmp_path)
    r = client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                    data={"task": "t", "options": "[1,2]"})
    assert r.status_code == 422
    assert client.get("/v1/jobs/nope").status_code == 404
    assert client.delete("/v1/jobs/nope").status_code == 404


# -- SSE ---------------------------------------------------------------------

def test_sse_streams_snapshots_to_terminal(tmp_path):
    _, _, client = make(tmp_path, steps=3)
    jid = submit(client)
    states, fractions = [], []
    with client.stream("GET", f"/v1/jobs/{jid}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data:"):
                snap = json.loads(line[5:])
                states.append(snap["state"])
                if snap.get("progress"):
                    fractions.append(snap["progress"]["fraction"])
                if snap["state"] in ("done", "failed", "cancelled"):
                    break
    assert states[-1] == "done"
    assert fractions == sorted(fractions) and len(fractions) >= 3


def test_executor_queuefull_raises(tmp_path):
    gate = threading.Event()
    seg = FakeSegmenter(gate=gate)
    ex = LocalExecutor(seg, workdir=tmp_path, max_pending=1)
    jid, jdir = ex.new_job_dir()
    ex.submit(jid, jdir, jdir / "x", "t", {})
    time.sleep(0.05)                                       # now running; queue empty
    j2, d2 = ex.new_job_dir()
    ex.submit(j2, d2, d2 / "x", "t", {})                   # fills the 1-slot queue
    j3, d3 = ex.new_job_dir()
    with pytest.raises(QueueFull):
        ex.submit(j3, d3, d3 / "x", "t", {})
    gate.set()


# -- wire-shape rules (catalog-only tasks, source descriptor, identities) ----

def test_unknown_task_is_404_with_examples(tmp_path):
    _, _, client = make(tmp_path)
    r = client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                    data={"task": "/etc/passwd"})
    assert r.status_code == 404
    assert "total_fast" in r.json()["detail"]      # examples, not just a refusal


def test_source_descriptor_validation(tmp_path):
    _, _, client = make(tmp_path)

    def post(source, with_file=True):
        files = {"file": ("x.nii.gz", b"d")} if with_file else None
        return client.post("/v1/jobs", files=files,
                           data={"task": "total_fast", "source": json.dumps(source)})

    assert post([{"kind": "upload"}, {"kind": "upload"}]).status_code == 422   # multi-channel
    assert post([{"kind": "url", "url": "http://x"}], with_file=False).status_code == 422
    assert "reserved" in post([{"kind": "url"}], with_file=False).json()["detail"]
    assert post([{"kind": "mystery"}]).status_code == 422
    assert post([{"kind": "upload", "part": "other"}]).status_code == 422
    assert post([{"kind": "upload"}], with_file=False).status_code == 422
    r = post([{"kind": "idc", "crdc_series_uuid": "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"}], with_file=True)
    assert r.status_code == 422                    # idc + an upload is contradictory


def test_upload_identity_is_sha256(tmp_path):
    import hashlib
    _, _, client = make(tmp_path)
    jid = submit(client)
    s = wait_state(client, jid, ("done",))
    want = "sha256:" + hashlib.sha256(b"\x1f\x8bdata").hexdigest()
    assert s["input_identity"] == [want]


def test_idc_source_fetches_at_dispatch(tmp_path, monkeypatch):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    fetched = {}

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir()
        (d / "slice.dcm").write_bytes(b"dcm")
        fetched["series"] = series
        return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "idc", "crdc_series_uuid": "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"}])})
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    s = wait_state(client, jid, ("done",))
    assert fetched["series"] == "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    assert s["input_identity"] == ["idc:0be27d1c-9410-47ff-9c9f-a44b26a4bd55"]
    assert seg.calls[0][0].endswith("series")      # segment saw the fetched directory


def test_multichannel_model_rejected_at_submit(tmp_path):
    seg = FakeSegmenter()
    seg.describe = lambda task: {"name": task, "channel_names": {"0": "T1", "1": "T2"}}
    ex = LocalExecutor(seg, workdir=tmp_path)
    client = TestClient(create_app(ex))
    r = client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                    data={"task": "total_fast"})
    assert r.status_code == 422
    assert "2 input channels" in r.json()["detail"]


def test_health_advertises_source_kinds(tmp_path):
    _, _, client = make(tmp_path)
    kinds = client.get("/v1/health").json()["sources"]
    assert "upload" in kinds                       # idc presence depends on the extra


def test_real_segmenter_describe_enrichment(tmp_path):
    from nnseg import Segmenter
    d = Segmenter(weights=tmp_path).describe("total_fast")   # empty root: nothing installed
    assert d["folds_default"] == [0]
    assert d["configuration"] is None
    assert d["weights_installed"] and all(e["installed"] is False for e in d["weights_installed"])
    assert d["channel_names"] is None


def test_idc_not_found_names_probed_buckets(tmp_path, monkeypatch):
    """Bucket drift must fail loudly and diagnosably, never as a bare not-found."""
    obstore_store = pytest.importorskip("obstore.store")

    class EmptyStore:
        @staticmethod
        def from_url(url, config=None):
            return EmptyStore()

        def list(self, prefix=None):
            return iter(())

    monkeypatch.setattr(obstore_store, "S3Store", EmptyStore)
    from nnseg.errors import InputError
    from nnseg.serve import IDC_BUCKETS, _fetch_idc_series
    with pytest.raises(InputError) as e:
        _fetch_idc_series("dead-beef", tmp_path)
    for bucket in IDC_BUCKETS:
        assert bucket in str(e.value)


def test_idc_identifier_fields_are_explicit(tmp_path, monkeypatch):
    """Identifier semantics live in field names, not in format sniffing: the
    ambiguous 'series' and the not-yet-supported 'series_instance_uid' each get a
    precise 422 at submit; a malformed crdc_series_uuid gets a format 422."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    _, _, client = make(tmp_path)

    def post(fields):
        return client.post("/v1/jobs", data={"task": "total_fast",
                           "source": json.dumps([{"kind": "idc", **fields}])})

    r = post({"series_instance_uid": "1.3.6.1.4.1.14519.5.2.1.2932.1975.25507"})
    assert r.status_code == 422 and "/v1/resolve" in r.json()["detail"]
    assert "idc_version" in r.json()["detail"]     # the reserved shape names its version slot
    r = post({"series_instance_uid": "1.3.6.1.4.1.14519", "idc_version": 21})
    assert r.status_code == 422 and "/v1/resolve" in r.json()["detail"]
    r = post({"crdc_series_uuid": "0be27d1c-9410-47ff-9c9f-a44b26a4bd55", "idc_version": 21})
    assert r.status_code == 422 and "already pinned" in r.json()["detail"]
    r = post({"series": "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"})
    assert r.status_code == 422 and "crdc_series_uuid" in r.json()["detail"]
    r = post({"crdc_series_uuid": "not-a-uuid-at-all"})
    assert r.status_code == 422 and "8-4-4-4-12" in r.json()["detail"]
    r = post({})
    assert r.status_code == 422


def test_sse_poll_branch_for_pushless_executors(tmp_path):
    """Executors without push (the ModalExecutor shape) stream via server-side
    polling of the same snapshots - same contract, different transport."""
    seg = FakeSegmenter(steps=3)
    ex = LocalExecutor(seg, workdir=tmp_path)
    ex.supports_push = False
    client = TestClient(create_app(ex))
    jid = submit(client)
    states = []
    with client.stream("GET", f"/v1/jobs/{jid}/events") as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                snap = json.loads(line[5:])
                states.append(snap["state"])
                if snap["state"] in ("done", "failed", "cancelled"):
                    break
    assert states[-1] == "done"


def test_real_segmenter_accepts_cancel_override(tmp_path, monkeypatch):
    """Regression from the first Modal smoke: Segmenter.segment's override
    whitelist rejected `cancel`, which pipeline.segment legitimately takes - both
    executors pass it, and only a fake segmenter let the tests miss it."""
    from nnseg import Segmenter, pipeline

    seen = {}
    monkeypatch.setattr(pipeline, "segment",
                        lambda image, task, **kw: seen.update(kw) or FakeSeg())
    token = object()
    Segmenter(weights=tmp_path).segment("x.nii.gz", "total_fast",
                                        cancel=token, progress=None)
    assert seen["cancel"] is token


# -- result cache, path surface, tiering -------------------------------------

def make_cached(tmp_path, **kw):
    seg = FakeSegmenter(**{k: v for k, v in kw.items() if k in ("gate", "steps", "fail")})
    ex = LocalExecutor(seg, workdir=tmp_path / "work", cache_dir=tmp_path / "cache",
                       max_pending=kw.get("max_pending", 4))
    return seg, ex, TestClient(create_app(ex, token=kw.get("token")))


def test_result_cache_hit_skips_compute(tmp_path):
    seg, ex, client = make_cached(tmp_path)
    a = submit(client)
    wait_state(client, a, ("done",))
    b = submit(client)                        # identical upload -> identical identity
    s = client.get(f"/v1/jobs/{b}").json()
    assert s["state"] == "done" and s["cached"] is True
    assert s["result"]["volumes_ml"] == {"spleen": 210.0}
    assert len(seg.calls) == 1                # computed once, served twice
    r = client.get(f"/v1/jobs/{b}/result")
    assert r.status_code == 200 and r.content.startswith(b"\x1f\x8b")


def test_no_cache_forces_recompute_and_options_key(tmp_path):
    seg, ex, client = make_cached(tmp_path)
    a = submit(client); wait_state(client, a, ("done",))
    b = submit(client, options={"no_cache": True}); wait_state(client, b, ("done",))
    assert len(seg.calls) == 2
    assert client.get(f"/v1/jobs/{b}").json().get("cached") is None
    c = submit(client, options={"interp": "nearest"})   # different options = new key
    wait_state(client, c, ("done",))
    assert len(seg.calls) == 3
    assert seg.calls[-1][2] == {"interp": "nearest"}    # no_cache never reaches segment


def test_idc_path_surface_blocking_and_cache(tmp_path, monkeypatch):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    seg = FakeSegmenter(steps=3)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"; d.mkdir(); (d / "s.dcm").write_bytes(b"d")
        return d

    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    r = client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd",
                   headers={"Prefer": "wait=10"})
    assert r.status_code == 200, r.text                 # blocked through the compute
    assert r.headers["preference-applied"] == "wait=10"
    assert "etag" in r.headers and r.content.startswith(b"\x1f\x8b")
    r2 = client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd")
    assert r2.status_code == 200 and len(seg.calls) == 1    # second read: pure cache
    meta = client.get(f"/v1/idc/{u}/total_fast/meta.json")
    assert meta.status_code == 200 and meta.json()["volumes_ml"] == {"spleen": 210.0}


def test_idc_path_wait_zero_gives_202(tmp_path, monkeypatch):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    gate = threading.Event()
    seg = FakeSegmenter(gate=gate)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"; d.mkdir(exist_ok=True); (d / "s.dcm").write_bytes(b"d")
        return d

    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    r = client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd",
                   headers={"Prefer": "respond-async"})
    assert r.status_code == 202
    assert r.headers["retry-after"] and r.headers["cache-control"] == "no-store"
    jid = r.json()["job"]
    r2 = client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd",
                    headers={"Prefer": "wait=0"})
    assert r2.status_code == 202 and r2.json()["job"] == jid   # single flight
    gate.set()
    wait_state(client, jid, ("done",))
    assert client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd").status_code == 200


def test_token_tiering_local(tmp_path, monkeypatch):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    seg = FakeSegmenter(steps=2)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"; d.mkdir(exist_ok=True); (d / "s.dcm").write_bytes(b"d")
        return d

    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex, token="s3cret"))
    auth = {"Authorization": "Bearer s3cret"}
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    # anonymous: health/tasks fine, jobs and compute-on-miss are not
    assert client.get("/v1/health").status_code == 200
    assert client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                       data={"task": "total_fast"}).status_code == 401
    assert client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd").status_code == 404
    assert len(seg.calls) == 0                          # the miss spent nothing
    # authed: materialize
    r = client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd",
                   headers={**auth, "Prefer": "wait=10"})
    assert r.status_code == 200
    # now anonymous reads the cache
    assert client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd").status_code == 200
    assert len(seg.calls) == 1


def test_public_app_is_read_only_by_construction(tmp_path):
    from nnseg.serve import ResultCache, create_public_app, result_key
    cache = ResultCache(tmp_path / "c")
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    key_fn = lambda identity, task, opts=None: result_key((identity,), task, opts or {}, ["w=1"])
    src = tmp_path / "labels.seg.nrrd"; src.write_bytes(b"\x1f\x8bx")
    cache.put(key_fn(f"idc:{u}", "total_fast"), src, {"volumes_ml": {"spleen": 1.0}}, {})
    app = create_public_app(key_fn, cache.get, lambda: ["total_fast"])
    client = TestClient(app)
    assert client.get("/v1/health").json()["mode"] == "public-cache"
    assert client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd").status_code == 200
    miss = client.get(f"/v1/idc/{'a'*8}-1111-2222-3333-{'b'*12}/total_fast/labels.seg.nrrd")
    assert miss.status_code == 404 and "authenticated" in miss.json()["detail"]
    assert client.post("/v1/jobs").status_code in (404, 405)   # no job routes exist


def test_idc_delete_evicts_and_cancels_inflight(tmp_path, monkeypatch):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    gate = threading.Event()
    seg = FakeSegmenter(gate=gate, steps=2)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"; d.mkdir(exist_ok=True); (d / "s.dcm").write_bytes(b"d")
        return d

    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex, token="s3cret"))
    auth = {"Authorization": "Bearer s3cret"}
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"

    assert client.delete(url).status_code == 401                  # anonymous: never
    assert client.delete(url, headers=auth).status_code == 404    # nothing yet

    gate.set()
    assert client.get(url, headers={**auth, "Prefer": "wait=10"}).status_code == 200
    r = client.delete(url, headers=auth)
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert client.delete(url, headers=auth).status_code == 404    # already gone
    assert client.get(url, headers={**auth, "Prefer": "wait=10"}).status_code == 200
    assert len(seg.calls) == 2                                    # recomputed after evict

    gate.clear()
    r202 = client.get(url, headers={**auth, "Prefer": "wait=0"})  # start compute #3... cached!
    assert r202.status_code == 200                                # cache hit from run 2
    client.delete(url, headers=auth)                              # clear again
    r202 = client.get(url, headers={**auth, "Prefer": "wait=0"})
    assert r202.status_code == 202                                # now computing, gated
    jid = r202.json()["job"]
    rd = client.delete(url, headers=auth)                         # delete mid-flight
    assert rd.json().get("cancelled_job") == jid
    gate.set()
    t0 = time.time()
    st = {}
    while time.time() - t0 < 5:
        st = client.get(f"/v1/jobs/{jid}", headers=auth).json()
        if st.get("state") in ("cancelled", "done"):
            break
        time.sleep(0.02)
    assert st.get("state") in ("cancelled", "done")


def test_head_probe_never_computes_and_distinguishes_states(tmp_path, monkeypatch):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    gate = threading.Event()
    seg = FakeSegmenter(gate=gate, steps=2)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"; d.mkdir(exist_ok=True); (d / "s.dcm").write_bytes(b"d")
        return d

    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"

    assert client.head(url).status_code == 404      # absent - and crucially:
    assert len(seg.calls) == 0                      # the probe started NOTHING

    r = client.get(url, headers={"Prefer": "wait=0"})   # initiate (gated)
    assert r.status_code == 202 and r.json()["initiated"] is True
    r2 = client.get(url, headers={"Prefer": "wait=0"})  # join the flight
    assert r2.status_code == 202 and r2.json()["initiated"] is False
    assert r2.json()["job"] == r.json()["job"]
    assert client.head(url).status_code == 202          # probe sees in-flight

    gate.set()
    jid = r.json()["job"]
    wait_state(client, jid, ("done",))
    assert client.head(url).status_code == 200          # probe sees materialized
    assert len(seg.calls) == 1


def test_anonymous_watches_authorized_flight(tmp_path, monkeypatch):
    """User decision 2026-08-24: an anonymous request for a resource an
    authorized job is materializing gets 202 (check back) - and may long-poll
    to the file - but can never initiate."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    gate = threading.Event()
    seg = FakeSegmenter(gate=gate, steps=2)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"; d.mkdir(exist_ok=True); (d / "s.dcm").write_bytes(b"d")
        return d

    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex, token="s3cret"))
    auth = {"Authorization": "Bearer s3cret"}
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"

    assert client.get(url).status_code == 404              # nothing in flight: 404
    r = client.get(url, headers={**auth, "Prefer": "wait=0"})
    assert r.status_code == 202                            # authorized initiates
    anon = client.get(url, headers={"Prefer": "wait=0"})
    assert anon.status_code == 202                         # anonymous watches
    assert anon.json()["state"] == "materializing"
    assert "job" not in anon.json()                        # no job vocabulary leaked
    assert client.head(url).status_code == 202             # probe agrees
    t0 = time.time()                           # wait=0 does not wait for the
    while not seg.calls and time.time() - t0 < 5:          # dispatcher to ENTER
        time.sleep(0.01)                       # segment(); a slow runner needs a beat
    assert len(seg.calls) == 1                             # one compute entered; watching added none
    gate.set()
    blocked = client.get(url, headers={"Prefer": "wait=10"})
    assert blocked.status_code == 200                      # anonymous long-poll to bytes
    assert len(seg.calls) == 1                             # one compute total


def test_public_app_shows_inflight_and_waits(tmp_path):
    from nnseg.serve import ResultCache, create_public_app, result_key
    cache = ResultCache(tmp_path / "c")
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    key_fn = lambda identity, task, opts=None: result_key((identity,), task, opts or {}, ["w=1"])
    flights = {}
    app = create_public_app(key_fn, cache.get, lambda: ["total_fast"],
                            inflight=lambda k: flights.get(k))
    client = TestClient(app)
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"
    assert client.get(url).status_code == 404
    flights[key_fn(f"idc:{u}", "total_fast")] = {"progress": {"stage": "predict", "fraction": 0.4}}
    r = client.get(url, headers={"Prefer": "wait=0"})
    assert r.status_code == 202
    assert r.json()["progress"]["stage"] == "predict"
    assert client.head(url).status_code == 202
    src = tmp_path / "l.seg.nrrd"; src.write_bytes(b"\x1f\x8bx")
    cache.put(key_fn(f"idc:{u}", "total_fast"), src, {}, {})
    assert client.get(url).status_code == 200              # materialized mid-watch


def test_202s_carry_progress_headers(tmp_path):
    """Progress rides 202 HEADERS too, so HEAD probes (which cannot have a
    body) and header-only clients see stage and fraction."""
    from nnseg.serve import ResultCache, create_public_app, result_key
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    key_fn = lambda identity, task, opts=None: result_key((identity,), task, opts or {}, ["w=1"])
    flights = {key_fn(f"idc:{u}", "total_fast"):
               {"progress": {"stage": "predict", "fraction": 0.42}}}
    cache = ResultCache(tmp_path / "c")
    app = create_public_app(key_fn, cache.get, lambda: ["total_fast"],
                            inflight=lambda k: flights.get(k))
    client = TestClient(app)
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"
    h = client.head(url)
    assert h.status_code == 202
    assert h.headers["nnseg-stage"] == "predict"
    assert h.headers["nnseg-fraction"] == "0.420"
    g = client.get(url, headers={"Prefer": "wait=0"})
    assert g.status_code == 202 and g.headers["nnseg-fraction"] == "0.420"


# -- prefetch: the CPU downloader runs parallel to the GPU job ---------------

def _idc_submit(client, uuid):
    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "idc",
                                                             "crdc_series_uuid": uuid}])})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def test_prefetch_overlaps_running_job(tmp_path, monkeypatch):
    """While job A holds the GPU, job B's series is downloaded; B's dispatch
    then uses the prefetched directory instead of fetching again."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    fetched = []

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "slice.dcm").write_bytes(b"dcm")
        fetched.append(series)
        return d

    gate = threading.Event()
    seg = FakeSegmenter(gate=gate)
    ex = LocalExecutor(seg, workdir=tmp_path, fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    ua = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    ub = "77aabbcc-9410-47ff-9c9f-a44b26a4bd55"
    a = _idc_submit(client, ua)
    wait_state(client, a, ("running",))
    b = _idc_submit(client, ub)

    t0 = time.time()                       # A is gated: any B fetch now overlaps
    while ub not in fetched:
        assert time.time() - t0 < 5, f"B was not prefetched; log {fetched}"
        time.sleep(0.01)
    assert client.get(f"/v1/jobs/{a}").json()["state"] == "running"

    gate.set()
    wait_state(client, a, ("done",))
    sb = wait_state(client, b, ("done",))
    assert fetched == [ua, ub]             # exactly once each - no refetch at dispatch
    assert sb["input_identity"] == [f"idc:{ub}"]
    assert seg.calls[1][0].endswith("series")


def test_prefetch_failure_falls_back_to_inline_fetch(tmp_path, monkeypatch):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    calls = []

    def fake_fetch(series, jobdir):
        calls.append(series)
        if len(calls) == 2:                # the prefetch attempt for B
            raise RuntimeError("bucket hiccup")
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "slice.dcm").write_bytes(b"dcm")
        return d

    gate = threading.Event()
    seg = FakeSegmenter(gate=gate)
    ex = LocalExecutor(seg, workdir=tmp_path, fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    a = _idc_submit(client, "0be27d1c-9410-47ff-9c9f-a44b26a4bd55")
    wait_state(client, a, ("running",))
    b = _idc_submit(client, "77aabbcc-9410-47ff-9c9f-a44b26a4bd55")
    t0 = time.time()
    while len(calls) < 2:                  # wait out the failed prefetch
        assert time.time() - t0 < 5
        time.sleep(0.01)
    gate.set()
    wait_state(client, a, ("done",))
    sb = wait_state(client, b, ("done",))
    assert sb["state"] == "done"
    assert len(calls) == 3                 # A, failed prefetch, B's inline retry


# -- SeriesCache: the evicting LRU input cache -------------------------------

def _cache(tmp_path, fetched, payload=b"x" * 100, delay=0.0, fail_on=()):
    from nnseg.serve import SeriesCache

    def fetch(series, entry):
        if delay:
            time.sleep(delay)
        if series in fail_on:
            raise RuntimeError("bucket hiccup")
        d = entry / "series"
        d.mkdir()
        (d / "slice.dcm").write_bytes(payload)
        fetched.append(series)
        return d

    return SeriesCache(tmp_path / "sc", fetch, budget_bytes=250)


def test_series_cache_reuses_across_calls(tmp_path):
    fetched = []
    sc = _cache(tmp_path, fetched)
    p1 = sc.get_or_fetch("u1")
    p2 = sc.get_or_fetch("u1")
    assert p1 == p2 and fetched == ["u1"]
    assert sc.has("u1")


def test_series_cache_reader_waits_out_writer(tmp_path):
    fetched = []
    sc = _cache(tmp_path, fetched, delay=0.3)
    t = threading.Thread(target=lambda: sc.prefetch("u1"))
    t.start()
    time.sleep(0.05)                       # writer holds the claim now
    assert sc.staging("u1")
    p = sc.get_or_fetch("u1")              # blocks on the marker, no second fetch
    t.join()
    assert p.name == "series" and fetched == ["u1"]


def test_series_cache_failed_writer_leaves_nothing(tmp_path):
    fetched = []
    sc = _cache(tmp_path, fetched, fail_on={"u1"})
    assert sc.prefetch("u1") is False
    assert not sc.has("u1") and not sc.staging("u1")


def test_series_cache_lru_eviction_on_budget(tmp_path):
    fetched = []
    sc = _cache(tmp_path, fetched)         # budget 250, entries are 100 bytes + marker
    sc.get_or_fetch("u1")
    time.sleep(0.02)
    sc.get_or_fetch("u2")
    time.sleep(0.02)
    sc.get_or_fetch("u1")                  # touch: u1 is now newer than u2
    time.sleep(0.02)
    sc.get_or_fetch("u3")                  # 3 x 100 > 250: evict LRU = u2
    assert sc.has("u1") and sc.has("u3") and not sc.has("u2")
    sc.get_or_fetch("u2")                  # refetch after eviction
    assert fetched == ["u1", "u2", "u3", "u2"]


def test_same_series_two_tasks_fetches_once(tmp_path, monkeypatch):
    """The user's multi-task-per-image case: total_fast then total on one
    series downloads it exactly once."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    fetched = []

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "slice.dcm").write_bytes(b"dcm")
        fetched.append(series)
        return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    uu = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    src = json.dumps([{"kind": "idc", "crdc_series_uuid": uu}])
    ja = client.post("/v1/jobs", data={"task": "total_fast", "source": src}).json()["id"]
    jb = client.post("/v1/jobs", data={"task": "total", "source": src}).json()["id"]
    wait_state(client, ja, ("done",))
    wait_state(client, jb, ("done",))
    assert fetched == [uu]                 # one download for both tasks
    assert seg.calls[0][1] == "total_fast" and seg.calls[1][1] == "total"


def test_prefetch_prereads_next_input(tmp_path, monkeypatch):
    """The read half of the IO-prefetch pipeline: while job A holds the GPU,
    job B's series is fetched AND read; B's segment receives the image object."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    fetched, read = [], []

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "slice.dcm").write_bytes(b"dcm")
        fetched.append(series)
        return d

    class FakeImage:
        def __init__(self, path):
            self.path = str(path)

        def __str__(self):
            return f"PREREAD:{self.path}"

    def fake_read(path):
        read.append(str(path))
        return FakeImage(path)

    gate = threading.Event()
    seg = FakeSegmenter(gate=gate)
    ex = LocalExecutor(seg, workdir=tmp_path, fetch_idc_fn=fake_fetch, read_fn=fake_read)
    client = TestClient(create_app(ex))
    ua = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    ub = "77aabbcc-9410-47ff-9c9f-a44b26a4bd55"
    a = _idc_submit(client, ua)
    wait_state(client, a, ("running",))
    b = _idc_submit(client, ub)

    t0 = time.time()                       # A is gated: read-ahead must land now
    while not read:
        assert time.time() - t0 < 5, "B was not pre-read"
        time.sleep(0.01)
    assert client.get(f"/v1/jobs/{a}").json()["state"] == "running"
    assert ub in read[0]

    gate.set()
    wait_state(client, a, ("done",))
    wait_state(client, b, ("done",))
    assert fetched == [ua, ub]
    assert seg.calls[1][0].startswith("PREREAD:")     # the object, not the path
    assert seg.calls[0][0].endswith("series")         # A itself read inline


def test_prefetch_prereads_next_upload(tmp_path, monkeypatch):
    """Uploads generalize the same mechanism: bytes are already local, so the
    pre-reader hides the read (the dominant warm cost for .nii.gz inputs)."""
    read = []

    class FakeImage:
        def __init__(self, path):
            self.path = str(path)

        def __str__(self):
            return f"PREREAD:{self.path}"

    def fake_read(path):
        read.append(str(path))
        return FakeImage(path)

    gate = threading.Event()
    seg = FakeSegmenter(gate=gate)
    ex = LocalExecutor(seg, workdir=tmp_path, read_fn=fake_read)
    client = TestClient(create_app(ex))
    a = client.post("/v1/jobs", files={"file": ("a.nii.gz", b"\x1f\x8baaaa")},
                    data={"task": "total_fast"}).json()["id"]
    wait_state(client, a, ("running",))
    b = client.post("/v1/jobs", files={"file": ("b.nii.gz", b"\x1f\x8bbbbb")},
                    data={"task": "total_fast"}).json()["id"]

    t0 = time.time()                       # A is gated: B's read must land now
    while not read:
        assert time.time() - t0 < 5, "B was not pre-read"
        time.sleep(0.01)
    assert client.get(f"/v1/jobs/{a}").json()["state"] == "running"
    assert read[0].endswith("b.nii.gz")

    gate.set()
    wait_state(client, a, ("done",))
    wait_state(client, b, ("done",))
    assert seg.calls[1][0].startswith("PREREAD:")     # B got the object
    assert seg.calls[0][0].endswith("a.nii.gz")       # A read inline


# -- pluggable data sources ---------------------------------------------------

class ToySource:
    """A programmatic source: numbered specimens from a fake repository."""
    prefix = "toy"
    id_pattern = r"sp[0-9]{3}"
    description = "test specimens"

    def __init__(self):
        self.fetched = []

    def enabled(self):
        return True

    def identity(self, ident):
        return f"toy:{ident}"

    def describe(self):
        return {"prefix": self.prefix, "id_pattern": self.id_pattern,
                "enabled": True, "description": self.description}

    def fetch(self, ident, dest_dir):
        d = dest_dir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "img.nii.gz").write_bytes(b"\x1f\x8b" + ident.encode())
        self.fetched.append(ident)
        return d


def test_custom_source_end_to_end(tmp_path):
    """A new repository = one class: submit by kind, path surface, caching,
    prefetch keys - all come from the registry."""
    from nnseg.sources import IDCSource
    toy = ToySource()
    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       sources=[IDCSource(), toy])
    client = TestClient(create_app(ex))

    srcs = client.get("/v1/sources").json()["sources"]
    assert [x["prefix"] for x in srcs] == ["idc", "toy"]

    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "toy", "id": "sp042"}])})
    assert r.status_code == 202, r.text
    s = wait_state(client, r.json()["id"], ("done",))
    assert s["input_identity"] == ["toy:sp042"]
    assert toy.fetched == ["sp042"]

    # the path surface exists for the new prefix, serving the cached result
    r2 = client.get("/v1/toy/sp042/total_fast/labels.seg.nrrd")
    assert r2.status_code == 200
    assert client.get("/v1/toy/zzz/total_fast/labels.seg.nrrd").status_code == 422
    assert client.head("/v1/toy/sp999/total_fast/labels.seg.nrrd").status_code == 404

    # completed-segmentations listing shows it with its path
    segs = client.get("/v1/segmentations").json()["segmentations"]
    assert any(e["identity"] == ["toy:sp042"]
               and e["path"] == "/v1/toy/sp042/total_fast/labels.seg.nrrd"
               for e in segs)


def test_segmentations_listing_shape(tmp_path, monkeypatch):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(b"d")
        return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    jid = _idc_submit(client, u)
    wait_state(client, jid, ("done",))
    t0 = time.time()                       # the cache put lands just after "done"
    while True:
        segs = client.get("/v1/segmentations").json()["segmentations"]
        if segs or time.time() - t0 > 2:
            break
        time.sleep(0.01)
    assert len(segs) == 1
    e = segs[0]
    assert e["task"] == "total_fast" and e["identity"] == [f"idc:{u}"]
    assert e["bytes"] > 0 and e["computed"]
    assert e["path"] == f"/v1/idc/{u}/total_fast/labels.seg.nrrd"


def test_slashed_identifiers_hash_into_the_cache(tmp_path):
    """Identifiers that cannot be directory names (slashes, dot names) map to
    deterministic hashed entries inside the root - safe by construction, and
    sources with DOI-style ids need no special casing."""
    from nnseg.serve import SeriesCache
    from nnseg.sources import registry

    fetched = []

    def fetch(key, entry):
        d = entry / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "f").write_bytes(b"x")
        fetched.append(key)
        return d

    root = tmp_path / "sc"
    sc = SeriesCache(root, fetch)
    for weird in ("doi:10.1234/abc.def", "a/../../etc", "..", "."):
        got = sc.get_or_fetch(weird)
        assert got.resolve().is_relative_to(root.resolve())   # never escapes
        assert sc.has(weird)
        assert sc._entry(weird).name.startswith("h_")
        assert (sc._entry(weird) / ".key").read_text() == weird
    assert sc.get_or_fetch("doi:10.1234/abc.def") and len(fetched) == 4  # cached

    class Doi:
        prefix = "doi"
        id_pattern = r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+"
        description = ""

        def enabled(self):
            return True

    assert "doi" in registry([Doi()])          # slashed patterns are fine now


def test_tcia_source_flattens_and_blocks_zip_slip(tmp_path, monkeypatch):
    import io
    import zipfile

    from nnseg.sources import TCIASource

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("CPTAC/1-1.dcm", b"DICM1")
        z.writestr("../../evil.txt", b"nope")
        z.writestr("LICENSE", b"CC BY")
        z.writestr("sub/dir/", b"")

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda url, timeout=0: FakeResp(buf.getvalue()))
    src = TCIASource()
    got = src.fetch("1.2.3.4.5.6.7", tmp_path)
    names = sorted(p.name for p in got.iterdir())
    assert names == ["1-1.dcm", "LICENSE", "evil.txt"]   # flattened, inside dest
    assert not (tmp_path.parent / "evil.txt").exists()
    assert import_re_fullmatch(src.id_pattern, "1.3.6.1.4.1.14519.5.2.1.7085.1")
    assert not import_re_fullmatch(src.id_pattern, "1." + "2" * 70)  # >64 chars


def import_re_fullmatch(pat, s):
    import re as _re
    return _re.fullmatch(pat, s) is not None


def test_openneuro_is_a_data_only_source(tmp_path, monkeypatch):
    import io

    from nnseg.sources import openneuro_source, registry

    src = openneuro_source()
    assert "openneuro" in registry([src])
    seen = {}

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(url, timeout=0):
        seen["url"] = url
        return FakeResp(b"\x1f\x8bmri")

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    got = src.fetch("ds000001/sub-01/anat/sub-01_T1w.nii.gz", tmp_path)
    assert seen["url"].endswith("openneuro.org/ds000001/sub-01/anat/sub-01_T1w.nii.gz")
    assert (got / "sub-01_T1w.nii.gz").exists()          # basename keeps the format


def test_catalog_remaps_match_installed_weights():
    """For every TS task part whose weights are installed locally, the remap's
    local labels must exist in the checkpoint's dataset.json AND map to a
    global whose name matches the checkpoint's name for that local. This is
    the drift guard that would have caught the stale total_mr class map
    (organs silently dropping 14 structures and mislabeling 6)."""
    import json
    from pathlib import Path

    import pytest

    data = json.loads((Path("src/nnseg/data/ts_tasks.json")).read_text())
    tasks = data["tasks"] if isinstance(data, dict) else data
    roots = [Path.home() / ".totalsegmentator/nnunet/results"]
    checked = 0
    for t in tasks:
        gmap = {int(k): v for k, v in (t.get("label_map") or {}).items()}
        parts = t.get("union") or []
        for part in parts:
            wid = part.get("weights_id")
            remap = part.get("label_remap") or {}
            djs = [dj for r in roots if r.exists()
                   for dj in r.glob(f"Dataset{wid}_*/**/dataset.json")]
            if not djs or not remap:
                continue
            labels = json.loads(djs[0].read_text())["labels"]
            local_names = {int(v): k for k, v in labels.items() if isinstance(v, int)}
            for local, global_ in remap.items():
                assert int(local) in local_names, (
                    f"{t['name']}/{part.get('name')}: remap local {local} not in "
                    f"Dataset{wid} checkpoint labels")
                assert gmap.get(int(global_)) == local_names[int(local)], (
                    f"{t['name']}/{part.get('name')}: local {local} "
                    f"({local_names[int(local)]}) mapped to global {global_} "
                    f"({gmap.get(int(global_))})")
            checked += 1
    if checked == 0:
        pytest.skip("no TS weights installed locally")


def test_prepare_endpoint_installs_via_job(tmp_path):
    """POST /v1/tasks/{task}/prepare queues a weights install through the same
    queue; the job's result is the task's description."""
    prepared = []

    seg = FakeSegmenter()
    seg.prepare = lambda t: (prepared.append(t) or {"name": t, "materialized": True})
    ex = LocalExecutor(seg, workdir=tmp_path)
    client = TestClient(create_app(ex))
    r = client.post("/v1/tasks/total_fast/prepare")
    assert r.status_code == 202 and r.json()["kind"] == "prepare"
    s = wait_state(client, r.json()["id"], ("done",))
    assert prepared == ["total_fast"]
    assert s["result"]["materialized"] is True
    assert client.post("/v1/tasks/nope/prepare").status_code == 404


def test_prepare_requires_auth_when_token_set(tmp_path):
    seg = FakeSegmenter()
    seg.prepare = lambda t: {"name": t}
    ex = LocalExecutor(seg, workdir=tmp_path)
    client = TestClient(create_app(ex, token="s3cret"))
    assert client.post("/v1/tasks/total_fast/prepare").status_code == 401
    r = client.post("/v1/tasks/total_fast/prepare",
                    headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 202


def test_all_task_name_forms_converge_to_one_cache_key(tmp_path, monkeypatch):
    """short, eco:name, and eco:name@version address the same resource: the
    canonical name drives the result key, so a result computed under one form
    is a cache hit under every other."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(b"d")
        return d

    seg = FakeSegmenter()
    seg.resolve_task = lambda t: {"total_fast": "ts:total_fast",
                                  "ts:total_fast": "ts:total_fast",
                                  "ts:total_fast@v1": "ts:total_fast"}.get(t) or (
        (_ for _ in ()).throw(LookupError(t)))
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"

    r = client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd",
                   headers={"Prefer": "wait=30"})
    assert r.status_code == 200                      # computed under the short form
    assert len(seg.calls) == 1
    for form in ("ts:total_fast", "ts:total_fast@v1", "total_fast"):
        r2 = client.get(f"/v1/idc/{u}/{form}/labels.seg.nrrd")
        assert r2.status_code == 200, form           # cache hit, no recompute
    assert len(seg.calls) == 1
    assert client.get(f"/v1/idc/{u}/bogus/labels.seg.nrrd").status_code == 404


def test_preview_renders_and_serves(tmp_path, monkeypatch):
    """A real (synthetic) volume through a fake segmenter that writes a real
    seg.nrrd: the preview lands in the cache entry, serves at preview.png on
    the path surface, and the listing links it."""
    pytest.importorskip("matplotlib")
    import numpy as np
    import SimpleITK as sitk

    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        a = np.full((24, 32, 32), -1000, np.int16)
        a[8:16, 10:22, 10:22] = 40
        img = sitk.GetImageFromArray(a)
        sitk.WriteImage(img, str(d / "vol.nii.gz"))
        return d

    class SavingSeg(FakeSegmenter):
        def segment(self, image, task, *, progress=None, cancel=None, **options):
            self.calls.append((str(image), task, options))

            class R:
                def save(_, path):
                    a = np.zeros((24, 32, 32), np.uint8)
                    a[9:15, 12:20, 12:20] = 1
                    img = sitk.GetImageFromArray(a)
                    img.SetMetaData("Segment0_Name", "blob")
                    img.SetMetaData("Segment0_LabelValue", "1")
                    img.SetMetaData("Segment0_Color", "0.9 0.3 0.2")
                    sitk.WriteImage(img, str(path))
                    return path
                schema = type("S", (), {"names": {1: "blob"}})()
                def volumes_ml(_):
                    return {"blob": 1.0}
                provenance = {"task": "total_fast"}
            return R()

    seg = SavingSeg()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    r = client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd",
                   headers={"Prefer": "wait=30"})
    assert r.status_code == 200
    p = wait_artifact(client, f"/v1/idc/{u}/total_fast/preview.png")
    assert p.status_code == 200
    assert p.headers["content-type"] == "image/png"
    assert p.content[:8] == b"\x89PNG\r\n\x1a\n" and len(p.content) > 5000
    wait_artifact(client, f"/v1/idc/{u}/total_fast/preview.png")
    segs = client.get("/v1/segmentations").json()["segmentations"]
    e = next(x for x in segs if x["identity"] == [f"idc:{u}"])
    assert e["preview"] == f"/v1/idc/{u}/total_fast/preview.png"
    # a result without a preview 404s cleanly rather than erroring
    assert client.get(f"/v1/idc/{u}/total/preview.png").status_code == 404
    # grid-variant labels (different shape from the input) still render: the
    # preview resamples the grayscale onto the labels grid
    from nnseg.preview import render_preview
    import SimpleITK as _sitk
    import numpy as _np
    fine = _sitk.GetImageFromArray(_np.zeros((48, 64, 64), _np.uint8))
    fine.SetSpacing((0.5, 0.5, 0.5))
    fine.SetMetaData("Segment0_Name", "blob")
    fine.SetMetaData("Segment0_LabelValue", "1")
    fine.SetMetaData("Segment0_Color", "0.9 0.3 0.2")
    arr = _sitk.GetArrayFromImage(fine)
    arr[18:30, 24:40, 24:40] = 1
    fine2 = _sitk.GetImageFromArray(arr)
    fine2.CopyInformation(fine)
    for k in fine.GetMetaDataKeys():
        fine2.SetMetaData(k, fine.GetMetaData(k))
    vp = tmp_path / "variant.seg.nrrd"
    _sitk.WriteImage(fine2, str(vp))
    coarse_input = _sitk.GetImageFromArray(
        _np.full((24, 32, 32), -1000, _np.int16))
    coarse_input.SetSpacing((1.0, 1.0, 1.0))
    out = render_preview(coarse_input, vp, tmp_path / "variant_preview.png")
    assert out is not None and out.stat().st_size > 3000


def test_grid_variant_1mm_is_a_distinct_addressable_resource(tmp_path, monkeypatch):
    """labels_res-1mm.seg.nrrd keys and computes under {"grid": 1.0}: distinct
    cache entry from the default, jobs-API submits converge onto the same key
    (int 1 normalizes to 1.0), the listing derives the variant path, and
    DELETE evicts only the variant."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(b"d")
        return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    base = f"/v1/idc/{u}/total_fast"

    r0 = client.get(f"{base}/labels.seg.nrrd", headers={"Prefer": "wait=30"})
    assert r0.status_code == 200
    assert seg.calls[-1][2] == {}                        # default: no options
    r1 = client.get(f"{base}/labels_res-1mm.seg.nrrd", headers={"Prefer": "wait=30"})
    assert r1.status_code == 200
    assert seg.calls[-1][2] == {"grid": 1.0}             # the variant's options
    assert len(seg.calls) == 2                           # two distinct computes
    assert r0.headers["etag"] != r1.headers["etag"]      # two distinct resources

    # jobs-API convergence: {"grid": 1} (int) lands on the same key -> no recompute
    j = client.post("/v1/jobs", data={"task": "total_fast",
                                      "options": json.dumps({"grid": 1}),
                                      "source": json.dumps([{"kind": "idc",
                                                             "crdc_series_uuid": u}])})
    assert j.status_code == 202
    s = wait_state(client, j.json()["id"], ("done",))
    assert s["cached"] is True and len(seg.calls) == 2   # cache hit, not a third run

    segs = client.get("/v1/segmentations").json()["segmentations"]
    paths = {e.get("path") for e in segs}
    assert f"{base}/labels.seg.nrrd" in paths
    assert f"{base}/labels_res-1mm.seg.nrrd" in paths

    assert client.head(f"{base}/labels_res-1mm.seg.nrrd").status_code == 200
    d = client.delete(f"{base}/labels_res-1mm.seg.nrrd")
    assert d.status_code == 200 and d.json()["deleted"]
    assert client.head(f"{base}/labels_res-1mm.seg.nrrd").status_code == 404
    assert client.head(f"{base}/labels.seg.nrrd").status_code == 200   # default untouched


def test_statistics_computed_and_served_json_and_tsv(tmp_path, monkeypatch):
    """First-order statistics land beside the labels at completion and serve
    as canonical JSON plus a derived TSV whose column names carry units."""
    import numpy as np
    import SimpleITK as sitk

    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        a = np.full((20, 20, 20), -1000, np.int16)
        a[5:15, 5:15, 5:15] = 100                       # the blob is 100 HU
        img = sitk.GetImageFromArray(a)
        img.SetSpacing((2.0, 2.0, 2.0))
        sitk.WriteImage(img, str(d / "vol.nii.gz"))
        return d

    class SavingSeg(FakeSegmenter):
        def segment(self, image, task, *, progress=None, cancel=None, **options):
            self.calls.append((str(image), task, options))

            class R:
                def save(_, path):
                    a = np.zeros((20, 20, 20), np.uint8)
                    a[5:15, 5:15, 5:15] = 1
                    img = sitk.GetImageFromArray(a)
                    img.SetSpacing((2.0, 2.0, 2.0))
                    img.SetMetaData("Segment0_Name", "blob")
                    img.SetMetaData("Segment0_LabelValue", "1")
                    img.SetMetaData("Segment0_Color", "0.2 0.6 0.9")
                    sitk.WriteImage(img, str(path))
                    return path
                schema = type("S", (), {"names": {1: "blob"}})()
                def volumes_ml(_):
                    return {"blob": 8.0}
                provenance = {}
            return R()

    seg = SavingSeg()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    base = f"/v1/idc/{u}/total_fast"
    assert client.get(f"{base}/labels.seg.nrrd",
                      headers={"Prefer": "wait=30"}).status_code == 200

    j = wait_artifact(client, f"{base}/statistics.json")
    assert j.status_code == 200
    stats = j.json()
    assert stats["units"]["intensity"] == "hu"
    s0 = stats["structures"][0]
    assert s0["structure"] == "blob"
    assert s0["voxels"] == 1000                          # 10^3 voxels
    assert abs(s0["volume_ml"] - 8.0) < 1e-6             # 1000 x 8 mm^3
    assert s0["mean"] == 100.0 and s0["std"] == 0.0      # uniform blob

    t = client.get(f"{base}/statistics.tsv")
    assert t.status_code == 200
    header = t.text.splitlines()[0].split("\t")
    assert "volume_ml" in header and "mean_hu" in header  # units in columns
    assert "centroid_r_mm" in header
    row = t.text.splitlines()[1].split("\t")
    assert row[0] == "blob" and row[3] == "8.0"

    wait_artifact(client, f"{base}/statistics.json")
    segs = client.get("/v1/segmentations").json()["segmentations"]
    e = next(x for x in segs if x["identity"] == [f"idc:{u}"])
    assert e["statistics"] == f"{base}/statistics.tsv"
    assert client.get(f"{base}/statistics_res-1mm.json").status_code == 404


def test_artifact_pending_vs_definitive_absence(tmp_path, monkeypatch):
    """While the overlap thread runs, artifact GETs say 202 retry-later (and
    Prefer: wait blocks until placed); once it finished, a missing artifact
    is a definitive 404 - a poller can never wait forever on a dead end."""
    import numpy as np
    import SimpleITK as sitk

    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        a = np.full((16, 16, 16), -1000, np.int16)
        a[4:12, 4:12, 4:12] = 50
        sitk.WriteImage(sitk.GetImageFromArray(a), str(d / "vol.nii.gz"))
        return d

    class SavingSeg(FakeSegmenter):
        def segment(self, image, task, *, progress=None, cancel=None, **options):
            self.calls.append((str(image), task, options))

            class R:
                def save(_, path):
                    a = np.zeros((16, 16, 16), np.uint8)
                    a[5:10, 5:10, 5:10] = 1
                    img = sitk.GetImageFromArray(a)
                    img.SetMetaData("Segment0_Name", "blob")
                    img.SetMetaData("Segment0_LabelValue", "1")
                    img.SetMetaData("Segment0_Color", "0.5 0.5 0.5")
                    sitk.WriteImage(img, str(path))
                    return path
                schema = type("S", (), {"names": {1: "blob"}})()
                def volumes_ml(_):
                    return {"blob": 1.0}
                provenance = {}
            return R()

    seg = SavingSeg()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    base = f"/v1/idc/{u}/total_fast"
    assert client.get(f"{base}/labels.seg.nrrd",
                      headers={"Prefer": "wait=30"}).status_code == 200

    # Prefer: wait on the artifact blocks until the overlap thread places it
    r = client.get(f"{base}/statistics.json", headers={"Prefer": "wait=10"})
    assert r.status_code == 200
    assert r.json()["structures"][0]["structure"] == "blob"

    # once the thread finished, a variant that never computed is definitive
    assert client.get(f"{base}/statistics_res-1mm.json").status_code == 404

    # artifacts disabled: absence is definitive immediately, never a 202
    ex2 = LocalExecutor(seg, workdir=tmp_path / "w2", cache_dir=tmp_path / "rc2",
                        fetch_idc_fn=fake_fetch, artifacts=())
    client2 = TestClient(create_app(ex2))
    assert client2.get(f"{base}/labels.seg.nrrd",
                       headers={"Prefer": "wait=30"}).status_code == 200
    assert client2.get(f"{base}/statistics.json").status_code == 404
    assert client2.get(f"{base}/preview.png").status_code == 404


def test_statistics_get_can_initiate_the_whole_chain(tmp_path, monkeypatch):
    """The refined rule: an authorized GET of statistics.tsv WITH Prefer: wait
    materializes segment -> statistics in one request. Without the header it
    stays read-only (bulk <img>-style GETs can never compute); anonymous
    stays read-only regardless."""
    import numpy as np
    import SimpleITK as sitk

    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        a = np.full((16, 16, 16), -1000, np.int16)
        a[4:12, 4:12, 4:12] = 70
        sitk.WriteImage(sitk.GetImageFromArray(a), str(d / "vol.nii.gz"))
        return d

    class SavingSeg(FakeSegmenter):
        def segment(self, image, task, *, progress=None, cancel=None, **options):
            self.calls.append((str(image), task, options))

            class R:
                def save(_, path):
                    a = np.zeros((16, 16, 16), np.uint8)
                    a[5:10, 5:10, 5:10] = 1
                    img = sitk.GetImageFromArray(a)
                    img.SetMetaData("Segment0_Name", "blob")
                    img.SetMetaData("Segment0_LabelValue", "1")
                    img.SetMetaData("Segment0_Color", "0.5 0.5 0.5")
                    sitk.WriteImage(img, str(path))
                    return path
                schema = type("S", (), {"names": {1: "blob"}})()
                def volumes_ml(_):
                    return {"blob": 1.0}
                provenance = {}
            return R()

    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    base = f"/v1/idc/{u}/total_fast"

    # authorized + Prefer: one GET runs the whole chain and returns the TSV
    seg = SavingSeg()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    r = client.get(f"{base}/statistics.tsv", headers={"Prefer": "wait=15"})
    assert r.status_code == 200, r.text
    assert r.text.splitlines()[1].startswith("blob\t")
    assert len(seg.calls) == 1                            # it computed

    # without Prefer: read-only, 404 with the hint, nothing computed
    seg2 = SavingSeg()
    ex2 = LocalExecutor(seg2, workdir=tmp_path / "w2", cache_dir=tmp_path / "rc2",
                        fetch_idc_fn=fake_fetch)
    client2 = TestClient(create_app(ex2))
    r2 = client2.get(f"{base}/statistics.tsv")
    assert r2.status_code == 404 and "Prefer" in r2.json()["detail"]
    assert seg2.calls == []

    # anonymous with Prefer on a token-gated server: still read-only
    seg3 = SavingSeg()
    ex3 = LocalExecutor(seg3, workdir=tmp_path / "w3", cache_dir=tmp_path / "rc3",
                        fetch_idc_fn=fake_fetch)
    client3 = TestClient(create_app(ex3, token="s3cret"))
    r3 = client3.get(f"{base}/statistics.tsv", headers={"Prefer": "wait=15"})
    assert r3.status_code == 404 and seg3.calls == []


def test_wait_zero_expresses_intent_with_impatience(tmp_path, monkeypatch):
    """Prefer: wait=0 on an artifact GET fires the job and returns 202 with
    progress headers immediately - intent without patience. Polling with
    wait=0 rides the same single flight; a later patient GET collects."""
    import numpy as np
    import SimpleITK as sitk

    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        a = np.full((16, 16, 16), -1000, np.int16)
        a[4:12, 4:12, 4:12] = 70
        sitk.WriteImage(sitk.GetImageFromArray(a), str(d / "vol.nii.gz"))
        return d

    class SavingSeg(FakeSegmenter):
        def segment(self, image, task, *, progress=None, cancel=None, **options):
            self.calls.append((str(image), task, options))
            if self.gate is not None:
                self.gate.wait(timeout=5)

            class R:
                def save(_, path):
                    a = np.zeros((16, 16, 16), np.uint8)
                    a[5:10, 5:10, 5:10] = 1
                    img = sitk.GetImageFromArray(a)
                    img.SetMetaData("Segment0_Name", "blob")
                    img.SetMetaData("Segment0_LabelValue", "1")
                    img.SetMetaData("Segment0_Color", "0.5 0.5 0.5")
                    sitk.WriteImage(img, str(path))
                    return path
                schema = type("S", (), {"names": {1: "blob"}})()
                def volumes_ml(_):
                    return {"blob": 1.0}
                provenance = {}
            return R()

    gate = threading.Event()
    seg = SavingSeg(gate=gate)
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    url = f"/v1/idc/{u}/total_fast/statistics.tsv"

    r = client.get(url, headers={"Prefer": "wait=0"})     # fire...
    assert r.status_code == 202 and "Retry-After" in r.headers
    r2 = client.get(url, headers={"Prefer": "wait=0"})    # ...poll: same flight
    assert r2.status_code == 202
    gate.set()
    r3 = wait_artifact_with_prefer(client, url)
    assert r3.status_code == 200
    assert r3.text.splitlines()[1].startswith("blob\t")
    assert len(seg.calls) == 1                            # single flight held


def wait_artifact_with_prefer(client, url, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(url, headers={"Prefer": "wait=5"})
        if r.status_code == 200:
            return r
        time.sleep(0.05)
    return r


def test_query_params_cannot_inject_key_material(tmp_path, monkeypatch):
    """Regression for the closure-default injection: _opts/_tok must not be
    query parameters. A variant URL with ?_opts= must keep variant semantics;
    junk ?_opts must not 500; the OpenAPI schema must not expose them."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(b"d")
        return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    base = f"/v1/idc/{u}/total_fast"
    assert client.get(f"{base}/labels.seg.nrrd",
                      headers={"Prefer": "wait=30"}).status_code == 200

    # only the default is cached: the variant URL must 404/202 regardless of
    # query-string games, never serve the default entry
    for q in ("?_opts=", "?_opts=zzz", "?_tok=", "?_opts=%7B%7D"):
        r = client.head(f"{base}/labels_res-1mm.seg.nrrd{q}")
        assert r.status_code in (404, 202), (q, r.status_code)
        r2 = client.head(f"{base}/labels.seg.nrrd{q}")
        assert r2.status_code == 200, (q, r2.status_code)   # and no 500s
    spec = client.get("/openapi.json").json()
    txt = json.dumps(spec)
    assert "_opts" not in txt and "_tok" not in txt


def test_public_twin_variant_urls_key_on_variant_options(tmp_path):
    """Regression for the arity-fallback bug: the twin must never serve a
    differently-keyed entry under a variant URL. With only the default entry
    cached, the res-1mm URL is a 404 - not the default bytes."""
    from nnseg.serve import ResultCache, create_public_app, result_key

    cache = ResultCache(tmp_path / "rc")
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    key_fn = lambda identity, task, opts=None: result_key((identity,), task,
                                                          opts or {}, ["w=1"])
    src = tmp_path / "labels.seg.nrrd"
    src.write_bytes(b"\x1f\x8bdefault")
    cache.put(key_fn(f"idc:{u}", "total_fast"), src, {"names": {}}, {})
    app = create_public_app(key_fn, cache.get, lambda: ["total_fast"])
    client = TestClient(app)
    assert client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd").status_code == 200
    assert client.get(f"/v1/idc/{u}/total_fast/labels_res-1mm.seg.nrrd").status_code == 404
    assert client.head(f"/v1/idc/{u}/total_fast/labels_res-1mm.seg.nrrd").status_code == 404
    # and the variant serves once its OWN entry exists
    v = tmp_path / "v.seg.nrrd"
    v.write_bytes(b"\x1f\x8bvariant")
    cache.put(key_fn(f"idc:{u}", "total_fast", {"grid": 1.0}), v, {"names": {}}, {})
    r = client.get(f"/v1/idc/{u}/total_fast/labels_res-1mm.seg.nrrd")
    assert r.status_code == 200 and r.content == b"\x1f\x8bvariant"


def test_pinned_entries_survive_eviction_and_live_writers_survive_timeouts(tmp_path):
    """Review C5: (a) a pinned in-use entry is never LRU-evicted even over
    budget; (b) a waiter at claim-timeout extends for a writer that is still
    producing files, and reclaims only a truly dead claim."""
    import threading as _t
    import time as _time

    from nnseg.serve import SeriesCache

    fetched = []

    def fetch(key, entry):
        d = entry / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "f").write_bytes(b"x" * 120)
        fetched.append(key)
        return d

    sc = SeriesCache(tmp_path / "sc", fetch, budget_bytes=150)
    sc.get_or_fetch("u1")
    sc.pin("u1")
    try:
        sc.get_or_fetch("u2")                 # over budget; u1 pinned -> u2's
        assert sc.has("u1")                   # commit evicts nothing usable...
        sc.get_or_fetch("u3")                 # ...now u2 (unpinned) must go
        assert sc.has("u1") and not sc.has("u2")
    finally:
        sc.unpin("u1")

    # live-writer fence: slow writer keeps writing; waiter must not destroy it
    sc2 = SeriesCache(tmp_path / "sc2", fetch, claim_timeout=0.3)

    def slow_fetch(key, entry):
        d = entry / "series"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            (d / f"f{i}").write_bytes(b"y")
            _time.sleep(0.15)                 # heartbeat via file mtimes
        return d

    sc2.fetch = slow_fetch
    t = _t.Thread(target=lambda: sc2.prefetch("s1"))
    t.start()
    _time.sleep(0.05)
    got = sc2.get_or_fetch("s1")              # waits ~0.9s > claim_timeout
    t.join()
    assert got.exists() and len(list(got.iterdir())) == 6   # writer's work intact

    # dead claim: bare directory, no writes -> reclaimed after timeout
    sc3 = SeriesCache(tmp_path / "sc3", fetch, claim_timeout=0.2)
    (sc3.root / "dead").mkdir()
    _time.sleep(0.3)
    assert sc3.get_or_fetch("dead").exists()  # reclaimed and fetched
    assert "dead" in fetched


def test_segmentations_listing_requires_auth_when_token_set(tmp_path):
    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc")
    client = TestClient(create_app(ex, token="s3cret"))
    assert client.get("/v1/segmentations").status_code == 401
    r = client.get("/v1/segmentations",
                   headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_first_install_rekeys_at_completion(tmp_path, monkeypatch):
    """Review C7: a job keyed while weights versions were 'unknown' re-keys
    at completion, so the entry is findable by every later probe instead of
    orphaned under the unknown-key."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(b"d")
        return d

    class InstallingSeg(FakeSegmenter):
        installed = False

        def describe(self, task):
            d = super().describe(task)
            if self.installed:
                d["weights_installed"] = [{"id": "297", "version": "v2"}]
            return d

        def segment(self, image, task, *, progress=None, cancel=None, **options):
            out = super().segment(image, task, progress=progress,
                                  cancel=cancel, **options)
            type(self).installed = True        # weights land during the run
            return out

    InstallingSeg.installed = False
    seg = InstallingSeg()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"
    assert client.get(url, headers={"Prefer": "wait=30"}).status_code == 200
    # post-install probes key with the REAL versions and must hit
    assert client.head(url).status_code == 200
    r = client.get(url)
    assert r.status_code == 200
    assert len(seg.calls) == 1                 # no recompute


def test_pending_marker_set_before_entry_visible(tmp_path, monkeypatch):
    """Review C8: the artifacts-pending marker must be observable no later
    than the cache entry itself - a probe that finds the entry but no artifact
    must read 'pending', never a definitive 404."""
    import nnseg.preview
    from nnseg import serve as serve_mod
    monkeypatch.setattr(nnseg.preview, "load_oriented_pair",
                        lambda *a, **kw: ("pair",))   # force the artifact path
    seen = []
    orig_put = serve_mod.ResultCache.put

    def spying_put(self, key, *a, **kw):
        seen.append(key in ex._artifacts_pending)
        return orig_put(self, key, *a, **kw)

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc")
    ex.cache.put = spying_put.__get__(ex.cache, serve_mod.ResultCache)
    client = TestClient(create_app(ex))
    jid = submit(client)
    wait_state(client, jid)
    assert seen == [True], "cache.put ran before the pending marker was set"
    # and the worker eventually clears it
    t0 = time.time()
    while ex._artifacts_pending and time.time() - t0 < 5:
        time.sleep(0.02)
    assert not ex._artifacts_pending


def test_result_cache_put_overwrite_atomic(tmp_path):
    """Review C8: an overwriting put must swap files by rename - an open
    reader keeps the old complete bytes, and no .tmp debris remains."""
    from nnseg.serve import RESULT_NAME, ResultCache
    rc = ResultCache(tmp_path / "rc", keep=5)
    src = tmp_path / "labels.seg.nrrd"
    src.write_bytes(b"OLD" * 100)
    rc.put("k1", src, {"v": 1}, {"m": 1})
    entry = rc.get("k1")[0]
    with open(entry, "rb") as held:          # a client mid-download
        src.write_bytes(b"NEW" * 200)
        rc.put("k1", src, {"v": 2}, {"m": 2})
        assert held.read() == b"OLD" * 100   # rename preserved the old inode
    labels, result = rc.get("k1")
    assert labels.read_bytes() == b"NEW" * 200
    assert result == {"v": 2}
    assert not list((tmp_path / "rc").rglob("*.tmp"))


def test_events_refetches_after_subscribe(tmp_path, monkeypatch):
    """Review C8: a job that turns terminal between the first status fetch and
    the SSE subscribe must not leave the stream idling on the stale snapshot
    (no event for the transition was ever pushed)."""
    seg, ex, client = make(tmp_path)
    jid = submit(client)
    wait_state(client, jid)

    real = LocalExecutor.status_of
    calls = {"n": 0}

    def stale_once(self, j):
        calls["n"] += 1
        s = real(self, j)
        if calls["n"] == 1 and s is not None:
            s = dict(s)
            s["state"] = "running"           # the pre-subscribe view
        return s

    monkeypatch.setattr(LocalExecutor, "status_of", stale_once)
    out = {}

    def read():
        with client.stream("GET", f"/v1/jobs/{jid}/events") as r:
            out["body"] = b"".join(r.iter_raw())

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(8)
    assert not t.is_alive(), "events stream hung on a stale pre-subscribe snapshot"
    assert b'"done"' in out["body"]


def test_result_purged_bytes_is_410(tmp_path):
    """Review C9: a done job whose labels file was purged answers 410, not a
    500 from streaming a missing file."""
    seg, ex, client = make(tmp_path)
    jid = submit(client)
    wait_state(client, jid)
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 200
    _, p = ex.result_file(jid)
    import pathlib
    pathlib.Path(p).unlink()
    r = client.get(f"/v1/jobs/{jid}/result")
    assert r.status_code == 410


def test_result_filename_is_stem_sanitized(tmp_path):
    """Review C9: canonical eco:name task names must not leak a ':' into the
    download filename."""
    seg, ex, client = make(tmp_path)
    jid = submit(client)
    wait_state(client, jid)
    cd = client.get(f"/v1/jobs/{jid}/result").headers.get("content-disposition", "")
    assert ":" not in cd and "/" not in cd.split("filename=")[-1]


def test_queuefull_cleans_job_dir(tmp_path):
    """Review C9: a submit refused with QueueFull must not leave the freshly
    created job directory behind."""
    gate = threading.Event()                   # never set: first job blocks
    seg, ex, client = make(tmp_path, gate=gate, max_pending=1)
    try:
        submit(client)                         # running (blocked on the gate)
        submit(client)                         # fills the pending bound
        before = {d.name for d in tmp_path.iterdir() if d.is_dir()}
        r = client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                        data={"task": "total_fast"})
        assert r.status_code == 429
        r2 = client.post("/v1/tasks/total_fast/prepare")
        assert r2.status_code == 429
        after = {d.name for d in tmp_path.iterdir() if d.is_dir()}
        assert after == before, f"leaked job dirs: {sorted(after - before)}"
    finally:
        gate.set()
        ex.close()


def test_preference_applied_only_with_prefer(tmp_path, monkeypatch):
    """Review C9: Preference-Applied is an echo of an applied preference
    (RFC 7240) - it must not appear when the client sent no Prefer header."""
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(b"d")
        return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "1be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"
    # F8 (user decision): a plain GET is a READ even when authorized - the
    # Prefer header is the intent to compute, exactly as on every artifact
    r = client.get(url)
    assert r.status_code == 404 and "Prefer" in r.json()["detail"]
    assert len(seg.calls) == 0
    r = client.get(url, headers={"Prefer": "wait=30"})
    assert r.status_code == 200
    assert r.headers.get("Preference-Applied") == "wait=30"
    r = client.get(url)                        # cached read, no Prefer sent
    assert r.status_code == 200
    assert "preference-applied" not in {k.lower() for k in r.headers}
    assert len(seg.calls) == 1


def test_task_names_with_path_separators_refused(tmp_path):
    """Review B9: a wire task name is a catalog name only - separator or
    dot-leading forms must die at canon_task, before any resolver that might
    pass a filesystem path through (Segmenter.resolve_task's identity
    fallback would; the in-process folder-path freedom must not cross the
    wire)."""
    seg, ex, client = make(tmp_path)
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    for bad in ("../weights", "a/b", "..", ".hidden", "a\\b", "ts:", "@v1", ""):
        r = client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                        data={"task": bad})
        assert r.status_code in (404, 422), (bad, r.status_code)
        # the path surface swallows slashes into {ident:path}, so a slashed
        # "task" simply fails identifier validation - but dotted and escaped
        # forms reach canon_task and must 404 there
        if "/" not in bad and bad:
            g = client.get(f"/v1/idc/{u}/{bad}/labels.seg.nrrd")
            assert g.status_code in (404, 422), (bad, g.status_code)
    assert len(seg.calls) == 0


def test_public_twin_full_parity(tmp_path):
    """Review R4: the twin IS create_app's route code. The artifact routes
    exist and serve (old F6: they were missing), filenames and cache headers
    match the main app (old F5: they drifted), and the mutating surface does
    not exist at all - not 401, absent."""
    from nnseg.serve import ResultCache, create_public_app, result_key
    cache = ResultCache(tmp_path / "c")
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    key_fn = lambda identity, task, opts=None: result_key(
        (identity,), task, opts or {}, ["w=1"])
    key = key_fn(f"idc:{u}", "total_fast")
    src = tmp_path / "labels.seg.nrrd"
    src.write_bytes(b"\x1f\x8bx")
    cache.put(key, src, {"names": {"1": "spleen"}}, {"identity": [f"idc:{u}"],
                                                     "task": "total_fast"})
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNGxx")
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps({"units": {"intensity": "hu"}, "structures": []}))
    assert cache.add_artifact(key, "preview.png", png)
    assert cache.add_artifact(key, "statistics.json", sj)

    app = create_public_app(key_fn, cache.get, lambda: ["total_fast"],
                            list_fn=cache.list)
    client = TestClient(app)
    base = f"/v1/idc/{u}/total_fast"

    r = client.get(f"{base}/labels.seg.nrrd")
    assert r.status_code == 200
    assert f"total_fast_{u[:8]}.seg.nrrd" in r.headers["content-disposition"]
    assert r.headers["etag"] == f'"{key[:32]}"'
    assert "max-age" in r.headers.get("cache-control", "")

    assert client.get(f"{base}/meta.json").status_code == 200
    p = client.get(f"{base}/preview.png")
    assert p.status_code == 200 and p.content.startswith(b"\x89PNG")
    t = client.get(f"{base}/statistics.tsv")
    assert t.status_code == 200 and t.text.startswith("structure\t")
    assert client.get(f"{base}/statistics.json").status_code == 200
    assert client.get("/v1/segmentations").status_code == 200   # lister opt-in

    assert client.delete(f"{base}/labels.seg.nrrd").status_code in (404, 405)
    assert client.post("/v1/tasks/total_fast/prepare").status_code in (404, 405)
    assert client.post("/v1/jobs").status_code in (404, 405)
    assert client.get("/v1/jobs").status_code in (404, 405)
    # and no Prefer header can make the twin compute
    u2 = "1be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    r = client.get(f"/v1/idc/{u2}/total_fast/statistics.tsv",
                   headers={"Prefer": "wait=30"})
    assert r.status_code == 404


def test_public_twin_without_lister_is_listing_free(tmp_path):
    from nnseg.serve import ResultCache, create_public_app, result_key
    cache = ResultCache(tmp_path / "c")
    key_fn = lambda identity, task, opts=None: result_key(
        (identity,), task, opts or {}, ["w=1"])
    app = create_public_app(key_fn, cache.get, lambda: ["total_fast"])
    assert TestClient(app).get("/v1/segmentations").status_code in (404, 405)


def test_task_allowlist_is_load_bearing_for_identity_fallback(tmp_path):
    """Adversarial round R-1: the old B9 test passed with the allowlist
    deleted, because every fake resolver already refused bad names. This one
    models the actual threat - a catalog whose resolve_task falls back to
    identity (Segmenter does exactly that for catalogs without resolve()) -
    so only the wire allowlist stands between a path and the pipeline."""
    class Passthrough(FakeSegmenter):
        def resolve_task(self, t):
            return str(t)                  # identity fallback: accepts anything

    seg = Passthrough()
    ex = LocalExecutor(seg, workdir=tmp_path)
    client = TestClient(create_app(ex))
    for bad in ("../weights", "..", ".hidden", "a\\b", "/etc/passwd"):
        r = client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                        data={"task": bad})
        assert r.status_code in (404, 422), (bad, r.status_code)
    assert len(seg.calls) == 0
    # and a legitimate name still flows through the same resolver
    ok = client.post("/v1/jobs", files={"file": ("x.nii.gz", b"d")},
                     data={"task": "total_fast"})
    assert ok.status_code == 202


def test_cancel_of_duplicate_does_not_clobber_survivor_marker(tmp_path):
    """Adversarial round F3: cancelling one of two queued duplicate jobs
    (same content, same key) must leave the survivor's single-flight marker
    in place - popping unconditionally made probes 404 while the survivor
    still ran and left DELETE unable to reach it."""
    gate = threading.Event()
    seg = FakeSegmenter(gate=gate)
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc")
    client = TestClient(create_app(ex))
    try:
        j0 = submit(client)                # occupies the dispatcher (blocked)
        j1 = submit(client)                # queued, keyed K
        j2 = submit(client)                # queued duplicate, marker now = j2
        key = next(k for k, v in ex._inflight.items() if v == j2)
        assert client.delete(f"/v1/jobs/{j1}").status_code == 200
        assert ex._inflight.get(key) == j2, "survivor's marker was clobbered"
        assert client.delete(f"/v1/jobs/{j2}").status_code == 200
        assert key not in ex._inflight     # own marker: popped
    finally:
        gate.set()
        ex.close()


def test_add_artifact_overwrite_is_atomic(tmp_path):
    """Adversarial round: add_artifact must swap by rename like put - an open
    reader keeps complete bytes, and concurrent writers cannot share a temp."""
    from nnseg.serve import ResultCache
    rc = ResultCache(tmp_path / "rc", keep=5)
    src = tmp_path / "l.seg.nrrd"
    src.write_bytes(b"\x1f\x8bx")
    rc.put("k", src, {}, {})
    a1 = tmp_path / "p1.png"
    a1.write_bytes(b"OLDPNG" * 50)
    assert rc.add_artifact("k", "preview.png", a1)
    entry = rc.get("k")[0].parent / "preview.png"
    with open(entry, "rb") as held:
        a2 = tmp_path / "p2.png"
        a2.write_bytes(b"NEWPNG" * 90)
        assert rc.add_artifact("k", "preview.png", a2)
        assert held.read() == b"OLDPNG" * 50
    assert entry.read_bytes() == b"NEWPNG" * 90
    assert not list((tmp_path / "rc").rglob("*.tmp"))


def test_cache_only_status_rechecks_cache_before_failed(tmp_path):
    """Adversarial round F6: the writer fills the cache BEFORE clearing its
    inflight marker; a reader probing in the same order can miss both and
    must recheck the cache before synthesizing a terminal 'failed'."""
    from nnseg.serve import CacheOnlyExecutor, ResultCache
    cache = ResultCache(tmp_path / "c")
    calls = {"n": 0}

    def get_flip(key):                     # miss on the first read, hit after
        calls["n"] += 1                    # (the completion window, collapsed)
        return None if calls["n"] == 1 else cache.get(key)

    src = tmp_path / "l.seg.nrrd"
    src.write_bytes(b"\x1f\x8bx")
    cache.put("K", src, {}, {})
    ex = CacheOnlyExecutor(get_flip, lambda i, t, o=None: "K",
                           lambda: ["total_fast"], inflight_fn=lambda k: None)
    assert ex.status_of("K")["state"] == "done"


def test_twin_artifact_watchability_matches_labels(tmp_path):
    """Adversarial round 4a: an anonymous caller watching an authorized
    flight sees 202 on preview/statistics exactly as on labels - the
    auth/Prefer gates guard initiation, not watching."""
    from nnseg.serve import ResultCache, create_public_app, result_key
    cache = ResultCache(tmp_path / "c")
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    key_fn = lambda identity, task, opts=None: result_key(
        (identity,), task, opts or {}, ["w=1"])
    flights = {key_fn(f"idc:{u}", "total_fast"):
               {"progress": {"stage": "predict", "fraction": 0.5}}}
    app = create_public_app(key_fn, cache.get, lambda: ["total_fast"],
                            inflight=lambda k: flights.get(k))
    client = TestClient(app)
    for what in ("labels.seg.nrrd", "preview.png", "statistics.tsv",
                 "statistics.json"):
        r = client.get(f"/v1/idc/{u}/total_fast/{what}",
                       headers={"Prefer": "wait=0"})   # watch, don't hold
        assert r.status_code == 202, (what, r.status_code)
    # a true miss (no flight) stays a definitive 404 on artifacts
    u2 = "1be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    assert client.get(f"/v1/idc/{u2}/total_fast/preview.png").status_code == 404


def test_preference_applied_on_artifact_200(tmp_path, monkeypatch):
    """Adversarial round 4b: a statistics GET that applied Prefer: wait
    echoes Preference-Applied exactly as the labels GET does."""
    import nnseg.preview
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    monkeypatch.setattr(nnseg.preview, "load_oriented_pair",
                        lambda *a, **kw: None)   # no artifacts computed

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(b"d")
        return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "2be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    r = client.get(f"/v1/idc/{u}/total_fast/labels.seg.nrrd",
                   headers={"Prefer": "wait=30"})
    assert r.status_code == 200
    # place a statistics artifact by hand, then read it WITH Prefer
    key = next(d.name for d in (tmp_path / "rc").iterdir())
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps({"units": {"intensity": "hu"}, "structures": []}))
    ex.cache.add_artifact(key, "statistics.json", sj)
    r = client.get(f"/v1/idc/{u}/total_fast/statistics.tsv",
                   headers={"Prefer": "wait=5"})
    assert r.status_code == 200
    assert r.headers.get("Preference-Applied") == "wait=5"
    r = client.get(f"/v1/idc/{u}/total_fast/statistics.tsv")
    assert "preference-applied" not in {k.lower() for k in r.headers}


def test_claim_teardown_respects_ownership(tmp_path):
    """Adversarial round F1: after a false-dead reclaim, the original
    writer's cleanup must not delete the successor's claim - and the
    heartbeat keeps a disk-silent live writer from being declared dead."""
    from nnseg.serve import SeriesCache
    sc = SeriesCache(tmp_path / "sc", fetch_fn=lambda *a: None)
    entry = sc._entry("s1")
    entry.mkdir(parents=True)
    token_a = sc._claim_owner(entry)       # writer 1's claim
    # reclaim: graveyard rename + re-claim by writer 2
    import shutil as _sh
    grave = entry.parent / (entry.name + ".stale0")
    entry.rename(grave)
    _sh.rmtree(grave)
    entry.mkdir(parents=True)
    token_b = sc._claim_owner(entry)
    (entry / "series").mkdir()
    (entry / "series" / "f").write_bytes(b"x")
    sc._teardown_claim(entry, token_a)     # writer 1's late cleanup
    assert entry.exists(), "stale writer deleted the successor's claim"
    assert (entry / "series" / "f").exists()
    sc._teardown_claim(entry, token_b)     # the owner may
    assert not entry.exists()

    # heartbeat: a fetch that writes nothing still reads as alive
    sc.claim_timeout = 0.6
    e2 = sc._entry("s2")
    e2.mkdir(parents=True)
    sc._claim_owner(e2)
    hb = sc._hb_start(e2)
    try:
        time.sleep(0.9)                    # > claim_timeout, zero writes
        assert sc._writer_alive(e2), "live writer declared dead"
    finally:
        hb.set()


def _idc_app(tmp_path, monkeypatch, seg=None):
    from nnseg import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(b"d")
        return d

    seg = seg or FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fake_fetch)
    return seg, ex, TestClient(create_app(ex))


def test_purged_record_resolves_via_cache_not_failure(tmp_path, monkeypatch):
    """Opus verification round: a job record purged mid-watch resolves
    through the cache - 200 promptly when the bytes exist, 404 'not
    materialized' when they don't. Never a synthesized 502 for a job that
    may have succeeded."""
    seg, ex, client = _idc_app(tmp_path, monkeypatch)
    u = "3be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"
    assert client.get(url, headers={"Prefer": "wait=30"}).status_code == 200
    key = next(d.name for d in (tmp_path / "rc").iterdir())

    monkeypatch.setattr(LocalExecutor, "find_inflight", lambda self, k: "zombie")
    monkeypatch.setattr(LocalExecutor, "status_of", lambda self, j: None)
    hit = ex.cache.get(key)

    def gated_get(self, k):                # miss at route entry, hit in-loop
        calls["n"] += 1
        return None if calls["n"] == 1 else (hit if k == key else None)

    calls = {"n": 0}
    monkeypatch.setattr(LocalExecutor, "cache_get", gated_get)
    t0 = time.time()
    r = client.get(url, headers={"Prefer": "wait=30"})
    assert r.status_code == 200
    assert time.time() - t0 < 10, "burned the deadline on a purged record"
    # bytes gone too: absence, not failure
    monkeypatch.setattr(LocalExecutor, "cache_get", lambda self, k: None)
    r = client.get(url, headers={"Prefer": "wait=30"})
    assert r.status_code == 404
    assert "failed" not in r.text


def test_cancelled_flight_artifact_is_absent_not_forever_202(tmp_path, monkeypatch):
    """Opus verification round: a cancelled flight must not leave artifact
    URLs answering 'materializing' 202 forever - terminal without a result
    is absence."""
    seg, ex, client = _idc_app(tmp_path, monkeypatch)
    u = "4be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    monkeypatch.setattr(LocalExecutor, "find_inflight", lambda self, k: "zombie")
    monkeypatch.setattr(LocalExecutor, "status_of",
                        lambda self, j: {"state": "cancelled"})
    r = client.get(f"/v1/idc/{u}/total_fast/statistics.tsv",
                   headers={"Prefer": "wait=0"})
    assert r.status_code == 404, r.status_code
    # failed flight, anonymous watcher: 404 with no job vocabulary or error
    app2 = create_app(ex, token="s3cret")
    client2 = TestClient(app2)
    monkeypatch.setattr(LocalExecutor, "status_of",
                        lambda self, j: {"state": "failed", "error": "boom"})
    r = client2.get(f"/v1/idc/{u}/total_fast/preview.png",
                    headers={"Prefer": "wait=0"})
    assert r.status_code == 404 and "boom" not in r.text


def test_listing_advertises_only_resolvable_links(tmp_path, monkeypatch):
    """Opus verification round: the listing filters unknown-task and
    stale-key entries - a stale link 404s WITH a recompute hint, so
    following the listing's own remedy would duplicate GPU work."""
    from nnseg.serve import result_key, weights_versions_of
    seg, ex, client = _idc_app(tmp_path, monkeypatch)
    u = "5be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    src = tmp_path / "l.seg.nrrd"
    src.write_bytes(b"\x1f\x8bx")
    good = result_key((f"idc:{u}",), "total_fast", {},
                      weights_versions_of(seg, "total_fast"))
    ex.cache.put(good, src, {}, {"identity": [f"idc:{u}"], "task": "total_fast",
                                 "options": {}})
    ex.cache.put("stalekey123", src, {}, {"identity": [f"idc:{u}"],
                                          "task": "total_fast", "options": {}})
    ex.cache.put("ghosttask456", src, {}, {"identity": [f"idc:{u}"],
                                           "task": "no_such_task", "options": {}})
    got = {e["key"] for e in client.get("/v1/segmentations").json()["segmentations"]}
    assert good in got
    assert "stalekey123" not in got        # link would 404-and-suggest-recompute
    assert "ghosttask456" not in got       # task this catalog cannot serve


def test_preference_echo_uniform_and_varied(tmp_path, monkeypatch):
    """Opus verification round: the echo is uniform (labels cache hits too),
    respond-async is echoed as itself, and the 200s declare Vary: Prefer."""
    seg, ex, client = _idc_app(tmp_path, monkeypatch)
    u = "6be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    url = f"/v1/idc/{u}/total_fast/labels.seg.nrrd"
    assert client.get(url, headers={"Prefer": "wait=30"}).status_code == 200
    r = client.get(url, headers={"Prefer": "wait=7"})   # cache hit WITH Prefer
    assert r.headers.get("Preference-Applied") == "wait=7"
    assert r.headers.get("Vary") == "Prefer"
    r = client.get(url, headers={"Prefer": "respond-async"})
    assert r.headers.get("Preference-Applied") == "respond-async"
    r = client.get(url)
    assert "preference-applied" not in {k.lower() for k in r.headers}


def test_result_cache_temp_names_unique_per_writer(tmp_path, monkeypatch):
    from nnseg.serve import ResultCache
    rc = ResultCache(tmp_path / "rc", keep=5)
    src = tmp_path / "l.seg.nrrd"
    src.write_bytes(b"\x1f\x8bx")
    seen = []
    import os as _os
    real = _os.replace
    monkeypatch.setattr("os.replace",
                        lambda a, b: (seen.append(Path(a).name), real(a, b))[1])
    from pathlib import Path
    rc.put("k", src, {}, {})
    rc.put("k", src, {}, {})
    labels_tmps = [n for n in seen if n.startswith("labels")]
    assert len(labels_tmps) == 2 and labels_tmps[0] != labels_tmps[1]


def test_live_writer_is_never_reclaimed(tmp_path):
    """Round 4 (HIGH): a waiter that exhausts its extensions against a
    writer that is STILL ALIVE gives up loudly - reclaiming a live writer
    aliased two writers onto one path and committed a MIXED series as
    complete (silently corrupt cache). One writer per series, always."""
    from nnseg.errors import ResourceError
    from nnseg.serve import SeriesCache
    hold = threading.Event()

    def slow_fetch(key, entry):
        d = entry / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "f").write_bytes(b"x")
        hold.wait(20)                      # alive (heartbeating) but slow
        return d

    sc = SeriesCache(tmp_path / "sc", fetch_fn=slow_fetch)
    sc.claim_timeout = 0.15
    t = threading.Thread(target=lambda: sc.get_or_fetch("s1"), daemon=True)
    t.start()
    time.sleep(0.1)                        # writer holds the claim
    t0 = time.time()
    with pytest.raises(ResourceError):
        sc.get_or_fetch("s1")
    assert 0.4 < time.time() - t0 < 8      # 4x timeout, then gave up
    assert sc._entry("s1").exists(), "live writer's claim was destroyed"
    hold.set()
    t.join(5)
    assert sc.has("s1")                    # the one writer committed cleanly


def test_stale_pending_marker_ages_out(tmp_path):
    """Round 4: a hung/killed overlap thread's pending marker must not be
    immortal - refuse-if-present made it unrecoverable, so artifact GETs
    202'd forever. Stale markers age out (Local mirror of the Modal sweep)."""
    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc")
    ex._artifacts_pending["K"] = ("dead-jid", time.time() - 1000)
    assert ex.artifact_state("K") == "absent"    # aged out (and cleaned)
    assert "K" not in ex._artifacts_pending
    ex._artifacts_pending["K"] = ("live-jid", time.time())
    assert ex.artifact_state("K") == "pending"
