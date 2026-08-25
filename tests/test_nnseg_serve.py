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
    key_fn = lambda identity, task: result_key((identity,), task, {}, ["w=1"])
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
    assert len(seg.calls) == 1                             # one compute entered; watching added none
    gate.set()
    blocked = client.get(url, headers={"Prefer": "wait=10"})
    assert blocked.status_code == 200                      # anonymous long-poll to bytes
    assert len(seg.calls) == 1                             # one compute total


def test_public_app_shows_inflight_and_waits(tmp_path):
    from nnseg.serve import ResultCache, create_public_app, result_key
    cache = ResultCache(tmp_path / "c")
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    key_fn = lambda identity, task: result_key((identity,), task, {}, ["w=1"])
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
    key_fn = lambda identity, task: result_key((identity,), task, {}, ["w=1"])
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
