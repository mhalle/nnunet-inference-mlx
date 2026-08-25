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
