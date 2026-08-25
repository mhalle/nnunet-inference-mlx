"""The Slicer module's protocol classes against a REAL server over real HTTP.

The module (medseg workspace, slicer/NNSegRemote/NNSegRemote.py) implements the wire
protocol with `requests` so Slicer needs no pip installs; its RemoteProtocol and
RemoteJobRunner import headless. Here they drive an actual uvicorn server running
create_app over a LocalExecutor with the fake segmenter - requests-based client,
FastAPI server, SSE stream, worker thread, downloads: the whole client stack the
Slicer widget sits on, minus Qt. Skipped when the workspace module is not checked
out beside this repo (CI clones only this repo).
"""
import importlib.util
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("requests")
pytest.importorskip("fastapi")

MODULE = Path(__file__).resolve().parents[2] / "slicer" / "NNSegRemote" / "NNSegRemote.py"
if not MODULE.exists():
    pytest.skip("workspace Slicer module not present", allow_module_level=True)

spec = importlib.util.spec_from_file_location("nnseg_slicer_module", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from test_nnseg_serve import FakeSegmenter  # noqa: E402

from nnseg.serve import LocalExecutor, create_app  # noqa: E402


def _spawn_server(tmp_path, **fake_kw):
    import uvicorn
    seg = FakeSegmenter(**fake_kw)
    ex = LocalExecutor(seg, workdir=tmp_path)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(create_app(ex), host="127.0.0.1",
                                           port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    return f"http://127.0.0.1:{port}", seg, server


def _wait(runner, timeout=15.0):
    t0 = time.time()
    while not runner.done and time.time() - t0 < timeout:
        time.sleep(0.05)
    assert runner.done, f"runner still going: {runner.snapshot}"


def test_upload_roundtrip_over_real_http(tmp_path):
    url, seg, server = _spawn_server(tmp_path / "srv", steps=4)
    try:
        proto = mod.RemoteProtocol(url)
        assert proto.health()["name"] == "nnseg"
        assert "total_fast" in proto.tasks()
        up = tmp_path / "scan.nii.gz"
        up.write_bytes(b"\x1f\x8bdata")
        runner = mod.RemoteJobRunner(proto, upload_path=up, task="total_fast",
                                     out_dir=tmp_path / "out")
        _wait(runner)
        assert runner.error is None
        assert runner.snapshot["state"] == "done"
        assert runner.snapshot["result"]["names"]["1"] == "spleen"
        assert runner.result_path is not None and runner.result_path.exists()
        assert runner.result_path.read_bytes().startswith(b"\x1f\x8b")
        assert seg.calls and seg.calls[0][1] == "total_fast"
    finally:
        server.should_exit = True


def test_cancel_over_real_http(tmp_path):
    url, seg, server = _spawn_server(tmp_path / "srv", steps=400)
    try:
        proto = mod.RemoteProtocol(url)
        up = tmp_path / "scan.nii.gz"
        up.write_bytes(b"d")
        runner = mod.RemoteJobRunner(proto, upload_path=up, task="total_fast",
                                     out_dir=tmp_path / "out")
        t0 = time.time()
        while runner.job_id is None and time.time() - t0 < 5:
            time.sleep(0.02)
        time.sleep(0.2)                   # let it get into the tick loop
        runner.cancel()
        _wait(runner)
        assert runner.snapshot["state"] == "cancelled"
        assert runner.result_path is None
    finally:
        server.should_exit = True
