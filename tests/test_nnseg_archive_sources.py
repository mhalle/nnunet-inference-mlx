"""The archive-reading source family, tested against a local in-process HTTP
server that serves Range requests - no network, real bytes."""
import http.server
import io
import json
import threading
import zipfile
from pathlib import Path

import pytest

from nnseg.errors import InputError
from nnseg.sources import (ArchiveReadingSource, HuggingFaceSource, RangeFile,
                           ZenodoSource, registry)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        body = self.payload
        if rng:
            lo, hi = rng.split("=")[1].split("-")
            lo, hi = int(lo), min(int(hi), len(body) - 1)
            chunk = body[lo:hi + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(body)}")
        else:
            chunk = body
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def range_server():
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("case1/ct.nii.gz", b"\x1f\x8b" + b"CT" * 500)
        z.writestr("case1/seg.nii.gz", b"\x1f\x8b" + b"SG" * 300)
        z.writestr("case2/ct.nii.gz", b"\x1f\x8b" + b"XX" * 400)
        z.writestr("../evil.txt", b"nope")
    _RangeHandler.payload = zbuf.getvalue()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/archive.zip", len(_RangeHandler.payload)
    srv.shutdown()


def test_rangefile_reads_match_direct_bytes(range_server):
    url, size = range_server
    rf = RangeFile(url, size, block=256)
    rf.seek(-100, 2)
    tail = rf.read(100)
    assert tail == _RangeHandler.payload[-100:]
    rf.seek(10)
    assert rf.read(50) == _RangeHandler.payload[10:60]
    assert rf.requests >= 1


class _Src(ArchiveReadingSource):
    prefix = "tst"
    id_pattern = r"[a-z.]+(?:![A-Za-z0-9._/]+)?"

    def __init__(self, url, size):
        self._url, self._size = url, size
        self.seen_credentials = []

    def resolve(self, outer, credentials=None):
        self.seen_credentials.append(credentials)
        return self._url, self._size


def test_member_extraction_fetches_member_only(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    d = src.fetch("archive.zip!case1/ct.nii.gz", tmp_path)
    assert (d / "ct.nii.gz").read_bytes().startswith(b"\x1f\x8bCT")
    assert sorted(p.name for p in d.iterdir()) == ["ct.nii.gz"]


def test_folder_prefix_extracts_the_case(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    d = src.fetch("archive.zip!case1/", tmp_path)
    assert sorted(p.name for p in d.iterdir()) == ["ct.nii.gz", "seg.nii.gz"]


def test_zip_slip_member_lands_flat(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    d = src.fetch("archive.zip!../evil.txt", tmp_path)
    assert (d / "evil.txt").exists()                 # flattened by basename
    assert not (tmp_path.parent / "evil.txt").exists()


def test_plain_file_downloads_whole(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    d = src.fetch("archive.zip", tmp_path)
    assert (d / "archive.zip").stat().st_size == size


def test_credentials_reach_the_resolver(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    src.fetch("archive.zip!case2/ct.nii.gz", tmp_path, credentials="tok123")
    assert "tok123" in src.seen_credentials


def test_missing_member_is_a_clear_error(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    with pytest.raises(InputError, match="no such member"):
        src.fetch("archive.zip!nope/", tmp_path)


def test_grammars_and_registry():
    import re
    reg = registry(None)
    assert set(reg) >= {"idc", "tcia", "openneuro", "zenodo", "hf"}
    z = reg["zenodo"].id_pattern
    assert re.fullmatch(z, "6802614/Totalsegmentator_dataset.zip!Totalsegmentator_dataset/s0001/ct.nii.gz")
    assert re.fullmatch(z, "6802614/labels.json")
    h = reg["hf"].id_pattern
    sha = "a" * 40
    assert re.fullmatch(h, f"org-x/data.set@{sha}/images/a.nii.gz!inner/file.nii.gz")
    assert not re.fullmatch(h, "org/data@main/images/a.nii.gz")   # floating refs refused
    assert not re.fullmatch(h, f"org/data@{'a' * 39}/x.nii.gz")


def test_token_header_reaches_fetch_and_never_leaks(tmp_path):
    """NNSeg-Source-Token flows to the source's fetch and appears nowhere a
    client can read: not in status, not in the jobs listing, not in cache
    meta. The prefetch (which has no request context) fails without the
    token and the dispatcher's inline fetch with it succeeds - the existing
    fallback covers restricted prefetches by design."""
    import json as _json

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nnseg.serve import LocalExecutor, create_app
    from test_nnseg_serve import FakeSegmenter, wait_state

    seen = []

    class TokSource:
        prefix = "tok"
        id_pattern = r"[a-z0-9]+"
        description = ""

        def enabled(self):
            return True

        def identity(self, i):
            return f"tok:{i}"

        def describe(self):
            return {"prefix": "tok", "id_pattern": self.id_pattern,
                    "enabled": True, "description": ""}

        def fetch(self, ident, dest_dir, *, credentials=None):
            seen.append(credentials)
            if credentials != "sekrit":
                raise InputError("token required")
            d = dest_dir / "series"
            d.mkdir(parents=True, exist_ok=True)
            (d / "x.nii.gz").write_bytes(b"\x1f\x8b" + ident.encode())
            return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       sources=[TokSource()])
    client = TestClient(create_app(ex))
    r = client.post("/v1/jobs",
                    data={"task": "total_fast",
                          "source": _json.dumps([{"kind": "tok", "id": "abc123"}])},
                    headers={"NNSeg-Source-Token": "tok=sekrit"})
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    st = wait_state(client, jid, ("done",))
    assert "sekrit" in seen                          # the fetch got the token
    assert "sekrit" not in _json.dumps(st)           # ...and no client surface does
    assert "sekrit" not in client.get("/v1/jobs").text
    assert "sekrit" not in client.get("/v1/segmentations").text
    assert "source_tokens" not in st


def test_slashed_identifiers_get_the_full_path_surface(tmp_path, monkeypatch):
    """Multi-segment identifiers (hf paths, zenodo recid/file!member) live in
    ordinary URLs via greedy right-to-left routes: blocking GET initiates,
    HEAD probes, meta and preview serve, DELETE evicts, and the listing links
    the slashed path."""
    import json as _json

    import numpy as np
    import SimpleITK as sitk
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nnseg.serve import LocalExecutor, create_app
    from test_nnseg_serve import FakeSegmenter

    class DeepSource:
        prefix = "deep"
        id_pattern = r"[a-z0-9]+/[a-z0-9@.]+/[a-z0-9._/!-]+"
        description = ""

        def enabled(self):
            return True

        def identity(self, i):
            return f"deep:{i}"

        def describe(self):
            return {"prefix": "deep", "id_pattern": self.id_pattern,
                    "enabled": True, "description": ""}

        def fetch(self, ident, dest_dir, *, credentials=None):
            d = dest_dir / "series"
            d.mkdir(parents=True, exist_ok=True)
            a = np.full((16, 24, 24), -1000, np.int16)
            a[4:12, 6:18, 6:18] = 40
            sitk.WriteImage(sitk.GetImageFromArray(a), str(d / "vol.nii.gz"))
            return d

    class SavingSeg(FakeSegmenter):
        def segment(self, image, task, *, progress=None, cancel=None, **options):
            self.calls.append((str(image), task, options))

            class R:
                def save(_, path):
                    a = np.zeros((16, 24, 24), np.uint8)
                    a[5:10, 8:16, 8:16] = 1
                    img = sitk.GetImageFromArray(a)
                    img.SetMetaData("Segment0_Name", "blob")
                    img.SetMetaData("Segment0_LabelValue", "1")
                    img.SetMetaData("Segment0_Color", "0.2 0.6 0.9")
                    sitk.WriteImage(img, str(path))
                    return path
                schema = type("S", (), {"names": {1: "blob"}})()
                def volumes_ml(_):
                    return {"blob": 1.0}
                provenance = {}
            return R()

    seg = SavingSeg()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       sources=[DeepSource()])
    client = TestClient(create_app(ex))
    ident = "org1/repo@abc123.def/images/case7.nii.gz!inner/ct.nii.gz"
    url = f"/v1/deep/{ident}/total_fast/labels.seg.nrrd"

    r = client.get(url, headers={"Prefer": "wait=30"})   # blocking GET initiates
    assert r.status_code == 200, r.text
    assert client.head(url).status_code == 200           # probe sees it
    m = client.get(f"/v1/deep/{ident}/total_fast/meta.json")
    assert m.status_code == 200
    try:
        import matplotlib  # noqa: F401
        from test_nnseg_serve import wait_artifact
        p = wait_artifact(client, f"/v1/deep/{ident}/total_fast/preview.png")
        assert p.status_code == 200 and p.content[:4] == b"\x89PNG"
    except ImportError:
        pass
    segs = client.get("/v1/segmentations").json()["segmentations"]
    e = next(x for x in segs if x["identity"] == [f"deep:{ident}"])
    assert e["links"]["labels"] == url                   # slashed path listed
    d = client.delete(url)
    assert d.status_code == 200 and d.json()["deleted"]
    assert client.head(url).status_code == 404           # gone
    assert len(seg.calls) == 1


def test_archive_cache_isolates_credentials(range_server, tmp_path):
    """Round 4 (HIGH): the archive cache keys on (outer, credentials), so a
    caller with no token cannot reuse an earlier caller's authenticated
    archive - resolve() (the access gate + token binding) runs for each."""
    url, size = range_server
    src = _Src(url, size)
    for sub in ("a", "b", "c"):
        (tmp_path / sub).mkdir()
    src.fetch("archive.zip!case1/ct.nii.gz", tmp_path / "a", credentials="ALICE")
    src.fetch("archive.zip!case2/ct.nii.gz", tmp_path / "b", credentials=None)
    # both callers triggered a resolve(): the tokenless one was NOT served
    # from Alice's cached archive
    assert "ALICE" in src.seen_credentials and None in src.seen_credentials
    # same credential reuses the cached archive (one resolve for two members)
    before = len(src.seen_credentials)
    src.fetch("archive.zip!case1/seg.nii.gz", tmp_path / "c", credentials="ALICE")
    assert len(src.seen_credentials) == before        # cache hit, no re-resolve


def test_rangefile_refuses_status_200(tmp_path):
    """Round 4: a server that ignores Range and streams the whole body (200)
    is refused, not pulled into memory."""
    class _Ignore(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)                    # ignores Range
            self.send_header("Content-Length", "8")
            self.end_headers()
            self.wfile.write(b"WHOLEBODY"[:8])

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ignore)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rf = RangeFile(f"http://127.0.0.1:{srv.server_address[1]}/x", 8, block=4)
        with pytest.raises(InputError, match="Range"):
            rf.read(4)
    finally:
        srv.shutdown()


def test_dotdot_identifier_refused(range_server, tmp_path):
    """Round 4: a '..' segment in the OUTER identifier is refused (it would
    steer the resolve/template URL off the operator's prefix)."""
    url, size = range_server
    src = _Src(url, size)
    with pytest.raises(InputError, match=r"\.\."):
        src.fetch("../../secret/archive.zip!case1/ct.nii.gz", tmp_path)


def test_zenodo_access_fails_closed(monkeypatch):
    """Round 4: absent/empty access fields must read as restricted, not open."""
    import urllib.request

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=None):
        return FakeResp(json.dumps({"files": [{"key": "x.nii.gz", "size": 10,
                        "links": {"content": "http://h/x"}}]}).encode())

    import nnseg.sources as srcmod
    monkeypatch.setattr(srcmod._OPENER, "open", fake_open)
    z = ZenodoSource()                                 # allow_restricted=False
    with pytest.raises(InputError, match="undeclared access|restricted|only fetches"):
        z.resolve("1234567/x.nii.gz")


def test_redirect_strips_auth_on_scheme_downgrade():
    """Round-4 sign-off: the token is dropped on an https->http downgrade
    even on the same host (cleartext exposure), while an http->https upgrade
    keeps it - matching httpx."""
    import nnseg.sources as srcmod
    opener = srcmod._safe_opener()
    (h,) = [x for x in opener.handlers
            if type(x).__name__ == "_StripCrossHostAuth"]
    import urllib.request

    def mk(url):
        r = urllib.request.Request(url, headers={"Authorization": "Bearer T"})
        return r

    # https -> http, same host: stripped
    new = h.redirect_request(mk("https://h/a"), None, 302, "", {},
                             "http://h/b")
    assert not any(k.lower() == "authorization" for k in new.headers)
    # http -> https, same host: kept (safe upgrade)
    new = h.redirect_request(mk("http://h/a"), None, 302, "", {},
                             "https://h/b")
    assert any(k.lower() == "authorization" for k in new.headers)
    # https -> https, same host: kept
    new = h.redirect_request(mk("https://h/a"), None, 302, "", {},
                             "https://h/b")
    assert any(k.lower() == "authorization" for k in new.headers)
    # cross host: stripped
    new = h.redirect_request(mk("https://h/a"), None, 302, "", {},
                             "https://evil/b")
    assert not any(k.lower() == "authorization" for k in new.headers)
