"""A small client for the nnseg REST job protocol (`nnseg remote ...`).

Talks to any server that implements the contract in :mod:`nnseg.serve` - a local
`nnseg serve`, a lab GPU box, or the Modal deployment. Progress arrives over the SSE
stream when the server offers it and silently falls back to polling: both surfaces
carry the same idempotent status snapshots, so the fallback changes latency, not
meaning. Needs ``httpx`` (the ``remote`` extra).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .errors import InputError, NnsegError


class RemoteError(NnsegError):
    """The server refused or failed a request."""


class RemoteClient:
    def __init__(self, server: str, *, token: str | None = None, timeout: float = 60.0):
        try:
            import httpx
        except ImportError as e:
            raise InputError("the remote client needs httpx: uv sync --extra remote "
                             "(or pip install 'nnseg[remote]')") from e
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(base_url=server.rstrip("/"), headers=headers,
                                  timeout=timeout, follow_redirects=True)

    # -- plumbing ------------------------------------------------------------
    def _json(self, method: str, path: str, **kw) -> dict:
        r = self._http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RemoteError(f"{method} {path} -> {r.status_code}: {detail}")
        return r.json()

    # -- the protocol --------------------------------------------------------
    def health(self) -> dict:
        return self._json("GET", "/v1/health")

    def tasks(self) -> list[str]:
        return self._json("GET", "/v1/tasks")["tasks"]

    def describe(self, task: str) -> dict:
        return self._json("GET", f"/v1/tasks/{task}")

    def submit(self, image, task: str, **options) -> str:
        """``image`` is a local file to upload, or ``"<source>:<identifier>"``
        (e.g. ``"idc:<crdc_series_uuid>"``) to have the server fetch the input
        from one of its registered data sources. A path that exists locally
        always wins over the shorthand reading."""
        data = {"task": task, "options": json.dumps(options)}
        img = str(image)
        prefix = img.split(":", 1)[0] if ":" in img else ""
        if prefix.isidentifier() and prefix.islower() and not Path(img).exists():
            ident = img.split(":", 1)[1]
            src = {"kind": prefix, "id": ident}
            if prefix == "idc":
                src["crdc_series_uuid"] = ident
            data["source"] = json.dumps([src])
            r = self._json("POST", "/v1/jobs", data=data)
        else:
            p = Path(image)
            with open(p, "rb") as f:
                r = self._json("POST", "/v1/jobs", data=data,
                               files={"file": (p.name, f, "application/octet-stream")})
        return r["id"]

    def status(self, job_id: str) -> dict:
        return self._json("GET", f"/v1/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict:
        return self._json("DELETE", f"/v1/jobs/{job_id}")

    def fetch(self, job_id: str, output) -> Path:
        out = Path(output)
        with self._http.stream("GET", f"/v1/jobs/{job_id}/result") as r:
            if r.status_code >= 400:
                r.read()
                raise RemoteError(f"result -> {r.status_code}: {r.text}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return out

    # -- progress ------------------------------------------------------------
    def events(self, job_id: str):
        """Status snapshots from the SSE stream; raises on transport failure -
        callers that need robustness use :meth:`wait`, which falls back to polling."""
        with self._http.stream("GET", f"/v1/jobs/{job_id}/events",
                               timeout=None) as r:
            if r.status_code >= 400:
                r.read()
                raise RemoteError(f"events -> {r.status_code}")
            for line in r.iter_lines():
                if line.startswith("data:"):
                    yield json.loads(line[5:].strip())

    def wait(self, job_id: str, *, on_status=None, poll_interval: float = 0.5) -> dict:
        """Block until the job is terminal; returns the final status. Prefers SSE,
        falls back to polling on any stream problem."""
        terminal = ("done", "failed", "cancelled")
        last = None
        try:
            for snap in self.events(job_id):
                last = snap
                if on_status:
                    on_status(snap)
                if snap["state"] in terminal:
                    return snap
        except Exception:
            pass                                   # stream unavailable - poll instead
        while True:
            snap = self.status(job_id)
            if snap != last and on_status:
                on_status(snap)
            last = snap
            if snap["state"] in terminal:
                return snap
            time.sleep(poll_interval)

    def run(self, image, task: str, output, *, on_status=None, **options) -> dict:
        """submit + wait + fetch: the whole round trip. Returns the final status."""
        jid = self.submit(image, task, **options)
        final = self.wait(jid, on_status=on_status)
        if final["state"] == "done":
            self.fetch(jid, output)
        elif final["state"] == "failed":
            raise RemoteError(f"job {jid} failed: {final.get('error', 'unknown')}")
        return final
