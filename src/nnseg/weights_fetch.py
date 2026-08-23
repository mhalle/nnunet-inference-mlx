"""Provision model weights on demand from their official source.

TotalSegmentator publishes each dataset as a zip on its own GitHub releases; the manifest
(``data/ts_weights.json``, id -> url + optional sha256) maps a weights id to its asset. This
module downloads and unpacks the ones a task needs into a weights root, skipping any already
present - so a fresh machine (or a cloud volume) fills its cache on first run and never
re-downloads. Nothing is redistributed by us: the URLs point at wasserth's releases.

Stdlib only (urllib + zipfile), so importing nnseg never pulls a download stack it will not use.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MANIFEST = Path(__file__).parent / "data" / "ts_weights.json"


def _manifest() -> dict:
    raw = json.loads(MANIFEST.read_text())
    return raw.get("weights") or raw


def is_present(weights_id, root) -> bool:
    return bool(sorted(Path(root).glob(f"Dataset{weights_id}_*")))


def fetch_one(weights_id, root, *, progress=None) -> Path:
    """Download and unpack one dataset into ``root`` if it is not already there.

    Returns the ``Dataset{id}_*`` directory. Verifies sha256 when the manifest gives one.
    Unpacks to a temp dir and moves into place, so an interrupted download never leaves a
    half-populated model folder that ``is_present`` would accept.
    """
    root = Path(root)
    existing = sorted(root.glob(f"Dataset{weights_id}_*"))
    if existing:
        return existing[0]
    entry = _manifest().get(str(weights_id))
    if entry is None:
        raise KeyError(f"weights id {weights_id} is not in the manifest ({MANIFEST.name})")
    url, expected = entry["url"], entry.get("sha256")
    say = progress or (lambda m: None)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmp = Path(tmp)
        archive = tmp / f"Dataset{weights_id}.zip"
        say(f"downloading Dataset{weights_id} from {url.rsplit('/', 1)[-1]}")
        h = hashlib.sha256()
        with urllib.request.urlopen(url) as r, open(archive, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
        if expected and h.hexdigest() != expected:
            raise ValueError(f"sha256 mismatch for Dataset{weights_id}: {h.hexdigest()} != {expected}")
        say(f"unpacking Dataset{weights_id}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        unpacked = next(p for p in tmp.iterdir() if p.is_dir() and p.name.startswith(f"Dataset{weights_id}"))
        dest = root / unpacked.name
        os.replace(unpacked, dest)
    return dest


def ensure_task_weights(task, root, *, catalog=None, progress=None) -> list[Path]:
    """Fetch every model a task needs (single, or all union parts). Idempotent."""
    from .tasks import TaskCatalog
    spec = (catalog or TaskCatalog("totalsegmentator")).get(task) if isinstance(task, str) else task
    ids = []
    if spec.single is not None:
        ids.append(spec.single)
    ids += [p.weights_id for p in spec.union]
    return [fetch_one(i, root, progress=progress) for i in ids]
