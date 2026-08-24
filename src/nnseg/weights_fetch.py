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
import re
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


def _no_entry(weights_id) -> "ModelNotFound":
    from .errors import ModelNotFound
    wid = str(weights_id)
    if wid in LICENSE_GATED:
        return ModelNotFound(
            f"Dataset{wid} ({LICENSE_GATED[wid]}) is not a public TotalSegmentator release asset - "
            f"it is served from TotalSegmentator's licensed backend. Obtain a license from "
            f"totalsegmentator.com, run `totalseg_set_license`, and let TotalSegmentator download "
            f"it into the weights root; nnseg will then find it.")
    return ModelNotFound(
        f"no manifest entry for Dataset{wid}; try `nnseg weights refresh` to pick up newly "
        f"published weights, or place the model folder under the weights root yourself")


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
        raise _no_entry(weights_id)
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


def ensure_task_weights(task, root, *, catalog=None, progress=None, _seen=None) -> list[Path]:
    """Fetch every model a task needs. Recurses through cascade ``crop_from_task`` stages, so a
    task that crops from another task (teeth <- craniofacial_structures) pulls that chain too.
    Idempotent."""
    from .tasks import TaskCatalog
    cat = catalog or TaskCatalog("totalsegmentator")
    spec = cat.get(task) if isinstance(task, str) else task
    seen = _seen if _seen is not None else set()
    paths = [fetch_one(i, root, progress=progress) for i in spec.weights_ids]
    for st in spec.cascade:
        if st.crop_from_task and st.crop_from_task not in seen:
            seen.add(st.crop_from_task)
            paths += ensure_task_weights(st.crop_from_task, root, catalog=cat, progress=progress, _seen=seen)
    return paths


# -- keeping the manifest current ------------------------------------------------------------
# The manifest is the provisioning mechanism for any machine that does not already have weights
# on disk - a cloud volume, a fresh container, someone else's laptop - so a gap in it means a
# task simply cannot run there. TotalSegmentator publishes weights as release assets, and adds
# to them over time, so the manifest has to be refreshable rather than hand-maintained.
TS_REPO = "wasserth/TotalSegmentator"

# TotalSegmentator serves some models from its own licensed backend rather than as public
# release assets (``commercial_models`` in totalsegmentator/map_to_binary.py): a POST to
# backend.totalsegmentator.com carrying a license number, not a URL anyone can fetch. A URL
# manifest structurally cannot cover them, so they are recorded here and reported as
# "needs a license" rather than "missing" - the difference between an actionable message and
# what looks like a broken manifest.
LICENSE_GATED = {
    "301": "heartchambers_highres", "303": "face", "304": "appendicular_bones",
    "409": "brain_structures", "481": "tissue_types", "485": "tissue_4_types",
    "507": "coronary_arteries_LEGACY", "509": "coronary_arteries", "514": "pulmonary_artery_landmarks",
    "710": "renal_arteries", "713": "aorta_annulus", "716": "aortic_dissection",
    "855": "appendicular_bones_mr", "856": "face_mr", "857": "thigh_shoulder_muscles",
    "920": "aortic_sinuses", "925": "tissue_types_mr",
}
ASSET_RE = re.compile(r"^Dataset(\d+)_.*\.zip$")


def _api(url: str, token: str | None = None) -> list:
    """One GitHub API GET, following pagination. Stdlib only, like the rest of this module."""
    out, page = [], 1
    while True:
        req = urllib.request.Request(f"{url}?per_page=100&page={page}",
                                     headers={"Accept": "application/vnd.github+json",
                                              "User-Agent": "nnseg"})
        tok = token or os.environ.get("GITHUB_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=60) as r:
            batch = json.loads(r.read())
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def discover_release_assets(repo: str = TS_REPO, *, token: str | None = None,
                            progress=None) -> dict[str, dict]:
    """Every ``Dataset<id>_*.zip`` published as a release asset, newest release first.

    Returns ``{weights id: {url, name, tag, size, sha256?}}``. When a dataset appears in more
    than one release the newest wins, which is what TotalSegmentator itself would install.
    Unauthenticated GitHub allows 60 requests/hour; set ``GITHUB_TOKEN`` to lift that.
    """
    say = progress or (lambda s: None)
    say(f"listing releases of {repo}")
    releases = _api(f"https://api.github.com/repos/{repo}/releases", token)
    releases.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    found: dict[str, dict] = {}
    for rel in releases:
        for a in rel.get("assets") or ():
            m = ASSET_RE.match(a.get("name", ""))
            if not m or m.group(1) in found:          # newest release wins
                continue
            entry = {"url": a["browser_download_url"], "name": a["name"],
                     "tag": rel.get("tag_name", ""), "size": a.get("size")}
            digest = a.get("digest") or ""            # "sha256:..." on newer GitHub API responses
            if digest.startswith("sha256:"):
                entry["sha256"] = digest.split(":", 1)[1]
            found[m.group(1)] = entry
    say(f"found {len(found)} dataset assets across {len(releases)} releases")
    return found


def refresh_manifest(path=MANIFEST, *, repo: str = TS_REPO, token: str | None = None,
                     add_missing: bool = True, update_existing: bool = False,
                     write: bool = True, progress=None) -> dict:
    """Merge newly published weights into the manifest.

    Adds entries that are absent. Existing entries are **left alone** unless
    ``update_existing``: repointing a dataset at a newer release changes which weights get
    downloaded, and therefore the segmentations - not something to do silently. The return value
    reports what changed either way, so ``write=False`` is a dry run.
    """
    say = progress or (lambda s: None)
    current = _manifest()
    upstream = discover_release_assets(repo, token=token, progress=progress)

    def url_of(entry):
        return entry.get("url") if isinstance(entry, dict) else entry

    added = {w: e for w, e in upstream.items() if w not in current}
    newer = {w: e for w, e in upstream.items()
             if w in current and url_of(current[w]) != e["url"]}
    merged = dict(current)
    if add_missing:
        merged.update(added)
    if update_existing:
        merged.update(newer)
    say(f"manifest: {len(current)} -> {len(merged)} entries; {len(added)} missing upstream entries"
        f"{' added' if add_missing else ' NOT added'}, {len(newer)} point at a newer release"
        f"{' (applied)' if update_existing else ' (left alone)'}")
    if write and merged != current:
        Path(path).write_text(
            json.dumps({"weights": dict(sorted(merged.items(), key=lambda kv: int(kv[0])))}, indent=2) + "\n")
        say(f"wrote {path}")
    return {"added": added, "newer_upstream": newer, "total": len(merged), "path": str(path)}


def coverage(catalog=None) -> dict:
    """Which catalog tasks the manifest can provision, which need a license, which are missing.

    Three outcomes, not two: a task nnseg cannot download because TotalSegmentator gates it
    behind a license is a different situation from one whose URL we simply do not have, and a
    caller (or a UI) should be able to tell them apart.
    """
    from .tasks import TaskCatalog
    cat = catalog or TaskCatalog("totalsegmentator")
    have = _manifest()
    ok, licensed, missing = [], {}, {}
    for name in cat.names():
        absent = [str(w) for w in cat.get(name).weights_ids if str(w) not in have]
        if not absent:
            ok.append(name)
        elif all(w in LICENSE_GATED for w in absent):
            licensed[name] = absent
        else:
            missing[name] = absent
    return {"covered": ok, "license_required": licensed, "missing": missing,
            "n_weights": len(have), "n_tasks": len(cat)}
