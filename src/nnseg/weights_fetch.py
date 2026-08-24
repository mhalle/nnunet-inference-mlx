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


def _sort_key(item):
    """Numeric ids sort numerically; any non-numeric id sorts after them, alphabetically."""
    k = item[0]
    return (0, int(k), "") if str(k).isdigit() else (1, 0, str(k))


# Written into each unpacked model folder by fetch_one. A sidecar rather than one index at the
# weights root: only the fetch that created the folder writes it, so concurrent fetches into a
# shared volume cannot race, and the record travels with the folder if it is copied elsewhere.
SIDECAR = ".nnseg-version.json"


def installed_version(folder) -> dict | None:
    """What :func:`fetch_one` recorded when it installed this model folder, if anything.

    ``None`` means nnseg did not install it - TotalSegmentator may have, or it was copied in by
    hand. That is reported as unknown rather than guessed at from the manifest: guessing would
    be wrong in exactly the case versioning exists for, where an older version is on disk and
    the manifest has since moved on.
    """
    f = Path(folder)
    for cand in (f / SIDECAR, f.parent / SIDECAR):    # accept a model folder or its dataset dir
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except (json.JSONDecodeError, OSError):
                return None
    return None


def _write_sidecar(dest: Path, weights_id, tag: str, entry: dict, sha256: str | None) -> None:
    import datetime
    rec = {"id": dataset_key(weights_id), "tag": tag, "sha256": sha256 or entry.get("sha256"),
           "url": entry.get("url"), "name": entry.get("name"),
           "installed": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "by": "nnseg"}
    try:
        (dest / SIDECAR).write_text(json.dumps(rec, indent=2) + "\n")
    except OSError:                                   # a read-only weights root must not fail a fetch
        pass


def dataset_key(weights_id) -> str:
    """Canonical manifest key for a dataset id: unpadded decimal.

    TotalSegmentator publishes both ``Dataset008_HepaticVessel`` and ``Dataset297_...``, so the
    same dataset can be written 8 or 008. Without canonicalizing, those become two entries for
    one model.
    """
    t = str(weights_id).strip()
    return str(int(t)) if t.isdigit() else t


def _manifest(path=None) -> dict:
    raw = json.loads(Path(path or MANIFEST).read_text())
    # `raw.get("weights") or raw` would fall through to the wrapper for an EMPTY manifest,
    # leaking the key "weights" in as a dataset id. Test for the key, not its truthiness.
    return _normalize(raw["weights"] if "weights" in raw else raw)


def _normalize(entries: dict) -> dict:
    """Accept both manifest shapes and return the current one.

    An entry is ``{"current": tag, "versions": {tag: {...}}}``: what upstream published is a
    *fact* that refresh may rewrite freely, while which one to install is a *decision* that only
    a human changes. Keeping them parallel means switching versions is a one-token edit and no
    version is a special case. Legacy flat entries (``{"url": ...}``) are lifted into that shape
    on read, so an old manifest still works.
    """
    out: dict = {}
    for raw, e in entries.items():
        wid = dataset_key(raw)
        if not (isinstance(e, dict) and "versions" in e):
            e = e if isinstance(e, dict) else {"url": e}
            tag = e.get("tag") or "unversioned"
            e = {"current": tag, "versions": {tag: e}}
        if wid in out:                                # 8 and 008 are the same dataset
            merged = dict(out[wid]["versions"]); merged.update(e["versions"])
            keep = out[wid]["current"] if out[wid]["current"] != "unversioned" else e["current"]
            out[wid] = {"current": keep if keep in merged else next(iter(merged)), "versions": merged}
        else:
            out[wid] = e
    return out


def selected(entry: dict, tag: str | None = None) -> dict:
    """The version of an entry that would be installed - ``tag`` if given, else ``current``."""
    versions = entry["versions"]
    want = tag or entry.get("current")
    if want not in versions:
        raise KeyError(f"version {want!r} not in manifest; have {sorted(versions)}")
    return versions[want]


def is_present(weights_id, root) -> bool:
    from .tasks import _dataset_dirs
    return bool(_dataset_dirs(Path(root), weights_id))


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


def fetch_one(weights_id, root, *, tag: str | None = None, progress=None) -> Path:
    """Download and unpack one dataset into ``root`` if it is not already there.

    ``tag`` installs a specific published version instead of the manifest's ``current`` one.

    Returns the ``Dataset{id}_*`` directory. Verifies sha256 when the manifest gives one.
    Unpacks to a temp dir and moves into place, so an interrupted download never leaves a
    half-populated model folder that ``is_present`` would accept.
    """
    root = Path(root)
    from .tasks import _dataset_dirs
    existing = _dataset_dirs(root, weights_id)
    if existing:
        return existing[0]
    entry = _manifest().get(dataset_key(weights_id))
    if entry is None:
        raise _no_entry(weights_id)
    chosen_tag = tag or entry.get("current")
    chosen = selected(entry, tag)
    if not chosen.get("url"):                        # a placeholder, e.g. a license-gated dataset
        raise _no_entry(weights_id)
    url, expected = chosen["url"], chosen.get("sha256")
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
        prefix = f"Dataset{int(str(weights_id)):03d}" if str(weights_id).isdigit() else f"Dataset{weights_id}"
        unpacked = next(p for p in tmp.iterdir() if p.is_dir()
                        and (p.name.startswith(f"Dataset{weights_id}") or p.name.startswith(prefix)))
        _write_sidecar(unpacked, weights_id, chosen_tag, chosen, h.hexdigest())
        dest = root / unpacked.name
        os.replace(unpacked, dest)                    # sidecar is inside, so the move stays atomic
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
    for rel in releases:                              # newest first, so current defaults to newest
        tag = rel.get("tag_name", "")
        for a in rel.get("assets") or ():
            m = ASSET_RE.match(a.get("name", ""))
            if not m:
                continue
            wid = m.group(1)
            entry = {"url": a["browser_download_url"], "name": a["name"], "size": a.get("size")}
            digest = a.get("digest") or ""            # "sha256:..." on newer GitHub API responses
            if digest.startswith("sha256:"):
                entry["sha256"] = digest.split(":", 1)[1]
            slot = found.setdefault(dataset_key(wid), {"current": tag, "versions": {}})
            slot["versions"].setdefault(tag, entry)
    n_ver = sum(len(v["versions"]) for v in found.values())
    say(f"found {len(found)} datasets ({n_ver} published versions) across {len(releases)} releases")
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
    current = _manifest(path)
    upstream = discover_release_assets(repo, token=token, progress=progress)

    merged = {w: {"current": e["current"], "versions": dict(e["versions"])} for w, e in current.items()}
    added, new_versions, repointed, migrated = {}, {}, {}, {}
    for wid, up in upstream.items():
        if wid not in merged:
            if add_missing:
                added[wid] = up
                merged[wid] = {"current": up["current"], "versions": dict(up["versions"])}
            continue
        slot = merged[wid]
        fresh = {t: v for t, v in up["versions"].items() if t not in slot["versions"]}
        if fresh:
            new_versions[wid] = sorted(fresh)
            slot["versions"].update(fresh)            # facts: always kept up to date
        # A legacy flat entry carries no tag. Name it by matching its URL against what upstream
        # published - NOT by adopting upstream's newest, which would silently repoint it (297 is
        # published as both v2.0.0 and v2.0.4, and TotalSegmentator itself pins v2.0.0).
        if slot.get("current") == "unversioned":
            was = (slot["versions"].get("unversioned") or {}).get("url")
            match = next((t for t, v in slot["versions"].items()
                          if t != "unversioned" and v.get("url") == was), None)
            if match:
                slot["versions"].pop("unversioned")
                slot["current"] = match
                migrated[wid] = match
            else:
                say(f"  ! Dataset{wid}: current URL matches no published asset; leaving as-is")
        elif update_existing and slot["current"] != up["current"]:
            slot["current"] = up["current"]              # a decision, never made silently
            repointed[wid] = up["current"]

    behind = {w: (merged[w]["current"], upstream[w]["current"])
              for w in upstream if w in merged and merged[w]["current"] != upstream[w]["current"]}
    say(f"manifest: {len(current)} -> {len(merged)} datasets; {len(added)} added, "
        f"{len(new_versions)} gained versions, {len(behind)} not on upstream's newest"
        f"{' (repointed)' if update_existing else ' (left alone)'}")
    if write and merged != current:
        Path(path).write_text(json.dumps(
            {"weights": dict(sorted(merged.items(), key=_sort_key))}, indent=2) + "\n")
        say(f"wrote {path}")
    return {"added": added, "new_versions": new_versions, "behind_upstream": behind,
            "migrated": migrated,
            "total": len(merged), "path": str(path)}


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
        absent = [dataset_key(w) for w in cat.get(name).weights_ids if dataset_key(w) not in have]
        if not absent:
            ok.append(name)
        elif all(w in LICENSE_GATED for w in absent):
            licensed[name] = absent
        else:
            missing[name] = absent
    return {"covered": ok, "license_required": licensed, "missing": missing,
            "n_weights": len(have), "n_tasks": len(cat)}
