"""Model ecosystems: where models come from, as pluggable registries.

The mirror of :mod:`nnseg.sources` - sources answer "where inputs come from",
ecosystems answer "where models come from". A :class:`ModelEcosystem` names its
tasks, installs their weights, and materializes each task's :class:`TaskSpec`.
The rule learned from the stale total_mr class map applies throughout:
**the checkpoint is the spec** - a catalog holds only what the checkpoint
cannot know (where to download it, and how to compose multiple models).

Three ecosystems ship:

- ``ts`` - the TotalSegmentator catalog. Its tasks are *compositions* (unions,
  cascades, remaps) that exist only as application logic, so it carries a full
  task registry (guarded by the remap drift test).
- ``moose`` - MOOSE/moosez models. Bare, self-describing nnU-Net checkpoints
  on public release assets: the manifest holds name -> url + folder, and the
  spec is read from the installed checkpoint's own dataset.json.
- ``native`` - local model folders the operator registers explicitly.

An :class:`EcosystemCatalog` federates a registry of ecosystems behind the
same interface :class:`nnseg.tasks.TaskCatalog` exposes, so ``Segmenter`` and
the server run unchanged. Tasks whose weights are not yet installed are
listed but unmaterialized: ``info()`` says so without downloading, ``get()``
installs on demand (the same behavior TS weights ids always had).

Naming has three layers (user decision 2026-08-25): the **canonical name is
ecosystem-qualified** (``ts:total_fast``, ``moose:clin_ct_fast_organs``) and
is what listings, result-cache keys, and provenance carry; the **short name**
is a resolution convenience, accepted when exactly one ecosystem offers it
and refused with the qualified candidates when ambiguous - so two ecosystems
may legitimately ship the same short name; beneath both sits the **hash**,
the content-addressed result key. Nothing rejects collisions anymore; only
the ambiguous short form becomes unusable.
"""
import json
import re
from pathlib import Path

from .errors import InputError, ModelNotFound
from .tasks import TaskCatalog, TaskSpec

MOOSE_MANIFEST = Path(__file__).parent / "data" / "moose_weights.json"


class ModelEcosystem:
    """One model repository: task names, weight installation, spec loading."""

    name: str = ""
    description: str = ""

    def tasks(self) -> list:
        raise NotImplementedError

    def materialized(self, task: str, root) -> bool:
        """Whether spec() can answer without installing anything."""
        raise NotImplementedError

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        """Install the task's weights under ``root`` (idempotent). ``version``
        pins a release: an already-installed different version is an error,
        never silently served."""
        raise NotImplementedError

    def spec(self, task: str, root) -> TaskSpec:
        """The task's spec; requires materialized() unless the ecosystem
        carries composition data of its own."""
        raise NotImplementedError

    def info(self, task: str, root) -> dict:
        """Cheap metadata that never downloads: ecosystem, materialized, and
        whatever is knowable pre-install."""
        out = {"name": task, "ecosystem": self.name,
               "materialized": self.materialized(task, root)}
        if out["materialized"]:
            spec = self.spec(task, root)
            out["modality"] = spec.modality
            out["structures"] = sorted(spec.label_map.values())
        return out


class TSEcosystem(ModelEcosystem):
    """TotalSegmentator: composed tasks from the shipped registry JSON; weights
    by id from the release-asset manifest (license-gated models refuse with an
    actionable message). Always materialized - the composition data carries
    the label maps, and the remap drift test keeps them honest."""

    name = "ts"
    description = "TotalSegmentator task catalog"

    def __init__(self):
        self._catalog = TaskCatalog("totalsegmentator")

    def tasks(self) -> list:
        return self._catalog.names()

    def materialized(self, task: str, root) -> bool:
        return True

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        from .weights_fetch import ensure_task_weights, installed_version
        paths = ensure_task_weights(task, root, catalog=self._catalog,
                                    progress=progress, tag=version)
        if version is not None:
            for pth in paths:
                rec = installed_version(pth) or {}
                # exact-or-error: an UNKNOWN installed tag (no sidecar - TS's
                # own downloader or a hand copy) must NOT silently satisfy a
                # pin. The whole point of @version is reproducibility.
                if rec.get("tag") != version:
                    have = rec.get("tag") or "unknown (no version sidecar)"
                    raise ModelNotFound(
                        f"{task}@{version}: {Path(pth).name} is installed at "
                        f"tag {have!r} - remove it to install the pinned version")

    def spec(self, task: str, root) -> TaskSpec:
        return self._catalog.get(task)


class MooseEcosystem(ModelEcosystem):
    """MOOSE (moosez): bare nnU-Net checkpoints from public GitHub release
    assets, installed under ``<root>/moose/<Dataset folder>`` and read through
    ``TaskSpec.from_model_folder`` - labels come from each checkpoint's own
    dataset.json, so there is no class map here to drift. The manifest
    (``data/moose_weights.json``, regenerated by tools/gen_moose_manifest.py
    from the moosez registry) holds only name -> url + folder + release tag."""

    name = "moose"
    description = "MOOSE (moosez) model zoo"

    def __init__(self, manifest=None):
        raw = json.loads(Path(manifest or MOOSE_MANIFEST).read_text())
        self._entries = raw["tasks"] if isinstance(raw, dict) else raw

    def tasks(self) -> list:
        return sorted(self._entries)

    def _folder(self, task: str, root) -> Path:
        return Path(root).expanduser() / "moose" / self._entries[task]["folder"]

    def materialized(self, task: str, root) -> bool:
        if task not in self._entries:
            return False
        d = self._folder(task, root)
        return d.is_dir() and any(d.rglob("dataset.json"))

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        from .weights_fetch import installed_version
        entry = self._entries[task]
        if version is not None:
            # check the INSTALLED tag, always - matching the manifest tag is
            # not proof the bytes ON DISK are that release (a regenerated
            # manifest bumps the tag; the old folder keeps its old sidecar)
            rec = installed_version(self._folder(task, root)) or {}
            if rec.get("tag") == version:
                return                     # provably the pinned bytes
            if self.materialized(task, root):
                have = rec.get("tag") or "unknown (no version sidecar)"
                raise ModelNotFound(
                    f"{task}@{version}: installed at tag {have!r} - remove "
                    f"{self._folder(task, root).name} to install the pinned version")
            if version != entry.get("tag"):
                raise ModelNotFound(
                    f"{task}@{version}: this manifest offers tag "
                    f"{entry.get('tag')!r} only")
        if self.materialized(task, root):
            return
        dest_parent = Path(root).expanduser() / "moose"
        dest_parent.mkdir(parents=True, exist_ok=True)
        _download_and_extract_zip(entry["url"], dest_parent, progress=progress,
                                  sha256=entry.get("sha256"))
        folder = self._folder(task, root)
        if not folder.is_dir():
            raise ModelNotFound(
                f"moose asset for {task!r} unpacked without the expected folder "
                f"{entry['folder']!r} - the manifest may be stale; regenerate it "
                "with tools/gen_moose_manifest.py")
        from .weights_fetch import _write_sidecar
        _write_sidecar(folder, task, entry.get("tag", "unknown"),
                       {"url": entry["url"]}, None)

    def spec(self, task: str, root) -> TaskSpec:
        if not self.materialized(task, root):
            raise ModelNotFound(
                f"moose task {task!r} is not installed under {root}; prepare it "
                "first (weights install on demand when the task runs)")
        return TaskSpec.from_model_folder(self._folder(task, root), name=task)

    def info(self, task: str, root) -> dict:
        out = super().info(task, root)
        m = re.match(r"(clin|preclin)_(ct|mr|pt|fdg_pt|pt_fdg)_", task)
        if "modality" not in out and m:
            out["modality"] = m.group(2).upper().replace("FDG_PT", "PT").replace("PT_FDG", "PT")
        out["tag"] = self._entries.get(task, {}).get("tag")
        return out


class NativeEcosystem(ModelEcosystem):
    """Local model folders registered by the operator: always materialized,
    nothing to install. The folder is read through from_model_folder, so the
    checkpoint's dataset.json is the spec here too."""

    name = "native"
    description = "operator-registered local nnU-Net model folders"

    def __init__(self, models: dict | None = None):
        self._models = {str(k): Path(v) for k, v in (models or {}).items()}

    def tasks(self) -> list:
        return sorted(self._models)

    def materialized(self, task: str, root) -> bool:
        return task in self._models and self._models[task].is_dir()

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        if not self.materialized(task, root):
            raise ModelNotFound(f"native task {task!r}: folder "
                                f"{self._models.get(task)} does not exist")
        if version is not None:
            from .weights_fetch import installed_version
            rec = installed_version(self._models[task]) or {}
            if rec.get("tag") != version:
                raise ModelNotFound(
                    f"{task}@{version}: native folder records "
                    f"{rec.get('tag') or 'no version metadata'}")

    def spec(self, task: str, root) -> TaskSpec:
        self.ensure(task, root)
        return TaskSpec.from_model_folder(self._models[task], name=task)


def _download_and_extract_zip(url: str, dest_parent: Path, *, progress=None,
                              sha256: str | None = None) -> None:
    """Fetch a release zip and unpack it under ``dest_parent`` atomically,
    verifying ``sha256`` when the manifest supplies one and refusing any
    member whose path would escape (zip-slip)."""
    import os
    import shutil
    import tempfile
    import urllib.request
    import zipfile
    if progress:
        progress(f"downloading {url.rsplit('/', 1)[-1]}")
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=dest_parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=1800) as r:
                while chunk := r.read(1 << 20):
                    tmp.write(chunk)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise InputError(f"weights download failed: {e}") from e
    if sha256:
        import hashlib
        h = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != sha256:
            tmp_path.unlink(missing_ok=True)
            raise InputError(f"weights zip digest mismatch: expected {sha256}, "
                             f"got {h.hexdigest()}")
    # extract into a sibling temp dir, then os.replace into place: an
    # interrupted unpack must never leave a folder that materialized() calls
    # complete (dataset.json present, checkpoints missing) - the exact trap
    # fetch_one's temp+rename avoids on the TS side
    staging = Path(tempfile.mkdtemp(dir=dest_parent, prefix=".unzip-"))
    try:
        with zipfile.ZipFile(tmp_path) as z:
            base = staging.resolve()
            for m in z.infolist():
                target = (staging / m.filename).resolve()
                if not target.is_relative_to(base):
                    raise InputError(f"zip member escapes destination: {m.filename!r}")
            z.extractall(staging)
        for child in staging.iterdir():        # move the unpacked top-level
            dest = dest_parent / child.name     # folder(s) into place atomically
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            os.replace(child, dest)
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def registry(ecosystems=None) -> dict:
    """Normalize a list of ecosystems into ``{name: ecosystem}``. Duplicate
    ecosystem names are rejected; duplicate *task* names across ecosystems are
    fine - the canonical ``eco:task`` form disambiguates, and only the short
    form goes ambiguous."""
    out = {}
    for e in (default_ecosystems() if ecosystems is None else list(ecosystems)):
        if not e.name or ":" in e.name or e.name in out:
            raise ValueError(f"bad or duplicate ecosystem name {e.name!r}")
        out[e.name] = e
    return out


def default_ecosystems() -> list:
    return [TSEcosystem(), MooseEcosystem()]


class EcosystemCatalog:
    """A TaskCatalog-compatible federation over an ecosystem registry.

    Canonical names are ``eco:task``; short names resolve when unique. ``get()``
    materializes on demand (installing weights if needed - the same
    on-first-use behavior TS weights ids always had); ``info()`` never
    downloads. A TaskSpec or a model-folder path passes through ``get``
    untouched, as with TaskCatalog."""

    def __init__(self, ecosystems=None, *, root=None):
        self.registry = registry(ecosystems)
        self.root = root
        self._short: dict[str, list] = {}
        for ename, e in self.registry.items():
            for t in e.tasks():
                self._short.setdefault(t, []).append(ename)

    def names(self) -> list:
        return sorted(f"{ename}:{t}" for t, enames in self._short.items()
                      for ename in enames)

    def resolve(self, name: str) -> tuple:
        """``(ecosystem, short_task, canonical, version)`` for any name form.

        The grammar is ``[eco:]name[@version]`` - three layers plus the hash
        beneath: the short name resolves when exactly one ecosystem offers it,
        the canonical form is ecosystem-qualified, and ``@version`` pins a
        weights release at install time (the canonical name stays unversioned;
        actual versions live in the result key's weights component). Unknown
        names and ambiguous short names raise LookupError - the ambiguity
        error lists the qualified candidates to use instead."""
        name = str(name)
        version = None
        if "@" in name:
            name, _, version = name.rpartition("@")
            if not version or not name:
                raise LookupError(f"malformed task name {name!r}@{version!r}")
        if ":" in name:
            ename, _, short = name.partition(":")
            eco = self.registry.get(ename)
            if eco is None or ename not in self._short.get(short, ()):
                raise LookupError(f"unknown task {name!r}")
            return eco, short, f"{ename}:{short}", version
        enames = self._short.get(name)
        if not enames:
            raise LookupError(f"unknown task {name!r}; {len(self._short)} known, "
                              f"e.g. {self.names()[:6]}")
        if len(enames) > 1:
            raise LookupError(f"short name {name!r} is ambiguous; use one of "
                              + ", ".join(f"{e}:{name}" for e in sorted(enames)))
        return self.registry[enames[0]], name, f"{enames[0]}:{name}", version

    def __contains__(self, name) -> bool:
        try:
            self.resolve(name)
            return True
        except LookupError:
            return False

    def __len__(self) -> int:
        return sum(len(v) for v in self._short.values())

    def ecosystem_of(self, name: str):
        try:
            return self.resolve(name)[0]
        except LookupError:
            return None

    def get(self, name) -> TaskSpec:
        if isinstance(name, TaskSpec):
            return name
        eco, short, canonical, version = self.resolve(name)
        if version is not None or not eco.materialized(short, self.root):
            eco.ensure(short, self.root, version=version)
        spec = eco.spec(short, self.root)
        if spec.name != canonical:
            import dataclasses
            spec = dataclasses.replace(spec, name=canonical)
        return spec

    __getitem__ = get

    def info(self, name: str) -> dict:
        eco, short, canonical, version = self.resolve(name)
        out = eco.info(short, self.root)
        out["name"] = canonical
        if version is not None:
            out["version_requested"] = version
        return out

    def prepare(self, name: str, progress=None) -> dict:
        """Install the task's weights now and return its full info."""
        eco, short, canonical, version = self.resolve(name)
        eco.ensure(short, self.root, progress=progress, version=version)
        out = eco.info(short, self.root)
        out["name"] = canonical
        return out
