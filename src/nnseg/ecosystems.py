"""Model ecosystems: where models come from, as pluggable registries.

The mirror of :mod:`nnseg.sources` - sources answer "where inputs come from",
ecosystems answer "where models come from". A :class:`ModelEcosystem` names its
tasks, installs their weights, and materializes each task's :class:`TaskSpec`.
The rule learned from the stale total_mr class map applies throughout:
**the checkpoint is the spec** - a catalog holds only what the checkpoint
cannot know (where to download it, and how to compose multiple models).

**An ecosystem is a catalog, not a runtime.** What runs a task is an *engine*
(:mod:`nnseg.engines.registry`), named by ``ModelEcosystem.engine``; many
ecosystems map to one engine. The three nnU-Net catalogs below all run on
``nnunetv2``; FastSurfer and SynthStrip each bring a catalog *and* an engine.

The nnU-Net catalogs:

- ``ts`` - the TotalSegmentator catalog. Its tasks are *compositions* (unions,
  cascades, remaps) that exist only as application logic, so it carries a full
  task registry (guarded by the remap drift test).
- ``moose`` - MOOSE/moosez models. Bare, self-describing nnU-Net checkpoints
  on public release assets: the manifest holds name -> url + folder, and the
  spec is read from the installed checkpoint's own dataset.json.
- ``custom`` - local model folders the operator registers explicitly.

Engine catalogs (present only where their engine is enabled, so the catalog can
never list a task no worker can run): ``fastsurfer``, ``synthstrip``.

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
import os
import re
import shutil
import tempfile
from pathlib import Path

from .engines import registry as _registry
from .errors import InputError, ModelNotFound, UnsupportedModel
from .tasks import TaskCatalog, TaskSpec

MOOSE_MANIFEST = Path(__file__).parent / "data" / "moose_weights.json"


class ModelEcosystem:
    """One model catalog: task names, weight installation, spec loading.

    An ecosystem is what the *user* selects. What actually runs its tasks is an
    :mod:`~nnseg.engines.registry` engine, named by :attr:`engine` - many
    ecosystems map to one engine (``ts``, ``moose`` and ``custom`` are three
    catalogs of nnU-Net models, all run by ``nnunetv2``).

    An ecosystem whose engine has no :class:`~nnseg.tasks.TaskSpec` - FastSurfer
    and SynthStrip run their own networks - sets :attr:`has_task_spec` to False
    and inherits the whole interface below unchanged; it does not need to
    override anything to refuse.

    Two axes, deliberately independent, because a catalog of engine models needs
    one of each and welding them together is what would make it fight this class:
    :attr:`has_task_spec` says whether the nnU-Net pipeline can run the task;
    :meth:`materialized` / :meth:`ensure` say where the weights come from. The
    per-task hooks :meth:`weights_identity` and :meth:`describe_task` exist for
    the same reason - a one-model engine answers both from constants, a catalog
    answers them from its manifest and its installed models.
    """

    name: str = ""
    description: str = ""
    #: Which registry engine runs this ecosystem's tasks.
    engine: str = _registry.NNUNETV2
    #: False when tasks are run by an engine's own network rather than an
    #: nnU-Net TaskSpec (drives ``spec()`` and the ``task_spec`` info flag).
    #: Independent of where the weights live - an ecosystem can have no TaskSpec
    #: and still install per task; see :meth:`materialized` / :meth:`ensure`.
    has_task_spec: bool = True
    #: Pre-install metadata for ecosystems that know it statically (a one-model
    #: engine knows its own modality and label set). A catalog leaves these None
    #: and answers per task instead - see :meth:`describe_task`.
    modality: str | None = None
    structures: list | None = None

    def tasks(self) -> list:
        raise NotImplementedError

    def weights_identity(self, task: str, root) -> list | None:
        """This task's contribution to the result-cache key, or None.

        Defaults to the engine's constant identity, which is what a one-model
        engine has (its weights ship with its image). A *catalog* of models
        overrides this to answer per task - it already holds a manifest with
        versions in memory, so this stays cheap. It must be cheap: ``info()``
        runs once per task on ``/v1/tasks`` and ``/v1/version``, so anything that
        walks the weights volume here lands on two hot endpoints.

        Returns None when the engine has no constant and the ecosystem declares
        nothing - the nnU-Net path, where the identity is computed per task from
        the spec's weights ids by :meth:`nnseg.segmenter.Segmenter.describe`.
        """
        identity = _registry.ENGINES[self.engine].weights_identity
        return identity() if identity is not None else None

    def describe_task(self, task: str, root) -> dict:
        """Per-task metadata an ecosystem can read from the model itself once it
        is installed - ``modality`` and ``structures``. Empty by default; the
        nnU-Net ecosystems get theirs from the TaskSpec, one-model engines from
        their class attributes, and a catalog reads its model's own metadata
        (the "the checkpoint is the spec" rule)."""
        return {}

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
        carries composition data of its own.

        Refuses with :class:`UnsupportedModel` when the ecosystem has no
        TaskSpec at all (an engine's own network), and with
        ``NotImplementedError`` otherwise - a half-written ecosystem must keep
        looking like a bug, not like an unsupported model.
        """
        if not self.has_task_spec:
            env = _registry.ENGINES[self.engine].enabled_env
            raise UnsupportedModel(
                f"{self.name} is an engine, not an nnU-Net task: it runs on the "
                f"{self.engine} engine, which has no TaskSpec. This server runs "
                f"nnU-Net models in-process; deploy with "
                f"{env}=1 to serve it from an engine worker.")
        raise NotImplementedError

    def info(self, task: str, root) -> dict:
        """Cheap metadata that never downloads: ecosystem, engine, materialized,
        and whatever is knowable pre-install.

        Uniform for every ecosystem, engines included - one shape for clients.
        ``structures`` is always a list; ``weights_installed`` appears when the
        engine carries a constant weights identity (engines bake their weights
        into their image), and is otherwise computed per task from the spec's
        install sidecars by :meth:`nnseg.segmenter.Segmenter.describe`.
        """
        out = {"name": task, "ecosystem": self.name, "engine": self.engine,
               "task_spec": self.has_task_spec,
               "materialized": self.materialized(task, root)}
        if self.modality is not None:
            out["modality"] = self.modality
        if self.structures is not None:
            out["structures"] = list(self.structures)
        identity = self.weights_identity(task, root)
        if identity is not None:
            out["weights_installed"] = identity
        if out["materialized"]:
            if self.has_task_spec:
                spec = self.spec(task, root)
                out["modality"] = spec.modality
                out["structures"] = sorted(spec.label_map.values())
            else:
                # a catalog of engine models reads its own metadata per task
                out.update(self.describe_task(task, root))
        return out


class TSEcosystem(ModelEcosystem):
    """TotalSegmentator: composed tasks from the shipped registry JSON; weights
    by id from the release-asset manifest (license-gated models refuse with an
    actionable message). Always materialized - the composition data carries
    the label maps, and the remap drift test keeps them honest."""

    name = "ts"
    description = "TotalSegmentator task catalog"

    def __init__(self):
        self._catalog = TaskCatalog("ts")

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


class CustomEcosystem(ModelEcosystem):
    """The operator's own model folders: always materialized, nothing to install.
    The folder is read through from_model_folder, so the checkpoint's
    dataset.json is the spec here too."""

    name = "custom"
    description = "operator-registered local nnU-Net model folders"

    def __init__(self, models: dict | None = None):
        self._models = {str(k): Path(v) for k, v in (models or {}).items()}

    def tasks(self) -> list:
        return sorted(self._models)

    def materialized(self, task: str, root) -> bool:
        return task in self._models and self._models[task].is_dir()

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        if not self.materialized(task, root):
            raise ModelNotFound(f"custom task {task!r}: folder "
                                f"{self._models.get(task)} does not exist")
        if version is not None:
            from .weights_fetch import installed_version
            rec = installed_version(self._models[task]) or {}
            if rec.get("tag") != version:
                raise ModelNotFound(
                    f"{task}@{version}: custom folder records "
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


class EngineEcosystem(ModelEcosystem):
    """An ecosystem whose tasks are run by an engine's own network rather than an
    nnU-Net TaskSpec.

    That is the ONLY thing this class asserts. Whether the weights ship with the
    engine's image is a separate question, answered by
    :class:`ImageBakedEcosystem` below - an engine ecosystem is free to install
    per task instead (a catalog of models does), and conflating the two is what
    would force a many-task engine catalog to fight this class.
    """

    has_task_spec = False


class ImageBakedEcosystem(EngineEcosystem):
    """An engine ecosystem whose weights ship inside the engine's own image:
    always materialized, nothing to install. The one-model engines
    (FastSurfer, SynthStrip, VoxTell) are all this shape.

    Subclasses declare data only - name, engine, task_names, modality, structures.
    """

    #: Task names this engine offers.
    task_names: tuple = ()

    def tasks(self) -> list:
        return list(self.task_names)

    def materialized(self, task: str, root) -> bool:
        return True                      # weights ship with the engine's image

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        return None


class FastSurferEcosystem(ImageBakedEcosystem):
    """FastSurfer whole-brain parcellation (2.5D view-aggregation, not nnU-Net).
    Its checkpoints are baked into the FastSurfer worker image."""

    name = "fastsurfer"
    engine = "fastsurfer"
    description = "FastSurfer whole-brain parcellation (engine)"
    task_names = ("brain",)
    modality = "MR (T1)"

    @property
    def structures(self) -> list:
        """The real DKTatlas label names, from the engine's own LUT - so a client
        can enumerate them like any other task's."""
        from .engines.fastsurfer import load_lut
        return sorted(v["name"] for v in load_lut().values())


def engine_of(ecosystem) -> str:
    """The engine name an ecosystem declares. Ecosystems are duck-typed here (an
    object with ``name``/``tasks()``/``spec()`` is enough), so one that predates
    the engine layer - or a test stand-in - falls back to the default engine."""
    return getattr(ecosystem, "engine", _registry.NNUNETV2)


def registry(ecosystems=None) -> dict:
    """Normalize a list of ecosystems into ``{name: ecosystem}``. Duplicate
    ecosystem names are rejected; duplicate *task* names across ecosystems are
    fine - the canonical ``eco:task`` form disambiguates, and only the short
    form goes ambiguous."""
    out = {}
    for e in (default_ecosystems() if ecosystems is None else list(ecosystems)):
        if not e.name or ":" in e.name or e.name in out:
            raise ValueError(f"bad or duplicate ecosystem name {e.name!r}")
        # An unknown engine would route silently to the default worker at spawn;
        # catching the typo here keeps "which engine runs this?" answerable.
        if engine_of(e) not in _registry.ENGINES:
            raise ValueError(f"ecosystem {e.name!r} declares unknown engine "
                             f"{engine_of(e)!r}; known: {sorted(_registry.ENGINES)}")
        out[e.name] = e
    return out


class SynthStripEcosystem(ImageBakedEcosystem):
    """SynthStrip brain extraction (skull-strip): a contrast-agnostic learned
    brain-mask UNet, not nnU-Net. Weights are baked into the worker image."""

    name = "synthstrip"
    engine = "synthstrip"
    description = "SynthStrip brain extraction / skull-strip (engine)"
    task_names = ("mask",)
    modality = "MR (any contrast)"
    structures = ["Brain"]


class VoxTellEcosystem(ImageBakedEcosystem):
    """VoxTell free-text promptable segmentation. Unlike every other catalog entry,
    ``voxtell:text`` has **no fixed label set** - the prompts are an input, passed
    as ``options={"prompts": [...]}``, and they hash into the result-cache key. So
    ``structures`` is deliberately absent from info(): what it segments is whatever
    the caller asks for."""

    name = "voxtell"
    engine = "voxtell"
    description = ("VoxTell free-text promptable segmentation (engine); "
                   'prompts are an input: options={"prompts": ["liver", ...]}')
    task_names = ("text",)
    modality = "CT / MR / PET"


MONAI_MANIFEST = Path(__file__).parent / "data" / "monai_bundles.json"


#: Whose job co-registration is - published on every multi-input task so a client
#: reads it up front instead of discovering it from a refusal. A multi-channel
#: network consumes ONE tensor, so its channels must already share a grid, and
#: producing that is a registration step belonging upstream where the caller can
#: see and check it. Slicer registers before it ever calls us; doing it silently
#: inside an inference call would be a geometry decision taken on someone else's
#: behalf, which is the shape of every geometry bug this project has paid for.
ASSUMED_PREREGISTERED = {
    "mode": "assumed-preregistered", "owner": "caller",
    "note": "channels are stacked in the model's declared order and must already "
            "be co-registered on a common grid; nnseg does not register or "
            "resample them, and refuses inputs whose grids differ",
}


def _ordered_channel_def(channel_def) -> list:
    """A model's declared channel names, in channel order.

    Keys are strings in the JSON (``"0"``, ``"1"``, ...), so they are sorted
    numerically rather than lexically - a ten-channel model must not put
    ``"10"`` between ``"1"`` and ``"2"``. Returns ``[]`` when nothing usable is
    declared, which the caller reads as "this model did not name its inputs".
    """
    if not isinstance(channel_def, dict):
        return []
    try:
        items = sorted(channel_def.items(), key=lambda kv: int(kv[0]))
    except (TypeError, ValueError):
        return []
    return [str(v) for _, v in items]


class MonaiEcosystem(EngineEcosystem):
    """The MONAI model zoo: a CATALOG of bundles run by the ``monai`` engine.

    The first ecosystem in the "many tasks on a new engine" shape - MOOSE is many
    tasks on the *existing* nnU-Net engine, because its models are nnU-Net
    checkpoints, while a MONAI bundle brings its own network *and* its own
    transform chain. So this is an :class:`EngineEcosystem` (no nnU-Net TaskSpec)
    that nonetheless installs per task, which is exactly the pair of axes the
    base class keeps apart.

    **The bundle is the spec**: labels, modality and channel count are read from
    each installed bundle's own ``configs/metadata.json``, never from the
    manifest, which holds only download + listing facts (the rule that keeps this
    from repeating the stale total_mr class map). See medseg/docs/monai-bundles.md.
    """

    name = "monai"
    engine = "monai"
    description = "MONAI model zoo bundles (engine)"

    def __init__(self, manifest: Path | None = None):
        raw = json.loads(Path(manifest or MONAI_MANIFEST).read_text())
        self._bundles = raw.get("bundles", raw)

    def tasks(self) -> list:
        return sorted(self._bundles)

    def _entry(self, task: str) -> dict:
        try:
            return self._bundles[task]
        except KeyError:
            raise ModelNotFound(
                f"unknown monai bundle {task!r}; this build curates "
                f"{sorted(self._bundles)}") from None

    def _dir(self, task: str, root) -> Path:
        # version in the path: two versions of a bundle can coexist, and an
        # @version pin then resolves to its own directory rather than fighting.
        return Path(root) / "monai" / f"{task}_v{self._entry(task)['version']}"

    def materialized(self, task: str, root) -> bool:
        d = self._dir(task, root)
        return (d / "configs" / "metadata.json").is_file()

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        """Install the bundle through MONAI's own downloader.

        Deliberately not a zip fetch of the manifest's ``url``: the zoo has moved
        hosting, and its newest entries point at a Hugging Face *repo page* rather
        than a downloadable archive (which is also why they publish no checksum).
        ``monai.bundle.download`` is the one thing that knows all of
        monaihosting / huggingface_hub / github / ngc, so the manifest supplies
        the name and version and MONAI resolves where that actually lives.
        """
        entry = self._entry(task)
        if version is not None and version != entry["version"]:
            raise ModelNotFound(
                f"{task}@{version}: this build curates {task} v{entry['version']}; "
                "regenerate the manifest to serve another version")
        if self.materialized(task, root):
            return
        from monai.bundle import download          # worker-side only; not on the api image

        dest = self._dir(task, root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"downloading MONAI bundle {task} v{entry['version']}")
        # MONAI unpacks to <bundle_dir>/<name>; we keep versioned directories so two
        # versions can coexist, so download into a temp parent and move into place.
        staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".bundle-"))
        try:
            download(name=task, version=entry["version"], bundle_dir=str(staging),
                     source=entry.get("source") or "monaihosting", progress=False)
            unpacked = staging / task
            if not (unpacked / "configs" / "metadata.json").is_file():
                raise ModelNotFound(
                    f"{task}: the downloaded bundle has no configs/metadata.json under "
                    f"{unpacked} - the layout is not what this ecosystem expects")
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            os.replace(unpacked, dest)             # atomic: never a half-installed bundle
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def bundle_metadata(self, task: str, root) -> dict:
        """The installed bundle's own metadata.json (the spec)."""
        return json.loads((self._dir(task, root) / "configs" / "metadata.json").read_text())

    def bundle_root(self, task: str, root) -> Path:
        """Where the installed bundle lives - what the engine runs."""
        return self._dir(task, root)

    def weights_identity(self, task: str, root) -> list:
        """Per bundle+version, not one constant for the whole engine: two bundles
        must not collide on one cached result. Read from the manifest already in
        memory, so ``/v1/tasks`` stays cheap."""
        entry = self._entry(task)
        out = {"id": task, "version": entry["version"]}
        if entry.get("checksum"):        # the zoo omits it on recent releases
            out["sha1"] = entry["checksum"]
        return [out]

    def describe_task(self, task: str, root) -> dict:
        """Modality, structures, input roles and restore behavior, all read from
        the installed bundle."""
        from .schemas import input_specs, single_input

        fmt = (self.bundle_metadata(task, root).get("network_data_format") or {})
        inp = (fmt.get("inputs") or {}).get("image") or {}
        channel_def = ((fmt.get("outputs") or {}).get("pred") or {}).get("channel_def") or {}
        out = {"structures": [str(v) for k, v in sorted(channel_def.items(),
                                                        key=lambda kv: int(kv[0]))
                              if str(v).lower() != "background"]}
        modality = str(inp["modality"]) if inp.get("modality") else None
        if modality:
            out["modality"] = modality

        n_in = int(inp.get("num_channels") or 1)
        roles = _ordered_channel_def(inp.get("channel_def"))
        if n_in == 1:
            out["inputs"] = single_input(modality)
        elif len(roles) == n_in:
            out["inputs"] = input_specs(roles, modality=modality)
            out["channel_names"] = roles
        else:
            # The bundle claims N channels and names fewer - renalStructures_CECT
            # declares 3 and names one ("image"). There is no honest way to bind
            # the rest: position is exactly what cannot be trusted here, since
            # MONAI's BraTS bundle orders T1c first where nnU-Net's own BraTS
            # convention puts FLAIR there. So: refuse, and say why.
            out["inputs"] = None
            out["inputs_incomplete"] = {
                "channels": n_in, "named": roles,
                "reason": f"the bundle declares {n_in} input channels but names "
                          f"{len(roles)}; nnseg will not bind inputs by position"}
            out["channel_names"] = roles or [f"channel_{i}" for i in range(n_in)]
        out["behavior"] = {"restore": self._restore_fact(task, root)}
        if out.get("inputs") and len(out["inputs"]) > 1:
            out["behavior"]["alignment"] = dict(ASSUMED_PREREGISTERED)
        return out

    def _restore_fact(self, task: str, root) -> dict:
        """How this bundle brings its prediction back to the input grid.

        A *fact*, not a knob: the bundle's own postprocessing decides, and the
        two orders in the wild give materially different boundaries.
        ``spleen_ct_segmentation`` inverts its spacing transform BEFORE argmax,
        so probabilities are resampled and the boundary is graded;
        ``wholeBody_ct_segmentation`` argmaxes first and inverts a labelmap with
        ``nearest_interp``, which snaps every boundary to the model's grid. We do
        not override either - running the bundle's own config is the whole point
        of this engine - so the least we can do is let a client see which one it
        is getting before choosing a model.
        """
        try:
            from .engines.monai_bundle import inference_config
            cfg = inference_config(self.bundle_root(task, root))
            if cfg.suffix != ".json":
                # YAML would need a parser the lean API image does not carry
                raise ValueError(f"{cfg.suffix} config")
            post = json.loads(cfg.read_text()).get("postprocessing") or {}
            transforms = post.get("transforms") if isinstance(post, dict) else None
            order = [str((t or {}).get("_target_", "")) for t in (transforms or [])
                     if isinstance(t, dict)]
            invert = next((i for i, t in enumerate(order) if t.endswith("Invertd")), None)
            argmax = next((i for i, t in enumerate(order) if t.endswith("AsDiscreted")), None)
        except Exception as e:                      # unreadable/absent/YAML: say so
            return {"mode": "unknown", "owner": "bundle", "note": f"not determined ({e})"}
        if invert is None:
            # nothing inverts, so the prediction stays on the model's grid and
            # nnseg's own nearest-neighbour resample is what the caller gets
            return {"mode": "label-nearest", "owner": "nnseg",
                    "note": "the bundle does not invert its spacing transform; "
                            "nnseg resamples the labelmap to the input grid"}
        if argmax is not None and argmax < invert:
            return {"mode": "label-nearest", "owner": "bundle",
                    "note": "this bundle argmaxes before inverting its spacing "
                            "transform, so boundaries are snapped to the model's grid"}
        return {"mode": "graded", "owner": "bundle",
                "note": "this bundle inverts its spacing transform before argmax, "
                        "so class probabilities are resampled and argued after"}

    def info(self, task: str, root) -> dict:
        out = super().info(task, root)
        entry = self._entry(task)
        # listing facts, so an uninstalled task still says something useful
        out.setdefault("modality", entry.get("modality"))
        out["bundle_version"] = entry["version"]
        if entry.get("task"):
            out["summary"] = entry["task"]
        if not out.get("materialized"):
            out["n_structures"] = max(int(entry.get("n_labels", 1)) - 1, 0)
            out.update(self._preinstall_inputs(entry, out.get("modality")))
        return out

    def _preinstall_inputs(self, entry: dict, modality) -> dict:
        """What can honestly be said about a bundle's inputs before it is
        installed.

        The manifest records the channel *count* (generator-derived from the
        bundle's own metadata), which is enough for the single-input case that
        covers most of the zoo. It does not always carry the channel *names*, and
        a multi-channel task without names cannot be described as one image
        without lying about it - so that case reports no inputs and says where
        the names come from.
        """
        from .schemas import input_specs, single_input

        n_in = int(entry.get("in_channels") or 1)
        roles = _ordered_channel_def(entry.get("channel_def"))
        if len(roles) == n_in:
            out = {"inputs": input_specs(roles, modality=modality)}
            if n_in > 1:
                # the caller has to know this BEFORE assembling a request, not
                # after being refused for it
                out["behavior"] = {"alignment": dict(ASSUMED_PREREGISTERED)}
            return out
        if n_in <= 1:
            return {"inputs": single_input(modality)}
        return {"inputs": None,
                "inputs_hint": f"this bundle takes {n_in} input channels; their "
                               "names are read from the bundle once installed"}


#: The ecosystems each engine contributes when its engine is enabled. The
#: nnU-Net catalogs are always present; engine catalogs appear only where their
#: engine does, so the catalog can never list a task no worker can run.
_ENGINE_ECOSYSTEMS = {"fastsurfer": FastSurferEcosystem,
                      "synthstrip": SynthStripEcosystem,
                      "voxtell": VoxTellEcosystem,
                      "monai": MonaiEcosystem}


def default_ecosystems() -> list:
    """The catalogs this deployment serves: the nnU-Net ones, plus one per
    enabled engine. Enablement is the registry's answer (read from the
    environment per call), so the catalog and the workers cannot disagree."""
    ecos = [TSEcosystem(), MooseEcosystem()]
    ecos += [cls() for engine, cls in _ENGINE_ECOSYSTEMS.items()
             if _registry.enabled(engine)]
    return ecos


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

    def engine_of(self, name: str) -> str | None:
        """Which engine runs ``name`` - the catalog-side answer to the question
        the Modal dispatcher answers from the task's grammar. ``None`` if the
        name does not resolve."""
        eco = self.ecosystem_of(name)
        return None if eco is None else engine_of(eco)

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
