"""A configured segmenter: execution policy set once, models kept warm between calls.

``nnseg.segment(image, task)`` is the right shape for a script - everything decided per call,
nothing retained. A Slicer module, an MCP tool or a REST endpoint wants the opposite: the
device, weights root and dtype are properties of the *deployment*, fixed at startup, and
reloading a checkpoint on every request is the difference between a service and a toy.

:class:`Segmenter` holds that policy plus a :class:`~nnseg.cache.ModelCache`, and exposes the
same operation as a method. Per-call arguments still win, so a caller can override anything for
one request without building a second Segmenter.
"""
from __future__ import annotations

from pathlib import Path

from .cache import ModelCache
from .errors import NnsegError
from .tasks import TaskCatalog
from .weights import as_store

# Set once per deployment, and overridable per call.
POLICY = ("device", "dtype", "weights", "folds", "accumulate", "batch_size", "allow_transpose",
          "resampling_order", "envelope_mm", "convention", "interp", "grid", "configuration")


class Segmenter:
    """Segment with a fixed execution policy and warm models.

    >>> seg = Segmenter(device="cuda", weights="/weights", cache_models=2)
    >>> r = seg.segment("scan.nii.gz", "total_fast")
    >>> r.mask("liver").sum()

    ``weights`` says where model files come from - a :class:`~nnseg.weights.WeightsStore`, a
    directory, or ``None`` for the ecosystem default. ``cache_models`` is how many models stay
    resident. Keep it small: warm weights compete for
    the same device memory the sliding-window accumulator needs (see :mod:`nnseg.cache`).
    """

    def __init__(self, *, device: str = "auto", dtype: str = "fp16", weights=None,
                 catalog=None, folds=(0,), accumulate: str = "auto", batch_size="auto",
                 resampling_order: int = 3, envelope_mm: float | None = 20.0,
                 convention: str = "auto", interp: str = "linear", grid="input",
                 configuration: str | None = None, cache_models: int = 1,
                 allow_transpose: bool = False):
        self.models = ModelCache(capacity=cache_models)
        self.weights = as_store(weights)
        if catalog is None:
            from .ecosystems import EcosystemCatalog
            catalog = EcosystemCatalog(root=self.weights.root)
        self.catalog = catalog
        self.policy = dict(device=device, dtype=dtype, weights=self.weights, folds=folds,
                           accumulate=accumulate, batch_size=batch_size,
                           resampling_order=resampling_order, envelope_mm=envelope_mm,
                           convention=convention, interp=interp, grid=grid,
                           configuration=configuration, allow_transpose=allow_transpose)

    def resolve_task(self, task) -> str:
        """The canonical (ecosystem-qualified, unversioned) name for any
        accepted form - short, eco:name, or eco:name@version."""
        if hasattr(self.catalog, "resolve"):
            return self.catalog.resolve(task)[2]
        return str(task)

    def prepare(self, task, *, progress=None) -> dict:
        """Install the task's weights now (idempotent) and return describe().

        The deliberate form of what first use does implicitly - a server or a
        UI calls this to warm a task before the data arrives."""
        if hasattr(self.catalog, "prepare"):
            self.catalog.prepare(task, progress=progress)
        else:
            self.weights.ensure(task, catalog=self.catalog)
        return self.describe(task)

    # -- the operation ------------------------------------------------------
    def segment(self, image, task, **overrides):
        """Segment ``image`` with ``task``; any policy argument may be overridden for this call."""
        from .pipeline import segment
        unknown = set(overrides) - set(POLICY) - {"progress", "outside", "cancel"}
        if unknown:
            raise TypeError(f"unknown argument(s) {sorted(unknown)}; policy is {sorted(POLICY)}")
        kw = {**self.policy, **overrides}
        return segment(image, task, catalog=self.catalog, models=self.models, **kw)

    __call__ = segment

    def submit(self, image, task, *, on_progress=None, **overrides):
        """Start a segmentation on a worker thread and return a :class:`~nnseg.job.Job`.

        The caller's loop stays free: poll ``job.progress`` from a UI timer, ``job.cancel()`` on
        a Cancel button, ``job.result()`` when done. Runs on one device at a time - a second job
        reports stage ``"queued"`` until the first releases the device.
        """
        from .job import Job
        unknown = set(overrides) - set(POLICY)
        if unknown:
            raise TypeError(f"unknown argument(s) {sorted(unknown)}; policy is {sorted(POLICY)}")
        kw = {**self.policy, **overrides}

        def run(reporter):
            from .pipeline import segment
            return segment(image, task, catalog=self.catalog, models=self.models,
                           cancel=reporter.cancel, progress=reporter, **kw)

        return Job(run, device=kw["device"], on_progress=on_progress,
                   name=getattr(task, "name", str(task)))

    # -- introspection: what can this thing do, before it does it -----------
    def tasks(self) -> list[str]:
        """Every task name in the catalog."""
        return self.catalog.names()

    def _introspection(self, d: dict) -> dict:
        """Add the block every task carries, whatever engine runs it: which
        images it takes (``inputs``), what a caller may send (``parameters``),
        and what the engine does that a caller cannot change (``behavior``).

        Uniform on purpose. A client should not have to know which engine is
        behind a task in order to build a valid request for it - that mapping is
        this tier's job, done once, rather than every client's job done N times.

        Never overwrites what an ecosystem already answered from the model's own
        metadata: ``inputs: None`` from a bundle that declared its channels
        incompletely means "not bindable", and must not be quietly replaced by a
        plausible default.
        """
        from .engines.registry import ENGINES, NNUNETV2
        from .schemas import declared_inputs, parameter_groups
        eng = ENGINES.get(d.get("engine") or NNUNETV2) or ENGINES[NNUNETV2]
        d.setdefault("parameters", parameter_groups(eng.parameters,
                                                    processing=eng.processing_knobs))
        if eng.behavior:
            d.setdefault("behavior", dict(eng.behavior))
        d["inputs"] = declared_inputs(d)
        return d

    def describe(self, task) -> dict:
        """What a task is and what it needs, without running or downloading anything.

        Beyond the spec itself: the policy knobs a caller can override per job
        (``folds_default``, ``configuration``), what is actually installed under this
        store's root - with the version/sha the install sidecar recorded, never a
        guess from the manifest - and ``channel_names`` when an installed model's
        ``dataset.json`` is there to read. All best-effort: an unconfigured weights
        root reports ``installed: false`` rather than raising.
        """
        from .tasks import _resolve_spec           # torch-free: describe must not need torch
        info = None
        if isinstance(task, str) and hasattr(self.catalog, "info"):
            try:
                info = self.catalog.info(task)
            except LookupError:
                info = None
            if info is not None and not info.get("materialized", True):
                # weights not installed yet: report what is knowable without
                # downloading anything, and how to materialize the rest
                return self._introspection(
                    {**info, "folds_default": list(self.policy["folds"]),
                     "hint": "structures are read from the checkpoint once "
                             "installed; prepare() or first use installs it"})
            if info is not None and not info.get("task_spec", True):
                # This ecosystem's tasks have no nnU-Net TaskSpec (an engine runs
                # its own network), so _resolve_spec would raise; its describe IS
                # its info - and it carries weights_installed, which the
                # result-cache key needs. Branch on `task_spec`, never on the
                # `engine` key: every ecosystem names an engine, so a truthy test
                # on `engine` would take this path for every task and silently
                # drop structures/weights from describe - which moves every
                # result-cache key and re-splits the API and worker keys.
                return self._introspection(
                    {**info, "folds_default": list(self.policy["folds"])})
        spec = _resolve_spec(task, self.catalog)
        from .engines.registry import NNUNETV2
        d = {"name": spec.name, "lineage": spec.lineage, "modality": spec.modality,
             # Which runtime would run this. A TaskSpec or a model folder resolves
             # to the default engine by definition; a catalog task reports what its
             # ecosystem declared. Reported for every task so a client never has to
             # infer it from the name.
             "engine": (info or {}).get("engine", NNUNETV2),
             "shape": spec.shape, "n_structures": len(spec.label_map),
             "structures": [spec.label_map[k] for k in sorted(spec.label_map)],
             "weights": [str(w) for w in spec.weights_ids],
             "folds_default": list(self.policy["folds"]),
             "configuration": self.policy["configuration"]}
        installed, channels = [], None
        for wid in spec.weights_ids:
            entry = {"id": str(wid), "installed": False}
            try:
                if self.weights.have(wid):
                    folder = self.weights.resolve(wid, configuration=self.policy["configuration"])
                    entry["installed"] = True
                    from .weights_fetch import installed_version
                    side = installed_version(folder)
                    if side:
                        entry["version"] = side.get("tag")
                        entry["sha256"] = side.get("sha256")
                    if channels is None:
                        import json
                        ds = Path(folder) / "dataset.json"
                        if ds.exists():
                            j = json.loads(ds.read_text())
                            channels = j.get("channel_names") or j.get("modality")
            except NnsegError:
                pass                      # no root configured / unresolvable: stays not-installed
            installed.append(entry)
        d["weights_installed"] = installed
        d["channel_names"] = channels
        return self._introspection(d)

    def structures(self, task) -> list[str]:
        """The structure names a task produces, in label order."""
        return self.describe(task)["structures"]

    # -- warm models --------------------------------------------------------
    def warm(self, task) -> int:
        """Load this task's models now, so the first real call does not pay for it.

        Returns how many are resident afterwards. Only useful with ``cache_models >= 1``.
        """
        from .tasks import _resolve_spec, _uses_nnunet_preprocessing
        spec = _resolve_spec(task, self.catalog)
        store = as_store(self.policy["weights"],
                         layout="nnunetv2" if _uses_nnunet_preprocessing(spec) else "ts")
        for wid in spec.weights_ids:
            folder = store.resolve(wid, configuration=self.policy["configuration"])
            self.models.get(folder, folds=self.policy["folds"], device=self.policy["device"],
                            dtype=self.policy["dtype"], accumulate=self.policy["accumulate"],
                            batch_size=self.policy["batch_size"])
        return len(self.models)

    def fetch(self, task) -> int:
        """Download whatever this task needs, without running anything. Returns model count."""
        return len(self.weights.ensure(task, catalog=self.catalog))

    def clear(self) -> None:
        """Drop every warm model and free the device memory."""
        self.models.clear()

    def __repr__(self) -> str:
        return (f"Segmenter(device={self.policy['device']!r}, dtype={self.policy['dtype']!r}, "
                f"{len(self.catalog)} tasks, {self.weights!r}, {self.models!r})")
