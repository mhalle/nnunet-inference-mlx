"""The engine registry: which runtimes exist, and what each one is.

**Ecosystem vs Engine.** An *ecosystem* is what the user selects - a catalog of
tasks with the ``eco:task@version`` grammar (``ts``, ``moose``, ``custom``,
``fastsurfer``, ``synthstrip``). An *engine* is the runtime that actually runs a
task: its container image, its compute, its weights identity. **Many ecosystems
map to one engine** - ``ts``, ``moose`` and ``custom`` are three catalogs of
nnU-Net models, all run by the ``nnunetv2`` engine.

This module is the single source of truth for that mapping and for every fact
that used to be spelled once per engine per call site: the enable flag, the
weights identity that keys the result cache, and the ecosystem -> engine route.
Adding an engine is one row here plus a worker class in
:mod:`nnseg.modal_app` that declares its image and compute.

Deliberately a **static registry, not a plugin framework** - engines are a
closed set we ship and test together (YAGNI). It is also deliberately **light**:
no torch, no SimpleITK, not even an import of the engine modules. ``importing
nnseg`` must stay torch-free (see ``docs/dependency-discipline.md``), and
``info()`` on the lean API image reads these constants, so the version literals
live *here* and the engine modules re-export them. The one dependency it does
take is pydantic, for the parameter schemas below - 27 ms to import and 7.8 MB
on disk, against a core that already requires SimpleITK's 183 MB, and it buys
the published schema, the enforced validation and the OpenAPI document from one
declaration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from ..schemas import GRADED_RESTORE, NoParams, Params, VoxTellParams

# The nnU-Net runtime. Named for the upstream major version it runs: nnU-Net v1
# has a different checkpoint layout and would need its own loader, hence its own
# engine, rather than a flag on this one.
NNUNETV2 = "nnunetv2"


def _truthy(value: str | None) -> bool:
    return (value or "0") not in ("0", "false", "no", "")


@dataclass(frozen=True)
class Engine:
    """One runtime. ``weights_identity`` is the engine's contribution to the
    result-cache key.

    It is a **constant** (or ``None``), never a function of the task, for two
    reasons. Engines bake their weights into their image, so there is nothing
    per-task to look up. And ``info()`` is called once per task by ``/v1/tasks``
    and ``/v1/version`` - 70+ tasks on a default catalog - so a per-task lookup
    here would put a filesystem walk over a mounted weights Volume in the path of
    two hot endpoints on the lean API container. The nnU-Net engine's weights are
    per-task and install-time, so its identity stays ``None`` here and is computed
    by :meth:`nnseg.segmenter.Segmenter.describe` from the spec's weights ids.
    """

    name: str
    enabled_env: str | None = None          # None = always on (the default engine)
    weights_identity: Callable[[], list[dict]] | None = None
    # The engine's compute entry point, for runtimes that can run in-process.
    # None today for all three: engine compute is reached through the Modal
    # worker's _compute hook. The field exists so local dispatch is a function
    # body, not a new seam - see EcosystemCatalog.engine_of.
    compute: Callable | None = None
    #: Whether a stored result may be SERVED for a repeat request (RFC 9111
    #: `no-cache` when False - results are still stored, so artifacts, the job's
    #: `key` and its `links` all keep working; `no-store` is a different thing we
    #: do not offer). False suits an engine whose request space is effectively
    #: unbounded, where a lookup almost never hits and the entries only push
    #: useful ones out of a shared LRU - VoxTell, whose free-text prompts hash
    #: into the key, is the case this exists for.
    serve_from_cache: bool = True
    #: The engine's own parameters, as a pydantic model. Generates the JSON
    #: Schema ``describe()`` publishes AND validates the request at submit, from
    #: one declaration. Our processing knobs are a separate group and are not
    #: listed here - see :func:`nnseg.schemas.parameter_groups`.
    parameters: type[Params] = NoParams
    #: Whether this engine can consume more than one input image. Introspection
    #: precedes capability on purpose: a multi-channel model may *declare* its
    #: roles (so a client can see what it would need) while the engine that runs
    #: it still refuses to be handed them.
    multi_input: bool = False
    #: Read-only facts about what this engine does to a result - published so a
    #: client can see behavior it cannot change. Empty when the answer is
    #: per-task rather than per-engine (MONAI, where each bundle's own
    #: postprocessing decides).
    behavior: dict = field(default_factory=dict)
    #: Whether nnseg's processing knobs (``grid``, ``interp``, ...) apply. False
    #: for engines that run someone else's chain end to end, where offering them
    #: would be advertising a knob we do not turn.
    processing_knobs: bool = True
    description: str = ""


def _fastsurfer_identity() -> list[dict]:
    return [{"id": "fastsurfer", "version": "vinn-v2"}]


def _synthstrip_identity() -> list[dict]:
    return [{"id": "synthstrip", "version": "v1"}]


def _voxtell_identity() -> list[dict]:
    return [{"id": "voxtell", "version": "v1.1"}]


# No _monai_identity: the MONAI engine runs a CATALOG of bundles, so the identity
# is per bundle+version and comes from MonaiEcosystem.weights_identity, not from a
# constant here. That is what Engine.weights_identity=None means.


ENGINES: dict[str, Engine] = {
    NNUNETV2: Engine(
        name=NNUNETV2,
        # the only engine whose surrounding pipeline is ours, so the only one
        # where our processing knobs are real
        behavior=GRADED_RESTORE,
        description="nnU-Net v2 networks (TotalSegmentator, MOOSE, and stock models)",
    ),
    "fastsurfer": Engine(
        name="fastsurfer",
        enabled_env="NNSEG_FASTSURFER",
        weights_identity=_fastsurfer_identity,
        behavior=GRADED_RESTORE,
        processing_knobs=False,
        description="FastSurferVINN 2.5D view-aggregation parcellation",
    ),
    "synthstrip": Engine(
        name="synthstrip",
        enabled_env="NNSEG_SYNTHSTRIP",
        weights_identity=_synthstrip_identity,
        behavior=GRADED_RESTORE,
        processing_knobs=False,
        description="SynthStrip brain extraction (signed distance transform)",
    ),
    "voxtell": Engine(
        name="voxtell",
        enabled_env="NNSEG_VOXTELL",
        weights_identity=_voxtell_identity,
        # free text means an unbounded key space and interactive, low-reuse
        # requests: memoizing them evicts results that do get re-read
        serve_from_cache=False,
        parameters=VoxTellParams,
        processing_knobs=False,
        behavior={"restore": {
            "mode": "none", "owner": "engine",
            "note": "masks come back on the reader's grid; nnseg does not resample"}},
        description="VoxTell free-text promptable segmentation (prompts are input)",
    ),
    "monai": Engine(
        name="monai",
        enabled_env="NNSEG_MONAI",
        weights_identity=None,          # per bundle - see MonaiEcosystem
        processing_knobs=False,
        # no `behavior` constant: each bundle's own postprocessing decides
        # whether it inverts probabilities or a labelmap, so the fact is per task
        # and read from the installed bundle - see MonaiEcosystem.describe_task
        description="MONAI model zoo bundles (each carries its own transforms)",
    ),
}

# Which engine runs each ecosystem's tasks. Ecosystems not listed here run on the
# default engine, so the nnU-Net catalogs need no entry.
ECOSYSTEM_ENGINE: dict[str, str] = {
    "fastsurfer": "fastsurfer",
    "synthstrip": "synthstrip",
    "voxtell": "voxtell",
    "monai": "monai",
}


def engine_for(ecosystem: str) -> Engine:
    """The engine that runs ``ecosystem``'s tasks (the default engine if the
    ecosystem declares none)."""
    return ENGINES[ECOSYSTEM_ENGINE.get(ecosystem, NNUNETV2)]


def engine_for_task(task: str) -> Engine:
    """The engine for a canonical ``eco:task`` name. Routes on the *grammar*
    rather than on hardcoded task prefixes, and falls back to the default engine
    for a bare name (every wire form is canonicalized before it reaches here)."""
    return engine_for(str(task).partition(":")[0])


def enabled(name: str) -> bool:
    """Whether this deployment enables ``name``. Read from the environment on
    every call (never cached) so a test can ``monkeypatch.setenv`` it; callers
    that must decide at import time - Modal resolves decorators then - snapshot
    the result themselves."""
    eng = ENGINES[name]
    return eng.enabled_env is None or _truthy(os.environ.get(eng.enabled_env))


def enabled_engines() -> list[str]:
    """Names of the engines this deployment can actually run."""
    return [n for n in ENGINES if enabled(n)]


def engine_env_vars() -> tuple[str, ...]:
    """Every engine enable flag, for forwarding into a container's environment.
    Derived, so a new engine cannot be forgotten here (a knob that exists at
    deploy time but not in the container is a bug this project has already hit)."""
    return tuple(e.enabled_env for e in ENGINES.values() if e.enabled_env)
