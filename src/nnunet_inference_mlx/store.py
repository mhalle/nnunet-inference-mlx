"""ModelStore — an explicit, owned, read-through store of models.

A ``ModelStore`` resolves a model *id* (integer dataset id for nnU-Net/TS,
string folder name for MOOSE) to a model, fetching and building lazily and
keeping what it makes. It is the explicit, caller-owned replacement for the
old module-global engine cache: nothing is process-global, behavior never
changes silently on RAM/env detection, and every piece of state is
inspectable and freeable.

Two layers, one object (a read-through stack):

* **disk** (downloaded / cold) — model files under ``model_root_dir``.
  Verbs: ``download`` / ``delete_downloads`` / ``downloaded``.
* **memory** (loaded / hot) — built ``LoadedModel`` s, bounded by
  ``max_memory_mb`` (LRU-evicted to fit). Verbs: ``load`` / ``unload`` / ``loaded``.

``get(id)`` returns cold :class:`ModelData` (config + weights, no GPU).
``load(id)`` returns a hot ``LoadedModel``, building on miss and caching it.
The whole vocabulary is models + readiness; nothing user-facing is an "engine".

The two transforms — read (folder → ModelData) and build (ModelData →
LoadedModel) — are injectable, so the store's mechanics are testable without
real weights or a GPU.
"""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Hashable, Iterable, Sequence

from .model_data import ModelData
from .values import BuildOptions


# Visible default memory budget for resident models (MB). NOT RAM-detected —
# an explicit constant you can see and override.
DEFAULT_MAX_MEMORY_MB = 4000


# ---------------------------------------------------------------------------
# Ecosystem conventions: id → folder, default locations
# ---------------------------------------------------------------------------


def _nnunet_resolve(root: Path, id) -> Path:
    matches = sorted(root.glob(f"Dataset{id}_*"))
    if not matches:
        raise FileNotFoundError(f"no Dataset{id}_* under {root}")
    return _config_subfolder(matches[0])


def _moose_resolve(root: Path, id) -> Path:
    model_dir = root / str(id)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"MOOSE model folder not found: {model_dir}")
    return _config_subfolder(model_dir)


def _config_subfolder(model_dir: Path) -> Path:
    """The inner ``{trainer}__{plans}__{config}`` folder (two '__')."""
    configs = sorted(
        p for p in model_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name.count("__") == 2
    )
    if not configs:
        raise FileNotFoundError(
            f"no nnU-Net config folder (trainer__plans__config) in {model_dir}"
        )
    return configs[0]


def _nnunet_model_dir(root: Path, id) -> Path:
    """The per-model directory (parent of the config folder) — the unit a
    download writes and a delete removes."""
    matches = sorted(root.glob(f"Dataset{id}_*"))
    if not matches:
        raise FileNotFoundError(f"no Dataset{id}_* under {root}")
    return matches[0]


def _nnunet_downloaded(root: Path) -> list[int]:
    ids = []
    for p in sorted(root.glob("Dataset*_*")):
        if p.is_dir():
            stem = p.name[len("Dataset"):].split("_", 1)[0]
            if stem.isdigit():
                ids.append(int(stem))
    return ids


def _moose_downloaded(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


# ecosystem → (id→config-folder, id→model-dir, list-downloaded, env vars, default path)
_ECOSYSTEMS = {
    "nnunet": (_nnunet_resolve, _nnunet_model_dir, _nnunet_downloaded,
               ("nnUNet_results",), None),
    "totalsegmentator": (_nnunet_resolve, _nnunet_model_dir, _nnunet_downloaded,
                         ("TOTALSEG_WEIGHTS_PATH",),
                         Path("~/.totalsegmentator/nnunet/results")),
    "moose": (_moose_resolve, lambda root, id: root / str(id), _moose_downloaded,
              ("NNUNET_MLX_MOOSE_MODELS", "MOOSE_MODELS"), None),
}


def _resolve_model_root_dir(ecosystem: str, explicit) -> Path | None:
    """Precedence: explicit arg → env var(s) → built-in default. Visible."""
    if explicit is not None:
        return Path(explicit).expanduser()
    if ecosystem not in _ECOSYSTEMS:
        return None
    _, _, _, env_vars, default = _ECOSYSTEMS[ecosystem]
    for var in env_vars:
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser()
    return default.expanduser() if default is not None else None


# ---------------------------------------------------------------------------
# Default transforms (read folder → ModelData; build ModelData → LoadedModel)
# ---------------------------------------------------------------------------


def _default_read(folder, *, folds="all", dtype=None, ecosystem="local", id=None) -> ModelData:
    return ModelData.read_folder(folder, folds=folds, dtype=dtype,
                                 ecosystem=ecosystem, id=id)


def _default_build(model_data: ModelData, options: BuildOptions):
    from .build import build_model  # phase 3
    return build_model(model_data, options)


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------


class ModelStore:
    """An explicit, owned store of models for one ecosystem.

    Parameters
    ----------
    ecosystem :
        ``"nnunet"`` / ``"totalsegmentator"`` / ``"moose"``. Determines the
        id→folder convention and the default location.
    model_root_dir :
        Local root holding the model folders. ``None`` → resolve from the
        ecosystem's env var(s) then built-in default (precedence is explicit).
    max_memory_mb :
        Budget for resident (loaded) models; LRU-evicted to fit. Visible
        default, never RAM-detected.
    options :
        Default :class:`BuildOptions` (the cache key: configuration, folds,
        batch_size, compile, dtype). Run knobs (step_size, mirroring) are
        per-call, not part of build identity.
    read, build :
        Injectable transforms (folder→ModelData, ModelData→LoadedModel).
        Default to the real ones; tests pass fakes.
    """

    def __init__(
        self,
        ecosystem: str,
        *,
        model_root_dir: str | os.PathLike | None = None,
        max_memory_mb: float = DEFAULT_MAX_MEMORY_MB,
        options: BuildOptions = BuildOptions(),
        read: Callable | None = None,
        build: Callable | None = None,
    ):
        if ecosystem not in _ECOSYSTEMS:
            raise ValueError(
                f"unknown ecosystem {ecosystem!r}; known: {sorted(_ECOSYSTEMS)}"
            )
        self.ecosystem = ecosystem
        self.model_root_dir = _resolve_model_root_dir(ecosystem, model_root_dir)
        self.max_memory_mb = float(max_memory_mb)
        self.options = options
        self._read = read or _default_read
        self._build = build or _default_build
        resolve, model_dir, downloaded, _, _ = _ECOSYSTEMS[ecosystem]
        self._resolve_folder = resolve
        self._model_dir = model_dir
        self._list_downloaded = downloaded
        self._loaded: "OrderedDict[Hashable, object]" = OrderedDict()

    # ----- root -----
    def _require_root(self) -> Path:
        if self.model_root_dir is None:
            raise FileNotFoundError(
                f"model_root_dir for ecosystem {self.ecosystem!r} is unknown. "
                f"Pass model_root_dir=, set the ecosystem env var, or rely on its "
                f"default."
            )
        return self.model_root_dir

    # ----- cold (disk) layer -----
    def get(self, id, *, folds=None, dtype=None) -> ModelData:
        """Read a cold :class:`ModelData` for ``id`` (no GPU). Reads the
        folder fresh each call; the data is not retained."""
        folder = self._resolve_folder(self._require_root(), id)
        return self._read(folder, folds=folds if folds is not None else "all",
                          dtype=dtype, ecosystem=self.ecosystem, id=id)

    def downloaded(self) -> list:
        """Ids whose model files are present on disk."""
        root = self.model_root_dir
        if root is None or not root.is_dir():
            return []
        return self._list_downloaded(root)

    def download(self, ids, *, build: bool = False) -> None:
        """Ensure ``ids`` are present locally (and built if ``build=True``).

        Remote fetch is not yet wired — for now this asserts local presence
        and raises with an actionable message if a model is missing. ``build``
        additionally loads each into memory.
        """
        present = set(self.downloaded())
        missing = [i for i in _as_list(ids) if i not in present]
        if missing:
            raise FileNotFoundError(
                f"models {missing} not present under {self.model_root_dir} and "
                f"remote download is not yet wired; place them locally."
            )
        if build:
            self.load(ids)

    def delete_downloads(self, ids) -> None:
        """Delete the on-disk files for ``ids`` (destructive; re-fetch needed)."""
        root = self._require_root()
        for i in _as_list(ids):
            self.unload(i)  # drop any loaded copy first
            try:
                shutil.rmtree(self._model_dir(root, i))
            except FileNotFoundError:
                pass

    # ----- hot (memory) layer -----
    def load(self, ids, *, options: BuildOptions | None = None):
        """Build (or reuse) the loaded model(s) for ``ids`` and keep them resident.

        Single id → one :class:`~nnunet_inference_mlx.build.LoadedModel`; an
        iterable → a list. LRU-evicts others to fit ``max_memory_mb``.
        """
        if _is_single(ids):
            return self._load_one(ids, options)
        return [self._load_one(i, options) for i in ids]

    def _load_one(self, id, options):
        opts = options or self.options
        key = (id, opts)
        if key in self._loaded:
            self._loaded.move_to_end(key)
            return self._loaded[key]
        model_data = self.get(id, folds=opts.folds if opts.folds != "all" else None)
        loaded = self._build(model_data, opts)
        self._loaded[key] = loaded
        self._evict_to_fit()
        return loaded

    def _evict_to_fit(self) -> None:
        # Evict oldest first; never evict the most-recently-added (it's at the
        # end). A single model larger than the budget degrades to no-reuse.
        while len(self._loaded) > 1 and self.loaded_mb > self.max_memory_mb:
            _, victim = self._loaded.popitem(last=False)
            _close(victim)

    def loaded(self) -> list[tuple]:
        """``[(id, memory_mb), ...]`` for resident loaded models (LRU order)."""
        return [(key[0], _mb(m)) for key, m in self._loaded.items()]

    @property
    def loaded_mb(self) -> float:
        return sum(_mb(m) for m in self._loaded.values())

    def unload(self, ids) -> None:
        """Free the memory held by ``ids`` (download kept). No-op if absent."""
        wanted = set(_as_list(ids))
        for key in [k for k in self._loaded if k[0] in wanted]:
            _close(self._loaded.pop(key))

    def unload_all(self) -> None:
        for m in self._loaded.values():
            _close(m)
        self._loaded.clear()

    # ----- lifecycle / inspection -----
    def __enter__(self) -> "ModelStore":
        return self

    def __exit__(self, *exc) -> None:
        self.unload_all()

    def __len__(self) -> int:
        return len(self._loaded)

    def __repr__(self) -> str:
        return (
            f"ModelStore(ecosystem={self.ecosystem!r}, "
            f"model_root_dir={str(self.model_root_dir)!r}, "
            f"max_memory_mb={self.max_memory_mb}, loaded={len(self._loaded)})"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_single(ids) -> bool:
    return isinstance(ids, (int, str))


def _as_list(ids) -> list:
    return [ids] if _is_single(ids) else list(ids)


def _mb(model) -> float:
    return float(getattr(model, "memory_mb", 0.0))


def _close(model) -> None:
    close = getattr(model, "close", None)
    if callable(close):
        close()


__all__ = ["ModelStore", "DEFAULT_MAX_MEMORY_MB"]
