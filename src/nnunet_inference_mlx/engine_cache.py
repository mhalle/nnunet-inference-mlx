"""Process-wide InferenceEngine cache.

Building an :class:`~nnunet_inference_mlx.engine.InferenceEngine` from a model
folder pays a one-time ~2–3 s cost: deserialize the bundle, build the MLX
network, compile, run a warmup forward. For workflows that touch the same
model many times — batch inference over a folder, a multi-stage cascade that
re-enters the same model, an interactive UI that segments after each click —
keeping the engine alive across calls eliminates that cost.

This module provides a single process-wide cache keyed on whatever inputs
materially affect engine state (model folder, configuration, folds, step
size, batch size, mirroring, compile setting). Look-ups return ``None`` when
caching is disabled; insertions are no-ops in the same case, so call sites
can use the same code path with or without caching enabled.

Auto-tiering
------------
Each cached engine holds ~600 MB of MLX state (weights + compiled graph +
working buffers). Five cached engines is ~3 GB. On a 64 GB Mac that's
trivial; on a 16 GB Mac it can crowd out the rest of an inference pipeline.
Caching is auto-disabled on Macs with < 32 GB unified memory — the same
threshold the ``Predictor`` uses for its Metal cache fraction.

Override via the ``NNUNET_MLX_CACHE_ENGINES`` env var (``"1"`` / ``"0"``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Hashable, Iterable

import mlx.core as mx

from .engine import InferenceEngine, ModelBundle


__all__ = [
    "cache_enabled",
    "clear_engine_cache",
    "get_cached_engine",
    "cache_engine",
    "cached_engine_from_folder",
    "cached_engine_from_task",
    "cached_engine_from_moose_model",
    "resolve_moose_config_folder",
]


_CACHE: dict[Hashable, InferenceEngine] = {}


def cache_enabled() -> bool:
    """Whether engine caching is on for this process.

    Auto-tiered by unified memory size (≥32 GB → on, else off). Override
    via the ``NNUNET_MLX_CACHE_ENGINES`` env var (``"1"``, ``"0"``,
    ``"true"``, ``"false"``, ``"yes"``, ``"no"``).
    """
    env = os.environ.get("NNUNET_MLX_CACHE_ENGINES")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no")
    try:
        ram_gb = mx.device_info().get("memory_size", 0) / 1e9
    except Exception:
        return False
    return ram_gb >= 32


def get_cached_engine(key: Hashable) -> InferenceEngine | None:
    """Look up a cached engine by key.

    Returns ``None`` if caching is disabled or the key is not present.
    Use any hashable key — the convention in this module's helpers is a
    tuple of (model_folder_str, configuration, folds, step_size, compile,
    batch_size, use_mirroring), but callers with bespoke needs can pick
    their own scheme.
    """
    if not cache_enabled():
        return None
    return _CACHE.get(key)


def cache_engine(key: Hashable, engine: InferenceEngine) -> None:
    """Store an engine in the cache under ``key``.

    No-op when caching is disabled, so call sites can call this
    unconditionally after building an engine and let the auto-tier decide
    whether the cache actually retains it.
    """
    if cache_enabled():
        _CACHE[key] = engine


def clear_engine_cache() -> None:
    """Release all cached engines and free Metal buffers.

    Use between unrelated workflows in a long-running process when you want
    to reclaim engine memory without exiting Python. Safe to call when the
    cache is empty.
    """
    for engine in list(_CACHE.values()):
        try:
            engine.close()
        except Exception:
            # close() may not exist on older bundles; swallow rather than
            # leak a half-closed cache state.
            pass
    _CACHE.clear()
    try:
        mx.clear_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cache-key construction + high-level "give me an engine for this folder"
# helpers. These are the common-case APIs.
# ---------------------------------------------------------------------------


def _normalize_folds(folds: int | Iterable[int] | str | None) -> tuple[int, ...] | str:
    """Canonicalize folds for cache-key hashing."""
    if folds is None:
        return (0,)
    if isinstance(folds, str):
        return folds  # e.g. "all"
    if isinstance(folds, int):
        return (folds,)
    return tuple(int(f) for f in folds)


def _engine_key(
    model_folder: Path,
    configuration: str | None,
    folds: int | Iterable[int] | str | None,
    step_size: float,
    compile: bool,
    batch_size: int | None,
    use_mirroring: bool,
) -> tuple:
    """Build a cache key from everything that affects engine state.

    Anything that would change the compiled graph or the loaded weights
    must be in the key. Verbose/progress flags don't change state, so they
    are deliberately excluded — flipping verbosity shouldn't bust the cache.
    """
    return (
        str(model_folder),
        configuration,
        _normalize_folds(folds),
        float(step_size),
        bool(compile),
        batch_size,
        bool(use_mirroring),
    )


def cached_engine_from_folder(
    model_folder: str | Path,
    *,
    configuration: str | None = None,
    folds: int | Iterable[int] | str | None = None,
    step_size: float = 0.5,
    compile: bool = True,
    batch_size: int | None = None,
    use_mirroring: bool = False,
    verbose: bool = False,
    progress: bool = False,
    dtype: str | mx.Dtype | None = None,
) -> InferenceEngine:
    """Get a cached :class:`InferenceEngine` for ``model_folder``, building if needed.

    Cache key derives from everything that affects engine state. Callers can
    flip ``verbose`` / ``progress`` without invalidating the cached engine.

    When caching is disabled (small-RAM Macs, or
    ``NNUNET_MLX_CACHE_ENGINES=0``), this function still works — it just
    builds a fresh engine on every call.
    """
    folds_for_load = folds if folds is not None else 0
    key = _engine_key(
        Path(model_folder), configuration, folds_for_load,
        step_size, compile, batch_size, use_mirroring,
    )

    engine = get_cached_engine(key)
    if engine is not None:
        return engine

    bundle = ModelBundle.from_folder(
        Path(model_folder), folds=folds_for_load, dtype=dtype,
    )
    engine = InferenceEngine(
        bundle,
        configuration=configuration,
        step_size=step_size,
        compile=compile,
        batch_size=batch_size,
        use_mirroring=use_mirroring,
        verbose=verbose,
        progress=progress,
    )
    cache_engine(key, engine)
    return engine


def cached_engine_from_task(
    task_id: int,
    *,
    folds: int | Iterable[int] | str | None = None,
    configuration: str | None = None,
    step_size: float = 0.5,
    compile: bool = True,
    batch_size: int | None = None,
    use_mirroring: bool = False,
    verbose: bool = False,
    progress: bool = False,
    weights_dir: str | Path | None = None,
    trainer: str | None = None,
    plans: str | None = None,
    model: str | None = None,
    dtype: str | mx.Dtype | None = None,
) -> InferenceEngine:
    """Get a cached engine for ``task_id`` via the :class:`WeightsLayout` registry.

    Resolves the model folder using the layout registry (the same path
    ``ModelBundle.from_task`` uses) and then delegates to
    :func:`cached_engine_from_folder`. Cache hits never touch disk.

    ``trainer`` / ``plans`` / ``model`` override the resolved layout's
    defaults, needed for datasets that ship multiple trainer variants
    (e.g. TS's Dataset291 has both ``nnUNetTrainer`` and
    ``nnUNetTrainerNoMirroring``).
    """
    from .engine import discover_weights, _find_model_folder

    # Folder resolution is glob-only — no weight I/O, so it's cheap to do
    # before the cache lookup and we get a stable cache key.
    if weights_dir is None:
        resolved_dir, layout = discover_weights()
        eff_trainer = trainer if trainer is not None else layout.trainer
        eff_plans = plans if plans is not None else layout.plans
        eff_model = model if model is not None else layout.model
    else:
        resolved_dir = Path(weights_dir).expanduser()
        eff_trainer, eff_plans, eff_model = trainer, plans, model

    model_folder = _find_model_folder(
        task_id, resolved_dir,
        trainer=eff_trainer, plans=eff_plans, model=eff_model,
    )

    return cached_engine_from_folder(
        model_folder,
        configuration=configuration,
        folds=folds,
        step_size=step_size,
        compile=compile,
        batch_size=batch_size,
        use_mirroring=use_mirroring,
        verbose=verbose,
        progress=progress,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# MOOSE model resolution
# ---------------------------------------------------------------------------
#
# MOOSE (ENHANCE-PET) stores nnU-Netv2 models under a flat models directory,
# one folder per model named ``Dataset{NNN}_{Name}`` (e.g. ``Dataset123_Organs``).
# Inside each is the standard nnU-Net config folder ``{trainer}__{plans}__{res}``
# containing ``plans.json`` / ``dataset.json`` / ``fold_*``. This is the same
# on-disk shape TS uses; only the *outer* identifier differs — MOOSE keys by
# the full folder name (a string), not an integer dataset ID.


def _default_moose_models_dir() -> Path | None:
    """Best-effort default MOOSE models directory.

    Resolution order:
      1. ``NNUNET_MLX_MOOSE_MODELS`` env var (our override).
      2. ``MOOSE_MODELS`` env var.
      3. The installed ``moosez`` package's models dir, if importable.

    Returns ``None`` if none resolve — the caller then raises a clear error.
    """
    for var in ("NNUNET_MLX_MOOSE_MODELS", "MOOSE_MODELS"):
        val = os.environ.get(var)
        if val:
            return Path(val).expanduser()
    try:
        import moosez.system as _moose_system  # type: ignore

        return Path(_moose_system.MODELS_DIRECTORY_PATH)
    except Exception:
        return None


def resolve_moose_config_folder(
    folder_name: str, models_dir: str | Path | None = None,
) -> Path:
    """Resolve a MOOSE model folder name to its nnU-Net config folder.

    ``folder_name`` is MOOSE's ``KEY_FOLDER_NAME`` (e.g. ``"Dataset123_Organs"``).
    Returns the inner ``{trainer}__{plans}__{res}`` folder that
    :func:`cached_engine_from_folder` consumes (the one holding
    ``plans.json`` / ``dataset.json`` / ``fold_*``).

    ``models_dir`` defaults to :func:`_default_moose_models_dir`.

    Raises ``FileNotFoundError`` with an actionable message if the models
    directory is unknown, the model folder is absent, or it contains no
    nnU-Net config subfolder.
    """
    base = Path(models_dir).expanduser() if models_dir else _default_moose_models_dir()
    if base is None:
        raise FileNotFoundError(
            "MOOSE models directory is unknown. Set NNUNET_MLX_MOOSE_MODELS "
            "(or MOOSE_MODELS), pass models_dir=..., or install moosez."
        )

    model_dir = base / folder_name
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"MOOSE model folder not found: {model_dir}. "
            f"(models_dir={base})"
        )

    # nnU-Net config subfolder: name has exactly two '__' separators
    # (trainer__plans__resolution), matching MOOSE's own selection logic.
    configs = sorted(
        p for p in model_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name.count("__") == 2
    )
    if not configs:
        raise FileNotFoundError(
            f"No nnU-Net config folder (trainer__plans__resolution) found in "
            f"{model_dir}."
        )
    return configs[0]


def cached_engine_from_moose_model(
    folder_name: str,
    *,
    models_dir: str | Path | None = None,
    folds: int | Iterable[int] | str | None = None,
    configuration: str | None = None,
    step_size: float = 0.5,
    compile: bool = True,
    batch_size: int | None = None,
    use_mirroring: bool = False,
    verbose: bool = False,
    progress: bool = False,
    dtype: str | mx.Dtype | None = None,
) -> InferenceEngine:
    """Get a cached engine for a MOOSE model identified by folder name.

    Resolves ``folder_name`` (e.g. ``"Dataset123_Organs"``) under the MOOSE
    models directory to its nnU-Net config folder, then delegates to
    :func:`cached_engine_from_folder` (so cache hits never touch disk).

    This is the MOOSE-side counterpart to :func:`cached_engine_from_task`
    (which resolves integer nnU-Net dataset IDs under the TS/nnU-Net results
    dir). Same engine, same cache — only the folder resolution differs.
    """
    model_folder = resolve_moose_config_folder(folder_name, models_dir)
    return cached_engine_from_folder(
        model_folder,
        configuration=configuration,
        folds=folds,
        step_size=step_size,
        compile=compile,
        batch_size=batch_size,
        use_mirroring=use_mirroring,
        verbose=verbose,
        progress=progress,
        dtype=dtype,
    )
