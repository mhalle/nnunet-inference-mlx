"""
Layered inference stack.

Layer 1 (model + weight primitives) lives in model.py / plans.py / weights.py.
This module implements the layers above:

* ``ModelBundle`` — pure I/O artifact: plans, dataset, ``fold_weights`` (list).
* ``Predictor``  — one compiled MLX network instance, weight-swappable. Does
  one forward pass on an input array. No sliding window, no normalization.
* ``SlidingWindowEngine`` — sliding-window scaffolding over a ``Predictor``:
  Gaussian map, streaming accumulator, shape cache, normalization.
* ``FoldEnsemble``  — wraps either a ``Predictor`` or a ``SlidingWindowEngine``.
  Loops the fold weight dicts via ``Predictor.reload_weights`` between
  forwards, averages softmax.
* ``InferenceEngine`` — thin back-compat facade. Given a bundle, builds the
  right composition (Predictor + SlidingWindowEngine, optionally wrapped in
  FoldEnsemble for multi-fold bundles). The 90% caller stays one line.

Downstream consumers compose what they need:

    TS / single fold:                      Predictor → SlidingWindowEngine
    MOOSE / chunked, multi-fold ensemble:  Predictor → SlidingWindowEngine → FoldEnsemble
    nnInteractive / single 192³ patch:     Predictor.forward(patch)  (skip Layers 3-4)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import mlx.core as mx
import numpy as np

from .inference import (
    choose_batch_size,
    compute_gaussian,
    compute_sliding_window_steps,
    predict_sliding_window_streaming,
)
from .plans import build_network_from_plans
from .preprocessing import ct_normalization, get_normalization_params, zscore_normalization
from .weights import (
    discover_folds,
    fuzzy_load_weights,
    load_checkpoint_with_metadata,
    load_model_weights,
)


# ---------------------------------------------------------------------------
# ModelBundle — all I/O lives here
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS_DIR = None  # resolved lazily


def _default_weights_dir() -> Path:
    if "nnUNet_results" in os.environ:
        return Path(os.environ["nnUNet_results"])
    raise FileNotFoundError(
        "No weights directory specified. Either pass weights_dir= "
        "or set the nnUNet_results environment variable."
    )


def _find_model_folder(task_id: int, weights_dir: Path) -> Path:
    """Resolve task_id to a model folder path."""
    matches = sorted(weights_dir.glob(f"Dataset{task_id}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No model found for task {task_id} in {weights_dir}."
        )
    dataset_dir = matches[0]
    # Find the single trainer subfolder
    trainer_dirs = sorted(dataset_dir.glob("*__*__*"))
    if not trainer_dirs:
        raise FileNotFoundError(
            f"No trainer folder found in {dataset_dir}."
        )
    return trainer_dirs[0]


@dataclass
class ModelBundle:
    """Pure I/O artifact: plans + dataset + N fold weight dicts + checkpoint metadata.

    A bundle says nothing about *how* inference runs — that's the caller's
    composition (Predictor, SlidingWindowEngine, FoldEnsemble). A bundle is
    just "here is what's on disk."

    Attributes
    ----------
    plans : dict
        Parsed plans.json (architecture, patch size, normalization).
    dataset : dict
        Parsed dataset.json (labels, channel names).
    fold_weights : list[dict[str, mx.array]]
        One MLX weight dict per loaded fold. Always non-empty; length 1 for
        single-fold bundles.
    metadata : dict
        First fold's checkpoint metadata (everything except network_weights):
        ``init_args``, ``trainer_name``, ``inference_allowed_mirroring_axes``,
        etc. Used by Predictor/InferenceEngine to auto-detect configuration
        name and other inference hints.
    fold_ids : tuple[int, ...]
        The fold IDs that were loaded, in the order they appear in
        ``fold_weights``.
    """

    plans: dict
    dataset: dict
    fold_weights: list[dict[str, mx.array]]
    metadata: dict = field(default_factory=dict)
    fold_ids: tuple[int, ...] = ()

    @property
    def mirroring_axes(self) -> tuple[int, ...]:
        """Spatial axes the trained model allows TTA mirroring along.

        Returns the ``inference_allowed_mirroring_axes`` captured at training
        time, or ``()`` when the model wasn't trained with mirroring (e.g.
        TotalSegmentator's ``NoMirroring`` variants). Spatial axes are in
        ``(Z, Y, X) == (D, H, W)`` order.
        """
        axes = (self.metadata or {}).get("inference_allowed_mirroring_axes")
        return tuple(axes) if axes else ()

    @staticmethod
    def _resolve_trainer_folder(path: Path) -> Path:
        if (path / "plans.json").exists():
            return path
        trainer_dirs = sorted(path.glob("*__*__*"))
        if trainer_dirs:
            return trainer_dirs[0]
        return path

    @staticmethod
    def from_folder(
        path: str | Path,
        folds: int | Iterable[int] | str = "all",
    ) -> ModelBundle:
        """Load from a local model folder.

        Parameters
        ----------
        path : str or Path
            Path to the trainer folder (``.../nnUNetTrainer__nnUNetPlans__3d_fullres``)
            or the dataset folder (``.../Dataset297_...``).
        folds : int, iterable[int], or "all", default "all"
            Which fold weights to load.

            * ``int`` — single fold (length-1 bundle).
            * ``iterable[int]`` — multi-fold ensemble in caller order.
            * ``"all"`` — auto-detect every ``fold_*`` subdir (sorted).

            The default loads whatever is on disk, which works for
            single-fold release builds (e.g. TotalSegmentator) and
            multi-fold trained models (e.g. MOOSE) without the caller
            needing to know upfront.

        Returns
        -------
        ModelBundle
            ``fold_weights`` has one entry per loaded fold.
        """
        path = ModelBundle._resolve_trainer_folder(Path(path).expanduser())
        fold_ids = ModelBundle._normalize_folds(folds, path)

        plans = json.loads((path / "plans.json").read_text())
        dataset = json.loads((path / "dataset.json").read_text())

        fold_weights: list[dict[str, mx.array]] = []
        metadata: dict = {}
        for i, f in enumerate(fold_ids):
            w, meta = load_checkpoint_with_metadata(path, fold=f)
            fold_weights.append(w)
            if i == 0:
                metadata = meta

        return ModelBundle(
            plans=plans,
            dataset=dataset,
            fold_weights=fold_weights,
            metadata=metadata,
            fold_ids=fold_ids,
        )

    @staticmethod
    def _normalize_folds(
        folds: int | Iterable[int] | str, path: Path
    ) -> tuple[int, ...]:
        if isinstance(folds, str):
            if folds != "all":
                raise ValueError(f"folds= must be int, iterable, or 'all'; got {folds!r}")
            discovered = discover_folds(path)
            if not discovered:
                raise FileNotFoundError(f"No fold_* subdirs in {path}")
            return discovered
        if isinstance(folds, int):
            return (folds,)
        ids = tuple(int(f) for f in folds)
        if not ids:
            raise ValueError("folds= must contain at least one fold ID.")
        return ids

    @staticmethod
    def from_task(
        task_id: int,
        folds: int | Iterable[int] | str = "all",
        weights_dir: str | Path | None = None,
    ) -> ModelBundle:
        """Load by task ID from the weights directory.

        See :meth:`from_folder` for ``folds`` semantics.

        Parameters
        ----------
        task_id : int
            nnU-Net dataset/task ID (e.g. 297).
        weights_dir : str or Path, optional
            Where to look for models. Defaults to ``$nnUNet_results``.
        """
        if weights_dir is None:
            weights_dir = _default_weights_dir()
        weights_dir = Path(weights_dir).expanduser()

        model_folder = _find_model_folder(task_id, weights_dir)
        return ModelBundle.from_folder(model_folder, folds=folds)


# ---------------------------------------------------------------------------
# ShapeContext — precomputed per-shape state
# ---------------------------------------------------------------------------

@dataclass
class ShapeContext:
    """Precomputed state for a given volume shape."""

    shape: tuple[int, int, int]
    pad_widths: list[tuple[int, int]]
    needs_padding: bool
    padded_shape: tuple[int, int, int]
    slicers: list[tuple[int, int, int]]
    crop_slices: tuple[slice, ...]
    n_patches: int


# ---------------------------------------------------------------------------
# Configuration resolution — shared by Predictor and InferenceEngine
# ---------------------------------------------------------------------------

def _resolve_configuration(
    plans: dict, metadata: dict, configuration: str | None
) -> str:
    """Pick which plans configuration to use.

    Order: explicit caller arg > checkpoint init_args > only-configuration in
    plans > "3d_fullres" fallback.
    """
    if configuration is not None:
        return configuration
    init_cfg = (metadata or {}).get("init_args", {}).get("configuration")
    if init_cfg:
        return init_cfg
    configs = list(plans.get("configurations", {}).keys())
    if len(configs) == 1:
        return configs[0]
    return "3d_fullres"


# ---------------------------------------------------------------------------
# Layer 2 — Predictor: one compiled, weight-swappable MLX network
# ---------------------------------------------------------------------------

class Predictor:
    """One MLX network instance: compiled, warmed up, weight-swappable.

    Owns the network, the compile, the warmup, and the Metal cache discipline.
    Knows nothing about sliding windows, ensembling, or normalization — those
    are the layers above.

    Use directly for raw-forward workflows (e.g. nnInteractive, which feeds
    one 192³ patch at a time). Wrap with :class:`SlidingWindowEngine` for
    nnU-Net-style whole-volume inference, and/or :class:`FoldEnsemble` for
    multi-fold averaging.

    Attributes
    ----------
    network : mlx.nn.Module
        The underlying MLX network. Public so callers that need raw access
        (custom forward loops, parameter inspection) don't have to reach
        into private state.
    patch_size : tuple[int, int, int]
        Spatial patch size from the chosen plans configuration.
    num_classes : int
        Number of output channels (segmentation heads).
    num_input_channels : int
        Network input channel count. Derived from ``dataset["channel_names"]``
        unless overridden.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        configuration: str | None = None,
        num_input_channels: int | None = None,
        compile: bool = True,
        verbose: bool = False,
        cache_limit_fraction: float = 0.3,
    ):
        if not bundle.fold_weights:
            raise ValueError("Bundle has no fold_weights.")
        self._bundle = bundle
        self._verbose = verbose

        plans = bundle.plans
        dataset = bundle.dataset
        cfg_name = _resolve_configuration(plans, bundle.metadata, configuration)
        if cfg_name not in plans.get("configurations", {}):
            raise KeyError(
                f"Configuration {cfg_name!r} not found in plans; "
                f"available: {list(plans['configurations'])}"
            )
        config = plans["configurations"][cfg_name]
        self.configuration = cfg_name
        self.patch_size = tuple(config["patch_size"])
        self.num_classes = len(dataset["labels"])

        if num_input_channels is None:
            num_input_channels = len(
                dataset.get("channel_names", dataset.get("modality", {}))
            )
        self.num_input_channels = num_input_channels

        # Limit Metal cache before any allocation. Without this, MLX caches
        # ~9.5GB of buffers after the first forward pass on constrained Macs.
        mem_info = mx.device_info()
        system_ram = mem_info.get("memory_size", 16 * 1024**3)
        mx.set_cache_limit(int(system_ram * cache_limit_fraction))

        self.network = build_network_from_plans(
            plans,
            cfg_name,
            num_input_channels,
            self.num_classes,
            deep_supervision=False,
        )
        self.reload_weights(bundle.fold_weights[0])

        # mx.compile wraps a callable; weight updates via network.load_weights
        # still propagate because the compiled function closes over the module
        # and re-reads its parameters on each call.
        self._compiled = mx.compile(self.network) if compile else self.network

        if verbose:
            print(
                f"Predictor: warming up "
                f"(patch={self.patch_size}, in_ch={num_input_channels}, "
                f"classes={self.num_classes})"
            )
        dummy = mx.random.normal((1, *self.patch_size, num_input_channels))
        mx.eval(self._compiled(dummy))
        del dummy

    def reload_weights(self, weights: dict[str, mx.array]) -> None:
        """Swap weights into the existing network in place.

        Used by :class:`FoldEnsemble` between folds, and by any caller that
        wants to cycle through multiple weight sets without paying the
        rebuild + compile + warmup cost.
        """
        try:
            self.network.load_weights(list(weights.items()))
        except Exception:
            fuzzy_load_weights(self.network, weights, verbose=self._verbose)

    def forward(self, x: mx.array) -> mx.array:
        """One forward pass on a batched, channels-last input.

        Parameters
        ----------
        x : mx.array
            Shape ``(B, *patch_size, num_input_channels)``.

        Returns
        -------
        mx.array
            Shape ``(B, *patch_size, num_classes)``.
        """
        return self._compiled(x)

    __call__ = forward

    def close(self) -> None:
        """Release the compiled graph and clear the Metal cache."""
        self._compiled = None
        self.network = None
        mx.clear_cache()


# ---------------------------------------------------------------------------
# Layer 3 — SlidingWindowEngine: nnU-Net whole-volume scaffolding
# ---------------------------------------------------------------------------

class SlidingWindowEngine:
    """Sliding-window inference over a :class:`Predictor`.

    Adds Gaussian importance weighting, the streaming accumulator, the shape
    cache, and per-channel normalization. Takes a raw ``(Z, Y, X)`` volume
    and returns logits ``(K, Z, Y, X)``.

    Single-fold ensemble: just construct one of these. For multi-fold
    averaging, wrap with :class:`FoldEnsemble`.
    """

    def __init__(
        self,
        predictor: Predictor,
        step_size: float = 0.5,
        batch_size: int | None = None,
        use_mirroring: bool = False,
        verbose: bool = False,
        progress: bool = False,
    ):
        self.predictor = predictor
        self.step_size = step_size
        self.verbose = verbose
        self.progress = progress
        # Mirroring axes auto-read from the bundle's checkpoint metadata.
        # If the model wasn't trained with mirroring, use_mirroring=True is
        # silently a no-op (mirror_axes=() short-circuits the TTA loop).
        self.use_mirroring = use_mirroring
        self.mirror_axes = predictor._bundle.mirroring_axes if use_mirroring else ()

        plans = predictor._bundle.plans
        config = plans["configurations"][predictor.configuration]
        norm_schemes = config.get("normalization_schemes", ["CTNormalization"])
        self._norm_schemes = norm_schemes
        self._norm_params = {}
        for ch in range(predictor.num_input_channels):
            scheme = norm_schemes[ch] if ch < len(norm_schemes) else norm_schemes[0]
            if scheme == "CTNormalization":
                self._norm_params[ch] = get_normalization_params(plans, ch)

        self._gaussian = compute_gaussian(
            predictor.patch_size, sigma_scale=1.0 / 8, value_scaling_factor=10
        )

        if batch_size is None:
            batch_size = max(
                1,
                choose_batch_size(
                    predictor.patch_size, predictor.num_classes, dtype_bytes=4
                ),
            )
        self._batch_size = batch_size

        self._shape_cache: dict[tuple, ShapeContext] = {}

    # Convenience accessors —————————————————————————————————————————————
    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.predictor.patch_size

    @property
    def num_classes(self) -> int:
        return self.predictor.num_classes

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """Apply the configuration's per-channel normalization to a (Z,Y,X) volume."""
        data = volume.astype(np.float32)
        ch = 0
        scheme = (
            self._norm_schemes[ch]
            if ch < len(self._norm_schemes)
            else self._norm_schemes[0]
        )
        if scheme == "CTNormalization":
            params = self._norm_params[ch]
            data = ct_normalization(
                data,
                mean=params["mean"],
                std=params["std"],
                lower_clip=params["lower_clip"],
                upper_clip=params["upper_clip"],
            )
        elif scheme == "ZScoreNormalization":
            data = zscore_normalization(data)
        return data

    def prepare(self, shape: tuple[int, int, int]) -> ShapeContext:
        """Precompute padding / slicers / crop slices for a given (Z,Y,X) shape."""
        if shape in self._shape_cache:
            return self._shape_cache[shape]

        pad_widths = []
        for s, p in zip(shape, self.patch_size):
            total = max(0, p - s)
            pad_widths.append((total // 2, total - total // 2))
        needs_padding = any(a > 0 or b > 0 for a, b in pad_widths)
        padded_shape = (
            tuple(s + a + b for s, (a, b) in zip(shape, pad_widths))
            if needs_padding
            else shape
        )
        steps = compute_sliding_window_steps(
            padded_shape, self.patch_size, self.step_size
        )
        slicers = [
            (sz, sy, sx) for sz in steps[0] for sy in steps[1] for sx in steps[2]
        ]
        crop_slices = tuple(
            slice(a, s - b) if (a > 0 or b > 0) else slice(None)
            for s, (a, b) in zip(padded_shape, pad_widths)
        )
        ctx = ShapeContext(
            shape=shape,
            pad_widths=pad_widths,
            needs_padding=needs_padding,
            padded_shape=padded_shape,
            slicers=slicers,
            crop_slices=crop_slices,
            n_patches=len(slicers),
        )
        self._shape_cache[shape] = ctx
        return ctx

    def predict(self, volume: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Run sliding-window inference. Returns logits ``(K, Z, Y, X)``."""
        if normalize:
            volume = self.normalize(volume)
        else:
            volume = volume.astype(np.float32)
        return predict_sliding_window_streaming(
            network=self.predictor._compiled,
            input_image=volume[np.newaxis],
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            tile_step_size=self.step_size,
            use_gaussian=True,
            use_mirroring=self.use_mirroring and bool(self.mirror_axes),
            mirror_axes=self.mirror_axes,
            batch_size=self._batch_size,
            use_fp16=False,
            verbose=self.verbose,
            progress=self.progress,
        )

    def close(self) -> None:
        self._shape_cache.clear()
        self.predictor.close()


# ---------------------------------------------------------------------------
# Layer 4 — FoldEnsemble: orthogonal multi-fold averaging
# ---------------------------------------------------------------------------

class FoldEnsemble:
    """Softmax-averaging fold ensemble over a Predictor or SlidingWindowEngine.

    Loops the bundle's ``fold_weights`` via :meth:`Predictor.reload_weights`
    between forwards. Skips the loop entirely when there's only one fold,
    so wrapping a single-fold bundle is a no-op cost.

    Returns averaged probabilities (already softmaxed), not logits — the
    average-of-softmaxes is the standard nnU-Net ensemble convention. Same
    shape semantics otherwise; ``argmax(axis=0)`` still yields the segmentation.
    """

    def __init__(
        self,
        backend: Predictor | SlidingWindowEngine,
        fold_weights: Sequence[dict[str, mx.array]] | None = None,
    ):
        self.backend = backend
        if fold_weights is None:
            predictor = backend if isinstance(backend, Predictor) else backend.predictor
            fold_weights = predictor._bundle.fold_weights
        self.fold_weights = list(fold_weights)
        if not self.fold_weights:
            raise ValueError("FoldEnsemble needs at least one fold.")

    def _predictor(self) -> Predictor:
        if isinstance(self.backend, Predictor):
            return self.backend
        return self.backend.predictor

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Run the backend's predict / forward across all folds, average softmax."""
        if len(self.fold_weights) == 1:
            self._predictor().reload_weights(self.fold_weights[0])
            return self._single(*args, **kwargs)

        acc: np.ndarray | None = None
        for w in self.fold_weights:
            self._predictor().reload_weights(w)
            out = self._single(*args, **kwargs)
            probs = softmax_inplace(np.asarray(out, dtype=np.float32))
            if acc is None:
                acc = probs
            else:
                acc += probs
        acc /= len(self.fold_weights)
        return acc

    def _single(self, *args, **kwargs):
        if isinstance(self.backend, SlidingWindowEngine):
            return self.backend.predict(*args, **kwargs)
        return self.backend.forward(*args, **kwargs)

    def close(self) -> None:
        self.backend.close()


# ---------------------------------------------------------------------------
# InferenceEngine — back-compat facade for the 90% case
# ---------------------------------------------------------------------------

class InferenceEngine:
    """One-line entry point for whole-volume inference.

    Given a :class:`ModelBundle`, builds the right composition:

    * Single-fold bundle  → ``Predictor → SlidingWindowEngine``
    * Multi-fold bundle   → ``Predictor → SlidingWindowEngine → FoldEnsemble``

    The single-fold path returns raw logits (back-compat). The multi-fold
    path returns averaged softmax probabilities. ``argmax(axis=0)`` on either
    output yields the segmentation, so callers that only argmax don't need
    to branch.

    Need more control (no sliding window, custom batching, per-patch ensemble)?
    Build the layers directly — ``Predictor``, ``SlidingWindowEngine``,
    ``FoldEnsemble`` — and skip this facade.

    Example
    -------
    >>> bundle = ModelBundle.from_task(297)
    >>> engine = InferenceEngine(bundle)
    >>> logits = engine.predict(volume)  # (Z, Y, X) → (K, Z, Y, X)
    """

    def __init__(
        self,
        bundle: ModelBundle,
        configuration: str | None = None,
        step_size: float = 0.5,
        compile: bool = True,
        batch_size: int | None = None,
        use_mirroring: bool = False,
        verbose: bool = False,
        progress: bool = False,
        num_input_channels: int | None = None,
    ):
        self._predictor = Predictor(
            bundle,
            configuration=configuration,
            num_input_channels=num_input_channels,
            compile=compile,
            verbose=verbose,
        )
        self._sliding = SlidingWindowEngine(
            self._predictor,
            step_size=step_size,
            batch_size=batch_size,
            use_mirroring=use_mirroring,
            verbose=verbose,
            progress=progress,
        )
        self._backend = (
            FoldEnsemble(self._sliding, bundle.fold_weights)
            if len(bundle.fold_weights) > 1
            else self._sliding
        )
        if verbose:
            print(
                f"InferenceEngine ready: patch={self.patch_size}, "
                f"classes={self.num_classes}, "
                f"folds={len(bundle.fold_weights)}"
            )

    # Expose the layered objects for callers that want them.
    @property
    def predictor(self) -> Predictor:
        return self._predictor

    @property
    def sliding_window(self) -> SlidingWindowEngine:
        return self._sliding

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self._predictor.patch_size

    @property
    def num_classes(self) -> int:
        return self._predictor.num_classes

    @property
    def num_channels(self) -> int:
        return self._predictor.num_input_channels

    @property
    def batch_size(self) -> int:
        return self._sliding._batch_size

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        return self._sliding.normalize(volume)

    def prepare(self, shape: tuple[int, int, int]) -> ShapeContext:
        return self._sliding.prepare(shape)

    def predict(self, volume: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Run inference. Single-fold → logits; multi-fold → averaged probs."""
        return self._backend.predict(volume, normalize=normalize)

    def close(self) -> None:
        """Release the compiled graph and clear the MLX Metal cache."""
        self._backend.close()

    def __enter__(self) -> InferenceEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            mx.clear_cache()
        except Exception:
            pass


def softmax_inplace(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities in-place along axis 0.

    Parameters
    ----------
    logits : np.ndarray
        Shape (K, ...), float32. Modified in-place.

    Returns
    -------
    np.ndarray
        The same array, now containing probabilities that sum to 1
        along axis 0.
    """
    logits -= logits.max(axis=0, keepdims=True)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=0, keepdims=True)
    return logits
