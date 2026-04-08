"""
ModelBundle + InferenceEngine — clean separation of I/O and computation.

ModelBundle handles all file I/O: finding models, converting weights, loading.
InferenceEngine is pure computation: takes a ModelBundle, returns numpy arrays.

    bundle = ModelBundle.from_task(297)
    engine = InferenceEngine(bundle)
    logits = engine.predict(volume)  # (Z, Y, X) → (K, Z, Y, X)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

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
    convert_model_folder,
    fuzzy_load_weights,
    load_model_weights,
    load_weights_safetensors,
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


class ModelBundle:
    """Model weights, architecture plans, and dataset metadata.

    All file I/O happens here. InferenceEngine receives a ModelBundle
    and never touches the filesystem.

    Attributes
    ----------
    plans : dict
        Parsed plans.json (architecture, patch size, normalization).
    dataset : dict
        Parsed dataset.json (labels, channel names).
    weights : dict[str, mx.array]
        Model parameters keyed by name, ready for load_weights().
    """

    def __init__(self, plans: dict, dataset: dict, weights: dict[str, mx.array]):
        self.plans = plans
        self.dataset = dataset
        self.weights = weights

    @staticmethod
    def from_folder(path: str | Path, fold: int = 0) -> ModelBundle:
        """Load from a local model folder.

        Parameters
        ----------
        path : str or Path
            Path to the model folder containing plans.json, dataset.json,
            and fold_N/ with weights. Can be either the trainer folder
            (e.g. .../nnUNetTrainer__nnUNetPlans__3d_fullres) or the
            dataset folder (e.g. .../Dataset297_...).
        fold : int
            Which fold's weights to load.
        """
        path = Path(path).expanduser()

        # If pointed at a dataset folder, find the trainer subfolder
        if not (path / "plans.json").exists():
            trainer_dirs = sorted(path.glob("*__*__*"))
            if trainer_dirs:
                path = trainer_dirs[0]

        plans = json.loads((path / "plans.json").read_text())
        dataset = json.loads((path / "dataset.json").read_text())
        weights = load_model_weights(path, fold=fold)

        return ModelBundle(plans=plans, dataset=dataset, weights=weights)

    @staticmethod
    def from_task(
        task_id: int,
        fold: int = 0,
        weights_dir: str | Path | None = None,
        auto_convert: bool = True,
    ) -> ModelBundle:
        """Load by task ID from the weights directory.

        Finds the model folder, converts .pth to .safetensors if needed
        (requires torch, one-time), then loads.

        Parameters
        ----------
        task_id : int
            nnU-Net dataset/task ID (e.g. 297).
        fold : int
            Which fold's weights to load.
        weights_dir : str or Path, optional
            Where to look for models. Defaults to ~/.totalsegmentator/nnunet/results
            or $TOTALSEG_WEIGHTS_PATH.
        auto_convert : bool
            If True, convert .pth to .safetensors automatically when
            safetensors are not found. Requires torch.
        """
        if weights_dir is None:
            weights_dir = _default_weights_dir()
        weights_dir = Path(weights_dir).expanduser()

        model_folder = _find_model_folder(task_id, weights_dir)

        # Auto-convert if needed: any .pth without a sibling .safetensors gets
        # converted once via torch. After conversion the runtime is torch-free.
        if auto_convert:
            fold_dir = model_folder / f"fold_{fold}"
            needs_convert = any(
                p.with_suffix(".safetensors").exists() is False
                for p in fold_dir.glob("*.pth")
            )
            if needs_convert:
                print(f"Converting weights to safetensors (one-time, requires torch)...")
                convert_model_folder(model_folder)

        return ModelBundle.from_folder(model_folder, fold=fold)


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
# InferenceEngine — pure computation, no I/O
# ---------------------------------------------------------------------------

class InferenceEngine:
    """In-memory nnU-Net inference engine.

    Takes a ModelBundle (pre-loaded weights and config). Does pure computation.
    No file I/O, no downloads, no path resolution.

    Parameters
    ----------
    bundle : ModelBundle
        Pre-loaded model weights, plans, and dataset metadata.
    step_size : float
        Sliding window overlap (0.5 = 50% overlap).
    compile : bool
        Use mx.compile for fused operations.
    batch_size : int, optional
        Patches per forward pass. None = auto-detect from Metal limits.
    verbose : bool
        Print progress during inference.

    Example
    -------
    >>> bundle = ModelBundle.from_task(297)
    >>> engine = InferenceEngine(bundle)
    >>> logits = engine.predict(volume)  # (Z, Y, X) → (K, Z, Y, X)
    """

    def __init__(
        self,
        bundle: ModelBundle,
        configuration: str = "3d_fullres",
        step_size: float = 0.5,
        compile: bool = True,
        batch_size: int | None = None,
        verbose: bool = False,
        progress: bool = False,
    ):
        self.step_size = step_size
        self.verbose = verbose
        self.progress = progress

        plans = bundle.plans
        dataset = bundle.dataset

        config = plans["configurations"][configuration]
        self.patch_size = tuple(config["patch_size"])
        self.num_classes = len(dataset["labels"])
        self.num_channels = len(
            dataset.get("channel_names", dataset.get("modality", {}))
        )

        # Normalization parameters — extracted once
        norm_schemes = config.get("normalization_schemes", ["CTNormalization"])
        self._norm_schemes = norm_schemes
        self._norm_params = {}
        for ch in range(self.num_channels):
            scheme = norm_schemes[ch] if ch < len(norm_schemes) else norm_schemes[0]
            if scheme == "CTNormalization":
                self._norm_params[ch] = get_normalization_params(plans, ch)

        # Build network and load weights
        network = build_network_from_plans(
            plans,
            configuration,
            self.num_channels,
            self.num_classes,
            deep_supervision=False,
        )

        try:
            network.load_weights(list(bundle.weights.items()))
        except Exception:
            fuzzy_load_weights(network, bundle.weights, verbose=verbose)

        # Limit Metal cache to avoid memory pressure on constrained machines.
        # Without this, MLX caches ~9.5GB of buffers after the first forward
        # pass, leaving insufficient room for the accumulator and volume data.
        mem_info = mx.device_info()
        system_ram = mem_info.get("memory_size", 16 * 1024**3)
        cache_limit = int(system_ram * 0.3)  # 30% of system RAM
        mx.set_cache_limit(cache_limit)

        # Compile
        if compile:
            self._net = mx.compile(network)
        else:
            self._net = network

        # Warmup — first compiled forward is slow
        if verbose:
            print(f"InferenceEngine: warming up "
                  f"(patch={self.patch_size}, classes={self.num_classes})")
        dummy = mx.random.normal((1, *self.patch_size, self.num_channels))
        mx.eval(self._net(dummy))
        del dummy

        # Gaussian importance map — depends only on patch_size
        self._gaussian = compute_gaussian(
            self.patch_size, sigma_scale=1.0 / 8, value_scaling_factor=10
        )

        # Batch size
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            self._batch_size = choose_batch_size(
                self.patch_size, self.num_classes, dtype_bytes=4
            )
            self._batch_size = max(1, self._batch_size)

        # Shape cache
        self._shape_cache: dict[tuple, ShapeContext] = {}

        if verbose:
            print(
                f"InferenceEngine ready: "
                f"patch={self.patch_size}, classes={self.num_classes}, "
                f"batch={self._batch_size}"
            )

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """Normalize a raw volume.

        Parameters
        ----------
        volume : np.ndarray
            Raw volume, shape (Z, Y, X), any numeric dtype.

        Returns
        -------
        np.ndarray
            Normalized volume, shape (Z, Y, X), float32.
        """
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
        """Precompute everything that depends on volume shape.

        Parameters
        ----------
        shape : tuple
            Spatial shape (Z, Y, X).

        Returns
        -------
        ShapeContext
            Reusable context for this shape.
        """
        if shape in self._shape_cache:
            return self._shape_cache[shape]

        pad_widths = []
        for s, p in zip(shape, self.patch_size):
            total = max(0, p - s)
            pad_widths.append((total // 2, total - total // 2))
        needs_padding = any(a > 0 or b > 0 for a, b in pad_widths)

        if needs_padding:
            padded_shape = tuple(
                s + a + b for s, (a, b) in zip(shape, pad_widths)
            )
        else:
            padded_shape = shape

        steps = compute_sliding_window_steps(
            padded_shape, self.patch_size, self.step_size
        )
        slicers = [
            (sz, sy, sx)
            for sz in steps[0]
            for sy in steps[1]
            for sx in steps[2]
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

    def predict(
        self, volume: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        """Run inference on a volume.

        Parameters
        ----------
        volume : np.ndarray
            Shape (Z, Y, X). Raw HU values if normalize=True,
            or pre-normalized float32 if normalize=False.

        Returns
        -------
        np.ndarray
            Logits, shape (K, Z, Y, X), float32.
        """
        if normalize:
            volume = self.normalize(volume)
        else:
            volume = volume.astype(np.float32)

        input_image = volume[np.newaxis]  # (1, Z, Y, X)

        logits = predict_sliding_window_streaming(
            network=self._net,
            input_image=input_image,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            tile_step_size=self.step_size,
            use_gaussian=True,
            use_mirroring=False,
            batch_size=self._batch_size,
            use_fp16=False,
            verbose=self.verbose,
            progress=self.progress,
        )

        return logits  # (K, Z, Y, X)


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
