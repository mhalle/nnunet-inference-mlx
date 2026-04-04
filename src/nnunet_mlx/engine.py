"""
InferenceEngine — streamlined in-memory inference for nnU-Net models.

No file I/O. Numpy in, numpy out. Model loaded and compiled once.

    engine = InferenceEngine(task_id=297)
    logits = engine.predict(volume)  # (Z, Y, X) → (K, Z, Y, X)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
from .predict import find_model_folder
from .preprocessing import ct_normalization, get_normalization_params, zscore_normalization
from .weights import fuzzy_load_weights, load_model_weights


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


class InferenceEngine:
    """In-memory nnU-Net inference engine.

    Loads the model once on init. Takes numpy arrays, returns numpy arrays.
    No NIfTI, no file I/O.

    Parameters
    ----------
    task_id : int
        nnU-Net dataset/task ID (e.g. 297 for TotalSegmentator fast).
    fold : int
        Which fold's weights to load.
    trainer : str
        nnU-Net trainer name.
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
    >>> engine = InferenceEngine(task_id=297)
    >>> volume = np.random.randn(167, 167, 236).astype(np.float32)
    >>> logits = engine.predict(volume)  # (118, 167, 167, 236)
    """

    def __init__(
        self,
        task_id: int = 297,
        fold: int = 0,
        trainer: str = "nnUNetTrainer_4000epochs_NoMirroring",
        step_size: float = 0.5,
        compile: bool = True,
        batch_size: int | None = None,
        verbose: bool = False,
    ):
        self.step_size = step_size
        self.verbose = verbose

        # Locate model and load plans
        model_folder = find_model_folder(task_id, trainer=trainer)
        plans = json.loads((model_folder / "plans.json").read_text())
        dataset = json.loads((model_folder / "dataset.json").read_text())

        config = plans["configurations"]["3d_fullres"]
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

        # Build network
        network = build_network_from_plans(
            plans,
            "3d_fullres",
            self.num_channels,
            self.num_classes,
            deep_supervision=False,
        )

        # Load weights
        weights = load_model_weights(model_folder, fold=fold)
        try:
            network.load_weights(list(weights.items()))
        except Exception:
            fuzzy_load_weights(network, weights, verbose=verbose)

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
                f"InferenceEngine ready: task={task_id}, "
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

        # Single-channel: apply the first channel's normalization
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
        # NoNormalization: already float32

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

        # Padding
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

        # Sliding window positions
        steps = compute_sliding_window_steps(
            padded_shape, self.patch_size, self.step_size
        )
        slicers = [
            (sz, sy, sx)
            for sz in steps[0]
            for sy in steps[1]
            for sx in steps[2]
        ]

        # Crop slices to undo padding
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

        # Add channel dim: (Z, Y, X) → (1, Z, Y, X) for sliding window
        input_image = volume[np.newaxis]

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
        )

        return logits  # (K, Z, Y, X)
