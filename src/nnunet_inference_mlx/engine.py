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

from dataclasses import dataclass, field
from typing import Sequence

import mlx.core as mx
import numpy as np

from .inference import (
    choose_batch_size,
    compute_gaussian,
    compute_sliding_window_steps,
    predict_sliding_window,
)
from .labels import (
    convert_logits_to_segmentation,
    has_regions,
    label_dtype,
    regions_class_order,
    sigmoid_inplace,
)
from .plans import build_network_from_plans
from .preprocessing import ct_normalization, get_normalization_params, zscore_normalization
from .weights import fuzzy_load_weights


# ---------------------------------------------------------------------------
# ModelBundle — all I/O lives here
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS_DIR = None  # resolved lazily


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
    def has_regions(self) -> bool:
        """True if this dataset uses region-based labels (BraTS-style).

        Region-based means some label values in ``dataset.json`` are lists of
        underlying classes rather than single ints. The model emits one
        sigmoid head per region, not a softmax across classes. Post-
        processing differs (per-region threshold + paint priority) and the
        fold-ensemble averaging differs (sigmoid-mean vs softmax-mean).
        """
        return has_regions(self.dataset)

    @property
    def regions_class_order(self) -> tuple[int, ...]:
        """Paint-priority order for region-based label conversion.

        Empty tuple for standard datasets. For region-based datasets, raises
        ``ValueError`` if dataset.json is missing ``regions_class_order``.
        """
        return regions_class_order(self.dataset)

    @property
    def target_spacing(self) -> tuple[float, float, float]:
        """Voxel spacing (Z, Y, X) in mm the model was trained to expect.

        Callers resample their input to this spacing before
        ``engine.predict``. nnU-Net's plans.json stores ``spacing`` per
        configuration in (Z, Y, X) order; we return it as-is.

        If multiple configurations are in plans.json, returns the spacing
        for the first one. Pass an explicit configuration via the
        Predictor constructor and consult its plans directly if you need
        a non-default configuration's spacing.
        """
        configs = self.plans.get("configurations", {})
        if not configs:
            raise KeyError("plans.json has no configurations.")
        # Default to the metadata's init_args.configuration if available,
        # otherwise the first one in the dict.
        init_cfg = (self.metadata or {}).get("init_args", {}).get("configuration")
        cfg = configs.get(init_cfg) or next(iter(configs.values()))
        spacing = cfg.get("spacing")
        if spacing is None:
            raise KeyError(
                "No 'spacing' field in plans configuration; cannot determine "
                "target spacing."
            )
        # plans.json stores spacing in *transposed* axis order (the order the
        # network sees). To present canonical-order spacing to callers (Z, Y, X
        # = caller's numpy axes), apply ``transpose_backward``. For models with
        # identity transpose this is a no-op.
        plans_spacing = tuple(float(s) for s in spacing)
        tb = self.transpose_backward
        if tb == (0, 1, 2):
            return plans_spacing
        return tuple(plans_spacing[i] for i in tb)

    @property
    def transpose_forward(self) -> tuple[int, int, int]:
        """Axis permutation applied to canonical-order volumes before they reach the network.

        Read from ``plans.json``'s top-level ``transpose_forward``. Defaults
        to ``(0, 1, 2)`` (identity) when absent. nnU-Net's training pipeline
        applies this permutation to every input volume before patching, so
        the model has only ever seen volumes in transposed-axis order.

        For TS Datasets 291–298 this is identity; some research nnU-Net
        models have non-identity transposes (e.g., ``(2, 0, 1)``).
        :meth:`InferenceEngine.predict_logits` and siblings handle the
        transpose round-trip automatically so callers always work in
        canonical axes; this property is exposed for users composing
        primitive layers manually.
        """
        t = self.plans.get("transpose_forward", (0, 1, 2))
        return tuple(int(x) for x in t)

    @property
    def transpose_backward(self) -> tuple[int, int, int]:
        """Inverse permutation of :attr:`transpose_forward`.

        Used to convert model-order logits back to canonical-order. Read
        from ``plans.json``'s top-level ``transpose_backward`` (which
        nnU-Net stores alongside ``transpose_forward`` for convenience —
        it's always ``np.argsort(transpose_forward)``). Defaults to
        ``(0, 1, 2)`` (identity) when absent.
        """
        t = self.plans.get("transpose_backward", (0, 1, 2))
        return tuple(int(x) for x in t)

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
        cache_limit_fraction: float | None = None,
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
        # Number of network output heads. For standard softmax models, that's
        # one per label (background included as a softmax class). For region-
        # based models, it's one per foreground region — background is
        # implicit, and regions of size 1 may be written as bare ints (e.g.
        # ``"ET": 3``) rather than 1-element lists. Either way they count as
        # a head. Matches nnUNetv2's LabelManager.num_segmentation_heads.
        labels = dataset["labels"]
        if has_regions(dataset):
            self.num_classes = sum(
                1 for k in labels if k != "background"
            )
        else:
            self.num_classes = len(labels)

        if num_input_channels is None:
            num_input_channels = len(
                dataset.get("channel_names", dataset.get("modality", {}))
            )
        self.num_input_channels = num_input_channels

        # Limit Metal cache before any allocation. Without this, MLX caches
        # ~9.5GB of buffers after the first forward pass on constrained Macs.
        # cache_limit_fraction=None auto-tiers by detected unified-memory
        # size: small Macs get 0.30 to leave room for accumulators, big Macs
        # get 0.50 so MLX doesn't keep evicting compiled-graph buffers between
        # forward passes. Pass an explicit fraction to override.
        mem_info = mx.device_info()
        system_ram = mem_info.get("memory_size", 16 * 1024**3)
        if cache_limit_fraction is None:
            ram_gb = system_ram / 1e9
            if ram_gb >= 32:
                cache_limit_fraction = 0.50
            else:
                cache_limit_fraction = 0.30
        self.cache_limit_fraction = cache_limit_fraction
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
        return predict_sliding_window(
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
    """Probability-averaging fold ensemble over a Predictor or SlidingWindowEngine.

    Loops the bundle's ``fold_weights`` via :meth:`Predictor.reload_weights`
    between forwards. Skips the loop entirely when there's only one fold,
    so wrapping a single-fold bundle is a no-op cost.

    Averaging math depends on the model's label scheme:

    * Standard N-class models: softmax-then-average (heads are softmax-related).
    * Region-based models: sigmoid-then-average (each head is independent).

    Pass ``region_based=True`` to force the sigmoid path; defaults to False
    so the facade can construct it conventionally. Returns averaged
    probabilities. ``argmax(axis=0)`` still yields the segmentation for
    standard models; for region-based, use
    :func:`labels.convert_logits_to_segmentation` (with ``threshold=0.5``,
    since the output is post-sigmoid).
    """

    def __init__(
        self,
        backend: Predictor | SlidingWindowEngine,
        fold_weights: Sequence[dict[str, mx.array]] | None = None,
        region_based: bool = False,
    ):
        self.backend = backend
        if fold_weights is None:
            predictor = backend if isinstance(backend, Predictor) else backend.predictor
            fold_weights = predictor._bundle.fold_weights
        self.fold_weights = list(fold_weights)
        if not self.fold_weights:
            raise ValueError("FoldEnsemble needs at least one fold.")
        self.region_based = region_based

    def _predictor(self) -> Predictor:
        if isinstance(self.backend, Predictor):
            return self.backend
        return self.backend.predictor

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Run the backend's predict / forward across all folds, average probabilities."""
        if len(self.fold_weights) == 1:
            self._predictor().reload_weights(self.fold_weights[0])
            return self._single(*args, **kwargs)

        squash = sigmoid_inplace if self.region_based else softmax_inplace
        acc: np.ndarray | None = None
        for w in self.fold_weights:
            self._predictor().reload_weights(w)
            out = self._single(*args, **kwargs)
            probs = squash(np.asarray(out, dtype=np.float32))
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

    A private compute core: build it from a :class:`ModelBundle` you construct
    directly. End users go through ``ModelStore`` / ``build_model`` / ``segment``
    (which read folders via ``ModelData.read_folder`` and own the bundle).

    Example
    -------
    >>> bundle = ModelBundle(plans=plans, dataset=dataset,
    ...                      fold_weights=[weights], metadata={}, fold_ids=(0,))
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
        self._bundle = bundle
        self._backend = (
            FoldEnsemble(
                self._sliding,
                bundle.fold_weights,
                region_based=bundle.has_regions,
            )
            if len(bundle.fold_weights) > 1
            else self._sliding
        )
        if verbose:
            scheme = "region-based" if bundle.has_regions else "standard"
            print(
                f"InferenceEngine ready: patch={self.patch_size}, "
                f"classes={self.num_classes}, "
                f"folds={len(bundle.fold_weights)}, labels={scheme}"
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

    @property
    def bundle(self) -> ModelBundle:
        """The underlying :class:`ModelBundle` (plans, dataset, weights, metadata).

        Use the curated properties (``target_spacing``, ``has_regions``,
        ``regions_class_order``, ``label_dtype``) for routine access; this
        is the escape hatch for anything else from the bundle metadata.
        """
        return self._bundle

    @property
    def target_spacing(self) -> tuple[float, float, float]:
        """Voxel spacing ``(Z, Y, X)`` in mm the model expects as input.

        Callers resample their input to this spacing before
        :meth:`predict_logits` / :meth:`predict_segmentation`. Returned
        in numpy axis order (Z, Y, X), not SITK order (X, Y, Z).
        """
        return self._bundle.target_spacing

    @property
    def has_regions(self) -> bool:
        """``True`` for BraTS-style models with independent sigmoid heads
        (region-based label scheme), ``False`` for standard mutually-exclusive
        classes (softmax)."""
        return self._bundle.has_regions

    @property
    def regions_class_order(self) -> tuple[int, ...]:
        """Paint-priority tuple for region-based label conversion.

        Empty tuple for standard datasets. For region-based datasets, the
        i-th value is the label ID that region i paints into the output
        segmentation, applied in order so later regions overwrite earlier
        ones at overlapping voxels.
        """
        return self._bundle.regions_class_order

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        return self._sliding.normalize(volume)

    def prepare(self, shape: tuple[int, int, int]) -> ShapeContext:
        return self._sliding.prepare(shape)

    def _apply_transpose_forward(self, volume_zyx: np.ndarray) -> np.ndarray:
        """Permute a canonical-order volume to the model's expected axis order.

        nnU-Net training pipelines apply ``transpose_forward`` to every input
        before patching; the model has only ever seen volumes in that
        permuted order. For models with identity transpose this is a no-op.
        """
        tf = self._bundle.transpose_forward
        if tf == (0, 1, 2):
            return volume_zyx
        return np.transpose(volume_zyx, axes=tf)

    def _apply_transpose_backward(self, predictions: np.ndarray) -> np.ndarray:
        """Permute model-order ``(K, *spatial)`` predictions back to canonical.

        The K axis stays at position 0; only the three spatial axes are
        permuted. For models with identity transpose this is a no-op.
        """
        tb = self._bundle.transpose_backward
        if tb == (0, 1, 2):
            return predictions
        # K (axis 0) is unchanged; spatial axes shift by 1.
        return np.transpose(predictions, axes=(0, tb[0] + 1, tb[1] + 1, tb[2] + 1))

    def predict(self, volume: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Run inference and return per-channel predictions as numpy.

        * Standard model, single fold  → raw logits, shape ``(K, Z, Y, X)``.
        * Standard model, multi fold   → averaged softmax probabilities.
        * Region-based, single fold    → per-region logits.
        * Region-based, multi fold     → averaged per-region sigmoid probs.

        Input ``volume`` is in canonical (Z, Y, X) numpy axis order. The
        engine internally applies ``transpose_forward`` from plans.json so
        the model sees data in its trained-on axis order, then applies
        ``transpose_backward`` to the output so the returned predictions
        are also in canonical axis order. For models with identity
        transpose (e.g., all of TS) this round-trip is a no-op.

        Use :meth:`predict_segmentation` to skip the post-processing branch
        — it handles all four cases and returns integer labels directly.
        Use :meth:`predict_logits` to receive the same data as ``mx.array``
        for chaining with MLX operations (``inverse_resample_argmax``,
        ``inverse_resample_paint``, multi-model arithmetic) without an
        explicit ``mx.array(...)`` wrap.
        """
        volume_model = self._apply_transpose_forward(volume)
        pred_model = self._backend.predict(volume_model, normalize=normalize)
        return self._apply_transpose_backward(pred_model)

    def predict_logits(self, volume: np.ndarray, normalize: bool = True) -> mx.array:
        """Run inference and return the per-channel predictions as ``mx.array``.

        Returns the same data as :meth:`predict` but wrapped as an
        ``mx.array`` in unified memory, ready to be passed directly to
        ``inverse_resample_argmax`` / ``inverse_resample_paint`` /
        ``mx.softmax`` / etc. without a per-caller ``mx.array(...)`` wrap.

        Input volume is in canonical (Z, Y, X) order; output logits are
        also in canonical (K, Z, Y, X) order (the internal
        ``transpose_forward`` / ``transpose_backward`` round-trip is
        handled by :meth:`predict`).

        Note: the sliding-window accumulator runs in numpy, so this method
        does not avoid the per-volume numpy materialization that happens
        inside the inference loop. The win is API ergonomics, not memory
        — equivalent to ``mx.array(engine.predict(volume))`` but expressed
        once where it makes architectural sense.
        """
        return mx.array(self.predict(volume, normalize=normalize))

    def predict_segmentation(
        self,
        volume: np.ndarray,
        normalize: bool = True,
        dtype: str | np.dtype | None = None,
    ) -> np.ndarray:
        """Run inference and return an integer segmentation map.

        Wraps :meth:`predict` with the correct post-processing for the
        bundle's label scheme:

        * Standard datasets → ``argmax(axis=0)`` on the predictor output.
        * Region-based datasets → per-region threshold + paint priority
          from ``regions_class_order``. Threshold is at 0 for single-fold
          (logits) and 0.5 for multi-fold (averaged sigmoid probs).

        Parameters
        ----------
        volume : np.ndarray
            Input shape ``(Z, Y, X)``.
        normalize : bool
            Apply the model's per-channel normalization before inference.
        dtype : str, np.dtype, or None
            Output integer dtype. ``None`` (default) auto-picks the smallest
            unsigned dtype that fits every label value in the dataset
            (``uint8`` / ``uint16`` / ``uint32``). Pass an explicit dtype
            to override.

        Returns
        -------
        np.ndarray
            Shape ``(Z, Y, X)``, integer dtype.
        """
        # Use self.predict to get the transpose round-trip (canonical-order in,
        # canonical-order out). For identity-transpose models this is a no-op.
        pred = self.predict(volume, normalize=normalize)
        if self._bundle.has_regions and len(self._bundle.fold_weights) > 1:
            # multi-fold ensemble already applied sigmoid before averaging
            threshold = 0.5
        else:
            # single-fold returns raw logits; argmax doesn't care about scale
            threshold = 0.0
        return convert_logits_to_segmentation(
            pred, self._bundle.dataset, threshold=threshold, dtype=dtype
        )

    @property
    def label_dtype(self) -> np.dtype:
        """Smallest unsigned integer dtype that fits this bundle's labels.

        What :meth:`predict_segmentation` would default to when called
        without an explicit ``dtype=`` override.
        """
        return label_dtype(self._bundle.dataset)

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
