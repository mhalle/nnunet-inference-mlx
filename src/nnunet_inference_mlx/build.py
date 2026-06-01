"""build_model — the one place GPU state is allocated.

``build_model(model_data, options)`` compiles a :class:`ModelData` into a
:class:`LoadedModel`: the runnable form of a model, with weights resident and
the network compiled. ``LoadedModel`` exposes ``.predict(volume)`` /
``.segment(volume)`` (and the metadata views), plus ``.memory_mb`` and
``.close()`` so a :class:`ModelStore` can size and free it.

``predict`` / ``segment`` compose the toolkit stages —
``preprocess.to_model_frame → infer.sliding_window`` (→ ``postprocess.restore``
for ``segment``) — so the single-model path *is* the decomposed pipeline. The
compute core (sliding window, normalization, transpose, fold ensemble,
region/argmax) still lives in ``InferenceEngine``; at the Phase 5 cutover that
machinery is rehomed under ``infer`` and the old facade is removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .model_data import ModelData
from .values import BuildOptions, LabelSchema, Prediction, Segmentation, Volume

if TYPE_CHECKING:
    from .engine import InferenceEngine


class LoadedModel:
    """A model loaded and ready to run (compiled network + resident weights).

    Returned by :func:`build_model` and by ``ModelStore.load``. Holds GPU
    state — bounded and freed by its owning store, or by ``.close()`` /
    ``with`` if you built it directly.
    """

    def __init__(self, engine: "InferenceEngine", model_data: ModelData,
                 *, memory_mb: float):
        self._engine: "InferenceEngine | None" = engine
        self.model_data = model_data
        self._memory_mb = float(memory_mb)

    # ----- metadata views -----
    @property
    def config(self) -> ModelData:
        return self.model_data

    @property
    def schema(self) -> LabelSchema:
        return self.model_data.schema

    @property
    def memory_mb(self) -> float:
        return self._memory_mb

    # ----- run -----
    def predict(
        self,
        volume: Volume,
        *,
        reorient_to: str | None = "RAS",
        interpolation: str = "auto",
        step_size: float = 0.5,
        use_mirroring: bool = False,
    ) -> Prediction:
        """Per-class model output at the model's native (training) spacing.

        Logits have been first-class since the engine's ``predict_logits``
        primitive — this surfaces them as a :class:`Prediction` value.
        Stops *before* the inverse resample back to the input grid (the lossy
        trip), so a caller can branch on the raw K-channel surface: uncertainty
        maps, multi-model arithmetic, custom thresholding, the sub-voxel logit
        render. ``segment`` is this plus restore.

        Composed from the toolkit stages: ``preprocess.to_model_frame`` then
        ``infer.sliding_window``. ``activation`` records what the values are:
        ``"logits"`` for a single fold; ``"softmax"`` (standard) or
        ``"sigmoid"`` (region) for a fold-ensembled model.
        """
        if self._engine is None:
            raise RuntimeError("LoadedModel has been closed")
        from .infer import sliding_window
        from .preprocess import to_model_frame

        model_vol, _plan = to_model_frame(
            volume, self.model_data,
            reorient_to=reorient_to, interpolation=interpolation,
        )
        return sliding_window(self, model_vol,
                              step_size=step_size, use_mirroring=use_mirroring)

    def segment(
        self,
        volume: Volume,
        *,
        reorient_to: str | None = "RAS",
        interpolation: str = "auto",
        peak_working_memory_mb: int | None = None,
        remove_small_components_mm3: float = 0.0,
        step_size: float = 0.5,
        use_mirroring: bool = False,
        output_spacing: "float | tuple[float, float, float] | None" = None,
        output_scaling: float | None = None,
        at_model_spacing: bool = False,
        output_interpolation: str = "linear",
    ) -> Segmentation:
        """Segment a single-channel :class:`Volume` → :class:`Segmentation`.

        The full pipeline, composed from the toolkit stages:
        ``to_model_frame → sliding_window → restore``.

        ``output_interpolation`` controls the inverse resample: ``"linear"``
        (default) interpolates the logits then argmax (higher fidelity, like
        nnU-Net); ``"nearest"`` argmaxes at model spacing then NN-resamples the
        label map (much faster on large grids, TS-style, stair-stepped).

        Output resolution (mutually exclusive; default = the input grid):
        ``output_spacing`` (absolute mm, scalar or (Z,Y,X)), ``output_scaling``
        (resolution multiplier; 2 = finer, 0.5 = coarser), or
        ``at_model_spacing`` (the model's native training grid). The result
        always stays in the input's orientation/space; only sampling changes.
        """
        if self._engine is None:
            raise RuntimeError("LoadedModel has been closed")
        from .infer import sliding_window
        from .postprocess import drop_small_components, restore
        from .preprocess import to_model_frame

        if sum(x is not None and x is not False for x in
               (output_spacing, output_scaling, at_model_spacing or None)) > 1:
            raise ValueError("pass at most one of output_spacing / output_scaling / at_model_spacing")

        model_vol, plan = to_model_frame(
            volume, self.model_data,
            reorient_to=reorient_to, interpolation=interpolation,
        )
        prediction = sliding_window(self, model_vol,
                                    step_size=step_size, use_mirroring=use_mirroring)
        target_spacing = self.model_data.target_spacing_zyx if at_model_spacing else output_spacing
        segmentation = restore(prediction, plan,
                               target_spacing=target_spacing,
                               target_scaling=output_scaling,
                               interpolation=output_interpolation,
                               peak_working_memory_mb=peak_working_memory_mb)
        if remove_small_components_mm3 > 0:
            segmentation = drop_small_components(
                segmentation, min_volume_mm3=remove_small_components_mm3,
                in_place=True,
            )
        return segmentation

    # ----- lifecycle -----
    def close(self) -> None:
        """Release the GPU/compute state. Idempotent."""
        self._engine = None
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    def __enter__(self) -> "LoadedModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_model(
    model_data: ModelData,
    options: BuildOptions = BuildOptions(),
    *,
    step_size: float = 0.5,
    use_mirroring: bool = False,
) -> LoadedModel:
    """Compile :class:`ModelData` into a runnable :class:`LoadedModel`.

    The single allocation point for GPU state in the toolkit.

    ``options`` (:class:`BuildOptions`) holds the build-identity knobs (and is
    the model store's cache key). ``step_size`` / ``use_mirroring`` are
    *run* knobs — they don't change what's built, so they're plain kwargs here
    (not part of identity). They're applied at construction for now because the
    current engine bakes them; once the engine internals are rehomed they move
    to per-call ``segment`` arguments.
    """
    from .engine import InferenceEngine, ModelBundle

    # Carry the real on-disk checkpoint metadata (mirroring axes, init_args) and
    # ensure the resolved configuration is present. ModelData already picked the
    # config (e.g. "3d_fullres"); without it the engine and ``bundle.target_spacing``
    # fall back to the *first* config in plans.json — "2d" for TS part models
    # (a 2-element spacing). We set it only if the metadata doesn't already carry it.
    configuration = options.configuration or model_data.config_name
    init_args = {**(model_data.metadata.get("init_args") or {})}
    init_args.setdefault("configuration", configuration)
    metadata = {**model_data.metadata, "init_args": init_args}
    bundle = ModelBundle(
        plans=dict(model_data.plans),
        dataset=dict(model_data.dataset),
        fold_weights=list(model_data.fold_weights),
        metadata=metadata,
        fold_ids=tuple(range(model_data.num_folds)),
    )
    engine = InferenceEngine(
        bundle,
        configuration=configuration,
        step_size=step_size,
        compile=options.compile,
        batch_size=options.batch_size,
        use_mirroring=use_mirroring,
        verbose=False,
    )
    # Resident footprint ≈ weights across folds (the dominant term).
    memory_mb = model_data.weights_mb * max(1, model_data.num_folds)
    return LoadedModel(engine, model_data, memory_mb=memory_mb)


__all__ = ["LoadedModel", "build_model"]
