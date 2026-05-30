"""build_model — the one place GPU state is allocated.

``build_model(model_data, options)`` compiles a :class:`ModelData` into a
:class:`LoadedModel`: the runnable form of a model, with weights resident and
the network compiled. ``LoadedModel`` exposes ``.segment(volume)`` (and the
metadata views), plus ``.memory_mb`` and ``.close()`` so a :class:`ModelStore`
can size and free it.

During migration the heavy lifting (sliding window, normalization, transpose
+ orientation, fold ensemble, region/argmax, resampling) is reused from the
proven ``InferenceEngine`` + ``predict_with_resampling`` path; at cutover that
internal machinery is rehomed under here and the old facade is removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .model_data import ModelData
from .values import BuildOptions, LabelSchema, Segmentation, Volume

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
    def segment(
        self,
        volume: Volume,
        *,
        reorient_to: str | None = "LPS",
        peak_working_memory_mb: int | None = None,
        remove_small_components_mm3: float = 0.0,
    ) -> Segmentation:
        """Segment a single-channel :class:`Volume` → :class:`Segmentation`.

        Reuses the full forward-resample → infer → inverse-resample path with
        orientation/transpose handling; the result is in the input's geometry.
        """
        if self._engine is None:
            raise RuntimeError("LoadedModel has been closed")
        from .imageio import sitk_to_segmentation, volume_to_sitk
        from .resampling import predict_with_resampling

        seg_sitk = predict_with_resampling(
            self._engine,
            volume_to_sitk(volume),
            reorient_to=reorient_to,
            peak_working_memory_mb=peak_working_memory_mb,
            remove_small_components_mm3=remove_small_components_mm3,
        )
        return sitk_to_segmentation(seg_sitk, self.schema)

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

    bundle = ModelBundle(
        plans=dict(model_data.plans),
        dataset=dict(model_data.dataset),
        fold_weights=list(model_data.fold_weights),
        metadata={},
        fold_ids=tuple(range(model_data.num_folds)),
    )
    engine = InferenceEngine(
        bundle,
        configuration=options.configuration,
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
