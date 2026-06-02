"""infer — run a built model over a model-frame volume → Prediction.

``sliding_window(model, volume)`` is the compute step: it runs the loaded
model's sliding-window inference on a model-frame :class:`Volume` (already
reoriented + resampled by ``preprocess.to_model_frame``) and returns a
:class:`Prediction` (per-class logits/probabilities) at the same geometry.

``activation`` records what the values are: ``"logits"`` for a single fold,
or the ensembled ``"softmax"`` (standard) / ``"sigmoid"`` (region) for a
multi-fold model — matching what the engine actually returns.

``step_size`` / ``use_mirroring`` are accepted per call. Until the engine's
compute internals are rehomed under this namespace (Phase 5), they are applied
by overriding the loaded model's sliding-window engine for the duration of the
call (the model owns its engine and calls are sequential), then restored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .values import Prediction, Volume

if TYPE_CHECKING:
    from .build import LoadedModel


def sliding_window(
    model: "LoadedModel",
    volume: Volume,
    *,
    step_size: float = 0.5,
    use_mirroring: bool = False,
    batch_size: int | None = None,
) -> Prediction:
    """Run the model over a (single-channel) model-frame volume → Prediction.

    The engine normalizes and applies ``transpose_forward``/``backward``
    internally; the returned :class:`Prediction` is in the volume's geometry
    and canonical ``(K, Z, Y, X)`` axis order.
    """
    engine = model._engine
    if engine is None:
        raise RuntimeError("LoadedModel has been closed")

    if volume.num_channels != 1:
        raise NotImplementedError(
            f"infer.sliding_window handles single-channel volumes; got "
            f"{volume.num_channels} channels {volume.channels}"
        )
    vol_np = np.asarray(volume.data[..., 0]).astype(np.float32, copy=False)

    sw = engine.sliding_window  # the SlidingWindowEngine (public accessor)
    saved = (sw.step_size, sw.use_mirroring, sw._batch_size)
    sw.step_size, sw.use_mirroring = float(step_size), bool(use_mirroring)
    if batch_size is not None:                        # else keep auto-chosen default
        sw._batch_size = int(batch_size)
    try:
        logits = engine.predict_logits(vol_np)        # (K, Z, Y, X) mx.array
    finally:
        sw.step_size, sw.use_mirroring, sw._batch_size = saved

    if model.model_data.num_folds > 1:
        activation = "sigmoid" if model.schema.is_region_model else "softmax"
    else:
        activation = "logits"

    return Prediction(
        data=logits,
        geometry=volume.geometry,
        schema=model.schema,
        activation=activation,
    )


__all__ = ["sliding_window"]
