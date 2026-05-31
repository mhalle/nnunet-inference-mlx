"""Compose the pipeline by hand from the pure-fn stage namespaces.

``segment`` / ``LoadedModel.segment`` are conveniences over four composable,
no-hidden-state namespaces — each a plain function over value types:

    preprocess.to_model_frame(volume, model_data) -> (Volume, RestorePlan)
    infer.sliding_window(model, volume)           -> Prediction
    postprocess.restore(prediction, plan)         -> Segmentation   (logit-resample → input grid)
    postprocess.to_labels(prediction)             -> Segmentation   (argmax at model grid)
    geometry.bbox_of_labels / crop / paste        -> FOV ops (Volume/Segmentation-native)

This is the seam for custom pipelines (uncertainty, ensembling, manual FOV
cropping, sub-volume inference) without forking the facade.

    uv run python examples/05_toolkit_namespaces.py scan.nii.gz seg.nii.gz
"""

from __future__ import annotations

import sys

from nnunet_inference_mlx import (
    ModelStore, NiftiReader, NiftiWriter, infer, postprocess, preprocess,
)


def main(inp: str, out: str) -> None:
    store = ModelStore("totalsegmentator")
    model = store.load(297)
    vol = NiftiReader().read(inp)

    # The same three steps `LoadedModel.segment` performs, made explicit:
    model_vol, plan = preprocess.to_model_frame(vol, model.model_data)  # reorient(RAS)+resample
    prediction = infer.sliding_window(model, model_vol)                 # → Prediction (logits)
    seg = postprocess.restore(prediction, plan)                        # → input grid

    NiftiWriter().write(out, seg)
    print(f"wrote {out}: {seg.geometry.shape_zyx}")
    print("plan:", plan.inference_orientation, "→", plan.source_orientation,
          "| model spacing", plan.model_spacing_zyx)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
