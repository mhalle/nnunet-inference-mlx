"""Logits are first-class; output resolution is a knob.

``LoadedModel.predict`` returns a :class:`Prediction` — the per-class logit
volume at the model's native spacing — *before* the lossy inverse resample. You
can branch on it (uncertainty, thresholding, multi-model arithmetic) or convert
it. ``segment``/``LoadedModel.segment`` render labels back to the input grid by
resampling the *logits* then argmax (higher quality than NN-resampling labels),
and can render at any resolution.

    uv run python examples/03_logits_and_resolution.py scan.nii.gz seg.nii.gz

Extras for the small-component cleanup: ``--extra postprocessing``.
"""

from __future__ import annotations

import sys

from nnunet_inference_mlx import ModelStore, NiftiReader, NiftiWriter, postprocess


def main(inp: str, out: str) -> None:
    store = ModelStore("totalsegmentator")
    model = store.load(297)                       # total_fast (3 mm)
    vol = NiftiReader().read(inp)

    # First-class logits at the model's native spacing.
    pred = model.predict(vol)                     # → Prediction(K, Z, Y, X), activation="logits"
    print(f"logits {tuple(pred.data.shape)} @ {pred.geometry.spacing_zyx} mm ({pred.activation})")

    # Same-grid labels straight from the logits (no resample):
    coarse = postprocess.to_labels(pred)
    print(f"labels at model spacing: {coarse.geometry.shape_zyx}")

    # Full pipeline back to the input grid, with dust cleanup and a finer output grid.
    seg = model.segment(
        vol,
        remove_small_components_mm3=200.0,        # drop <0.2 ml blobs (needs [postprocessing])
        output_scaling=1.0,                       # 1=input grid; 2=finer; 0.5=coarser; or output_spacing=MM
    )
    NiftiWriter().write(out, seg)
    print(f"wrote {out}: {seg.geometry.shape_zyx} @ {seg.geometry.spacing_zyx} mm")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
