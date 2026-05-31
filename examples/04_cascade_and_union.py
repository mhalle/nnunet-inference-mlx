"""Multi-model tasks: cascade (crop FOV) and label-union — same one-liner.

A task's *shape* (single / cascade / label_union) is in its recipe; ``segment``
dispatches on it. The TotalSegmentator ``total`` task is a 5-part label-union
(organs/vertebrae/cardiac/muscles/ribs) folded into one 117-class output. A
cascade task runs a coarse model, crops to a target region, then a fine model
inside that box and pastes back — all driven by the recipe.

    uv run python examples/04_cascade_and_union.py scan.nii.gz seg.nii.gz [TASK]

Default TASK is "total" (the full 1.5 mm union — slower, ~5 part models).
"""

from __future__ import annotations

import sys

from nnunet_inference_mlx import (
    CascadeStep, ModelStore, NiftiReader, NiftiWriter, TaskCatalog, TaskSpec, segment,
)


def main(inp: str, out: str, task: str = "total") -> None:
    store = ModelStore("totalsegmentator", max_memory_mb=16000)
    catalog = TaskCatalog("totalsegmentator")
    image = NiftiReader().read(inp)

    spec = catalog.get(task)
    print(f"task {spec.qualified_name}: shape={spec.shape}")

    # `segment` runs the union (or cascade) transparently, reporting progress.
    seg = segment(spec, image, store=store, catalog=catalog,
                  progress=lambda m: print(f"  {m}"))
    NiftiWriter().write(out, seg)
    print(f"wrote {out}: {seg.geometry.shape_zyx}")

    # You can also build a recipe inline instead of looking one up — e.g. a cascade:
    #   spec = TaskSpec(name="x", source="ts", modality="CT", shape="cascade",
    #                   cascade=(CascadeStep(weights_id=298, crop_to_classes=(...,)),
    #                            CascadeStep(weights_id=297)),
    #                   label_map={...})
    #   segment(spec, image, store=store)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
