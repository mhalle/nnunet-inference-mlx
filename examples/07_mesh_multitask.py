"""End-to-end CT → SurfaceNets meshes for a multi-task (label_union) task.

For label_union tasks (e.g. ``ts:total`` = organs ∪ vertebrae ∪ cardiac
∪ muscles ∪ ribs at 1.5 mm), we run each sub-model independently, mesh
each at the same scale, remap each sub-task's labels to the global
label namespace, and write a separate VTP per sub-task. Open all the
VTPs together in Slicer for the union view.

Why per-sub-task VTPs (not ``mesh_concat``): each sub-task's
preprocessing produces a slightly different Geometry (cropping/padding
differs by sub-model), and ``mesh_concat`` requires identical Geometry
+ identical Schema-by-identity. Per-sub-task VTPs sidestep both
constraints cleanly while still letting the user view the union.

Caveat: each sub-task's mesh has "artifact background surfaces" where
voxels outside its own class set are labelled 0 (background). E.g. the
organs sub-task draws a rib-shaped background hole because it doesn't
know about ribs, but the ribs sub-task draws the actual rib surface
in the same physical location. You'll see parallel sheets where two
sub-tasks share an anatomical boundary. A fully-fused mesh would
composite all sub-task logit volumes into one global K-channel volume
before extraction — deferred (would need a composite-logits API).

Usage:
    uv run --with vtk --with connected-components-3d \\
        python examples/07_mesh_multitask.py CT.nii.gz out_basename [TASK] [SCALE]

Args:
    CT.nii.gz      Input CT volume.
    out_basename   Path prefix; per-sub-task files are written as
                   ``{basename}_{i}_{name}.vtp``.
    TASK           ``label_union`` task name (default: ``total``).
    SCALE          Optional float; passed to ``to_mesh(..., scale=...)``
                   for memory-bounded upsampling. e.g. ``1.25`` upsamples
                   each sub-task's mesh to 1.25× its training spacing
                   (~1.2 mm for TS sub-tasks). Omit for native spacing.

Memory note (chest scan, TS sub-task at 1.25× scale):
    Source prediction: ~4 GB (K=24)
    Full-grid reduced state: ~5 GB
    Per-slab K-channel peak: ~1-2 GB
    Total per-sub-task peak: ~10-11 GB    (fits on M2 17 GB)
Drop scale (or omit it) if you see OOM signs.
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import vtk

from nnunet_inference_mlx import (
    ModelStore,
    NiftiReader,
    TaskCatalog,
    infer,
    mesh_cleanup,
    mesh_to_vtk_polydata,
    postprocess,
    preprocess,
)
from nnunet_inference_mlx.values import LabelSchema, Mesh


# The same recipe used in example 06.
RECIPE = dict(
    confidence_margin=1.0,
    drop_components_below_mm3=50.0,
    project_to_surface=True,
    emit_normals=True,
)


def _remap_boundary_labels(
    mesh: Mesh, label_remap: dict[int, int], global_schema: LabelSchema,
) -> Mesh:
    """Translate sub-task ``boundary_labels`` to the global TS label
    namespace and attach the shared global schema.

    Sub-task label 0 (background) stays 0 in the global namespace;
    non-background sub-task labels are looked up in ``part.label_remap``.
    """
    bl = mesh.boundary_labels
    lookup = np.zeros(int(max(label_remap.keys())) + 1, dtype=np.int32)
    for k, v in label_remap.items():
        lookup[int(k)] = int(v)
    flat = bl.ravel()
    nonzero = flat > 0
    remapped = flat.copy()
    remapped[nonzero] = lookup[flat[nonzero]]
    return replace(
        mesh,
        boundary_labels=remapped.reshape(bl.shape),
        schema=global_schema,
    )


def _write_vtp(mesh: Mesh, path: Path) -> None:
    pd = mesh_to_vtk_polydata(mesh)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(path)); w.SetInputData(pd); w.Write()


def main(
    inp: str, out_basename: str,
    task: str = "total", scale: float | None = None,
) -> None:
    print(f"=== {task} on {inp} ===\n", flush=True)

    print(f"loading image ...", flush=True)
    image = NiftiReader().read(inp)
    print(f"  shape={image.shape_zyx}, spacing={image.geometry.spacing_zyx}",
          flush=True)

    store = ModelStore("totalsegmentator")
    spec = TaskCatalog("totalsegmentator").get(task)
    if spec.shape != "label_union":
        raise ValueError(
            f"This example expects a label_union task; {task!r} has "
            f"shape={spec.shape!r}. For single-model tasks see example 06."
        )

    print(f"task {task!r}: label_union of {len(spec.union)} sub-models, "
          f"{len(spec.label_map)} global labels\n", flush=True)

    # Single shared schema so all sub-meshes carry the same global namespace.
    global_schema = LabelSchema(names=spec.label_map)

    t_total = time.time()
    for i, part in enumerate(spec.union):
        print(f"--- Part {i+1}/{len(spec.union)}: {part.name} "
              f"(Dataset{part.weights_id}) ---", flush=True)
        t = time.time()
        model = store.load(part.weights_id)
        print(f"  weights:     {time.time() - t:5.1f}s", flush=True)

        t = time.time()
        model_vol, _ = preprocess.to_model_frame(
            image, model.model_data, reorient_to="RAS",
        )
        print(f"  preprocess:  {time.time() - t:5.1f}s  "
              f"shape={model_vol.geometry.shape_zyx} "
              f"sp={tuple(round(s, 2) for s in model_vol.geometry.spacing_zyx)}",
              flush=True)

        t = time.time()
        prediction = infer.sliding_window(model, model_vol)
        mx.eval(prediction.data)
        K = int(prediction.data.shape[0])
        size_gb = (K * prediction.data.shape[1] * prediction.data.shape[2]
                   * prediction.data.shape[3] * 4) / 1e9
        print(f"  inference:   {time.time() - t:5.1f}s  K={K}  "
              f"pred-size={size_gb:.2f}GB", flush=True)

        kwargs = dict(RECIPE)
        if scale is not None:
            kwargs["scale"] = float(scale)

        t = time.time()
        mesh = postprocess.to_mesh(prediction, **kwargs)
        print(f"  to_mesh:     {time.time() - t:5.1f}s  "
              f"verts={mesh.num_points} quads={mesh.num_quads}  "
              f"grid={mesh.geometry.shape_zyx}", flush=True)

        # Remap boundary_labels to the global namespace, attach the shared
        # schema, run mesh_cleanup, write.
        mesh = _remap_boundary_labels(mesh, part.label_remap, global_schema)

        t = time.time()
        mesh = mesh_cleanup(mesh)
        out = Path(f"{out_basename}_{i+1}_{part.name}.vtp")
        _write_vtp(mesh, out)
        print(f"  cleanup+wr:  {time.time() - t:5.1f}s  "
              f"verts={mesh.num_points} quads={mesh.num_quads}  → {out}",
              flush=True)

        # Free per-sub-task memory before the next iteration.
        del model, prediction, model_vol, mesh
        gc.collect()
        mx.metal.clear_cache()

    print(f"\n*** TOTAL: {time.time() - t_total:.1f}s ***", flush=True)
    print(f"VTPs written under {out_basename}_*_*.vtp — drag all into Slicer "
          f"for the union view.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    args = sys.argv[1:]
    inp, out = args[0], args[1]
    task = args[2] if len(args) > 2 else "total"
    scale = float(args[3]) if len(args) > 3 else None
    main(inp, out, task, scale)
