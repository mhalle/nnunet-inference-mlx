"""End-to-end CT → SurfaceNets mesh, with the recommended cleanup recipe.

Single-model task only — for multi-model (label_union) tasks like
``ts:total``, see ``examples/07_mesh_multitask.py``.

Pipeline:
    image → preprocess.to_model_frame → infer.sliding_window
          → postprocess.to_mesh (with cleanup kwargs)
          → mesh_cleanup (mesh-side polish)
          → mesh_to_vtk_polydata → .vtp file

The recipe in ``RECIPE`` below is the one validated on chest TS-fast
and TS-full. See ``docs/mesh-pipeline.md`` for what each knob does and
when to tune it.

Usage:
    uv run --with vtk --with connected-components-3d \\
        python examples/06_mesh_output.py CT.nii.gz mesh_out [TASK] [SCALE]

Args:
    CT.nii.gz    Input CT volume.
    mesh_out     Output basename (`.vtp` and `.npz` files written).
    TASK         Single-model task name (default: total_fast).
    SCALE        Optional float >0: upsample/downsample factor at mesh
                 extraction. e.g. 1.25 for a 1.25× upsample (streaming).
                 Omit for native-spacing mesh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import vtk

from nnunet_inference_mlx import (
    ModelStore,
    NiftiReader,
    TaskCatalog,
    infer,
    mesh_cleanup,
    mesh_to_npz,
    mesh_to_vtk_polydata,
    postprocess,
    preprocess,
)


# The recommended kwargs for ``to_mesh`` on TS-style data. See
# docs/mesh-pipeline.md for what each one does.
RECIPE = dict(
    confidence_margin=1.0,            # drop edges incident to isolated low-margin voxels
    drop_components_below_mm3=50.0,   # cc3d-based small-component filter
    project_to_surface=True,          # Newton-step binary cells onto L_i = L_j
    emit_normals=True,                # field-gradient normals
)


def main(inp: str, out_basename: str, task: str = "total_fast",
         scale: float | None = None) -> None:
    store = ModelStore("totalsegmentator")
    catalog = TaskCatalog("totalsegmentator")
    spec = catalog.get(task)
    if spec.shape != "single":
        raise NotImplementedError(
            f"This example handles single-model tasks only; {task!r} is "
            f"a {spec.shape!r} task. For label_union tasks (e.g. ``total``), "
            f"see ``examples/07_mesh_multitask.py``."
        )

    print(f"loading {inp} ...", flush=True)
    image = NiftiReader().read(inp)
    print(f"  shape={image.shape_zyx}, spacing={image.geometry.spacing_zyx}",
          flush=True)

    print(f"loading task {task!r} (weights {spec.single}) ...", flush=True)
    model = store.load(spec.single)

    print("preprocess (reorient + resample to model spacing) ...", flush=True)
    model_vol, _plan = preprocess.to_model_frame(
        image, model.model_data, reorient_to="RAS",
    )
    print(f"  model-frame shape={model_vol.geometry.shape_zyx}, "
          f"spacing={model_vol.geometry.spacing_zyx}", flush=True)

    print("sliding-window inference ...", flush=True)
    prediction = infer.sliding_window(model, model_vol)
    print(f"  prediction (K, Z, Y, X) = {tuple(prediction.data.shape)} "
          f"({prediction.activation})", flush=True)

    kwargs = dict(RECIPE)
    if scale is not None:
        kwargs["scale"] = float(scale)
        print(f"SurfaceNets-from-logits at scale={scale} "
              f"(streaming, slab-based) ...", flush=True)
    else:
        print(f"SurfaceNets-from-logits at native spacing ...", flush=True)
    mesh = postprocess.to_mesh(prediction, **kwargs)
    print(f"  mesh: {mesh.num_points} verts, {mesh.num_quads} quads, "
          f"grid={mesh.geometry.shape_zyx} sp={mesh.geometry.spacing_zyx}",
          flush=True)

    print("mesh_cleanup (drop tiny isolated regions + low-pass smooth) ...",
          flush=True)
    mesh = mesh_cleanup(mesh)
    print(f"  cleaned: {mesh.num_points} verts, {mesh.num_quads} quads",
          flush=True)

    npz_path = Path(out_basename).with_suffix(".npz")
    mesh_to_npz(mesh, npz_path)
    print(f"wrote {npz_path} (toolkit-canonical)", flush=True)

    vtp_path = Path(out_basename).with_suffix(".vtp")
    pd = mesh_to_vtk_polydata(mesh)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path))
    writer.SetInputData(pd)
    writer.Write()
    print(f"wrote {vtp_path} (drag into Slicer)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    args = sys.argv[1:]
    inp, out = args[0], args[1]
    task = args[2] if len(args) > 2 else "total_fast"
    scale = float(args[3]) if len(args) > 3 else None
    main(inp, out, task, scale)
