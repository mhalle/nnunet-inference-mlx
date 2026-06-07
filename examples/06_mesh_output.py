"""End-to-end logit-based surface extraction: CT → SurfaceNets mesh.

Loads a single-channel CT volume, runs a TotalSegmentator single-model
task at the network's training spacing, and emits a SurfaceNets dual
mesh whose vertex positions come from edge-crossing interpolation in the
continuous logit field (no labelmap stepping-stone, no smoothing — the
``feature/logit-mesh`` MVP).

    uv run --with vtk python examples/06_mesh_output.py CT.nii.gz mesh_out [TASK]

Produces ``mesh_out.npz`` (toolkit-canonical, lossless) and ``mesh_out.vtp``
(vtkXMLPolyData; drag into Slicer to visually validate).

``TASK`` defaults to ``"total_fast"`` (TS 3 mm, single Dataset297 model —
fastest end-to-end). Pass another single-model task name to exercise its
training-spacing mesh.
"""

from __future__ import annotations

import sys
from pathlib import Path

from nnunet_inference_mlx import (
    ModelStore,
    NiftiReader,
    TaskCatalog,
    infer,
    mesh_to_npz,
    mesh_to_vtk_polydata,
    postprocess,
    preprocess,
)


def main(inp: str, out_basename: str, task: str = "total_fast") -> None:
    store = ModelStore("totalsegmentator")
    catalog = TaskCatalog("totalsegmentator")
    spec = catalog.get(task)
    if spec.shape != "single":
        raise NotImplementedError(
            f"This example currently supports only single-model tasks; "
            f"{task!r} is a {spec.shape!r} task. Multi-task composite via "
            f"`mesh_concat` is the next phase."
        )

    print(f"loading {inp} ...", flush=True)
    image = NiftiReader().read(inp)
    print(f"  shape={image.shape_zyx}, spacing={image.geometry.spacing_zyx}", flush=True)

    print(f"loading task {task!r} (weights {spec.single}) ...", flush=True)
    model = store.load(spec.single)

    print("forward (reorient + resample to model spacing) ...", flush=True)
    model_vol, _plan = preprocess.to_model_frame(image, model.model_data,
                                                 reorient_to="RAS")
    print(f"  model-frame shape={model_vol.shape_zyx}, "
          f"spacing={model_vol.geometry.spacing_zyx}", flush=True)

    print("sliding-window inference ...", flush=True)
    prediction = infer.sliding_window(model, model_vol)
    print(f"  prediction (K, Z, Y, X) = {tuple(prediction.data.shape)} "
          f"({prediction.activation})", flush=True)

    print("SurfaceNets-from-logits ...", flush=True)
    mesh = postprocess.to_mesh(prediction)
    print(f"  mesh: {mesh.num_points} verts, {mesh.num_quads} quads", flush=True)

    npz_path = Path(out_basename).with_suffix(".npz")
    mesh_to_npz(mesh, npz_path)
    print(f"wrote {npz_path} (toolkit-canonical)", flush=True)

    vtp_path = Path(out_basename).with_suffix(".vtp")
    pd = mesh_to_vtk_polydata(mesh)
    import vtk
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path))
    writer.SetInputData(pd)
    writer.Write()
    print(f"wrote {vtp_path} (drag into Slicer)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
