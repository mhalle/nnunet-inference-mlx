# Surface mesh extraction (SurfaceNets from logits)

`postprocess.to_mesh(prediction)` extracts a **multi-material SurfaceNets dual
mesh** directly from the K-channel logit volume — no labelmap intermediate.
Vertex positions come from sub-voxel edge-crossing interpolation in the
continuous logit field; normals come from the gradient of the log-margin
field; quads carry the VTK `BoundaryLabels` convention so the result drops
straight into Slicer / VTK pipelines.

This document is the working primer. See `mesh.py`, `meshio.py`, and
`postprocess.py` docstrings for API-level detail; this page is the
"what to do, what not to do, and why" overlay.

---

## The recipe

```python
from nnunet_inference_mlx import (
    ModelStore, NiftiReader, TaskCatalog,
    infer, preprocess, postprocess,
    mesh_cleanup, mesh_to_vtk_polydata,
)
import vtk

# 1. Inference at the model's native training spacing
image = NiftiReader().read("scan.nii.gz")
spec = TaskCatalog("totalsegmentator").get("total_fast")     # single-model task
model = ModelStore("totalsegmentator").load(spec.single)
vol, _ = preprocess.to_model_frame(image, model.model_data, reorient_to="RAS")
prediction = infer.sliding_window(model, vol)

# 2. Mesh at desired output resolution
mesh = postprocess.to_mesh(
    prediction,
    # --- output resolution (optional) ---
    # scale=1.25,                       # 1.25× upsample
    # output_spacing_zyx=(1.2,1.2,1.2), # or specify mm directly
    # --- noise floor cleanup ---
    confidence_margin=1.0,            # spike-voxel edge filter
    drop_components_below_mm3=50.0,   # cc3d small-component drop
    # --- geometry refinement ---
    project_to_surface=True,          # Newton step onto decision surface
    emit_normals=True,                # field-gradient normals
)

# 3. Mesh-side polish (drop tiny disconnected regions + low-pass smooth)
mesh = mesh_cleanup(mesh)             # defaults: min_region_cells=200,
                                      #           smooth_iters=30,
                                      #           smooth_passband=0.05

# 4. Write a VTP for Slicer
pd = mesh_to_vtk_polydata(mesh)
w = vtk.vtkXMLPolyDataWriter()
w.SetFileName("scan_mesh.vtp"); w.SetInputData(pd); w.Write()
```

That recipe is the one the user endorsed after iterating through 10
variants on TS-fast and TS-full chest data. **Use this as the starting
point**; tune knobs only if you see specific problems.

---

## What each recipe knob does (in plain language)

| Knob | Default | What it does | When to change |
|---|---|---|---|
| `confidence_margin=1.0` | `0.0` (off) | Drops mesh edges incident to **isolated low-confidence voxels** (`n_same==0` AND `margin < threshold`). Non-destructive — labels unchanged. | Raise to 1.5–2.0 to catch slightly higher-margin spikes; set 0 if you see holes. |
| `drop_components_below_mm3=50.0` | `0.0` (off) | Runs `cc3d` connected components on the labelmap and drops any with physical volume below threshold. Only catches **truly disconnected** noise islands. | Raise to 200 for aggressive cleanup (TS default); set 0 for max recall. |
| `project_to_surface=True` | `False` | One Newton step per **binary-cell vertex** toward `L_i = L_j = 0`. Places vertex on the actual decision surface, not the centroid of its edge crossings. | Off if you want strict centroid placement. |
| `emit_normals=True` | `False` | Per-vertex normals from `∇(L_i − L_j)` (closed-form trilinear gradient). | Off if you'll re-bake via `mesh_compute_normals` after smoothing. |
| `scale` / `output_spacing_zyx` / `output_shape_zyx` | `None` | Activate the **streaming pipeline** to mesh at a different grid than the prediction's. At most one may be set. | Upsample for finer detail (memory permitting); downsample for previews. |
| `peak_working_memory_mb=1024` | 1024 | Per-slab budget for the streaming pipeline. | Raise on machines with more RAM; lower if memory pressure. |

`mesh_cleanup` adds three more knobs (`min_region_cells`, `smooth_iters`,
`smooth_passband`); the defaults are tuned for upsampled chest meshes.

---

## Memory model — IMPORTANT

The trilinear gather in `resample_prediction` (the whole-volume version)
materialises **8 K-channel intermediate arrays** during compute, so peak
memory is **~9× the output prediction size**. On M2 17 GB this is a
real ceiling — the failure mode below ~12 GB peak is a machine crash
(OS OOM-kills before MLX raises).

### `resample_prediction` (whole-volume, eager)

- Refuses any case whose peak would exceed `memory_ceiling_gb=12.0` (default)
- For chest TS-fast (K=118): only **downsamples** and **identity** fit; any meaningful upsample is refused
- Don't bump the ceiling unless you have ≥24 GB

### `to_mesh(prediction, scale=...)` and friends (streaming)

- Routes through `surfacenets_logits_at_target` — slab-streams the K-channel work, never holds the full upsampled volume
- For chest TS-full sub-tasks (K=19–27) at 1.25× upsample: peak ~10–11 GB, fits on M2 17 GB
- For chest TS-fast at 1.5× upsample: peak ~8 GB
- Memory is bounded by `peak_working_memory_mb` regardless of how big the output grid is

**Rule of thumb:** if you'd ever want to materialise the upsampled
prediction (vs just the mesh), use `resample_prediction`. Otherwise
use the `to_mesh(..., scale=...)` streaming path.

---

## Multi-task tasks (`label_union` shape — e.g. TS-full)

`TaskCatalog("totalsegmentator").get("total")` is a **`label_union` task**:
5 sub-models (organs, vertebrae, cardiac, muscles, ribs) at 1.5 mm,
unified into a 117-label namespace via `part.label_remap`.

The current `to_mesh` works on a single `Prediction`. For multi-task,
the pattern is **per-sub-task VTPs with global-namespace labels**, then
view together in Slicer. `mesh_concat` won't work across sub-tasks
because each sub-task has its own Geometry (different cropping) and
each `to_mesh` returns a Mesh with its sub-task's local schema. See
`examples/07_mesh_multitask.py` for the full pattern.

Sketch:

```python
spec = TaskCatalog("totalsegmentator").get("total")
for i, part in enumerate(spec.union):
    model = store.load(part.weights_id)
    vol, _ = preprocess.to_model_frame(image, model.model_data, reorient_to="RAS")
    prediction = infer.sliding_window(model, vol)

    mesh = postprocess.to_mesh(prediction, scale=1.25, confidence_margin=1.0,
                                drop_components_below_mm3=50.0,
                                project_to_surface=True, emit_normals=True)
    mesh = mesh_cleanup(mesh)

    # Remap sub-task labels (1..N) to global namespace (e.g. 25..50 for vertebrae)
    bl_remapped = remap_with(part.label_remap, mesh.boundary_labels)
    mesh = replace(mesh, boundary_labels=bl_remapped)

    write_vtp(mesh, f"out_{i+1}_{part.name}.vtp")
```

Caveat: each sub-task's mesh has "artifact" background surfaces where
voxels outside its own classes are labeled 0 (background). E.g. the
organs sub-task draws "rib-shaped background holes" because it doesn't
know about ribs. The fix (deferred) would composite all 5 logit
volumes into one 117-channel volume before extraction.

---

## Per-anatomical-label coloring in Slicer

Each quad's `boundary_labels` is `(Label0, Label1)` (background goes
in Label1 by VTK convention). To color a mesh by label in Slicer,
attach a single-component **scalar** to the polydata:

```python
import numpy as np, vtk
from vtk.util import numpy_support as ns

# Compute per-quad "label" scalar: non-bg value, or the smaller for non-bg pairs
bl = ns.vtk_to_numpy(pd.GetCellData().GetArray("BoundaryLabels")).reshape(-1, 2)
label0, label1 = bl[:, 0].astype(np.int32), bl[:, 1].astype(np.int32)
is_bg = (label0 == 0) | (label1 == 0)
scalar = np.where(is_bg, np.where(label0 == 0, label1, label0),
                          np.minimum(label0, label1)).astype(np.int32)

sca = ns.numpy_to_vtk(scalar, deep=True, array_type=vtk.VTK_INT)
sca.SetName("label")
pd.GetCellData().AddArray(sca)
pd.GetCellData().SetActiveScalars("label")
```

In Slicer:
1. Drag the VTP into the scene
2. Models module → expand the mesh → **Display** tab
3. **Scalars** → check "Active" → pick `label`
4. **Color table** → `Labels` (discrete categorical) or any palette
5. **Display range** → the label range of the sub-task (e.g. 92–117 for ribs)

For TS-full with all 5 sub-tasks in one scene, set the same color table
and display range **1–117** on all five — you get a unified palette
across the full anatomy.

---

## Performance characteristics (M2 Pro)

### TS-fast chest (K=118, native 3 mm spacing, ~22M voxels)

| Recipe | Time | Verts |
|---|---|---|
| `plain` to_mesh (no kwargs) | ~3.5 s | ~388k |
| + `project_to_surface` + `emit_normals` | ~4.0 s | ~388k |
| + `confidence_margin=1.0` | ~5.0 s | ~385k |
| + `mesh_cleanup` | +0.5 s | ~382k |
| Streaming **1.5× upsample** (scale=1.5) | ~29 s | ~824k |

### TS-full chest at 1.25× upsample, per sub-task

Each sub-task: ~5–6 min inference + ~1.5–2 min mesh (1.25× streaming).
Full 5-sub-task pipeline: **~38 min total**, peak memory ~10–11 GB,
~2.3M vertices total across 5 sub-task VTPs.

---

## What doesn't work / known limitations

| Issue | Status |
|---|---|
| Region-model meshing (BraTS-style sigmoid heads) | `to_mesh` raises `NotImplementedError` |
| Cross-sub-task mesh fusion | `mesh_concat` requires identical Geometry + identical Schema-by-identity; sub-tasks have different geometries. Workaround: per-sub-task VTPs (above) |
| 2× isotropic upsample of TS-fast on M2 17 GB | Streaming refuses (peak ~25 GB output, even slab-streamed reduces state to ~9 GB — tight) |
| Pyramid spikes at single-voxel argmax flips | Real artifact at noise-floor voxels. `confidence_margin` catches isolated cases; face-connected spike clusters need `mesh_cleanup` |
| `confidence_margin` with relaxed `n_same` (≤2) | **Tried and DEAD END** — puts holes in legitimate thin anatomy. Strict `n_same==0` is the right floor; use `mesh_cleanup` for the rest |
| Multi-component triple-junction vertex placement | Uses cell center (no zig-zag); pyramid pop-ups at multi-comp cells. Gauss-Newton on triple-line locus is the principled fix; deferred |
| `cc3d` only catches **disconnected** components | Face-attached spike clusters survive cc3d. Use `mesh_cleanup` for those |

---

## The output: where things live, what coords

- `Mesh.points` is `(N, 3)` float32 in **(Z, Y, X) order** — left-handed
  permutation of the usual (X, Y, Z). Exporters (`mesh_to_vtk_polydata`,
  `mesh_to_npz`, `mesh_smooth`, `mesh_cleanup`) reverse the order at
  write time. Don't call `np.cross` on these directly — the result is
  geometrically inverted.
- `Mesh.quads` is `(M, 4)` int32 vertex indices.
- `Mesh.boundary_labels` is `(M, 2)` int32 — `(Label0, Label1)` per VTK
  `vtkSurfaceNets3D` convention (background goes in `Label1`).
- `Mesh.normals` (if present) is `(N, 3)` float32 in (Z, Y, X) order;
  same swap rules apply.
- `Mesh.geometry` carries spacing, shape, origin, and direction — used
  by exporters to convert grid coords to world mm.
- `Mesh.schema` carries `names: {label_int: name_str}` so VTP cell-data
  arrays can be queried by name.

---

## Tests

51 mesh tests in `tests/test_mesh_*.py` — all passing. The streaming
path is exercised via the chest-scale workflows. There are NO synthetic
unit tests for streaming yet — if you change `surfacenets_logits_at_target`,
verify against a real prediction.

---

## Pointers to the implementation

| File | What's there |
|---|---|
| `src/nnunet_inference_mlx/mesh.py` | All algorithm helpers — `surfacenets_logits`, `surfacenets_logits_at_target`, `_argmax_and_margin`, `_edge_crossings`, `_cell_components`, `_cell_dual_vertices`, `_gradient_refine`, `_gradient_refine_at_source`, `_slab_stream_reduced`, `_compute_spike_mask` |
| `src/nnunet_inference_mlx/values.py` | `Mesh` value type |
| `src/nnunet_inference_mlx/postprocess.py` | `to_mesh`, `resample_prediction`, `_resolve_mesh_target_geometry` |
| `src/nnunet_inference_mlx/meshio.py` | npz IO, vtkPolyData export, `mesh_smooth`, `mesh_compute_normals`, `mesh_cleanup` |
| `src/nnunet_inference_mlx/labels.py` | `mesh_concat` (for the same-geometry use case) |
| `src/nnunet_inference_mlx/resampling.py` | `_kchannel_trilinear_full`, `_cascade_kchannel_to_target` (shared with `resample_prediction`) |
| `examples/06_mesh_output.py` | Single-task end-to-end |
| `examples/07_mesh_multitask.py` | Multi-task TS-full pattern |
