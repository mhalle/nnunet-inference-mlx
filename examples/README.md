# Examples

Runnable scripts for the toolkit API. Each is self-contained — read the
docstring at the top. Run with `uv run` so dependencies are present.

| Script | What it shows |
|---|---|
| [`01_single_volume.py`](01_single_volume.py) | NIfTI in, NIfTI out — `ModelStore` + `segment`, the minimal call |
| [`02_batch_folder.py`](02_batch_folder.py) | A folder of CTs reusing one resident model (the `ModelStore` *is* the cache — bounded, owned, no globals) |
| [`03_logits_and_resolution.py`](03_logits_and_resolution.py) | First-class logits (`LoadedModel.predict` → `Prediction`), `postprocess` conversions, and output-resolution control |
| [`04_cascade_and_union.py`](04_cascade_and_union.py) | Multi-model tasks — the TS `total` label-union (and cascade recipes) via the same `segment` one-liner |
| [`05_toolkit_namespaces.py`](05_toolkit_namespaces.py) | Compose the pipeline by hand: `preprocess.to_model_frame → infer.sliding_window → postprocess.restore` |
| [`06_mesh_output.py`](06_mesh_output.py) | **CT → SurfaceNets mesh** (single-task) with the recommended cleanup recipe (`confidence_margin`, `cc3d`, `project_to_surface`, `mesh_cleanup`). Optional `scale` arg for memory-bounded upsampling. See [`docs/mesh-pipeline.md`](../docs/mesh-pipeline.md) for the design |
| [`07_mesh_multitask.py`](07_mesh_multitask.py) | **Multi-task TS-full meshes** — per-sub-task VTPs at the global TS-117 label namespace. The pattern for `label_union` tasks where `mesh_concat` doesn't apply |

## Running them

```bash
# from the repo root — uv installs everything (mlx, SimpleITK, …)
uv run python examples/01_single_volume.py scan.nii.gz seg.nii.gz
```

For the common cases there are also two CLIs (no script needed):

```bash
uv run nnmlx segment total_fast scan.nii.gz seg.nii.gz       # native CLI
uv run TotalSegmentator -i scan.nii.gz -o segmentations       # TotalSegmentator drop-in
```

## Where do the weights come from?

The `ModelStore` resolves a model root from (in precedence order) the
`model_root_dir=` argument → ecosystem env vars (`$TOTALSEG_WEIGHTS_PATH`,
`$nnUNet_results`) → the built-in default `~/.totalsegmentator/nnunet/results`.
Download TotalSegmentator weights once via the real TS CLI (or any tool that
populates that directory):

```bash
TotalSegmentator -i scan.nii.gz -o /tmp/out --fast   # downloads task 297 (total_fast)
TotalSegmentator -i scan.nii.gz -o /tmp/out          # downloads 291-295 (total union)
```

Then load by id or by task name:

```python
from nnunet_inference_mlx import ModelStore, TaskCatalog, NiftiReader, segment
store, catalog = ModelStore("totalsegmentator"), TaskCatalog("totalsegmentator")
seg = segment("total_fast", NiftiReader().read("scan.nii.gz"), store=store, catalog=catalog)
```
