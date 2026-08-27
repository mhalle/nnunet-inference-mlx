# nnunet-inference-mlx

MLX inference for [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) models on Apple Silicon. Runs trained nnU-Net checkpoints natively on Metal with no PyTorch dependency at runtime.

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is a self-configuring medical-image segmentation framework that consistently achieves state-of-the-art results across biomedical datasets. This package brings those trained models to Mac users with native Metal acceleration. It also integrates with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (the widely-used CT segmentation tool with ~117 anatomical classes).

> This is alpha-level software. We aim to demonstrate MLX's value for nnU-Net inference on Apple Silicon, not to maintain a long-term fork — the goal is for these ideas to flow back upstream.

## Highlights

- **Torch-free `.pth` loading.** Pretrained nnU-Net checkpoints are read directly via a vendored restricted unpickler. No PyTorch at runtime, no separate conversion step.
- **Layered inference architecture.** `Predictor` → `SlidingWindowEngine` → `FoldEnsemble` → `InferenceEngine` — compose what you need. Single-patch interactive use bypasses the wrapper; batch use composes the whole stack.
- **Process-wide engine cache.** Auto-tiered by detected RAM (≥32 GB on by default). ~1500× speedup on warm fetch — batch and cascade workflows pay the load cost once per process.
- **Multi-stage cascades** via `run_workflow`. MOOSE-style body-detector → high-res organ-specific patterns in ~5 lines of user code. FOV cropping between stages with automatic geometry preservation.
- **Path-B inverse resampling.** Trilinear interpolation on the K-channel logit volume with argmax-at-the-end, slab-streamed in unified memory. Smoother boundaries than label-NN, affordable at K≥100 because we slab.
- **Region-based labels** (BraTS-style sigmoid heads with paint-priority order). Auto-detected from `dataset.json`.
- **Multi-label component cleanup** via [cc3d](https://github.com/seung-lab/connected-components-3d). ~10× faster than SITK, ~90× faster than scipy.
- **Surface mesh extraction from logits.** Multi-material SurfaceNets dual mesh straight from the K-channel logit volume — no labelmap step. Vertex positions from sub-voxel logit-field interpolation, gradient-field normals, slab-streamed upsampling to arbitrary output resolution. See [docs/mesh-pipeline.md](docs/mesh-pipeline.md).
- **Pluggable weights-folder discovery** (`WeightsLayout` registry). Built-in layouts for nnU-Net and TotalSegmentator; downstream packages register their own.
- **6× faster than PyTorch CPU**, ~1.4× faster than PyTorch MPS on the same hardware. Identical voxel output to upstream nnU-Net.

## Installation

```bash
uv add nnunet-inference-mlx              # recommended (uv)
pip install nnunet-inference-mlx         # or pip
```

`SimpleITK` is a core dependency (image I/O + resampling are part of
segmenting); `uv run` installs everything needed. Optional extras add only
peripheral features:

| Extra | What it pulls in | Enables |
|---|---|---|
| `[postprocessing]` | `connected-components-3d` | `segment --remove-small-components-mm3` (multi-label dust) |
| `[remote]` | `httpx` | Remote weight download (`store.download`) + HTTP-range `.pth` loading |
| `[test]` | `pytest` | Running the test suite |

```bash
pip install 'nnunet-inference-mlx[preprocessing,postprocessing]'
```

Extras are **role-scoped**: torch and the inference stack live in the `torch` extra (and
the engine extras), so `import nnseg` is itself **torch-free** — a describe-only front-end
or a task listing never pays for a multi-GB CUDA torch, and each deployment image installs
only what its role needs. Contributors: the rules that keep imports and images lean are in
[docs/dependency-discipline.md](docs/dependency-discipline.md) (enforced by
`tests/test_nnseg_layering.py`).

## Quick start

```python
from nnunet_inference_mlx import ModelBundle, InferenceEngine, predict_nifti

# Load a model (auto-discovers TS weights from ~/.totalsegmentator/nnunet/results,
# nnU-Net weights from $nnUNet_results)
bundle = ModelBundle.from_task(297, folds=0)
engine = InferenceEngine(bundle)

# Run inference, write the segmentation
predict_nifti(engine, "scan.nii.gz", "seg.nii.gz")
```

See [`examples/`](examples/) for batch, cascade, and path-B variants.

## Common workflows

### Single-volume inference

```python
from nnunet_inference_mlx import ModelBundle, InferenceEngine, predict_nifti

engine = InferenceEngine(ModelBundle.from_task(297, folds=0))
predict_nifti(engine, "scan.nii.gz", "seg.nii.gz")
```

### Batch processing with engine cache

```python
from pathlib import Path
from nnunet_inference_mlx import cached_engine_from_task, predict_nifti

engine = cached_engine_from_task(297, folds=0)  # loaded once
for ct in Path("scans/").glob("*.nii.gz"):
    predict_nifti(engine, ct, f"out/{ct.stem}_seg.nii.gz")
# Engine survives across calls; load cost amortized to once per process.
```

### Path-B inverse resampling with SITK

```python
import SimpleITK as sitk
from nnunet_inference_mlx import cached_engine_from_task, predict_with_resampling

engine = cached_engine_from_task(297, folds=0)
img = sitk.ReadImage("scan.nii.gz")
seg = predict_with_resampling(engine, img,
                              remove_small_components_mm3=200.0)
sitk.WriteImage(seg, "seg.nii.gz")
```

`predict_with_resampling` handles forward resample (CPU/SITK) → inference (Metal) → inverse resample (Metal, path-B trilinear + argmax with slab streaming) → optional cc3d cleanup, all in one call. Output geometry matches input.

### Two-stage cascade

```python
import SimpleITK as sitk
from nnunet_inference_mlx import (
    Stage, run_workflow, cached_engine_from_task,
)

body  = cached_engine_from_task(298, folds=0)              # low-res body detector
liver = cached_engine_from_task(291, folds=0,              # high-res organ model
                                trainer="nnUNetTrainerNoMirroring")

stages = [
    Stage(engine=body,  crop_to_classes=(BODY_TRUNK,), dilation_mm=10.0),
    Stage(engine=liver, remove_small_components_mm3=200.0),
]

seg = run_workflow(sitk.ReadImage("scan.nii.gz"), stages, verbose=True)
sitk.WriteImage(seg, "seg.nii.gz")
```

The second stage runs only inside the cropped FOV (bbox of `BODY_TRUNK` from the first stage). Output is pasted back into the original geometry.

### Region-based / BraTS-style models

Models whose `dataset.json` declares regions (lists of underlying classes per output) are detected automatically. `InferenceEngine.predict_segmentation()` applies the right post-processing:

```python
engine = InferenceEngine(ModelBundle.from_folder(brats_folder))
seg = engine.predict_segmentation(volume)
# Standard models: argmax. Region models: per-region sigmoid threshold +
# paint priority via dataset.json's `regions_class_order`.
```

## Architecture

The package is split into composable layers so consumers can pick the right entry point:

```
ModelBundle   ── plans + dataset + N fold weights, pure I/O artifact
    │
    ▼
Predictor                 ── one compiled MLX network, weight-swappable
    │                        (nnInteractive-style single-patch use sits here)
    ▼
SlidingWindowEngine       ── Gaussian-weighted sliding window
    │
    ▼
FoldEnsemble (optional)   ── softmax/sigmoid averaging across folds
    │
    ▼
InferenceEngine           ── back-compat one-call facade
```

Higher-level building blocks compose on top:

- `predict_nifti` / `predict_folder` — NIfTI I/O around an engine
- `predict_with_resampling` — full path-B pipeline (SITK in, SITK out)
- `run_workflow` — multi-stage cascade orchestrator with FOV cropping
- `cached_engine_from_task` / `cached_engine_from_folder` — process-wide engine cache

The geometric primitives (`Bbox`, `compute_fg_bbox`, `crop_image`, `paste_segmentation`) are exported as public API for bespoke pipelines (nnInteractive sub-volumes, manual FOV limiting).

## Auto-tiering by RAM

A few places adapt automatically to detected unified memory:

| Knob | < 32 GB Mac | ≥ 32 GB Mac |
|---|---|---|
| `Predictor.cache_limit_fraction` | 0.30 | 0.50 |
| `inverse_resample_argmax` slab budget | 200 MB | 2000 MB |
| Engine cache enabled? | off | on |

Override any of them explicitly when the heuristic doesn't match your workload (`NNUNET_MLX_CACHE_ENGINES=1` to force the cache on, `peak_working_memory_mb=N` to set the inverse-resample slab budget, etc.).

## Weights

Pretrained TotalSegmentator `.pth` checkpoints load directly — no conversion step. The `WeightsLayout` registry auto-discovers:

- `$nnUNet_results` (standard nnU-Net)
- `$TOTALSEG_WEIGHTS_PATH` (TS env var)
- `~/.totalsegmentator/nnunet/results` (TS default install location)

Downstream packages can register their own layouts (`register_weights_layout(WeightsLayout(...))`).

## TotalSegmentator integration

```bash
uv run TotalSegmentator -i scan.nii.gz -o output/ -d mlx
```

```python
from totalsegmentator.python_api import totalsegmentator
totalsegmentator(input="scan.nii.gz", output="output/", device="mlx")
```

The `-d mlx` flag dispatches to this package's engine. No weight conversion required — checkpoints in `~/.totalsegmentator/nnunet/results` are read directly.

## Surface mesh extraction (SurfaceNets from logits)

Generate a multi-material **dual mesh** directly from the K-channel logit
volume — no labelmap intermediate. Vertex positions are sub-voxel
interpolations of the continuous logit field; normals come from the
field gradient; quads carry VTK's `BoundaryLabels` convention so the
result drops straight into Slicer.

```python
from nnunet_inference_mlx import (
    NiftiReader, TaskCatalog, ModelStore,
    infer, preprocess, postprocess,
    mesh_cleanup, mesh_to_vtk_polydata,
)

image = NiftiReader().read("scan.nii.gz")
spec = TaskCatalog("totalsegmentator").get("total_fast")
model = ModelStore("totalsegmentator").load(spec.single)

vol, _ = preprocess.to_model_frame(image, model.model_data, reorient_to="RAS")
prediction = infer.sliding_window(model, vol)

# Mesh at native model spacing
mesh = postprocess.to_mesh(
    prediction,
    confidence_margin=1.0,            # drop edges at isolated low-margin voxels
    drop_components_below_mm3=50.0,   # small-component filter
    project_to_surface=True,          # Newton step onto decision surface
    emit_normals=True,                # field-gradient normals
)
mesh = mesh_cleanup(mesh)             # mesh-side polish (drop tiny regions + smooth)

import vtk
w = vtk.vtkXMLPolyDataWriter()
w.SetFileName("scan_mesh.vtp"); w.SetInputData(mesh_to_vtk_polydata(mesh)); w.Write()
```

For higher-resolution output: `to_mesh(prediction, scale=1.25, ...)` activates
**memory-bounded slab streaming** — meshes at 1.25× the model's native grid
without ever materialising the upsampled K-channel volume. See `examples/07_mesh_multitask.py`
for the multi-task TS-full pattern (`ts:total` = 117-class union of 5 sub-models).

📖 Full docs: [**docs/mesh-pipeline.md**](docs/mesh-pipeline.md) — the recipe,
memory model, multi-task pattern, and known limitations.
📁 Examples: [`06_mesh_output.py`](examples/06_mesh_output.py) (single-task)
and [`07_mesh_multitask.py`](examples/07_mesh_multitask.py) (TS-full).

## Benchmarks

Real abdominal CT (255×178×256, 1.49 mm spacing) on M2 Mac 16 GB RAM:

### 3 mm fast mode (single model, K=118)

| Backend | Wall time |
|---|---|
| **MLX** | **8 s** |
| MPS | 12 s |
| CPU | 54 s |

### 1.5 mm full mode (5-model ensemble)

| Backend | Wall time |
|---|---|
| **MLX** | **3.2 min** |
| MPS | 4.5 min |
| CPU | 45+ min |

100% voxel agreement vs. upstream PyTorch inference.

### Engine cache (process-wide, warm vs cold)

| | Time |
|---|---|
| Cold load (build + compile + warmup) | ~4.5 s |
| Cache hit | ~3 ms |

### Postprocessing (cc3d vs scipy, K=117 CT)

| Method | Time |
|---|---|
| `cc3d.dust` | **80 ms** |
| SITK ScalarConnectedComponent + RelabelComponent | 820 ms |
| scipy per-label loop (TS-style) | 7800 ms |

### Projected scaling with RAM

| RAM | Batch size | Est. full-res time |
|---|---|---|
| 16 GB | 1 | 3.2 min |
| 32 GB | 2–3 | ~2.0 min |
| 64 GB | 5–6 | ~1.2 min |
| 96 GB+ | 7–8 | ~1 min |

## Supported models

- **PlainConvUNet** (nnU-Net default) — fully tested
- **ResidualEncoderUNet** (TS large models) — implemented, used by `Dataset291`–`295` weights
- **Standard label scheme** (mutually-exclusive classes via softmax) — supported
- **Region-based label scheme** (BraTS-style, sigmoid heads + paint priority) — supported
- **Old-format plans.json** (TotalSegmentator-era) — supported
- **New-format plans.json** (`network_arch_init_kwargs`) — supported

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python ≥ 3.10
- MLX ≥ 0.25

## Citations

If you use this package, please cite the original nnU-Net and TotalSegmentator papers:

**nnU-Net:** Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nat Methods* 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z

**TotalSegmentator:** Wasserthal, J., Breit, H.-C., Meyer, M.T. et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence* 5(5) (2023). https://doi.org/10.1148/ryai.230024

## License

Apache 2.0 (same as nnU-Net).
