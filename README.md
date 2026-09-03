# nnseg

`nnseg` runs nnU-Net-family segmentation models on PyTorch - TotalSegmentator, MOOSE,
MRSegmentator, and any stock nnU-Net v2 model folder - on an Apple Silicon GPU (MPS), a CUDA
card, or the CPU. It is a library with a command line and a small REST server on top.

This page is what a new user needs to run it. The design record is in `docs/` and the
workspace it lives in; the API is documented in the docstrings.

## Requirements

- **Apple Silicon** (M1 or later) or a CUDA machine. There are no PyTorch wheels for Intel
  Macs anymore, so an Intel Mac cannot run it.
- **Python 3.10 to 3.14** (3.12 is what the tests run on).
- **Memory:** 16 GB runs every task. The 3 mm `total_fast` task is comfortable; whole-body
  `total` at 1.5 mm fits (the sliding-window accumulator moves to host memory when the GPU
  budget is short) but takes about 25 minutes on an M2. More memory means the accumulator
  stays on the GPU and the run is faster; nothing is hard-coded to a laptop.
- About **1.1 GB** of packages (torch is most of it), plus model weights: 160 MB for
  `total_fast`, 1.1 GB for MRSegmentator, a few hundred MB per TotalSegmentator part.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "nnunet-inference-mlx[torch] @ git+https://github.com/mhalle/nnunet-inference-mlx@feature/nnseg"
```

`pip install` of the same URL works too. The distribution is still named `nnunet-inference-mlx`
for historical reasons (the repository began as an MLX port, whose source remains as the
reference nnseg is checked against); what it installs is the `nnseg` package and the `nnseg`
command, nothing else. Add `serve` to the extras for the local server, `remote` for the client:

```bash
uv pip install "nnunet-inference-mlx[torch,serve,remote] @ git+https://github.com/mhalle/nnunet-inference-mlx@feature/nnseg"
```

For a command-line user, a tool install puts `nnseg` on PATH in its own environment, away
from any project:

```bash
uv tool install --python 3.12 "nnunet-inference-mlx[torch,serve,remote] @ git+https://github.com/mhalle/nnunet-inference-mlx@feature/nnseg"
```

and `uvx --from "nnunet-inference-mlx[torch] @ git+https://github.com/mhalle/nnunet-inference-mlx@feature/nnseg" nnseg ...`
runs it without installing anything.

Check it:

```bash
nnseg tasks
```

lists every task the catalog knows, with the engine, modality, and whether its weights are
already on disk. `nnseg tasks --installed` shows what will run without a download.

## Weights

Weights live under one root, shared with TotalSegmentator's own installation so the two never
download the same model twice:

| Source | Location |
|---|---|
| default | `~/.totalsegmentator/nnunet/results` |
| environment | `TOTALSEG_WEIGHTS_PATH`, then `nnUNet_results` |
| command line | `--model-root DIR` |

They download on first use: TotalSegmentator models from the TotalSegmentator GitHub releases
(with the sha256 the manifest records), MOOSE and MRSegmentator from their own hosting. To
provision ahead of time:

```bash
nnseg weights fetch total          # every part the task needs
nnseg weights coverage             # what the manifest can provision, and what it cannot
```

`coverage` marks the TotalSegmentator tasks whose weights are behind TotalSegmentator's
commercial license (`appendicular_bones`, `brain_structures`, `coronary_arteries`,
`heartchambers_highres`, `tissue_types`, ...). nnseg does not handle that license: install those
with TotalSegmentator's own `totalseg_set_license` flow into the same root and nnseg will find
them.

## Segment from the command line

```bash
nnseg segment scan.nii.gz --task total_fast -o labels.nii.gz
```

Input: NIfTI, NRRD, MetaImage, or a DICOM series directory. Output format follows the extension
(`.nii.gz`, `.nrrd`, `.seg.nrrd`, `.mha`); labels come back on the input grid, in the input's
orientation. Task names are `ecosystem:task`, and a bare name is looked up across ecosystems
(`total_fast` is `ts:total_fast`).

Useful options:

| Option | Meaning |
|---|---|
| `--spacing 1.0` | isotropic output spacing in mm instead of the input grid |
| `--interp nearest` | TotalSegmentator's label semantics; the default `linear` gives sub-voxel boundaries from the logits |
| `--device mps|cuda|cpu` | default `auto` |
| `--dtype fp16|bf16|fp32` | default `fp16` (the network runs fp16 on MPS) |
| `--envelope 20` | restrict inference to the body plus this margin in mm; `0` for the whole volume |
| `--accumulate device|host` | force the sliding-window accumulator's placement; `auto` decides from free memory |

What to expect on an M2 for `total_fast` on a 709 x 768 x 768 chest CT, one run per process:

| Stage | First run after install | Later runs |
|---|---|---|
| read + orientation | 8 s | 8 s |
| model load (checkpoint, architecture, GPU upload) | 38 s | 5 s |
| network (8 patches, fp16 MPS) | 19 s | 15 s |
| total | 68 s | 30 s |

The first run after an install pays about 30 s once for compiling and caching (torch's MPS
kernels, bytecode); the second run in the same environment loads the model in 5 s. The Python
API and the server keep models resident, so a second case in the same process pays only the
read and the network.

## Python API

```python
from nnseg import segment, Segmenter

r = segment("scan.nii.gz", "total_fast")        # a Segmentation
r.save("labels.nii.gz")
liver = r.mask("liver")                          # boolean array on the output grid
r.present()                                      # {label: name} for what was found
r.volumes_ml()

seg = Segmenter(cache_models=5)                  # models stay warm across calls
for path in paths:
    seg.segment(path, "total").save(path.with_suffix(".labels.nii.gz"))
job = seg.submit("scan.nii.gz", "total", on_progress=print)   # off-thread, cancellable
```

`segment()` takes the same options as the command line as keyword arguments (`grid=1.0`,
`interp="nearest"`, `device="mps"`, `envelope_mm=20`, ...). A stock nnU-Net model folder
works as a task: `segment("scan.nii.gz", "/path/to/Dataset123_x/nnUNetTrainer__nnUNetPlans__3d_fullres")`.
Errors are one family, `nnseg.NnsegError` (`InputError`, `ModelNotFound`,
`UnsupportedModel`, `ResourceError`, `Cancelled`).

## Local server

The server is the same job protocol nnseg deploys on Modal, run on the machine itself:

```bash
nnseg serve --port 8790 --token choose-a-secret
```

It builds a `Segmenter` with warm models (`--cache-models 5` by default, enough for a whole
`total` union), queues jobs, streams progress, and keeps a durable result cache under
`~/.cache/nnseg/results`. Without `--token` a request can read health, the task list, and
cached results, but never compute. The client:

```bash
export NNSEG_SERVER=http://127.0.0.1:8790
nnseg remote --token choose-a-secret tasks
nnseg remote --token choose-a-secret submit scan.nii.gz --task total_fast -o labels.seg.nrrd
```

`submit` uploads, shows progress, and downloads the labels; `--no-wait` returns a job id for
`status`, `fetch`, and `cancel`. The endpoints are under `/v1/` (`/v1/health`, `/v1/tasks`,
`/v1/jobs`); the OpenAPI document is at `/docs`. On this M2 a `total_fast` job through the
server produced labels voxel-identical to the command line's.

## What nnseg does not do yet

- Multi-channel nnU-Net inputs, region (sigmoid) heads, and the `3d_lowres`, cascade and `2d`
  configurations are not on the nnU-Net path.
- The FastSurfer, SynthStrip, VoxTell and MONAI engines each need their own environment
  (their dependency pins conflict with the torch path) and are run as separate server
  processes; locally they are not part of the default install.
- Versioning: `nnseg.__version__` is nnseg's own number and is what the server reports; the
  distribution's version belongs to the repository as a whole.
