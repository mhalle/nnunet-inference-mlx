# nnunet-mlx

MLX inference backend for [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) on Apple Silicon. Runs nnU-Net models natively on Metal without PyTorch.

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is a self-configuring framework for medical image segmentation that consistently achieves state-of-the-art results across a wide range of biomedical datasets. It automates the entire segmentation pipeline -- architecture selection, preprocessing, training, and postprocessing -- making expert-level performance accessible without manual tuning. This package brings nnU-Net's trained models to Apple Silicon with native Metal acceleration.

This package also integrates with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), the widely-used tool for automatic segmentation of 117 anatomical structures in CT images.

> **Note:** This is an alpha-level port. It is not our intent to maintain a fork of nnU-Net. The goal is to demonstrate the potential performance improvement of using MLX to accelerate nnU-Net inference for Mac users. We hope that these changes can be incorporated back into the official [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) package in the future.

## Features

- Drop-in MLX backend for TotalSegmentator (`-d mlx`)
- 6x faster than PyTorch CPU, 1.4x faster than PyTorch MPS on Apple Silicon
- Memory-efficient segmentation mode for large volumes
- Auto-scales batch size to available Metal memory
- Integrates directly with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

## Installation

We recommend [uv](https://docs.astral.sh/uv/) for managing Python environments:

```bash
uv add nnunet-mlx
```

Or with pip:

```bash
pip install nnunet-mlx
```

## Usage with TotalSegmentator

```bash
# Convert weights (one-time, requires torch)
uv run --with torch nnunet-mlx-convert --all .

# Run TotalSegmentator with MLX backend
uv run TotalSegmentator -i scan.nii.gz -o output/ -d mlx
```

Or from Python:

```python
from totalsegmentator.python_api import totalsegmentator

result = totalsegmentator(input="scan.nii.gz", output="output/", device="mlx")
```

## Standalone usage

```python
from nnunet_mlx import nnUNetv2_predict_mlx

nnUNetv2_predict_mlx(
    dir_in="input/",
    dir_out="output/",
    task_id=297,
    trainer="nnUNetTrainer_4000epochs_NoMirroring",
)
```

## Benchmarks

Tested on a real abdominal CT (255x178x256, 1.49mm spacing) on an M2 Mac with 17GB RAM:

### 3mm fast mode (single model, 118 classes)

| Backend | Prediction time |
|---------|----------------|
| **MLX** | **8s** |
| MPS | 12s |
| CPU | 54s |

### 1.5mm full mode (5 models)

| Backend | Prediction time |
|---------|----------------|
| **MLX** | **3.2 min** |
| MPS | 4.5 min |
| CPU | 45+ min |

Results are identical across all backends (verified 100% voxel agreement).

### Projected scaling with RAM

| RAM | Batch size | Est. full-res time |
|-----|-----------|-------------------|
| 17GB | 1 | 3.2 min |
| 32GB | 2-3 | ~2.0 min |
| 64GB | 5-6 | ~1.2 min |
| 96GB+ | 7-8 | ~1 min |

## Weight conversion

Models are stored as PyTorch `.pth` checkpoints. Convert them to safetensors for torch-free runtime:

```bash
# Convert all downloaded TotalSegmentator models (torch needed for conversion only)
uv run --with torch nnunet-mlx-convert --all .

# Convert a specific model folder
uv run --with torch nnunet-mlx-convert ~/.totalsegmentator/nnunet/results/Dataset297_*/nnUNet*/

# Convert a single checkpoint
uv run --with torch nnunet-mlx-convert path/to/checkpoint_final.pth
```

After conversion, the runtime dependencies are `mlx`, `numpy`, `nibabel`, `scipy`, and `safetensors`. No PyTorch required.

## How it works

The package reimplements nnU-Net's inference pipeline in MLX:

- **model.py** -- PlainConvUNet and ResidualEncoderUNet architectures (Conv3d, InstanceNorm, LeakyReLU, ConvTranspose3d)
- **weights.py** -- PyTorch-to-MLX weight conversion with NCDHW-to-NDHWC transposition
- **inference.py** -- Sliding window prediction with Gaussian weighting, segmentation mode with top-2 confidence tracking
- **plans.py** -- Parses nnU-Net plans.json (both old and new formats) to build the correct architecture
- **preprocessing.py** -- CTNormalization matching nnU-Net's implementation
- **predict.py** -- `nnUNetv2_predict_mlx()`, the main entry point

Key optimizations:
- `mx.compile` for fused conv-norm-relu operations (~1.8x speedup)
- Segmentation mode: accumulates only labels + confidence per voxel instead of all class logits, enabling inference on large volumes with minimal memory
- Auto batch sizing based on Metal buffer limits

## Supported models

- PlainConvUNet (nnU-Net default) -- fully tested
- ResidualEncoderUNet -- architecture implemented, weight mapping not yet verified
- Old-style plans format (TotalSegmentator models) -- supported
- New-style plans format (network_arch_init_kwargs) -- supported

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.10
- MLX >= 0.22

## Citations

If you use this package, please cite the original nnU-Net and TotalSegmentator papers:

**nnU-Net:**
Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nat Methods* 18, 203--211 (2021). https://doi.org/10.1038/s41592-020-01008-z

**TotalSegmentator:**
Wasserthal, J., Breit, H.-C., Meyer, M.T. et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence* 5(5) (2023). https://doi.org/10.1148/ryai.230024

## License

Same as nnU-Net (Apache 2.0).
