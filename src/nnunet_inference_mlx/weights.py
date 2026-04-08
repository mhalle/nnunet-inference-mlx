"""
Weight conversion and loading for MLX nnU-Net.

Supports:
  - Converting PyTorch .pth checkpoints to .safetensors (one-time, needs torch)
  - Loading .safetensors at runtime (no torch needed)
  - Automatic detection: loads safetensors if available, falls back to .pth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx


def convert_pytorch_weights(
    pt_state_dict: dict,
    key_map: dict[str, str] | None = None,
) -> dict[str, mx.array]:
    """Convert a PyTorch nnU-Net state dict to MLX format.

    Main operations:
      - Skip duplicate keys (all_modules.*, decoder.encoder.*)
      - Conv3d weights: (out, in, D, H, W) -> (out, D, H, W, in)
      - ConvTranspose3d weights: (in, out, D, H, W) -> (out, D, H, W, in)
      - 1D tensors (bias, norm): no change
      - Key remapping: remove extra .0 from Sequential wrapping
    """
    mlx_weights = {}

    for pt_key, tensor in pt_state_dict.items():
        if ".all_modules." in pt_key:
            continue
        if pt_key.startswith("decoder.encoder."):
            continue

        if hasattr(tensor, "numpy"):
            arr = tensor.cpu().numpy()
        else:
            arr = np.asarray(tensor)

        mlx_key = _remap_pt_key(pt_key)

        if key_map and pt_key in key_map:
            mlx_key = key_map[pt_key]

        # Transpose 5D conv weights
        if arr.ndim == 5:
            if "transpconv" in pt_key or "ConvTranspose" in pt_key:
                # PyTorch ConvTranspose3d: (in_ch, out_ch, D, H, W)
                # MLX ConvTranspose3d:     (out_ch, D, H, W, in_ch)
                arr = arr.transpose(1, 2, 3, 4, 0)
            else:
                # PyTorch Conv3d: (out_ch, in_ch, D, H, W)
                # MLX Conv3d:     (out_ch, D, H, W, in_ch)
                arr = arr.transpose(0, 2, 3, 4, 1)

        mlx_weights[mlx_key] = mx.array(arr)

    return mlx_weights


def _remap_pt_key(key: str) -> str:
    """Remap a PyTorch state dict key to match MLX module hierarchy."""
    parts = key.split(".")
    result = []
    i = 0
    while i < len(parts):
        result.append(parts[i])
        if (parts[i] == "stages" and i + 3 < len(parts)
                and parts[i + 1].isdigit()
                and parts[i + 2] == "0"
                and parts[i + 3] in ("convs", "blocks")):
            result.append(parts[i + 1])
            i += 3
        else:
            i += 1
    return ".".join(result)


def fuzzy_load_weights(network, mlx_weights: dict, verbose: bool = False):
    """Match PyTorch keys to MLX keys by adjusting hierarchy."""
    import mlx.nn as nn

    model_keys = set()
    for k, _ in nn.utils.tree_flatten(network.parameters()):
        model_keys.add(k)

    mapped = {}
    unmapped = []
    for key, val in mlx_weights.items():
        if key in model_keys:
            mapped[key] = val
        else:
            parts = key.split(".")
            new_parts = []
            skip_next = False
            for j, p in enumerate(parts):
                if skip_next:
                    skip_next = False
                    continue
                if (p == "stages" and j + 2 < len(parts)
                        and parts[j + 1].isdigit() and parts[j + 2] == "0"):
                    new_parts.append(p)
                    new_parts.append(parts[j + 1])
                    skip_next = True
                else:
                    new_parts.append(p)
            candidate = ".".join(new_parts)
            candidate = candidate.replace(".all_modules.0.", ".conv.")
            candidate = candidate.replace(".all_modules.1.", ".norm.")
            candidate = candidate.replace(".all_modules.2.", ".nonlin.")

            if candidate in model_keys:
                mapped[candidate] = val
            else:
                unmapped.append(key)

    if unmapped and verbose:
        print(f"Warning: {len(unmapped)} unmapped weight keys")
        for k in unmapped[:5]:
            print(f"  {k}")

    network.load_weights(list(mapped.items()))


# ---------------------------------------------------------------------------
# Safetensors I/O
# ---------------------------------------------------------------------------

WEIGHT_LAYOUT_MLX = "mlx_ndhwc"
WEIGHT_LAYOUT_TORCH = "torch_ncdhw"


def save_weights_safetensors(mlx_weights: dict[str, mx.array], path: str | Path):
    """Save MLX-layout weights to safetensors format.

    Stamps ``weight_layout=mlx_ndhwc`` into the safetensors metadata header so
    future loaders (this one or any other) can decide unambiguously whether
    the on-disk tensors need transposing.
    """
    from safetensors.numpy import save_file
    np_weights = {k: np.array(v) for k, v in mlx_weights.items()}
    save_file(
        np_weights,
        str(path),
        metadata={
            "weight_layout": WEIGHT_LAYOUT_MLX,
            "format_version": "1",
        },
    )


def load_weights_safetensors(
    path: str | Path,
    source_layout: str = WEIGHT_LAYOUT_MLX,
) -> dict[str, mx.array]:
    """Load weights from a safetensors file, transposing if necessary.

    Reads the ``weight_layout`` entry from the safetensors metadata header
    and chooses the load path:

    * ``mlx_ndhwc`` — file already holds MLX-layout tensors, load as-is.
    * ``torch_ncdhw`` — file holds PyTorch-layout tensors (e.g. written by
      ``nnUNetTrainer`` directly), transpose conv weights to MLX layout.

    If the file has no ``weight_layout`` entry (e.g. legacy MLX files written
    before this convention, or torch files written by older nnU-Net), the
    caller's ``source_layout`` hint is used. Defaults to ``mlx_ndhwc`` so
    pre-existing MLX-converted files keep loading without changes.

    Loading goes through ``safetensors.numpy``, never through torch — works
    on Apple Silicon with no PyTorch installed.
    """
    from safetensors import safe_open
    from safetensors.numpy import load_file

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
    layout = meta.get("weight_layout", source_layout)

    np_state_dict = load_file(str(path))

    if layout == WEIGHT_LAYOUT_MLX:
        return {k: mx.array(v) for k, v in np_state_dict.items()}
    if layout == WEIGHT_LAYOUT_TORCH:
        # convert_pytorch_weights handles the transpose, key remapping, and
        # filtering. Its tensor branch accepts numpy arrays via np.asarray.
        return convert_pytorch_weights(np_state_dict)
    raise ValueError(
        f"Unknown weight_layout {layout!r} in {path}. "
        f"Expected one of: {WEIGHT_LAYOUT_MLX!r}, {WEIGHT_LAYOUT_TORCH!r}."
    )


def convert_model_folder(model_folder: str | Path, checkpoint_name: str = "checkpoint_final.pth"):
    """Convert all .pth checkpoints in a model folder to safetensors.

    Converts each fold's checkpoint. After this, the MLX runtime can
    load weights without torch.

    Requires torch (one-time conversion).
    """
    import torch

    model_folder = Path(model_folder)
    converted = 0
    for fold_dir in sorted(model_folder.glob("fold_*")):
        pth_path = fold_dir / checkpoint_name
        if not pth_path.exists():
            continue
        safetensors_path = fold_dir / checkpoint_name.replace(".pth", "_mlx.safetensors")
        if safetensors_path.exists():
            continue

        ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
        mlx_weights = convert_pytorch_weights(ckpt["network_weights"])
        save_weights_safetensors(mlx_weights, safetensors_path)
        converted += 1
        print(f"  Converted {fold_dir.name}/{checkpoint_name} -> {safetensors_path.name}")

    return converted


def load_model_weights(
    model_folder: str | Path,
    fold: int = 0,
    checkpoint_name: str = "checkpoint_final.pth",
) -> dict[str, mx.array]:
    """Load weights for a model fold, preferring safetensors over .pth.

    File resolution order, from most to least preferred:

    1. ``<base>.safetensors`` — the canonical layout written by nnUNetTrainer
       (PyTorch-layout tensors with a ``weight_layout=torch_ncdhw`` metadata
       header). Models trained with new nnU-Net land here directly with no
       conversion step. The loader transposes on load via the metadata tag.
    2. ``<base>_mlx.safetensors`` — the legacy MLX-pre-transposed format
       written by this package's own ``nnunet-inference-mlx-convert`` CLI on
       prior releases. Loaded as-is.
    3. ``<base>.pth`` — legacy PyTorch pickle. Loaded via ``torch.load``
       (one-time torch dependency at conversion time only).

    Returns an MLX weight dict ready for ``network.load_weights``.
    """
    model_folder = Path(model_folder)
    fold_dir = model_folder / f"fold_{fold}"
    base = checkpoint_name.replace(".pth", "")

    # 1. Canonical nnUNet safetensors layout (torch-layout tensors).
    nnunet_path = fold_dir / f"{base}.safetensors"
    if nnunet_path.exists():
        return load_weights_safetensors(nnunet_path, source_layout=WEIGHT_LAYOUT_TORCH)

    # 2. Legacy MLX-pre-transposed safetensors.
    legacy_path = fold_dir / f"{base}_mlx.safetensors"
    if legacy_path.exists():
        return load_weights_safetensors(legacy_path, source_layout=WEIGHT_LAYOUT_MLX)

    # 3. Last resort: .pth (requires torch).
    pth_path = fold_dir / checkpoint_name
    if pth_path.exists():
        import torch
        ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
        return convert_pytorch_weights(ckpt["network_weights"])

    raise FileNotFoundError(
        f"No weights found in {fold_dir}. "
        f"Expected one of: {nnunet_path.name}, {legacy_path.name}, {checkpoint_name}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def convert_weights_cli():
    """Command-line tool to convert all TotalSegmentator models to safetensors."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PyTorch nnU-Net weights to MLX safetensors format"
    )
    parser.add_argument("path",
                        help="Path to checkpoint_final.pth or model folder")
    parser.add_argument("-o", "--output",
                        help="Output path (only for single checkpoint)")
    parser.add_argument("--all", action="store_true",
                        help="Convert all models in ~/.totalsegmentator/")
    args = parser.parse_args()

    if args.all:
        import os
        if "TOTALSEG_WEIGHTS_PATH" in os.environ:
            weights_dir = Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
        else:
            weights_dir = Path.home() / ".totalsegmentator" / "nnunet" / "results"

        total = 0
        for dataset_dir in sorted(weights_dir.glob("Dataset*")):
            for model_dir in sorted(dataset_dir.glob("*__*__*")):
                print(f"Converting {model_dir.name}...")
                total += convert_model_folder(model_dir)
        print(f"\nConverted {total} checkpoints total.")
    else:
        path = Path(args.path)
        if path.is_file() and path.suffix == ".pth":
            # Single checkpoint
            import torch
            from safetensors.numpy import save_file

            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            mlx_weights = convert_pytorch_weights(ckpt["network_weights"])
            np_weights = {k: np.array(v) for k, v in mlx_weights.items()}
            out_path = args.output or str(path).replace(".pth", "_mlx.safetensors")
            save_file(np_weights, out_path)
            print(f"Saved {len(np_weights)} tensors to {out_path}")
        elif path.is_dir():
            # Model folder
            n = convert_model_folder(path)
            print(f"Converted {n} checkpoints.")
        else:
            print(f"Error: {path} is not a .pth file or directory")
