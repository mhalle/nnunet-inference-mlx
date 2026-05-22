"""
Weight loading for MLX nnU-Net.

Reads TotalSegmentator release ``.pth`` checkpoints directly via the
vendored torch-free unpickler. No torch dependency, no on-disk conversion
step — point :func:`load_model_weights` at the release tree and go.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx

from ._torchfree import load_pth


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


def load_model_weights(
    model_folder: str | Path,
    fold: int = 0,
    checkpoint_name: str = "checkpoint_final.pth",
) -> dict[str, mx.array]:
    """Load weights for a model fold from a TotalSegmentator release ``.pth``.

    Reads ``<fold_dir>/<checkpoint_name>`` directly via the vendored
    torch-free unpickler — no torch, no on-disk conversion. Returns an MLX
    weight dict ready for ``network.load_weights``.
    """
    fold_dir = Path(model_folder) / f"fold_{fold}"
    pth_path = fold_dir / checkpoint_name
    if not pth_path.exists():
        raise FileNotFoundError(f"No weights at {pth_path}")
    return convert_pytorch_weights(load_pth(str(pth_path)))
