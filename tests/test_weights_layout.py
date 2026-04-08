"""Tests for the canonical safetensors loader.

The loader handles exactly one on-disk format: the nnU-Net canonical layout
(``<base>.safetensors`` containing PyTorch-layout tensors with a
``weight_layout=torch_ncdhw`` metadata header). Files without the metadata
tag, or with an unrecognized layout, are rejected. There is no longer a
fallback hint or a legacy MLX-pre-transposed file format.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file

from nnunet_inference_mlx.weights import (
    WEIGHT_LAYOUT_TORCH,
    convert_pth_to_safetensors,
    load_model_weights,
    load_weights_safetensors,
)


# ---------------------------------------------------------------------------
# Synthetic state dicts
# ---------------------------------------------------------------------------

def _torch_layout_conv_weight(out_ch: int = 2, in_ch: int = 1, k: int = 3) -> np.ndarray:
    """A 5D conv weight in PyTorch layout: (out_ch, in_ch, kD, kH, kW)."""
    return np.arange(out_ch * in_ch * k * k * k, dtype=np.float32).reshape(
        out_ch, in_ch, k, k, k
    )


def _expected_mlx_layout(arr: np.ndarray) -> np.ndarray:
    """Apply the torch_ncdhw -> mlx_ndhwc conv transpose."""
    return arr.transpose(0, 2, 3, 4, 1)


def _save_canonical_safetensors(
    path: Path, *, with_metadata: bool = True, key: str = "encoder.0.weight"
) -> np.ndarray:
    """Write a synthetic torch-layout safetensors file. Returns the source
    weight (pre-transpose) so callers can compare against the loaded result."""
    weight = _torch_layout_conv_weight()
    metadata = {"weight_layout": WEIGHT_LAYOUT_TORCH} if with_metadata else None
    np_save_file({key: weight}, str(path), metadata=metadata)
    return weight


# ---------------------------------------------------------------------------
# load_weights_safetensors
# ---------------------------------------------------------------------------

def test_canonical_layout_is_transposed_on_load(tmp_path: Path) -> None:
    src = _save_canonical_safetensors(tmp_path / "ckpt.safetensors")

    weights = load_weights_safetensors(tmp_path / "ckpt.safetensors")

    key = next(iter(weights))
    loaded = np.array(weights[key])
    expected = _expected_mlx_layout(src)
    assert loaded.shape == expected.shape == (2, 3, 3, 3, 1)
    np.testing.assert_array_equal(loaded, expected)


def test_missing_metadata_header_raises(tmp_path: Path) -> None:
    """A file without the weight_layout entry is ambiguous and rejected.
    The loader does not guess."""
    _save_canonical_safetensors(tmp_path / "ckpt.safetensors", with_metadata=False)
    with pytest.raises(ValueError, match="no weight_layout metadata"):
        load_weights_safetensors(tmp_path / "ckpt.safetensors")


def test_unknown_layout_raises(tmp_path: Path) -> None:
    weight = _torch_layout_conv_weight()
    np_save_file(
        {"encoder.0.weight": weight},
        str(tmp_path / "ckpt.safetensors"),
        metadata={"weight_layout": "jax_nhwdc"},
    )
    with pytest.raises(ValueError, match="Unsupported weight_layout"):
        load_weights_safetensors(tmp_path / "ckpt.safetensors")


# ---------------------------------------------------------------------------
# convert_pth_to_safetensors round-trip
# ---------------------------------------------------------------------------

def test_convert_pth_writes_canonical_layout(tmp_path: Path) -> None:
    """Synthesize a .pth checkpoint, convert it, then verify the output file
    has the right metadata header and round-trips through the MLX loader."""
    import torch

    src = _torch_layout_conv_weight()
    pth_path = tmp_path / "checkpoint_final.pth"
    torch.save(
        {"network_weights": {"encoder.0.weight": torch.from_numpy(src)}},
        str(pth_path),
    )

    out = convert_pth_to_safetensors(pth_path)
    assert out == pth_path.with_suffix(".safetensors")

    with safe_open(str(out), framework="numpy") as f:
        meta = f.metadata() or {}
    assert meta.get("weight_layout") == WEIGHT_LAYOUT_TORCH
    assert meta.get("format_version") == "1"

    weights = load_weights_safetensors(out)
    np.testing.assert_array_equal(
        np.array(weights["encoder.0.weight"]),
        _expected_mlx_layout(src),
    )


# ---------------------------------------------------------------------------
# load_model_weights file resolution
# ---------------------------------------------------------------------------

def _make_fold(tmp_path: Path, fold: int = 0) -> Path:
    fold_dir = tmp_path / f"fold_{fold}"
    fold_dir.mkdir()
    return fold_dir


def test_load_model_weights_reads_canonical_safetensors(tmp_path: Path) -> None:
    fold_dir = _make_fold(tmp_path)
    src = _save_canonical_safetensors(fold_dir / "checkpoint_final.safetensors")

    weights = load_model_weights(tmp_path, fold=0)

    np.testing.assert_array_equal(
        np.array(weights["encoder.0.weight"]),
        _expected_mlx_layout(src),
    )


def test_load_model_weights_falls_back_to_pth(tmp_path: Path) -> None:
    """When no .safetensors exists, the loader uses torch.load on the .pth.
    This is the slow path; subsequent loads should run convert first."""
    import torch

    fold_dir = _make_fold(tmp_path)
    src = _torch_layout_conv_weight()
    torch.save(
        {"network_weights": {"encoder.0.weight": torch.from_numpy(src)}},
        str(fold_dir / "checkpoint_final.pth"),
    )

    weights = load_model_weights(tmp_path, fold=0)
    np.testing.assert_array_equal(
        np.array(weights["encoder.0.weight"]),
        _expected_mlx_layout(src),
    )


def test_load_model_weights_missing_raises(tmp_path: Path) -> None:
    _make_fold(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_model_weights(tmp_path, fold=0)
