"""Tests for the dual-layout safetensors loader.

These cover the matrix of (file name, header metadata, source_layout hint)
combinations the loader has to handle so that:

* New nnUNetTrainer-written ``<base>.safetensors`` files load correctly
  (PyTorch-layout, transposed on load).
* Legacy ``<base>_mlx.safetensors`` files written by older releases of this
  package load correctly (MLX-layout, no transpose).
* The metadata header always wins over the caller's hint when present, so
  unusual cases (a file moved across naming conventions) are still correct.
* The file resolution order in ``load_model_weights`` prefers the canonical
  nnUNet layout over the legacy MLX-specific name when both exist.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file as np_load_file
from safetensors.numpy import save_file as np_save_file

from nnunet_inference_mlx.weights import (
    WEIGHT_LAYOUT_MLX,
    WEIGHT_LAYOUT_TORCH,
    load_model_weights,
    load_weights_safetensors,
    save_weights_safetensors,
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


def _save_torch_layout_safetensors(
    path: Path, *, with_metadata: bool, key: str = "encoder.0.weight"
) -> np.ndarray:
    """Write a synthetic torch-layout safetensors file. Returns the source
    weight (pre-transpose) so callers can compare against the loaded result."""
    weight = _torch_layout_conv_weight()
    metadata = {"weight_layout": WEIGHT_LAYOUT_TORCH} if with_metadata else None
    np_save_file({key: weight}, str(path), metadata=metadata)
    return weight


def _save_mlx_layout_safetensors(
    path: Path, *, with_metadata: bool, key: str = "encoder.0.weight"
) -> np.ndarray:
    """Write a synthetic MLX-layout safetensors file (already transposed)."""
    weight = _expected_mlx_layout(_torch_layout_conv_weight())
    metadata = {"weight_layout": WEIGHT_LAYOUT_MLX} if with_metadata else None
    np_save_file({key: weight}, str(path), metadata=metadata)
    return weight


# ---------------------------------------------------------------------------
# Header-driven layout dispatch
# ---------------------------------------------------------------------------

def test_torch_layout_with_metadata_is_transposed(tmp_path: Path) -> None:
    src = _save_torch_layout_safetensors(tmp_path / "ckpt.safetensors", with_metadata=True)

    weights = load_weights_safetensors(tmp_path / "ckpt.safetensors")

    key = next(iter(weights))
    loaded = np.array(weights[key])
    expected = _expected_mlx_layout(src)
    assert loaded.shape == expected.shape == (2, 3, 3, 3, 1)
    np.testing.assert_array_equal(loaded, expected)


def test_mlx_layout_with_metadata_is_loaded_as_is(tmp_path: Path) -> None:
    src = _save_mlx_layout_safetensors(tmp_path / "ckpt.safetensors", with_metadata=True)

    weights = load_weights_safetensors(tmp_path / "ckpt.safetensors")

    key = next(iter(weights))
    loaded = np.array(weights[key])
    np.testing.assert_array_equal(loaded, src)


def test_torch_layout_without_metadata_uses_source_hint(tmp_path: Path) -> None:
    """When the file has no weight_layout entry, the caller's hint kicks in.
    A file written by older nnUNet would land here."""
    src = _save_torch_layout_safetensors(tmp_path / "ckpt.safetensors", with_metadata=False)

    weights = load_weights_safetensors(
        tmp_path / "ckpt.safetensors", source_layout=WEIGHT_LAYOUT_TORCH
    )

    loaded = np.array(weights[next(iter(weights))])
    np.testing.assert_array_equal(loaded, _expected_mlx_layout(src))


def test_mlx_layout_without_metadata_uses_default_hint(tmp_path: Path) -> None:
    """The default source_layout is mlx_ndhwc, so legacy MLX files written
    before this package stamped the metadata still load correctly."""
    src = _save_mlx_layout_safetensors(tmp_path / "ckpt.safetensors", with_metadata=False)

    weights = load_weights_safetensors(tmp_path / "ckpt.safetensors")

    loaded = np.array(weights[next(iter(weights))])
    np.testing.assert_array_equal(loaded, src)


def test_metadata_header_overrides_caller_hint(tmp_path: Path) -> None:
    """If the file says torch_ncdhw, that wins even if the caller passed
    source_layout=mlx_ndhwc. Metadata is the source of truth."""
    src = _save_torch_layout_safetensors(tmp_path / "ckpt.safetensors", with_metadata=True)

    weights = load_weights_safetensors(
        tmp_path / "ckpt.safetensors", source_layout=WEIGHT_LAYOUT_MLX
    )

    np.testing.assert_array_equal(
        np.array(weights[next(iter(weights))]),
        _expected_mlx_layout(src),
    )


def test_unknown_layout_raises(tmp_path: Path) -> None:
    weight = _torch_layout_conv_weight()
    np_save_file(
        {"encoder.0.weight": weight},
        str(tmp_path / "ckpt.safetensors"),
        metadata={"weight_layout": "jax_nhwdc"},
    )
    with pytest.raises(ValueError, match="Unknown weight_layout"):
        load_weights_safetensors(tmp_path / "ckpt.safetensors")


# ---------------------------------------------------------------------------
# save_weights_safetensors stamps metadata
# ---------------------------------------------------------------------------

def test_save_stamps_mlx_layout_metadata(tmp_path: Path) -> None:
    weights = {
        "encoder.0.weight": mx.array(_expected_mlx_layout(_torch_layout_conv_weight()))
    }
    save_weights_safetensors(weights, tmp_path / "out.safetensors")

    with safe_open(str(tmp_path / "out.safetensors"), framework="numpy") as f:
        meta = f.metadata() or {}
    assert meta.get("weight_layout") == WEIGHT_LAYOUT_MLX
    assert meta.get("format_version") == "1"


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    src = _expected_mlx_layout(_torch_layout_conv_weight())
    save_weights_safetensors(
        {"encoder.0.weight": mx.array(src)}, tmp_path / "out.safetensors"
    )

    loaded = load_weights_safetensors(tmp_path / "out.safetensors")
    np.testing.assert_array_equal(np.array(loaded["encoder.0.weight"]), src)


# ---------------------------------------------------------------------------
# load_model_weights file resolution order
# ---------------------------------------------------------------------------

def _make_fold(tmp_path: Path, fold: int = 0) -> Path:
    fold_dir = tmp_path / f"fold_{fold}"
    fold_dir.mkdir()
    return fold_dir


def test_load_model_weights_prefers_nnunet_layout(tmp_path: Path) -> None:
    """When both <base>.safetensors and <base>_mlx.safetensors exist, the
    canonical nnUNet layout wins. The two files contain *different* weights;
    the test asserts the loader picked the nnUNet one."""
    fold_dir = _make_fold(tmp_path)
    distinctive_torch = np.full((2, 1, 3, 3, 3), 7.0, dtype=np.float32)
    distinctive_mlx = np.full((2, 3, 3, 3, 1), -42.0, dtype=np.float32)
    np_save_file(
        {"encoder.0.weight": distinctive_torch},
        str(fold_dir / "checkpoint_final.safetensors"),
        metadata={"weight_layout": WEIGHT_LAYOUT_TORCH},
    )
    np_save_file(
        {"encoder.0.weight": distinctive_mlx},
        str(fold_dir / "checkpoint_final_mlx.safetensors"),
        metadata={"weight_layout": WEIGHT_LAYOUT_MLX},
    )

    weights = load_model_weights(tmp_path, fold=0)
    loaded = np.array(weights["encoder.0.weight"])
    # If nnUNet layout was chosen, we get the transposed version of the 7.0 file.
    np.testing.assert_array_equal(loaded, distinctive_torch.transpose(0, 2, 3, 4, 1))


def test_load_model_weights_falls_back_to_legacy_mlx(tmp_path: Path) -> None:
    fold_dir = _make_fold(tmp_path)
    distinctive_mlx = np.full((2, 3, 3, 3, 1), -42.0, dtype=np.float32)
    np_save_file(
        {"encoder.0.weight": distinctive_mlx},
        str(fold_dir / "checkpoint_final_mlx.safetensors"),
        metadata={"weight_layout": WEIGHT_LAYOUT_MLX},
    )

    weights = load_model_weights(tmp_path, fold=0)
    np.testing.assert_array_equal(np.array(weights["encoder.0.weight"]), distinctive_mlx)


def test_load_model_weights_missing_raises(tmp_path: Path) -> None:
    _make_fold(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_model_weights(tmp_path, fold=0)
