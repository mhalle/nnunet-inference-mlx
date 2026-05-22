"""Smoke tests for the torch-free .pth loader path.

End-to-end weight loading is exercised by the engine/equivalence tests when
real TotalSegmentator fixtures are present. This file only covers the
``load_model_weights`` file-resolution contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nnunet_inference_mlx.weights import load_model_weights


def test_load_model_weights_missing_raises(tmp_path: Path) -> None:
    (tmp_path / "fold_0").mkdir()
    with pytest.raises(FileNotFoundError):
        load_model_weights(tmp_path, fold=0)
