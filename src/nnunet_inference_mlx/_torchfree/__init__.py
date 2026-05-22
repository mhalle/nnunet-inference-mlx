"""Vendored torch-free PyTorch checkpoint loader.

Reads zip-format ``.pth`` checkpoints (PyTorch >= 1.6) into numpy arrays
without importing torch. Used by :mod:`nnunet_inference_mlx.weights` to
load TotalSegmentator release weights directly.

Public entry points: :func:`torchfree_load.load_pth`,
:func:`torchfree_load.load_pth_url`, :func:`torchfree_load.smart_load_url`.
"""

from .torchfree_load import (
    load_pth,
    load_pth_url,
    load_from_zip,
    plan_load,
    smart_load_url,
)

__all__ = [
    "load_pth",
    "load_pth_url",
    "load_from_zip",
    "plan_load",
    "smart_load_url",
]
