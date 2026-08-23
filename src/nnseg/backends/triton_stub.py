"""CUDA backend placeholder. The Metal kernel's structure (one thread per
output voxel, 8-corner gather, running argmax / paint over K) ports directly;
it needs a CUDA box to write and validate against the torch backend."""
from __future__ import annotations


def available() -> bool:
    return False


def run(*args, **kwargs):
    raise NotImplementedError("nnseg: the Triton backend is not implemented yet; use backend='torch' on CUDA")
