"""nnseg - nnU-Net-family segmentation on torch (MPS / CUDA / CPU).

Two layers in one package, kept apart by a rule the tests enforce
(``tests/test_nnseg_layering.py``): the **kernel layer** - ``grid``, ``mapping``, ``tables``,
``restore``, ``resample``, ``shuffleup``, ``backends``, ``reference`` - depends on nothing but
torch and numpy and knows nothing about tasks, plans, weights or files, so it stays a leaf that
could be lifted out as its own package the day something outside wants it. The **pipeline
layer** - ``io``, ``preprocess``, ``frame``, ``network``, ``pipeline``, ``cli`` - reads images
through SimpleITK, drives nnU-Net's networks, and composes tasks from the toolkit's catalog.
"""
# Eager imports are the torch-FREE surface: importing nnseg costs only these. The
# inference path (torch) is loaded lazily on first access (see _LAZY / __getattr__
# below), so `import nnseg` pulls no torch - a lean consumer (the serve front-end, a
# describe-only caller) never pays for it. Enforced by tests/test_nnseg_layering.py.
from . import errors, io
from .job import Job
from .progress import CancelToken, Progress
from .errors import (Cancelled, InputError, ModelNotFound, NnsegError, ResourceError,
                     UnsupportedModel)
from .frame import Frame
from .grid import Grid
from .mapping import Mapping
from .result import Segmentation
from .segmenter import Segmenter
from .cache import ModelCache
from .weights import WeightsStore
from .tasks import TaskCatalog, TaskSpec
from .reference import margins
from .tables import AxisTable, build_tables

# Torch-pulling exports, loaded on first ATTRIBUTE access (PEP 562). These live in the
# modules that import torch at top level (network / pipeline / resample / restore /
# shuffleup); keeping them lazy is what makes `import nnseg` torch-free.
_LAZY = {
    "segment": ("pipeline", "segment"),
    "TorchModel": ("network", "TorchModel"),
    "resample_data": ("resample", "resample_data"),
    "available_backends": ("restore", "available_backends"),
    "resample_argmax": ("restore", "resample_argmax"),
    "resample_paint": ("restore", "resample_paint"),
    "to_labels": ("restore", "to_labels"),
    "ShuffleUp3d": ("shuffleup", "ShuffleUp3d"),
    "swap_transposed": ("shuffleup", "swap_transposed"),
}
# The backends subpackage imports torch (each backend module does), so it's lazy too -
# accessed as `nnseg.backends`, and pulled anyway by the inference path when it runs.
_LAZY_SUBMODULES = ("backends",)


def __getattr__(name: str):
    import importlib
    if name in _LAZY_SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    else:
        spec = _LAZY.get(name)
        if spec is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f".{spec[0]}", __name__), spec[1])
    globals()[name] = value            # cache: __getattr__ won't fire again for it
    return value


def __dir__():
    return sorted([*globals(), *_LAZY])


# Part of the result-cache key (serve.result_key): bumping it invalidates cached results.
# 0.2.0 does exactly that - multi-model tasks before it ran parts 2..N on the first
# model's normalization, so every cached `total` result predating it is degraded.
__version__ = "0.2.0"
__all__ = [
    # the API most callers need
    "segment", "Segmenter", "Segmentation", "Job", "Progress", "CancelToken", "ModelCache", "TaskCatalog", "TaskSpec", "io",
    # errors, catchable as a family or individually
    "NnsegError", "InputError", "ModelNotFound", "UnsupportedModel", "ResourceError",
    "Cancelled", "errors",
    # geometry and models, for callers composing their own pipeline
    "Frame", "Grid", "Mapping", "TorchModel",
    # kernel layer
    "AxisTable", "ShuffleUp3d", "available_backends", "backends", "build_tables", "margins",
    "resample_argmax", "resample_data", "resample_paint", "swap_transposed", "to_labels",
    "__version__",
]
