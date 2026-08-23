"""nnseg - nnU-Net-family segmentation on torch (MPS / CUDA / CPU).

Two layers in one package, kept apart by a rule the tests enforce
(``tests/test_nnseg_layering.py``): the **kernel layer** - ``grid``, ``mapping``, ``tables``,
``restore``, ``resample``, ``shuffleup``, ``backends``, ``reference`` - depends on nothing but
torch and numpy and knows nothing about tasks, plans, weights or files, so it stays a leaf that
could be lifted out as its own package the day something outside wants it. The **pipeline
layer** - ``io``, ``preprocess``, ``frame``, ``network``, ``pipeline``, ``cli`` - reads images
through SimpleITK, drives nnU-Net's networks, and composes tasks from the toolkit's catalog.
"""
from . import backends, errors, io
from .errors import (Cancelled, InputError, ModelNotFound, NnsegError, ResourceError,
                     UnsupportedModel)
from .frame import Frame
from .grid import Grid
from .mapping import Mapping
from .network import TorchModel
from .pipeline import segment
from .result import Segmentation
from .segmenter import Segmenter
from .store import ModelStore
from .tasks import TaskCatalog, TaskSpec
from .reference import margins
from .resample import resample_data
from .restore import available_backends, resample_argmax, resample_paint, to_labels
from .shuffleup import ShuffleUp3d, swap_transposed
from .tables import AxisTable, build_tables

__version__ = "0.1.0"
__all__ = [
    # the API most callers need
    "segment", "Segmenter", "Segmentation", "ModelStore", "TaskCatalog", "TaskSpec", "io",
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
