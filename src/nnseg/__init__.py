"""nnseg - nnU-Net-family segmentation on torch (MPS / CUDA / CPU), with labelgrid restore.

The torch path of the toolkit. Reuses the framework-neutral core of
``nnunet_inference_mlx`` (task catalog, weight resolution, labels, value types) and adds:
``network`` (nnU-Net model folder -> fp16-capable network + sliding window),
``preprocess`` (canonical frame, forward resample, normalization), ``frame`` (the geometry
record that composes the output-grid mapping), ``pipeline`` (segment()).
"""
from . import io
from .frame import Frame
from .network import TorchModel
from .pipeline import segment

__version__ = "0.1.0"
__all__ = ["Frame", "TorchModel", "io", "segment", "__version__"]
