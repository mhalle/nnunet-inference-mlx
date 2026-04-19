"""nnU-Net inference on Apple Silicon via MLX."""

from .engine import InferenceEngine, ModelBundle, softmax_inplace
from .io import (
    load_nifti_zyx,
    predict_folder,
    predict_nifti,
    save_segmentation_zyx,
)
from .model import PlainConvUNet, ResidualEncoderUNet
from .plans import build_network_from_plans
from .preprocessing import preprocess_volume
from .weights import (
    convert_model_folder,
    convert_pth_to_safetensors,
    convert_pytorch_weights,
    load_model_weights,
    load_weights_safetensors,
)

__all__ = [
    "InferenceEngine",
    "ModelBundle",
    "softmax_inplace",
    "PlainConvUNet",
    "ResidualEncoderUNet",
    "build_network_from_plans",
    "convert_model_folder",
    "convert_pth_to_safetensors",
    "convert_pytorch_weights",
    "load_model_weights",
    "load_weights_safetensors",
    "preprocess_volume",
    "load_nifti_zyx",
    "save_segmentation_zyx",
    "predict_nifti",
    "predict_folder",
]
