"""nnU-Net inference on Apple Silicon via MLX."""

from .engine import (
    FoldEnsemble,
    InferenceEngine,
    ModelBundle,
    Predictor,
    SlidingWindowEngine,
    softmax_inplace,
)
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
    convert_pytorch_weights,
    discover_folds,
    load_checkpoint_with_metadata,
    load_model_weights,
)

__all__ = [
    # Layered primitives
    "ModelBundle",
    "Predictor",
    "SlidingWindowEngine",
    "FoldEnsemble",
    "InferenceEngine",          # back-compat facade
    "softmax_inplace",
    # Model + plans + weights
    "PlainConvUNet",
    "ResidualEncoderUNet",
    "build_network_from_plans",
    "convert_pytorch_weights",
    "load_model_weights",
    "load_checkpoint_with_metadata",
    "discover_folds",
    # Preprocessing + NIfTI I/O
    "preprocess_volume",
    "load_nifti_zyx",
    "save_segmentation_zyx",
    "predict_nifti",
    "predict_folder",
]
