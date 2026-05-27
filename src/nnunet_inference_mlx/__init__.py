"""nnU-Net inference on Apple Silicon via MLX."""

from .engine import (
    FoldEnsemble,
    InferenceEngine,
    ModelBundle,
    Predictor,
    SlidingWindowEngine,
    WeightsLayout,
    discover_weights,
    list_weights_layouts,
    register_weights_layout,
    softmax_inplace,
)
from .engine_cache import (
    cache_enabled,
    cache_engine,
    cached_engine_from_folder,
    cached_engine_from_task,
    clear_engine_cache,
    get_cached_engine,
)
from .labels import (
    convert_logits_to_segmentation,
    has_regions,
    label_dtype,
    paint_union,
    regions_class_order,
    remap_labels,
    sigmoid_inplace,
)
from .io import (
    load_nifti_zyx,
    predict_folder,
    predict_nifti,
    save_segmentation_zyx,
)
from .resampling import (
    get_orientation,
    inverse_resample_argmax,
    inverse_resample_paint,
    predict_with_resampling,
    reorient,
    resample_image_to_target,
)
from .postprocessing import remove_small_components
from .preprocessing import resample_volume
from .tasks import (
    CascadeStep,
    TaskSpec,
    UnionPart,
    get_task,
    list_registered_tasks,
    list_tasks_by_modality,
    register_task,
    run_named_task,
    unregister_task,
)
from .workflow import (
    Bbox,
    ParallelStage,
    Stage,
    compute_fg_bbox,
    crop_image,
    paste_segmentation,
    run_label_union_workflow,
    run_workflow,
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
    "sigmoid_inplace",
    # Weights layout discovery (nnU-Net + TS by default; downstream extensible)
    "WeightsLayout",
    "discover_weights",
    "list_weights_layouts",
    "register_weights_layout",
    # Process-wide engine cache
    "cache_enabled",
    "cache_engine",
    "cached_engine_from_folder",
    "cached_engine_from_task",
    "clear_engine_cache",
    "get_cached_engine",
    # Label-scheme post-processing
    "convert_logits_to_segmentation",
    "has_regions",
    "label_dtype",
    "paint_union",
    "regions_class_order",
    "remap_labels",
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
    # Resampling + SITK-based predict_with_resampling (opt-in via [preprocessing] extra)
    "get_orientation",
    "inverse_resample_argmax",
    "inverse_resample_paint",
    "predict_with_resampling",
    "reorient",
    "resample_image_to_target",
    "resample_volume",
    # Postprocessing (opt-in via [postprocessing] extra; cc3d backend)
    "remove_small_components",
    # Multi-stage workflow + geometric primitives
    "Bbox",
    "ParallelStage",
    "Stage",
    "compute_fg_bbox",
    "crop_image",
    "paste_segmentation",
    "run_label_union_workflow",
    "run_workflow",
    # Declarative task registry + named-task dispatcher
    "CascadeStep",
    "TaskSpec",
    "UnionPart",
    "get_task",
    "list_registered_tasks",
    "list_tasks_by_modality",
    "register_task",
    "run_named_task",
    "unregister_task",
]
