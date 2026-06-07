"""nnU-Net inference on Apple Silicon via MLX."""

# ---------------------------------------------------------------------------
# Toolkit API (0.10; the composable, no-hidden-state surface).
# Three nouns + one verb: TaskCatalog (name→recipe), ModelStore (id→model,
# read-through + bounded), segment (the headline). Frozen value types as the
# currency; preprocess/infer/postprocess/geometry as pure-fn namespaces.
# ---------------------------------------------------------------------------
from . import geometry, infer, postprocess, preprocess
from .build import LoadedModel, build_model
from .catalog import TaskCatalog
from .imageio import ArrayReader, DicomReader, NiftiReader, NiftiWriter
from .model_data import ModelData
from .segment import segment
from .store import ModelStore
from .values import (
    BuildOptions,
    Geometry,
    LabelSchema,
    Mesh,
    Prediction,
    Region,
    RestorePlan,
    Segmentation,
    Volume,
)
from .mesh import surfacenets_logits
from .meshio import (
    mesh_compute_normals, mesh_from_npz, mesh_smooth, mesh_to_npz,
    mesh_to_vtk_polydata,
)

from .engine import (
    FoldEnsemble,
    InferenceEngine,
    ModelBundle,
    Predictor,
    SlidingWindowEngine,
    softmax_inplace,
)
from .labels import (
    convert_logits_to_segmentation,
    has_regions,
    label_dtype,
    mesh_concat,
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
    reorient,
    resample_image_to_target,
)
from .postprocessing import remove_small_components
from .preprocessing import resample_volume
from .tasks import (
    AmbiguousTaskError,
    CascadeStep,
    TaskSpec,
    UnionPart,
    WeightsId,
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
    # ===== toolkit API (0.10) =====
    # nouns + verb
    "TaskCatalog",
    "ModelStore",
    "segment",
    "build_model",
    "LoadedModel",
    "ModelData",
    # value types (the inter-stage currency)
    "Geometry",
    "Volume",
    "Segmentation",
    "Prediction",
    "Mesh",
    "LabelSchema",
    "Region",
    "RestorePlan",
    "BuildOptions",
    # mesh extraction + serialization
    "surfacenets_logits",
    "mesh_to_npz",
    "mesh_from_npz",
    "mesh_to_vtk_polydata",
    "mesh_smooth",
    "mesh_compute_normals",
    # image IO (format plug-ins)
    "NiftiReader",
    "DicomReader",
    "ArrayReader",
    "NiftiWriter",
    # pure-fn stage namespaces (use as preprocess.to_model_frame, etc.)
    "preprocess",
    "infer",
    "postprocess",
    "geometry",
    # ===== legacy surface (removed at the rest of the Phase 5 cutover) =====
    # Layered primitives
    "ModelBundle",
    "Predictor",
    "SlidingWindowEngine",
    "FoldEnsemble",
    "InferenceEngine",          # private compute core
    "softmax_inplace",
    "sigmoid_inplace",
    # Label-scheme post-processing
    "convert_logits_to_segmentation",
    "has_regions",
    "label_dtype",
    "mesh_concat",
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
    # Resampling primitives (SITK-backed; SITK is a core dependency)
    "get_orientation",
    "inverse_resample_argmax",
    "inverse_resample_paint",
    "reorient",
    "resample_image_to_target",
    "resample_volume",
    # Postprocessing (opt-in via [postprocessing] extra; cc3d backend)
    "remove_small_components",
    # Recipe types (the declarative task vocabulary; lookup is via TaskCatalog)
    "AmbiguousTaskError",
    "CascadeStep",
    "TaskSpec",
    "UnionPart",
    "WeightsId",
]
