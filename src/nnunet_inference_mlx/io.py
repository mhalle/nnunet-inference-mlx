"""NIfTI convenience helpers.

Optional layer on top of InferenceEngine. Handles the boilerplate that
clients (e.g. TotalSegmentator) would otherwise re-implement: NIfTI load,
axis-order conversion, argmax, save. nibabel is imported lazily so the
core engine stays free of file-format dependencies.

Axis convention
---------------
nibabel reports volumes in (X, Y, Z); nnU-Net works in (Z, Y, X). All
helpers here transpose at the boundary so callers stay in nibabel order.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .engine import InferenceEngine


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel is required for NIfTI helpers. "
            "Install with: pip install nibabel"
        ) from e
    return nib


def load_nifti_zyx(path: str | Path) -> tuple[np.ndarray, "object"]:
    """Load a NIfTI file and return (volume_zyx, nibabel_image).

    The returned volume is a transposed view in (Z, Y, X) order ready
    for InferenceEngine.predict. The image is returned so callers can
    reuse its affine/header when saving outputs.
    """
    nib = _require_nibabel()
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data.transpose(2, 1, 0), img


def save_segmentation_zyx(
    seg_zyx: np.ndarray,
    path: str | Path,
    reference: "object",
) -> None:
    """Save a (Z, Y, X) segmentation as NIfTI, transposing back to (X, Y, Z).

    `reference` is the nibabel image whose affine/header should be reused
    (typically the input image returned by load_nifti_zyx).
    """
    nib = _require_nibabel()
    seg_xyz = seg_zyx.transpose(2, 1, 0)
    out = nib.Nifti1Image(seg_xyz, reference.affine, reference.header)
    nib.save(out, str(path))


def predict_nifti(
    engine: "InferenceEngine",
    in_path: str | Path,
    out_path: str | Path,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Run inference on a single NIfTI file and write the segmentation.

    Returns the segmentation in (Z, Y, X) order for callers that want it.
    """
    vol_zyx, img = load_nifti_zyx(in_path)
    logits = engine.predict(vol_zyx)
    seg_zyx = np.argmax(logits, axis=0).astype(dtype)
    save_segmentation_zyx(seg_zyx, out_path, img)
    return seg_zyx


def predict_folder(
    engine: "InferenceEngine",
    dir_in: str | Path,
    dir_out: str | Path,
    pattern: str = "*_0000.nii.gz",
    fallback_pattern: str = "*.nii.gz",
    progress: bool = False,
) -> list[Path]:
    """Run inference on every NIfTI in a folder.

    Mirrors nnU-Net's input convention: files are expected to end in
    ``_0000.nii.gz``; if none are found, falls back to ``*.nii.gz``.
    Output filenames have the channel suffix stripped.

    Returns the list of written segmentation paths.
    """
    dir_in = Path(dir_in)
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob(str(dir_in / pattern)))
    if not files:
        files = sorted(glob(str(dir_in / fallback_pattern)))

    written: list[Path] = []
    for fpath in files:
        fname = Path(fpath).name
        out_name = fname.replace("_0000.nii.gz", ".nii.gz")
        out_path = dir_out / out_name
        if progress:
            print(f"  Processing {fname}")
        predict_nifti(engine, fpath, out_path)
        written.append(out_path)
    return written
