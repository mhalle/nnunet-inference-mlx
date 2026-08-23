"""Image IO through SimpleITK - the same reader nnU-Net itself defaults to.

TotalSegmentator is ``Nifti1Image`` all the way down; nnU-Net has a reader registry whose
default is ``SimpleITKIO``, reorienting to RAS with ``sitk.DICOMOrient``. nnseg follows
nnU-Net: SimpleITK reads NIfTI, NRRD, MetaImage, DICOM series and more, carries direction
cosines (so oblique acquisitions survive the round trip), and hands back arrays already in
(Z, Y, X) order - no transposes.

Geometry is the toolkit's :class:`Geometry` value (spacing/shape in Z, Y, X; origin and
direction in SimpleITK's X, Y, Z), so the neutral core is shared rather than duplicated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CANONICAL = "RAS"


def _sitk():
    import SimpleITK as sitk
    return sitk


def geometry_of(image) -> "Geometry":
    from .values import Geometry
    return Geometry(
        spacing_zyx=tuple(float(s) for s in reversed(image.GetSpacing())),
        shape_zyx=tuple(int(s) for s in reversed(image.GetSize())),
        origin_xyz=tuple(float(o) for o in image.GetOrigin()),
        direction_xyz=tuple(float(d) for d in image.GetDirection()),
    )


def orientation_of(image) -> str:
    sitk = _sitk()
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())


def read(path) -> tuple[np.ndarray, "Geometry", str]:
    """Read any SimpleITK-supported image (or a DICOM series directory).

    Returns ``(array (Z, Y, X), canonical geometry, original orientation code)``: the array is
    already reoriented to RAS, which is the frame nnU-Net's readers put every input in and the
    frame the networks were trained in - feeding LPS mirrors left and right silently.
    """
    sitk = _sitk()
    p = Path(path)
    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(p))
        if not files:
            raise FileNotFoundError(f"no DICOM series found in {p}")
        reader.SetFileNames(files)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(p))
    if image.GetDimension() != 3:
        raise ValueError(f"expected a 3D image; {p} has {image.GetDimension()} dimensions")
    original = orientation_of(image)
    image = sitk.DICOMOrient(image, CANONICAL)
    return sitk.GetArrayFromImage(image), geometry_of(image), original


def to_image(array_zyx: np.ndarray, geometry: "Geometry"):
    """(Z, Y, X) array + geometry -> a SimpleITK image in the canonical frame."""
    sitk = _sitk()
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_zyx))
    image.SetSpacing(tuple(float(s) for s in reversed(geometry.spacing_zyx)))
    image.SetOrigin(tuple(float(o) for o in geometry.origin_xyz))
    image.SetDirection(tuple(float(d) for d in geometry.direction_xyz))
    return image


def restore_orientation(image, original: str):
    """Undo the canonical reorientation, so the output sits in the input's own frame."""
    return _sitk().DICOMOrient(image, original)


def write(image, path, *, compress: bool = True) -> None:
    _sitk().WriteImage(image, str(path), compress)
