"""Image IO — readers/writers between formats and :class:`Volume`.

Format plug-ins: bytes/paths/arrays in → ``Volume`` out, ``Segmentation`` /
``Volume`` → file out. Everything goes through in-memory SITK images; nothing
is forced through a temp file. Reading is always an explicit step (the reader
is the swappable seam) — there is no path-sniffing dispatch.

Also hosts the :class:`Geometry` ↔ SITK bridge (the one place SITK conventions
meet our channels-last ``(Z, Y, X, C)`` value types), so the value types
themselves stay backend-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from .values import Geometry, LabelSchema, Segmentation, Volume


def _require_sitk():
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SimpleITK is required for image IO. Install with "
            "`pip install nnunet-inference-mlx[preprocessing]`."
        ) from e


# ---------------------------------------------------------------------------
# Geometry ↔ SITK
# ---------------------------------------------------------------------------


def geometry_from_sitk(image) -> Geometry:
    """Build a :class:`Geometry` from a SITK image's metadata."""
    size_xyz = image.GetSize()                       # (X, Y, Z)
    spacing_xyz = image.GetSpacing()
    return Geometry(
        spacing_zyx=(spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]),
        shape_zyx=(size_xyz[2], size_xyz[1], size_xyz[0]),
        origin_xyz=tuple(image.GetOrigin()),
        direction_xyz=tuple(image.GetDirection()),
    )


def array_to_sitk(data_zyx: np.ndarray, geometry: Geometry):
    """Wrap a ``(Z, Y, X)`` numpy array as a SITK image with ``geometry``."""
    sitk = _require_sitk()
    img = sitk.GetImageFromArray(np.ascontiguousarray(data_zyx))
    img.SetSpacing((geometry.spacing_zyx[2], geometry.spacing_zyx[1], geometry.spacing_zyx[0]))
    img.SetOrigin(geometry.origin_xyz)
    img.SetDirection(geometry.direction_xyz)
    return img


def volume_to_sitk(volume: Volume):
    """A single-channel SITK image from a :class:`Volume`.

    Multi-channel volumes aren't supported on this path yet (used by the
    single-channel inference spine); pass a 1-channel volume.
    """
    if volume.num_channels != 1:
        raise NotImplementedError(
            f"volume_to_sitk handles single-channel volumes; got "
            f"{volume.num_channels} channels {volume.channels}"
        )
    data = np.asarray(volume.data[..., 0]).astype(np.float32, copy=False)
    return array_to_sitk(data, volume.geometry)


def sitk_to_volume(image, *, channels: Sequence[str] = ("CT",)) -> Volume:
    """A single-channel :class:`Volume` from a SITK image.

    The image is cast to **float32** here (clinical CT is typically int16).
    This is deliberate: everything downstream — the forward resample in
    ``preprocess.to_model_frame`` and the network input — works in float, so
    interpolated values are never rounded to integers. (The legacy
    ``predict_with_resampling`` resampled the raw int16 image, rounding
    interpolated HU; on real CT that flipped ~0.03% of boundary voxels at
    argmax. Float resampling matches nnU-Net v2's reference preprocessing.)
    """
    sitk = _require_sitk()
    arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)  # (Z, Y, X)
    data = mx.array(arr)[..., None]                                     # (Z, Y, X, 1)
    return Volume(data=data, geometry=geometry_from_sitk(image), channels=tuple(channels))


def sitk_to_segmentation(image, schema: LabelSchema) -> Segmentation:
    """A :class:`Segmentation` from a SITK integer label image."""
    sitk = _require_sitk()
    arr = sitk.GetArrayFromImage(image)                                # (Z, Y, X) int
    return Segmentation(data=mx.array(np.asarray(arr)),
                        geometry=geometry_from_sitk(image), schema=schema)


# ---------------------------------------------------------------------------
# Readers / writers (format plug-ins)
# ---------------------------------------------------------------------------


class NiftiReader:
    """Read a NIfTI file into a single-channel :class:`Volume`."""

    def read(self, path: str | Path, *, channels: Sequence[str] = ("CT",)) -> Volume:
        sitk = _require_sitk()
        return sitk_to_volume(sitk.ReadImage(str(path)), channels=channels)


class DicomReader:
    """Read a DICOM series *directory* into a single-channel :class:`Volume`."""

    def read(self, directory: str | Path, *, channels: Sequence[str] = ("CT",)) -> Volume:
        sitk = _require_sitk()
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(directory))
        if not files:
            raise FileNotFoundError(f"no DICOM series found in {directory}")
        reader.SetFileNames(files)
        return sitk_to_volume(reader.Execute(), channels=channels)


class ArrayReader:
    """Wrap an in-memory array (+ geometry) as a :class:`Volume`. No disk."""

    def read(self, array, geometry: Geometry, *,
             channels: Sequence[str] = ("CT",)) -> Volume:
        data = array if isinstance(array, mx.array) else mx.array(np.asarray(array))
        if data.ndim == 3:
            data = data[..., None]
        return Volume(data=data, geometry=geometry, channels=tuple(channels))


class NiftiWriter:
    """Write a :class:`Segmentation` or single-channel :class:`Volume` to NIfTI."""

    def write(self, path: str | Path, obj: Segmentation | Volume) -> None:
        sitk = _require_sitk()
        if isinstance(obj, Segmentation):
            img = array_to_sitk(np.asarray(obj.data), obj.geometry)
        else:
            img = volume_to_sitk(obj)
        sitk.WriteImage(img, str(path))


__all__ = [
    "geometry_from_sitk",
    "array_to_sitk",
    "volume_to_sitk",
    "sitk_to_volume",
    "sitk_to_segmentation",
    "NiftiReader",
    "DicomReader",
    "ArrayReader",
    "NiftiWriter",
]
