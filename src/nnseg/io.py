"""Image IO through SimpleITK - the same reader nnU-Net itself defaults to.

TotalSegmentator is ``Nifti1Image`` all the way down; nnU-Net has a reader registry whose
default is ``SimpleITKIO``. nnseg follows
nnU-Net: SimpleITK reads NIfTI, NRRD, MetaImage, DICOM series and more, carries direction
cosines (so oblique acquisitions survive the round trip), and hands back arrays already in
(Z, Y, X) order - no transposes.

Orientation is **not** uniform across the ecosystem: TotalSegmentator canonicalizes to RAS,
while nnU-Net's default readers keep the stored axis order and only the opt-in ``*WithReorient``
variants canonicalize. :func:`read` takes ``reorient`` and :func:`reader_reorients` reads the
model's declared choice out of its plans.

Geometry is the toolkit's :class:`Geometry` value (spacing/shape in Z, Y, X; origin and
direction in SimpleITK's X, Y, Z), so the neutral core is shared rather than duplicated.
"""
from __future__ import annotations

from pathlib import Path

from .errors import InputError

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


def read(path, *, reorient: bool = True) -> tuple[np.ndarray, "Geometry", str]:
    """Read any SimpleITK-supported image (or a DICOM series directory).

    Returns ``(array (Z, Y, X), geometry, original orientation code)``.

    With ``reorient=True`` (the default, and what TotalSegmentator does) the array comes back
    in RAS; feeding a model LPS data mirrors left and right silently. But **nnU-Net's default
    reader does not reorient**: ``SimpleITKIO`` hands the array over in its stored axis order,
    and only the opt-in ``SimpleITKIOWithReorient`` / ``NibabelIOWithReorient`` canonicalize.
    A model trained through the plain reader therefore expects its own acquisition orientation,
    so pass ``reorient=False`` for those - see :func:`reader_reorients`.
    """
    sitk = _sitk()
    p = Path(path)
    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(p))
        if not files:
            raise InputError(f"no DICOM series found in {p}")
        reader.SetFileNames(files)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(p))
    if image.GetDimension() != 3:
        raise InputError(f"expected a 3D image; {p} has {image.GetDimension()} dimensions")
    original = orientation_of(image)
    if reorient:
        image = sitk.DICOMOrient(image, CANONICAL)
    return sitk.GetArrayFromImage(image), geometry_of(image), original


def reader_reorients(model_folder) -> bool:
    """Does this nnU-Net model's declared reader reorient to RAS?

    The plans name an ``image_reader_writer`` (``dataset.json``'s ``overwrite_image_reader_writer``
    wins when present). Only the ``*WithReorient`` variants canonicalize; ``SimpleITKIO`` and
    ``NibabelIO`` - the defaults, and what almost every trained model declares - do not. Getting
    this wrong mirrors left and right, which shows up as paired structures scoring Dice 0 while
    unpaired ones merely degrade.
    """
    import json
    f = Path(model_folder)
    name = None
    ds = f / "dataset.json"
    if ds.exists():
        name = json.loads(ds.read_text()).get("overwrite_image_reader_writer")
    if not name:
        pl = f / "plans.json"
        if pl.exists():
            name = json.loads(pl.read_text()).get("image_reader_writer")
    return bool(name) and "reorient" in str(name).lower()


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


def orientation_transform(geometry: "Geometry", target: str) -> tuple[tuple[int, int, int], tuple[bool, bool, bool], tuple[float, ...], tuple[float, float, float]]:
    """What ``DICOMOrient`` would do to an array with this geometry, as a torch-applicable recipe.

    Returns ``(perm, flips, direction_xyz, spacing_xyz)``: the new (Z, Y, X) axis ``k`` is the
    old axis ``perm[k]``, reversed if ``flips[k]``. Derived by running ``DICOMOrient`` itself on
    a 3x3x3 probe whose voxel values encode their own index, so nothing of SimpleITK's
    orientation logic is re-implemented - the probe *is* the answer. Direction and spacing of
    the result are size-independent, so the probe's are the real ones; the origin is not, and
    :func:`reorient` computes it from the real size.
    """
    sitk = _sitk()
    probe = sitk.GetImageFromArray(np.arange(27, dtype=np.int32).reshape(3, 3, 3))
    probe.SetSpacing(tuple(float(s) for s in reversed(geometry.spacing_zyx)))
    probe.SetOrigin(tuple(float(o) for o in geometry.origin_xyz))
    probe.SetDirection(tuple(float(d) for d in geometry.direction_xyz))
    out = sitk.DICOMOrient(probe, target)
    arr = sitk.GetArrayFromImage(out)                               # (3, 3, 3), values = old flat index
    old_idx = np.stack(np.unravel_index(arr, (3, 3, 3)), axis=-1)  # (3, 3, 3, 3): old (z, y, x) per new voxel
    perm, flips = [], []
    for k in range(3):
        first = old_idx[tuple(0 if a != k else 0 for a in range(3))]
        last = old_idx[tuple(0 if a != k else 2 for a in range(3))]
        delta = last - first
        (axis,) = np.nonzero(delta)[0]
        perm.append(int(axis))
        flips.append(bool(delta[axis] < 0))
    return tuple(perm), tuple(flips), tuple(float(d) for d in out.GetDirection()), tuple(float(s) for s in out.GetSpacing())


def reorient(array_zyx, geometry: "Geometry", target: str):
    """Reorient a (Z, Y, X) array - numpy or torch, on any device - to ``target`` exactly as
    ``DICOMOrient`` would, but as a permute + flip on the tensor where it lives.

    On a 418 M-voxel label volume ``DICOMOrient`` is ~4 s of single-threaded CPU; the same
    permutation on the GPU is milliseconds. Returns ``(array, Geometry)`` with the array on
    the host (numpy), ready for :func:`to_image`.
    """
    from .values import Geometry
    perm, flips, direction, spacing_xyz = orientation_transform(geometry, target)
    is_torch = hasattr(array_zyx, "permute")
    shape_old = tuple(int(n) for n in array_zyx.shape)
    if is_torch:
        t = array_zyx.permute(*perm)
        dims = [k for k, f in enumerate(flips) if f]
        if dims:
            t = t.flip(dims)
        out = t.contiguous().cpu().numpy()
    else:
        out = np.transpose(np.asarray(array_zyx), perm)
        for k, f in enumerate(flips):
            if f:
                out = np.flip(out, axis=k)
        out = np.ascontiguousarray(out)
    # origin: world position of the new voxel (0, 0, 0) = the old voxel at index 0, or n-1 on a
    # flipped axis, along each old axis
    old_idx_zyx = np.zeros(3)
    for k, (p_, f) in enumerate(zip(perm, flips)):
        old_idx_zyx[p_] = (shape_old[p_] - 1) if f else 0
    d = np.asarray(geometry.direction_xyz, dtype=np.float64).reshape(3, 3)
    sp_xyz = np.asarray(geometry.spacing_zyx, dtype=np.float64)[::-1]
    origin = np.asarray(geometry.origin_xyz, dtype=np.float64) + d @ (old_idx_zyx[::-1] * sp_xyz)
    geo = Geometry(spacing_zyx=tuple(reversed(spacing_xyz)), shape_zyx=tuple(int(n) for n in out.shape),
                   origin_xyz=tuple(float(o) for o in origin), direction_xyz=direction)
    return out, geo


def write(image, path, *, compress: bool = True) -> None:
    _sitk().WriteImage(image, str(path), compress)
