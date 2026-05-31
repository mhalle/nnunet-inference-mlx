"""preprocess — Volume-native steps that move an image into model frame.

Pure, composable wrappers over the proven SITK resampling/orientation
primitives in ``resampling.py``. Each takes a :class:`Volume` and returns a
new :class:`Volume` (or a Volume + a :class:`RestorePlan`). No GPU
allocation, no hidden state.

``to_model_frame`` is the headline: it reorients to a canonical orientation
and resamples to the model's training spacing, returning the model-frame
volume *and* the :class:`RestorePlan` that ``postprocess.restore`` uses to
get back to the caller's grid. The axis transpose
(``transpose_forward``/``backward``) is handled inside the engine at inference
time, so it never appears here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .imageio import (
    geometry_from_sitk,
    sitk_to_volume,
    volume_to_sitk,
)
from .values import RestorePlan, Volume

if TYPE_CHECKING:
    from .model_data import ModelData


def reorient(volume: Volume, code: str = "RAS") -> tuple[Volume, str]:
    """Reorient a single-channel :class:`Volume` to a DICOM orientation code.

    Returns the reoriented volume *and* the volume's original orientation
    code (so a caller can map results back). A no-op (returns the same data)
    when the volume is already in ``code``.
    """
    from .resampling import get_orientation, reorient as _reorient

    img = volume_to_sitk(volume)
    original = get_orientation(img)
    if original == code:
        return volume, original
    out = _reorient(img, code)
    return sitk_to_volume(out, channels=volume.channels), original


def resample(volume: Volume, spacing_zyx, *, interpolation: str = "linear") -> Volume:
    """Resample a single-channel :class:`Volume` to a target voxel spacing."""
    from .resampling import resample_image_to_target

    img = volume_to_sitk(volume)
    out = resample_image_to_target(img, tuple(spacing_zyx), interpolation=interpolation)
    return sitk_to_volume(out, channels=volume.channels)


def to_model_frame(
    volume: Volume,
    model_data: "ModelData",
    *,
    reorient_to: str | None = "RAS",
    interpolation: str = "linear",
) -> tuple[Volume, RestorePlan]:
    """Move a :class:`Volume` into the model's input frame.

    Reorients to ``reorient_to`` then resamples to ``model_data.target_spacing_zyx``.
    The default ``"RAS"`` is nnU-Net v2's universal canonical: its readers reorient
    every input to RAS before inference and back to the input's orientation on
    write (``nnunetv2.imageio.SimpleITKIO.read_images(orientation="RAS")`` via
    ``sitk.DICOMOrient``; ``NibabelIO`` likewise reorients to RAS). TotalSegmentator
    and MOOSE both run on nnU-Net v2, so RAS applies to all three ecosystems. The
    network is **not** left/right-equivariant, so it must see data in that trained
    orientation — feeding ``"LPS"`` mirrors L↔R and silently swaps left/right
    labels. (Override only for a model trained on an older nnU-Net whose reader
    did not canonicalize.) Returns the model-frame volume and a :class:`RestorePlan`
    capturing the inverse (the canonical source-spacing grid the logits map back
    onto, and the orientation to return to).

    ``reorient_to=None`` skips the reorient round-trip (only safe when the
    input is already canonical).
    """
    from .resampling import get_orientation, reorient as _reorient, resample_image_to_target

    img = volume_to_sitk(volume)
    source_orientation = get_orientation(img)
    if reorient_to is not None:
        img_canon = _reorient(img, reorient_to)
        inference_orientation = reorient_to
    else:
        img_canon = img
        inference_orientation = source_orientation

    inference_geometry = geometry_from_sitk(img_canon)
    model_spacing = tuple(model_data.target_spacing_zyx)
    resampled = resample_image_to_target(img_canon, model_spacing, interpolation=interpolation)
    model_vol = sitk_to_volume(resampled, channels=volume.channels)

    plan = RestorePlan(
        source_geometry=volume.geometry,
        source_orientation=source_orientation,
        inference_geometry=inference_geometry,
        inference_orientation=inference_orientation,
        model_spacing_zyx=model_spacing,
    )
    return model_vol, plan


__all__ = ["reorient", "resample", "to_model_frame"]
