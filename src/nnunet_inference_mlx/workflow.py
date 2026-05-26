"""Multi-stage segmentation workflows with FOV cropping between stages.

A *workflow* is a sequence of :class:`Stage`s. Each stage runs full inference
(forward resample → predict → inverse resample) via
:func:`predict_with_resampling`. Between stages, the output of the prior
stage can be used to crop the next stage's input to a tight bounding box
around structures of interest — the MOOSE-style cascade pattern.

Typical patterns
----------------

**Single-stage** (degenerate but supported)::

    seg = run_workflow(image, [Stage(engine=engine_full)])

**Two-stage cascade** (body detector → high-res organ model)::

    stages = [
        Stage(engine=engine_body, crop_to_classes=(BODY_TRUNK,)),
        Stage(engine=engine_organs),
    ]
    seg = run_workflow(image, stages)

**Three-stage** (body → liver → liver-segments)::

    stages = [
        Stage(engine=body_eng, crop_to_classes=(BODY_TRUNK,)),
        Stage(engine=organs_eng, crop_to_classes=(LIVER,)),
        Stage(engine=liver_seg_eng),
    ]
    seg = run_workflow(image, stages)

The geometric primitives (``Bbox``, ``compute_fg_bbox``, ``crop_image``,
``paste_segmentation``) are exported so callers building bespoke pipelines
(nnInteractive sub-volumes, manual FOV limiting, custom cascades) can use
the same building blocks.

Output geometry
---------------
The final segmentation is always returned in the *original* input image's
coordinate system. If any stage cropped, the cropped final-stage output is
pasted back into a full-shape canvas filled with background (``0``). The
returned SITK image has the same size, origin, spacing, and direction as
the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import numpy as np

from .resampling import _require_sitk, predict_with_resampling

if TYPE_CHECKING:
    from .engine import InferenceEngine
    import SimpleITK as sitk


__all__ = [
    "Bbox",
    "ParallelStage",
    "Stage",
    "compute_fg_bbox",
    "crop_image",
    "paste_segmentation",
    "run_label_union_workflow",
    "run_workflow",
]


# ---------------------------------------------------------------------------
# Bbox — voxel-coordinate bounding box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bbox:
    """Inclusive-start, exclusive-end voxel-coordinate bounding box.

    Axes are (Z, Y, X) to match the rest of the package's volume order.
    Indices are voxel indices in the volume the bbox refers to — the
    caller tracks which volume's coordinate space the bbox is in.
    """
    z_start: int
    z_end: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Shape of the cropped sub-volume."""
        return (
            self.z_end - self.z_start,
            self.y_end - self.y_start,
            self.x_end - self.x_start,
        )

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        """numpy-style slices for ``arr[bbox.slices]``."""
        return (
            slice(self.z_start, self.z_end),
            slice(self.y_start, self.y_end),
            slice(self.x_start, self.x_end),
        )

    def clamped(self, max_shape_zyx: tuple[int, int, int]) -> "Bbox":
        """Clamp to ``[0, max_shape_zyx)`` per axis."""
        return Bbox(
            z_start=max(0, self.z_start),
            z_end=min(max_shape_zyx[0], self.z_end),
            y_start=max(0, self.y_start),
            y_end=min(max_shape_zyx[1], self.y_end),
            x_start=max(0, self.x_start),
            x_end=min(max_shape_zyx[2], self.x_end),
        )

    def dilated(
        self,
        voxels_zyx: int | tuple[int, int, int],
        *,
        max_shape_zyx: tuple[int, int, int] | None = None,
    ) -> "Bbox":
        """Expand outward by ``voxels_zyx`` per axis. Optionally clamp to ``max_shape_zyx``."""
        if isinstance(voxels_zyx, int):
            dz = dy = dx = voxels_zyx
        else:
            dz, dy, dx = voxels_zyx
        new = Bbox(
            z_start=self.z_start - dz, z_end=self.z_end + dz,
            y_start=self.y_start - dy, y_end=self.y_end + dy,
            x_start=self.x_start - dx, x_end=self.x_end + dx,
        )
        if max_shape_zyx is not None:
            new = new.clamped(max_shape_zyx)
        return new

    def compose(self, sub: "Bbox") -> "Bbox":
        """Chain a sub-bbox (in this bbox's coords) into the outer coord system.

        If ``self`` represents the region of the original volume that the
        current crop occupies, and ``sub`` is a further crop expressed in
        the current crop's coordinates, this returns where ``sub`` lives
        in the original's coordinates.
        """
        return Bbox(
            z_start=self.z_start + sub.z_start,
            z_end=self.z_start + sub.z_end,
            y_start=self.y_start + sub.y_start,
            y_end=self.y_start + sub.y_end,
            x_start=self.x_start + sub.x_start,
            x_end=self.x_start + sub.x_end,
        )

    @classmethod
    def full(cls, shape_zyx: tuple[int, int, int]) -> "Bbox":
        """A bbox covering the whole volume of the given shape."""
        Z, Y, X = shape_zyx
        return cls(0, Z, 0, Y, 0, X)


# ---------------------------------------------------------------------------
# Bbox computation + crop/paste
# ---------------------------------------------------------------------------


def compute_fg_bbox(
    labels_zyx: np.ndarray,
    *,
    classes: Iterable[int] | None = None,
    dilation_mm: float = 0.0,
    spacing_zyx: tuple[float, float, float] | None = None,
) -> Bbox | None:
    """Find the foreground bounding box of a label volume.

    Parameters
    ----------
    labels_zyx :
        Integer label volume in (Z, Y, X) order.
    classes :
        Iterable of class IDs to treat as foreground. ``None`` (default)
        treats any nonzero label as foreground. Pass a tuple/list to crop
        around specific structures only (e.g. ``classes=(LIVER, SPLEEN)``).
    dilation_mm :
        Expand the bbox outward by this physical distance per axis. Requires
        ``spacing_zyx``. Common values: ``5`` for crisp-boundary organs
        (liver, kidneys), ``10`` for softer-boundary structures (muscle,
        fat) or when downstream resampling needs safety margin.
    spacing_zyx :
        Voxel spacing in mm, same axis order as ``labels_zyx``. Required
        when ``dilation_mm > 0``.

    Returns
    -------
    Bbox or None
        The bounding box, or ``None`` if no foreground voxels were found.
        ``None`` is the signal callers can use to skip cropping for the
        next stage when the target class wasn't detected.
    """
    if classes is None:
        fg = labels_zyx > 0
    else:
        # Don't force the classes array to the labels dtype — a user
        # asking about class 999 on a uint8 label volume should get
        # "not found", not an OverflowError. Filter out values outside
        # the labels dtype's range, then build a same-dtype index.
        info = np.iinfo(labels_zyx.dtype)
        in_range = [c for c in classes if info.min <= c <= info.max]
        if not in_range:
            return None
        fg = np.isin(labels_zyx, np.asarray(in_range, dtype=labels_zyx.dtype))

    if not fg.any():
        return None

    # Per-axis projections — much cheaper than np.where on the full mask.
    z_any = fg.any(axis=(1, 2))
    y_any = fg.any(axis=(0, 2))
    x_any = fg.any(axis=(0, 1))

    z_indices = np.where(z_any)[0]
    y_indices = np.where(y_any)[0]
    x_indices = np.where(x_any)[0]

    bbox = Bbox(
        z_start=int(z_indices[0]),
        z_end=int(z_indices[-1]) + 1,
        y_start=int(y_indices[0]),
        y_end=int(y_indices[-1]) + 1,
        x_start=int(x_indices[0]),
        x_end=int(x_indices[-1]) + 1,
    )

    if dilation_mm > 0:
        if spacing_zyx is None:
            raise ValueError(
                "spacing_zyx is required when dilation_mm > 0 "
                "(needed to convert mm to voxels)."
            )
        dilation_vox = tuple(
            max(1, int(round(dilation_mm / s))) for s in spacing_zyx
        )
        bbox = bbox.dilated(dilation_vox, max_shape_zyx=labels_zyx.shape)

    return bbox


def crop_image(image_sitk: "sitk.Image", bbox: Bbox) -> "sitk.Image":
    """Extract a sub-volume from a SITK image, preserving world geometry.

    Origin is shifted so the returned image's voxel (0,0,0) maps to the
    same world coordinate as the cropped voxel in the input. Spacing and
    direction are preserved. Use :func:`paste_segmentation` to project a
    cropped-space label volume back into the original's voxel grid.
    """
    sitk = _require_sitk()
    # SITK axis order is (X, Y, Z); our Bbox is (Z, Y, X).
    extract_index_xyz = [bbox.x_start, bbox.y_start, bbox.z_start]
    extract_size_xyz = [
        bbox.x_end - bbox.x_start,
        bbox.y_end - bbox.y_start,
        bbox.z_end - bbox.z_start,
    ]
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(extract_index_xyz)
    roi.SetSize(extract_size_xyz)
    return roi.Execute(image_sitk)


def paste_segmentation(
    small_seg_zyx: np.ndarray,
    full_shape_zyx: tuple[int, int, int],
    bbox: Bbox,
    *,
    fill: int = 0,
) -> np.ndarray:
    """Paste a cropped-space label volume back into a full-shape canvas.

    The canvas is filled with ``fill`` (default ``0`` = background), then
    ``small_seg_zyx`` is written into the slot defined by ``bbox``.
    ``small_seg_zyx.shape`` must equal ``bbox.shape_zyx``.
    """
    if small_seg_zyx.shape != bbox.shape_zyx:
        raise ValueError(
            f"small_seg shape {small_seg_zyx.shape} does not match "
            f"bbox shape {bbox.shape_zyx}"
        )
    out = np.full(full_shape_zyx, fill, dtype=small_seg_zyx.dtype)
    out[bbox.slices] = small_seg_zyx
    return out


# ---------------------------------------------------------------------------
# Workflow orchestrator
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    """One stage of a multi-stage segmentation workflow.

    Parameters
    ----------
    engine :
        An already-built :class:`InferenceEngine`. Use
        :func:`cached_engine_from_task` or :func:`cached_engine_from_folder`
        from the engine_cache module to compose with the cache.
    crop_to_classes :
        If set, after this stage runs, the foreground bbox of these class
        IDs in this stage's output is used to crop the *next* stage's
        input. ``None`` (default) means no cropping after this stage
        (the next stage gets the same input this stage saw).
    dilation_mm :
        Safety margin added to the bbox in physical units (mm) on every
        axis. Default ``10`` is generous enough for soft-tissue structures
        and clips minimally on crisp-boundary organs. Set lower for
        tighter cascade memory; higher when downstream resampling needs
        more context.
    interpolation :
        Forward-resample interpolation for this stage. ``"linear"``
        (default), ``"bspline"``, or ``"nearest"``.
    peak_working_memory_mb :
        Memory budget for the inverse-resample slab loop in this stage.
        ``None`` (default) auto-tiers from system RAM.
    remove_small_components_mm3 :
        If > 0, drop label components below this physical volume from
        this stage's output. ``0`` (default) disables. Requires the
        ``[postprocessing]`` extra.
    """
    engine: "InferenceEngine"
    crop_to_classes: tuple[int, ...] | None = None
    dilation_mm: float = 10.0
    interpolation: str = "linear"
    peak_working_memory_mb: int | None = None
    remove_small_components_mm3: float = 0.0


def run_workflow(
    image_sitk: "sitk.Image",
    stages: list[Stage],
    *,
    verbose: bool = False,
) -> "sitk.Image":
    """Run a multi-stage workflow with optional FOV cropping between stages.

    Each stage runs full :func:`predict_with_resampling` (forward resample
    on CPU, inference on Metal, inverse resample on Metal) on its input.
    When a stage has ``crop_to_classes`` set, the foreground bbox of those
    classes in that stage's output (in voxel coords) is used to crop the
    *next* stage's input via :func:`crop_image`.

    The final output is pasted back into the original image's voxel grid
    if any cropping happened. The returned image has the same geometry
    (size, origin, spacing, direction) as ``image_sitk``.

    Parameters
    ----------
    image_sitk :
        Input image at acquisition spacing.
    stages :
        Ordered list of :class:`Stage`s. The last stage's
        ``crop_to_classes`` is ignored (nothing follows it to crop).
    verbose :
        Print per-stage timing and bbox info.

    Returns
    -------
    sitk.Image
        Segmentation at acquisition spacing, geometry matching the input.

    Raises
    ------
    ValueError
        If ``stages`` is empty.
    """
    sitk = _require_sitk()
    if not stages:
        raise ValueError("stages must be a non-empty list.")

    # --- Reorient to canonical LPS once for the whole workflow --------
    # Each stage's predict_with_resampling would otherwise reorient
    # independently; doing it here amortizes the reorient cost across
    # all stages and ensures every stage's crop bbox is computed in
    # the same canonical voxel grid.
    from .resampling import get_orientation, reorient
    original_orient = get_orientation(image_sitk)
    image_sitk_canonical = reorient(image_sitk, "LPS")

    in_size_xyz = image_sitk_canonical.GetSize()
    original_shape_zyx = (in_size_xyz[2], in_size_xyz[1], in_size_xyz[0])

    # cumulative_bbox tracks where the current input lives in the ORIGINAL
    # (canonical) image's voxel coords. Initially the full volume.
    cumulative_bbox = Bbox.full(original_shape_zyx)
    current_input = image_sitk_canonical
    current_seg: "sitk.Image" | None = None

    for i, stage in enumerate(stages):
        if verbose:
            print(
                f"[workflow] stage {i+1}/{len(stages)}: "
                f"input shape ZYX={(current_input.GetSize()[2], current_input.GetSize()[1], current_input.GetSize()[0])}"
            )

        # reorient_to=None skips the per-stage reorient (we already did
        # it once at the workflow boundary).
        current_seg = predict_with_resampling(
            stage.engine,
            current_input,
            interpolation=stage.interpolation,
            peak_working_memory_mb=stage.peak_working_memory_mb,
            remove_small_components_mm3=stage.remove_small_components_mm3,
            reorient_to=None,
        )

        # If this is the last stage, no point computing a next-stage crop.
        if i == len(stages) - 1:
            break

        if stage.crop_to_classes is None:
            if verbose:
                print(f"[workflow] stage {i+1}: no crop_to_classes, passing input through")
            continue

        seg_array = sitk.GetArrayFromImage(current_seg)
        spacing_xyz = current_seg.GetSpacing()
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

        sub_bbox = compute_fg_bbox(
            seg_array,
            classes=stage.crop_to_classes,
            dilation_mm=stage.dilation_mm,
            spacing_zyx=spacing_zyx,
        )

        if sub_bbox is None:
            if verbose:
                print(
                    f"[workflow] stage {i+1}: classes {stage.crop_to_classes} not "
                    f"found in output; skipping crop for next stage"
                )
            continue

        if verbose:
            print(
                f"[workflow] stage {i+1}: crop sub-bbox shape ZYX={sub_bbox.shape_zyx} "
                f"(classes={stage.crop_to_classes}, dilation_mm={stage.dilation_mm})"
            )

        current_input = crop_image(current_input, sub_bbox)
        cumulative_bbox = cumulative_bbox.compose(sub_bbox)

    # Paste final seg back if any cropping happened.
    assert current_seg is not None
    if cumulative_bbox.shape_zyx == original_shape_zyx:
        out_sitk = current_seg
    else:
        seg_array = sitk.GetArrayFromImage(current_seg)
        out_array = paste_segmentation(
            seg_array, original_shape_zyx, cumulative_bbox,
        )
        out_sitk = sitk.GetImageFromArray(out_array)
        out_sitk.CopyInformation(image_sitk_canonical)

    # --- Reorient final segmentation back to caller's orientation -----
    # `reorient` is a no-op when original_orient == "LPS".
    out_sitk = reorient(out_sitk, original_orient)

    return out_sitk


# ---------------------------------------------------------------------------
# Multi-task label-union orchestrator (TS full-mode pattern)
# ---------------------------------------------------------------------------


@dataclass
class ParallelStage:
    """One task in a multi-task label-union workflow.

    Each task runs independently against the same input volume, predicts
    a task-local label segmentation, gets its labels remapped into a
    shared unified label space, and paints into a running union.

    The "parallel" in the name is logical, not literal: on a single
    Apple Silicon GPU the tasks still run sequentially (Metal serializes
    GPU work). The point of the dataclass is that the tasks are
    *independent* — no inter-stage cropping, no dependency graph — and
    fold together by paint priority into a single output.

    Parameters
    ----------
    engine :
        An already-built :class:`InferenceEngine`. Use
        :func:`cached_engine_from_task` to amortize load cost across
        tasks that share weights.
    label_remap :
        ``{task_local_id: unified_id}`` lookup table. Task-local IDs not
        in this dict become background in the unified output — explicit
        drop. See :func:`remap_labels`.
    part_name :
        Optional human-readable name for logging (e.g. ``"vertebrae"``,
        ``"organs"``). Not used by the orchestrator.
    """

    engine: "InferenceEngine"
    label_remap: dict[int, int]
    part_name: str | None = None


def run_label_union_workflow(
    image_sitk: "sitk.Image",
    stages: list[ParallelStage],
    *,
    peak_working_memory_mb: int | None = None,
    reorient_to: str | None = "LPS",
    verbose: bool = False,
) -> "sitk.Image":
    """Run independent multi-task inference and fold outputs by paint priority.

    This is the TotalSegmentator full-mode pattern: each "part" (vertebrae,
    organs, ribs, cardiac, muscles, ...) is a separate trained model that
    predicts in its own task-local label space; the user wants a single
    segmentation in a shared 117-class space. Each stage's task-local IDs
    are remapped into the unified space via ``label_remap``, then painted
    into a running union — last stage in the list wins overlapping voxels.

    Unlike :func:`run_workflow` (sequential cascade with inter-stage cropping),
    this orchestrator runs every stage against the *same* input. There's no
    inter-stage data dependency.

    The recipe is intentionally thin — every step is a public top-level
    function the caller could call themselves:

    1. ``reorient`` to canonical (``reorient_to``, default ``"LPS"``)
    2. For each stage:

       - :func:`predict_with_resampling` (returns SITK seg at input geometry)
       - :func:`~nnunet_inference_mlx.remap_labels` (task IDs → unified IDs)
       - :func:`~nnunet_inference_mlx.paint_union` (overwrite-where-nonzero)

    3. ``reorient`` back to original

    Callers who want a non-standard variant (custom paint priority, hold
    all intermediate logits in RAM and do logit-confidence merging,
    serialize intermediates, etc.) build the same recipe themselves from
    the primitives — this function isn't doing anything they can't do.

    Parameters
    ----------
    image_sitk :
        Input SITK image at any orientation / spacing.
    stages :
        List of :class:`ParallelStage`. Order matters: later stages
        overwrite earlier stages at voxels where both fire. Empty list
        raises ``ValueError``.
    peak_working_memory_mb :
        Memory budget passed to each stage's inverse-resample slab loop.
        ``None`` (default) auto-tiers from system RAM.
    reorient_to :
        Canonical orientation. ``"LPS"`` (default) for nnU-Net / TS.
        Pass ``None`` to skip the reorient round-trip (only safe when
        the input is already in canonical orientation).
    verbose :
        Print per-stage progress.

    Returns
    -------
    sitk.Image
        Unified segmentation at the original input's geometry (size,
        spacing, origin, direction). Dtype is the smallest unsigned int
        that fits every unified target ID across all stages.

    Raises
    ------
    ValueError
        If ``stages`` is empty.

    Examples
    --------
    The TotalSegmentator full-mode 5-part union (sketch)::

        from nnunet_inference_mlx import (
            cached_engine_from_task, ParallelStage, run_label_union_workflow,
        )

        # Each engine is a different TS part model. The remaps go from
        # task-local labels into the 117-class unified TS scheme.
        stages = [
            ParallelStage(cached_engine_from_task(291), TS_REMAP_ORGANS,    "organs"),
            ParallelStage(cached_engine_from_task(292), TS_REMAP_VERTEBRAE, "vertebrae"),
            ParallelStage(cached_engine_from_task(293), TS_REMAP_CARDIAC,   "cardiac"),
            ParallelStage(cached_engine_from_task(294), TS_REMAP_MUSCLES,   "muscles"),
            ParallelStage(cached_engine_from_task(295), TS_REMAP_RIBS,      "ribs"),
        ]
        seg = run_label_union_workflow(image, stages)
    """
    from .labels import paint_union, remap_labels
    from .resampling import get_orientation, reorient

    sitk = _require_sitk()
    if not stages:
        raise ValueError("stages must be a non-empty list.")

    # --- Reorient once at boundary ------------------------------------
    if reorient_to is not None:
        original_orient = get_orientation(image_sitk)
        image_canonical = reorient(image_sitk, reorient_to)
    else:
        original_orient = None
        image_canonical = image_sitk

    in_size_xyz = image_canonical.GetSize()
    out_shape_zyx = (in_size_xyz[2], in_size_xyz[1], in_size_xyz[0])

    # --- Pick unified dtype from the union of all stages' target IDs --
    # Use the smallest unsigned int that fits every unified ID across
    # all stages. TS's 117-class scheme fits uint8; cohorts with more
    # tasks may push to uint16.
    max_target = 0
    for s in stages:
        if s.label_remap:
            max_target = max(max_target, max(int(v) for v in s.label_remap.values()))
    if max_target <= np.iinfo(np.uint8).max:
        out_dtype = np.dtype(np.uint8)
    elif max_target <= np.iinfo(np.uint16).max:
        out_dtype = np.dtype(np.uint16)
    else:
        out_dtype = np.dtype(np.uint32)

    unified = np.zeros(out_shape_zyx, dtype=out_dtype)

    # --- Per-stage: predict → remap → paint --------------------------
    for i, stage in enumerate(stages):
        if verbose:
            name = stage.part_name or f"<stage {i+1}>"
            print(
                f"[label_union] stage {i+1}/{len(stages)} ({name}): "
                f"input ZYX={out_shape_zyx}"
            )

        # reorient_to=None — we already did the reorient once at boundary.
        seg_img = predict_with_resampling(
            stage.engine, image_canonical,
            reorient_to=None,
            peak_working_memory_mb=peak_working_memory_mb,
        )
        seg_array = sitk.GetArrayFromImage(seg_img)

        remapped = remap_labels(seg_array, stage.label_remap, out_dtype=out_dtype)
        paint_union(unified, remapped)

    # --- Wrap as SITK at canonical geometry ---------------------------
    out_sitk = sitk.GetImageFromArray(unified)
    out_sitk.CopyInformation(image_canonical)

    # --- Reorient back ------------------------------------------------
    if original_orient is not None:
        out_sitk = reorient(out_sitk, original_orient)

    return out_sitk
