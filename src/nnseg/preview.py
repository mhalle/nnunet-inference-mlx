"""Preview rendering: a three-plane overlay PNG generated at the one moment
it is nearly free - job completion, while the input is still staged and the
labels were just written.

Rules learned rendering by hand all week, now encoded once: both volumes go
through the same reader and the same reorientation operator (RAS), every
panel uses the physical per-plane aspect ratio, colors and names come from
the .seg.nrrd's own segment table so the preview can never disagree with the
artifact, and slices are chosen where the labels actually are (max label
area per plane). Windowing keys off the data: CT gets a soft-tissue window,
everything else a robust percentile stretch.

**The display convention is stated, derived, and drawn - never implied by
array order.** A RAS array shown with ``imshow`` puts the patient's right on
the image right (neurological) simply because that is the axis order the
loader chose, and nothing said so; the previews shipped that way for a week
before anyone noticed. So :data:`DISPLAY` names the convention
(``radiological``, 3D Slicer's default: patient right on the image left in
axial and coronal, sagittal viewed from the patient's left so anterior is on
the image left), :func:`display_planes` turns the convention into an
orientation *frame* per panel and gets there through
:func:`nnseg.io.orientation_transform` - the same DICOMOrient probe the
pipeline trusts, applied as a view - and the R/L, A/P, S/I edge letters are
read back from the resulting direction cosines, so the picture states what
it is and a wrong convention is visible rather than silent.

Everything here is best-effort by contract: callers wrap it in try/except,
because a preview must never fail a job. matplotlib is imported lazily and
its absence just means no preview.
"""
from pathlib import Path

__all__ = ["render_preview", "load_oriented_pair", "display_planes", "DISPLAY", "DISPLAY_FRAMES"]

#: The display convention every preview is drawn in. ``radiological`` is what
#: 3D Slicer and every reading room show; ``neurological`` is its mirror.
DISPLAY = "radiological"

#: Per-panel orientation frame for each convention, as a DICOM orientation code
#: naming the direction each array axis (x, y, z) points *toward*. Panels are
#: drawn with matplotlib ``origin="lower"``, so column 0 is the image left and
#: row 0 the image bottom. Radiological: patient Left toward the image right
#: (so right on the left), Anterior/Superior up; the sagittal is viewed from
#: the patient's left, so Posterior toward the image right. Neurological is the
#: RAS array as-is, sagittal viewed from the right.
DISPLAY_FRAMES = {
    "radiological": {"axial": "LAS", "coronal": "LAS", "sagittal": "LPS"},
    "neurological": {"axial": "RAS", "coronal": "RAS", "sagittal": "RAS"},
}

#: Which (row axis, column axis) of the frame's (x=0, y=1, z=2) each panel shows.
_PANEL_AXES = {"axial": (1, 0), "coronal": (2, 0), "sagittal": (2, 1)}
_OPPOSITE = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def load_oriented_pair(image, labels_path):
    """The shared load for completion artifacts: both volumes through the same
    reader and the same RAS operator, grayscale resampled onto the labels'
    grid for variants - done ONCE, then handed to preview and statistics
    (each was re-reading the gzip-compressed labelmap on its own, measured as
    part of a ~24 s per-job regression on 400M-voxel torsos)."""
    import numpy as np
    import SimpleITK as sitk

    from .io import read_image

    seg = sitk.ReadImage(str(labels_path))
    img = image if isinstance(image, sitk.Image) else read_image(image)
    img_r, seg_r = (sitk.DICOMOrient(v, "RAS") for v in (img, seg))
    if (img_r.GetSize() != seg_r.GetSize()
            or not np.allclose(img_r.GetSpacing(), seg_r.GetSpacing(), atol=1e-4)):
        img_r = sitk.Resample(img_r, seg_r, sitk.Transform(),
                              sitk.sitkLinear, -1024.0, img_r.GetPixelID())
    return seg, img_r, seg_r


def display_planes(gray, lab, geometry, *, display: str = DISPLAY) -> list:
    """The three preview panels for ``display``, derived from the geometry.

    ``gray`` and ``lab`` are (Z, Y, X) arrays sharing ``geometry`` (a
    :class:`nnseg.values.Geometry`, any orientation). For each panel the
    volumes are brought into the convention's frame (:data:`DISPLAY_FRAMES`)
    as a transpose+flip *view* - no copy, no resample, and no hand-written
    flip: the permutation comes from :func:`nnseg.io.orientation_transform`,
    i.e. from running ``DICOMOrient`` itself on a probe. The slice is then the
    one with the most labeled area, and the edge letters are read back from
    the frame's direction cosines, so they cannot disagree with the pixels.

    Returns ``[(name, gray2d, lab2d, aspect, markers), ...]`` in axial,
    coronal, sagittal order, where ``aspect`` is row spacing / column spacing
    and ``markers`` is ``{"left", "right", "bottom", "top"}`` -> letter.
    """
    import numpy as np
    import SimpleITK as sitk

    from . import io as nio

    frames = DISPLAY_FRAMES[display]           # KeyError for an unknown convention: loud
    fg_all = lab > 0
    out = []
    for name in ("axial", "coronal", "sagittal"):
        perm, flips, direction, spacing_xyz = nio.orientation_transform(geometry, frames[name])
        def view(a):
            v = np.transpose(a, perm)
            dims = [k for k, f in enumerate(flips) if f]
            return np.flip(v, axis=dims) if dims else v
        g, l, fg = view(gray), view(lab), view(fg_all)
        # the frame is the truth about the axes now: read the code back from the cosines
        code = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(direction)
        assert code == frames[name], (code, frames[name])
        row_ax, col_ax = _PANEL_AXES[name]
        slice_ax = ({0, 1, 2} - {row_ax, col_ax}).pop()
        # arrays are (Z, Y, X): frame axis k is array axis 2 - k
        area = fg.sum(axis=tuple(2 - a for a in (row_ax, col_ax)))
        idx = int(np.argmax(area))
        take = [slice(None)] * 3
        take[2 - slice_ax] = idx
        g2, l2 = g[tuple(take)], l[tuple(take)]
        if row_ax < col_ax:                    # keep rows = row_ax, columns = col_ax
            g2, l2 = g2.T, l2.T
        aspect = float(spacing_xyz[row_ax]) / float(spacing_xyz[col_ax])
        markers = {"right": code[col_ax], "left": _OPPOSITE[code[col_ax]],
                   "top": code[row_ax], "bottom": _OPPOSITE[code[row_ax]]}
        out.append((name, g2, l2, aspect, markers))
    return out


def render_preview(image, labels_path, out_png, *, title: str | None = None,
                   dpi: int = 110, pair=None, display: str = DISPLAY) -> Path | None:
    """Write a three-plane overlay PNG; returns the path, or None if the
    preview could not be made (missing matplotlib, unreadable inputs, ...).

    ``image`` is whatever the pipeline consumed - a path (file or DICOM
    directory) or a SimpleITK image. ``labels_path`` is the saved
    .seg.nrrd, whose metadata provides names and colors. ``display`` names
    the convention the panels are drawn in (see :data:`DISPLAY`).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import SimpleITK as sitk

        from .io import read_image

        seg, img_r, seg_r = pair or load_oriented_pair(image, labels_path)
        gray = sitk.GetArrayFromImage(img_r)       # native dtype: window per-slice
        lab = sitk.GetArrayFromImage(seg_r)
        if gray.shape != lab.shape:
            return None

        if float(gray.min()) < -200:               # CT: soft-tissue window
            def window(sl):
                return np.clip((sl.astype(np.float32) + 160.0) / 400.0, 0, 1)
        else:                                      # MR / PET: robust stretch
            lo, hi = np.percentile(gray, (1, 99.5))
            span = max(float(hi - lo), 1e-6)

            def window(sl):
                return np.clip((sl.astype(np.float32) - lo) / span, 0, 1)

        colors = {}
        for k in seg.GetMetaDataKeys():
            if k.startswith("Segment") and k.endswith("_Color"):
                i = k[len("Segment"):-len("_Color")]
                try:
                    v = int(seg.GetMetaData(f"Segment{i}_LabelValue"))
                except (ValueError, RuntimeError):
                    continue
                colors[v] = tuple(float(x) for x in seg.GetMetaData(k).split()[:3])
        present = [int(v) for v in np.unique(lab) if v]
        if not present:
            return None
        rgba = np.zeros((max(present) + 1, 4))
        for v in present:
            rgba[v] = (*colors.get(v, (1.0, 0.2, 1.0)), 0.5)

        from .io import geometry_of
        planes = display_planes(gray, lab, geometry_of(img_r), display=display)

        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), facecolor="black")
        for ax, (name, sl, ls, aspect, mk) in zip(axes, planes):
            ax.imshow(window(sl), cmap="gray", origin="lower", vmin=0, vmax=1, aspect=aspect)
            ax.imshow(rgba[np.clip(ls, 0, rgba.shape[0] - 1)], origin="lower", aspect=aspect)
            ax.set_title(name, fontsize=8, color="white")
            ax.axis("off")
            # the convention, written on the picture from the cosines it was drawn with
            for key, (x, y, ha, va) in {"left": (0.01, 0.5, "left", "center"),
                                        "right": (0.99, 0.5, "right", "center"),
                                        "top": (0.5, 0.99, "center", "top"),
                                        "bottom": (0.5, 0.01, "center", "bottom")}.items():
                ax.text(x, y, mk[key], color="white", fontsize=7, ha=ha, va=va,
                        transform=ax.transAxes)
        if title:
            fig.suptitle(title[:90], color="white", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94) if title else None)
        out = Path(out_png)
        fig.savefig(out, dpi=dpi, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        return out
    except Exception:
        return None
