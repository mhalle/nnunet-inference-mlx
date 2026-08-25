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

Everything here is best-effort by contract: callers wrap it in try/except,
because a preview must never fail a job. matplotlib is imported lazily and
its absence just means no preview.
"""
from pathlib import Path

__all__ = ["render_preview"]


def render_preview(image, labels_path, out_png, *, title: str | None = None,
                   dpi: int = 110) -> Path | None:
    """Write a three-plane overlay PNG; returns the path, or None if the
    preview could not be made (missing matplotlib, unreadable inputs, ...).

    ``image`` is whatever the pipeline consumed - a path (file or DICOM
    directory) or a SimpleITK image. ``labels_path`` is the saved
    .seg.nrrd, whose metadata provides names and colors.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import SimpleITK as sitk

        from .io import read_image

        seg = sitk.ReadImage(str(labels_path))
        img = image if isinstance(image, sitk.Image) else read_image(image)
        img_r, seg_r = (sitk.DICOMOrient(v, "RAS") for v in (img, seg))
        gray = sitk.GetArrayFromImage(img_r).astype(np.float32)
        lab = sitk.GetArrayFromImage(seg_r).astype(np.int32)
        if gray.shape != lab.shape:
            return None

        if gray.min() < -200:                      # CT: soft-tissue window
            gray = np.clip((gray + 160.0) / 400.0, 0, 1)
        else:                                      # MR / PET: robust stretch
            lo, hi = np.percentile(gray, (1, 99.5))
            gray = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)

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

        fg = lab > 0
        sx, sy, sz = img_r.GetSpacing()
        planes = [(int(np.argmax(fg.sum(axis=(1, 2)))), None, None, "axial", sy / sx),
                  (None, int(np.argmax(fg.sum(axis=(0, 2)))), None, "coronal", sz / sx),
                  (None, None, int(np.argmax(fg.sum(axis=(0, 1)))), "sagittal", sz / sy)]

        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), facecolor="black")
        for ax, (z, y, x, name, aspect) in zip(axes, planes):
            sl = gray[z] if z is not None else gray[:, y, :] if y is not None else gray[:, :, x]
            ls = lab[z] if z is not None else lab[:, y, :] if y is not None else lab[:, :, x]
            ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1, aspect=aspect)
            ax.imshow(rgba[np.clip(ls, 0, rgba.shape[0] - 1)], origin="lower", aspect=aspect)
            ax.set_title(name, fontsize=8, color="white")
            ax.axis("off")
        if title:
            fig.suptitle(title[:90], color="white", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94) if title else None)
        out = Path(out_png)
        fig.savefig(out, dpi=dpi, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        return out
    except Exception:
        return None
