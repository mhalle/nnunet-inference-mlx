"""First-order per-structure statistics, computed at the one moment they are
nearly free - job completion, input staged, labels just written.

Same discipline as previews: both volumes through the same reader and the
same RAS reorientation, the grayscale resampled onto the labels' grid when a
grid variant differs from the input, names taken from the .seg.nrrd's own
segment table. Centroids are physical RAS millimeters. Intensity units are
detected (CT -> "hu", otherwise "intensity" - raw stored values; SUV scaling
is deliberately out of scope) and recorded both in the JSON and in the
derived TSV's column names.

Radiomics (texture features) is deliberately NOT here: its parameters change
the numbers and therefore belong in the result key, as a separately planned
artifact. Best-effort by contract - statistics must never fail a job.
"""
import json
from pathlib import Path

__all__ = ["compute_statistics", "statistics_tsv"]


def compute_statistics(image, labels_path, out_json) -> Path | None:
    """Write per-structure first-order statistics as JSON; returns the path,
    or None if they could not be computed."""
    try:
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
        gray = sitk.GetArrayFromImage(img_r).astype(np.float64)
        lab = sitk.GetArrayFromImage(seg_r).astype(np.int64)
        if gray.shape != lab.shape:
            return None

        names = {}
        for k in seg.GetMetaDataKeys():
            if k.startswith("Segment") and k.endswith("_Name"):
                i = k[len("Segment"):-len("_Name")]
                try:
                    v = int(seg.GetMetaData(f"Segment{i}_LabelValue"))
                except (ValueError, RuntimeError):
                    continue
                names[v] = seg.GetMetaData(k)

        unit = "hu" if float(gray.min()) < -200 else "intensity"
        sp = seg_r.GetSpacing()
        voxel_ml = float(sp[0] * sp[1] * sp[2]) / 1000.0

        flat = lab.ravel()
        K = int(flat.max()) + 1
        counts = np.bincount(flat, minlength=K)

        # one sort, then every per-label distribution is a contiguous slice
        order = np.argsort(flat, kind="stable")
        vals = gray.ravel()[order]
        starts = np.searchsorted(flat[order], np.arange(K))
        ends = np.concatenate([starts[1:], [flat.size]])

        # vectorized centroids: mean index per axis, then to physical RAS mm
        zz, yy, xx = np.indices(lab.shape, sparse=True)
        sums = {ax: np.bincount(flat, weights=w.ravel() if w.size == flat.size
                                else np.broadcast_to(w, lab.shape).ravel(),
                                minlength=K)
                for ax, w in (("z", np.broadcast_to(zz, lab.shape)),
                              ("y", np.broadcast_to(yy, lab.shape)),
                              ("x", np.broadcast_to(xx, lab.shape)))}

        structures = []
        for v in sorted(names):
            n = int(counts[v]) if v < K else 0
            if n == 0:
                continue
            seg_vals = vals[starts[v]:ends[v]]
            cz, cy, cx = (sums["z"][v] / n, sums["y"][v] / n, sums["x"][v] / n)
            pt = seg_r.TransformContinuousIndexToPhysicalPoint(
                (float(cx), float(cy), float(cz)))
            centroid_ras = [-pt[0], -pt[1], pt[2]]        # LPS -> RAS
            p10, med, p90 = np.percentile(seg_vals, (10, 50, 90))
            structures.append({
                "structure": names[v], "label": int(v), "voxels": n,
                "volume_ml": round(n * voxel_ml, 3),
                "mean": round(float(seg_vals.mean()), 3),
                "std": round(float(seg_vals.std()), 3),
                "min": round(float(seg_vals.min()), 3),
                "max": round(float(seg_vals.max()), 3),
                "median": round(float(med), 3),
                "p10": round(float(p10), 3), "p90": round(float(p90), 3),
                "centroid_ras_mm": [round(float(c), 2) for c in centroid_ras],
            })
        out = {"units": {"intensity": unit, "volume": "ml", "centroid": "mm (RAS)"},
               "grid_spacing_mm": [round(float(s), 4) for s in sp],
               "structures": structures}
        path = Path(out_json)
        path.write_text(json.dumps(out, indent=1))
        return path
    except Exception:
        return None


def statistics_tsv(stats: dict) -> str:
    """The cached statistics JSON rendered as a TSV: one row per structure,
    units carried in the column names (user decision) - volume_ml, mean_hu /
    mean_intensity, centroid_r_mm, ... Same numbers by construction."""
    u = (stats.get("units") or {}).get("intensity", "intensity")
    cols = ["structure", "label", "voxels", "volume_ml",
            f"mean_{u}", f"std_{u}", f"min_{u}", f"max_{u}",
            f"median_{u}", f"p10_{u}", f"p90_{u}",
            "centroid_r_mm", "centroid_a_mm", "centroid_s_mm"]
    lines = ["\t".join(cols)]
    for s in stats.get("structures", []):
        c = s.get("centroid_ras_mm") or ["", "", ""]
        row = [s.get("structure", ""), s.get("label", ""), s.get("voxels", ""),
               s.get("volume_ml", ""), s.get("mean", ""), s.get("std", ""),
               s.get("min", ""), s.get("max", ""), s.get("median", ""),
               s.get("p10", ""), s.get("p90", ""), c[0], c[1], c[2]]
        lines.append("\t".join(str(x) for x in row))
    return "\n".join(lines) + "\n"
