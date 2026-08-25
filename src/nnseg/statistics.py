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


def compute_statistics(image, labels_path, out_json, *, pair=None) -> Path | None:
    """Write per-structure first-order statistics as JSON; returns the path,
    or None if they could not be computed. ``pair`` shares one load with the
    preview (see preview.load_oriented_pair)."""
    try:
        import numpy as np
        import SimpleITK as sitk

        from .preview import load_oriented_pair

        seg, img_r, seg_r = pair or load_oriented_pair(image, labels_path)
        gray = sitk.GetArrayFromImage(img_r)       # native dtype; cast subsets only
        lab = sitk.GetArrayFromImage(seg_r)
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

        # only labeled voxels matter: on a torso CT that is ~10-20 % of the
        # volume, and sorting 40M beats sorting 400M by an order of magnitude
        # (the full-volume argsort was measured costing ~20 s per job)
        fz, fy, fx = np.nonzero(lab)
        labs = lab[fz, fy, fx].astype(np.int32)
        vals_fg = gray[fz, fy, fx].astype(np.float32)
        K = int(labs.max()) + 1 if labs.size else 1
        counts = np.bincount(labs, minlength=K)

        order = np.argsort(labs, kind="stable")
        vals = vals_fg[order]
        starts = np.searchsorted(labs[order], np.arange(K))
        ends = np.concatenate([starts[1:], [labs.size]])

        # centroids from the foreground indices alone
        sums = {"z": np.bincount(labs, weights=fz, minlength=K),
                "y": np.bincount(labs, weights=fy, minlength=K),
                "x": np.bincount(labs, weights=fx, minlength=K)}

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
