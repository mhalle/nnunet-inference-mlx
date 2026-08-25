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

        # Only labeled voxels matter (~10-20 % of a torso). One flatnonzero,
        # coordinates by arithmetic - no triple index-array construction.
        Z, Y, X = lab.shape
        flat = lab.ravel()
        idx = np.flatnonzero(flat)
        labs = flat[idx].astype(np.int64)
        keep = labs > 0                    # segment ids are positive; a signed
        if not keep.all():                 # labelmap's negatives are not
            idx, labs = idx[keep], labs[keep]   # structures (and crash bincount)
        vals_fg = gray.ravel()[idx]
        K = int(labs.max()) + 1 if labs.size else 1
        counts = np.bincount(labs, minlength=K)

        fz = idx // (Y * X)
        rem = idx - fz * (Y * X)
        fy = rem // X
        fx = rem - fy * X
        sums = {"z": np.bincount(labs, weights=fz, minlength=K),
                "y": np.bincount(labs, weights=fy, minlength=K),
                "x": np.bincount(labs, weights=fx, minlength=K)}

        # Integer intensities (every native CT/MR grid): exact statistics from
        # a per-label histogram - one bincount instead of a 60M-element sort.
        # Percentiles are nearest-rank on the exact distribution. Float grids
        # (resampled variants) take the sort path below.
        hist = None
        if labs.size and np.issubdtype(vals_fg.dtype, np.integer):
            vmin = int(vals_fg.min())
            nb = int(vals_fg.max()) - vmin + 1
            # K*nb is the histogram's footprint; nb alone was capped, but K is
            # max_label_id+1 and a stray id in a stock model's dataset.json can
            # blow it up (a daemon-thread MemoryError is swallowed; a cgroup
            # OOM-kill is not). Fall to the sort path past ~128 MB.
            if 0 < nb <= 1 << 16 and K * nb <= 1 << 24:
                keys = labs * nb + (vals_fg.astype(np.int64) - vmin)
                hist = np.bincount(keys, minlength=K * nb).reshape(K, nb)
                bin_vals = np.arange(vmin, vmin + nb, dtype=np.float64)
        if hist is None:
            order = np.argsort(labs, kind="stable")
            vals = vals_fg[order].astype(np.float32)
            starts = np.searchsorted(labs[order], np.arange(K))
            ends = np.concatenate([starts[1:], [labs.size]])

        structures = []
        for v in sorted(names):
            n = int(counts[v]) if v < K else 0
            if n == 0:
                continue
            if hist is not None:
                h = hist[v]
                cum = h.cumsum()
                s1 = float((h * bin_vals).sum())
                s2 = float((h * bin_vals * bin_vals).sum())
                mean = s1 / n
                var = max(s2 / n - mean * mean, 0.0)
                nz = np.flatnonzero(h)
                vmin_v, vmax_v = bin_vals[nz[0]], bin_vals[nz[-1]]
                p10, med, p90 = (bin_vals[np.searchsorted(cum, q * n, side="left")]
                                 for q in (0.10, 0.50, 0.90))
                stats_row = (mean, var ** 0.5, vmin_v, vmax_v, med, p10, p90)
            else:
                seg_vals = vals[starts[v]:ends[v]]
                sv = np.sort(seg_vals)     # nearest-rank, matching the
                                           # histogram path's searchsorted(cum,
                                           # q*n, "left") on value-ordered bins
                def _nr(q, sv=sv, n=n):
                    k = min(n - 1, max(0, int(np.ceil(q * n)) - 1))
                    return float(sv[k])
                stats_row = (float(seg_vals.mean()), float(seg_vals.std()),
                             float(sv[0]), float(sv[-1]),
                             _nr(0.50), _nr(0.10), _nr(0.90))
            cz, cy, cx = (sums["z"][v] / n, sums["y"][v] / n, sums["x"][v] / n)
            pt = seg_r.TransformContinuousIndexToPhysicalPoint(
                (float(cx), float(cy), float(cz)))
            centroid_ras = [-pt[0], -pt[1], pt[2]]        # LPS -> RAS
            mean, std, vmn, vmx, med, p10, p90 = stats_row
            structures.append({
                "structure": names[v], "label": int(v), "voxels": n,
                "volume_ml": round(n * voxel_ml, 3),
                "mean": round(float(mean), 3), "std": round(float(std), 3),
                "min": round(float(vmn), 3), "max": round(float(vmx), 3),
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
    def _cell(x):
        # a third-party structure name with a tab/newline would shift or split
        # the row; collapse all whitespace-control to a single space
        return " ".join(str(x).split()) if isinstance(x, str) else str(x)

    u = (stats.get("units") or {}).get("intensity", "intensity")
    cols = ["structure", "label", "voxels", "volume_ml",
            f"mean_{u}", f"std_{u}", f"min_{u}", f"max_{u}",
            f"median_{u}", f"p10_{u}", f"p90_{u}",
            "centroid_r_mm", "centroid_a_mm", "centroid_s_mm"]
    lines = ["\t".join(cols)]
    for s in stats.get("structures", []):
        c = list(s.get("centroid_ras_mm") or [])
        c = (c + ["", "", ""])[:3]         # tolerate a short/absent centroid
        row = [s.get("structure", ""), s.get("label", ""), s.get("voxels", ""),
               s.get("volume_ml", ""), s.get("mean", ""), s.get("std", ""),
               s.get("min", ""), s.get("max", ""), s.get("median", ""),
               s.get("p10", ""), s.get("p90", ""), c[0], c[1], c[2]]
        lines.append("\t".join(_cell(x) for x in row))
    return "\n".join(lines) + "\n"
