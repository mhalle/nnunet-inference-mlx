"""Round 4 differential + property tests for the statistics numerics (the
module had no direct test before)."""
import json

import numpy as np
import pytest

SimpleITK = pytest.importorskip("SimpleITK")
import SimpleITK as sitk

from nnseg.statistics import compute_statistics, statistics_tsv


def _pair(gray, lab, names, tmp_path, spacing=(1.0, 1.0, 1.0)):
    gi = sitk.GetImageFromArray(gray)
    gi.SetSpacing(spacing)
    li = sitk.GetImageFromArray(lab.astype(np.uint16))
    li.SetSpacing(spacing)
    tmp_path = __import__("pathlib").Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    lp = tmp_path / "labels.seg.nrrd"
    for v, nm in names.items():
        li.SetMetaData(f"Segment{v}_Name", nm)
        li.SetMetaData(f"Segment{v}_LabelValue", str(v))
    sitk.WriteImage(li, str(lp))
    return gi, lp


def test_histogram_and_sort_paths_agree(tmp_path):
    """The int (histogram) and float (sort) paths must report identical
    percentiles - not 'within 1 unit'. numpy's linear interpolation invented
    values that disagreed by thousands of HU on multimodal structures."""
    rng = np.random.default_rng(20260825)
    worst = 0.0
    for case in range(120):
        shape = tuple(int(x) for x in rng.integers(12, 30, size=3))
        lab = np.zeros(shape, dtype=np.int64)
        K = int(rng.integers(2, 6))
        for v in range(1, K):
            mask = rng.random(shape) < 0.15
            lab[mask] = v
        # a distribution with heavy ties + negatives + bimodal structures
        gi = rng.integers(-1000, 1301, size=shape)
        if case % 3 == 0:                  # force a bimodal label
            gi = np.where(rng.random(shape) < 0.5, -1000, 1300)
        names = {v: f"s{v}" for v in range(1, K)}
        gint, lp = _pair(gi.astype(np.int16), lab, names, tmp_path / f"i{case}")
        gflt, _ = _pair(gi.astype(np.float32), lab, names, tmp_path / f"f{case}")
        ri = compute_statistics(gint, str(lp), tmp_path / f"ri{case}.json")
        rf = compute_statistics(gflt,
                                str((tmp_path / f"f{case}") / "labels.seg.nrrd"),
                                tmp_path / f"rf{case}.json")
        if ri is None and rf is None:
            continue
        a = {s["label"]: s for s in json.loads(ri.read_text())["structures"]}
        b = {s["label"]: s for s in json.loads(rf.read_text())["structures"]}
        assert a.keys() == b.keys()
        for v in a:
            for key in ("median", "p10", "p90", "min", "max"):
                d = abs(a[v][key] - b[v][key])
                worst = max(worst, d)
                assert d < 1e-6, (case, v, key, a[v][key], b[v][key])
    assert worst < 1e-6


def test_all_background_symmetric(tmp_path):
    """Empty foreground yields the same artifact (valid, empty structures)
    regardless of gray dtype - not None for int and JSON for float."""
    lab = np.zeros((8, 8, 8), dtype=np.int64)
    for dt in (np.int16, np.float32):
        gi, lp = _pair((np.zeros((8, 8, 8)) + 5).astype(dt), lab, {},
                       tmp_path / str(dt))
        out = compute_statistics(gi, str(lp), tmp_path / f"o{dt}.json")
        assert out is not None
        assert json.loads(out.read_text())["structures"] == []


def test_negative_labels_do_not_crash(tmp_path):
    """A signed labelmap with negative values must not crash bincount; the
    negatives are simply not structures."""
    lab = np.zeros((8, 8, 8), dtype=np.int64)
    lab[0, 0, 0] = -3
    lab[1, 1, 1] = 2
    gi, lp = _pair((np.zeros((8, 8, 8), dtype=np.int16) + 40), lab, {2: "kidney"},
                   tmp_path)
    out = compute_statistics(gi, str(lp), tmp_path / "o.json")
    labels = [s["label"] for s in json.loads(out.read_text())["structures"]]
    assert labels == [2]


def test_statistics_tsv_sanitizes_names():
    """A structure name with a tab/newline must not shift columns or split
    the row (moose/native names are third-party dataset.json data)."""
    stats = {"units": {"intensity": "hu"}, "structures": [
        {"structure": "liver\twith\ttabs", "label": 1, "voxels": 3,
         "volume_ml": 1.0, "mean": 1, "std": 0, "min": 1, "max": 1,
         "median": 1, "p10": 1, "p90": 1, "centroid_ras_mm": [1, 2, 3]},
        {"structure": "spleen\nnewline", "label": 2, "voxels": 3,
         "volume_ml": 1.0, "mean": 1, "std": 0, "min": 1, "max": 1,
         "median": 1, "p10": 1, "p90": 1, "centroid_ras_mm": [4]},  # short centroid
    ]}
    tsv = statistics_tsv(stats)
    lines = tsv.split("\n")
    assert lines[-1] == "" and not lines[-2].endswith("\r")
    lines = [ln for ln in lines if ln]     # drop the trailing newline only
    assert len(lines) == 3                 # header + 2 rows, no split
    ncol = len(lines[0].split("\t"))
    assert all(len(ln.split("\t")) == ncol for ln in lines), [ln.count("\t") for ln in lines]
