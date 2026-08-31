"""Build a duckn store from a `ranked_emit.py` / `ranked_emit_fastsurfer.py` output directory.

The emit step records `frame.to_meta()` (nnU-Net path) or the conformed `source_grid`
(FastSurfer) into every part, so the canonical geometry is stated rather than re-derived and
there is nothing here that could disagree with the run.

What still has to be computed, for the nnU-Net path only, is the model grid's TRUE spacing.
`frame.model_spacing` is the nominal request and nnseg documents it as informational; under the
corner rule the grid actually lands at (n_src-1)*s_src/(n_model-1) - 1.504063 where 1.5 was
asked for, in one measured case. Using the nominal value misplaces the far edge by over a
millimeter.

usage: uv run python tools/ranked_build_store.py RANKED_DIR OUT.duckn CASE [all|last]

The last argument selects which emitted parts to keep, and matters only for CASCADE tasks. A
cascade emits one part per stage: a coarse stage that finds the region of interest, then a fine
stage that segments it. `last` keeps only the fine stage - what the task name actually denotes -
and is the sensible default for a store meant to be read as "the X segmentation". `all` keeps
every stage, which is what a cascade replay needs but makes, for example, a `lung_vessels` store
that is mostly a 118-class copy of `total_fast`'s model. Multi-model (non-cascade) tasks are
unaffected: their parts are complementary, not sequential, so `last` would silently discard
four fifths of the task.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import zarr


def _ts_names(task):
    """TotalSegmentator label id -> name, from the installed catalog.

    Read from the task spec rather than a checked-in table: a copied label map is a snapshot
    that goes stale silently when the catalog moves, and a wrong name on a segment is the kind
    of error nothing downstream catches.
    """
    from nnseg.ecosystems import EcosystemCatalog
    from nnseg.tasks import _resolve_spec
    from nnseg.weights import as_store
    store = as_store(None, layout="ts")
    return dict(_resolve_spec(task, EcosystemCatalog(root=store.root)).label_map)


def _fastsurfer_lut() -> Path:
    """Locate FastSurfer's color LUT, importing it if possible and searching if not.

    Engines each get their own environment here (they pin conflicting numpy and torch ranges),
    so the builder normally runs *outside* the one holding FastSurferCNN - an import alone fails
    in the ordinary case, not the exotic one. Fall back to the per-engine venvs beside the repo.
    """
    try:
        import FastSurferCNN
        return Path(FastSurferCNN.__file__).parent / "config" / "FastSurfer_ColorLUT.tsv"
    except ImportError:
        pass
    root = Path(__file__).resolve().parent.parent
    for p in sorted(root.glob(".venvs/*/lib/python*/site-packages/FastSurferCNN/config/"
                              "FastSurfer_ColorLUT.tsv")):
        return p
    raise FileNotFoundError(
        "FastSurfer_ColorLUT.tsv not found - import FastSurferCNN failed and no per-engine "
        f"venv under {root}/.venvs/ contains it")


def _fastsurfer_names():
    """FreeSurfer aparc+aseg id -> name."""
    out = {}
    for line in _fastsurfer_lut().read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 2 and f[0].strip().isdigit():
            out[int(f[0])] = f[1].strip()
    return out


def names_for(engine, task, allow_unnamed=False):
    """Label id -> name. The engines do not share a label namespace: nnunetv2 parts carry the
    ecosystem's ids, FastSurfer carries FreeSurfer aparc+aseg ids.

    RAISES by default when the lookup fails. It degraded silently once - a changed FastSurfer
    LUT path renamed all 78 brain segments to `label_<id>` while the build reported success -
    and an unnamed store is not a smaller store, it is a wrong one that looks finished. Pass
    `allow_unnamed` to build anyway, deliberately.
    """
    try:
        names = _fastsurfer_names() if engine == "fastsurfer" else _ts_names(task)
    except Exception as exc:                       # noqa: BLE001 - any import/catalog problem
        msg = (f"no label names for engine={engine!r} task={task!r} "
               f"({exc.__class__.__name__}: {exc})")
        if not allow_unnamed:
            raise SystemExit(
                f"{msg}\n  The catalog or engine package is not importable from this "
                f"environment.\n  Fix the environment, or pass --allow-unnamed to accept "
                f"label_<id> segment names.") from exc
        print(f"  ! {msg}; segments will be named label_<id>", flush=True)
        return {}
    if not names:
        if not allow_unnamed:
            raise SystemExit(f"label map for {task!r} is empty - refusing to build a store "
                             "whose segments would all be label_<id>")
        print(f"  ! empty label map for {task!r}", flush=True)
    return names


def geometry(part):
    """(true spacing zyx, first-voxel-center origin xyz, direction) of the stored array.

    Two engines record their geometry differently, because their grids arise differently.
    FastSurfer states its conformed grid outright - the logits are native to it, there is no
    crop and the spacing is exactly 1 mm by construction. The nnU-Net path states a canonical
    frame plus a requested spacing, so the grid it actually landed on has to be derived: under
    the corner rule that is (n_src-1)*s_src/(n_model-1), not the nominal request.
    """
    if "source_grid" in part:                                      # fastsurfer
        g = part["source_grid"]
        return list(g["spacing_zyx"]), tuple(g["origin_xyz"]), g["direction_xyz"]
    c = part["frame"]["canonical"]
    model = [int(v) for v in part["model_grid"]]
    start = [int(v) for v in part["envelope"]["start"]]
    eff = [(n_s - 1) * s / (n_m - 1) if n_m > 1 else s
           for n_s, s, n_m in zip(c["shape_zyx"], c["spacing_zyx"], model)]
    D = np.asarray(c["direction_xyz"], float).reshape(3, 3)
    off_xyz = np.asarray([start[2] * eff[2], start[1] * eff[1], start[0] * eff[0]], float)
    origin = np.asarray(c["origin_xyz"], float) + D @ off_xyz      # the crop moves voxel 0
    return eff, tuple(float(v) for v in origin), c["direction_xyz"]


def extent(part):
    """(model grid, envelope start, stop) - FastSurfer has no envelope, so it is the whole grid.

    Half-open internally, because that is what slices a numpy array. The serialized form is
    duckn's inclusive `extent`; see :func:`as_extent`.
    """
    if "source_grid" in part:
        g = [int(v) for v in part["source_grid"]["shape_zyx"]]
        return g, [0, 0, 0], g
    return ([int(v) for v in part["model_grid"]],
            [int(v) for v in part["envelope"]["start"]],
            [int(v) for v in part["envelope"]["stop"]])


def as_extent(start, stop):
    """Half-open (start, stop) -> duckn's `[min_i, max_i, min_j, max_j, min_k, max_k]`.

    duckn has exactly one vocabulary for a voxel range and it is INCLUSIVE on both ends, from
    the `.seg.nrrd` Extent field it converts. Carrying a second, half-open convention in the
    same file is how off-by-one errors get written: a reader would have to remember which key
    means which. Python slices stay half-open in code, where they index arrays; only the stored
    form is converted.
    """
    return [int(v) for a, b in zip(start, stop) for v in (a, b - 1)]


def duckn_grid(rec):
    """A `grid_record` dict -> duckn's own geometry vocabulary.

    The internal record names axis order in its keys (`shape_zyx` beside `origin_xyz`) because
    it packs array-order and world-order quantities together. duckn does not need that: `axes`
    is positional and each entry carries a world `space_direction`, so the order is structural.
    Emitting the duckn form means a reader that can parse an array's geometry can parse a
    referenced grid's with the same code.
    """
    sp, D = rec["spacing_zyx"], np.asarray(rec["direction_xyz"], float).reshape(3, 3)
    cols = [D[:, 2], D[:, 1], D[:, 0]]                             # array axes 0,1,2
    return {"space": "left-posterior-superior",
            "space_origin": [round(float(v), 6) for v in rec["origin_xyz"]],
            "samples": [int(v) for v in rec["shape_zyx"]],
            "axes": [{"kind": "space", "centering": "cell", "unit": "mm",
                      "space_direction": [round(float(v), 9) for v in (c * s)]}
                     for c, s in zip(cols, sp)]}


BRICK = 32


def occupancy(ranks, support, K, smax, brick=BRICK):
    """``(K, Zb, Yb, Xb)`` uint8: the brick-max of each class's support-encoded deficit.

    Answers "can this brick be skipped for class c" without reading the brick. Two decisions
    keep it from being brittle:

    THE BRICK IS DECLARED, NOT INHERITED. Indexing per zarr chunk or per shard would couple this
    to the storage layout, so a rechunk or reshard would silently invalidate it. A declared
    spatial factor cannot: if it happens to align with the chunk grid the skipping is maximally
    efficient, and if it does not the index is still correct, only coarser.

    IT STORES THE BRICK-MAX, NOT A BOOLEAN. A boolean answers one question. The max, in the same
    encoding `support` already uses, answers every threshold question - class c wins somewhere in
    the brick iff the max is ``support_max``, and comes within tau of the winner iff
    ``gap(max) <= tau`` - at the same size and with no new convention.

    Conservative by construction: a max can only over-report presence, never miss it.
    """
    shape = ranks.shape[1:]
    nb = tuple(int(np.ceil(s / brick)) for s in shape)
    idx = np.zeros((K,) + nb, np.uint8)
    bz, by, bx = (np.arange(s) // brick for s in shape)
    flat = (bz[:, None, None] * nb[1] * nb[2]
            + by[None, :, None] * nb[2] + bx[None, None, :]).ravel()
    flatidx = idx.reshape(K, -1)
    for j in range(ranks.shape[0]):
        val = np.full(shape, smax, np.uint8) if j == 0 else support[j - 1]
        ok = (ranks[j] != 0).ravel()
        cls = (ranks[j].astype(np.int64) - 1).ravel()[ok]
        np.maximum.at(flatidx, (cls, flat[ok]), val.ravel()[ok])
    return idx, nb


def brick_geometry(direction, eff, origin, brick, nb):
    """duckn block for the coarse grid: cell-centred bricks, one `list` axis for the class.

    The last brick along an axis is partial when the shape is not a multiple of `brick`, so its
    true centre is nearer than this uniform grid says. That is left as-is deliberately: the
    array is a conservative index, not a measurement, and declaring a uniform grid keeps it a
    readable duckn array rather than a private layout.
    """
    D = np.asarray(direction, float).reshape(3, 3)
    cols = [D[:, 2], D[:, 1], D[:, 0]]
    off = np.asarray([(brick - 1) / 2 * eff[2], (brick - 1) / 2 * eff[1],
                      (brick - 1) / 2 * eff[0]], float)
    o = np.asarray(origin, float) + D @ off
    return {"duckn": {"version": "1.0", "space": "left-posterior-superior",
                      "space_origin": [round(float(v), 6) for v in o],
                      "axes": [{"kind": "list"}] + [
                          {"kind": "space", "centering": "cell", "unit": "mm",
                           "space_direction": [round(float(v), 9) for v in (c * s * brick)]}
                          for c, s in zip(cols, eff)]}}


def segment_extents(labels_zyx, values):
    """duckn `extent` per label: inclusive bbox in the array's storage order, one pass."""
    from scipy import ndimage as ndi
    out = {}
    for v, sl in zip(range(1, int(labels_zyx.max()) + 1), ndi.find_objects(labels_zyx)):
        if sl is None or v not in values:
            continue
        out[v] = [int(x) for s in sl for x in (s.start, s.stop - 1)]
    return out


def axes(direction_xyz, eff_zyx, list_axis):
    D = np.asarray(direction_xyz, float).reshape(3, 3)
    cols = [D[:, 2], D[:, 1], D[:, 0]]                             # array Z, Y, X
    sp = [{"kind": "space", "centering": "cell", "unit": "mm",
           "space_direction": [round(float(v), 9) for v in (c * s)]}
          for c, s in zip(cols, eff_zyx)]
    return ([{"kind": "list"}] + sp) if list_axis else sp


def attrs(direction, eff, origin, *, list_axis):
    return {"duckn": {"version": "1.0", "space": "left-posterior-superior",
                      "space_origin": [round(float(v), 6) for v in origin],
                      "axes": axes(direction, eff, list_axis)}}


CODEC = ("mode", "classes", "depth", "clip", "support_max", "rank_sentinel",
         "exhaustive", "max_tail")

CHUNK4, CHUNK3 = (1, 64, 64, 64), (64, 64, 64)


README = Path(__file__).parent / "ranked_store_README.md"


def write_readme(out):
    """Drop the format reference into the store.

    The arrays are self-describing only to a reader who already knows the conventions - that a
    rank plane is an address rather than a class, that zero means absent in all three arrays,
    that the labelmap is `ranks[0] - 1` and must not be recovered by argmax. A reader arriving
    cold (a person, or a model asked to make sense of the directory) has no way to infer those,
    and every one of them is a silent-wrong-answer trap. The file is generic: it describes the
    format, never this dataset.
    """
    if README.exists():
        shutil.copyfile(README, Path(out) / "README.md")


def layout(shape):
    """``(chunks, shards)``: 64^3 chunks packed into ONE shard per array.

    Loose chunks are the wrong default here. Measured on the five-part `total`: 3012 chunk
    files, median 154 bytes, 89 % under 4 KB - so a 6.97 MB store occupied 17.57 MB against a
    4 KiB allocation unit, and any HTTP client faced a request per chunk. One whole-array shard
    gives 13 data files for the same store at 7.25 MB, and costs nothing that matters:

      * empty chunks stay free - the shard index marks a missing inner chunk, which occupies
        no bytes, so the zero-sentinel elision that makes air cost nothing still applies;
      * partial reads survive - the index is fetched, then only the wanted inner chunks are
        range-read, so `ranks[0]` alone is still one plane's worth of IO, not the whole array;
      * the +3.6 % for the indexes is the whole overhead.

    The leading 1 in the 4-D chunk keeps a chunk from spanning the rank axis, so progressive
    refinement still reads plane by plane inside the shard.
    """
    chunks = CHUNK4 if len(shape) == 4 else CHUNK3
    return chunks, tuple(int(np.ceil(s / c) * c) for s, c in zip(shape, chunks))


def build(src, out, case, parts="all", allow_unnamed=False):
    src, out = Path(src), Path(out)
    meta = json.loads((src / "meta.json").read_text())
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.create_group(store=str(out))
    segs, order = [], []

    engine = next(iter(meta["parts"].values())).get("engine", "nnunetv2")
    NAMES = names_for(engine, meta.get("task"), allow_unnamed)

    items = list(meta["parts"].items())
    if parts == "last" and len(items) > 1:
        # only meaningful for a cascade, whose part names are `<task>:s<i>`; refuse to drop
        # parts of a multi-model task, where every part carries different structures
        if all(":s" in n for n, _ in items):
            dropped = [n for n, _ in items[:-1]]
            print(f"  cascade: keeping {items[-1][0]} only, dropping {dropped}", flush=True)
            items = items[-1:]
        else:
            print(f"  parts='last' ignored: {len(items)} complementary parts, not a cascade",
                  flush=True)

    for i, (name, part) in enumerate(items):
        eff, origin, direction = geometry(part)
        grid, start, stop = extent(part)
        g = root.create_group(f"parts/{i}")
        lut = [int(v) for v in part["labels"]]
        # Axis order is not spelled in these keys. duckn states it structurally - `axes` is
        # positional, one entry per array axis - so a per-axis list here is in array order by
        # the same rule, and a `_zyx` suffix would be our vocabulary, not the format's.
        block = {"version": "0.1",
                 **{k: part[k] for k in CODEC if k in part},
                 "model_grid": grid,
                 "envelope": as_extent(start, stop),               # inclusive, like duckn
                 "labels": lut, "part": name,
                 "task": part.get("task", meta.get("task"))}
        if "convention" in part:                                   # nnunetv2 only
            block["resample_alignment"] = part["convention"]
            block["nominal_spacing"] = [float(v) for v in part["spacing_zyx"]]
        if "labels_note" in part:
            block["labels_note"] = part["labels_note"]
        if "target_grid" in part:                                  # fastsurfer: the input grid
            block["target_grid"] = duckn_grid(part["target_grid"])
        g.attrs.update({"duckn": {"version": "1.0", "extensions": {"ranked": block}}})

        for arr_name in ("ranks", "support", "tail"):
            f = src / f"{name}_{arr_name}.npy"
            if not f.exists():
                continue
            a = np.load(f, mmap_mode="r")
            four = a.ndim == 4
            chunks, shards = layout(a.shape)
            z = g.create_array(arr_name, shape=a.shape, dtype=a.dtype,
                               chunks=chunks, shards=shards,
                               compressors=zarr.codecs.ZstdCodec(level=9),
                               attributes=attrs(direction, eff, origin, list_axis=four))
            z[:] = a
            del a

        # occupancy index: which bricks a class can possibly be in, so a reader after one
        # structure skips the rest without opening it
        rk_all = np.asarray(np.load(src / f"{name}_ranks.npy", mmap_mode="r"))
        su_all = np.asarray(np.load(src / f"{name}_support.npy", mmap_mode="r"))
        occ, nb = occupancy(rk_all, su_all, len(lut), part["support_max"])
        oz = g.create_array("occupancy", shape=occ.shape, dtype=occ.dtype,
                            chunks=occ.shape, shards=None,
                            compressors=zarr.codecs.ZstdCodec(level=9),
                            attributes=brick_geometry(direction, eff, origin, BRICK, nb))
        oz[:] = occ
        block["brick"] = [BRICK, BRICK, BRICK]
        g.attrs.update({"duckn": {"version": "1.0", "extensions": {"ranked": block}}})
        del su_all, occ

        # duckn's own per-segment bounding box. Worth writing rather than leaving None: with it,
        # "is this structure truncated by the field of view" is answerable by any reader - the
        # question that made a gallbladder look like it vanished between resolutions.
        wins = rk_all[0].astype(np.int64)
        del rk_all
        boxes = segment_extents(np.asarray(lut)[wins - 1], {int(x) for x in lut} - {0})
        for v in sorted({int(x) for x in lut} - {0}):
            if not any(s["label_value"] == v for s in segs):
                seg = {"id": f"c{v}", "name": NAMES.get(v, f"label_{v}"),
                       "label_value": v, "layer": i}
                if v in boxes:
                    seg["extent"] = boxes[v]
                segs.append(seg)
        del wins
        order.append({"index": i, "name": name})
        print(f"  parts/{i} {name:<12} grid {tuple(grid)} crop {tuple(start)} "
              f"eff {[round(v, 6) for v in eff]}", flush=True)

    # A group is a duckn Segment whose label_value is a list of segment ids - duckn's own way
    # of saying "union", so no invention is needed. The useful unions are per label namespace.
    def pick(pred):
        return [s["id"] for s in segs if pred(s["name"])]

    if engine == "fastsurfer":
        # NO whole-hemisphere group. FastSurfer's network emits 31 lh-numbered cortical
        # channels and only 14 rh-numbered ones; the missing right-hemisphere regions ride
        # inside lh-numbered channels and are separated by `split_cortex_labels`, which is
        # SPATIAL. So laterality is not a property of the stored labels for cortex, and a
        # `g_rh` union over them would quietly drop 17 regions. The engine says as much in
        # `labels_note`. The aseg structures below ARE lateralized in channel space (14 each),
        # so those group honestly - named for what they actually contain.
        spec = [("g_subcortical_left", "left subcortical structures",
                 lambda n: n.startswith("Left-")),
                ("g_subcortical_right", "right subcortical structures",
                 lambda n: n.startswith("Right-")),
                ("g_cortex", "cerebral cortex", lambda n: n.startswith("ctx-")),
                ("g_cerebellum", "cerebellum", lambda n: "Cerebellum" in n),
                ("g_ventricles", "ventricular system",
                 lambda n: "Ventricle" in n or n.endswith("-Vent"))]
    else:
        spec = [("g_lungs", "lungs", lambda n: n.startswith("lung_")),
                ("g_spine", "vertebral column", lambda n: n.startswith("vertebrae_"))]
    groups = [{"id": i, "name": nm, "label_value": v}
              for i, nm, p in spec if (v := pick(p))]

    root.attrs.update({"duckn": {"version": "1.0", "extensions": {
        "seg": {"version": "0.6", "terminologies": {"SCT": {
            "name": "SNOMED CT", "url": "http://snomed.info/sct",
            "url_template": "http://snomed.info/id/{code}"}},
            "segments": segs + groups},
        "nnseg": {"nnseg_version": dict(items)[order[0]["name"]].get("nnseg"),
                  "engine": engine, "task": meta["task"], "case": case,
                  "source_file": Path(meta["image"]).name, "part_order": order},
    }}})
    write_readme(out)
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{out.name}: {mb:.2f} MB\n")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ranked_dir")
    ap.add_argument("out")
    ap.add_argument("case")
    ap.add_argument("parts", nargs="?", default="all", choices=["all", "last"])
    ap.add_argument("--allow-unnamed", action="store_true",
                    help="build even if segment names cannot be resolved")
    a = ap.parse_args()
    build(a.ranked_dir, a.out, a.case, a.parts, a.allow_unnamed)
