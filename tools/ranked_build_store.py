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
import sys
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


def names_for(engine, task):
    """Label id -> name. The engines do not share a label namespace: nnunetv2 parts carry the
    ecosystem's ids, FastSurfer carries FreeSurfer aparc+aseg ids.

    Both lookups need the corresponding package/catalog installed, and each lives in its own
    environment - so a miss is expected, not exceptional, and degrades to `label_<id>` rather
    than failing the build.
    """
    try:
        return _fastsurfer_names() if engine == "fastsurfer" else _ts_names(task)
    except Exception as exc:                       # noqa: BLE001 - any import/catalog problem
        print(f"  ! no label names for engine={engine!r} task={task!r} ({exc.__class__.__name__}"
              f": {exc}); segments will be named label_<id>", flush=True)
        return {}


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
    """(model grid, envelope start, stop) - FastSurfer has no envelope, so it is the whole grid."""
    if "source_grid" in part:
        g = [int(v) for v in part["source_grid"]["shape_zyx"]]
        return g, [0, 0, 0], g
    return ([int(v) for v in part["model_grid"]],
            [int(v) for v in part["envelope"]["start"]],
            [int(v) for v in part["envelope"]["stop"]])


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


def build(src, out, case, parts="all"):
    src, out = Path(src), Path(out)
    meta = json.loads((src / "meta.json").read_text())
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.create_group(store=str(out))
    segs, order = [], []

    engine = next(iter(meta["parts"].values())).get("engine", "nnunetv2")
    NAMES = names_for(engine, meta.get("task"))

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
        block = {"version": "0.1",
                 **{k: part[k] for k in CODEC if k in part},
                 "model_grid_zyx": grid,
                 "envelope": {"start": start, "stop": stop},       # a range, both ends
                 "labels": lut, "part": name,
                 "task": part.get("task", meta.get("task"))}
        if "convention" in part:                                   # nnunetv2 only
            block["resample_alignment"] = part["convention"]
            block["nominal_spacing_zyx"] = [float(v) for v in part["spacing_zyx"]]
        for k in ("labels_note", "target_grid"):                   # fastsurfer carries these
            if k in part:
                block[k] = part[k]
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

        for v in sorted({int(x) for x in lut} - {0}):
            if not any(s["label_value"] == v for s in segs):
                segs.append({"id": f"c{v}", "name": NAMES.get(v, f"label_{v}"),
                             "label_value": v, "layer": i})
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
    build(*sys.argv[1:5])
