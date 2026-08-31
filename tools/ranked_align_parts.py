"""Pad every part of a store out to the shared model grid, so all parts have ONE origin.

A multi-model task crops each part to its own body envelope - and the envelope is recomputed
per model, because the body threshold depends on that model's normalization. So `total`'s five
parts land on four different offsets into one grid, and every reader has to carry the crop
arithmetic to line them up. Padding removes that: each array becomes the model grid itself,
`envelope.start` is [0,0,0] for all parts, and the origin is the frame's own.

The fill is not arbitrary, and it is not the zero sentinel either:

    ranks[0]  = 1     class 0 + 1, i.e. BACKGROUND - because ranks[0] is the argmax and every
                      voxel has a winner. Filling it with the sentinel would break the one
                      invariant the labelmap path relies on (`ranks[0] - 1` IS the labelmap).
    ranks[1:] = 0     sentinel: no other class is within the clip here.
    support   = 0     support counts up FROM the clip, so 0 means every runner-up is >= clip
                      behind - maximally confident.
    tail      = 0     no mass beyond the top N.

That is the same claim `segment(outside="background")` already makes when it restores a cropped
part to the output grid, so the padded store decodes to what the pipeline produces. It is still
a claim the network did not make, so the store records it rather than hiding it.

usage: uv run python tools/ranked_align_parts.py SRC.duckn DST.duckn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))   # sibling tools

import numpy as np
import zarr


def ranked_of(group):
    return group.attrs.asdict()["duckn"]["extensions"]["ranked"]


def start_stop(r):
    """Half-open (start, stop) from whichever serialized form is present.

    `envelope` is duckn's inclusive `[min_i, max_i, ...]`; older stores carried a half-open
    {start, stop} dict, and older still a flat `envelope_start_zyx`. Converted here rather than
    at every use, so the rest of this file thinks in slices.
    """
    if isinstance(r.get("envelope"), list):                 # duckn extent, inclusive
        e = r["envelope"]
        return [e[0], e[2], e[4]], [e[1] + 1, e[3] + 1, e[5] + 1]
    if isinstance(r.get("envelope"), dict):                 # earlier half-open form
        return list(r["envelope"]["start"]), list(r["envelope"]["stop"])
    start = list(r["envelope_start_zyx"])
    grid = list(r.get("model_grid") or r.get("model_grid_zyx") or r["model_grid_full_zyx"])
    return start, grid


FILL = {"ranks": None, "support": 0, "tail": 0}      # ranks handled per plane


def _eff(sp):
    return [float(np.linalg.norm(a["space_direction"]))
            for a in sp["ranks"].attrs.asdict()["duckn"]["axes"] if a["kind"] == "space"]


def _direction(sp):
    """Unit direction cosines, recovered from the axes' world vectors."""
    ax = [a for a in sp["ranks"].attrs.asdict()["duckn"]["axes"] if a["kind"] == "space"]
    eff = _eff(sp)
    D = np.zeros((3, 3))
    for k, a in enumerate(ax):
        D[:, 2 - k] = np.asarray(a["space_direction"], float) / eff[k]
    return D.reshape(-1).tolist()


def frame_origin(s, order):
    """The one origin every padded part must land on, computed once.

    Deriving it per part by subtracting that part's offset back off gives answers that differ
    in the last decimal, because the rounding happens after the subtraction. A nanometer is
    physically nothing, but a reader grouping arrays by identical origin would see five
    distinct grids, so the value is computed once here and shared.
    """
    out = []
    for i, _ in enumerate(order):
        sp = s[f"parts/{i}"]
        start, _ = start_stop(ranked_of(sp))
        at = sp["ranks"].attrs.asdict()["duckn"]
        ax = [a for a in at["axes"] if a["kind"] == "space"]
        eff = [float(np.linalg.norm(a["space_direction"])) for a in ax]
        D = np.zeros((3, 3))
        for k, a in enumerate(ax):
            D[:, 2 - k] = np.asarray(a["space_direction"], float) / eff[k]
        off = np.asarray([start[2] * eff[2], start[1] * eff[1], start[0] * eff[0]], float)
        out.append(np.asarray(at["space_origin"], float) - D @ off)
    out = np.asarray(out)
    spread = float(np.abs(out - out[0]).max())
    if spread > 1e-3:
        raise SystemExit(f"parts do not share a frame: origins spread by {spread:.6f} mm")
    print(f"  shared origin {[round(v, 6) for v in out.mean(0)]}  "
          f"(parts agree to {spread:.2e} mm)")
    return [round(float(v), 6) for v in out.mean(0)]


def align(src, dst):
    src, dst = Path(src), Path(dst)
    s = zarr.open_group(str(src), mode="r")
    import shutil
    shutil.rmtree(dst, ignore_errors=True)
    d = zarr.create_group(store=str(dst))
    d.attrs.update(s.attrs.asdict())                  # root metadata carries over unchanged

    order = s.attrs.asdict()["duckn"]["extensions"]["nnseg"]["part_order"]
    shared = frame_origin(s, order)

    # Segment extents are in the CROPPED array's coordinates. Padding moves every voxel by that
    # part's start offset, so an extent copied verbatim points at the wrong place - silently,
    # since it stays a plausible box inside the grid. Shift them with the data.
    starts = {i: start_stop(ranked_of(s[f"parts/{i}"]))[0] for i, _ in enumerate(order)}
    a = dict(d.attrs.asdict()["duckn"])
    ext_ = dict(a["extensions"])
    seg_ = dict(ext_["seg"])
    moved = 0
    out_segs = []
    for seg in seg_["segments"]:
        seg = dict(seg)
        if "extent" in seg and not isinstance(seg["label_value"], list):
            off = starts.get(seg.get("layer", 0), [0, 0, 0])
            if any(off):
                e = seg["extent"]
                seg["extent"] = [int(v) + off[k // 2] for k, v in enumerate(e)]
                moved += 1
        out_segs.append(seg)
    seg_["segments"] = out_segs
    ext_["seg"] = seg_
    a["extensions"] = ext_
    d.attrs["duckn"] = a
    if moved:
        print(f"  shifted {moved} segment extents into the padded grid", flush=True)
    for i, _ in enumerate(order):
        sp = s[f"parts/{i}"]
        r = ranked_of(sp)
        start, _stop = start_stop(r)
        grid = list(r.get("model_grid") or r.get("model_grid_zyx")
                    or r["model_grid_full_zyx"])
        dp = d.create_group(f"parts/{i}")

        origin = None
        for name in ("ranks", "support", "tail"):
            if name not in sp:
                continue
            a = np.asarray(sp[name])
            lead = a.shape[:-3]
            full = np.zeros(lead + tuple(grid), dtype=a.dtype)
            if name == "ranks":
                full[0] = 1                            # background everywhere, then paste
            sl = (Ellipsis,) + tuple(slice(o, o + n) for o, n in zip(start, a.shape[-3:]))
            full[sl] = a

            at = dict(sp[name].attrs.asdict())
            duckn = dict(at["duckn"])
            duckn["space_origin"] = shared     # the array now starts at the frame's own origin
            at["duckn"] = duckn
            origin = shared

            from ranked_build_store import layout            # one shard per array - see there
            chunks, shards = layout(full.shape)
            z = dp.create_array(name, shape=full.shape, dtype=full.dtype,
                                chunks=chunks, shards=shards,
                                compressors=zarr.codecs.ZstdCodec(level=9), attributes=at)
            z[:] = full
            del a, full

        meta = dict(sp.attrs.asdict()["duckn"]["extensions"]["ranked"])
        # the occupancy index must be REBUILT, not copied: padding changes which classes can
        # be found in the edge bricks (background now wins there), so a copied index would
        # under-report background and over-report nothing - conservative in the wrong direction
        from ranked_build_store import BRICK, brick_geometry, occupancy
        rk_p, su_p = np.asarray(dp["ranks"]), np.asarray(dp["support"])
        occ, nb = occupancy(rk_p, su_p, len(r["labels"]), r["support_max"])
        oz = dp.create_array("occupancy", shape=occ.shape, dtype=occ.dtype,
                             chunks=occ.shape, shards=None,
                             compressors=zarr.codecs.ZstdCodec(level=9),
                             attributes=brick_geometry(_direction(sp), _eff(sp),
                                                       shared, BRICK, nb))
        oz[:] = occ
        del rk_p, su_p, occ

        meta.pop("envelope_start_zyx", None)
        meta.pop("model_grid_zyx", None)
        meta.pop("model_grid_full_zyx", None)
        meta["envelope"] = [v for n in grid for v in (0, int(n) - 1)]   # inclusive, duckn form
        meta["model_grid"] = [int(v) for v in grid]
        # the array is no longer purely network output - say so
        meta["padded_from"] = {"start": [int(v) for v in start],
                               "fill": "background (ranks[0]=1, ranks[1:]=0, support=0, tail=0)"}
        dp.attrs.update({"duckn": {"version": "1.0", "extensions": {"ranked": meta}}})
        print(f"  parts/{i} {meta['part']:<12} {tuple(start)} -> (0,0,0)   grid {tuple(grid)}"
              f"   origin {origin}", flush=True)

    from ranked_build_store import write_readme
    write_readme(dst)                        # the padded copy needs the reference too
    mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"{dst.name}: {mb:.2f} MB")


if __name__ == "__main__":
    align(*sys.argv[1:3])
