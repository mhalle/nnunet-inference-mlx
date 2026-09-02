"""Add the triple-line `junction` layer to an existing ranked store, in place.

The builder writes the layer for new stores. This is for stores that already exist - the demo
packages - where re-emitting to rebuild is not on: it reads `ranks` and `support` back out of
the store, computes the layer with the same function the builder uses, writes the two arrays
with the store's own layout and geometry, records the decode parameters in the part block, and
refreshes the README so the store describes what it now contains.

usage: uv run python tools/ranked_add_junction.py STORE.duckn [STORE.duckn ...] [--force]

`--force` recomputes a layer that is already there. The truncation is the distance field's, so
the two layers agree on their reach; a store with no distance field gets two voxels.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranked_build_store import JUNCTION_MAX, junction_field, layout, write_readme  # noqa: E402


def _spacing(arr):
    ax = arr.attrs.asdict()["duckn"]["axes"]
    return [float(np.linalg.norm(a["space_direction"])) for a in ax if a.get("kind") == "space"]


def _attrs_like(arr, list_axis):
    d = copy.deepcopy(arr.attrs.asdict())
    axes = d["duckn"]["axes"]
    spatial = [a for a in axes if a.get("kind") != "list"]
    d["duckn"]["axes"] = ([{"kind": "list"}] + spatial) if list_axis else spatial
    return d


def add(store: Path, force=False):
    root = zarr.open_group(str(store), mode="r+")
    parts = root["parts"]
    names = sorted((k for k in parts.keys()), key=lambda k: int(k))
    for name in names:
        g = parts[name]
        attrs = g.attrs.asdict()
        block = attrs["duckn"]["extensions"]["ranked"]
        if "junction" in g and not force:
            print(f"  parts/{name}: junction present, skipping (--force to redo)", flush=True)
            continue
        t0 = time.perf_counter()
        ranks = np.asarray(g["ranks"][:])
        support = np.asarray(g["support"][:])
        spacing = _spacing(g["ranks"])
        trunc = float(block.get("distance_truncation") or 2.0 * min(spacing))
        jn, jp = junction_field(ranks, support, block["clip"], spacing, trunc)
        del ranks, support
        for nm in ("junction", "junction_pair"):
            if nm in g:
                del g[nm]
        chunks, shards = layout(jn.shape)
        jz = g.create_array("junction", shape=jn.shape, dtype=jn.dtype, chunks=chunks,
                            shards=shards, compressors=zarr.codecs.ZstdCodec(level=9),
                            attributes=_attrs_like(g["ranks"], list_axis=False))
        jz[:] = jn
        chunks, shards = layout(jp.shape)
        pz = g.create_array("junction_pair", shape=jp.shape, dtype=jp.dtype, chunks=chunks,
                            shards=shards, compressors=zarr.codecs.ZstdCodec(level=9),
                            attributes=_attrs_like(g["ranks"], list_axis=True))
        pz[:] = jp
        block["junction_truncation"] = round(trunc, 6)
        block["junction_max"] = JUNCTION_MAX
        g.attrs.update(attrs)
        frac = 100.0 * np.count_nonzero(jn) / jn.size
        print(f"  parts/{name}: junction {frac:.2f} % of {jn.size / 1e6:.1f} M voxels, "
              f"T = {trunc:.3f} mm, {time.perf_counter() - t0:.1f} s", flush=True)
    write_readme(store)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    if not args:
        sys.exit(__doc__)
    for a in args:
        print(a, flush=True)
        add(Path(a), force=force)
