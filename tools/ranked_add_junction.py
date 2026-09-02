"""Add the triple-line `junction` layer to an existing ranked store, in place.

The builder writes the layer for new stores. This is for stores that already exist - the demo
packages - where re-emitting to rebuild is not on: it reads `ranks` and `support` back out of
the store SLAB BY SLAB, computes the layer with the same functions the builder uses, writes
the two arrays slab-wise with the store's own geometry, records the decode parameters in the
part block, and refreshes the README so the store describes what it now contains.

Memory is a slab plus the tube, whatever the store's size: a store whose planes are half a
gigabyte is processed in a few tens of megabytes.

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
from ranked_build_store import (JUNCTION_SPAN, JUNCTION_ZERO, junction_sparse,  # noqa: E402
                                write_junction, write_readme)


def _spacing(arr):
    ax = arr.attrs.asdict()["duckn"]["axes"]
    return [float(np.linalg.norm(a["space_direction"])) for a in ax if a.get("kind") == "space"]


def _attrs_like(arr, list_axis):
    d = copy.deepcopy(arr.attrs.asdict())
    axes = d["duckn"]["axes"]
    spatial = [a for a in axes if a.get("kind") != "list"]
    d["duckn"]["axes"] = ([{"kind": "list"}] + spatial) if list_axis else spatial
    return d


def _record(root, trunc, n, shape):
    """Append a processing step to the store's duckn provenance, where it has one.

    An in-place addition is a processing step like any other, and a store whose provenance
    says two steps ran when three did is wrong in the one place a reader looks to find out
    what happened to it. One step per store, updated in place if this tool runs again.
    """
    attrs = root.attrs.asdict()
    prov = attrs.get("duckn", {}).get("extensions", {}).get("provenance")
    if prov is None:
        return
    steps = [s for s in prov.get("processing", []) if s.get("name") != "Triple-line junction layer"]
    steps.append({
        "name": "Triple-line junction layer",
        "description": "signed distance to the interface between the two leading structures, "
                       "from their logits, written in tubes around the lines where such an "
                       "interface meets a third label; added to an existing store in place",
        "software": {"name": "ranked_add_junction.py",
                     "url": "https://github.com/mhalle/nnunet-inference-mlx"},
        "parameters": {"junction_truncation": round(float(trunc), 6),
                       "junction_zero": JUNCTION_ZERO, "junction_span": JUNCTION_SPAN,
                       "voxels_written": int(n), "grid": [int(v) for v in shape]}})
    prov["processing"] = steps
    root.attrs.update(attrs)


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
        ranks, support = g["ranks"], g["support"]
        shape = tuple(ranks.shape[1:])
        spacing = _spacing(ranks)
        trunc = float(block.get("distance_truncation") or 2.0 * min(spacing))
        idx, q, a, b = junction_sparse(
            lambda z0, z1: ranks[0, z0:z1],
            lambda z0, z1: (ranks[:, z0:z1], support[:, z0:z1]),
            shape, block["clip"], spacing, trunc)
        n = write_junction(g, idx, q, a, b, shape, ranks.dtype,
                           _attrs_like(ranks, list_axis=False), _attrs_like(ranks, list_axis=True),
                           trunc, block)
        g.attrs.update(attrs)
        _record(root, trunc, n, shape)
        frac = 100.0 * n / float(np.prod(shape))
        print(f"  parts/{name}: junction {frac:.2f} % of {np.prod(shape) / 1e6:.1f} M voxels, "
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
