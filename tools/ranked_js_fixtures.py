"""Golden fixtures for a JavaScript ranked-store decoder.

A JS port of `nnseg.ranked.decode_groups` cannot be trusted by inspection - the
project rule is that geometry and decode changes get a bit-exactness check
against a known-good reference before they are believed. This tool produces that
reference, from the decoder that is already validated:

  OUT/manifest.json           what was decoded, expected sha256 per group,
                              spot samples, and geometry checkpoints computed
                              independently of any JS code
  OUT/margins.zarr            (G, Z, Y, X) uint8 quantized group margins from
                              the REAL demo store, written as a duckn array
                              whose lut value transform maps bytes back to
                              logits (0 = absent -> -clip, 128 = the surface)
  OUT/phantom.duckn           a tiny synthetic ranked store (K=5, depth 3,
                              unsharded, with genuinely missing chunks) for the
                              edge semantics a big healthy store never shows:
                              the zero sentinel, fill-value reads, a class that
                              is absent everywhere, a tail plane
  OUT/phantom_margins.zarr    golden decode of the phantom's groups

The quantized convention is `_quantize_margin`'s: byte 128 sits exactly on the
boundary, bytes span 1..255, and 0 is reserved so an unwritten chunk or cleared
texture reads "absent", which the lut states explicitly as -clip.

usage: uv run --with zarr python tools/ranked_js_fixtures.py STORE OUT
   eg: uv run --with zarr python tools/ranked_js_fixtures.py \
         data/duckn_demo/idc-torso1/total_fast.duckn ../../sdfview/test/fixtures
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

from nnseg.ranked import CLIP, RankedCode, decode_groups, encode

ZSTD = None  # zarr.codecs is import-time heavy; resolved in main()


# ---------------------------------------------------------------------------
# group resolution: segment ids -> label values -> channel indices
# ---------------------------------------------------------------------------

def _channels(segments, labels, want):
    """Resolve segment names to channel-index lists via the label LUT.

    A group segment's label_value is a list of segment IDS (strings like
    "c10"), each naming a segment whose own label_value is the shared label
    id; `labels[channel]` maps channel -> label id. Names are never matched
    by prefix - membership comes from the store's own tables.
    """
    by_id = {s["id"]: s for s in segments}
    by_name = {s["name"]: s for s in segments}
    chan_of_label = {int(lab): c for c, lab in enumerate(labels)}

    def one(seg):
        v = seg["label_value"]
        if isinstance(v, list):
            return [chan_of_label[int(by_id[ref]["label_value"])] for ref in v]
        return [chan_of_label[int(v)]]

    out = {}
    for name in want:
        if name not in by_name:
            raise SystemExit(f"segment {name!r} not in this store; have e.g. "
                             f"{sorted(by_name)[:8]}")
        out[name] = one(by_name[name])
    return out


# ---------------------------------------------------------------------------
# duckn attribute blocks for the fixture arrays
# ---------------------------------------------------------------------------

def _lut_transform(clip):
    """Byte -> logits, including the sentinel: 0 -> -clip, else (q-128)/127*clip."""
    values = [-clip] + [(q - 128) / 127 * clip for q in range(1, 256)]
    return [{"name": "lut", "parameters": {"first_value": 0, "values": values}}]


def _margin_attrs(spatial_axes, origin, space, clip, groups):
    return {"duckn": {
        "version": "1.1",
        "space": space,
        "space_origin": list(origin),
        "value_transforms": _lut_transform(clip),
        "axes": [{"kind": "list"}] + [dict(ax) for ax in spatial_axes],
        "extensions": {"nnseg_fixture": {
            "clip": clip,
            "byte_for_boundary": 128,
            "byte_for_absent": 0,
            "groups": [{"name": n, "channels": c} for n, c in groups.items()],
        }},
    }}


def _geometry_checkpoints(origin, directions, shape_zyx):
    """Index -> world and world -> texture-coordinate pairs, computed here.

    space_origin is the CENTER of voxel [0,0,0], so the value stored at index
    i sits at texture coordinate (i + 0.5) / n - the half-texel rule the JS
    geometry module must reproduce. Indices are array order (z, y, x); world
    is the store's space; texture coordinates are GL order (x fastest).
    """
    d = np.asarray(directions, dtype=np.float64)          # (3 axes, 3 world)
    o = np.asarray(origin, dtype=np.float64)
    nz, ny, nx = shape_zyx
    picks = [(0, 0, 0), (5, 7, 11), (nz - 1, ny - 1, nx - 1)]
    world_from_index, texture_from_world = [], []
    for iz, iy, ix in picks:
        w = o + iz * d[0] + iy * d[1] + ix * d[2]
        world_from_index.append({"index_zyx": [iz, iy, ix], "world": list(w)})
        tc = [(ix + 0.5) / nx, (iy + 0.5) / ny, (iz + 0.5) / nz]
        texture_from_world.append({"world": list(w), "texture_coordinate": tc})
    return {"world_from_index": world_from_index,
            "texture_from_world": texture_from_world}


# ---------------------------------------------------------------------------
# decode + describe one store
# ---------------------------------------------------------------------------

def _decode(store_dir, groups):
    g = zarr.open_group(str(store_dir), mode="r")["parts/0"]
    meta = dict(g.attrs["duckn"]["extensions"]["ranked"])
    ranks = np.asarray(g["ranks"])
    support = np.asarray(g["support"])
    tail = np.asarray(g["tail"]) if "tail" in g else None
    meta["shape"] = list(ranks.shape[1:])
    code = RankedCode(ranks=ranks, support=support, tail=tail, meta=meta)
    dec = decode_groups(code, list(groups.values()), quantize=True)
    arrays = g["ranks"].attrs["duckn"]
    spatial = [ax for ax in arrays["axes"] if ax.get("space_direction")]
    return dec.numpy(), meta, arrays, spatial


def _describe(dec, groups, clip):
    """Hashes and spot samples a JS test can assert without loading Python."""
    out = []
    for gi, (name, channels) in enumerate(groups.items()):
        vol = dec[gi]
        peak = np.unravel_index(int(np.argmax(vol)), vol.shape)
        picks = [(0, 0, 0), tuple(int(v) for v in peak),
                 tuple(int(v) // 2 for v in vol.shape),
                 (vol.shape[0] - 1, vol.shape[1] - 1, vol.shape[2] - 1)]
        out.append({
            "name": name,
            "channels": channels,
            "sha256_uint8": hashlib.sha256(vol.tobytes()).hexdigest(),
            "voxels_inside": int((vol > 128).sum()),
            "spot_samples": [{"index_zyx": list(p), "byte": int(vol[p])}
                             for p in picks],
        })
    return out


# ---------------------------------------------------------------------------
# the phantom: small, synthetic, and deliberately awkward
# ---------------------------------------------------------------------------

def _phantom_logits():
    """K=5 logits on a (20, 24, 32) grid: two overlapping spheres, one small
    ellipsoid, one class that is absent everywhere, over a zero background.
    A deterministic sub-quantum wiggle avoids exact ties without randomness."""
    shape = (20, 24, 32)
    directions = [[0.0, 0.0, 2.0], [0.0, 1.5, 0.0], [1.25, 0.0, 0.0]]
    origin = [5.0, -10.0, 3.0]
    iz, iy, ix = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    d = np.asarray(directions); o = np.asarray(origin)
    w = (o[None, None, None, :] + iz[..., None] * d[0] + iy[..., None] * d[1]
         + ix[..., None] * d[2])

    def ball(center, radius, steep):
        return steep * (radius - np.linalg.norm(w - np.asarray(center), axis=-1))

    lg = np.zeros((5,) + shape, np.float32)
    lg[1] = ball((20.0, 0.0, 14.0), 8.0, 4.0)
    lg[2] = ball((30.0, 2.0, 18.0), 7.0, 4.0)
    lg[3] = ball((12.0, 18.0, 34.0), 4.0, 6.0)
    lg[4] = -20.0                                  # absent everywhere
    lg += 0.01 * np.sin(ix * 1.7)[None] * np.cos(iy * 2.3 + iz)[None]
    return torch.from_numpy(lg), directions, origin


def _write_ranked_store(out, code, directions, origin):
    names = ["background", "sphere_a", "sphere_b", "blob", "nowhere"]
    segments = [{"id": f"c{i}", "name": n, "label_value": i, "layer": 0}
                for i, n in enumerate(names)]
    segments.append({"id": "g1", "name": "spheres",
                     "label_value": ["c1", "c2"], "layer": 0})
    root = zarr.create_group(store=str(out))
    root.attrs.update({"duckn": {"version": "1.0", "extensions": {
        "seg": {"version": "0.6", "segments": segments}}}})
    part = root.create_group("parts/0")
    meta = dict(code.meta)
    meta.update({"labels": list(range(5)), "part": "phantom"})
    part.attrs.update({"duckn": {"version": "1.0",
                                 "extensions": {"ranked": meta}}})
    spatial = [{"kind": "space", "centering": "cell", "unit": "mm",
                "space_direction": dvec} for dvec in directions]

    def attrs(ndim):
        axes = ([{"kind": "list"}] + spatial) if ndim == 4 else spatial
        return {"duckn": {"version": "1.0", "space": "left-posterior-superior",
                          "space_origin": origin, "axes": axes}}

    for name, a in (("ranks", code.ranks), ("support", code.support),
                    ("tail", code.tail)):
        chunks = (1, 16, 16, 16) if a.ndim == 4 else (16, 16, 16)
        z = part.create_array(name, shape=a.shape, dtype=a.dtype,
                              chunks=chunks, shards=None,
                              compressors=ZSTD(level=9), attributes=attrs(a.ndim))
        z[:] = a
    return {"directions": directions, "origin": origin}


# ---------------------------------------------------------------------------

def main(store_dir, out_dir):
    global ZSTD
    ZSTD = zarr.codecs.ZstdCodec
    store_dir, out = Path(store_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(store_dir), mode="r")
    segments = root.attrs["duckn"]["extensions"]["seg"]["segments"]
    part_meta = root["parts/0"].attrs["duckn"]["extensions"]["ranked"]
    groups = _channels(segments, part_meta["labels"],
                       ["lungs", "liver", "aorta", "vertebral column"])

    dec, meta, arr_attrs, spatial = _decode(store_dir, groups)
    clip = float(meta["clip"])
    mz = zarr.create_group(store=str(out / "margins.zarr"), overwrite=True)
    z = mz.create_array("margins", shape=dec.shape, dtype=dec.dtype,
                        chunks=(1, 64, 64, 64), shards=None,
                        compressors=ZSTD(level=9),
                        attributes=_margin_attrs(spatial, arr_attrs["space_origin"],
                                                 arr_attrs["space"], clip, groups))
    z[:] = dec

    ph_logits, ph_dirs, ph_origin = _phantom_logits()
    ph_code = encode(ph_logits, depth=3, clip=CLIP)
    _write_ranked_store(out / "phantom.duckn", ph_code, ph_dirs, ph_origin)
    ph_groups = {"sphere_a": [1], "spheres": [1, 2], "nowhere": [4]}
    ph_code.meta["shape"] = list(ph_code.ranks.shape[1:])
    ph_dec = decode_groups(ph_code, list(ph_groups.values()),
                           quantize=True).numpy()
    pz = zarr.create_group(store=str(out / "phantom_margins.zarr"),
                           overwrite=True)
    zp = pz.create_array("margins", shape=ph_dec.shape, dtype=ph_dec.dtype,
                         chunks=(1, 16, 16, 16), shards=None,
                         compressors=ZSTD(level=9),
                         attributes=_margin_attrs(
                             [{"kind": "space", "centering": "cell", "unit": "mm",
                               "space_direction": dv} for dv in ph_dirs],
                             ph_origin, "left-posterior-superior", CLIP, ph_groups))
    zp[:] = ph_dec

    ranks_chunks = sum(1 for _ in (out / "phantom.duckn/parts/0/ranks/c")
                       .rglob("*") if _.is_file())
    manifest = {
        "source_store": str(store_dir.resolve()),
        "quantized_margin": {"clip": clip, "byte_for_boundary": 128,
                             "byte_for_absent": 0, "lowest_written_byte": 1},
        "real": {
            "shape_zyx": meta["shape"],
            "depth": int(meta["depth"]),
            "groups": _describe(dec, groups, clip),
            "geometry": _geometry_checkpoints(
                arr_attrs["space_origin"],
                [ax["space_direction"] for ax in spatial], meta["shape"]),
        },
        "phantom": {
            "shape_zyx": list(ph_code.ranks.shape[1:]),
            "depth": 3,
            "stored_rank_chunk_files": ranks_chunks,
            "groups": _describe(ph_dec, ph_groups, CLIP),
            "geometry": _geometry_checkpoints(
                ph_origin, ph_dirs, list(ph_code.ranks.shape[1:])),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    total = ph_code.ranks.shape[0] * np.prod(
        [-(-s // c) for s, c in zip(ph_code.ranks.shape[1:], (16, 16, 16))])
    print(f"real store: {dec.shape} decoded, {len(groups)} groups")
    for d in manifest["real"]["groups"]:
        print(f"  {d['name']:>16}: inside {d['voxels_inside']:>8}  "
              f"{d['sha256_uint8'][:16]}")
    print(f"phantom: ranks chunks stored {ranks_chunks} of {int(total)} "
          f"(missing chunks are the fill-value test)")
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
