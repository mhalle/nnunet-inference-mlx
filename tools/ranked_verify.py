"""Check a ranked store against the invariants its README states.

Every failure this session was a SILENT one. A sixth positional argument that never arrived, so
`envelope_mm` kept its default and the run was subtly not what was asked for. A label-name
lookup that failed and renamed 78 segments to `label_<id>` while the build reported success. An
occupancy index copied instead of rebuilt, so it under-reported the one class the padding had
changed. None of these raised; all of them produced a store that looked fine.

So the pipeline needs a step that can say no. These are the README's claims turned into
assertions - if a check here fails, either the store is wrong or the documentation is, and
either way somebody has to look.

Cheap checks always run. `--deep` adds the ones that touch every voxel: occupancy
conservatism, segment extents, and the decode identity.

usage: uv run python tools/ranked_verify.py STORE.duckn [--deep] [--quiet]
       uv run python tools/ranked_verify.py DIR --all [--deep]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

REQUIRED_RANKED = ("version", "mode", "classes", "depth", "clip", "support_max",
                   "rank_sentinel", "labels", "part", "task", "model_grid", "envelope")


class Report:
    def __init__(self, name, quiet=False):
        self.name, self.quiet, self.fail, self.warn, self.ok = name, quiet, [], [], 0

    def check(self, cond, msg, *, warn_only=False):
        if cond:
            self.ok += 1
        elif warn_only:
            self.warn.append(msg)
        else:
            self.fail.append(msg)
        return bool(cond)

    def emit(self):
        status = "FAIL" if self.fail else ("warn" if self.warn else "ok")
        if not (self.quiet and status == "ok"):
            print(f"  {self.name:<44} {self.ok:>3} checks  {status}")
        for m in self.fail:
            print(f"      FAIL  {m}")
        for m in self.warn:
            print(f"      warn  {m}")
        return not self.fail


def verify(path: Path, deep: bool = False, quiet: bool = False) -> bool:
    r = zarr.open_group(str(path), mode="r")
    root = r.attrs.asdict().get("duckn", {})
    ext = root.get("extensions", {})
    rep = Report(f"{path.parent.name}/{path.name}", quiet)

    rep.check((path / "README.md").exists(), "no README.md - the format reference is missing")
    rep.check("seg" in ext, "no seg extension on the root group")
    rep.check("nnseg" in ext, "no nnseg block on the root group")
    pv = ext.get("provenance") or {}
    rep.check(bool(pv), "no duckn provenance extension - nothing says where this came from")
    rep.check(bool(pv.get("sources")), "provenance has no sources - the case is unidentifiable",
              warn_only=True)
    steps = pv.get("processing") or []
    rep.check(bool(steps), "provenance has no processing steps - nothing says what made this")
    rep.check(all(s.get("software", {}).get("name") for s in steps),
              "a processing step names no software")
    rep.check(all(s.get("name") for s in steps),
              "a processing step has no name (required by the extension)")
    order = ext.get("nnseg", {}).get("part_order", [])
    rep.check(bool(order), "empty part_order - paint order is external knowledge and must "
                           "be recorded")

    segs = ext.get("seg", {}).get("segments", [])
    leaves = [s for s in segs if not isinstance(s["label_value"], list)]
    rep.check(all("name" in s and not str(s["name"]).startswith("label_") for s in leaves),
              f"{sum(1 for s in leaves if str(s.get('name','')).startswith('label_'))} of "
              f"{len(leaves)} segments are unnamed (label_<id>) - the name lookup degraded")
    rep.check(all(0 <= s.get("layer", 0) < max(len(order), 1) for s in leaves),
              "a segment's layer is not a valid part index")

    origins, directions = set(), set()
    for i, _p in enumerate(order):
        if f"parts/{i}" not in r:
            rep.check(False, f"parts/{i} declared in part_order but absent")
            continue
        g = r[f"parts/{i}"]
        m = g.attrs.asdict().get("duckn", {}).get("extensions", {}).get("ranked", {})
        miss = [k for k in REQUIRED_RANKED if k not in m]
        rep.check(not miss, f"parts/{i}: ranked block missing {miss}")
        if miss:
            continue

        soft = m.get("softmax") or {}
        rep.check(bool(soft), f"parts/{i}: no softmax block - a reader cannot tell which "
                              "normalization these classes competed in")
        if soft:
            rep.check(soft.get("classes") == m["classes"],
                      f"parts/{i}: softmax.classes {soft.get('classes')} != classes "
                      f"{m['classes']}")
            rep.check(bool(soft.get("weights")), f"parts/{i}: softmax names no weights")

        K, smax, sent = m["classes"], m["support_max"], m["rank_sentinel"]
        rep.check(len(m["labels"]) == K,
                  f"parts/{i}: labels has {len(m['labels'])} entries for {K} classes")
        rep.check(sent == 0, f"parts/{i}: rank_sentinel is {sent}, not 0 - the whole "
                             "'zero means absent' contract assumes 0")

        env, grid = m["envelope"], m["model_grid"]
        rep.check(len(env) == 6, f"parts/{i}: envelope must be 6 inclusive bounds, got {len(env)}")
        if len(env) == 6:
            lo, hi = env[0::2], env[1::2]
            rep.check(all(a <= b for a, b in zip(lo, hi)),
                      f"parts/{i}: envelope has an inverted bound {env} - inclusive, not half-open?")
            rep.check(all(0 <= a and b < n for a, b, n in zip(lo, hi, grid)),
                      f"parts/{i}: envelope {env} escapes model_grid {grid}")

        ranks = g["ranks"]
        rep.check(ranks.ndim == 4, f"parts/{i}: ranks must be 4-D, got {ranks.ndim}")
        shape = tuple(ranks.shape[1:])
        if len(env) == 6:
            want = tuple(b - a + 1 for a, b in zip(env[0::2], env[1::2]))
            rep.check(shape == want,
                      f"parts/{i}: array is {shape} but envelope says {want}")
        rep.check("support" in g, f"parts/{i}: no support array")
        if "support" in g:
            rep.check(g["support"].shape[0] == ranks.shape[0] - 1,
                      f"parts/{i}: support has {g['support'].shape[0]} planes for "
                      f"{ranks.shape[0]} rank planes (want one fewer)")
            rep.check(tuple(g["support"].shape[1:]) == shape,
                      f"parts/{i}: support grid {tuple(g['support'].shape[1:])} != ranks {shape}")
        rep.check(m.get("exhaustive") or "tail" in g,
                  f"parts/{i}: not exhaustive but no tail array", warn_only=True)

        if "distance" in g:
            # The quantum is truncation/max, so without both the array is a uint8 with no
            # scale - the same roles `clip`/`support_max` play for `support`.
            rep.check("distance_truncation" in m,
                      f"parts/{i}: distance array but no distance_truncation - "
                      "the field cannot be decoded")
            rep.check("distance_max" in m,
                      f"parts/{i}: distance array but no distance_max - "
                      "the field cannot be decoded")
            rep.check(tuple(g["distance"].shape) == shape,
                      f"parts/{i}: distance grid {tuple(g['distance'].shape)} "
                      f"!= ranks {shape} - it must be ONE 3-D field, like tail")
            t = m.get("distance_truncation")
            rep.check(t is None or t > 0, f"parts/{i}: distance_truncation {t} is not positive")

        # Centering is the sample-count-to-extent relationship, so it decides what a resampler
        # holds fixed. It was hardcoded to "cell" on grids the corner rule produced, which is
        # duckn's "node" - harmless while duckn's resample() ignored the field, a half-voxel
        # shift now that it honors it. Restated here rather than imported: a verifier that
        # imports the builder inherits the builder's mistakes.
        want_data = {"corner": "node", "center": "cell"}.get(m.get("resample_alignment"))
        for nm in ("ranks", "support", "tail", "occupancy"):
            if nm not in g:
                continue
            ax = g[nm].attrs.asdict().get("duckn", {}).get("axes", [])
            cen = {a.get("centering") for a in ax if a.get("space_direction")}
            rep.check(len(cen) <= 1,
                      f"parts/{i}/{nm}: spatial axes disagree on centering {cen}")
            # occupancy is a brick summary whose samples do own a cell, whatever grid the
            # data arrays landed on.
            expect = "cell" if nm == "occupancy" else want_data
            rep.check(expect is None or cen == {expect},
                      f"parts/{i}/{nm}: centering {cen or 'unset'} but the grid is "
                      f"{m.get('resample_alignment')!r}-aligned, which is {expect}")
            # a single-chunk array is already one file; sharding it would add an index for
            # nothing. Only multi-chunk arrays need it.
            nchunks = int(np.prod([int(np.ceil(s / c))
                                   for s, c in zip(g[nm].shape, g[nm].chunks)]))
            rep.check(g[nm].shards is not None or nchunks == 1,
                      f"parts/{i}/{nm}: {nchunks} loose chunks - ~2.5x the disk and one "
                      "request per chunk", warn_only=True)

        at = g["ranks"].attrs.asdict().get("duckn", {})
        ax = at.get("axes", [])
        rep.check(len(ax) == 4 and ax[0].get("kind") == "list",
                  f"parts/{i}: ranks axes must be [list, space, space, space], got "
                  f"{[a.get('kind') for a in ax]}")
        rep.check("space_origin" in at, f"parts/{i}: ranks has no space_origin")
        origins.add(tuple(round(float(v), 4) for v in at.get("space_origin", [])))
        directions.add(tuple(round(float(x), 6) for a in ax if a.get("kind") == "space"
                             for x in a.get("space_direction", [])))

        if "occupancy" in g:
            rep.check("brick" in m, f"parts/{i}: occupancy present but no declared brick - "
                                    "the index would be tied to the storage layout")
            if "brick" in m:
                b = m["brick"]
                want = (K,) + tuple(int(np.ceil(s / bb)) for s, bb in zip(shape, b))
                rep.check(tuple(g["occupancy"].shape) == want,
                          f"parts/{i}: occupancy is {tuple(g['occupancy'].shape)}, want {want}")

        if deep:
            rk0 = np.asarray(ranks[0])
            rep.check(not (rk0 == 0).any(),
                      f"parts/{i}: ranks[0] holds the sentinel at "
                      f"{int((rk0 == 0).sum())} voxels - every voxel must have a winner")
            lut = np.asarray(m["labels"])
            glob = lut[rk0.astype(np.int64) - 1]
            if "occupancy" in g and "brick" in m:
                occ, b = np.asarray(g["occupancy"]), m["brick"][0]
                missed = 0
                for c in range(K):
                    if lut[c] == 0:
                        continue
                    hit = glob == lut[c]
                    if not hit.any():
                        continue
                    z, y, x = np.nonzero(hit)
                    truth = np.zeros(occ.shape[1:], bool)
                    truth[z // b, y // b, x // b] = True
                    missed += int((truth & (occ[c] != smax)).sum())
                rep.check(missed == 0,
                          f"parts/{i}: occupancy missed {missed} bricks that contain the class "
                          "- the index is NOT conservative and skipping it loses data")
            for s in leaves:
                if s.get("layer", 0) != i or "extent" not in s:
                    continue
                hit = glob == s["label_value"]
                if not hit.any():
                    rep.check(False, f"parts/{i}: segment {s['name']} has an extent but no voxels")
                    continue
                idx = np.nonzero(hit)
                want = [int(v) for d in range(3) for v in (idx[d].min(), idx[d].max())]
                rep.check(list(s["extent"]) == want,
                          f"parts/{i}: {s['name']} extent {s['extent']} != actual {want}")

    softmaxes = [(r[f"parts/{i}"].attrs.asdict().get("duckn", {}).get("extensions", {})
                  .get("ranked", {}).get("softmax") or {}).get("weights")
                 for i, _ in enumerate(order) if f"parts/{i}" in r]
    if len(order) > 1 and all(softmaxes):
        rep.check(len(set(softmaxes)) == len(softmaxes)
                  or len(set(softmaxes)) == 1,
                  f"parts share weights inconsistently ({softmaxes}) - either every part is a "
                  "distinct model or they are all one; a mix means the part split is wrong",
                  warn_only=True)

    if len(order) > 1:
        rep.check(len(origins) == 1,
                  f"parts do not share one origin ({len(origins)} distinct) - a reader must "
                  "then carry per-part offsets", warn_only=True)
        rep.check(len(directions) == 1,
                  f"parts do not share one orientation ({len(directions)} distinct)")
    return rep.emit()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--all", action="store_true", help="verify every *.duckn under path")
    ap.add_argument("--deep", action="store_true", help="add the whole-volume checks")
    ap.add_argument("--quiet", action="store_true", help="print only failures")
    a = ap.parse_args()

    stores = sorted(a.path.glob("*/*.duckn")) if a.all else [a.path]
    if not stores:
        sys.exit(f"no stores found under {a.path}")
    print(f"verifying {len(stores)} store(s){' (deep)' if a.deep else ''}")
    bad = [s for s in stores if not verify(s, a.deep, a.quiet)]
    print(f"\n{len(stores) - len(bad)}/{len(stores)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
