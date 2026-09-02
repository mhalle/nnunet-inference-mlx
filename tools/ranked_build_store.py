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


# nnseg names the forward resample by where it aligns samples; duckn names the same fact by
# what a sample represents. They are one decision under two vocabularies, so the store derives
# duckn's word from nnseg's rather than stating it twice and letting the two drift.
CENTERING = {"corner": "node", "center": "cell"}


def geometry(part):
    """(true spacing zyx, first-voxel-center origin xyz, direction, centering) of the array.

    Two engines record their geometry differently, because their grids arise differently.
    FastSurfer states its conformed grid outright - the logits are native to it, there is no
    crop and the spacing is exactly 1 mm by construction. That grid is an image grid, so its
    samples are cell centres. The nnU-Net path states a canonical frame plus a requested
    spacing, so the grid it actually landed on has to be derived, and HOW depends on which
    convention the resample used:

      corner (TotalSegmentator, scipy.zoom)  holds the first and last sample centres, so
          spacing is (n_src-1)*s_src/(n_model-1) and voxel 0 does not move  -> duckn `node`
      center (nnU-Net native, skimage)       holds the field of view, so spacing is
          n_src*s_src/n_model and voxel 0 moves in by half the spacing change -> duckn `cell`

    Neither is the nominal request. Declaring the nominal spacing, or declaring `cell` for a
    corner-rule grid, misplaces the volume by up to half a voxel - silently, because the sample
    values are unaffected and only the stated geometry is wrong.
    """
    if "source_grid" in part:                                      # fastsurfer
        g = part["source_grid"]
        return list(g["spacing_zyx"]), tuple(g["origin_xyz"]), g["direction_xyz"], "cell"
    c = part["frame"]["canonical"]
    model = [int(v) for v in part["model_grid"]]
    start = [int(v) for v in part["envelope"]["start"]]
    convention = part.get("convention", "corner")
    centering = CENTERING[convention]
    if centering == "node":
        eff = [(n_s - 1) * s / (n_m - 1) if n_m > 1 else s
               for n_s, s, n_m in zip(c["shape_zyx"], c["spacing_zyx"], model)]
        shift = [0.0, 0.0, 0.0]                                    # voxel 0 stays put
    else:
        eff = [n_s * s / n_m for n_s, s, n_m in zip(c["shape_zyx"], c["spacing_zyx"], model)]
        shift = [(e - s) / 2 for e, s in zip(eff, c["spacing_zyx"])]
    D = np.asarray(c["direction_xyz"], float).reshape(3, 3)
    off_zyx = [sh + st * e for sh, st, e in zip(shift, start, eff)]  # centring, then the crop
    off_xyz = np.asarray([off_zyx[2], off_zyx[1], off_zyx[0]], float)
    origin = np.asarray(c["origin_xyz"], float) + D @ off_xyz      # the crop moves voxel 0
    return eff, tuple(float(v) for v in origin), c["direction_xyz"], centering


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


def duckn_grid(rec, centering="cell"):
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
            "axes": [{"kind": "space", "centering": centering, "unit": "mm",
                      "space_direction": [round(float(v), 9) for v in (c * s)]}
                     for c, s in zip(cols, sp)]}


BRICK = 32
DISTANCE_VOXELS = 2.0        # truncation, in voxels of the finest axis
DISTANCE_MAX = 255           # on the surface; 0 is at or beyond the truncation
JUNCTION_MAX = 127           # half range of the signed byte; 128 is on the interface


def distance_field(ranks, support, clip, spacing, truncation):
    """``(Z, Y, X)`` uint8: how far the nearest surface is, in millimetres.

    ONE FIELD, NOT A STACK. It is the distance to the nearest place the argmax changes,
    whichever pair of classes forms it. A second field keyed to the next LOGIT rank was tried
    and dropped: it measures the ``l_winner = l_third`` level set, and since the runner-up is by
    definition above the third class, that level set generically sits buried under it - a place
    no surface is drawn. Note also that logit order and distance order genuinely disagree, at
    9-15 % of the voxels where both are close, because the distance to a class's interface is
    its deficit divided by the steepness of that pair's transition and ranking by deficit
    ignores the divisor. This field is immune, being found from the labelmap rather than a rank.

    WHY STORE IT, when it is derivable from `support` sitting beside it. Unlike decoding a
    margin, which is pointwise, this is NON-LOCAL: it needs a neighbourhood of radius
    `truncation`, so a client deriving it per brick needs halos, and one deriving it whole
    spends ~53 s on a five-part 1.5 mm case before the first frame. It is also easy to get
    wrong in ways that render plausibly rather than raise. It remains a derived VIEW, not a
    replacement - `support` carries confidence, alternatives, and the ability to re-decide,
    none of which survive the conversion to millimetres.

    THE ENCODING COUNTS UP FROM THE TRUNCATION, mirroring `support` counting up from the clip:
    `distance_max` is on the surface, 0 is at or beyond `distance_truncation`. That keeps "zero
    means nothing here" true in this array too, which is not cosmetic - the band is a few per
    cent of the volume, so whole chunks of zeros elide and cost nothing, and a reader can treat
    a missing chunk as "no surface here" without decoding it.

    No sign bit: the field is unsigned magnitude, and `ranks[0]` already says which side.

    TRUNCATION IS SET BY RECONSTRUCTION, NOT BY STORAGE. One voxel is too tight to be usable -
    it holds only the layer adjacent to the surface, while a central difference for a normal
    reaches 1.5 voxels and a trilinear corner reaches sqrt(3), so neither can be evaluated from
    the stored field. Two voxels is the floor, and it is expressed in voxels because the grid
    spacing was itself chosen to match the anatomy.
    """
    d = _crossing_distance(ranks, support, clip, spacing, truncation)
    q = np.rint((1.0 - d / truncation) * DISTANCE_MAX)
    return np.where(d < truncation, np.clip(q, 0, DISTANCE_MAX), 0).astype(np.uint8)


def _deficit_at(rank_cols, support_cols, want, clip):
    """Logit deficit of class ``want`` at gathered positions.

    ``rank_cols`` / ``support_cols`` are ``(planes, N)`` columns gathered at N voxels; a dense
    per-voxel version of this ran full-volume comparisons for values discarded everywhere but
    the crossings. Deficits are stated relative to each voxel's own winner, which is what makes
    them subtractable: the reference cancels in a difference, so ``deficit(B) - deficit(A)`` is
    ``l_A - l_B`` regardless of who wins where. A class absent from the rank list is no better
    than the clip, which is the bound the encoding already guarantees.
    """
    d = np.full(want.shape, np.float32(clip))
    d[rank_cols[0] == want] = 0.0
    for j in range(1, rank_cols.shape[0]):
        hit = rank_cols[j] == want
        d[hit] = (1.0 - support_cols[j - 1][hit].astype(np.float32) / 255.0) * clip
    return d


def _crossing_distance(ranks, support, clip, spacing, truncation):
    """Distance in mm to the nearest surface: where the argmax changes.

    WHICH SURFACE IS FOUND BY THE LABELMAP, NOT BY A RANK PAIR. An earlier version watched the
    (winner, runner-up) pair for a sign change, which misses an argmax change whenever the class
    that overtakes is not the local runner-up - at one voxel l_A > l_B > l_D, at its neighbour
    l_D > l_A > l_B: the winner changed and that pair never crossed. `win[a] != win[b]` has no
    such gap, and it needs no logits at all.

    WHERE IT SITS comes from the pair that actually swaps. Deficits are stated against each
    voxel's own winner, so for P winning at `a` and Q winning at `b`, the signed field l_P - l_Q
    is +deficit_Q(a) at one end and -deficit_P(b) at the other. It crosses zero once, and linear
    interpolation puts the surface there - no divisor, and nothing to go wrong at a fold, which
    is what ``m / |grad m|`` on the winner's margin got wrong (that field is folded: it falls to
    zero at the interface and rises again beyond it, so a difference across the crossing measures
    the fold, and at a symmetric fold it measures zero).

    Everything beyond the seeded voxels is filled by propagation. Logits are gathered ONLY at
    the flipped edges - a fraction of a per cent of the volume - not evaluated densely.
    """
    win = ranks[0]
    d = np.full(win.shape, np.inf, np.float32)

    for axis, step in enumerate(float(v) for v in spacing):
        lo = [slice(None)] * win.ndim
        hi = [slice(None)] * win.ndim
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        flip = win[lo] != win[hi]
        if not flip.any():
            continue
        at = np.nonzero(flip)                          # `a` side of each flipped edge
        bt = list(at)
        bt[axis] = at[axis] + 1                        # `b` side, one step along the axis
        bt = tuple(bt)
        # deficit of the far winner here, and of the near winner there
        dq_a = _deficit_at(ranks[(slice(None),) + at], support[(slice(None),) + at],
                           win[bt], clip)
        dp_b = _deficit_at(ranks[(slice(None),) + bt], support[(slice(None),) + bt],
                           win[at], clip)
        denom = dq_a + dp_b
        # a tie splits the edge; it is the only sensible reading and it is rare
        t = np.divide(dq_a, denom, out=np.full_like(dq_a, 0.5), where=denom > 1e-9)
        np.minimum.at(d, at, t * step)
        np.minimum.at(d, bt, (1.0 - t) * step)

    return _eikonal(d, spacing, truncation)


def _eikonal(d, spacing, truncation):
    """Propagate seeded crossings outward by solving |grad d| = 1.

    NOT a min-plus sweep. `d = min(d, neighbour + h)` along each axis in turn measures a taxicab
    distance: a diagonal comes out as dx + dy rather than sqrt(dx^2 + dy^2), up to sqrt(2) too
    large in 2-D and sqrt(3) in 3-D. Shading reads a gradient, so the error appears as facets on
    every surface not aligned with an axis, and as |grad d| clustering near sqrt(2) instead of 1.

    A PLANAR TEST CANNOT SEE THIS - along its own normal the two agree exactly, which is how it
    survived one. tests/test_ranked_distance.py covers it with a sphere, measured
    against its analytic distance -- |grad d| is NOT a discriminating statistic
    here, because a narrow band is mostly clamped at the truncation.

    The Godunov update solves the Eikonal equation: with the smaller neighbour a_i on each axis,
    find d satisfying sum_i max(d - a_i, 0)^2 / h_i^2 = 1, trying one, two, then three active
    axes. Seeded voxels keep their interpolated sub-voxel values.

    ONLY THE BAND IS PROCESSED. A dense version of this update spent 34 s per 52 Mvoxel part
    running full-volume iterations to move values on the ~1 % of voxels near a surface. Each
    iteration advances influence by at most one voxel from a finite value, so dilating the seed
    mask by the iteration count (Chebyshev) contains every voxel any iteration could touch, and
    a band voxel's neighbours outside the band were never updated by the dense version either -
    they hold the same `big` in both. Bit-identical by construction, and verified against the
    dense implementation on a real 52 Mvoxel part.
    """
    h = np.asarray([float(v) for v in spacing], np.float32)
    big = np.float32(truncation * 4.0)
    n_iter = int(np.ceil(truncation / float(h.min()))) + 4

    finite = np.isfinite(d)
    if not finite.any():
        return np.minimum(d, big).astype(np.float32)

    # the reachable set: seeds dilated by n_iter voxels, separably, one voxel per pass
    band = finite.copy()
    for _ in range(n_iter):
        for axis in range(d.ndim):
            lo = [slice(None)] * d.ndim
            hi = [slice(None)] * d.ndim
            lo[axis], hi[axis] = slice(0, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            band[lo] |= band[hi]
            band[hi] |= band[lo]

    # pad by one voxel of `big` so neighbour gathers never leave the array
    padded = np.full(tuple(n + 2 for n in d.shape), big, np.float32)
    core = tuple(slice(1, -1) for _ in d.shape)
    padded[core] = np.minimum(d, big)

    bz, by, bx = np.nonzero(band)
    ny, nx = padded.shape[1], padded.shape[2]
    flat = ((bz + 1) * ny + (by + 1)) * nx + (bx + 1)
    seed = finite[bz, by, bx]
    strides = (ny * nx, nx, 1)

    r = padded.ravel()
    cur = r[flat]
    axes = [(np.minimum, s, np.float32(hv)) for s, hv in zip(strides, h)]
    for _ in range(n_iter):
        # per-axis smaller neighbour, then a 3-element sort network carrying h alongside
        # (anisotropic spacing travels with its axis through the swaps)
        trip = [(np.minimum(r[flat - s], r[flat + s]), np.full(flat.shape, hv, np.float32))
                for _, s, hv in axes]
        for i, j in ((0, 1), (1, 2), (0, 1)):
            ai, hi_v = trip[i]
            aj, hj_v = trip[j]
            swap = ai > aj
            trip[i] = (np.where(swap, aj, ai), np.where(swap, hj_v, hi_v))
            trip[j] = (np.where(swap, ai, aj), np.where(swap, hi_v, hj_v))
        (a0, h0), (a1, h1), (a2, h2) = trip
        w0, w1, w2 = 1.0 / (h0 * h0), 1.0 / (h1 * h1), 1.0 / (h2 * h2)

        sol = a0 + h0                                          # one active axis
        use2 = sol > a1
        A2, B2 = w0 + w1, a0 * w0 + a1 * w1
        C2 = a0 * a0 * w0 + a1 * a1 * w1
        disc2 = B2 * B2 - A2 * (C2 - 1.0)
        d2 = (B2 + np.sqrt(np.maximum(disc2, 0.0))) / A2       # two active axes
        ok2 = use2 & (disc2 >= 0) & (d2 <= a2)
        sol = np.where(ok2, d2, sol)
        A3, B3 = A2 + w2, B2 + a2 * w2
        C3 = C2 + a2 * a2 * w2
        disc3 = B3 * B3 - A3 * (C3 - 1.0)
        d3 = (B3 + np.sqrt(np.maximum(disc3, 0.0))) / A3       # three active axes
        sol = np.where(use2 & ~ok2 & (disc3 >= 0), d3, sol)

        cur = np.where(seed, cur, np.minimum(cur, np.minimum(sol, big)))
        r[flat] = cur

    return padded[core]


def junction_field(ranks, support, clip, spacing, truncation, reach=None):
    """``(junction, pair)``: the signed distance to the interface between two structures, near
    every TRIPLE LINE, and which two structures it is.

    WHAT THE MAIN FIELD CANNOT SAY. `distance` is the distance to the nearest surface, whichever
    surface that is. Along a triple line - where the interface between two structures comes up
    to meet the surface against a third label, background included - the nearest surface is the
    outer one, so the field is silent about where the two structures divide it. A renderer
    deciding which of the two owns a point on that shared surface then has only the labelmap,
    which is voxel-quantized, and draws the division as a staircase. The information exists in
    the logits: the margin between the two structures is continuous along the surface, its zero
    is the true division at sub-voxel precision, and it continues PAST the surface into the
    third region as a virtual sheet - which is what makes it interpolable at the surface itself,
    where half of any stencil lies in that third region.

    ONE SIGNED FIELD PER VOXEL, FOR ONE PAIR. At each voxel of the tube around a triple line,
    `pair` names the two leading real (non-background) classes in logit order, stored
    canonically by class index, and `junction` is the signed distance in millimetres to the
    level set where their logits are equal, positive on the first class's side:
    (l_a - l_b) / |grad (l_a - l_b)|. That is the deficit DIFFERENCE over its own gradient -
    never the winner's margin over its gradient, which is folded. The gradient is a central
    difference of the same two classes' deficit difference at the six axis neighbours, each read
    from that neighbour's own rank list (a class absent from a list is floored at the clip), so
    the pair is evaluated consistently across the stencil whoever wins at each tap.

    SPARSE BY CONSTRUCTION. Cells whose eight corners carry three or more labels are where a
    third region meets a two-structure interface - the triple lines, background counting as a
    label - and the field is written only within `reach` voxels of their corners. Everywhere
    else the byte is 0, the sentinel, so the array is a set of thin tubes and compresses to
    almost nothing. Along a two-structure interface away from any triple line it is 0 too: a
    reader's own pair field already places that interface exactly.

    ENCODING. 128 is on the interface, 128 +- `junction_max` are +-`junction_truncation`, 0 is
    absent. Zero is the sentinel here as in every other array. The sign lives in the byte
    because a sign is exactly what the main field lacks and this one exists to supply.

    COST. Cheap, and dominated by the two full-volume passes rather than by the field itself:
    on a 52 Mvoxel part, 0.26 s to find the triple cells, 0.18 s to dilate the tubes, and
    0.4 s to gather deficits at the 0.3 % of voxels inside them - 0.8 s in all. The torch
    twin in nnseg.ranked is byte-identical and takes 1.4 s on MPS; it is for the CUDA worker,
    where the arrays already live.
    """
    win = ranks[0]
    Z, Y, X = win.shape
    h = np.asarray([float(v) for v in spacing], np.float32)
    if reach is None:
        reach = int(np.ceil(truncation / float(h.min()))) + 1

    # Triple-line cells: three or more distinct labels among a cell's eight corners. A cell
    # with two labels is an ordinary interface; one with three is where a third region meets
    # it. The renderer's own cells are these same cells.
    corners = [win[dz:Z - 1 + dz, dy:Y - 1 + dy, dx:X - 1 + dx]
               for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    lo = corners[0]
    hi = corners[0]
    for c in corners[1:]:
        lo = np.minimum(lo, c)
        hi = np.maximum(hi, c)
    third = np.zeros(lo.shape, bool)
    for c in corners:
        third |= (c != lo) & (c != hi)
    tube = np.zeros(win.shape, bool)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                tube[dz:Z - 1 + dz, dy:Y - 1 + dy, dx:X - 1 + dx] |= third
    for _ in range(reach):
        for axis in range(3):
            a_ = [slice(None)] * 3
            b_ = [slice(None)] * 3
            a_[axis], b_[axis] = slice(0, -1), slice(1, None)
            a_, b_ = tuple(a_), tuple(b_)
            tube[a_] |= tube[b_]
            tube[b_] |= tube[a_]

    junction = np.zeros(win.shape, np.uint8)
    pair = np.zeros((2,) + win.shape, ranks.dtype)
    idx = np.nonzero(tube)
    N = idx[0].size
    if N == 0:
        return junction, pair

    # The pair: the first two real classes in each voxel's rank list. Background is class 0,
    # which `ranks` holds as 1; the sentinel 0 is not a class.
    cols = ranks[(slice(None),) + idx]
    real = (cols != 1) & (cols != 0)
    a = np.zeros(N, cols.dtype)
    b = np.zeros(N, cols.dtype)
    seen = np.zeros(N, np.int64)
    for j in range(cols.shape[0]):
        r = real[j]
        first, second = r & (seen == 0), r & (seen == 1)
        a[first] = cols[j][first]
        b[second] = cols[j][second]
        seen += r
    have = seen >= 2
    swap = have & (b < a)
    a, b = np.where(swap, b, a), np.where(swap, a, b)

    # m = l_a - l_b = deficit(b) - deficit(a), at the voxel and at its axis neighbours.
    z, y, x = idx

    def m_at(zz, yy, xx):
        q = (zz, yy, xx)
        rc = ranks[(slice(None),) + q]
        sc = support[(slice(None),) + q]
        return _deficit_at(rc, sc, b, clip) - _deficit_at(rc, sc, a, clip)

    m0 = m_at(z, y, x)
    grad2 = np.zeros(N, np.float32)
    for axis, (arr, n) in enumerate(((z, Z), (y, Y), (x, X))):
        plus = [z, y, x]
        minus = [z, y, x]
        plus[axis] = np.minimum(arr + 1, n - 1)
        minus[axis] = np.maximum(arr - 1, 0)
        span = (plus[axis] - minus[axis]).astype(np.float32) * h[axis]
        diff = m_at(*plus) - m_at(*minus)
        g = np.divide(diff, span, out=np.zeros_like(diff), where=span > 0)
        grad2 += g * g
    gmag = np.sqrt(grad2)
    # Both classes saturated at the clip on every tap: the interface is beyond reach, and the
    # sign of m is all that is known. Clamp to the truncation on that side.
    s = np.divide(m0, gmag, out=np.sign(m0) * np.float32(truncation), where=gmag > 1e-6)
    s = np.clip(s, -truncation, truncation)
    q = np.clip(np.rint(128.0 + s / truncation * JUNCTION_MAX), 1, 255).astype(np.uint8)
    q[~have] = 0
    junction[idx] = q
    pair[(0,) + idx] = np.where(have, a, 0).astype(ranks.dtype)
    pair[(1,) + idx] = np.where(have, b, 0).astype(ranks.dtype)
    return junction, pair


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


def axes(direction_xyz, eff_zyx, list_axis, centering):
    D = np.asarray(direction_xyz, float).reshape(3, 3)
    cols = [D[:, 2], D[:, 1], D[:, 0]]                             # array Z, Y, X
    sp = [{"kind": "space", "centering": centering, "unit": "mm",
           "space_direction": [round(float(v), 9) for v in (c * s)]}
          for c, s in zip(cols, eff_zyx)]
    return ([{"kind": "list"}] + sp) if list_axis else sp


def attrs(direction, eff, origin, *, list_axis, centering):
    return {"duckn": {"version": "1.0", "space": "left-posterior-superior",
                      "space_origin": [round(float(v), 6) for v in origin],
                      "axes": axes(direction, eff, list_axis, centering)}}


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


def generator_steps(meta, items, engine):
    """duckn `processing` steps: what ran, with what, and with which parameters.

    Two steps, because they are genuinely separable and can fail independently - the network
    produced a distribution, then this tool laid it out as a store. A reader debugging a wrong
    store needs to know which of those to doubt.
    """
    first = dict(items)[next(iter(dict(items)))]
    nnseg_v = first.get("nnseg")
    models = [p.get("softmax", {}) for _n, p in items if p.get("softmax")]
    seg = {"name": "Segmentation",
           "description": f"{meta.get('task')} via the {engine} engine; the pre-argmax logit "
                          "field was captured between the network and the restore",
           "software": {"name": "nnseg", "version": nnseg_v,
                        "url": "https://github.com/mhalle/nnunet-inference-mlx"},
           "parameters": {"task": meta.get("task"), "engine": engine,
                          "depth": meta.get("depth"), "clip": meta.get("clip"),
                          "envelope_mm": meta.get("envelope_mm"),
                          "device": meta.get("device")}}
    if models:
        seg["method"] = {"name": meta.get("task"),
                         "version": ", ".join(sorted({str(m.get("version")) for m in models})),
                         "models": models}
    return [seg,
            {"name": "Ranked encoding and store layout",
             "description": "top-N ranks with quantized gaps, packed as zarr v3 with one shard "
                            "per array and a per-brick occupancy index",
             "software": {"name": "ranked_build_store.py", "version": nnseg_v},
             "parameters": {"depth": meta.get("depth"), "clip": meta.get("clip"),
                            "brick": [BRICK, BRICK, BRICK], "parts_kept": "all"}}]


def build(src, out, case, parts="all", allow_unnamed=False,
          distance_voxels=DISTANCE_VOXELS):
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
        eff, origin, direction, centering = geometry(part)
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
        if "softmax" in part:
            # the identity of the normalization these classes competed in. Two parts share a
            # softmax only if this matches; margins are not comparable otherwise. See the
            # store README section 3.1.
            block["softmax"] = part["softmax"]
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
                               attributes=attrs(direction, eff, origin, list_axis=four,
                                                centering=centering))
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

        if distance_voxels:
            # An emit on a CUDA worker computes the field where the arrays already are and
            # ships it alongside; use that when it answers the same request. The recorded
            # truncation travels with the field - a precomputed array is only decodable
            # against the truncation IT was quantized to.
            pre = src / f"{name}_distance.npy"
            if pre.exists() and part.get("distance_voxels") == float(distance_voxels):
                dist = np.load(pre)
                trunc = float(part["distance_truncation"])
                print(f"    distance: precomputed at emit (T={trunc:.3f} mm)", flush=True)
            else:
                if pre.exists():
                    print(f"    distance: emit used {part.get('distance_voxels')} voxels, "
                          f"{distance_voxels} requested - recomputing", flush=True)
                trunc = float(distance_voxels) * min(eff)
                dist = distance_field(rk_all, su_all, part["clip"], eff, trunc)
            chunks, shards = layout(dist.shape)
            dz = g.create_array("distance", shape=dist.shape, dtype=dist.dtype,
                                chunks=chunks, shards=shards,
                                compressors=zarr.codecs.ZstdCodec(level=9),
                                attributes=attrs(direction, eff, origin, list_axis=False,
                                                 centering=centering))
            dz[:] = dist
            # Decode parameters, not descriptions: the quantum is truncation/max, so without
            # them the array is a uint8 with no scale. They sit beside `clip`/`support_max`,
            # which play exactly the same roles for `support`.
            block["distance_truncation"] = round(trunc, 6)
            block["distance_max"] = DISTANCE_MAX
            del dist
            # The triple-line layer, at the same truncation: what `distance` cannot say about
            # where two structures divide a shared surface. Thin tubes, so it costs nothing.
            # An emit that computed it on the worker ships it alongside, like the distance.
            pre_j = src / f"{name}_junction.npy"
            pre_p = src / f"{name}_junction_pair.npy"
            if (pre_j.exists() and pre_p.exists()
                    and part.get("junction_truncation") == round(trunc, 6)):
                jn, jp = np.load(pre_j), np.load(pre_p)
                print("    junction: precomputed at emit", flush=True)
            else:
                jn, jp = junction_field(rk_all, su_all, part["clip"], eff, trunc)
            chunks, shards = layout(jn.shape)
            jz = g.create_array("junction", shape=jn.shape, dtype=jn.dtype,
                                chunks=chunks, shards=shards,
                                compressors=zarr.codecs.ZstdCodec(level=9),
                                attributes=attrs(direction, eff, origin, list_axis=False,
                                                 centering=centering))
            jz[:] = jn
            chunks, shards = layout(jp.shape)
            pz = g.create_array("junction_pair", shape=jp.shape, dtype=jp.dtype,
                                chunks=chunks, shards=shards,
                                compressors=zarr.codecs.ZstdCodec(level=9),
                                attributes=attrs(direction, eff, origin, list_axis=True,
                                                 centering=centering))
            pz[:] = jp
            block["junction_truncation"] = round(trunc, 6)
            block["junction_max"] = JUNCTION_MAX
            print(f"    junction: {100.0 * np.count_nonzero(jn) / jn.size:.2f} % of voxels "
                  f"in the triple-line tubes", flush=True)
            del jn, jp
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
        # duckn specifies where "what produced this" goes: provenance.processing, each step
        # naming its software. Writing it here rather than inventing a field means a duckn
        # reader finds the generator in the place the spec says to look. `sources` and
        # `attribution` are filled in per case by ranked_demo_provenance.py.
        "provenance": {"version": "1.0", "processing": generator_steps(meta, items, engine)},
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
    ap.add_argument("--distance-voxels", type=float, default=DISTANCE_VOXELS,
                    help="truncation of the distance planes, in voxels "
                         f"(default {DISTANCE_VOXELS}); 0 omits them")
    a = ap.parse_args()
    build(a.ranked_dir, a.out, a.case, a.parts, a.allow_unnamed, a.distance_voxels)
