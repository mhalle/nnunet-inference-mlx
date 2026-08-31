# What is in this directory

A **ranked segmentation store**: the output *distribution* of one or more segmentation models,
not just the labels they chose. A labelmap answers *which class won*; this keeps the second
answer — *by how much* — for roughly a third of a byte per voxel.

It is a [zarr](https://zarr.dev) v3 hierarchy. Everything below is generic to the format; no
knowledge of the particular images, models, or anatomy is needed to read it.

---

## 1. Layout

```
zarr.json                  root group
parts/
  0/  1/  2/ ...           one group per MODEL that ran (a "part")
      zarr.json            the ranked metadata block for that part  <- read this first
      ranks/               (N, Z, Y, X)  uint8 or uint16
      support/             (N-1, Z, Y, X) uint8
      tail/                (Z, Y, X)     uint8   (may be absent)
```

Parts are numbered, not named — a name lives in metadata (`part`). Some tasks are a single
model (one part); some composite several models covering different structures.

Read `parts/<i>/zarr.json` → `attributes.duckn.extensions.ranked` before touching the arrays.
It carries `classes`, `depth`, `clip`, `support_max`, `rank_sentinel`, `labels`, and the grid.

---

## 2. The three arrays

Every array is indexed `[..., Z, Y, X]` in **array order** (slowest axis first).

### `ranks` — who is here, best first

`ranks[j, z, y, x]` is **`class_index + 1`** of the *j*-th best class at that voxel.

- **Plane `j` is a rank, not a class.** `ranks[1]` means "the runner-up *here*", and the
  runner-up is a different class at different voxels. **Ranks are addresses, not values.**
- **`0` means "no class here"** (`rank_sentinel`). The `+1` shift exists because class `0` is
  usually a legitimate class (background), so a bare `0` could not mean "absent".
- **`ranks[0]` is the argmax and never holds the sentinel** — every voxel has a winner.
  Therefore **`ranks[0] - 1` *is* the labelmap**, with no special cases.

### `support` — how far behind, counting up from the clip

`support[j]` describes the class named by `ranks[j+1]` (so `support` has one fewer plane than
`ranks`; there is nothing to store for the winner, which trails by definition by zero).

Convert a stored byte to **logits behind the winner**:

```
gap = (1 - support / support_max) * clip          # support_max is normally 255
```

- `support == support_max` → gap 0 → **tied with the winner**.
- `support == 0` → gap ≥ `clip` → **at the clip**: this class is indistinguishable from absent.

Counting *up from the clip* rather than down from the winner is what makes `0` mean "nothing
here" in this array too.

### `tail` — what was thrown away

`tail[z,y,x] / support_max` is the fraction of probability mass **outside** the top `N` classes.
Probabilities can be renormalized exactly with `Z = Z_top / (1 - tail)`.

It is often all zeros at useful depths and may be **absent entirely** (`exhaustive: true`, i.e.
`depth >= classes`, or the writer dropped it). Treat a missing `tail` as zero.

### Zero means "nothing here" in all three

That is deliberate and load-bearing: an unwritten chunk, a `calloc`'d buffer, a zero-cleared GPU
texture, or a sparse default all decode to the **safe** answer. It is also why the arrays are
tiny — regions where nothing is close to the winner (air, deep interiors) are uniformly zero and
compress away, and empty chunks are not stored at all.

---

## 3. Two different fields come out of the same bytes

This is the one trap in the format. Decoding gives you a per-class value, and there are **two**
of them, differing in exactly one place — the winner's own channel.

```
deficit  d_c = l_c - max_j     l_j      the winner is 0, everyone else negative
margin   m_c = l_c - max_{j!=c} l_j     the winner gets +its lead over the runner-up
```

**Use `deficit` to reconstruct labels.** It is the logits shifted by a per-voxel constant shared
by *every* channel — a gauge transformation — so interpolating it and taking the argmax is
exactly interpolating the logits and taking the argmax.

**Use `margin` to render or mesh one structure.** Its zero level set is that structure's surface,
it is positive inside by however much the class leads, and it has an interior gradient to shade
with. `deficit` is flat zero throughout the interior and has no usable isosurface.

**Do not substitute one for the other.** At voxel centers the winner wins either way, so an
argmax over margins looks correct — and then breaks once any interpolation stencil mixes voxels
with different winners, because the lead is added to a *different* channel at each voxel.
Nearest-neighbor resampling cannot expose this at all, which is how such a bug survives review.

---

## 4. Decoding

**Never interpolate a stored plane directly.** Plane *j* does not mean a fixed thing across
voxels. Scatter to dense per-class first, then interpolate.

```python
import numpy as np, zarr

g    = zarr.open_group("<this directory>", mode="r")["parts/0"]
meta = g.attrs["duckn"]["extensions"]["ranked"]
K, clip, smax = meta["classes"], meta["clip"], meta["support_max"]

ranks   = np.asarray(g["ranks"])                    # (N, Z, Y, X)  values are class + 1
support = np.asarray(g["support"])                  # (N-1, Z, Y, X)
gap     = lambda s: (1.0 - s / smax) * clip         # bytes -> logits behind the winner

n_rank, n_sup = ranks.shape[0], support.shape[0]
# what a class NOT named at this voxel is worth: no better than the last one that was
floor = -gap(support[n_sup - 1]) if n_sup >= n_rank else np.float32(-clip)

# Allocate K+1 channels and scatter with `ranks` UNSHIFTED. Because ranks holds class+1, the
# sentinel 0 lands in scratch channel 0 and every real class lands at class+1. Subtracting 1
# from ranks instead would send the sentinel to index -1, i.e. silently into the LAST class.
full = np.empty((K + 1,) + ranks.shape[1:], np.float32)
full[1:] = floor
for j in range(n_rank - 1, 0, -1):                  # losers first
    np.put_along_axis(full, ranks[j:j+1].astype(np.intp), -gap(support[j-1])[None], axis=0)
np.put_along_axis(full, ranks[0:1].astype(np.intp), 0.0, axis=0)      # winner last
deficit = full[1:]                                  # channel c is class c

labels = ranks[0].astype(np.int64) - 1              # the labelmap: read it, do not re-derive
margin = deficit.copy()                             # margin for every class, everywhere:
lead   = gap(support[0])                            # the winner's lead over the runner-up
np.put_along_axis(margin, labels[None], lead[None], axis=0)
```

Use `put_along_axis`, **not** `putmask` — putmask fills cyclically from a flattened array and
does not align positionally, which silently scrambles the stack.

⚠ **Take the labelmap from `ranks[0]`; do not recover it with `argmax(deficit)`.** The two
disagree at voxels where the runner-up quantized to "tied": one stored step is
`clip / support_max` logits, so `support == support_max` means the true gap was under half of
that — not that it was zero. At such a voxel several classes decode to exactly 0.0, and
`argmax` resolves the tie by lowest class index, which need not be the class that actually won.
Measured on one real store: 506 of 1.43 M voxels had a quantized-tied runner-up, and in 9 of
them `argmax` returned it instead of the stored winner. `ranks[0]` is exact and free.

**Shortcuts that avoid the dense stack entirely:**

- the labelmap is `ranks[0] - 1`; read only that plane;
- one structure's margin needs only the planes where it appears, plus the floor;
- a **union** of classes S has margin `max(d_c for c in S) - max(d_c for c not in S)`, which is
  one pass over the rank planes and never needs all K channels resident.

---

## 5. Labels

`labels` in the part metadata maps **channel index → label id**: `labels[c]` is the id that
channel `c` denotes in whatever labeling scheme the run used. Channel ids are local to a model;
label ids are shared across the store's parts.

```python
label_volume = np.asarray(meta["labels"])[ranks[0].astype(np.int64) - 1]
```

If a `labels_note` field is present, **read it** — it warns where the mapping is not the whole
story (for example, when a downstream step splits a channel spatially, so laterality or a
similar attribute is *not* recoverable from the LUT alone).

Human-readable names, when present, are in the root group's
`attributes.duckn.extensions.seg.segments`, each with `label_value` and `name`. An entry whose
`label_value` is a **list** is a group (a union of other segments), not a class.

---

## 6. Geometry

Each array carries a `duckn` block in its attributes:

- `space` — the world frame the coordinates are in (e.g. `"left-posterior-superior"`).
- `space_origin` — world coordinates of the **center of voxel `[0,0,0]`**.
- `axes` — one entry per **array** axis, in array order. Each spatial entry's
  `space_direction` is a world-space vector whose **length is that axis's spacing**, so a
  non-axis-aligned (oblique) acquisition is represented exactly. A leading entry of
  `{"kind": "list"}` marks the non-spatial rank axis of a 4-D array.

```python
world = space_origin + sum(index[axis] * space_direction[axis] for spatial axes)
```

The part metadata also records the grid the model computed on:

- `model_grid_zyx` — the full grid.
- `envelope: {start, stop}` — the half-open sub-box actually stored. `start` is `[0,0,0]` when
  the whole grid is present. **If `start` is non-zero, the array is a crop and its
  `space_origin` already accounts for the offset** — do not apply it twice.
- `padded_from`, if present, means voxels outside the original computed region were **filled
  in, not inferred**; the field says what value was used. Those voxels are an assertion, not a
  model output.

Do not trust a nominal spacing field (e.g. `nominal_spacing_zyx`) as the true grid spacing — it
records what was *requested*. The `space_direction` vectors are the actual geometry.

---

## 7. Storage

zarr v3, one **shard** per array (`sharding_indexed`), 64³ inner chunks, zstd, `fill_value: 0`.
Any zarr v3 reader opens this as an ordinary directory; no special store class is needed.

Consequences worth knowing:

- **Empty inner chunks cost nothing** — the shard index marks them missing. This is why a store
  covering mostly air is small.
- **Partial reads work**: fetch the shard index, then range-read only the inner chunks you want.
  The leading chunk dimension of a 4-D array is `1`, so a chunk never spans the rank axis and
  reading `ranks[0]` alone touches only that plane's chunks.
- Reading progressively deeper is reading more planes in order; a reader may stop early.

---

## 8. Quick sanity checks

If you are unsure a store is intact:

- `ranks[0]` should contain **no zeros** (every voxel has a winner).
- The decoded `deficit` of the winner should be **exactly 0.0** everywhere.
- `argmax(deficit)` should equal `ranks[0] - 1` **except at exact ties** — see the warning in
  §4. A handful of disagreeing voxels is expected and correct; a large fraction is not, and
  points at a wrong `clip`, a mis-scaled `gap`, or an off-by-one in the scatter.
- `support` planes should be mostly `0` (most classes at most voxels are past the clip);
  a nearly-uniform high value suggests the wrong `clip` or a mis-scaled decode.
- `max_tail` in the metadata bounds the worst single voxel's discarded mass; if it is tiny, the
  `tail` array carries almost nothing and can be ignored.

## 9. Precision

Values are quantized to one byte over the range `clip`, so one step is `clip / support_max`
logits (≈ 0.03 at the common `clip = 8`, `support_max = 255`) and any decoded gap is accurate to
half a step. That is the format's error floor: differences finer than a step are not
recoverable, which is why exact ties appear in decoded output that were not ties in the model's
logits. `clip` also bounds what is representable at all — a class more than `clip` logits behind
the winner is stored as absent and decodes to the floor, not to its true value.
