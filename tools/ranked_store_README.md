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
      occupancy/           (K, Zb, Yb, Xb) uint8  a skip index; see §7.1 (may be absent)
```

**A part is one softmax, and that is the unit that matters.** The classes within a part competed
in a single normalization; classes in different parts did not. This is not the same as "one
task" — a five-model task like `total` is five separate softmaxes. See §3.1.

Parts are numbered, not named — a name lives in metadata (`part`). Some tasks are a single
model (one part); some composite several models covering different structures.

Read `parts/<i>/zarr.json` → `attributes.duckn.extensions.ranked` before touching the arrays.
It carries `classes`, `depth`, `clip`, `support_max`, `rank_sentinel`, `labels`, and the grid.

---

## 2. The three arrays

These three hold the data. `occupancy`, if present, is a derived skip index and carries no
information of its own — see §7.1; `distance`, if present, is a derived geometric view.

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

### `distance` — where the nearest surface is, in millimetres (optional)

Present only when the store was built with it. One 3-D field on the data grid, like `tail`: the
distance from this voxel to the **nearest surface** — the nearest place the argmax changes,
whichever pair of classes forms it. Two part-block fields decode it, and the array cannot be
read without them:

```python
d_mm = (1 - distance / distance_max) * distance_truncation   # distance_max = on the surface
```

Like `support`, it counts **up** from the far end: `distance_max` is at the surface, `0` is at
or beyond `distance_truncation`. No sign is stored — the field is unsigned magnitude, and
`ranks[0]` already says which side of the surface a voxel is on.

**Scope: this is the distance to the nearest surface of the whole labelmap.** It renders the
whole scene, or any single structure, directly. For a **selection** of structures it is a
preview, not the answer: its zero set includes surfaces internal to the selection and surfaces
of everything else. `ranks[0]` and `ranks[1]` can classify the nearest surface (outer boundary
iff exactly one side is selected) for an instant preview-grade mask — but do **not** gate a
sub-voxel shell's opacity with any sampled membership mask: a voxel-quantized gate on a
sub-voxel surface renders as stipple, and no amount of supersampling removes it. The quality
path is to **re-seed**: on the fetched region, seed edges where selection membership flips,
propagate with the Eikonal update, and render that field with no gates anywhere — it is zero
exactly on the selection's boundary. Measured cost: about a second on a lungs-sized crop.

The sub-voxel crossing positions come from this array itself: at a flip edge,
`t = d_a / (d_a + d_b)` with the decoded distances on either side. For a locally planar
surface that is exact — the perpendicular-versus-along-edge cosine cancels in the ratio — so
the reseed needs only `ranks[0]` and `distance`. Measured against deficit-derived crossings
on a real five-lobe selection: median 4 µm per edge, 1.4 % of the band moved more than a
tenth of a voxel, and about 1 % of rendered pixels differ, confined to thin seams where a
flip voxel's nearest surface is a different one (a fissure behind it). Seeding from the
deficits (`t = deficit_far / (deficit_far + deficit_near)`, gathered from `ranks`+`support`)
remains the exact option for those seams.

There is deliberately only one field. A second one keyed to the next *logit* rank was tried and
dropped: it measures the level set where `l_winner = l_third`, and since the runner-up is by
definition above the third class, that level set generically sits buried under it — a place no
surface is drawn. Note also that logit order and distance order genuinely disagree, at 9–15 %
of the voxels where both are close: the distance to a class's interface is its deficit divided
by the steepness of that pair's transition, and ranking by deficit ignores the divisor. This
field is immune, being found from the labelmap rather than from a rank.

It is a **derived view, not a replacement for `support`**. Distance says where a surface is;
it cannot say how confident the model was, what the alternatives were, or let you re-decide
after the fact. Everything that needs those still reads the logits. What distance buys is a
shared unit: margins are logits scoped to one softmax and are not comparable across parts, so a
five-part composite can only be combined by paint order — millimetres let the same parts be
combined by nearest surface, and make cross-part geometric questions answerable. It is stored
rather than derived on read because the derivation is **non-local** (a neighbourhood of radius
`distance_truncation`, so per-brick derivation needs halos and whole-volume derivation costs
tens of seconds) and easy to get wrong in ways that render plausibly rather than raise.

Three cautions if you compute your own:

- **Find surfaces with the labelmap, not with a rank pair.** Watching the (winner, runner-up)
  pair for a sign change misses every argmax change where the class that overtakes is not the
  local runner-up. `ranks0[a] != ranks0[b]` across an edge is exact.
- **Do not divide the winner's margin by its own gradient.** That field is *folded* — it falls
  to zero at the surface and rises again beyond it — so a difference taken across the crossing
  measures the fold, not the slope, and at a symmetric fold it measures zero. Interpolate the
  *signed* field of the pair that actually swaps.
- **Propagate with an Eikonal (Godunov) update, not min-plus sweeps.** `min(d, neighbour + h)`
  measures a taxicab distance — up to √3 too large off-axis — and the error shades as facets.
  A planar test cannot catch either of the last two; a sphere against its analytic distance
  catches both.

The truncation is set by reconstruction, not by storage: a central difference for a normal
reaches 1.5 voxels and a trilinear corner reaches √3, so at `T` = 1 voxel neither can be
evaluated from the stored field. Two voxels is the floor.

### `junction` — where two structures divide a shared surface (optional)

Present only when the store was built with it, beside `distance`. Two arrays and two decode
fields in the part block, and the arrays cannot be read without them:

```
junction        (Z, Y, X)     uint8    signed distance to the interface between the pair
junction_pair   (2, Z, Y, X)  ranks'   the pair, as class + 1, lower class index first

s_mm = (junction - 128) / junction_max * junction_truncation      # only where junction != 0
```

**What it answers that `distance` cannot.** `distance` is the distance to the nearest surface,
whichever it is. Along a **triple line** — where the interface between two structures comes up
to meet the surface against a third label, background included — the nearest surface is the
outer one, so `distance` is silent about where the two structures divide it. A reader deciding
which of the two owns a point on that shared surface then has only the labelmap, which is
voxel-quantized, and draws the division as a staircase. The information exists in the logits:
the margin between the two structures is continuous along the surface, its zero is the true
division at sub-voxel precision, and it continues *past* the surface into the third region as
a virtual sheet. `junction` is that sheet's signed distance, `(l_a - l_b) / |grad (l_a - l_b)|`,
in millimetres, **positive on the side of `junction_pair[0]`**, the lower class index.

**Where it is written.** Only within a few voxels of cells whose eight corners carry three or
more labels — the triple lines. Everywhere else, including along a plain two-structure
interface far from any third region, the byte is 0: there a reader's own pair field (the signed
`distance` of one structure) already places the interface exactly. The array is a set of thin
tubes and compresses to almost nothing.

**How to read it at a surface point.** Take the eight corners of the cell, keep those whose
`junction_pair` is the pair in question (in either order — flip the sign if the order is
reversed), renormalize their trilinear weights, and interpolate `s`. Do *not* interpolate across
corners carrying a different pair. The result is a signed distance along the surface to the
division between the two structures, continuous through the third region, so it can be
filtered at whatever width the display needs — one pixel's footprint — instead of being blended
at the voxel's width, which is a blur when close and nothing when far.

**Two cautions.**

- It is a deficit *difference* over its own gradient, never the winner's margin over its
  gradient: the winner's margin is folded and measures zero at a symmetric fold.
- The gradient is taken from the same two classes at every tap of the stencil, each read from
  that voxel's own rank list, whoever wins there. A class absent from a list is no better than
  the clip, so far from the interface the field saturates at the truncation. That is the
  intended reading: beyond the truncation there is nothing to divide.

### Zero means "nothing here" in every array

That is deliberate and load-bearing — in `distance` too, where 0 means "no surface within the
truncation": an unwritten chunk, a `calloc`'d buffer, a zero-cleared GPU texture, or a sparse
default all decode to the **safe** answer. It is also why the arrays are
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

### 3.1 A margin is scoped to one softmax

There is no absolute confidence in a class to threshold. Softmax has a gauge freedom — adding a
constant to every logit changes nothing — so only *differences* are identifiable, and every
stored value is a comparison against the classes that competed. "Certainty about the liver" is
always "certainty the liver beats whatever else is here."

**Which softmax is recorded.** Each part's `ranked.softmax` names the normalization its classes
competed in:

```json
"softmax": {"engine": "nnunetv2", "weights": "291", "classes": 25,
            "version": "v2.0.4", "sha256": "...", "folder": "..."}
```

Two parts share a softmax **only if this matches**. Do not infer it from the task name — a task
can be five models — nor from the folder name, which does not identify the weights version
(`Dataset297` ships as both v2.0.0 and v2.0.4 and unpacks to the same folder). `version` may be
`"unknown"` when the weights were installed by something other than nnseg and left no version
sidecar; that is reported rather than guessed, because guessing is wrong in exactly the case
versioning exists for.

**Within one part** the classes partition the volume: every voxel has exactly one winner. So

- a set `S` and its **complement** describe literally one surface — `m_S == -m_complement`,
  exactly, verified to 0.0;
- two disjoint sets never both claim a voxel, and where they touch there is no gap and no
  double claim (though other classes may lie between them, so two *proper* subsets do not in
  general share a surface).

**Across parts nothing enforces any of that.** Each part places its own zero level set against
its own competitors, in its own gauge. Two margins from different parts are not comparable
numbers, so overlaps cannot be arbitrated by confidence — they are resolved by paint order,
which is what `part_order` in the root metadata is for. Measured on a five-part task: 0.010 %
of voxels claimed by two parts, none by three. Small, but it is not zero and nothing makes it
zero.

Practically: compose freely inside a part, and treat any composite spanning parts — or spanning
tasks, which additionally means different grids — as needing an explicit resolution rule.

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

A segment may also carry:

- `layer` — which part owns its voxels. Two segments with different `layer` values came from
  different softmaxes; see §3.1 before comparing or compositing them.
- `extent` — `[min_i, max_i, min_j, max_j, min_k, max_k]`, **inclusive**, in the array's storage
  order, non-spatial axes not counted. Absent when the class does not appear. Use it to skip
  straight to a structure, and to test truncation as described in §6.

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

**Axis order is never spelled in a key.** It is structural: `axes` has one entry per array
axis, in array order, so any other per-axis list in this file is in that same order. A
referenced grid (e.g. `target_grid`) is written in exactly this vocabulary too — `space`,
`space_origin`, `samples`, `axes` — so one parser handles both.

The part metadata also records the grid the model computed on:

- `model_grid` — the full grid, as a sample count per array axis.
- `envelope` — the sub-box actually stored, as `[min_i, max_i, min_j, max_j, min_k, max_k]`
  with **both bounds inclusive**, the same convention as a segment's `extent` below. It is
  `[0, n_i-1, 0, n_j-1, 0, n_k-1]` when the whole grid is present. **If a minimum is non-zero
  the array is a crop, and its `space_origin` already accounts for the offset** — do not apply
  it twice.
- `brick` — the edge of one occupancy brick, per axis. See §7.1.
- `padded_from`, if present, means voxels outside the original computed region were **filled
  in, not inferred**; the field says what value was used. Those voxels are an assertion, not a
  model output.

Do not trust a nominal spacing field (`nominal_spacing`) as the true grid spacing — it records
what was *requested*. The `space_direction` vectors are the actual geometry; a request of 1.5 mm
routinely lands at 1.504 mm.

### `centering` — what a sample owns, and what a resample must hold fixed

Each spatial axis declares `centering`. It is **not** a statement about where the samples are:
under either value a sample sits at `space_origin + index * space_direction`, and `space_origin`
is the first sample's position either way. It states the array's **extent** — whether the
footprint runs half a voxel past the outermost samples.

| value | meaning | extent along an axis of `n` samples |
|---|---|---|
| `cell` | each sample owns a cell; its position is the cell center | `n * spacing` |
| `node` | samples sit on the cell boundaries | `(n - 1) * spacing` |

That is what a resampler holds fixed, so reading it wrong displaces the volume by half a voxel
— silently, because the sample values are unaffected and only the stated geometry is wrong.

In these stores:

- **`ranks`, `support` and `tail` are `node`** whenever the part's `resample_alignment` is
  `corner`, which is what nnU-Net / TotalSegmentator pipelines use: the forward resample held
  the first and last sample centers, so the spacing is `(n_src - 1) * s_src / (n_model - 1)`
  and voxel 0 did not move. A part aligned `center` is `cell` instead, with spacing
  `n_src * s_src / n_model` and voxel 0 moved in by half the spacing change. An engine whose
  logits are native to an acquired grid — FastSurfer's conformed volume — is `cell` and carries
  no `resample_alignment`, because nothing was resampled.
- **`occupancy` is always `cell`**, whatever the data arrays are. It is a brick summary and a
  brick genuinely owns its box; its `space_origin` already sits at the first brick's center.

`resample_alignment` and `centering` are one fact in two vocabularies — the pipeline's word for
where it aligned samples, and duckn's word for what a sample represents. Both are derived from a
single value when the store is built, so they cannot drift apart; if you ever find them
disagreeing, the store is wrong and `resample_alignment` is the one to trust.

### Is a structure cut off by the field of view?

Each segment carries duckn's `extent`, the inclusive bounding box of its voxels. Compare it
against `model_grid` (mapping through `envelope` if the array is a crop): a structure whose
extent reaches a face of the model grid is **truncated by the acquisition**, and no volume or
area computed from it means anything. This is not a rare edge case — on a chest CT the liver,
kidneys and gallbladder all reach the inferior face, and a structure that looks like it
"disappeared" between two resolutions is usually one that was never wholly in the scan.

---

## 6.1 Where did this come from, and what made it?

The root group carries duckn's `provenance` extension, which is the specified home for both
questions — so a duckn reader finds them where the spec says to look rather than in a field of
ours:

- `sources` — one entry per input, with `identifier`, `doi`, `url`, `description`, `created`.
- `processing` — the steps that produced this store, in order, each naming its `software` (name
  and version) and the `parameters` it ran with. Segmentation and store layout are separate
  steps because they fail independently: a reader doubting a store needs to know which to doubt.
- `attribution` — licence and citation.

### What this store deliberately does NOT carry

**The source's DICOM tags.** No patient or study UID, no frame of reference, no manufacturer,
kernel, kVp, slice thickness or pixel spacing. That is duckn's rule for a derived array
(`dicom` extension §10.3) and it is a correctness rule, not tidiness: *this array was not
acquired at 120 kVp with a soft-tissue reconstruction kernel — it was computed.* A segmentation
advertising a reconstruction kernel is describing a scan it is not. Inheritance across a
derivation has no defined semantics either: some attributes survive resampling, some are
invalidated, and an array derived from several sources could inherit contradictory ones.

So the store names its source and stops. `provenance.sources[].identifier` resolves in the
archive it names — a `crdc_series_uuid` for IDC, a `ds<number>/<path>` for OpenNeuro — and that
identifier is what the fetch tooling takes, so the input is re-obtainable from the store alone.

**This costs nothing for a DICOM writer.** Producing a conformant SEG from one of these is a
deliberate construction, not a metadata copy: it mints new instance and series identifiers,
marks the result as derived, references the source instances through DICOM's own mechanisms,
and carries forward Patient and Study — *from the source*, which it must have access to anyway,
not from a stale copy of a header kept here (`dicom` extension §10.4).

`nnseg.case_detail` therefore holds only what is not an inherited tag: the case name, and any
cross-reference to an independent artifact — for example an existing published segmentation of
the same source, which nobody could reconstruct from the source header and which a later
comparison needs.

---

## 6.2 The client renderer contract

Eight rules, each purchased with a visible artifact. Reference implementations:
`tools/ranked_render_selection.py` (selection reseed, shell, glass) and
`tools/ranked_render_composite.py` (two stores, two grids, one ray).

1. **No labelmap is ever an opacity mask.** Ranks are addresses: sample nearest, use for
   identity and sign. A membership mask multiplied into sub-voxel opacity renders as stipple
   no supersampling removes.
2. **Sign before you interpolate.** Build `s = member ? +d : -d` at voxel level, THEN
   trilinear. The stored magnitude is folded; interpolating it across a surface erases the
   crossing.
3. **For a selection, reseed - never mask.** Seed edges where selection membership flips,
   crossings from the baked distances (`t = d_a/(d_a+d_b)`), propagate with the Godunov
   update (min-plus is taxicab and shades as facets). ~1 s on a lungs-sized crop; removes
   every gate downstream.
4. **Opacity by analytic per-segment integration.** Point-sampling a feature thinner than a
   few steps makes each pixel's opacity depend on sample phase. The tent/ramp integral with
   `s` linear per segment is closed-form; phase drops out exactly.
5. **Normals from a separately smoothed copy.** Shading moves no geometry, so smooth its
   field freely (bounded by half the thinnest structure); outward normal `-grad s`;
   two-sided for shells.
6. **Identity from a dilated colour map, nearest.** The outside half of a shell must know
   which structure it wraps.
7. **Compose across stores in millimetres, per sample, in the world frame the metadata
   declares.** Never resample one store onto another's grid - that is what keeps native
   trees intact inside coarse glass.
8. **Style is functions of `s` and `n`.** Ramp width = antialiased edge; tent = shell;
   Fresnel on `n.v` = glass; specular composited through transmittance but not gated by it,
   normalized per surface. Set operations are field Booleans: union `max`, intersection
   `min`, difference `min(s_A, -s_B)` - exact in sign everywhere, approximate in magnitude
   near the other operand (reseed the composite boundary if magnitude matters). Margins
   never combine across softmaxes.

Marching step is bounded by anatomy, not by shell width: below half the thinnest structure
you must not drop. And everything here reads `ranks[0]` + `distance` only; `support` enters
for confidence-graded fill and exact-seam seeding, nothing else.

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

### The two `junction` arrays are sharded per slab, not per array

Every other array is one shard (see above). `junction` and `junction_pair` are sharded per
**64-slice slab** along the first axis instead. A whole-array shard can only be written whole,
which means materializing the dense volume; these arrays are almost entirely zero, and they
are computed and written slab by slab precisely so that nothing dense is ever held. A reader
sees no difference except a few more files: absent inner chunks are still absent from each
slab's index, and a whole all-zero slab is not stored at all.

### 7.1 `occupancy` — skipping bricks you do not need

`occupancy[c, bz, by, bx]` is the **maximum** over that brick of class `c`'s support-encoded
deficit. `brick` in the part metadata gives the brick edge in voxels.

```
class c wins somewhere in the brick   <->  occupancy[c] == support_max
c comes within tau of the winner      <->  gap(occupancy[c]) <= tau
```

So a reader after one structure, or one confidence threshold, can skip every brick that fails
the test without opening it. Measured: the median structure occupies **0.7 – 6 %** of bricks,
and the index costs **0.05 – 0.4 %** of the store.

Two properties are load-bearing:

- **Conservative.** A maximum can over-report presence but never miss it, so skipping a brick
  the index rejects is always safe. Verified across every store and structure: zero bricks
  containing a class went unflagged.
- **Independent of storage layout.** `brick` is a declared spatial factor, *not* the chunk or
  shard shape. Rechunking or resharding the data cannot invalidate it. If the brick happens to
  align with the chunk grid the skipping is maximally efficient; if not, the index is still
  correct, only coarser.

It is also a readable duckn array in its own right — a coarse occupancy map, with `space_origin`
shifted by half a brick and `space_direction` scaled by it. One caveat: where the shape is not a
multiple of `brick`, the last brick along that axis is partial, so its true centre is nearer than
the uniform grid declares. That is accepted deliberately; this is a conservative index, not a
measurement.

The array is optional. A store without it is complete; a reader without it just does more IO.

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
