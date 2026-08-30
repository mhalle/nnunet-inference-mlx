# Ranked probabilities — keeping what the labelmap throws away

**The format and algorithm behind `nnseg.ranked` and `segment(..., probabilities=...)`.**
A labelmap answers *which class won*. This keeps the second answer — *by how much* — for
about a third of a byte per voxel, computed from the logits at the one moment they exist.

Written 2026-08-29. Every number below was measured on real TotalSegmentator output
(chest.nii and CT_Abdo), not estimated.

## Why bother

`argmax` is a hard decision made per voxel and then discarded. A voxel labeled *rib* at
p = 0.95 and one that merely beat 26 rivals at p = 0.34 are indistinguishable afterwards.
Three consequences we hit in practice:

- **Boundaries are sub-voxel, and labels are not.** The logits cross smoothly; the labelmap
  snaps to the grid. Rendering or meshing from labels bakes in the staircase.
- **"Missing anatomy" questions become unanswerable.** Was a structure *absent from the
  output*, or *present but uncertain*? The labelmap cannot distinguish these; the margin can,
  and that distinction is what located the normalization bug (see
  `docs/` history and the volume-render session).
- **Overlap is destroyed.** Multi-part tasks composite by paint order (later wins) and region
  heads threshold-and-paint, so a voxel that is plausibly two things becomes one thing.

The full softmax answers all of this and costs 5.6 GB per case in fp32. This format answers
it in single-digit megabytes.

## What is stored

Per model, on that model's own grid, three arrays:

| array | shape | dtype | meaning |
|---|---|---|---|
| `ranks` | `(N, Z, Y, X)` | uint8 / uint16 | `class + 1` of the N best channels; **0 = not this class** |
| `support` | `(N-1, Z, Y, X)` | uint8 | how far each trails the winner: 255 = tied, **0 = at the clip** |
| `tail` | `(Z, Y, X)` | uint8 | probability mass beyond the top N; omitted when exhaustive |

`ranks[0]` is the argmax and is bit-exact — `ranks[0] - 1` *is* the labelmap, with no special
cases, because every voxel has a winner and plane 0 therefore never holds the sentinel.

Three properties of that layout are load-bearing and are the whole design.

### 1. Margins, not probabilities

Softmax has a per-voxel gauge freedom: adding a constant to every logit changes nothing. Only
*differences* between logits are identifiable, so differences are what we store.

That is not merely tidier. It buys three things:

- **Uniform quantization.** A gap in logits quantizes evenly; a probability does not, and
  would waste resolution near 0 and 1 where nothing interesting happens.
- **Correct interpolation.** The linear interpolant of a logit difference is right; the linear
  interpolant of a softmax output is not, and the discrepancy is worst exactly at boundaries.
  This is the same lemma the fused restore rests on. It holds for a difference taken against a
  reference *shared by all channels* — see `deficit` below, and the trap that the per-structure
  `margin` is a different quantity that must not be substituted here.
- **No softmax at all.** The gaps *are* logit differences, so `topk` on the logits yields
  ranks and gaps directly. No exponentials, no normalization, no `log` round trip. Only the
  tail needs a normalizer, and that is one `logsumexp`-shaped pass, not a K-channel volume.

> The reference implementation this format grew from took *probabilities* and immediately
> applied `log()` to recover logits. That round trip existed only because its CLI consumed
> saved `.npz` files. Inside a pipeline the logits are already in hand, and streaming from
> them reproduces the reference stores **byte-for-byte** with no intermediate files.

### 2. Zero means "nothing here", in every array

`ranks` stores `class + 1` because class 0 is legitimately *background* — a bare 0 could not
mean "absent" without the shift. `support` counts **up from the clip** rather than down from
the winner, so it too is 0 when the class is not there.

The payoff is that `fill_value = 0` is correct for all three arrays, so:

- an unwritten chunk decodes to "absent", which is what makes whole-block skipping sound;
- a reader that forgets the fill value, a `calloc`'d buffer, a zero-cleared GPU texture, or a
  sparse-array default all produce the **safe** answer.

Storing the deficit instead inverts this: 0 would mean *tied with the winner*, so the same
mistakes would produce the maximally wrong answer. This was a known trap in the original
design, handled there by setting the fill to `GAP_MAX`; inverting the encoding removes the
trap instead of documenting it.

It is also **smaller**, which was not the motivation but is a clean confirmation. With
`class + 1` every value is a small integer, so the high byte of a uint16 is always zero and
byte-shuffling compressors get that plane for free. With a 65535 sentinel the high byte is a
noisy bilevel mask that has to be encoded:

| ranks encoding | size (5-layer store) |
|---|---|
| high sentinel (65535) | 1.48 MB |
| zero sentinel (`class + 1`) | **0.93 MB** |

### 3. The clip bounds each class's spatial support

Beyond `clip` logits behind the winner, a class is indistinguishable from absent (8 logits is
a probability ratio of ~3000:1). Past it, the rank plane is masked to the sentinel and the
support saturates at 0, so confident regions become *uniform* — which is what makes depth
nearly free and lets a store skip empty blocks entirely.

Measured on the five-part `total`, **95.0 % to 99.7 %** of all gap entries sit at the clip.

`clip` trades range against precision and is a per-store parameter, not a constant. A smaller
clip quantizes finer **and** stores smaller (more entries saturate), at the cost of dynamic
range for low-ranked classes and the tail. A rendering tier and an analysis tier legitimately
want different points on that curve.

## Depth is nearly free

Because planes past the first few sit at the clip almost everywhere and mask away, raising
the depth costs a few percent rather than a proportional amount:

| depth | chest `total_fast` (K=118) | 5-part `total` |
|---|---|---|
| 3 | 1.24 MB | 3.51 MB |
| 4 | 1.30 MB | — |
| 6 | 1.31 MB | 3.74 MB |

Before the sentinel fix the same 3 → 6 step cost **2.5×** (chest) and **3.6×** (five-part).
After it, 6 % and 7 %. Depth was only ever expensive because of noise the sentinel removes.

**Therefore the default is depth 6, not 3.** A depth-6 8-bit store is *smaller* than a
depth-3 store in the original encoding (3.74 vs 19.41 MB) while carrying twice the depth.

## The tail, and when it stops earning its place

`tail` records the mass discarded by truncating at N, as a fraction, so probabilities can be
renormalized exactly: `Z = Z_top / (1 - tail)`. It is the subtlest part of the format — and
depth makes it redundant. At depth 6:

| layer | max_tail |
|---|---|
| organs | 0.015 |
| vertebrae | 0.009 |
| muscles | 0.004 |
| cardiac | 0.000 |
| ribs | 0.000 |
| chest `total_fast` (K=118) | 0.002 |

`max_tail` is a **worst-single-voxel** statistic, not a typical one: at depth 3 only 0.09 % of
voxels have any nonzero tail at all. At depth 6, dropping the tail entirely would cost at most
1.5 % — less than the 8-bit quantization error already accepted below. So a deep store can be
just **ranks + support**, two arrays in which zero means "nothing here".

`max_tail` is recorded in metadata precisely so a consumer can check whether the depth it was
given is deep enough for what it wants to do, rather than trusting a default.

## Quantization: what 8 bits costs

`support` spends 255 levels over the clip, a step of 31.4 mlogit (against 0.122 mlogit at 16
bits). Measured across every layer of both datasets:

- **probability error** — max |Δp| 0.0039–0.0046, mean ~2 × 10⁻⁵. Uniform across K = 19 and
  K = 118 alike; it is set by the step, not the class count or depth.
- **geometry** — near boundaries the gap gradient runs ~0.9–1.0 logit/mm, so the quantization
  displaces a surface by **15–18 µm**: about 1 % of a 1.5 mm voxel, and three orders of
  magnitude below the ~9 % volumetric difference between restore methods.

`ranks` uses uint8 when `K + 1` fits and uint16 otherwise. The dtype is **chosen from K and
declared in metadata**, never assumed: `class + 1` breaks silently above 254 classes, which no
TotalSegmentator model reaches (the largest is 118) but a custom or MONAI ecosystem model
could. Narrowing ranks saves almost nothing on disk — byte shuffle had already collected that
— but halves RAM and texture-upload cost, which is why it is still worth doing.

## The algorithm

Encoding, per z-slab, on whatever device the logits are already on:

```
top, idx = topk(logits, N, dim=0)        # descending; top[0] is the winner
gaps      = top[0:1] - top               # >= 0, and gaps[0] is exactly 0
support   = round(clamp(1 - gaps[1:]/clip, 0, 1) * 255)
ranks     = idx + 1;  ranks[1:][gaps[1:] >= clip] = 0
tail      = clamp((Z_full - Z_top) / Z_full, 0, 1)      # only place all K are read
            where Z_full = sum_k exp(l_k - l_1), Z_top = sum_j exp(-gaps_j)
```

Slabbing bounds peak memory to one slab promoted to fp32, not the whole volume. Slab size is
performance-neutral in practice (32/64/128 measured within noise on an L40S), so it is a
memory knob, not a tuning knob.

Decoding is the inverse, and it yields **two different fields from the same bytes**. They
agree at voxel centers and diverge under interpolation, which is exactly the kind of
difference that hides in testing — so which one to use is not a matter of taste.

```
margin   m_c = +gap      where c wins      (its lead over the runner-up)
                -gap_j   where c ranks j   (its deficit behind the winner)
                -clip    where c is absent

deficit  d_c =  0        where c wins      ( = m_c with the lead removed )
                -gap_j   where c ranks j
                -clip    where c is absent
```

**`deficit` is the field a restore must interpolate.** (Measured in
[ranked-reconstruction.md](ranked-reconstruction.md) §1.) It is `l_c - max_j l_j`, which differs
from the logits by a per-voxel constant *shared by every channel* — a gauge transformation —
so interpolating it and taking the argmax is exactly interpolating the logits and taking the
argmax. `nnseg.ranked.deficit(code, channel)`.

**`margin` is the field a renderer or mesher wants.** It is `l_c - max_{j != c} l_j`: positive
inside by however much `c` leads, zero *on* `c`'s surface, negative outside. It has an
interior gradient to shade with and a non-degenerate zero level set, which `deficit` (flat
zero throughout the interior) does not. `nnseg.ranked.margin(code, channel)`.

The trap is that `margin` is **not** gauge-equivalent: it adds the lead to one channel only,
which is channel-dependent. At a voxel center the winner still wins, so an argmax over
margins looks correct; once a trilinear stencil mixes voxels with *different* winners, it is
not. Measured on a real K=118 case, restoring through `margin` agrees with the logits on
99.43 % of sub-voxel samples and only **84 %** of near-tie samples, against 99.98 % / 99.4 %
through `deficit`. Nearest-neighbor restore cannot see the difference at all, because it
never mixes voxels — which is precisely how such a bug survives.

`nnseg.ranked.probabilities(code)` is the head-specific decode: `p_j = exp(-g_j) / Z` with
`Z = Z_top / (1 - tail)`, exact when exhaustive. Where a rank holds the sentinel the class is
absent, its id is `-1`, **and its probability is reported as 0** — both halves matter, because
`-1` under `np.take_along_axis` indexes the *last* class and would silently attribute the
voxel's mass to whatever sorts last. Mask on `ids >= 0` before using ids as indices.

## Region (sigmoid) heads

Overlapping regions — BraTS-style nested tumor labels, anything `to_labels(mode="regions")`
handles — are independent Bernoullis. Nothing sums to one, several channels can be present at
once, and **ranking buys nothing**: top-N compression works precisely *because* softmax
classes are mutually exclusive.

But the useful part survives, and simplifies. For a sigmoid head the logit **is** already a
margin, referenced to the decision threshold rather than to a winner, and the gauge argument
does not apply (the reference is pinned at 0, where p = 0.5). So `encode_regions` stores one
plane per region, no ranks and no tail:

```
m_c = l_c - logit(threshold)
q   = round(clamp(m_c/clip, -1, 1) * 127) + 128     # 1..255; 128 is exactly the boundary
```

Two deliberate details. The threshold is **folded in at encode time**, so zero is the decision
boundary in every file whatever produced it — a renderer or mesher then needs no head type at
all. And the boundary lands *on* a quantization level rather than between two, which keeps 0
reserved as the fill sentinel, so an unwritten block still reads as "absent" here too.

Cost scales with the number of regions rather than sub-linearly in K, but spatial sparsity
still applies per region: exterior blocks go unwritten, interiors are a saturated constant
that compresses to nearly nothing, and only the boundary shell carries data.

**Margins from different heads must not be compared**, exactly as margins from different
layers must not be. A softmax margin says "more likely this class than any other"; a sigmoid
margin says "more likely present than absent, independently of everything else".

## One layer per model, one file per layer

Each TotalSegmentator part is its own network with its own softmax over its own class set, and
parts overlap spatially. **Logits are comparable only within a model.** The merged labelmap
hides this by compositing in paint order; the format must not.

That non-comparability is also the argument for keeping parts in *separate files* rather than
separate groups of one store:

- the grids may legitimately differ (per-part body envelopes), and a combined store would have
  to refuse or paper over that;
- it matches the pipeline, where one part's logits exist and are freed before the next;
- it matches the serve tier, where artifacts are single files fetched independently — a client
  wanting only ribs fetches only ribs, and one part failing does not invalidate the rest.

What co-location gave for free was *discovery*, which a manifest replaces (the result JSON's
`links` map already has the slot). Each file is self-describing: `meta` carries the class
count, depth, clip, dtypes, `max_tail`, the model spacing, the envelope offset and full model
grid, the channel → global-label map, the orientation and the resampling convention — and
`frame`, the **spatial extent**, which is what the next section is about.

## Re-restoring: a new mask on an arbitrary grid, without the network

Argmax after interpolation depends only on logit *differences*, and differences are exactly
what is stored. So a stored code is not a picture of one run: rebuild `Frame.from_meta`,
expand the per-class margin fields, and call the same `to_labels` the pipeline calls. The
restore becomes a decision that can be *re-made* — which matters because the restore is not a
minor knob (linear vs nearest moves rib volume ~9 %) while inference is the expensive,
irreversible half.

`Frame.to_meta()` / `Frame.from_meta()` carry the extent: the source grid, the crop-to-nonzero
sub-grid, the model shape and spacing, the convention, the original orientation, and the world
placement (origin plus direction cosines). Dropping any of it produces a frame that still
looks valid and silently shifts every restored label — the crop offset especially.

Measured on CT_Abdo, re-restored from the stored parts alone and compared against fresh
`segment()` runs at the same target. The five-part `total` onto a **1 mm isotropic** grid:

| restore | agreement |
|---|---|
| nearest | 99.992 % |
| linear, decoded through `margin` | 99.650 % |
| linear, decoded through `deficit` | **99.869 %** |

The gap between the last two is the decode field, not the format. Decomposing it on sampled
sub-voxel points against ground truth (interpolating the raw logits) separates every
candidate source:

| what varies | agreement (all points) | (near-tie points) |
|---|---|---|
| exact margins, all 118 classes, no clip — sanity | 100.000 % | — |
| `margin` field, depth 6, clip 8, uint8 | 99.431 % | 84.4 % |
| `deficit` field, depth 6, clip 8, uint8 | 99.736 % | 92.3 % |
| `deficit` field, depth 6, clip 20, uint8 | **99.985 %** | **99.4 %** |
| `deficit` field, depth 12, clip 20, uint8 | 99.987 % | 99.5 % |

Three things fall out. **Quantization is not the limit** — exact float gaps score 99.738 %
against uint8's 99.736 %, a difference of 0.002 %. **Depth is not the limit** — 6, 12, 24 and
all 118 are indistinguishable. What remains is the **decode field** (the largest single term)
and then the **clip**, whose floor at `-clip` lets a far-behind class compete once a stencil
reaches a voxel where it was truncated. Widening the clip to 20 removes most of what is left.

That also corrects an earlier, wrong reading of the same numbers: nearest looked near-exact
and linear looked deficient, which invited blaming the clip. Nearest was simply immune to a
bug in the decode.

Companding — spending the 255 levels non-uniformly to get fine resolution near a tie — was
tried and is **worse**: 99.35 % at gamma 2 against 99.65 % linear, despite a 100x finer step
near zero. Linear interpolation runs between a voxel where a class leads and one where it
trails, so the accuracy of *large* margins matters as much as small ones, and companding buys
the latter by destroying the former. Uniform support with a wider clip is the right shape.

Beyond a different grid, the same machinery re-decides the interpolation (so one store yields
both the TS-parity nearest labelmap and the logit-grade linear one), the label mapping, and
**confidence gating** — floor the background channel at *g* logits and only sufficiently
confident voxels keep their label, which a labelmap cannot express at all. On this case a
3-logit gate drops foreground from 2.44 M to 2.10 M voxels while keeping all 87 structures.

One limit: compositing across parts still needs a policy, because margins are not comparable
between models. Re-restoring replays that decision; it does not improve on it.

## What it costs — measured

Sizes are of the compressed store (blosc/zstd-5, 64³ chunks, empty chunks skipped).

**CT_Abdo, `total`, 1.5 mm, five models, 122 channels, 11.46 M voxels** (fp32 would be 5.59 GB):

| encoding | depth 3 | depth 6 |
|---|---|---|
| original (raw ranks, deficit gaps, uint16) | 19.41 MB | 70.13 MB |
| + ranks masked at the clip | 5.43 MB | 5.87 MB |
| + zero sentinel | 4.90 MB | 5.26 MB |
| + 8-bit | **3.51 MB** | **3.74 MB** |

**chest.nii, `total`, 1.5 mm, 32.0 M voxels**: 181.50 MB → **7.00 MB** (0.219 B/voxel, 2230×).

**chest.nii, `total_fast`, 3 mm, K = 118, 3.98 M voxels**: 7.42 MB → **1.31 MB** at depth 6.

Two things worth noticing. Bytes per voxel converge to **0.22–0.33 across every case**, despite
K ranging 19–118, depth 3–6, and 1.5 vs 3 mm grids — and the figure *falls* as volume grows
(chest 0.219 vs CT_Abdo 0.326), because after masking, the only voxels that cost anything are a
shell around boundaries. Size tracks **boundary area**, not voxel count or class count. And the
correctly-normalized store is *smaller* than the buggy one it replaced (19.4 vs 21.3 MB at
depth 3) — correct normalization makes the models more confident, and confidence is what this
compresses.

For scale, on CT_Abdo: input CT 7.75 MB, labelmap 0.28 MB, this format 3.74 MB, a dense fp16 +
zstd baseline 178 MB, fp32 5.59 GB. On the 32 M-voxel chest case the complete probabilistic
record of five models is **less than the input scan**.

### Time

Encoding rides on inference it cannot avoid:

- **M2 / MPS**: 9.6 s across five parts, **5.9 % of network time**.
- **L40S**: 0.20 s per 1.5 mm part (52–57 Mvox/s), i.e. **~2.8 % of a warm 36 s `total`**.

CUDA output is **bit-identical to CPU** for ranks, support, tail and regions. The device→host
copy per slab is well overlapped — slab size does not move the number.

## Where it lives

`src/nnseg/ranked.py` is **kernel layer**: torch and numpy only, no knowledge of tasks, plans,
weights or files, enforced by `tests/test_nnseg_layering.py`. That is what let the whole module
be shipped to a bare Modal image as a single file for the CUDA check.

The pipeline hook is in `segment()`, between the network and the restore — the only moment the
logits exist, so it costs one pass and no recomputation:

```python
from nnseg import ranked
seg = nnseg.segment(image, "total",
                    probabilities=ranked.RankedSpec(sink=my_writer, depth=6))
```

`RankedSpec` takes a **sink** rather than returning codes, because a multi-model task holds one
part's logits at a time and the uncompressed codes are far larger than the stored form — each
part is handed over as produced, written, and dropped. It is off by default: labels are a
fraction of the size and most callers want only those.

The code is produced on the model's own **envelope-cropped grid**, which is where the
distribution lives. Restoring it to the output grid first would inflate it and bake in one
interpolation choice; the metadata carries what is needed to place that grid in the world.

## Deliberately not decided: storage and serving

The encoder is settled; the container is not. Zarr v3 with 64³ chunks and `fill_value = 0` is
the natural fit — chunk skipping is exactly the sparsity this format creates, and 64³ is a GPU
brick — but two things are open:

- **`zarr` is in no nnseg extra**, and the Modal nnU-Net worker syncs `["torch", "serve",
  "idc", "cuda"]`. Adding it is an image change.
- **Artifact shape.** Every serve artifact today is a single file returned by `FileResponse`;
  a zarr store is a directory (893 files for a five-layer store). Either ship `.zarr.zip` —
  one file, fits the existing cache and `_place` mechanism, still range-readable by a client
  that reads the zip index since chunks are already compressed — or add a route that serves
  chunk paths, which gives a renderer true lazy per-chunk fetch. The zip first; the route when
  a client needs laziness.

A `probabilities` artifact should be **opt-in per request**, like preview and statistics.

## Reconstruction

How a reader gets a segmentation back out — the deficit/margin distinction, progressive
refinement, the adaptive floor, resampling order, and the measured cost of every derived
product — is in [ranked-reconstruction.md](ranked-reconstruction.md). It also corrects
several claims below, which are left in place with pointers rather than silently amended.

## Consumer contract

Two reads, and the choice follows from what you are producing, not from taste:

- **`deficit(code, channel)` — when the answer is a label map.** Gauge-equivalent to the
  logits, so interpolate then argmax.
- **`margin(code, channel)` — when the answer is a surface.** Signed, zero on the boundary,
  with an interior gradient. Both unify softmax and region heads behind one shape, so a
  renderer or mesher never learns which head produced a file.

Substituting one for the other is silent: they agree at voxel centers and diverge only under
interpolation, and nearest-neighbor restore never notices at all.

Two consumers already want the surface field:

- **Volume rendering.** Decode per-structure margins to uint8 textures and ray-march them. The
  ramp must be in **millimeters** (`m / |grad m|`), not logits: a fixed logit ramp is sub-voxel
  thin wherever the model is confident, which degenerates to a hard isosurface and shows
  trilinear facets.
- **Meshing.** The zero level set is the same surface the logit-mesh SurfaceNets pipeline
  builds, so a cached store can be meshed without re-running the GPU.

A third is nearly free: `max_tail`, the per-voxel tail, and the winner's margin are a **quality
channel**. "Is this structure absent, or present but uncertain?" is answerable from the store
alone, without re-running inference — which is how the multi-model normalization defect was
found.
