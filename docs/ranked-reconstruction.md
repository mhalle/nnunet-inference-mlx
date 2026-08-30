# Reconstruction from a ranked store — the algorithm, and what it costs

**How to get a segmentation back out of `nnseg.ranked`, at any grid.** The companion to
[ranked-probabilities.md](ranked-probabilities.md), which covers what is stored and why.
This document covers what a *reader* does with it.

Written 2026-08-30. Every number was measured on real TotalSegmentator output — CT_Abdo and
chest.nii, the five 1.5 mm `total` parts (K = 19–27) and the 3 mm `total_fast` model
(K = 118). Where a claim is an extrapolation it says so.

Several conclusions here **correct** earlier beliefs, including claims in the companion
document and in shipped code. Those are marked ⚠ and explained rather than quietly amended,
because each one passed casual inspection and would be re-derived the same wrong way.

---

## 1. The two fields, and the trap between them

Decoding yields **two different per-class fields from the same bytes**. They agree at voxel
centers and diverge under interpolation, which is exactly the kind of difference that hides
in testing.

```
margin   m_c = l_c - max_{j != c} l_j     winner gets +its lead
deficit  d_c = l_c - max_j     l_j        winner gets 0
```

They differ in **exactly one place**: the winner's own channel. Every other class has the
same value in both.

**`deficit` is the field a restore must interpolate.** It is the logits shifted by a
per-voxel constant *shared by every channel* — a gauge transformation — so interpolating it
and taking the argmax is exactly interpolating the logits and taking the argmax.

**`margin` is the field a renderer or mesher wants.** Its zero level set is that structure's
surface, it is positive inside by however much the class leads, and it has an interior
gradient to shade with. `deficit` is flat zero throughout the interior and has no usable
isosurface.

### ⚠ Why substituting one for the other is silent

`margin = deficit + lead·1[c is winner]`, and the lead varies per voxel and is awarded to a
*different class* at each voxel. Subtracting the max is safe because every channel gets the
same subtraction and it cancels in any comparison. Adding the lead is not, because only one
channel gets it.

At a voxel center the winner still wins either way, so an argmax over margins looks correct.
Once a trilinear stencil mixes voxels with different winners, it is not. Measured on K = 118:

| field interpolated | all samples | near-tie samples |
|---|---|---|
| `margin` | 99.431 % | **84.4 %** |
| `deficit` | 99.736 % | **92.3 %** |
| `deficit`, clip 20 | 99.985 % | 99.4 % |

A real case, two adjacent voxels and their midpoint:

| class | logit A | logit B | truth (mid) | deficit (mid) | margin (mid) |
|---|---|---|---|---|---|
| 0 | 30.516 | 31.344 | **30.930** | **−2.320** | −2.320 |
| 10 | 34.906 | 25.281 | 30.094 | −3.156 | **−1.227** |
| 51 | 26.719 | 31.594 | 29.156 | −4.094 | −3.969 |

Class 10 wins voxel A by 3.859, class 51 wins B by 0.250, class 0 wins neither but is
runner-up at both and truly wins the midpoint. `deficit` reproduces that. `margin` gives
class 10 half its lead as a bonus (+1.929), and it takes the midpoint. Class 0, having never
won anywhere, collects nothing.

**Nearest-neighbor restore cannot see this difference at all**, because it never mixes
voxels — which is precisely how such a bug survives.

---

## 2. Reconstruction: scatter, then interpolate

The stored arrays are **sparse and rank-indexed**, and plane *j* does not mean a fixed thing:
`support[0]` is "the runner-up's deficit **here**", and the runner-up is liver at one voxel
and spleen at the next. **Never interpolate a stored plane directly.** Ranks are addresses,
not values.

```
sparse (ranks, support)  --scatter-->  dense d_c per class  --interpolate-->  argmax --> label
```

```python
def reconstruct(ranks, support, clip, K, n_rank, n_sup):
    # floor: what a class not named here is worth. See §4.
    floor = -gap(support[n_sup-1]) if n_sup >= n_rank else -clip
    out = broadcast(floor, (K, Z, Y, X)).copy()
    for j in range(n_rank-1, 0, -1):              # losers first
        put_along_axis(out, ranks[j], -gap(support[j-1]))
    put_along_axis(out, ranks[0], 0.0)            # winner last: deficit 0
    return out
```

`gap(s) = (1 - s/255) * clip`. Use `put_along_axis`, **not** `putmask` — putmask fills
cyclically from a flattened array and does not align positionally, which silently scrambles
the stack.

### Memory

The dense stack lives at the **model** grid; `to_labels` samples from it into the output grid
writing one `uint8` per output voxel. Upsampling does **not** multiply the K-channel cost.
Per-part files bound it further: a five-part `total` never holds 118 channels, only ≤ 27.

### The bounded-candidate optimization

At any output point, only classes appearing in the eight surrounding voxels' rank planes can
win — a class absent from all eight sits at the floor at every corner and interpolates to the
floor, while every corner has some class at deficit 0. Measured: the union averages **7.8
classes** (bound 8 × depth = 48, p99 = 13), and **85.7 % of stencils have all eight corners
sharing one winner**, which is an *exact* early-out requiring no arithmetic — the shared
winner is 0 at every corner, everything else ≤ 0. Combined arithmetic ratio against the dense
path: **~105×**.

⚠ **Do not narrow the candidate set to the corners' winners.** A class that wins no corner
can still win the interpolated point — measured at 0.045 % of stencils, and it is exactly the
worked example in §1.

---

## 3. Progressive refinement: planes come in pairs

Planes arrive in stored order — `ranks[0]`, `support[0]`, `ranks[1]`, `support[1]`, … — and a
reader can stop early. Measured at near-tie sample points (the only places a decision can
flip), across **ten segmentations, two cases, six models**:

| segmentation | K | 1 pl | 2 pl | 3 pl | 4 pl | 5 pl | 6 pl |
|---|---|---|---|---|---|---|---|
| CT_Abdo / organs | 25 | 68.45 | 94.74 | 94.74 | 99.96 | 99.96 | 100.00 |
| CT_Abdo / vertebrae | 27 | 66.29 | 97.71 | 97.71 | 100.00 | 100.00 | 100.00 |
| CT_Abdo / cardiac | 19 | 70.86 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| CT_Abdo / muscles | 24 | 71.97 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| CT_Abdo / ribs | 27 | 71.93 | 95.86 | 95.86 | 100.00 | 100.00 | 100.00 |
| CT_Abdo / total_fast | **118** | 63.61 | 84.47 | 84.47 | 98.48 | 98.48 | 99.92 |
| chest.nii / organs | 25 | 70.31 | 98.36 | 98.36 | 100.00 | 100.00 | 100.00 |
| chest.nii / vertebrae | 27 | 65.07 | 97.26 | 97.26 | 100.00 | 100.00 | 100.00 |
| chest.nii / cardiac | 19 | 68.06 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| chest.nii / muscles | 24 | 62.32 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

**A rank plane alone changes nothing** — columns 2→3 and 4→5 are identical to two decimals in
every row. **A support plane alone is worse than useless** — dropping `ranks[1]` while keeping
`support[1]` scores 69.8 % against 94.7 % for two planes, because the true runner-up gets
pushed to the deeper floor along with everyone else.

**So planes must be added in pairs**, and a residual loop must terminate on pairs.

⚠ I first read this table as "support planes carry everything, rank planes are free" and was
wrong. `ranks[1]` looks free only because at that level the floor is still `−gap₁`, so placing
the runner-up there is a no-op. Both are load-bearing; neither works alone.

### Why two planes work at all

At a two-class boundary, the runner-up's deficit **is** the margin. For a pair A|B with
winners *p* and *q*:

- reconstruction says *p* takes the midpoint iff `m_A > m_B`
- truth says *p* takes the midpoint iff `gap(p at B) < gap(q at A)`

If *q* is runner-up at A and *p* at B, then `m_A = gap(q at A)` and `m_B = gap(p at B)` — the
two conditions are **the same inequality**. Exact, with no third-class information.

The premise holds where it matters: **near a boundary (margin < 2) the runner-up is one of the
six neighbors' winners 85.5 % of the time**, falling to 2.7 % over the whole volume — in the
interior the runner-up is arbitrary, but there the winner leads by 8 logits and nothing can
flip. The label plane names the competitor; the margin gives its distance.

The failure mode is the mirror image: a third class within half a logit of the runner-up
breaks the identity, and that happens on 93.7 % of voxels overall but only **1.7 % near a
boundary**. Junctions, not surfaces.

---

## 4. The adaptive floor — one plane, and the best one

The format stores N ranks and **N−1** supports, so classes beyond the deepest stored rank fall
back on the fixed `clip`. Storing **one more support plane** — the deficit of rank N+1, used
as a per-voxel floor for "everything else" — is the cheapest plane in the format. K = 118,
against interpolated raw logits:

| configuration | planes | all | near-tie |
|---|---|---|---|
| depth 2, clip floor | 3 | 99.612 % | 86.716 % |
| depth 2, **adaptive floor** | 4 | 99.767 % | **92.320 %** |
| depth 3, clip floor | 5 | 99.779 % | 92.513 % |
| depth 3, **adaptive floor** | **6** | **99.784 %** | **92.672 %** |
| depth 4, adaptive floor | 8 | 99.785 % | 92.706 % |
| depth 6 (as stored today) | 11 | 99.785 % | 92.706 % |

One extra plane buys more than a whole rank/support pair (86.7 → 92.3 for one plane, versus
86.7 → 92.5 for two). And **depth 3 with an adaptive floor — six planes — matches depth 6's
eleven planes to 0.001 points.**

It also subsumes the clip-tuning problem. A fixed clip is a global constant standing in for a
per-voxel quantity, which is why clip 8 → 16 mattered so much (see §10). The adaptive floor
stores the actual value instead.

**Recommendation: store N ranks and N supports.**

---

## 5. Dynamic depth and the residual loop

Convergence varies from **2 planes (cardiac, muscles) to 6 (K = 118)**, so a fixed depth is
wrong in both directions. Depth is already recorded per part in metadata, so variable depth
costs nothing structurally.

```
d = 1
while True:
    encode at depth d and d+1
    sample sub-voxel points BIASED TO NEAR-TIES (|top1 - top2| < 1 logit)
    if interpolated argmax agrees within tolerance: stop at d
    d += 1
```

Three requirements, each learned the hard way:

- **Decision-based, not value-based.** Measure whether the argmax changes, not whether values
  do. `max_tail` is a proxy and a poor one.

  ⚠ That is the criterion for a **labelmap** consumer, and it converges early for everyone
  else. §8.4 measures identical group masks at depth 3 and depth 6 while the liver's p95
  margin error moves 0.0149 → 0.2969 — so this loop would stop at 3 and hand a renderer a
  field 20× worse in the quantity it samples. A consumer that reads the field rather than the
  decision needs the value-based test rejected just above. Same loop, different stop.
- **Sample near boundaries.** Uniform sampling saturates at 99.9 % and stops discriminating;
  every distinction in this document is visible only in the near-tie column.
- **Terminate on pairs**, per §3.

---

## 6. Resampling

### Order of operations for multi-part tasks

**Resample each part in its own frame, then composite.** Compositing first interpolates
between margins measured against *different models' peer sets*.

| approach | all | foreground |
|---|---|---|
| composite → resample | 99.506 % | 98.692 % |
| **resample per part → composite** | **99.924 %** | **99.887 %** |

Nearly **10× less foreground error** from the same bytes. Compositing is a lossy projection:
do it last.

### The error is scale-invariant

Restoring to the input grid is an **upsample** in nearly every real case:

| case | model | input | factor |
|---|---|---|---|
| chest.nii | `total_fast` 3 mm | 1.0 × 0.651 mm | 3.0 / 4.61× (**64× by volume**) |
| chest.nii | `total` 1.5 mm | 1.0 × 0.651 mm | 1.5 / 2.3× |
| CT_Abdo | `total_fast` 3 mm | 1.49 mm | 2.01× |
| CT_Abdo | `total` 1.5 mm | 1.49 mm | 1.00× |

And the error does not grow with the factor — the boundary shell stays one output voxel thick
however finely you sample:

| factor | 2-plane | full depth |
|---|---|---|
| 2.0× | 99.048 % / Dice 0.9899 | 99.704 % / Dice 0.9982 |
| 1.5–2.3× | 99.064 % / Dice 0.9906 | 99.706 % / Dice 0.9984 |
| 3.0–4.61× | 99.064 % / Dice 0.9906 | 99.722 % / Dice 0.9984 |

**The native grid is exact** (100.000 %) for both tiers, because no interpolation occurs and
`ranks[0]` *is* the argmax.

### Interpolate the margin, never a derived form

`sigmoid(interp(m)) != interp(sigmoid(m))`. Compute masks and probabilities **after**
resampling:

| what is interpolated | all | at the boundary |
|---|---|---|
| the union **margin**, threshold 0 | 99.903 % | **89.6 %** |
| the union **posterior**, threshold 0.5 | 99.730 % | 69.6 % |
| the binary **mask**, threshold 0.5 | 99.553 % | 63.8 % |

Probability saturates, so interpolating it displaces the surface: margins of +6.9 → −0.1 put
the true crossing 98.6 % of the way across, while interpolated probability puts it at 95.2 %.
It is exact only in the symmetric case, which is the case that flatters the method.

### Repeated interpolation

Staged linear upsampling is **exact when each stage's grid contains the previous samples**
(dyadic refinement: max difference 7.6 × 10⁻⁶, labels identical). Through a non-node-preserving
intermediate it smooths — max deviation 7.9 logits, 99.73 % label agreement. The no-overshoot
property survives regardless (each stage is a convex combination, so the composition is one),
but the effective support widens, so the ≤ 8 × depth candidate bound applies only to a single
interpolation.

---

## 7. Derived products and what they cost

Measured on the five-part `total`, 11.5 M voxels, output at 1 mm isotropic.

| operation | reads | time |
|---|---|---|
| union mask, **native grid** | rank plane 0 only (8 % of bytes) | **62 ms** |
| union posterior (6 unions, 5 layers) | all planes + a 256-entry `exp` LUT | ~440 ms floor |
| single-structure mask, resampled | 1 channel | 0.23 s, 46 MB |
| union mask, resampled | 2 channels | (dominated by decode) |
| full label map, resampled | K channels | 6.79 s, 1146 MB |

**Single structure**: `margin(code, c)` → `to_labels(mode="regions", threshold=0)`. **30×
faster and 25× less memory** than restoring the whole label map and selecting. Agreement
99.9984 %, and the single-channel result is a strict **subset** — interpolating `m_c`
underestimates the true margin at junctions because `interp(max) ≥ max(interp)`, so the mask
is conservatively eroded by ~0.5 %.

**Union, same layer**: build `d_S = max over the group` and `d_notS = max over the rest`, then
argmax those two channels. **Exact at voxel centers**, because deficits within one model share
a per-voxel reference, so max-ing over a group is a legal operation. Under interpolation it
picks up the mirror-image convexity error (the union runs marginally *large*): 124 voxels of
38.6 M, 99.9997 %.

**Union, cross-layer**: `OR` of per-layer masks — each half is exact in its own layer, and the
margins never have to be compared. Contested voxels (claimed by 2+ parts) were **607 of 2.54 M
claimed**, 0.02 %.

**Posteriors**: within a layer, `P(union) = Σ P(members)` — exact, since the classes are
mutually exclusive. Because `exp(-gap)` depends only on a uint8 it is a **256-entry lookup
table, not a transcendental**, so the whole computation is table lookups and adds; all groups
accumulate in one pass.

**Cross-layer posteriors for disjoint structures ADD.** `P(A ∪ B) = P(A) + P(B)` is an axiom
for disjoint events, requiring no independence assumption. Verified on three disjoint pairs:
the sum never exceeded 1 (max 0.9983, zero violating voxels). **A sum above 1 is a
self-check** — it means both models claim the voxel, so the disjointness premise failed there.
`1 − ∏(1−p)` is the *independence* rule and is wrong here; `max` is worse.

⚠ Use the posterior (`probabilities()`) for this, not `sigmoid(margin)` — the latter is a
pairwise "beats its best rival" quantity, not a probability over the class set, and does not
add.

---

## 8. Consuming: meshers and renderers

The store is not a label map that a consumer re-derives; it is a **continuous field a
consumer samples**. Two applications exercise that differently.

### 8.1 SurfaceNets — sub-voxel placement

For every cell edge where the margin changes sign, solve `t = m_A / (m_A - m_B)` and compare
against the crossing computed from raw logits. Error in voxels (3 mm each):

| structure | 2-plane median / p95 | full depth median / p95 |
|---|---|---|
| liver | 0.0006 / **0.214** | 0.0003 / **0.0052** |
| lung_upper_lobe_left | 0.0011 / 0.336 | 0.0007 / 0.0317 |
| lung_lower_lobe_left | 0.0008 / 0.270 | 0.0006 / 0.0253 |
| autochthon_right | 0.0000 / 0.067 | 0.0000 / 0.0037 |

**Two planes place the typical vertex exactly** — a median of ~0.001 voxel is ~3 um. That is
the §3 identity at work: at a two-class boundary the reconstructed margin for `c` is its true
margin on *both* sides, so the crossing is right.

**But the tail is 0.21-0.34 voxels (0.6-1.0 mm)** against 0.005-0.04 at full depth. Those are
the junction vertices, and a mesh shows them as local bumps or a ragged seam exactly where two
organs abut. **Mesh quality is a tail property, so mesh from full depth.**

### 8.2 Volume rendering — sample level 0, bound with max-pooling

Sample `margin` trilinearly at ray positions: a signed field whose zero crossing is the
surface, with a gradient for shading. Everything in §6 applies — margin for surfaces, ramp in
**millimeters** (`m / |grad m|`) not logits, never interpolate a stored plane.

Space skipping needs a **conservative** proxy: one that never reports empty where something
is. The reduction matters, and the intuitive choice is wrong.

| structure | level | avg-pool: blocks MISSED | max-pool: missed | max-pool bloat |
|---|---|---|---|---|
| liver | 1 | **808** | 0 | 0.0 % |
| lung_upper_lobe_left | 1 | **631** | 0 | 0.0 % |
| gallbladder | 1 | **66** | 0 | 0.0 % |

⚠ **Average-pooling erodes, which is exactly hiding.** A ray using it skips regions that
contain the structure and geometry is lost silently. Earlier drafts of this reasoning claimed
coarse levels "err by eroding, so they cannot hide anything" — that is backwards.

**Max-pooling is exactly conservative** by construction: `max(m) > 0` in a block iff some
voxel in it is inside. Zero missed, zero bloat, at every level.

So the bound needs no coarse *values*, only "could this block contain `c`" — **one bit per
structure per block**, which is the per-chunk presence index of §11 arriving from another
direction. Consequently **a value pyramid is not needed at all**: its only justification was
LOD shading, and §8.3 says do not shade from coarse levels.

### 8.3 If you build a value pyramid anyway

Decimating the margin field versus **re-encoding each level from the logits** (surface shift,
median, and enclosed-volume change):

| structure | level | decimate | re-encode |
|---|---|---|---|
| liver | 1 (6 mm) | 0.868 mm / -4.3 % | **0.524 mm / -2.8 %** |
| lung_lower_lobe_left | 1 | 0.532 mm / -3.7 % | **0.406 mm / -1.4 %** |
| gallbladder | 1 | 0.470 mm / -22.1 % | 1.101 mm / -20.3 % |
| gallbladder | 2 (12 mm) | 2.306 mm / -65.9 % | **1.378 mm / -52.0 %** |

Re-encoding is better for large structures and costs ~14 % (a pyramid *down* is `1 + 1/8 +
1/64`). But **the erosion floor remains**: the gallbladder still loses half its volume at
12 mm. That loss is not an encoding artifact — it is information genuinely absent at that
resolution, since the model's own logits interpolated to 12 mm no longer put the structure
ahead anywhere. **The pyramid's limit is resolution, not representation**, and no encoding
fixes it. Record per level which structures reached zero voxels, so a renderer does not
conclude a gallbladder is absent from the scan.

### 8.4 Group fields: ~10 structures instead of 118

A renderer usually wants composite structures, and grouping is a **decode-time** choice — the
store keeps all classes.

Build `d_S = max over members` and `d_notS = max over the rest` in one pass over the encoded
planes (no K-channel stack), then `m_S = d_S - d_notS` is the union's margin field.

**Internal boundaries vanish, which is the real win.** Of 1,413 adjacent voxel pairs where two
lung lobes abut, `m_S` crosses zero at **exactly zero** of them. Rendering five lobes as five
fields puts a surface at all 1,413 — five shells pressed together, with cracks, z-fighting and
doubled shading. The union is **one manifold**. And it is faithful: 99.9992 % identical to the
OR of its member labels.

**One field per group, not two.** Interpolation is linear, so
`interp(d_S) > interp(d_notS)` and `interp(d_S - d_notS) > 0` are the same question. Measured
bit-identical at 3, 1.5 and 1 mm, and **18x faster** at 1 mm (0.17 s vs 3.06 s) because it
skips a channel of sampling and the argmax.

**Against the uncompressed logits** (not against another decoder), on the real organs part,
depth 6 / clip 8:

| group | members | Dice | FP | FN | margin err p50 | p95 | max |
|---|---|---|---|---|---|---|---|
| liver | 1 | 0.99982 | **0** | 126 | 0.0078 | 0.0149 | 3.24 |
| kidneys | 2 | 0.99988 | **0** | 23 | 0.0079 | 0.0149 | 3.22 |
| lungs | 5 | 0.99994 | **0** | 87 | 0.0079 | 0.0149 | 3.33 |
| gi tract | 5 | 0.99955 | **0** | 264 | 0.0078 | 0.0148 | 2.73 |

**False positives are structurally impossible.** `ranks[0]` is the bit-exact argmax, so the
store always agrees on *who won*; a member can only fail to be claimed, never be falsely
claimed. Every loss is the quantization dead zone — a win smaller than one level (clip/255)
decodes to exactly zero — which is 0.035 % of the liver. The p50/p95 errors are half a
quantum, the quantizer performing to spec.

⚠ **The max error is depth truncation, and it costs magnitude rather than mask.** A member
below rank N reads as absent (`-clip`) instead of at its true level, worth up to 3.3 logits.
Depth 3 and depth 6 give *identical* masks on real data — same Dice, same FP, same FN —
while the liver's p95 error goes 0.0149 → 0.2969. So **depth is a rendering decision, not a
labelmap one**: a labelmap consumer can take depth 3, a consumer that samples the field for
an opacity ramp should not.

Four rules for building it:

- **Group at the model grid, then resample.** The `max` walks the planes once at model
  resolution; only the single resulting field pays the target-resolution cost. Grouping after
  resampling does the expensive part K times.
- **Emit uint8 quantized margin**, not float32 — ten groups at 1 mm is 1.5 GB as float32 and
  386 MB as uint8, which is what a GPU wants anyway (128 = the surface).
- **Take membership from the label LUT**, never from name prefixes. `startswith("lung_")` is
  the filename-semantics failure of §11 in another costume.
- **A cross-part group is per-part.** Composite the resulting fields; do not merge them into
  one field, which would require comparing margins across models.

### 8.5 The decode reduces to one operation

Built, as `nnseg.ranked`:

```python
resident = ranked.to_device(code, "cuda")            # upload the planes once
fields   = ranked.decode_groups(resident, groups,    # expand only what was asked for
                                quantize=True)       # (G, Z, Y, X) uint8, 128 = surface
```

Grouping is a `max` while walking the rank planes; resolution is the target grid handed to
`to_labels`. Neither is baked into the store, so the same bytes answer "118 classes at 3 mm",
"ten groups at 1 mm", or "two groups at 6 mm" with no re-encoding — which is what lets a
viewer toggle *show lobes separately* against *show lungs* as a half-second re-extraction
rather than a different artifact.

**Resampling is deliberately not in the decode.** It belongs to the restore, and taking a
`spacing` here would bake a grid into an operation whose whole point is that no grid is baked
in. The earlier sketch of this signature said `decode(groups, spacing)`; that was wrong.

#### Residency: the small thing persists, the large thing does not

`to_device` exists because the encoded planes are what should live on the GPU, and the fields
they stand for should not. On the organs part:

| form | size | holds |
|---|---|---|
| compressed store (disk, wire) | 2 MB | every class |
| **decompressed, still encoded, resident** | **126 MB** | every class |
| the fp16 K-channel field it represents | 574 MB | the same thing, expanded |
| four expanded groups, uint8 | 46 MB | only what was asked for |

So the resident form is **4.6× smaller than the field it stands for** while still carrying
all 25 classes, and any grouping expands out of it transiently (~0.3 s, four groups). That
inverts the pipeline as it stands, where the gigabyte logit field is alive during the run and
nothing survives it.

Without `to_device`, `decode_groups` re-uploads 126 MB on *every* call — which is precisely
the interactive case this is for. The measured win was only **1.13× on MPS**, where unified
memory makes an upload nearly free; on a discrete GPU across PCIe it should be larger, and
that is not measured here.

⚠ A resident code is for `decode_groups`. `margin`, `deficit` and `probabilities` read the
host arrays and raise rather than indexing a tensor with numpy semantics; `to_device` does
not consume its argument, so the host form stays usable alongside.

### 8.6 Gradients: saturation, not quantization, and which stencil

§8.2 asks for a ramp in millimeters, `m / |grad m|`. Everything hard about that is in the
divisor.

**Quantization is never the limit.** One support level is `clip/255` logits, which at the
confidence slopes actually measured on the store is **4.6 um** (sharp boundary, `k` = 6.87
logits/mm) to **12.3 um** (soft, `k` = 2.55). Three orders of magnitude below a voxel.

**Saturation is the limit.** The field is only informative out to `clip/k`:

| | `k` (logits/mm) | usable band |
|---|---|---|
| heart, liver, spleen, stomach | 2.55–3.26 | 3.1 mm ≈ 2.1 voxels |
| trachea, ribs, esophagus | 5.42–5.64 | ~1.45 mm ≈ 1.0 voxel |
| aorta, pulmonary vein, vertebrae | 6.11–6.87 | **1.16 mm ≈ 0.78 voxels** |

Bone, vessels and airways — the thin structures where aliasing is worst — are exactly the
confident ones, and their band is under one voxel. **Do not widen the clip to fix this:**
clip 8 → 16 takes the organs part from 1.93 MB to **34.56 MB**, because live runner-up slots
go 2.1 % → 62.3 %. The cost is *sparsity*, not the quantization curve, which is why companding
recovers only 15 % (29.39 MB) — a different reason from §10's, and both point the same way.

⚠ **Differentiate the signed margin, never `gap1`.** `gap1 = |m_w|` is an absolute value with a
**crease at the surface**, so a central difference straddling the boundary differences it
across its own kink and collapses. Medians agree (1.07 versus 1.43 logits/mm) because the
crease only touches surface voxels — but **19 % of band voxels understate by more than 2x**,
producing distances at least 2x too large in precisely the anti-aliasing band. Build `m_c`
(§1), then differentiate.

**Use plain central differences.** Measured on the rib cage against the gradient of the
*unclipped* field, over the shell that renders (72 % of which has saturation inside its 3×3×3
neighborhood):

| estimator | median | p95 | > 10° |
|---|---|---|---|
| **central** | **0.56°** | **16.28°** | **16.1 %** |
| sobel `[1 2 1]` | 10.77° | 51.36° | 53.4 % |
| scharr `[3 10 3]` | 8.63° | 42.02° | 43.0 % |
| one-sided (avoid saturated) | 0.64° | 60.63° | 32.1 % |
| masked sobel | 9.81° | 51.39° | 49.1 % |

⚠ Smoothing is the natural reach and it is **15–19x worse**. Anatomy here is one to two voxels
thick, so a `[1 2 1]` kernel across the perpendicular axes reaches the opposite wall and the
background and blends them into the direction; masking the saturated samples out does not
rescue it, because there is no clean material within reach. One-sided differencing fails the
same way from the other side — on a thin shell the clean neighbor is often *along* the surface,
so it returns a tangential vector, visible as a crawling speckle. **The rule is minimal
support.**

**A composite's normal must be recomputed from `m_S`, not inherited.** A per-class normal is
the gradient of `top1 - top2` — the winner against whoever is runner-up *at that voxel* —
which is creased wherever the runner-up's identity changes. Along a rib that flips constantly
between background, the adjacent rib, and cartilage. Shading a union with it:

| | median | p95 | > 10° |
|---|---|---|---|
| whole shell | 1.84° | 54.02° | 28.7 % |
| internal seams (1.08 % of shell) | 28.99° | 79.12° | 85.0 % |
| exterior only | 1.69° | 53.10° | 28.1 % |

It renders as visible corrugation. Note the exterior disagrees too, although `m_w = m_S`
holds *pointwise* there — the stencil reaches into neighbors where the competitor identity has
changed. Regrouping **dissolves** internal competitors, so a union's normal is a different
field, not a transformation of a stored one. Build `m_S` (§8.4), then differentiate it.

**Why not store the gradient.** The encoder holds unclipped logits and could write it; on the
band (7.6 % of voxels) that costs +0.79 MB for magnitude (+41 %) or +2.03 MB for an octahedral
direction (+105 %). Rejected — see §10.

---

## 9. Compositing across parts

The composite label map is **derived, and worth shipping**: 0.222 MB, 5.8 % of the store,
89 % of the size of the five per-part maps it summarizes. It is the only thing most consumers
want, and deriving it requires the catalog's **paint order**, which is external knowledge.
Alphabetical is *not* composition order (`cardiac, muscles, organs, ribs, vertebrae` versus
`organs, vertebrae, cardiac, muscles, ribs`), so a reader doing the obvious thing gets a
different segmentation, silently, in the contested voxels.

**A composite margin is also worth shipping**, and needs no special caveat. The margin is
already a *locally referenced* quantity — within a single model, 20 of 25 classes appear as
the runner-up somewhere, so the peer set varies voxel to voxel regardless. Compositing widens
the pool of possible competitors without changing the kind of statement being made: *"at this
voxel, this label leads its best local competitor by m logits."*

The one genuine exception is the contested set, where paint order discarded a competing
claim. Those voxels are self-announcing — median margin 4.49 logits versus 8.00 elsewhere,
11.7 % below 1 logit versus 2.7 % — and **derivable** from the per-part label maps, so they
should not be materialized. A cached query result is not information.

### 9.1 The paint order is literally painter's algorithm

Upstream is unambiguous — `TotalSegmentator/totalsegmentator/nnunet.py:601-637`:

```python
seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)   # blank canvas
for idx, tid in enumerate(task_id):                            # fixed task order
    mapped_seg = lut[seg]                                      # part labels -> global
    np.copyto(seg_combined[img_part], mapped_seg, where=mapped_seg != 0)
```

Blank canvas, parts painted in task order, **last writer wins**, no probability consulted
anywhere. So deferring the merge to the consumer loses nothing, and gains: a consumer holding
margins can arbitrate better than task order.

It also barely matters. Of 2,371 k voxels claimed by at least one part, **2,370.6 k (99.96 %)
are claimed by exactly one**; 0.8 k are contested, all of them organs↔cardiac, at a median
margin difference of 2.92 logits (21 % within 1 logit).

| part | foreground voxels | |
|---|---|---|
| organs | 1,555 k | 13.56 % |
| muscles | 367 k | 3.20 % |
| cardiac | 234 k | 2.04 % |
| vertebrae | 115 k | 1.00 % |
| ribs | 102 k | 0.89 % |

This is why §8.4's fourth rule holds without costing anything: cross-part groups stay
per-part, and the arbitration that a shared margin scale would buy applies to 0.04 % of the
volume.

**For composing across tasks, engines and cascade stages** — what fits the store, what does
not, and why a pipeline is exact where a composite is heuristic — see
[ranked-composition.md](ranked-composition.md).

---

## 10. Rejected alternatives, with numbers

**Companding the support** (spending the 255 levels non-uniformly). Tested with the corrected
`deficit` field across clips 16–40 and gammas 0.7–3.0. Uniform wins decisively:

| clip | γ | near-tie |
|---|---|---|
| 16 | **1.0** | **99.21 %** |
| 20 | **1.0** | **99.43 %** |
| 40 | 1.5 | 81.54 % |
| 40 | 2.0 | 73.31 % |
| 40 | 0.7 | 85.36 % |

Fine resolution near a tie (0.31 mlogit versus 31) makes it **worse**. An interpolated decision
is a weighted combination across the whole stencil, so the far corners matter as much as the
near ones; interpolation wants **uniform accuracy across the range**, and any non-uniform
allocation starves terms in the same sum. If both reach and resolution are needed the answer
is more bits, not a cleverer curve.

**Encoding at a finer grid.** Upsampling with full depth and encoding two planes there does
recover accuracy monotonically — and loses anyway:

| store | voxels | Dice at 0.651 mm |
|---|---|---|
| **full depth @ 3 mm (native)** | **8,192** | **0.9999** |
| 2-plane @ 3.0 mm | 8,192 | 0.9824 |
| 2-plane @ 1.5 mm | 65,536 | 0.9950 |
| 2-plane @ 0.5 mm | 1,769,472 | 0.9993 |

216× the voxels to end up *behind* full depth at native. Upsampling adds no information — the
network computed at 3 mm — it only reduces three-way junctions per voxel, which is exactly
what the deeper planes encode directly and far more cheaply. Measured store cost of a 2×
level: **6.8×** (0.373 → 2.555 MB), with `margin` growing 7.4× (quasi-continuous, nothing to
collapse) against `labels` at 3.7×.

**Dropping the rank planes** (`labels` + `support[0]` + `support[1]`, no `ranks[1]`): 69.8 %
against 94.7 %. See §3.

**Cubic interpolation.** Reconstructs *better* on average (98.38 % versus 96.91 % for linear
on a downsample-and-restore test) but violates a structural invariant: **59,885 voxels with
deficit > 0**, max +6.2, which says a class beat the maximum. Linear produced exactly zero.
Staged linear — wider support without the negative lobes — was worse than a single step
(96.28 %), so width alone is not the benefit.

**A stored gradient plane.** The encoder holds unclipped logits, so it can compute
`|grad m_w|` before any saturation and write it on the band (7.6 % of voxels), which a decoder
cannot reconstruct afterward. Priced on the organs part:

| | size | delta |
|---|---|---|
| ranks + support (today) | 1.93 MB | — |
| + magnitude, uint8 | 2.71 MB | +41 % |
| + octahedral direction, 2 B | 3.96 MB | +105 % |
| + both | 4.74 MB | +146 % |

Rejected on both counts. **Magnitude**: plain central differences already land at 0.56° median
against it (§8.6), and the residual is invisible in a rendering. **Direction**: worse than
unnecessary — a stored normal bakes in a *competitor set*, and since grouping is a decode-time
choice (§8.4), the one grouping it was encoded with is the least useful one. Measured against
a union's own normal it disagrees on 28.7 % of the shell. A plane that must be recomputed for
every composite is dead weight.

**A truncated distance field for all structures.** The ranked layout does not transfer to
distance. Its premise is that the softmax is peaked — top-6 of 118 captures everything — but
proximity is not peaked: within 20 mm of a voxel there are a median of 3 structures, p95 **7**,
max 13 (at 40 mm: 8 / 16 / 27). Depth 4 covers the median and truncates the tail; a complete
R = 20 mm store needs depth ≈ 13, landing near the size of the logit store it was meant to
supplement. The distance plane also dominates the bytes (79 %), and it is the plane ranking
does nothing for — its redundancy is *spatial smoothness*, not class sparsity. Logits are
sparse in class space and want ranking; distance is smooth in physical space and wants
multiresolution (a downsampled-then-interpolated SDF is accurate to ~0.10 mm beyond 10 mm at
4x, 0.15 mm beyond 40 mm at 8x). Different redundancy, different encoder. An SDF store is also
a **cache**, not source data — 1.7 s per structure to rebuild from labels — where the ranked
store holds the only record of what the network said.

---

## 11. Storage layout

**A sharded zarr directory, not a zip.** Measured: 3.82 MB in 83 files versus 3.89 MB in one
zip of 1067 entries. A zip entry costs ~127 bytes (local header, central directory record,
filename twice); a shard index slot costs 16 bytes per chunk. Sparsity is 61 %, and zip's
zero-cost absence does not make up an 8× per-entry difference. The zip is *packaging*, the
directory is *storage*; converting is mechanical.

⚠ A `ZipStore` is append-only and cannot overwrite a key, so writing arrays and then
annotating the group appends a **second** `zarr.json` rather than replacing the first. Readers
take the last by convention, but any reader iterating local headers gets stale metadata.

**Tiers**, and the fetch cost of each:

| tier | contents | size | of full |
|---|---|---|---|
| 1 | `labels` — a valid segmentation alone | 0.27 MB | 7 % |
| 2 | + `margin` — confidence, surfaces, resampling at Dice 0.99 | 2.41 MB | 64 % |
| 3 | + refinement planes — Dice 0.998, exact unions, posteriors | 3.75 MB | 100 % |

Per-plane fetch is free provided **no chunk or shard spans the list axis** — shards then live
under per-plane prefixes and reading plane 0 touches only its own objects. Splitting into
separate 3D arrays buys per-plane *naming*, not per-plane access.

**A per-chunk class-presence index** costs 0.1–0.2 KB per layer and a median structure
occupies only 6–12 % of chunks, so a single-structure read touches ~8 % of the store (53 MB
of 631 MB raw). Define its granularity as an explicit field, not implicitly the chunk grid,
so it can follow a future sharding layout.

⚠ Names are addresses, not attributes. Order must be explicit in metadata — recovering it by
sorting names (numeric or alphabetic) makes the name load-bearing, which is the failure this
avoids. If a directory name disagrees with `meta.part`, the metadata wins; cross-checking them
makes the name authoritative through the back door. Reserve *structure* (`parts/`, `derived/`)
rather than names, so a part legitimately called `composite` collides with nothing.

---

## 12. Open questions

- **Depth defaults.** Depth 3 + adaptive floor matches depth 6 at K = 118, and the parts
  converge sooner. The residual loop (§5) should set it per part; the fixed default should
  move meanwhile.
- **The adaptive floor is not implemented.** `encode` stores N−1 supports.
- **Clip.** With an adaptive floor the fixed clip matters much less. Without one, the curve
  peaks at 12–16 for a 3 mm model and *reverses* by 24 (99.43 → 99.18 %) as the coarser step
  costs more than the added reach. Required reach should scale with logit change per voxel,
  hence with spacing — untested.
- **The 2-plane bias is systematic.** Every structure came out *larger* (+0.45 % to +1.30 %),
  because flooring non-winners at −m over-promotes third classes. It will accumulate in one
  direction across a cohort; matters for volumetry, not for viewing.
- **Region (sigmoid) heads** have no rank structure, so none of §3–§5 applies; `encode_regions`
  stores one margin plane per region and everything in §6 about *what* to interpolate still
  holds.
- **A conservative bound is not implemented.** §8.2 needs a max-pooled, 1-bit-per-structure
  hierarchy; the per-chunk presence index of §11 is the same object and is also unbuilt.
- ~~**Group extraction is not in the API.**~~ Closed: `ranked.decode_groups` (§8.5), with
  `ranked.to_device` for residency.
- ~~**Uint8 group output.**~~ Closed: `decode_groups(..., quantize=True)`, on the same
  128-is-the-boundary convention `encode_regions` stores.
- **The residual loop needs a second criterion.** §5 stops when the interpolated argmax stops
  changing, and is emphatic that the test be decision-based rather than value-based. That is
  right for a labelmap and blind to what a renderer needs: §8.4 measures depth 3 and depth 6
  giving **identical** group masks on real data while the liver's p95 margin error goes
  0.0149 → 0.2969. So a decision-converged store can be 20× worse in the very field a
  renderer samples for its opacity ramp, and the loop cannot tell. Dynamic depth is still the
  answer — but the loop has to know which error it is minimizing, and for the field consumer
  that is the value-based test §5 rejects for the other one.
- **All measurements are two cases.** The *shape* of every finding was consistent across ten
  segmentations, but absolute numbers will move.

Questions about composing several stores — cross-part calibration, a codec per layer type,
graded cascade inputs, stage addressing — are in
[ranked-composition.md](ranked-composition.md) §8.
