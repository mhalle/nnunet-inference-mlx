# Composing and pipelining ranked stores

*Written 2026-08-30. Companion to [ranked-reconstruction.md](ranked-reconstruction.md), which
covers getting things out of **one** store. This one is about several — across parts, across
engines, and across cascade stages. See also
[ranked-measurement.md](ranked-measurement.md), which reads volume and area off a store
without getting anything out of it.*

The premise is that a segmentation's stored output can be the **deliverable** rather than an
intermediate, so that combining happens after inference instead of during it. That works, but
not uniformly, and the boundary is not where it first appears.

---

## 1. Two independent axes

The natural instinct is to sort by engine — "nnU-Net works, FastSurfer doesn't." That is
wrong, and it is the same mistake as baking a grouping in early (§8.4 of the companion). Two
unrelated questions are involved:

- **Axis A — does the output fit the store?** A property of the output's *structure*.
- **Axis B — can the computation be deferred?** A property of the *dependency graph*.

A task can fail either independently. Cascades fail B while fitting A perfectly; VoxTell fits
A for any given prompt but fails B, because the prompt is an *input* and the catalog is
unbounded. Filing them in one list hides that.

⚠ Axis A is also more permissive than it first looks, and the instinct to sort by engine is
what obscures it. FastSurfer seg and SynthStrip both fit; region heads fit in a different
mode. The genuine Axis-A exclusion is narrow — outputs that are not voxel fields at all.

---

## 2. Axis A — what the store can hold

The ranked encoding assumes a **per-voxel ranked list of classes competing within one
softmax**. Anything with that shape fits, whatever produced it.

| structure | examples | fits |
|---|---|---|
| multi-class softmax over a partition | nnU-Net parts; FastSurfer seg; MONAI labelmap bundles; VoxTell for a fixed prompt | exactly |
| single graded channel | SynthStrip SDT | yes — one layer |
| overlapping sigmoid regions | BraTS tumor core / whole / enhancing | yes — different mode |
| non-voxel | FastSurfer surfaces, thickness | out of scope |

**FastSurfer seg is a first-class citizen**, not a special case. `engines/fastsurfer.py` already
captures the pre-argmax logit field and resamples *it* rather than a labelmap —
`restore_logits(logit_zyx, ...)` with `K = logit_zyx.shape[3]`, sized at 256³ × 79. That is
structurally an nnU-Net part with 79 classes, and encodes with `ranked` unchanged.

**SynthStrip is a better citizen than nnU-Net output**, not a worse one. Its network emits a
signed distance field in millimeters, and the mask is a threshold applied afterward
(`engines/synthstrip.py`):

```python
sdt_native = restore_sdt_gpu(...)   # graded field, resampled to the input grid
mask = sdt_native < border          # threshold AFTER the resample -> sub-voxel boundary
```

One layer whose margin is already metric and whose threshold is still live. It wants a
different codec from the ranked one — linear over a truncation band, far field coarse — but
the same container. **Same store, different codec per layer type.**

**Region heads fit too, in a different mode — do not repeat the claim that they do not.**
Overlapping regions are independent Bernoullis, so there is no winner and no runner-up, but
the logit *is already* the margin: `encode_regions` stores `m_c = l_c - threshold`, one plane
per region, positive inside. Folding the threshold in at encode time is what lets a consumer
treat zero as the boundary **without knowing the head type**, so a renderer or mesher sampling
margin fields works on them unchanged, and `margin()` / `probabilities()` carry explicit
region branches.

Compositing is in fact *simpler* here: with no competitor set, union is `max(m) > 0` and
intersection `min(m) > 0`, both exact — none of the `d_S - d_notS` construction of §3 is
needed.

Three things genuinely do not carry over:

- **Ranking.** §3–§5 of the companion (progressive refinement, adaptive floor, dynamic depth)
  presuppose a rank order. `encode_regions` stores K full planes, no sentinel sparsity, no
  truncation — fine at BraTS's K = 3, not a compression win at large K, though large-K region
  heads are not a thing that exists.
- **The labelmap view.** Overlapping regions cannot argmax to one label. That is a limit of the
  output *view*, not of the representation.
- **Automatic naming.** `engines/monai_bundle.py:87` refuses to interpret a bundle's region
  `channel_def`, deliberately — read as a labelmap, brats reported its background as "Tumor
  core: 6372 ml". An introspection gap, not a representation gap.

---

## 3. Composites: exact within a softmax, painter's across

**Within one part** the union margin `m_S = d_S - d_notS` is exact, because every class came
from one softmax and the margins are directly comparable. §8.4 of the companion covers the
construction and the four rules.

**Across parts there is no algebra**, because the parts are different models with different
preprocessing — the normalization-sharing bug of 2026-08-28 is the proof — so their logits
share no scale. What exists instead is painter's algorithm, and that is already what upstream
does: see §9.1 of the companion for the code and the measured contention (0.04 % of claimed
voxels).

The practical consequence is mild. Deferring painter's to the consumer is lossless, and a
consumer holding margins can do better than task order in the 0.04 % where it fires. Calling
that *principled* would need per-part temperature calibration; nobody needs it yet.

⚠ **A composite's shading normal is not a stored quantity.** Recomputing it is not an
optimization to skip — see §8.6 of the companion, where inheriting a per-class normal renders
as visible corrugation.

---

## 4. Pipelines: exact, and that is the surprise

Cascades are the largest Axis-B population — about 30 TS tasks declare a `"crop"` in
`map_tasks_config.py`, each cropping the fine stage to a box around another task's structures.
That is a data dependency on the network's **input**, so no amount of stored output lets you
skip inference. `lung_vessels` cannot be synthesized from a finished `total`.

But the interesting question is not whether inference can be skipped. It is whether stage 2,
run from a **stored** stage 1, gets what it would have gotten from the live logit field. It
does, exactly:

- Every cascade consumes **hard labels**. TS derives a bounding box from them; nnU-Net's own
  cascade one-hots the coarse labelmap into extra input channels
  (`convert_labelmap_to_one_hot`, `nnunetv2/inference/predict_from_raw_data.py:35`).
- `ranks[0]` **is** the argmax, stored losslessly — the zero sentinel only ever touches ranks 1
  and up. Verified against the live field on the organs part: **11,464,290 / 11,464,290 voxels
  identical.**

  ⚠ `topk` does not define an order among equal values, and CUDA's is not the CPU's, so
  without care the stored bytes depend on where they were encoded. Promotion does not help:
  `encode` casts to fp32 before `topk`, but fp16 -> fp32 is exact, so ties survive it — and
  fp16 is what the network returns. `_settle_ties` orders equals by ascending class index,
  which is `argmax`'s own convention (numpy and torch both return the first maximal index),
  so `ranks[0]` remains exactly the argmax every labelmap in this ecosystem was made with.
  That inherits argmax's existing bias toward background rather than adding a new one; the
  bound is **0.155 %** of the worst structure's volume, against ~9 % for the linear-vs-nearest
  choice already in the pipeline.

  With it, labels are bit-identical across backends (0 of 11,464,290, from 3,669 before).
  **One residual remains:** ordering the N selected entries cannot change *which* `topk`
  selected, so a tie straddling the depth boundary is still backend-dependent. On the real
  organs part at depth 6 that is 194,043 voxels — of which all but **16** are among classes
  beyond the clip, where both candidates mask to the sentinel and the choice never reaches the
  stored bytes. For those 16 the contested gap is 6.60–7.97 against a clip of 8, so the worst
  decoded margin differs by 1.40 logits, on the least-significant plane, for a class already
  more than 6.5 logits behind. Closing it needs a stable descending sort instead of `topk`
  (measured 3.4–3.7×) and is not worth it — unless the store is ever content-addressed by
  checksum, where any difference is a difference.
- The box a cascade actually consumes is therefore identical. For the five lung lobes that
  `lung_vessels` and `lung_nodules` crop to: `((94,240), (13,116), (52,218))`, 2.55 Mvox, from
  both the live logits and the store.

So a cascade replayed from a 2 MB artifact is not an approximation of the live run — it is the
same computation. **The pipeline case is exact where the composite case is heuristic**, which
is the reverse of what one would guess.

### 4.1 What margins add here, and what they do not

They do **not** make the crop box more robust. A bounding box is in principle hostage to one
stray voxel, and strays exist — the `lung_vessels` run had 1,060 connected components, 1,034 of
them under 50 voxels. But applying a confidence floor barely moves it:

| floor (logits) | lung voxels | box volume |
|---|---|---|
| 0.0 | 749,564 | 100.0 % |
| 0.5 | 743,885 | 100.0 % |
| 1.0 | 737,976 | 99.3 % |
| 4.0 | 699,285 | 99.3 % |
| 6.0 | 668,809 | 97.7 % |

An 11 % drop in voxels moves the extent 2.3 %, because a lung's extremes are confidently lung.
What margins give is **visibility into that stability**, which a labelmap cannot provide — not
a better box.

Where a stored approximation would genuinely start paying is a stage that consumed a *graded*
input. Feeding the margin field into a fine stage instead of one-hot is strictly more
information — "confidently lung" versus "barely lung" rather than a flat mask — but the trained
models expect one-hot, so that is a retraining question, not a drop-in.

---

## 5. What this buys

**Intermediate stages become durable, addressable artifacts instead of temp files.** TS writes
part segmentations to a temp directory and deletes them. If they are stored, stage 2 is
re-runnable, at any later date, with different parameters, from ~2 MB — and stage 1 is the
expensive half (`total` at 36 s warm on an L40S). The serve tier already has the addressing
(`/v1/<source>/<ident>/<task>/…`); this extends it from whole tasks to stages.

**The task graph becomes data.** Those ~30 `"crop"` dependencies are currently hard-wired
control flow. With stages stored, "can I run `liver_segments`?" becomes "is `liver` in the
store?" — a lookup rather than a pipeline rerun.

**Post-hoc operations that upstream left unwired become available.** `remove_outside`
(`map_tasks_config.py:376`, masking by `["heart","aorta","inferior_vena_cava"]` at 10 mm)
is a composite of two stored layers plus a dilation, not a pipeline step. It is commented out
upstream for `lung_vessels`, which is why 73 % of predicted `lung_airways` lands outside the
patient at −900 HU.

**The toolbox spans engines.** A brain MR study can carry FastSurfer's 79-class store,
SynthStrip's SDT, and a MONAI bundle's output in one container. These overlap by *containment*
rather than contention, which is a useful composite: "mask the seg by the brain, gated at
whatever border I choose now" is a read-time operation on two layers, with the threshold still
live because the SDT was stored graded rather than thresholded at inference.

---

## 6. What remains out

- **Cascade inference** (Axis B). The box is derivable; the network still has to run.
- **Unbounded catalogs.** VoxTell's prompt is an *input*, so its output fits the store per
  prompt but the catalog cannot be precomputed.
- **Non-voxel outputs.** Surfaces and thickness are a different kind of artifact.

---

## 7. The standing rule

Everything that broke while working this out broke by **fixing a grouping too early**: the
stored normal plane bakes a competitor set, composite-by-`max` underestimates because it does
not know the group, the merged labelmap bakes both the argmax and the paint order. Everything
that worked deferred the grouping to the point of use.

So: **a composite is a view, not a thing.** Store what each model said; decide groupings when
someone asks.

---

## 8. Open questions

- **Per-part temperature calibration** would make cross-part margins comparable and turn
  painter's into a real arbitration. Applies to 0.04 % of the volume, so it is not urgent.
- **A codec per layer type.** SynthStrip's SDT wants linear-over-a-band with a coarse far
  field; the ranked codec is wrong for it. The container should carry the codec name.
- **Graded cascade inputs** — training a fine stage on margins rather than one-hot.
- **Stage addressing.** Extending the serve cache from task results to pipeline stages needs a
  key that captures the stage's inputs, not just the task name.
