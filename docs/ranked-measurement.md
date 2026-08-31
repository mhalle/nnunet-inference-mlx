# Measuring from a ranked store — volume, area, and what counting gets wrong

**Volume and surface area as integrals of the margin field rather than properties of a
mask.** The fourth of the `nnseg.ranked` documents, after
[ranked-probabilities.md](ranked-probabilities.md) (what is stored),
[ranked-reconstruction.md](ranked-reconstruction.md) (getting one store back out) and
[ranked-composition.md](ranked-composition.md) (combining several). This one is about not
getting it out at all — reading a number off the field directly.

Written 2026-08-31. Phantom numbers are against closed-form or spectrally-converged truth
(`nnseg.phantoms`); real numbers are the 1.5 mm `total` organs part of **`idc-torso1`**
(52 M voxels, K = 25), the case described in
[ranked-reconstruction.md §0](ranked-reconstruction.md#0-the-two-cases).

---

## 1. Why this is a question at all

Both quantities are integrals:

    V = ∫ H(m) dx                A = ∫ δ(m) |∇m| dx

so nothing requires a raster. `nnseg.statistics` counts voxels because that is what a
labelmap offers, and counting volume is defensible. Counting **area** is not:

| body (1.5 mm) | counted volume | field volume | counted area | field area |
|---|---|---|---|---|
| sphere | +0.23 % | −0.28 % | **+49.33 %** | −0.18 % |
| ellipsoid | +0.20 % | −0.34 % | +45.94 % | −0.19 % |
| torus | −1.28 % | −0.59 % | +39.19 % | −0.23 % |
| shell (4 mm wall) | +1.38 % | −0.09 % | +50.01 % | −0.59 % |
| box 14³ | **−10.34 %** | −0.28 % | −7.02 % | −4.52 % |
| rounded box | +4.18 % | −0.33 % | +21.42 % | −0.59 % |
| star | +0.39 % | −0.28 % | +54.40 % | −0.28 % |

Face counting is not a coarse estimate of area. It is an estimate of a **different
quantity** — the staircase genuinely has more surface than the surface it stands for — and
it does not improve when the grid does:

| spacing | counted vol | field vol | counted area | field area |
|---|---|---|---|---|
| 3.00 mm | −2.67 % | −1.11 % | +50.40 % | −0.87 % |
| 2.00 mm | +0.84 % | −0.50 % | +50.88 % | −0.28 % |
| 1.50 mm | +0.23 % | −0.28 % | +49.33 % | −0.18 % |
| 1.00 mm | +0.12 % | −0.12 % | +50.88 % | −0.09 % |
| 0.75 mm | +0.19 % | −0.07 % | +50.67 % | −0.06 % |

Four refinements, no trend. The field converges at O(h²) beside it.

Counting's volume error is small but **erratic** — it swings from −10.3 % to +4.2 % across
shapes depending on how the body sits against the grid — and it saws by 1.10 % of a
sphere's volume and 2.69 % of its area under pure sub-voxel translation, where the field
moves 0.00 % and 0.14 %. That stability, not the accuracy, is the argument for longitudinal
work: a follow-up scan resampled half a voxel differently must not report a changed organ.

**On real data** the face-count/field area ratio is 1.51–1.68 across spleen, kidneys,
liver, stomach, lung and trachea, and the volumes agree to 0.01–0.95 %.

### 1a. The fair baseline is Crofton, not face counting

Face counting is the naive raster measure and beating it proves little. SimpleITK's
`LabelShapeStatisticsImageFilter.ComputePerimeter` is a **Crofton** estimator — proper
stereology, which corrects the staircase — and it is the measure
[medseg `docs/segmentation-storage-and-duckn.md` §Statistics](../../docs/segmentation-storage-and-duckn.md)
proposed shipping. It is good, and it is the comparison that matters:

| sphere r=20, 1.5 mm | face counting | Crofton | field |
|---|---|---|---|
| area error | +49.33 % | −0.70 % | −0.18 % |
| box 14³ (a crease) | −7.02 % | **−16.57 %** | −4.52 % |
| torus | +39.19 % | −2.13 % | −0.23 % |
| ellipsoid | +45.94 % | +0.02 % | −0.19 % |
| sub-voxel translation spread | 2.69 % | 0.85 % | **0.14 %** |

On a smooth body at one grid, Crofton is within a couple of percent and is a perfectly
reasonable number to publish. The field wins on the other three axes:

- **Stability**: 0.14 % against 0.85 % under pure sub-voxel translation — 6×. This is the
  property that matters for longitudinal work, where the question is whether an organ
  changed, not how big it is.
- **Convergence**: Crofton bounces (−0.80, +0.28, −0.70, +0.19, +0.18 % over 3.0 → 0.75 mm)
  where the field descends monotonically (−0.87 → −0.06 %). Crofton is unbiased-ish but
  noisy; the field is consistent, so refining the grid buys something.
- **Creases**: −4.5 % against −16.6 %.
- **It needs no raster.** Crofton consumes a labelmap, so it requires the restore; the field
  path reads the store directly.

So the honest claim is not "counting is hopeless" — it is that Crofton is a good single-grid
estimator and the field is a better one, most clearly where the answer has to be compared
against another answer.

---

## 2. The method

One sweep over the cells **between voxel centers**. Cells whose eight corners agree in sign
are full or empty and contribute exactly; straddling cells get the plane their corner values
imply, and the volume and area of a plane cutting a box are elementary (Scardovelli–Zaleski):

    V(α) = Σ_S (−1)^|S| max(α − ΣS, 0)³ / (6abc)
    A(α) = |∇| · Σ_S (−1)^|S| max(α − ΣS, 0)² / (2abc)

over subsets `S` of the normal's three components. **Area is the volume expression
differentiated in the level** — the co-area formula written for a box — so the two cannot
disagree about where the surface is. No mesher and no dependency beyond numpy.

Three implementation facts that are not optional:

**The general form divides by `6abc`.** A face perpendicular to an axis has two vanishing
components, and differencing cubes over a denominator of 1e-18 returns noise scaled by the
field's units. That is where a medical image spends most of its surface. The degenerate
cases are limits (one axis → a slab, two → a prism) and are taken as such, branching on the
**sorted** components so the test is on how flat the plane is, not which axis it faces. The
first version of this had the sphere right and the box off by 80 %, with a negative area.

**Only straddling cells work** — 5–29 % of a real organ's bounding box. That is the point of
not rasterizing.

**It streams.** The dense form allocates eight float64 copies of the volume: 3.3 GB for a
473 × 333 × 333 part, which is how the first version died on a real store. `volume_area`
crops to the structure's bounding box and sweeps z-slabs; the slab size is asserted not to
be part of the answer. Real cost is ~40–180 ms per structure on 52 M voxels, against ~170 ms
to decode the margin in the first place.

---

## 3. The clip is the largest error, and it is correctable

**Pass `clip`.** `measure.volume_area(m, spacing, clip=code.meta["clip"])`. Omitting it
costs 1–5 % of the area on real margin gradients.

A cell's corners reach half a cell diagonal from the surface — 2.6 mm at 1.5 mm spacing.
Wherever the margin climbs faster than `clip` over that distance, the far corners of
straddling cells sit **at** the clip. Those are bounds, not values, and averaging them into
the plane fit flattens it.

This is not a phantom curiosity. Measured on the real organs part:

| structure | \|∇m\| logits/mm | band / cell diagonal | straddling cells with a clipped corner |
|---|---|---|---|
| spleen | 5.79 | 0.53 | 68.7 % |
| kidney (R) | 6.35 | 0.48 | 80.3 % |
| liver | 4.14 | 0.74 | 30.5 % |
| stomach | 3.02 | 1.02 | 12.6 % |
| lung (UL) | 6.86 | 0.45 | 88.9 % |
| trachea | 7.24 | 0.43 | 94.7 % |

Sweeping a phantom across that same band-to-diagonal range:

| band / diagonal | 0.40 | 0.50 | 0.60 | 0.80 |
|---|---|---|---|---|
| sphere, ignoring the clip | −5.16 % | −4.74 % | −3.64 % | −0.48 % |
| sphere, `clip=` passed | −0.99 % | −0.65 % | −0.41 % | −0.15 % |
| cube | unchanged at every ratio, bit for bit |

At the mid-range ratio, every smooth body lands within 0.9 % once the clip is passed:

| body | counted area | field, clip ignored | field, clip passed |
|---|---|---|---|
| sphere | +49.33 % | −3.64 % | −0.41 % |
| ellipsoid | +45.94 % | −2.86 % | −0.38 % |
| torus | +39.19 % | −2.71 % | −0.40 % |
| shell | +50.01 % | −3.74 % | −0.42 % |
| rounded box | +21.42 % | −2.76 % | −0.83 % |
| star | +54.40 % | −4.22 % | −0.51 % |
| box | −7.02 % | −4.52 % | −4.52 % |

On the real store the correction recovers **+0.58 %** (stomach, widest band) to **+5.18 %**
(lung, narrowest) of the surface, tracking the gradient exactly as the dose-response
predicts. Volume moves at most 0.04 %: clipping never cost the volume anything, because the
misplacement is symmetric across the surface and cancels.

**Dropping the censored corners is the wrong fix.** `|ψ| ≥ clip` is information. A
flat-faced body at a narrow band has whole corner planes censored, and an unconstrained fit
rotates the normal freely — a cube went from −3.8 % to **−17.1 %**. So it is one step of a
censored regression: fit without them, then reactivate any whose prediction lands back
inside the band or on the wrong side, which is exactly the constraint they carry. Two steps
converge, and a cube comes out bit-for-bit unchanged.

**Raising `clip` to ~16 would also fix it**, with no estimator change, and was rejected:
`clip` bounds each class's spatial support, which is the whole compression argument
([ranked-probabilities.md](ranked-probabilities.md)). Doubling it roughly doubles the
support planes — 2.1 MB of a 4.66 MB part — and needs every store re-encoded. The estimator
correction is free and changes no bytes.

---

## 4. What does not help

**Subdivision.** Trilinear-upsampling the field and sweeping finer cells moves a sphere from
−0.282 % to −0.281 % for 67× the time. The error is the C⁰ trilinear interpolant, not the
quadrature; no rule inside its cells can beat the cells. The way past it is a **cubic
B-spline** interpolant, not a finer rule. There is deliberately no `refine` knob.

**Depth.** A class at its own surface is by definition tied with the winner there, so every
surface lives in `ranks[0]` and `ranks[1]`. On the real store at most two classes sit within
a logit of the winner at 99.997 % of voxels, three at 0.003 %, four at none. Depth matters
for the probability tail, not for a level set.

**Quantization.** The uint8 support contributes nothing measurable — clipping alone
reproduces the whole area loss to three decimals. Round-to-nearest is noise, and noise on a
surface *adds* area rather than removing it.

---

## 5. Known biases

**Sharp creases read ~4.5 % low in area.** A trilinear field cannot represent an edge, and
no fit over its corners will; subdividing makes it worse, since the finer interpolant rounds
more. Volume is unaffected (−0.28 %, the same as every smooth body) — rounding a corner
moves a second-order amount of volume and a first-order amount of surface. Anatomy has few
true creases, but a structure with one will under-report. This is the one remaining error
that needs the B-spline.

**Area is scale-dependent by nature.** It is a first-derivative functional: a ripple at fixed
amplitude moves area 5.6 % across spatial frequencies while volume moves 0.5 %. An area must
always be reported with the spacing it was measured on, and **must never be compared across
grids**. A volume may be.

**The domain is the box the voxel centers span**, half a voxel inside the array. A structure
touching the array edge is cut; real ones sit inside an envelope.

**Segmentation error is not captured and usually dominates.** `total` vs `total_fast` differ
by 14 % on autochthon and 36 % on gallbladder. Nothing here bounds that.

---

## 6. Status

`nnseg.measure` is the primitive. `nnseg.statistics.compute_statistics(..., ranked_code=)`
reports `volume_ml_field` and `area_cm2_field` **beside** `volume_ml`, never instead of it,
and passes the clip. Without a code the output is byte-identical to before — asserted,
because that is what makes shipping both safe. The field numbers carry their own
`field_grid_spacing_mm`: the code lives on the model grid, the labelmap has been restored
onto the input's, and for area that distinction is load-bearing.

**Not wired into the server.** `serve.artifact_overlap` computes from the in-RAM pair with
no input or disk dependence, and the ranked code is not there — carrying it in is a design
decision about that thread, not a mechanical edit, and the code only exists when a run asks
for it. So in practice the served `statistics.json` and `.tsv` are unchanged today.

**Not switched.** Both numbers ship until the comparison has run on a real cohort. Field
volume runs 0.01–0.95 % below counted on real organs, so published values would move under
1 % for large structures — and more for small ones, which is where the change is most
defensible and least comparable to history.

**Why not the SurfaceNets mesh** ([mesh-pipeline.md](mesh-pipeline.md)): its vertex placement
was tuned for appearance, and triangle areas off a dual mesh have their own bias. Measuring
the field directly has closed-form truth to validate against; the mesh path would need that
validation before it could be trusted, and it would still be measuring the same field.

---

## 7. Corrections made while working this out

**"Quantization is a low-pass filter that loses area."** Wrong. It is entirely the clip.
Round-to-nearest is noise, and noise adds area. The tell was there and was missed.

**"The stored form costs ≤ 0.06 %."** That was measured with subdivision on, which smooths
the quantization steps. At the shipped setting, volume is still free (≤ 0.03 %) but area was
1–5 % low until the clip correction.

**"Creases read ~6 % low."** 4.5 % at the shipped setting; the 6 % figure came from the
subdivided run, which is worse rather than better.

**"Counting's area is +50 %."** True of face counting, which is what the harness
implemented — but the raster measure actually proposed for this system was Crofton, which is
within a couple of percent. The first version of this document compared against the naive
baseline only and over-claimed; §1a is the fair comparison.

**"An 8-fold junction shows depth 6 is marginal."** An 8-fold junction *line* is
non-generic — in general position at most four regions meet in 3D, at isolated points. Real
anatomy agrees, and the phantom is a stress case rather than a forecast.
