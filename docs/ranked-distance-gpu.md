# The distance field on a GPU — plan (2026-09-01, not yet built)

The CPU implementation in `tools/ranked_build_store.py` (`distance_field`) computes the
nearest-surface distance for a five-part `total` in **9.9 s** after band-restriction (53 s
dense). That is fine for offline bakes. Two consumers want it faster:

1. **The server bake at scale.** Batch runs on Modal already have the ranked arrays on an
   L40S when the encode finishes; the distance field should be produced there, in the same
   device residency, rather than in a numpy pass afterward.
2. **The client, for selections.** The baked field is the distance to the nearest surface of
   the *whole* labelmap. An arbitrary selection of structures has its own outer boundary, so
   the client re-seeds from membership flips and re-propagates. That is the same algorithm at
   interactive stakes: it sits between the user clicking a structure list and the next frame.

## Why the algorithm is GPU-shaped already

The CPU version was written as a **Jacobi** iteration — every voxel updates from its
neighbors' previous values, no sweep ordering — precisely because ordered fast-marching
does not vectorize. Jacobi is the GPU-natural form; nothing about the math changes.

The two stages:

- **Seeding.** For each of the 6 edge directions: does the argmax flip, and where on the edge
  does the signed deficit difference cross zero. Comparisons and gathers, branchless.
- **Propagation.** `n_iter = ceil(T/h) + 4` Godunov updates: per axis take the smaller
  neighbor, 3-element sort network, closed-form 1/2/3-axis quadratics. Branchless
  (`where`-selected), fixed iteration count, float32 throughout (MPS has no float64; the CPU
  version is float32 already, so parity holds).

## Do it dense on the GPU

The CPU band-restriction (14× there) should **not** be ported. It exists because CPUs are
slow, not because dense is wrong: 52 Mvoxel × 6 iterations × ~30 flops ≈ 10 Gflop plus ~5 GB
of memory traffic, which is milliseconds-to-tens-of-milliseconds on an L40S and well under a
second on an M2. Dense removes the band bookkeeping (dilation, flat indices, scatter-back)
and with it the only part of the CPU code that took thought to keep bit-identical.

## Replace scatter with gather

The CPU seeding scatters (`np.minimum.at`, and an atomic-min on a GPU). The GPU formulation
should invert it: **each voxel computes its own seed** as the min over its ≤ 6 incident
edges — look at each neighbor, if the argmax differs compute the crossing parameter from the
two deficits, take `t·h` on this side. Pure gather, no atomics, no race. (This is also worth
back-porting to the CPU if seeding ever dominates again: `minimum.at` is the slowest ufunc
in numpy.)

Deficit lookup per voxel is a scan over the ≤ 6 rank planes comparing against the wanted
class — a handful of gathers, same as the CPU's `_deficit_at`.

## Three targets, in order

1. **torch, CUDA (the Modal bake).** ~80 lines against ops that all exist (`roll`/slice
   compare, advanced indexing, `where`, `sqrt`). Hook: `ranked_emit_modal` produces the
   distance array next to `ranks`/`support` before anything leaves the device. This is the
   highest-value, lowest-risk step and should go first.
2. **torch, MPS (local bakes and the debug renderer).** The same code; the only op to verify
   on MPS is nothing exotic — the kernel set above is all elementwise. If a gap appears, the
   fused-restore precedent applies: the project already ships Metal source through
   `torch.mps.compile_shader`.
3. **WebGPU (sdfview, selection re-seeding).** Ping-pong compute shader: seed pass (gather
   formulation, membership flips from `ranks[0]`/`ranks[1]` against the selection bitset),
   then `n_iter` Godunov passes between two textures. The Jacobi form needs no atomics
   anywhere. Couples to sdfview's phase plan, not to this repo.

## Validation

Same suite, one relaxation. `tests/test_ranked_distance.py` (planar exactness, sphere vs
analytic distance — the case that catches taxicab propagation, labelmap-vs-runner-up seeding)
runs against any implementation. GPU float reassociation means byte-identity with the CPU is
not guaranteed; the acceptance bound is **max |difference| ≤ 1 quantum** on the encoded uint8
field, asserted store-wide, not eyeballed. The history that motivates this rigor: three CPU
attempts, two of which produced plausible images from wrong fields (folded-gradient divisor,
chamfer propagation), each caught only by a test against analytic truth.

## Non-goals

- No fast-marching / ordered solver. The band is 2 voxels; Jacobi converges in `n_iter`
  passes and parallelizes trivially. Priority queues buy asymptotics this problem never uses.
- No GPU band restriction, per above.
- No wider truncation to enable sphere tracing. Empty-space skipping stays `occupancy`'s job.
