# Handoff — resampling/orientation perf + the duckn orientation bug (2026-06-01)

Snapshot for compaction. Pairs with the auto-memory note
`project_rearch.md` (which has the running findings log) and
`docs/rearch-status.md` (the 0.10.x state).

## TL;DR

- **Root-cause win:** the "broken/striped/moth-eaten" segmentation renders on
  `ct.nii`/`chest.nii` were **not** our code — they were a **duckn NIfTI-export
  bug** that mis-oriented the volumes 180° about S (LPS values written into an
  RAS-tagged sform without conversion). The model is not orientation-equivariant,
  so it produced garbage on the flipped input. **Fixed in duckn**; header-fixing
  the inputs gave clean segs (ribs 22–24/24 intact, 0 lung holes).
- **Perf work (all in `nnunet-inference-mlx`):** moved the *entire non-network
  pipeline onto the GPU* — fused Metal restore kernel, a per-axis MLX
  resampler (anti-aliased clamped cubic down / linear up), and a GPU reorient
  (transpose+flip, replacing SITK `DICOMOrient`). End-to-end fast mode on chest:
  **75 → 57.6 s**, output **bit-identical** to the prior path.
- **Benchmark verdict:** MLX **wins fast mode** (preprocessing-bound: 57.6 vs
  75 s) but **loses full mode** (conv-bound: 240 vs 159 s) because **MLX's fp32
  `conv3d` is ~1.3× slower than PyTorch-MPS's** (MPSGraph). Not dtype/memory —
  see profiling below. Only safe lever is bf16 (~1.14×); gap is MLX-internal.

## ⚠️ UNCOMMITTED WORK (do not lose)

Nothing was committed this session. Two repos:

**`nnunet-inference-mlx`** (branch `feature/medseg-rearch`, last commit `eae9780`):
- `src/.../resampling.py` — fused restore Metal kernel (`_FUSED_ARGMAX_SRC` /
  `_FUSED_PAINT_SRC`, `inverse_resample_argmax` default `use_fused_kernel=True`);
  `resample_volume_mlx` + `_RESAMPLE_1D_SRC` (per-axis AA-cubic/linear, clamped);
  `reorient_array_mlx` + `_LETTER_DIR`/`_code_direction` (GPU reorient).
- `src/.../preprocess.py` — `to_model_frame` default `interpolation="auto"`
  (GPU reorient → GPU resample, no SITK `DICOMOrient`).
- `src/.../build.py` — `LoadedModel.predict/segment` default `interpolation="auto"`.
- `src/.../postprocess.py` — `restore` uses GPU reorient for the inverse
  (both nearest path A + linear path B).
- `tests/test_resampling.py` — `TestMlxForwardResampler`, `TestGpuReorient`,
  `TestFusedKernelEquivalence` (+ updated). **276 fast tests pass.**

**`duckn`** (branch `main`, last commit `5fa6eb9`):
- `src/duckn/nifti_convert.py` — **the orientation fix**: `zarr_to_nifti` now
  reframes the affine from the store's `space` to RAS+ before writing sform/qform
  (`_space_to_ras_signs`, negate axes opposite RAS). Was writing LPS verbatim
  into an RAS sform.
- `tests/test_nifti_convert.py` — updated `test_dicom_sourced_zarr_to_nifti` to
  assert the RAS reframe (it had encoded the bug). One **pre-existing** failure
  remains: `test_different_qform_sform_roundtrip` (qform-restore, unrelated).

(The `TotalSegmentator` fork at `~/Dropbox/development/total-segmentator/TotalSegmentator`,
branch `feature/mlx-backend`, lib pinned to `nnunet-inference-mlx@4771d79`, is the
repro env for upstream MLX-via-TS — not ours to commit; only an untracked `uv.lock`.)

## What we built / changed (detail)

1. **Fused Metal restore kernel** — trilinear+argmax inline, ~100×. This was
   COMMITTED earlier this session (`542d0d4` kernel, `eae9780` changelog). The
   `resampler + reorient + clamp` additions below are the UNCOMMITTED part (they
   sit on top of the committed kernel in the same `resampling.py`).
2. **Per-axis MLX resampler** (`resample_volume_mlx`): factor-scaled Catmull-Rom
   (anti-aliased) on downsampling axes, linear on upsampling/near-identity
   (no ringing on thick-slice through-plane). 0.54 s on 418 M voxels. `"auto"`
   default. Visually verified clean; sharp-edge no-ringing regression test.
3. **GPU reorient** (`reorient_array_mlx`): permutation+flips from direction
   cosines, applied via `mx.transpose` + negative-step slicing (MLX 0.31.1 has
   **no `mx.flip`**; `mx.contiguous` exists). Bit-identical to `sitk.DICOMOrient`
   across 6 inputs × 3 targets. Replaced the ~5.5 s of CPU DICOMOrient
   (fwd 3.2 + inv 2.3) with ~0.9 s GPU.

## Key facts established (don't relitigate)

- **Orientation is fully correct** — GPU reorient bit-exact vs SITK; SAR/RAS/SPL
  all validated; output bit-identical to the SITK path. The duckn bug was the
  *input* geometry, now fixed.
- **Forward cubic vs linear:** ~0 segmentation difference (99.97%); cubic on CPU
  is ~15 s in *any* lib (scipy/skimage/SITK) — that 14.7 s is exactly TS's
  `Resampled` time. Our GPU AA-cubic is 0.2 s.
- **TS inverse = path A** (argmax at model spacing, NN-upsample labels) in *both*
  fast and full (it pre-resamples the image to plans spacing, so nnU-Net's logit
  resample is a no-op). Our default is path B (logit interp) = higher fidelity.
- **Full-mode network gap:** MLX fp32 conv3d ~1.3× slower than MPS fp32 (autocast
  is CUDA-only → MPS is fp32). batch=1 optimal. bf16 safe ~1.14×; fp16 garbage.

## Repro / data

- Corrected inputs (canonical, header-fixed, bytes intact): `~/tmp/data/ct.nii`
  (SPL, uint16), `~/tmp/data/chest.nii` (SPL, uint16). `~/tmp/data/chest.zmp` =
  duckn ZMP (DICOM/LPS) for the source-of-truth geometry.
- Upstream TS: `~/.local/bin/TotalSegmentator` v2.13.0, `-d mps`.
- Outputs in `~/tmp/out/`: `chest_default_full.nii.gz` (our auto), `*_MPS_*`,
  `*_REALTS_*` (PyTorch truth), `resampler_zoom_final.png` (visual).
- Benchmarks (chest, fast): MLX 57.6 s (fwd 2.1, net 48, inv 2.3) vs MPS 75 s.
  Full (ct.nii): MLX 240 s vs MPS 159 s.

## Open / next

- **Commit** the two repos (separate, focused commits) — pending user go-ahead.
- Optional: wire bf16 as opt-in `use_dtype` for ~1.14× on the network (won't flip
  full-mode verdict).
- Optional micro-opt: union re-runs `to_model_frame` per part (5×); could resample
  once and share across the 5 same-spacing parts (small win — preprocessing is
  already 0.5 s/part).
- Full-mode network gap is MLX `conv3d` vs MPSGraph — not closable from our code.
- Update `CHANGELOG.md [Unreleased]` for the resampler + GPU reorient (fused
  kernel entry already there).
