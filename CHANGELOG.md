# Changelog

## [0.8.2] - 2026-05-24

### Fixed — `transpose_forward` / `transpose_backward` handling

nnU-Net's training pipeline permutes input volume axes by `transpose_forward` from `plans.json` before patches reach the network. The model has only ever seen volumes in that transposed axis order. For models where `transpose_forward != (0, 1, 2)`, feeding the engine canonical-order volumes was producing silently-wrong predictions — same class of bug as the orientation issue in 0.8.1, just at a different axis-mapping layer.

For TS Datasets 291–298 these are all identity, so this fix has zero effect on TS workloads. For research nnU-Net models with non-identity transposes (e.g. `(2, 0, 1)` for axially-acquired thoracic models), this closes a real correctness hole.

### Added — bundle transpose properties

- **`ModelBundle.transpose_forward`** — read from `plans.json`, defaults to `(0, 1, 2)` when absent.
- **`ModelBundle.transpose_backward`** — same.
- **`ModelBundle.target_spacing`** now applies `transpose_backward` to the raw plans spacing, returning canonical-order spacing (matching MOOSE's convention). For identity-transpose models the behavior is unchanged.

### Changed — engine round-trip

`InferenceEngine.predict()` / `predict_logits()` / `predict_segmentation()` now:

1. Apply `transpose_forward` to the input volume (numpy `np.transpose`) before the sliding-window backend sees it
2. Run inference in the model's expected axis order
3. Apply `transpose_backward` to the output predictions (spatial axes; K dimension preserved at axis 0)

Caller-facing axis order is canonical `(Z, Y, X)` everywhere — both for inputs and for outputs. For identity-transpose models the round-trip is two no-ops; for non-identity models it makes the engine "transparent" so the caller doesn't need to know about the internal permutation.

`SlidingWindowEngine`, `Predictor`, and `FoldEnsemble` are unchanged — they remain raw primitives operating on whatever axis order they're given. The transpose handling lives in the `InferenceEngine` facade, matching the architectural rule: primitives do one thing, facades compose with metadata-awareness.

### Tests

14 new tests in `test_transpose_handling.py`. Total: **167 passing.**

Notable coverage:

- `test_target_spacing_non_identity_transpose` — verifies `bundle.target_spacing` returns canonical-order spacing
- `test_non_identity_round_trip_preserves_canonical_layout` — runs the engine with five different transpose pairs (identity, swaps, rotations, full reverse) on an asymmetric `(20, 24, 28)` input and verifies output shape is always `(K, 20, 24, 28)` — the engine is transparent regardless of the model's internal axis convention
- `test_apply_transpose_backward_undoes_forward` — tests the spatial-axis-only K-channel transpose helper directly

### Migration

For identity-transpose models (every TS model): no behavior change.

For non-identity-transpose models: existing callers passing canonical-order volumes get the right answer for the first time. Callers who were manually pre-transposing inputs to compensate for the bug should remove that compensation.

## [0.8.1] - 2026-05-24

### Fixed — canonical orientation handling in `predict_with_resampling` and `run_workflow`

Inputs with non-canonical direction matrices (oblique / reformatted scans where voxel axes don't align with anatomical axes) were silently producing badly fragmented segmentations: the sliding window scans along numpy axes assuming they map to canonical anatomical directions, but for a volume where, e.g., voxel-X = patient-superior, vertebrae stretched along the patient's cranio-caudal direction land as thin needles perpendicular to the sliding-window primary axis. Most patches saw only fragments.

Reproduced on a chest CT with SAR orientation (voxel X→S, Y→A, Z→R, spacing 1.0×0.65×0.65 mm). Before the fix:

| Task | FG voxels |
|---|---|
| Dataset291 (organs) | 16.3 M |
| **Dataset292 (vertebrae)** | **18.5 k** (almost entirely missed) |
| Dataset293 (cardiac) | 187 k |
| Dataset294 (muscles) | 5.46 M |
| Dataset295 (ribs) | 215 k |
| **Total FG** | **22.0 M (58 classes)** |

After the fix:

| Task | FG voxels |
|---|---|
| Dataset291 (organs) | 25.5 M (1.6×) |
| **Dataset292 (vertebrae)** | **2.30 M (124×)** |
| Dataset293 (cardiac) | 2.38 M (12.7×) |
| Dataset294 (muscles) | 11.3 M (2.1×) |
| Dataset295 (ribs) | 1.43 M (6.7×) |
| **Total FG** | **42.9 M (111 classes)** |

Inference itself ~26% faster aggregate — patches now contain anatomically coherent context so the model spends less compute on garbage paths.

### Changed — `predict_with_resampling` API

New keyword argument `reorient: str | None = "LPS"`. The function now:

1. Reorients the input image to canonical LPS via `sitk.DICOMOrient` before forward resample (if not already canonical),
2. Runs the full pipeline (forward resample → inference → inverse resample) in canonical orientation,
3. Reorients the output segmentation back to the caller's original orientation before returning.

Pass `reorient=None` to skip the round-trip when the caller knows the input is already canonical and wants to save the ~3–5 s reorient cost on huge volumes. Pass `reorient="RAS"` or any other DICOM-style orientation code to target a different canonical orientation.

For axis-aligned inputs (the typical case — clinical CTs in LPS, neuroimaging volumes in RAS) the reorient is a no-op and there's no behavior change. The fix only affects volumes with non-identity direction matrices.

### Changed — `run_workflow` reorient at workflow boundary

The orchestrator now reorients to LPS once at the workflow entry and reorients the final output back to the caller's orientation at exit. Each `Stage`'s internal `predict_with_resampling` call is told `reorient=None` (saving the per-stage reorient cost) since the workflow has already done it. Inter-stage crop bboxes are computed in canonical-orientation voxel coordinates, then composed in the same space.

### Tests

7 new orientation tests in `test_canonical_orientation.py`. Total: **153 passing**.

Notable: `test_sar_input_returns_sar_output` and `test_sar_cascade` explicitly verify that an input with SAR direction matrix produces output with the same SAR direction matrix after the LPS round-trip — geometry preserved through both the single-stage and cascade paths.

### Migration

Existing callers of `predict_with_resampling(engine, image)` and `run_workflow(image, stages)` need no changes — the reorient defaults to on. Callers who were already manually canonicalizing inputs can pass `reorient=None` to opt out.

## [0.8.0] - 2026-05-23

Small, single-purpose primitive additions that complete the mix-and-match toolkit story. No god-methods, no consolidation. Each addition does one thing and composes with what's already there.

### Added — engine bundle property accessors

Four read-only properties on `InferenceEngine` that surface metadata previously buried in `engine._bundle`:

- **`engine.target_spacing`** — `(Z, Y, X)` voxel spacing in mm the model expects as input. Callers resample to this spacing before predict.
- **`engine.has_regions`** — `True` for BraTS-style models with independent sigmoid heads, `False` for standard mutually-exclusive classes.
- **`engine.regions_class_order`** — paint-priority tuple of label values for region-based conversion. Empty for standard datasets.
- **`engine.bundle`** — the underlying `ModelBundle` (escape hatch for everything else in `dataset` / `plans` / `metadata`).

Pure delegations to `engine._bundle`. The curated names cover the routine cases; `engine.bundle` is for the rare reach-through.

### Added — `engine.predict_logits()` returning `mx.array`

The MLX-native sibling of `engine.predict()`. Returns the per-channel predictions as an `mx.array` in unified memory, ready to feed directly into `inverse_resample_argmax` / `inverse_resample_paint` / multi-model arithmetic without a per-caller `mx.array(...)` wrap. Same underlying data as `predict()`; same per-volume materialization cost (the sliding-window accumulator runs in numpy internally). The win is API ergonomics — the `mx.array(...)` ceremony moves once into the engine rather than every call site.

### Added — `inverse_resample_paint` (region-based scheme-aware inverse resample)

New primitive paired with `inverse_resample_argmax`. Same slab-streaming + K-channel trilinear-gather pass; differs only in the per-slab finishing step: per-region threshold (default 0.0 ↔ raw-logit "sigmoid > 0.5") + paint-priority overwrite from `regions_class_order`. Use for BraTS-style models where argmax across channels is silently wrong (it picks "the region with the highest sigmoid" instead of "all regions above threshold, painted by priority").

```python
seg = inverse_resample_paint(
    logits_target,
    out_shape_zyx=acq_shape,
    target_spacing_zyx=target_spacing,
    acq_spacing_zyx=acq_spacing,
    regions_class_order=engine.regions_class_order,
    threshold=0.0,   # 0.5 if input is post-sigmoid
)
```

Memory shape identical to `inverse_resample_argmax` (slab budget bounds the K-channel acquisition-spacing slab). Output dtype auto-picks from the maximum label value in `regions_class_order` — typically `uint8`.

### Added — `inverse_resample_argmax` accepts `mx.array | np.ndarray`

Polymorphic input. The caller-side `mx.array(numpy_logits)` wrap is internalized; the function accepts whatever logit-shaped tensor you have. Same one-time conversion cost when called with numpy, zero conversion when called with `mx.array`.

### Added — `resample_volume` (numpy forward resample primitive)

Pure-scipy sibling of `resample_image_to_target` (which takes a `sitk.Image`). For users composing numpy-native mix-and-match pipelines without an SITK dependency:

```python
vol_target = resample_volume(vol_acq, in_spacing_zyx=acq, out_spacing_zyx=target, order=3)
```

`order=3` (cubic) matches nnU-Net's training-time forward resample quality; pass `order=0` for label volumes. Adds an optional scipy dependency that's already commonly installed.

### Fixed — `predict_with_resampling` scheme dispatch for region-based models

Previously called `inverse_resample_argmax` unconditionally. For region-based models, the sigmoid-averaged ensemble output would get argmax'd — picking "highest sigmoid channel" instead of doing proper threshold + paint priority. Silently wrong at ambiguous voxels.

Now dispatches on `engine.has_regions`:

- Standard scheme → `inverse_resample_argmax`
- Region-based scheme → `inverse_resample_paint` with `regions_class_order=engine.regions_class_order` and `threshold=0.5` (multi-fold, post-sigmoid mean) or `threshold=0.0` (single-fold, raw logits)

The non-resample path (`engine.predict_segmentation`) already handled this correctly via `convert_logits_to_segmentation`; this brings the resample path into alignment.

### Tests

38 new pytest tests across three new files (`test_predict_logits_and_properties.py`, `test_resample_primitives.py`, `test_predict_with_resampling_scheme.py`), bringing total coverage to 146 tests, all passing.

Notable: `test_predict_with_resampling_scheme.py::test_region_based_dispatch_uses_paint` explicitly verifies the fixed BraTS bug — output labels must come from `regions_class_order` (`{0, 1, 2, 4}` in the test fixture), not channel indices (`{0, 1, 2, 3}` which would indicate argmax was incorrectly applied).

### Migration

`predict_with_resampling` callers don't need to change anything — the scheme dispatch is internal. Direct callers of `inverse_resample_argmax` who were passing region-based logits should switch to `inverse_resample_paint`.

Internal use of `engine._bundle.target_spacing` should migrate to `engine.target_spacing` (the underscore-prefixed access still works but is no longer the recommended pattern).

## [0.7.0] - 2026-05-23

### Added — pluggable weights-folder discovery (`WeightsLayout` + registry)

Different consumers of nnU-Net weights store them in slightly different places (vanilla nnU-Net under `$nnUNet_results`, TotalSegmentator under `$TOTALSEG_WEIGHTS_PATH` or `~/.totalsegmentator/nnunet/results`, MOOSE elsewhere). The `WeightsLayout` dataclass captures a layout as data — env var, default path, preferred trainer/plans/model naming — and a module-level registry walks layouts in order. Two built-in layouts cover nnU-Net and TS; downstream packages register their own.

- **`WeightsLayout`** — frozen dataclass: `name`, `env_var`, `default_path`, `trainer`, `plans`, `model`. `resolve_weights_dir()` returns the dir if the env var points at one, falls back to `default_path` if that exists, returns `None` otherwise.
- **`register_weights_layout(layout, *, prepend=False)`** — add a layout. Append by default; pass `prepend=True` to put it ahead of the built-ins.
- **`list_weights_layouts()`** — current lookup order.
- **`discover_weights()`** — first matching layout wins, returns `(path, layout)`. Raises `FileNotFoundError` with a diagnostic listing of what was tried when nothing resolves.

`ModelBundle.from_task()` now walks the registry when `weights_dir=None` and inherits the resolved layout's trainer preference automatically. New keyword args `trainer` / `plans` / `model` disambiguate Datasets that ship multiple trainer variants — needed for TS's `Dataset291` (both `nnUNetTrainer` and `nnUNetTrainerNoMirroring` under one Dataset folder).

### Added — process-wide engine cache (`engine_cache.py`)

Building an `InferenceEngine` from a model folder takes ~3–5 s on M2 base (read disk + build network + load weights + compile + warmup). For workflows that touch the same model many times — batch inference, multi-stage cascade, interactive UI — keeping the engine alive across calls eliminates that cost.

- **`cached_engine_from_folder(model_folder, *, configuration, folds, step_size, compile, batch_size, use_mirroring, ...)`** — high-level helper: build-and-cache or return-cached, keyed on everything that affects engine state.
- **`cached_engine_from_task(task_id, *, folds, trainer, plans, model, ...)`** — same, but resolves the folder via the `WeightsLayout` registry first. Drop-in replacement for TS-style `find_model_folder` + build-engine code.
- **`get_cached_engine(key)`** / **`cache_engine(key, engine)`** — low-level for callers managing custom keys (e.g. nnInteractive's session state).
- **`cache_enabled()`** — auto-tiers by detected unified memory (≥32 GB on by default; below disabled to avoid memory pressure from holding ~600 MB per cached engine). Override via the `NNUNET_MLX_CACHE_ENGINES` env var.
- **`clear_engine_cache()`** — release everything and clear Metal buffers; safe on empty cache.

Measured on TS Dataset297 fast task:
- Cold load: ~4.5 s
- Cache hit: ~3 ms (**~1500× speedup**)

Verbose / progress flags don't bust the cache; `step_size`, `use_mirroring`, `batch_size`, fold list, configuration, and folder all do.

### Added — multi-stage workflow orchestrator (`workflow.py`)

The MOOSE-style cascade pattern (low-res body detector → high-res organ model) and the generalization of TS's FOV-limited inference, as a first-class primitive. Each stage runs `predict_with_resampling`; between stages, the prior stage's output bbox optionally crops the next stage's input.

- **`Bbox`** — frozen dataclass for `(Z, Y, X)` voxel-coordinate boxes. `shape_zyx` / `slices` / `clamped()` / `dilated()` / `compose()` / `Bbox.full()`.
- **`compute_fg_bbox(labels, *, classes=None, dilation_mm=0, spacing_zyx=None)`** — find FG bbox of a label volume, optionally restricted to specific classes and dilated by a physical margin. Returns `None` when no FG is found, signalling "skip cropping" to workflow callers.
- **`crop_image(sitk_image, bbox)`** — extract a sub-volume preserving world-coordinate geometry (origin shifts to track the crop).
- **`paste_segmentation(small_seg, full_shape_zyx, bbox, *, fill=0)`** — paste a cropped-space label volume back into a full-shape canvas.
- **`Stage`** — `engine + crop_to_classes + dilation_mm + interpolation + peak_working_memory_mb + remove_small_components_mm3`. Default dilation is 10 mm.
- **`run_workflow(image_sitk, stages, *, verbose=False)`** — chain stages, crop input between stages where requested, paste-back at the end so output geometry matches input.

Two-stage cascade in user code:

```python
from nnunet_inference_mlx import Stage, run_workflow, cached_engine_from_task

body  = cached_engine_from_task(298, folds=0)              # low-res body detector
liver = cached_engine_from_task(291, folds=0,              # high-res organ model
                                trainer="nnUNetTrainerNoMirroring")

stages = [
    Stage(engine=body,  crop_to_classes=(BODY_TRUNK,), dilation_mm=10.0),
    Stage(engine=liver, remove_small_components_mm3=200.0),
]

seg = run_workflow(image_sitk, stages, verbose=True)
```

Composes naturally with the engine cache — each `Stage.engine` survives across workflow invocations. nnInteractive-style sub-volume re-runs use the same crop/paste primitives at finer granularity. The geometric primitives are exported as public API so callers building bespoke pipelines (FOV-limited inference, manual region updates) use the same building blocks.

### Fixed

- `compute_fg_bbox`: previously cast user-supplied class IDs to the labels array's dtype, which raised `OverflowError` for IDs outside the dtype's range (asking about class 999 on a `uint8` label volume crashed). Now filters out-of-range values up front and returns `None` when no in-range classes remain.

### Tests

97 new pytest tests across five new files (`test_weights_layout_registry.py`, `test_engine_cache.py`, `test_workflow.py`, `test_postprocessing.py`, `test_resampling.py`), all passing in ~3.7 s with no real weights or CT data required. Backfills coverage for the 0.6.0 resampling + postprocessing modules as well as the new 0.7.0 modules.

## [0.6.0] - 2026-05-23

### Added — spacing-aware resampling subsystem (`resampling.py`)

New module covering the forward + inverse resampling pieces that nnU-Net inference needs around model space. SimpleITK on the way in (CPU, B-spline/linear/nearest), MLX on the way out (Metal, K-channel trilinear with slab streaming). All resampling code is opt-in via the new `[preprocessing]` extra so the core package stays SimpleITK-free.

- **`resample_image_to_target(sitk_image, target_spacing_zyx, *, interpolation='linear')`** — forward resample an acquisition-spacing image to model target spacing via SITK. `interpolation` accepts `'linear'`, `'bspline'`, or `'nearest'`. Preserves geometry; output has the correct origin/spacing/direction.

- **`inverse_resample_argmax(logits_target, out_shape_zyx, target_spacing_zyx, acq_spacing_zyx, *, out_dtype=np.uint8, peak_working_memory_mb=None, cascade_downsample=False, verbose=False)`** — resample target-spacing K-channel logits to acquisition-spacing labels (path B: continuous-logit interpolation + argmax). Single Z-slab loop with explicit-coordinate trilinear over all K channels; slab depth auto-sized from `peak_working_memory_mb` so the K-channel slab fits the budget. When the whole output fits, equivalent to a one-shot materialize.

  `peak_working_memory_mb=None` (default) auto-detects from system RAM: 200 MB on `<32 GB` Macs, 2000 MB on `≥32 GB`.

  `cascade_downsample` (default `False`) enables a multi-step path for aggressive downsampling (source-ratio > 2× in any axis). Each step does a 2× K-channel downsample, preserving the continuous decision surface; final pass slabs to acquisition spacing. Trades more smoothing for less boundary aliasing; **not** a strict win on aggressive downsamples — small-vessel structures can vanish more than single-step preserves them. Documented with the trade-off in the docstring.

- **`predict_with_resampling(engine, image_sitk, *, interpolation='linear', peak_working_memory_mb=None, remove_small_components_mm3=0.0)`** — full path-B pipeline. Caller hands over a SITK image at any acquisition spacing, gets back a SITK image at the same spacing with integer labels. Forward resample on CPU/SITK, inference on Metal, inverse resample on Metal with slab+channel streaming, optional cc3d cleanup. K-channel logits are transient — never materialized at acquisition spacing.

  ```python
  import SimpleITK as sitk
  from nnunet_inference_mlx import InferenceEngine, predict_with_resampling

  img = sitk.ReadImage("scan.nii.gz")
  seg = predict_with_resampling(engine, img)              # default path B
  seg = predict_with_resampling(engine, img,
                                remove_small_components_mm3=200.0)  # + cleanup
  ```

Why path B: nnU-Net's standard inverse resampling is either nearest-neighbor on labels (aliased boundaries) or per-class one-hot resize (memory-prohibitive at K≥100). Trilinear interpolation of the K-channel logits with argmax-at-the-end gives smoother boundaries while remaining feasible on Apple Silicon — we slab-stream the K-channel inverse so the working set is bounded by a tunable budget (default ~200 MB on 16 GB Macs, 2 GB on 32 GB+). On unified memory the logits stay resident across the full pipeline without disk round-trips.

### Added — multi-label connected-component postprocessing (`postprocessing.py`)

New module exposing `remove_small_components` for dropping label islands below a physical-volume threshold. Backed by [cc3d](https://github.com/seung-lab/connected-components-3d) for the multi-label CC pass — two neighboring voxels are connected iff they share the same nonzero label, so disconnected pieces of the same class are filtered independently. Original label IDs are preserved.

- **`remove_small_components(labels, spacing_zyx, *, min_volume_mm3=200.0, connectivity=26, in_place=False)`** — drop components smaller than `min_volume_mm3`. Default threshold matches TotalSegmentator's `--remove_small_blobs` flag. Pass `0` for a no-op. Opt-in via the new `[postprocessing]` extra.

Measured on a 256×178×255 K=117 CT segmentation (`min_volume_mm3=665`):

| Method | Time | Speedup |
|---|---|---|
| `cc3d.dust` | **80 ms** | baseline |
| SITK `ScalarConnectedComponent + RelabelComponent + Mask` | 820 ms | 10× slower |
| scipy per-label loop (TS-style) | 7800 ms | 90× slower |

Wired into `predict_with_resampling` as a new `remove_small_components_mm3=0.0` keyword arg (off by default; pass `200.0` for TS-equivalent cleanup).

### Added — optional extras

- `[preprocessing]` = `["SimpleITK"]` — required for `resample_image_to_target`, `predict_with_resampling`, and any SITK-image I/O path.
- `[postprocessing]` = `["connected-components-3d"]` — required for `remove_small_components`.

Both raise a clear `ImportError` pointing users at the right `pip install` invocation when the underlying package is missing.

### Notes

- Path B inverse throughput is dominated by MLX's native trilinear+gather kernels (~11 ns / voxel-K on M2 base, ~3 ns / voxel-K on M1 Max). `peak_working_memory_mb` is a memory-bound knob, not a perf knob — slab size affects what fits, not how fast it runs.
- Slab boundaries are continuous: explicit-coordinate trilinear ensures interp values across slab borders match the values that would come from a single materialize pass. Earlier per-slab `mx.nn.Upsample` formulation had visible staircase artifacts in non-axial views; the current explicit-coordinate path is artifact-free.
- Output dtype on `predict_with_resampling` follows the bundle's label scheme: `uint8` for standard `K ≤ 255`, auto-widened for region-based / large-K datasets via `label_dtype`.

## [0.5.3] - 2026-05-22

### Changed — auto-tier Metal cache by detected unified memory

`Predictor` now accepts `cache_limit_fraction=None` (the new default) and auto-picks based on detected unified memory:

- **< 32 GB RAM**: 0.30 (unchanged from previous behavior; leaves room for sliding-window accumulators on constrained Macs).
- **≥ 32 GB RAM**: 0.50 (M1 Max / M3 Pro / Studio / Ultra). Maps to ~30 GB+ Metal cache on a 64 GB Mac — well under Apple's recommended GPU working-set size, but enough to keep compiled-graph buffers resident between forward passes instead of evicting and rebuilding for every patch.

Effect: small but consistent latency win on batch / multi-file workloads on big-RAM Macs. No change on 16 GB Macs. `self.cache_limit_fraction` is now a `Predictor` attribute callers can introspect to confirm what auto-detection picked. Explicit fractions still work as a hard override.

Why now: downstream consumers (TotalSegmentator's MLX backend in particular) want to ship engine-caching workflows that hold ≥5 InferenceEngines resident on big-RAM Macs. The previous 0.30 default sized the Metal cache for the constrained case and left ~50 GB unused on 64 GB systems, forcing more buffer churn than necessary in those flows.

## [0.5.2] - 2026-05-22

### Added — region-based label handling (BraTS-style models)

New `labels.py` module brings the LabelManager-equivalent post-processing piece that was the last gap vs `nnUNetPredictor`. Closes a silent-correctness hole: a BraTS-style checkpoint loaded into 0.5.1 and consumed via `np.argmax(engine.predict(vol), axis=0)` would have produced nonsense labels because the model's region heads are independent sigmoids, not a softmax distribution.

- **`labels.has_regions(dataset_json)`** — True if any label value is a list/tuple of underlying classes (the union form).
- **`labels.regions_class_order(dataset_json)`** — paint-priority tuple, empty for standard datasets.
- **`labels.label_dtype(dataset_json)`** — smallest unsigned integer dtype that fits every label value across both region member-class lists and `regions_class_order` output values. Returns `uint8` / `uint16` / `uint32` per nnUNetv2's convention.
- **`labels.convert_logits_to_segmentation(pred, dataset_json, threshold=0.0, dtype=None)`** — uniform post-processing. Standard datasets get `argmax`; region-based get threshold + paint-priority overwrite. Auto-detects scheme from `dataset_json`. `dtype=None` (default) auto-picks via `label_dtype`; pass explicit dtype to override.
- **`labels.sigmoid_inplace`** — float32-safe in-place sigmoid (matches `softmax_inplace` in style).

`ModelBundle` gains two derived properties:

- **`bundle.has_regions`** — True for BraTS-style datasets.
- **`bundle.regions_class_order`** — the paint-priority tuple.

### Added — `InferenceEngine.predict_segmentation()`

The one-call high-level path that returns integer labels directly. Handles all four (scheme × fold-count) combinations:

| Scheme | Single fold | Multi fold |
|---|---|---|
| Standard | logits → argmax | softmax-avg → argmax |
| Region-based | per-region logits → threshold + paint | sigmoid-avg → threshold + paint |

```python
seg = engine.predict_segmentation(volume)        # auto dtype
seg = engine.predict_segmentation(volume, dtype="uint16")  # explicit dtype
```

Auto-routes through the right scheme; auto-picks output dtype (the smallest unsigned int that fits every label value). Pass an explicit dtype to force a specific integer width regardless of what the dataset needs.

Also exposed: **`engine.label_dtype`** read-only property for introspecting what auto-detection would pick.

### Fixed — `Predictor.num_classes` for region-based datasets

Previously `Predictor.num_classes = len(dataset["labels"])` — which counts background plus all regions/labels indiscriminately. For region-based datasets this gave the wrong network output channel count (e.g. 4 instead of 3 for BraTS, where background is implicit). Now correctly counts foreground regions (`sum(k != "background")`) for region-based datasets while preserving the standard `len(labels)` rule for softmax datasets. Also handles bare-int size-1 regions like `"ET": 3` correctly — they count as a head, same as `"ET": [3]`.

### Changed — `FoldEnsemble` averaging

The fold-ensemble averaging branches on label scheme:

- **Standard** models (heads are softmax-related): softmax-then-average — unchanged.
- **Region-based** models (heads are independent sigmoids): sigmoid-then-average. Auto-set from `bundle.has_regions` when the facade builds the ensemble; can be forced via `FoldEnsemble(..., region_based=True)` for direct callers.

### Changed — `predict_nifti` default dtype

The `dtype` parameter on `nnunet_inference_mlx.io.predict_nifti` now defaults to `None` (auto-detect from dataset) instead of hardcoded `np.uint8`. Pass an explicit dtype to force one. Also internally routes through `engine.predict_segmentation` instead of a raw `np.argmax`, so region-based models produce correct labels through the NIfTI helper too.

### Migration

Purely additive; no API removals. Code calling `engine.predict()` + `np.argmax` continues to work unchanged for standard models. Code using region-based models that was silently broken in 0.5.1 (no public consumer that we know of) now works correctly via `engine.predict_segmentation()`.

## [0.5.1] - 2026-05-22

### Added — nnInteractive enabler set

Two small additive changes that close the last gaps for a third-party port (notably nnInteractive) to use `nnunet-inference-mlx` as its core instead of forking `model.py` / `weights.py`.

- **Nested `cfg["architecture"]` plans block.** `build_network_from_plans` now accepts the form
  ```json
  "architecture": {
    "network_class_name": "...ResidualEncoderUNet",
    "arch_kwargs":        {...}
  }
  ```
  emitted by modern nnUNetv2 trainers via the `dynamic_network_architectures` configuration manager. The flat `network_arch_init_kwargs` form (TS Dataset29x plans) still works. Lookup order: flat-top-level → flat-in-config → nested-in-config → "old plans" fallback. Pure additive, no behavior change on existing models.
- **`dtype=` weight cast on load.** `load_model_weights`, `load_checkpoint_with_metadata`, and `ModelBundle.from_folder` / `from_task` gain a `dtype` parameter. Pass `"float16"`, `"bfloat16"`, `"float32"`, or an `mx.Dtype` to cast on load. Default `None` preserves source precision. Verified 2.00× memory reduction on Dataset291 (125 MB → 62 MB). Callers using fp16 weights still need matching activation precision in the forward — this is a weight-cast convenience, not a full mixed-precision pipeline.

### Changed — internal cleanup

- **`predict_sliding_window_streaming` dropped** in favour of the simpler `predict_sliding_window` kernel. The streaming variant's rolling-Z accumulator was optimizing peak memory that isn't where TS's pressure actually sits (the 5-models-in-one-process Metal cache, addressed separately by `engine.close()`). `SlidingWindowEngine.predict` now calls the non-streaming kernel; ~231 lines removed from `inference.py`. Test suite runs ~15% faster on the same volumes. No public-API impact — the variant was never exported.

## [0.5.0] - 2026-05-22

### Added — layered inference architecture

The monolithic `InferenceEngine` is decomposed into four composable layers so downstream consumers (TotalSegmentator, MOOSE, nnInteractive, future ports) can pick what they need instead of inheriting concerns they don't.

- **`Predictor`** (Layer 2) — one compiled, weight-swappable MLX network. Owns `mx.compile`, warmup, Metal cache discipline. Exposes `forward(x)`, `reload_weights(w)`, public `.network`. Knows nothing about sliding windows, ensembling, or normalization.
- **`SlidingWindowEngine`** (Layer 3) — wraps a `Predictor` and adds Gaussian importance weighting, the streaming accumulator, the shape cache, and per-channel normalization. The whole-volume `(Z, Y, X) → (K, Z, Y, X)` path.
- **`FoldEnsemble`** (Layer 4) — orthogonal composable. Wraps either a `Predictor` (patch-level ensemble) or a `SlidingWindowEngine` (volume-level ensemble). Loops the bundle's fold weight dicts via `Predictor.reload_weights` between forwards and averages softmax. Single-fold wrap is a no-op cost.
- **`InferenceEngine`** survives as a thin back-compat facade — given a bundle, builds `Predictor → SlidingWindowEngine`, wraps in `FoldEnsemble` automatically when `len(bundle.fold_weights) > 1`. The 90% caller stays one line.

### Added — multi-fold bundle loading

- **`ModelBundle.from_folder(path, folds=…)`** and the same kwarg on `from_task`. Single union-typed parameter:
  - `folds=int` — single fold
  - `folds=Iterable[int]` — multi-fold ensemble
  - `folds="all"` (default) — auto-detect every `fold_*/` subdir
  
  The `"all"` default loads whatever is on disk, which works for single-fold release builds (e.g. TotalSegmentator) and multi-fold trained models (e.g. MOOSE) without the caller knowing upfront.
- **`ModelBundle.metadata: dict`** — first fold's non-weights checkpoint metadata (`init_args`, `trainer_name`, `inference_allowed_mirroring_axes`, …) captured during load. Used by `Predictor` to auto-detect the configuration name from `init_args["configuration"]` when the caller doesn't pass `configuration=`.
- **`ModelBundle.fold_ids: tuple[int, ...]`** — which folds were loaded, in order.
- **`load_checkpoint_with_metadata(folder, fold)`** in `weights.py` — returns `(mlx_weights, metadata)` for a single fold.
- **`discover_folds(folder)`** — list of int fold IDs present as `fold_*/` subdirs.

### Added — TTA mirroring

- **`use_mirroring: bool = False`** kwarg on `SlidingWindowEngine` and `InferenceEngine`. When enabled, flip-averages predictions along every spatial axis the trained model allows. Mirror axes auto-read from `bundle.metadata["inference_allowed_mirroring_axes"]`; if the model wasn't trained with mirroring (e.g. TotalSegmentator's `NoMirroring` variants), `use_mirroring=True` is silently a no-op. Default `False` for back-compat and because mirroring doubles cost per axis (3 spatial axes → 8 forwards per patch).
- **`ModelBundle.mirroring_axes`** read-only property — `tuple(metadata["inference_allowed_mirroring_axes"] or ())`. Removes a magic dict-key from consumer code.

This closes a latent gap on the MLX path: upstream `nnUNetPredictor` defaults TTA on when the model allows it, but TS's `mlx_predict.py` silently dropped its `tta=` kwarg. MOOSE relies on the same upstream default. Consumers can now opt in by passing `use_mirroring=True` (and should, on models trained with mirroring axes set).

### Added — `Predictor` parameters for future consumers

- **`num_input_channels=None`** override on `Predictor` and `InferenceEngine`. When `None`, derives from `dataset["channel_names"]` as today; nnInteractive passes `8` explicitly (1 image + 7 interaction channels).
- **`configuration=None`** auto-detect — order is explicit arg → `metadata["init_args"]["configuration"]` → the only configuration in plans → `"3d_fullres"` fallback. Lets nnInteractive's non-default `"3d_fullres_ps192_bs24"` config work without explicit caller knowledge.
- **`cache_limit_fraction=0.3`** parameter on `Predictor` — was previously hard-coded.

### Changed — breaking API

- **`ModelBundle.weights` → `ModelBundle.fold_weights`**. Singular dict becomes a list of dicts (length 1 for single-fold bundles). Known consumers (this package, TotalSegmentator's `mlx_predict.py`) never read the attribute directly, so impact is limited to anyone constructing a bundle by hand. Test fixtures updated.
- **`InferenceEngine.predict()` return type depends on bundle shape**: single-fold → raw logits (unchanged behavior); multi-fold → softmax-averaged probabilities (the standard nnU-Net ensemble convention). `np.argmax(axis=0)` on either output yields the segmentation, so argmax-only callers don't need to branch.
- **`InferenceEngine.__init__(configuration=...)`** default changed from `"3d_fullres"` to `None` (auto-detect). Explicit pass-through still works; `None` now reads the configuration from the checkpoint's `init_args`.

### Migration

- Callers of `ModelBundle.from_folder(path, fold=N)` must rewrite as `folds=N`. The `fold=` keyword no longer exists.
- The default behavior changed: `from_folder(path)` with no fold args now loads *all* available folds, not just fold 0. For TotalSegmentator-style single-fold release builds this is a no-op (there's only one fold to load). For multi-fold trained models the engine now auto-ensembles. Pass `folds=0` explicitly if you want fold-0-only on a multi-fold model.
- Code constructing `ModelBundle(plans, dataset, weights=...)` directly must pass `fold_weights=[weights]` instead. Engine/facade users (the 90% case) need no changes.
- Code that wants raw single-patch forward without sliding-window scaffolding (e.g. nnInteractive-style) should build `Predictor` directly: `Predictor(bundle).forward(patch)`. Skips Layers 3-4.

## [0.4.0] - 2026-05-22

### Added
- Vendored torch-free `.pth` loader at `src/nnunet_inference_mlx/_torchfree/` (`torchfree_load.py`, `rangefile.py`). Reads zip-format PyTorch checkpoints (>= 1.6) into numpy via a restricted unpickler — only storage rebuild symbols, `OrderedDict`, and numpy reconstructors are allowed. Materializes only the `network_weights` subtree, so optimizer state and other large storages are never read.
- `load_pth_url` and `smart_load_url` for HTTP-range remote loading of `.pth` checkpoints. Opt-in via the new `remote` extra (`pip install nnunet-inference-mlx[remote]` → `requests`, `remotezip`).
- NIfTI I/O helpers: `load_nifti_zyx`, `save_segmentation_zyx`, `predict_nifti`, `predict_folder`.
- `InferenceEngine.close()` plus context-manager / `__del__` support. Clears the MLX Metal cache between runs — needed for TotalSegmentator's full mode, which runs inference 5× in one process and previously OOM'd without explicit buffer release.

### Changed
- `load_model_weights` now reads `<fold_dir>/<checkpoint_name>` directly via the vendored loader. TotalSegmentator release `.pth` files load with no conversion step and no PyTorch import.
- Runtime dependencies trimmed to `mlx>=0.25`, `numpy`, `tqdm`. `safetensors` is no longer required.

### Removed
- **Breaking**: the entire safetensors load/convert pipeline. Removed public symbols `load_weights_safetensors`, `convert_pth_to_safetensors`, `convert_model_folder`, and the `WEIGHT_LAYOUT_TORCH` constant.
- **Breaking**: `convert_weights_cli` and the `nnunet-inference-mlx-convert` console script entry in `[project.scripts]`.
- **Breaking**: `ModelBundle.from_task(auto_convert=...)` parameter (auto-conversion no longer exists).
- `safetensors` from required dependencies.
- `convert = ["torch"]` optional extra (no path through the package needs torch anymore).

### Migration
- The typical inference workflow (`predict_nifti`, `InferenceEngine`, `ModelBundle.from_task(task_id)`) is unchanged — `.pth` files in `~/.totalsegmentator/nnunet/results/...` load directly.
- Code that imported any of the removed symbols, called the CLI, or passed `auto_convert=` must be updated. There is no migration shim.
- Users who deleted `.pth` files to keep only `.safetensors` siblings need to re-fetch: `rm -rf <model_folder> && totalseg_download_weights -t <task>`.

### Performance
- Cold `.pth` load (Dataset291 fold_0, M2 17 GB): ~162 ms — *faster* than the previous safetensors path's ~848 ms, because the torchfree reader skips the optimizer-state subtree (reads ~125 MB of network_weights, not the full 354 MB safetensors file).
- Vendored loader micro-optimization: removed a redundant `.copy()` in `_LazyStorage.array` that previously memcopied every storage. Downstream consumers (`mx.array`, `np.moveaxis + ascontiguousarray`) copy into their own buffers, so the upstream copy was wasted. ~12 ms saved per fold on warm cache; arrays returned from `load_pth` are now read-only views into the underlying zip bytes.

## [0.3.1] - 2026-04-07

### Added
- `progress: bool = False` parameter on `InferenceEngine` and on the three sliding-window functions (`predict_sliding_window`, `predict_sliding_window_streaming`, `predict_sliding_window_segmentation`). When `True`, a tqdm progress bar is shown for each patch processed during inference. Mirrors the equivalent bar that nnUNetPredictor shows on the PyTorch path.
- `tqdm` is now a runtime dependency (small, pure Python).

### Fixed
- The MLX inference path was missing a per-patch progress bar that the PyTorch/MPS path through `nnUNetPredictor` has always shown. Long inference runs now have visible progress feedback when the caller passes `progress=True`. The corresponding `mlx_predict.py` wrapper in TotalSegmentator now passes `progress=not quiet` to enable the bar by default.

## [0.3.0] - 2026-04-07

### Added
- `ModelBundle` and `InferenceEngine` are now separate classes — bundles hold weights/plans/dataset, engines hold inference-time configuration. Construct each independently or use `InferenceEngine(ModelBundle.from_task(...))`.
- Streaming sliding-window accumulator: rolling Z-direction buffer keeps memory bounded for large volumes. Skipped automatically when the volume fits in a single accumulator.
- `softmax_inplace` helper for converting logits to probabilities without an extra copy.
- `convert_pth_to_safetensors` public helper for one-shot conversion of legacy PyTorch checkpoints to the canonical layout.

### Changed
- **Adopt the nnU-Net canonical safetensors layout as the only on-disk format.** Files now live at `<base>.safetensors` (PyTorch-layout tensors with a `weight_layout=torch_ncdhw` metadata header), matching what `nnUNetTrainer` writes natively after the upstream safetensors PR. Models trained with new nnU-Net drop onto a Mac and load with no conversion step.
- The loader transposes conv weights at load time using `safetensors.numpy` (no torch round-trip). Runtime stays torch-free.
- `convert_model_folder` and the convert CLI now write the canonical layout directly. Output is byte-identical in shape and tagging to `nnUNetTrainer`'s native output.
- Package renamed from internal references to `nnunet-inference-mlx`; TotalSegmentator-specific defaults and hardcoding removed from `ModelBundle` and `InferenceEngine`. The package no longer assumes any particular weights directory.
- `nnUNet_results` environment variable is consulted before falling back to TotalSegmentator's default location.
- Metal cache limit set to 30% of system RAM by default for better large-volume behavior.

### Removed
- Legacy `<base>_mlx.safetensors` (MLX-pre-transposed) format. Existing files become orphaned and can be deleted; auto-conversion handles re-generation from `.pth` on first call.
- `save_weights_safetensors` and the `WEIGHT_LAYOUT_MLX` constant (unused after the rewrite).
- Hardcoded `Task` enum for TotalSegmentator model IDs.
- Source-layout fallback hint on `load_weights_safetensors`. The loader now requires the metadata header and rejects untagged files with an actionable error.

### Fixed
- Tests in `test_engine.py` use plain asserts instead of `return bool`, eliminating `PytestReturnNotNoneWarning`.

### Migration
- Pre-existing `_mlx.safetensors` files are no longer read. Delete them and let `ModelBundle.from_task` auto-convert from `.pth` on the next call (one-time torch dependency at conversion time only).
- Code that imported `save_weights_safetensors` should switch to `convert_pth_to_safetensors`.

## [0.2.0] - 2026-04-03

### Changed
- Replace custom `_AvgPool3d` with built-in `mlx.nn.AvgPool3d` in residual encoder blocks.
- Bump minimum MLX version from 0.22 to 0.25 (adds native 3D pooling, 3D conv speedups).

## [0.1.0] - 2025-04-14

Initial release: MLX inference backend for nnU-Net on Apple Silicon.
