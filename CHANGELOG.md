# Changelog

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
