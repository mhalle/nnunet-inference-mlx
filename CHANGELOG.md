# Changelog

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
