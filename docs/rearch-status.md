# Rearchitecture — status & handoff

**Branch:** `feature/medseg-rearch` (off `main`). **Tests:** 344 passed, 1 skipped,
2 deselected (`uv run pytest -m "not slow"`, ~7s). All work committed. **Phase 3b
is done** (decomposition); only **Phase 5** (cutover/delete) remains.

Read `docs/architecture-rearch.md` for the full target design + rationale. This
file is the *where-we-are / what's-next* handoff.

## What this rearchitecture is

Building the 1.0 toolkit API **alongside** the old one (which still works:
validated inference, the old tests still pass), then cutting over and deleting
the old surface. Governed by: composable toolkit, **no hidden state**,
format/source-agnostic plug-ins, IO separated from compute. The new modules
coexist with the old; the new API is **not wired into `__init__.py` yet** (that's
the Phase 5 cutover — deliberately avoiding two public surfaces mid-migration).

## Commits on the branch (oldest→newest)

```
88a56a6 phase 1: frozen value types + architecture doc
7ef65c8 phase 2: ModelArtifact + ModelStore (read-through, owned)
6e21e51 rename ModelArtifact -> ModelData
7c0b9da inline origin onto ModelData; LoadedModel + build_model
0fd983e phase 3 (spine): build_model -> LoadedModel + image IO
5084dca split EngineOptions -> BuildOptions (cache key) + run knobs
b1facf9 phase 4: TaskCatalog (explicit, no global) + run()
540b9b8 rename run() -> segment()
6fef99f Probabilities -> Prediction; add LoadedModel.predict (logits first-class)
dc13f26 status/handoff doc
56f8ef6 phase 3b: Volume-native geometry namespace (Box/bbox_of_labels/crop/paste)
76d5f11 phase 3b: preprocess/infer/postprocess namespaces + RestorePlan refit
0fb0e9a phase 3b: re-express segment/cascade/union over namespaces (drop workflow bridge)
```
(`main` also has unpushed `bdf216d` = MOOSE engine resolution + nested cascade,
the 0.9.3 groundwork, plus the 0.9.0–0.9.2 line; tags v0.9.0/v0.9.1/v0.9.2 are
local-only. Nothing pushed to origin since v0.9.0.)

## The settled vocabulary (every name reviewed in conversation)

Three user nouns + a one-liner; "engine" never appears at the user level.

| name | what it is | access |
|---|---|---|
| `TaskCatalog(ecosystem)` | name → recipe (frozen) | `catalog["total_fast"]`; bare resolves when unique, else qualify `"ts:total"` (`AmbiguousTaskError`) |
| `ModelStore(ecosystem, model_root_dir=, max_memory_mb=)` | id → model; read-through (download/load layers); owned, bounded, freeable | `get(id)→ModelData` (cold), `load(id)→LoadedModel` (hot) |
| `segment(task, image, *, store, catalog=)` | the headline one-liner → `Segmentation` | dispatches single/cascade/union |

Value types (`values.py`, frozen):
- `Geometry` (spacing_zyx, shape_zyx, origin_xyz, direction_xyz; hashable)
- `Volume` (channels-last `(Z,Y,X,C)` + Geometry + channel names)
- `Segmentation` (`(Z,Y,X)` int + Geometry + LabelSchema)
- `Prediction` (`(K,Z,Y,X)` float + Geometry + LabelSchema + `activation ∈ {logits,softmax,sigmoid}`) — **was `Probabilities`; renamed because single-fold output is raw logits, not probabilities**
- `RestorePlan` (source geometry/orientation/axis-perm/model-spacing — the inverse recipe, returned not hidden)
- `LabelSchema` (names + regions + paint_priority; `from_dataset_json`)
- `BuildOptions` (configuration, folds, batch_size, compile, dtype — **build identity = the store's cache key**; `step_size`/`use_mirroring` are NOT here — they're run knobs)

`ModelData` (`model_data.py`): cold model — plans/dataset/fold_weights + inlined
`ecosystem`/`id`/`version` (Provenance was inlined). Derived: schema,
target_spacing_zyx (transpose-applied), patch_size_zyx, num_input_channels,
num_folds, weights_mb. `read_folder()` delegates to old `ModelBundle.from_folder`
during migration.

`LoadedModel` (`build.py`): runnable form. `predict(volume)→Prediction` (native
output at training spacing, via existing `predict_logits`), `segment(volume)→
Segmentation` (full path via `predict_with_resampling`), `.memory_mb`, `.close()`,
context manager. Built by `build_model(model_data, options=BuildOptions(), *,
step_size=, use_mirroring=)`.

### Naming decisions made (don't relitigate)
- `ModelArtifact`→`ModelData`; `Provenance` inlined; `CompiledModel/Engine`→`LoadedModel`;
  `build_engine`→`build_model`; `EngineOptions`→`BuildOptions` (+ split out run knobs);
  `run`→`segment`; `Probabilities`→`Prediction`; `dir`→`model_root_dir`; `keep`→`max_memory_mb`;
  store verbs `download/downloaded/delete_downloads` (disk) + `load/unload/unload_all/loaded/loaded_mb` (memory);
  catalog access by name, store/cache by id; `NnUNetModels`/`MooseModels` were the
  *format* axis with access (`Http`/`LocalDir`) as a separate injected `Location` — but
  in the built `ModelStore` this is currently simplified to `ModelStore(ecosystem,
  model_root_dir=)` (the format×location split is documented in architecture-rearch.md
  as the extensibility seam, not yet fully built).

## What works end-to-end NOW (new API)

```python
from nnunet_inference_mlx.catalog import TaskCatalog
from nnunet_inference_mlx.store import ModelStore
from nnunet_inference_mlx.imageio import NiftiReader, NiftiWriter
from nnunet_inference_mlx.segment import segment

catalog = TaskCatalog("totalsegmentator")            # 51 tasks, explicit, no global
store   = ModelStore("totalsegmentator", model_root_dir="/data/ts", max_memory_mb=4000)
image   = NiftiReader().read("ct.nii.gz")            # or DicomReader / ArrayReader
seg     = segment("total_fast", image, store=store, catalog=catalog)
NiftiWriter().write("seg.nii.gz", seg)
```
All three shapes are expressed over the toolkit stages (Phase 3b): single =
`LoadedModel.segment` (itself `to_model_frame → sliding_window → restore`);
**cascade** = `model.segment` per stage + `geometry.bbox_of_labels/crop/paste`;
**union** = `model.segment` per part + `labels.remap_labels/paint_union`. No more
bridge to `run_workflow`/`run_label_union_workflow` — they're untouched by the new
path and deleted at Phase 5. Nested `crop_from_task` cascades resolve through the
catalog. All exercised with synthetic models (real build, no real weights, no GPU
needed beyond MLX) + SITK.

## Module ↔ test map (all committed, green)

| module | tests | status |
|---|---|---|
| `values.py` | `test_values.py` (26) | done |
| `model_data.py` + `store.py` | `test_store.py` (25) | done |
| `build.py` (`build_model`/`LoadedModel`/`predict`/`segment` — compose namespaces) | `test_build.py` | done |
| `imageio.py` (Geometry↔SITK, Nifti/Dicom/Array readers, writer) | `test_imageio.py` | done |
| `catalog.py` (`TaskCatalog`) | `test_catalog.py` | done |
| `segment.py` (`segment()` dispatch; cascade/union over namespaces) | `test_segment.py` | done |
| `geometry.py` (`Box`/`bbox_of_labels`/`crop`/`paste`, Volume-native) | `test_geometry.py` (12) | done (3b) |
| `preprocess.py` (`reorient`/`resample`/`to_model_frame`) | `test_decompose.py` | done (3b) |
| `infer.py` (`sliding_window` → `Prediction`) | `test_decompose.py` | done (3b) |
| `postprocess.py` (`to_labels`/`restore`/`drop_small_components`) | `test_decompose.py` (regression oracle) | done (3b) |

## Hidden globals eliminated

- old module-global engine `_CACHE` → owned `ModelStore` (explicit, bounded by
  `max_memory_mb`, freeable). **Old one still exists in `engine_cache.py`** — deleted at cutover.
- old module-global task `_REGISTRY` (in `tasks.py`) → owned `TaskCatalog`. **Old one
  still exists** — deleted at cutover.

## Phase 3b — DONE (decomposition into Volume-native pure-fn namespaces)

The careful one (touches the geometry glue that has historically caused bugs).
Landed across commits `56f8ef6` / `76d5f11` / `0fb0e9a`:
- `geometry.py`: `Box`, `bbox_of_labels(seg, classes, dilation_mm)`,
  `crop(volume, box)` (shifts world origin to the cropped corner — validated
  against SITK `RegionOfInterest` incl. oblique directions), `paste(patch,
  canvas_geometry, box)`.
- `preprocess.py`: `reorient(volume, code)`, `resample(volume, spacing)`,
  `to_model_frame(volume, model_data) -> (Volume, RestorePlan)` (reorient to
  canonical + resample to model spacing).
- `infer.py`: `sliding_window(loaded_model, volume, *, step_size, use_mirroring)
  -> Prediction`. The engine normalizes + transposes internally, so neither
  `to_model_frame` nor the `RestorePlan` touch axis permutation.
- `postprocess.py`: `to_labels(prediction)` (same-grid argmax/paint),
  `restore(prediction, plan)` (inverse-resample logits to source grid,
  scheme-aware, then reorient back — the high-quality logit-resample path),
  `drop_small_components`.
- `LoadedModel.predict/segment` now compose these; cascade re-expressed over
  `geometry`, union over `labels` — **no `workflow` bridge on the new path**.
- `RestorePlan` refit: dropped unused `axis_permutation`; added
  `inference_geometry` / `inference_orientation`.
- **Regression oracle** (`test_decompose.py`): `to_model_frame → sliding_window
  → restore` is asserted *bit-identical* to the old fused `predict_with_resampling`
  for both a canonical (LPS) and a reoriented (RAS) volume. This is the safety net
  that let the geometry glue be restructured without behavior drift.
- Caveat — **`step_size`/`use_mirroring` are threaded but not yet truly free**:
  `infer.sliding_window` applies them by temporarily overriding the loaded model's
  `engine.sliding_window` for the call (save/set/restore). The real fix (baking
  removed) is the engine rehome in Phase 5.

## What's NEXT

### Phase 5 — cutover + delete old surface (task #23)
- Wire the new API into `__init__.py` (export `TaskCatalog`, `ModelStore`, `segment`,
  `predict`?, `build_model`, `LoadedModel`, `ModelData`, value types, readers/writers).
- Move `ModelBundle.from_folder` folder-reading orchestration INTO
  `ModelData.read_folder` (it currently delegates to `ModelBundle`); then delete:
  - `engine_cache.py` globals: `_CACHE`, `cache_enabled`, `get_cached_engine`,
    `cache_engine`, `cached_engine_from_folder/_task/_moose_model`, `resolve_moose_config_folder`
  - `engine.py`: `WeightsLayout` + `_WEIGHTS_LAYOUTS` + `register_weights_layout` +
    `discover_weights` + `_find_model_folder`; `ModelBundle.from_dataset/from_task`
    (resolution leaves the artifact). Keep the compute internals (Predictor/
    SlidingWindowEngine/FoldEnsemble) — rehome under `build`/`infer`, don't rewrite.
  - `tasks.py`: the module-global registry (`_REGISTRY`, `register_task`, `get_task`,
    `run_named_task`, etc.) — `TaskSpec`/`CascadeStep`/`UnionPart`/`_taskspec_from_dict`/
    `AmbiguousTaskError` SURVIVE (reused by catalog.py); decide whether to rename
    `TaskSpec`→a recipe name or keep.
  - `workflow.py` (`Bbox`/`Stage`/`ParallelStage`/`run_workflow`/`run_label_union_workflow`/
    `compute_fg_bbox`/`crop_image`/`paste_segmentation`) and `resampling.predict_with_resampling`:
    **no longer used by the new path** (3b replaced them with `geometry`/`preprocess`/
    `postprocess`). Decide at cutover: delete, or keep as a public SITK-level convenience.
    Keep the lower-level `resampling` primitives (`resample_image_to_target`,
    `inverse_resample_*`, `reorient`, `get_orientation`) — the new namespaces wrap them.
  - update the old tests that exercise deleted symbols (`test_workflow*`, the
    `predict_with_resampling` regression assert in `test_decompose.py` once the oracle
    is retired).
- Decide: keep `InferenceEngine` as a private compute core, or fold fully into `build`.
  When rehoming, fold `step_size`/`use_mirroring` into the per-call sliding-window path
  so `infer.sliding_window` stops the temporary engine-attr override (see 3b caveat).

### Deferred / open (not blocking)
- MOOSE: the `ModelStore("moose", ...)` path + nested cascade + string ids all exist;
  remaining MOOSE-compat (the `refresh_moose_registry.py` generator, a downloaded-model
  verification, possible RescaleTo01/use_mask_for_norm normalization) is the old
  0.9.3 roadmap, independent of this rearch.
- format×location plug-in split (`Location`: LocalDir/Http) — documented in
  architecture-rearch.md; `ModelStore` currently takes `model_root_dir` directly.
  Remote download (`store.download` from URL) is stubbed (raises) — local only.
- A `segment` lacks a top-level `predict(task, image) -> Prediction` sibling; only
  `LoadedModel.predict` exists. Add the top-level sibling when convenient.
- Suggested release framing once cutover lands: this is a 1.0 (breaking) API.

## How to resume

1. `git checkout feature/medseg-rearch`; `uv run pytest -m "not slow"` (expect 344).
2. Phase 3b is done. Next is **Phase 5** (cutover): wire `__init__.py`, move
   folder-reading into `ModelData.read_folder`, then delete the old surface
   (`engine_cache` globals, `WeightsLayout`, `ModelBundle.from_dataset/from_task`,
   `tasks.py` global registry, and `workflow.py`/`predict_with_resampling` now that
   nothing on the new path uses them).
3. Keep committing per-step; keep the old API working until the Phase 5 cutover.
