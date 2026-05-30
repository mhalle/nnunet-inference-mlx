# Rearchitecture — status & handoff

**Branch:** `feature/medseg-rearch` (off `main`). **Tests:** 326 passed, 1 skipped,
2 deselected (`uv run pytest -m "not slow"`, ~7s). All work committed.

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
Single dispatches natively (`store.load(id).segment`); **cascade & union bridge to
the proven `run_workflow`/`run_label_union_workflow`** via `store.load(id)._engine`
+ Volume↔SITK (the migration seam). Nested `crop_from_task` cascades resolve
through the catalog. All exercised with synthetic models (real build, no real
weights, no GPU needed beyond MLX) + SITK.

## Module ↔ test map (all committed, green)

| module | tests | status |
|---|---|---|
| `values.py` | `test_values.py` (26) | done |
| `model_data.py` + `store.py` | `test_store.py` (25) | done |
| `build.py` (`build_model`/`LoadedModel`/`predict`/`segment`) | `test_build.py` | done (spine) |
| `imageio.py` (Geometry↔SITK, Nifti/Dicom/Array readers, writer) | `test_imageio.py` | done |
| `catalog.py` (`TaskCatalog`) | `test_catalog.py` | done |
| `segment.py` (`segment()` dispatch) | `test_segment.py` | done |

## Hidden globals eliminated

- old module-global engine `_CACHE` → owned `ModelStore` (explicit, bounded by
  `max_memory_mb`, freeable). **Old one still exists in `engine_cache.py`** — deleted at cutover.
- old module-global task `_REGISTRY` (in `tasks.py`) → owned `TaskCatalog`. **Old one
  still exists** — deleted at cutover.

## What's NEXT

### Phase 3b — decompose compute into Volume-native pure-fn namespaces (task #24)
The careful one (touches the geometry glue that has historically caused bugs:
orientation, transpose, resample, region/argmax, sub-voxel inverse).
- `preprocess`: `reorient`/`permute_axes`/`resample`/`normalize` over `Volume`,
  plus `to_model_frame(volume, model_data) -> (Volume, RestorePlan)`.
- `infer`: `sliding_window(loaded_model, volume) -> Prediction` (already have
  `LoadedModel.predict` returning a `Prediction` at training spacing — extend so it
  *also* yields/accepts a `RestorePlan` for the inverse).
- `postprocess`: `to_labels(prediction) -> Segmentation` (argmax / region-paint),
  `resample_prediction`/`restore(result, plan) -> Segmentation` (inverse-resample,
  incl. sub-voxel logit path), `drop_small_components`.
- geometry: `bbox_of_labels(seg, labels, margin_mm) -> Box`, `crop(volume, box)`,
  `paste(canvas, patch, box)` — Volume/Segmentation-native (the old workflow.py
  versions are SITK/numpy).
- Then **re-express `segment` as `predict → to_labels → restore`**, and **re-express
  cascade/union over these namespaces** so `segment.py` stops bridging to the old
  `run_workflow`/`run_label_union_workflow`. Make `step_size`/`use_mirroring`
  true per-call args to `infer` (they're baked at engine construction today; this
  is the rehome that frees them — see build.py TODO).
- Reuse the existing proven primitives in `resampling.py` (`resample_image_to_target`,
  `inverse_resample_argmax`/`inverse_resample_paint`, `reorient`, `get_orientation`)
  and `preprocessing.py` (normalization) — wrap them Volume-native, don't rewrite.

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
  - update the old tests that exercise deleted symbols.
- Decide: keep `InferenceEngine` as a private compute core, or fold fully into `build`.

### Deferred / open (not blocking)
- `EngineOptions`→`BuildOptions` split note: `step_size`/`use_mirroring` are passed to
  `build_model` as kwargs for now (engine bakes them); make per-call in 3b.
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

1. `git checkout feature/medseg-rearch`; `uv run pytest -m "not slow"` (expect 326).
2. Pick Phase 3b (decomposition) or Phase 5 (cutover). 3b makes 5's deletions clean
   (removes the bridge to old workflow first); doing 5 first is possible but leaves
   the cascade/union bridge in place.
3. Keep committing per-step; keep the old API working until the Phase 5 cutover.
