# Rearchitecture: the toolkit API (1.0 target)

This captures the API design settled in the long design conversation, and the
phased plan to get there. Governing principles (in priority order):

1. **Composable toolkit.** Every stage is an independently callable, swappable
   function over first-class values. No god-methods.
2. **No hidden state.** No module-level mutable globals, no implicit disk
   writes, no behavior that silently changes on env/RAM detection. All state is
   explicit, caller-owned, inspectable, and freeable.
3. **Format/source-agnostic, extensible.** Model sources and image formats are
   plug-ins; adding one is a small new implementation, never a near-duplicate
   top-level function or a core edit.
4. **IO separated from compute.** Reading artifacts (bytes/config/weights) is
   distinct from allocating GPU state — kept distinct, but the shapes are ours
   to choose.
5. Pre-1.0; Python 3.10+; MLX backend. No backward-compat obligations.

The predecessors (nnU-Net, TotalSegmentator) are what we're correcting: they
wrote hidden temp files mid-pipeline, assumed env-var directories, hardcoded
one on-disk layout, and couldn't accept arbitrary image formats or run purely
in-memory. None of that here.

---

## The user-facing vocabulary (three nouns)

A novice meets **`segment(task, image)`** and nothing else. Beyond that, the whole
surface is three nouns, each named for what it holds and how you access it:

| noun | holds | access | analogy |
|---|---|---|---|
| **TaskCatalog** | tasks (recipes) | by **name** (`catalog['total_fast']`) | the menu |
| **ModelStore** | models | by **id** (`store.get(297)`) | the supplier |
| *(run)* | — | orchestrates | the kitchen |

- A **catalog** is for *discovery* — human names you browse/look up; you don't
  know the id yet.
- A **store** is for *retrieval* — you (or a task) already have the id; fetch it.

The **task is the translator**: humans speak names → `TaskCatalog`; the machine
speaks ids → `ModelStore`. A task recipe is the one thing that maps name → ids.

### Readiness, not separate cache objects

A model in the store has a **readiness**, expressed as verbs, not as separate
objects the user juggles:

| | make ready | release | inspect |
|---|---|---|---|
| **disk** (downloaded / cold) | `download(ids)` | `delete_downloads(ids)` | `downloaded()` |
| **memory** (loaded / hot) | `load(ids)` | `unload(ids)` / `unload_all()` | `loaded()` |

Two inverse pairs: `download ⇄ delete_downloads` (disk, destructive word because
you re-fetch), `load ⇄ unload` (memory, non-destructive — the download stays).
"engine" is the *internal* name for a loaded model and never appears in a method
a user calls.

The `ModelStore` is internally a **read-through stack** — RAM (loaded) over disk
(downloaded) over remote (origin). `get(id)` cascades down on miss: in RAM?
return; on disk? load it; absent? download then load. The layering is
*encapsulated* (one object), but every control is explicit:

```python
store = ModelStore('totalsegmentator',
                   model_root_dir="/data/ts",   # local root (tree of Dataset###_*/...)
                   max_memory_mb=4000)           # GPU budget for loaded models (LRU-evict to fit)
```

`model_root_dir` (local, on-disk root) pairs with an implicit `model_root_url`
(remote download source, defaulted per ecosystem). `max_memory_mb` bounds the
**resident** loaded models — distinct from the per-run working-memory budget
(`peak_working_memory_mb`), which bounds the transient sliding-window slabs.

### Defaults & overrides (explicit, not hidden)

Defaults are real but *visible*: named constants, documented precedence
(explicit arg → env var → built-in default), and the resolved value is stored on
the object (`store.model_root_dir`). Env vars configure *paths* (fine); env vars
that *silently change behavior* (the old RAM-tier cache toggle) are gone.

### Freeing (the other half of ownership)

Because the store is owned, it is freeable — deterministically, not at GC whim:

```python
store.unload(297) / store.unload_all()   # free memory (reversible; download kept)
store.delete_downloads(ids)              # delete disk files (destructive; re-download)
with ModelStore(...) as store: ...       # unload_all() on exit; downloads kept
```

---

## The layering, top to bottom

```
L5  segment(task, image, store=…)                          one call; fans out over the task's ids
L4  TaskCatalog / recipes (SingleModel/Cascade/Union)  name→recipe; compose; refs resolved at load
L3  preprocess.* | infer.* | postprocess.* | geometry  pure fns over Volume/Prediction/Segmentation
L2  build_model(model_data, opts) -> LoadedModel       the only GPU allocation (one place)
L1  ModelStore (read-through: load/download)            id→model; readiness; bounded; freeable
L0  ModelData (frozen) | Volume (frozen)            pure data: weights+config | image+geometry
```

Each capability's home:

| capability | home |
|---|---|
| read ckpt (torch-free) → MLX arrays; per-ecosystem packaging | L1 source plug-ins → `ModelData` |
| image from NIfTI/DICOM/array, in-memory | L0 readers (`NiftiReader`, …) |
| build ~600 MB compute; reuse; free | L1/L2 `build_model` + the store's memory layer |
| reorient/permute/resample/normalize (+inverse) | L3 `preprocess.*` + `RestorePlan` + `postprocess.restore` |
| sliding window, Gaussian, mirroring, fold ensemble, region sigmoid | L3 `infer.sliding_window` → `Prediction` |
| argmax vs region-paint; subvoxel inverse; small-component drop | L3 `postprocess.*` |
| cascade (crop FOV → fine → paste) | L4 `Cascade` + L3 geometry crop/paste |
| label-union (remap + paint priority) | L4 `LabelUnion` + L3 `remap_labels`/`paint_priority` |
| name→pipeline dispatch, ecosystem-namespaced | L4 `TaskCatalog` |
| one-liner | L5 `run` |

---

## Core value types (frozen)

- `Geometry` — spacing_zyx, direction (cosines), origin, shape. Array order ZYX.
- `Volume` — `data: mx.array` channels-last `(Z, Y, X, C)`, `geometry`, `channels`.
- `Segmentation` — integer label `data (Z,Y,X)`, `geometry`, `schema`.
- `Prediction` — `data (K,Z,Y,X)` float, `geometry`, `schema`, `activation`.
- `RestorePlan` — original geometry + orientation + axis-permutation; the inverse
  recipe, **returned** from preprocess (never hidden), consumed by `postprocess.restore`.
- `LabelSchema` — int↔name, plus region definitions + paint priority (sigmoid models).
- `ModelData` — `config` + `schema` + `fold_weights` (MLX arrays) + `provenance`.
  **No GPU state.** Output of IO; input to `build_model`.
- `BuildOptions` — the build knob tail (folds, step_size, batch_size, mirroring,
  compile, dtype), frozen & hashable → doubles as the store's cache key.

Channels-last because the MLX port is channels-last end-to-end. SITK stays the
resampling/orientation backend (CPU, battle-tested).

---

## Recipes (frozen data, separate from execution)

```python
SingleModel(ref)                                   # one model
Cascade(stages=(CascadeStage(ref|task, crop_to, margin_mm), ...))
LabelUnion(members=(UnionMember(ref|task, relabel), ...), schema, priority)
```

A `ref` is `ModelRef(ecosystem, id)` — the ecosystem tag routes an id to the
right store. A recipe carries no weights, no GPU; it's printable, diffable,
serializable. Inter-task name references (e.g. teeth → craniofacial) are
resolved to ids when the `TaskCatalog` hands you the task, so at run time the
recipe is self-contained and `run` never needs the catalog.

`task.model_refs` — pure query (data out, no side effects) — feeds
`store.download(refs)` / `store.load(refs)`. The recipe exposes *what it needs*;
the store acts. (Never `task.init(store)` — that would invert the dependency,
making a frozen recipe know about runtime caches.)

---

## Usage (the feel)

```python
import medseg as ms

# one-liner
seg = ms.segment("total_fast", ms.NiftiReader().read("ct.nii.gz"))   # defaults wire catalog+store

# toolbox, every step by hand
store    = ms.ModelStore('totalsegmentator', model_root_dir="/data/ts")
artifact = store.get(297)
model    = ms.build_engine(artifact)
image    = ms.NiftiReader().read("ct.nii.gz")          # or DicomReader().read("series/")
prepared, restore = ms.preprocess.to_model_frame(image, model.config)
probs    = ms.infer.sliding_window(model, prepared, mirror_axes=(0,1,2))
seg      = ms.postprocess.restore(ms.postprocess.to_labels(probs, model.config.labels), restore)

# batch reuse + explicit memory + free
with ms.ModelStore('totalsegmentator', max_memory_mb=4000) as store:
    catalog = ms.TaskCatalog('totalsegmentator')
    task = catalog['total']                 # 5-part union
    store.load(task.model_refs)             # optional pre-warm (build now, fail-fast)
    for path in cohort:
        seg = ms.segment(task, ms.NiftiReader().read(path), store=store)
# store.unload_all() at exit; downloads kept
```

---

## Phased plan: build new alongside, then cut over

The current code works (244 tests, validated inference). Build the new surface
in a `medseg`-style namespace, get it green, then migrate `run_named_task`
callers and **delete** the old surface. Each phase is shippable and tested.

- **Phase 1 — value types** (`types`): Geometry, Volume, Segmentation,
  Prediction, RestorePlan, LabelSchema, BuildOptions. No deps. ← *start here*
- **Phase 2 — ModelData + ModelStore**: pure artifact; read-through store
  (format×location, download/load layers, memory budget, the readiness verbs,
  free, context manager). Synthetic-tree tests (no real weights).
- **Phase 3 — build_engine + stage namespaces**: `build_engine(artifact, opts)`
  (rework `InferenceEngine`); `preprocess` / `infer` / `postprocess` as pure-fn
  modules over the value types; image readers.
- **Phase 4 — recipes + TaskCatalog + run**: frozen recipes with `model_refs`;
  catalog that resolves nested refs at load; `run` fanning out over the catalog.
- **Phase 5 — cutover + delete**: migrate tests; delete the `_CACHE` global,
  `cached_engine_from_*`, `WeightsLayout` global registry,
  `ModelBundle.from_dataset/from_task`, `run_named_task`; update exports.

### What gets deleted

- module-global engine `_CACHE`, `cache_enabled`/`get_cached_engine`/`cache_engine`
- `cached_engine_from_folder/_task/_moose_model` (→ `ModelStore` methods)
- `WeightsLayout` + `_WEIGHTS_LAYOUTS` + `register_weights_layout` + `discover_weights`
  (→ explicit `ModelStore('ecosystem', model_root_dir=…)` + format×location)
- `ModelBundle.from_dataset`/`from_task` (resolution leaves the artifact)
- `run_named_task` (→ `run`); `TaskSpec`/`CascadeStep`/`UnionPart` reshaped into
  the frozen recipes (much survives — names, source-qualification, remap/paint)

### What survives intact (validated, aligned)

Source-qualified task names + `AmbiguousTaskError`; the `remap_labels` /
`paint_union` primitives; the region/argmax dispatch; LPS-reorient + transpose
handling; the generated `ts_tasks.json` (51 tasks) + its `uv run` admin CLI;
the sliding-window / fold-ensemble compute internals (rehomed, not rewritten).
