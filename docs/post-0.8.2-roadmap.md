# Post-0.8.2 roadmap

> **Update (0.9.1 landed):** declarative task registry (`TaskSpec`, `CascadeStep`, `UnionPart`) + `run_named_task` dispatcher + JSON-storage registry in `data/ts_tasks.json` (empty for now; generator deferred to 0.9.1.x). See CHANGELOG 0.9.1.
>
> **Update (0.9.0–0.9.2 landed):** Shipped — the label-union orchestrator + top-level primitives (`get_orientation`, `reorient`, `remap_labels`, `paint_union`, 0.9.0); the declarative task registry + `run_named_task` dispatcher (0.9.1); the TS registry generator + 50 populated TS tasks + source-qualified names + `int|str` weights IDs (0.9.2). Next milestones below: MOOSE compatibility (0.9.3, audit-grounded), CLI `mlxseg` (0.10.0). The MOOSE section reflects a real source audit of v3.1.6.

State as of this document: **v0.8.2 shipped to origin.** The package now has:

- Architectural foundations (Predictor / SlidingWindowEngine / FoldEnsemble / InferenceEngine)
- Engine cache with auto-tiered RAM detection (0.7.0)
- Workflow orchestrator + crop/paste primitives for sequential cascades (0.7.0)
- `predict_logits` mx.array-native primitive (0.8.0)
- `inverse_resample_paint` for region-based (BraTS-style) models (0.8.0)
- Polymorphic `inverse_resample_argmax` accepting `mx.array | np.ndarray` (0.8.0)
- Bundle property accessors (target_spacing, has_regions, regions_class_order, bundle) (0.8.0)
- `resample_volume` numpy forward-resample primitive (0.8.0)
- Scheme-aware dispatch in `predict_with_resampling` (0.8.0)
- **LPS reorient in `predict_with_resampling` + `run_workflow` (0.8.1) — closed the orientation correctness hole**
- **`transpose_forward` / `transpose_backward` handling in `InferenceEngine` (0.8.2) — closed the transpose correctness hole**
- 167 passing tests

Validated end-to-end on:
- CT_Abdo full-mode TS pipeline (~3:23 wall time, 88 classes)
- chest.nii full-mode with orientation fix (~35:54 wall time, 111 classes, 124× more vertebrae vs pre-fix)
- chest.nii fast-mode (~5:40 wall time, 108 classes)
- lung_vessels two-stage cascade (~9:53 wall time, 4 vessel classes)
- lung_vessels with 1.5× subvoxel upsample render (~11:45 wall time, 3.33× FG voxel ratio matches 1.5³ theory)
- Engine cache 2× run: 3.7s cold load → 1.6ms warm hit (2300× speedup)

## What's next (in suggested order)

### 0.9.0 — `_run_label_union` orchestrator (parallel multi-task pattern)

The pattern user proposed for TS full mode: per-task argmax + per-task path B + label union by paint priority. Currently buildable as ~25 lines of user code; this turns it into a primitive.

**Scope (~150 lines):**

- New file `src/nnunet_inference_mlx/multitask.py` (or extend `workflow.py`)
- New dataclass:
  ```python
  @dataclass
  class ParallelStage:
      engine: InferenceEngine
      label_remap: dict[int, int]    # task-local class ID → unified class ID
      part_name: str | None = None    # optional, for debug/logging
  ```
- New orchestrator:
  ```python
  def run_label_union_workflow(
      image_sitk, stages: list[ParallelStage],
      *, peak_working_memory_mb=None, reorient="LPS",
  ) -> sitk.Image
  ```
  Internally: LPS reorient once, forward resample once, loop per-task
  (predict_logits → inverse_resample_argmax|paint → remap+paint into
  unified), reorient output back.
- Export from `__init__.py`
- Tests: synthetic two-stage union, verify paint order, geometry preservation
- CHANGELOG + docs

**Why this is 0.9.0:** new public primitive, addresses the multi-task pattern that has no analog in `run_workflow` (which is sequential cascade only). MOOSE doesn't have this either.

---

### 0.9.0 — Declarative task registry + named-task dispatcher

Concrete proposal in chat: TS_TASKS as a Python dict (or shippable JSON) with three task shapes — single-model, cascade (single dependency), label-union (multi-task).

**Scope (~380 lines + tests + data):**

- New file `src/nnunet_inference_mlx/tasks.py`:
  - `TaskSpec` dataclass
  - `TS_TASKS` default registry covering TS's main public tasks
  - `TS_LABEL_SCHEMES` registry — the unified 117-class space mappings
  - `register_task(name, spec)` extension point
  - `list_registered_tasks()`
  - `run_named_task(name, image, *, registry=TS_TASKS) -> sitk.Image`
  - Internal helpers: `_resolve_class_names`, `_run_label_union`, cascade builder
- Default `TS_TASKS` registry covers at least: `total_fast`, `total_fastest`, `total`, `body`, `lung_vessels`, `head_glands_cavities`, `appendicular_bones`, `tissue_types`, `liver_vessels`
- Label scheme registry derived from TS's `map_to_binary.py` (script to regenerate when TS updates)
- Tests: registry validation, name resolution, cascade graph walking, multi-task dispatch
- Example script using `run_named_task("lung_vessels", img)`
- CHANGELOG + docs

**Why this is 0.9.0:** completes the "TS-equivalent at the API level" story. After this ships, `run_named_task("lung_vessels", img)` does what `totalsegmentator(input=..., task="lung_vessels")` does — but with our path-B, scheme-correct, orientation-correct, transpose-correct backend.

Can ship alongside the orchestrator above as a single 0.9.0 release, or split into 0.9.0 (orchestrator) + 0.9.1 (registry).

---

### 0.9.3 — MOOSE compatibility (audit-grounded)

MOOSE (Multi-organ objective segmentation, ENHANCE-PET) ships nnU-Netv2-trained models for whole-body CT/PET/MR segmentation. **This section is grounded in a source audit of MOOSE v3.1.6** (`~/Dropbox/development/moose/MOOSE`), not from memory — the TS audit taught us that auditing materially changes the plan.

**Audit findings (MOOSE v3.1.6):**

MOOSE is nnU-Netv2-native. `moosez/models.py`'s `Model` class reads `dataset.json` + `plans.json`, pulls `transpose_forward`/`transpose_backward`, and computes `voxel_spacing = [voxel_spacing_t[i] for i in transpose_backward]` — **byte-identical to our `ModelBundle.target_spacing`.** Config folders are standard `trainer__planner__resolution` with `fold_*` subdirs. So our loader, engine, transpose handling, resampling, and sliding window apply unchanged.

Catalog: **25 models — 22 CT, 2 PET (`PT`), 1 MR.** Only **2 cascades** (`clin_ct_body_composition` ← `clin_ct_fast_vertebrae`; `clin_ct_face` ← `clin_ct_body`), both referencing the crop model **by name**. The other 23 are single-model. **No `label_union`** — running multiple MOOSE models yields multiple *separate* segmentations (one multilabel file per model), not a merged unified-class map. So MOOSE exercises only our `single` and `cascade` shapes.

Model identity is a **string** (`"clin_ct_organs"`) plus `KEY_FOLDER_NAME` (`"Dataset123_Organs"`) and a `KEY_URL` download link. `KEY_LIMIT_FOV` is the cascade descriptor: `{model_to_crop_from, inference_fov_intensities: [lo,hi], label_intensity_to_crop_from, largest_component_only}` — richer than our `crop_to_classes` + `dilation_mm`.

**Done (groundwork shipped before this milestone):**

- ✅ Registry machinery with source qualification — `source="moose"` valid; `moose:total` / `ts:total` coexist; `AmbiguousTaskError` resolves collisions
- ✅ Source-agnostic dispatcher — `run_named_task` branches on `shape`, not source
- ✅ The two shapes MOOSE uses (`single`, `cascade`)
- ✅ Generator pattern (PEP 723 `uv run` admin CLI) — `refresh_moose_registry.py` parallels the TS one; MOOSE's `MODEL_METADATA` is already a clean dict (no AST/exec needed — *easier* than TS)
- ✅ **Weights identifier widened to `int | str`** (shipped 0.9.2) — MOOSE string folder names are storable/round-trippable
- ✅ **Source-aware engine resolution** (shipped 0.9.2) — `cached_engine_from_moose_model(folder_name, models_dir)` + `resolve_moose_config_folder`; `run_named_task` picks the factory by `source` and takes a `moose_models_dir=` param (env-var fallbacks + moosez auto-detect)
- ✅ **Nested-task cascade** (shipped 0.9.2) — `CascadeStep.crop_from_task` + recursive flattening in the dispatcher; proven on TS `teeth` (3-deep). Covers MOOSE's two FOV-limited models structurally.

**Remaining required changes:**

1. **`refresh_moose_registry.py` admin CLI** (~120 lines) — parallel to TS; `MODEL_METADATA` is a plain dict, so simpler. Emits `data/moose_tasks.json`. Map MOOSE's richer `limit_fov` (intensity-range crop, `largest_component_only`) onto `crop_from_task` + `crop_to_classes` (expand `[lo,hi]` ranges to class tuples; `largest_component_only` is a future crop-primitive flag).

2. **Verification pass** on a downloaded MOOSE CT model — confirm our loader reads its `plans.json`/`dataset.json`, the normalization scheme is one we support, and the engine produces a sane segmentation.

**Genuine unknowns — need a downloaded MOOSE model's `plans.json` to resolve:**

- **Normalization schemes.** We have CTNormalization / ZScoreNormalization / NoNormalization. If any model's `plans.json` uses `RescaleTo01Normalization` or `use_mask_for_norm`, add it (~15–30 lines each). PET/MR models are the likely culprits.
- **Multi-channel PET-CT.** `clin_pt_fdg_*` / PUMA models report `modality="FDG-PET-CT"` → `expected_modalities=['FDG-PET','CT']` → genuine 2-channel input. Our public `predict` path is single-channel; the 22 CT-only models don't need this.
- **RAS orientation.** Recent model URLs say `_ras_`. Likely handled by the plans transpose (our engine already round-trips), but confirm against a real model. Our `reorient(code)` primitive supports arbitrary targets if needed.

**Recommended scope split:**

- **MOOSE-CT parity** (~120 lines): the generator + verification above. The engine-resolution and nested-cascade groundwork already shipped in 0.9.2, so 0.9.3 is mostly the registry generator + a real-model check. Covers the CT catalog (incl. both cascades structurally).
- **MOOSE PET multi-channel**: deferred — only 2 models, heaviest change (multi-channel input path). Its own milestone if demand appears.

---

### 0.10.0 — CLI (`mlxseg`)

After the registry exists, the CLI is mostly plumbing.

**Scope (~200 lines):**

- New entry point in `pyproject.toml`:
  ```toml
  [project.scripts]
  mlxseg = "nnunet_inference_mlx.cli:main"
  ```
- New file `src/nnunet_inference_mlx/cli.py`:
  - `argparse` setup
  - `--task NAME` (registry lookup) or `--model-folder PATH` (direct)
  - `--in PATH`, `--out PATH` (NIfTI; or `--out DIR` for per-organ split)
  - `--folds 0,1,2,3,4`, `--no-cache`, `--verbose`
  - `--rois CLASS,CLASS,...` (label filtering — see below)
  - `--remove-small-blobs MM3`
  - `--ml` (multilabel output) vs `--per-organ` (separate files)
  - `--list-tasks` (uses `list_registered_tasks`)
- Output format detection: `.nii.gz`, `.nrrd`, `.seg.nrrd` (auto-Slicer-segmentation)
- DICOM input via SITK ImageSeriesReader if `--in` points to a directory
- CHANGELOG + new example showing CLI usage
- Update README to advertise the CLI

**Why this is 0.10.0:** discrete user-visible feature. The CLI is what makes the package a tool, not just a library.

---

### 0.11.0+ — Optional enhancements

These are real but not blocking. Pick based on demand.

- **Label filtering / `--rois` implementation** (~80 lines): TS-style "give me only liver and spleen" filter. Two designs to choose between:
  - Post-hoc: argmax over all K classes, then zero out everything not in the requested subset
  - Pre-hoc K-prune: slice the final conv's output channels to only the requested classes. Faster downstream (less softmax/inverse work) but doesn't normalize softmax. Better for performance.
- **Weight download with caching** (~80 lines, optional `[remote]` extra already exists): use `requests` + `remotezip` to fetch `.pth` archives over HTTP range requests, cache locally.
- **`use_mask_for_norm` for MR models** (~30 lines): for ZScoreNormalization with `use_mask_for_norm=True`, compute stats from the foreground mask only.
- **`3d_cascade_fullres` configuration support** (~200 lines): nnU-Net's internal cascade (lowres pass conditions a fullres pass via an extra input channel). None of TS or MOOSE uses it; some research models do.
- **DICOM-SEG output** (~100 lines, new `[dicom]` optional extra with `pydicom-seg`): clinical interop.
- **Volume / intensity statistics per ROI** (~150 lines): MOOSE has this for PET; useful for cohort studies.

---

## What we deliberately are NOT doing

Listed for completeness so we don't drift into them.

- **Multi-fold ensemble across different models** — we discussed this and concluded the logit scales aren't commensurate without explicit calibration. The "logit-first" benefits are within-single-model; cross-model consensus needs Platt/temperature scaling we don't have.
- **Pre-hoc K-pruning as a marquee feature** — interesting but speculative. Add only if `--rois` needs it for performance.
- **Top-K logit representation for persistent storage** — interesting at large K but not blocking. Revisit after we have a real "persistent logits" use case.
- **Sparse octree for K-channel logits** — far-future. Only if someone hits the dense-array memory wall at K=117 on whole-body volumes.
- **Multi-modal logit fusion** — speculative. Wait for a real use case before designing.

## Implementation order recommendation

If picking up after a context turnover:

1. `_run_label_union` orchestrator (the new primitive) — fits in `workflow.py` next to `run_workflow`
2. Task registry + named-task dispatcher (depends on #1)
3. CLI (depends on #2)

Each is independently shippable and has clear acceptance criteria.

## Where the package is at architecturally

The 0.8.x series closed the two silent-correctness holes (orientation and transpose). 0.7.x established the layered architecture and the engine cache. 0.8.0 added the logit-first primitives (`predict_logits`, `inverse_resample_paint`).

The remaining 0.9.x work is **orchestration on top of stable primitives** — no architectural commitments to revisit. The primitives are proven by:
- Real TS-equivalent runs on CT_Abdo and chest CT
- A working 2-stage cascade (lung_vessels)
- A demonstrated path-B subvoxel render at 1.5×
- 167 tests including non-identity transpose, SAR orientation, region-based models, and scheme dispatch

After 0.9.x + CLI ships, the package is a self-contained TS replacement at the user-facing API level.
