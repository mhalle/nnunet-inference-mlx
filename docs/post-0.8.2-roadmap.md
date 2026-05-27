# Post-0.8.2 roadmap

> **Update (0.9.1 landed):** declarative task registry (`TaskSpec`, `CascadeStep`, `UnionPart`) + `run_named_task` dispatcher + JSON-storage registry in `data/ts_tasks.json` (empty for now; generator deferred to 0.9.1.x). See CHANGELOG 0.9.1.
>
> **Update (0.9.0 landed):** the multi-task label-union orchestrator and the four supporting top-level primitives (`get_orientation`, `reorient`, `remap_labels`, `paint_union`) are now shipped. See CHANGELOG 0.9.0. Next milestones below: TS task data via generator script (0.9.1.x), MOOSE compatibility (0.9.2), CLI `mlxseg` (0.10.0).

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

### 0.9.2 — MOOSE compatibility

MOOSE (Multi-organ objective segmentation, QIMP Vienna) ships nnU-Netv2-trained models for whole-body CT/PET segmentation. Most of MOOSE's catalog already works through our engine — single-channel CT, plain cascade, standard normalization. A handful of small pieces close the gaps for full parity.

**Audit of current coverage (no work needed):**

- ✅ CT single-channel inference (`clin_ct_organs`, `clin_ct_lungs`, `clin_ct_ribs`, `clin_ct_vertebrae`, `clin_ct_fat_muscles`, `clin_ct_body`, `preclin_ct_all`)
- ✅ `CTNormalization`, `ZScoreNormalization`, `NoNormalization`
- ✅ Sequential cascade with FOV crop (the MOOSE body→organ pattern, via `run_workflow`)
- ✅ Multi-fold ensemble (softmax / sigmoid)
- ✅ Region-based heads
- ✅ Engine cache (helps MOOSE more than TS — MOOSE has many small models)
- ✅ LPS reorient + transpose handling
- ✅ Per-ROI volume stats (one-liner: `np.bincount(seg.ravel()) * np.prod(spacing_zyx)`; documented in example, not a primitive)

**What's missing (the actual scope):**

1. **True multi-channel input path** (~50 lines)

   `InferenceEngine.predict()` / `predict_logits()` currently accept `(Z, Y, X)` and `SlidingWindowEngine.normalize()` is hardcoded to `ch = 0`. The infrastructure (`num_input_channels`, per-channel scheme list, per-channel norm params) is already in place — only the public path needs to loop.

   Changes:
   - Accept `(C, Z, Y, X)` numpy arrays in `predict` / `predict_logits` / `predict_segmentation`
   - Extend `normalize()` to iterate over `num_input_channels`, dispatching per-channel scheme
   - `predict_with_resampling` accepts a `list[sitk.Image]` (one per modality) instead of a single image, resamples each to target spacing independently, stacks
   - Tests: synthetic 2-channel engine with two different per-channel norm schemes

   Required only for true PET/CT *fused* MOOSE models (some PUMA variants). PET-only and CT-only single-channel tasks already work without this.

2. **`RescaleTo01Normalization`** (~15 lines)

   nnU-Net's percentile-clip + rescale-to-[0,1] scheme. Used by some PET models. Just another branch in `apply_normalization`.

3. **MOOSE `WeightsLayout` entry** (~30 lines)

   MOOSE's folder layout matches nnU-Net's `nnUNet_results/...` convention with minor variations (the directory naming for fold checkpoints, sometimes flattened, sometimes wrapped in a MOOSE-specific top level). Need to verify against a real MOOSE install and either confirm the existing nnU-Net layout works as-is or add a MOOSE layout to the `WeightsLayout` registry.

4. **`MOOSE_TASKS` registry entries** (~50 lines + data)

   Same shape as `TS_TASKS` (0.9.1). Per-task descriptors with model paths, cascade dependencies, label remaps, organ-name lookups. Generated from MOOSE's `expected_modality.json` and `organ_indices.json` (their internal config files).

5. **`use_mask_for_norm`** (~30 lines, optional)

   For MR `preclin_mr_all` (rodent MR). ZScoreNormalization computed over the foreground mask only, not the full volume. Already on the roadmap as 0.11.0+; consider promoting if MR matters.

**Total scope:** ~145 lines (excluding MR) or ~175 lines (with MR). Plus a verification pass on real MOOSE weights.

**Why this is 0.9.2:** sits naturally after the TS registry (0.9.1) — same machinery, second consumer. Single-channel MOOSE tasks already work today via `cached_engine_from_folder` + `predict_with_resampling`; the registry + multi-channel path are the gap-closers for "full MOOSE replacement."

**Coordination with 0.9.1:** the `TaskSpec` design from 0.9.1 should be MOOSE-aware from day one — extending the dataclass with an optional `modalities: list[str]` field for multi-channel tasks, and making the registry name-resolved across both TS and MOOSE rather than baking in a TS-only assumption. That way 0.9.2 is purely additive (new task entries, new weights layout, new norm scheme), not a redesign.

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
