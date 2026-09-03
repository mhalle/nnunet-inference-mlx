# Changelog

## [Unreleased]

- **nnseg installs and runs from a clean Mac (2026-09-02).** Tried as an outsider would - fresh
  venv, `uv pip install "...[torch] @ git+https://...@feature/nnseg"`, empty weights root - and
  it failed twice before running anything. (1) uv applies `[tool.uv.sources]` to a git
  install, and the `nnunetv2 = { path = "../upstream/nnUNet" }` convenience source pointed at a
  path that exists only in the dev workspace; the source is gone (PyPI nnunetv2 2.8.1 is what
  nnseg is written against - stock APIs only), and the checkout goes in by hand for upstream
  work. (2) The wheel had no task catalog: hatch skips VCS-ignored files and the workspace
  `.gitignore`'s `data/` rule covered `src/nnseg/data`, so every task lookup died on a missing
  `ts_tasks.json`; `artifacts = ["src/nnseg/data/*.json"]` ships them regardless of gitignore
  (verified by building from the pushed branch's tree). With both fixed, `total_fast` on the
  chest CT ran on MPS in 68 s cold (38 s of it model load), 109/117 structures, 99.97 %
  agreement with the recorded result; a `nnseg serve` job on the same Mac was voxel-identical
  to the CLI. Around it: `nnseg tasks [--installed] [--json]` lists the catalog locally without
  importing torch ("installed" asks the weights store, not `materialized`, which for TS only
  says the spec is known); nnseg's own errors end as one `nnseg: <message>` line with status 2
  instead of a traceback (the missing-serve-extra `InputError` was reaching the user as a raw
  `ModuleNotFoundError`); single-model timing keys read `load:ts:total_fast`, not
  `load:ts:total_fast:ts:total_fast`; nnunetv2's non-CUDA `print` and its old-plans-format
  warning are silenced at the predictor. New: `docs/nnseg-getting-started.md`, the page a new
  user reads (requirements, install, weights, CLI, API, local server), linked from the README.

- **Previews are radiological, and say so (nnseg).** The three-plane preview showed the
  patient's right on the image right - not by decision but because the loader's RAS array
  was handed to `imshow` as-is, and nothing stated a convention. Found while checking
  MRSegmentator's left/right; confirmed on `ts:total_fast` from the same CT (liver plotted
  at column 327 of 512). `preview.DISPLAY` now names the convention (`radiological`, 3D
  Slicer's default: patient right on the image left in axial and coronal, sagittal viewed
  from the patient's left so anterior is on the image left), `preview.display_planes()`
  turns it into a per-panel orientation frame and gets there through
  `io.orientation_transform` - the same DICOMOrient probe the pipeline trusts, applied as
  a transpose+flip view, never a hand-written `[:, ::-1]` - and the R/L, A/P, S/I edge
  letters are read back from the resulting direction cosines and drawn on every panel, so
  a wrong convention is visible instead of silent. `neurological` remains available as a
  named option. Tested on a laterality phantom stored in five axis orders and read back
  from the PNG's pixels (`tests/test_nnseg_preview.py`). `load_oriented_pair` is
  unchanged (statistics still read the RAS pair), and cache keys do not move; previews
  already in a results cache keep their old appearance until regenerated.

- **MRSegmentator ecosystem (nnseg).** `mrsegmentator:base` (40 abdominal / pelvic / thoracic
  structures on MRI, also usable on CT) and `mrsegmentator:body_comp` (10 body-composition
  classes) join the catalog as a fourth nnU-Net ecosystem beside `ts`, `moose` and `custom` -
  the same shape as MOOSE (bare checkpoints on public assets, labels read from each installed
  checkpoint's own `dataset.json`, `@version` pins checked against the bytes on disk), and
  `nnseg serve` lists and installs them through the existing prepare path with no worker change.
  Two things are properties of upstream's *packaging* rather than of the checkpoints, and both
  are recorded in the manifest / spec rather than guessed: the zips are **flat** (a
  configuration folder with no `Dataset*` parent), so each one is installed under
  `<root>/mrsegmentator/<Dataset>/<trainer>__<plans>__<config>/` through a staging directory
  and one rename, with the zip's own `version.json` checked against the manifest tag; and
  MRSegmentator's reader **forces LPS** on top of plans that declare the non-reorienting
  `SimpleITKIO`, so the spec carries `orientation="LPS"`. That needed one small generalization:
  `TaskSpec.orientation` (None = follow the declared reader, as before), `io.read(target=...)`,
  and `pipeline.canonical_orientation_for()`, the one place the decision is made; provenance
  and ranked-store metadata gain `canonical_orientation` beside the unchanged
  `reoriented_to_ras`. Existing tasks are byte-for-byte unaffected (RAS for `ts`, stored order
  for plain nnU-Net models). `tools/gen_mrsegmentator_manifest.py` regenerates the manifest
  from upstream's `MODEL_REGISTRY` and reads dataset name, trainer and version out of each 1.1
  GB zip by Range request (a few MB, no download). nnseg's default fold policy (fold 0, no
  mirroring) matches upstream's `--fast`; pass `folds=[0,1,2,3,4]` for its default ensemble.
  Validated on three AMOS22 MRI cases against AMOS ground truth (fold 0): mean Dice 0.799 vs
  TS `total_mr` 0.792 on the 13 shared organs, higher on 9, every paired structure on the
  annotated side (`bench/results/mrseg_vs_tsmr_amos/` in the workspace). Still owed:
  voxel-level parity against MRSegmentator's own package output.

- **Volume and surface area measured from the field, not the mask (nnseg).** `nnseg.measure`
  integrates the margin field over the cells between voxel centers — full cells exactly,
  straddling cells by the plane their corners imply — so `V = int H(m)` and
  `A = int delta(m)|grad m|` are evaluated on the interpolant the store already defines.
  Area is the volume expression differentiated in the level, so the two cannot disagree
  about where the surface is. Against closed-form phantom truth at 1.5 mm, counted area is
  **+39 to +54 %** and does not converge (four grid refinements leave a sphere at +50.4,
  +50.9, +49.3, +50.9, +50.7 %) where the field is within 0.3 % and converges at O(h^2). The
  fair raster baseline is not face counting but SimpleITK's Crofton `ComputePerimeter`,
  which is within a couple of percent on a smooth body; the field beats it 0.14 % vs 0.85 %
  on sub-voxel stability, −4.5 % vs −16.6 % on a crease, converges monotonically where
  Crofton bounces, and needs no raster at all. Meanwhile
  counted volume swings −10.3 to +4.2 % across shapes where the field holds −0.6 to −0.1 %.
  Under pure sub-voxel translation counting saws by 1.10 % of a sphere's volume and 2.69 %
  of its area; the field by 0.00 % and 0.14 %. On real TotalSegmentator output the
  count/field area ratio is 1.51–1.68 across six structures. **Pass
  `clip=code.meta["clip"]`**: a straddling cell's far corners saturate wherever the margin
  climbs faster than the clip over half a cell diagonal, which is 30–95 % of cells at real
  gradients (3.0–7.2 logits/mm), and reading those bounds as values costs 1–5 % of the
  surface. `nnseg.statistics` passes it. Full account, including what does *not* help
  (subdivision, depth) and what is still biased (creases, −4.5 %), in
  [`docs/ranked-measurement.md`](docs/ranked-measurement.md).

- **`statistics.json` can carry both measurements (nnseg).** `compute_statistics` takes an
  optional `ranked_code` and then reports `volume_ml_field` and `area_cm2_field` beside
  `volume_ml`, with the field's own `field_grid_spacing_mm` — the code lives on the model
  grid, the labelmap has been restored onto the input's, and an area must never be compared
  across grids even where a volume may be. Without a code the output is byte-identical to
  before, so both can ship until the comparison has run on a real cohort. **The server does
  not pass one yet**: `artifact_overlap` computes from the in-RAM pair with no disk
  dependence, so the served `statistics.json` and `.tsv` are unchanged for now.

- **Analytic phantoms (nnseg).** `nnseg.phantoms` turns geometry with closed-form volume and
  area into logits, so anything reading the ranked field can be scored against calculus
  rather than against another measurement. A segmentation model cannot supply that — render
  a synthetic image, run a network on it, and the ground truth evaporates, because the
  decision surface is not the surface that was drawn. Sphere, ellipsoid, torus, shell, box,
  rounded box, star and an n-sector partition; closed forms cross-checked against a
  Gauss–Legendre × trapezoid rule with partials by autograd (the ellipsoid area agrees with
  `scipy.special.elliprg` to 12 digits). `tools/field_vs_counting.py` is the standalone
  comparison harness.

- **Fixed: multi-model tasks ran on one model's normalization (nnseg).** `segment()`
  cached the preprocessed frame by spacing alone, but the cached tensor had nnU-Net's
  **per-model** normalization baked in, so any task running several models at one
  spacing handed models 2..N the first model's intensity statistics. For `total` that
  meant the organs model's CT clip at +276 HU applied to vertebrae, cardiac, muscles
  and ribs, collapsing every bone density above it: on CT_Abdo, ribs 273 -> 412.9 mL,
  sternum 12.1 -> 59.6, costal cartilage 5.8 -> 128.6. `to_model_frame` is now
  `to_model_grid` (crop + resample, model-independent and therefore shareable, and a
  different type from a network input so it cannot be fed to one) plus `normalize_for`
  (per-model, fresh tensor), and the input carries a normalization fingerprint the
  consumer checks. **`__version__` is bumped to 0.2.0, which invalidates every cached
  serve result** - `total` results cached before this are degraded. Unaffected:
  `total_mr` (ZScoreNormalization reads no dataset statistics), single-model tasks,
  and the engines layer.

- **The job queue is durable (nnseg).** Records live in a sqlite `jobs.db` under
  the server's workdir, so a restart re-queues work instead of dropping it - a job
  is content-keyed and idempotent, so an interrupted run is re-runnable. It also
  gives job directories an owner, so the ones a previous run left behind are
  reclaimed rather than stranded forever holding their uploads. `--jobs-ttl-hours`
  bounds how long a record lasts (`--keep-finished` bounds memory and files).
  `tools/jobs.py` reads the store without a running server.

- **Named inputs, typed parameters, introspection (nnseg).** A task now publishes
  what it takes: `describe()` carries `inputs` (roles), `parameters` grouped by
  owner (the algorithm's knobs vs ours, as real JSON Schema), and `behavior` -
  read-only facts about what an engine does that a caller cannot change. Sources
  bind to roles **by name, never by position**: MONAI's BraTS bundle orders its
  channels T1c-first while nnU-Net's own convention puts FLAIR there, so a
  positional wire would silently mis-serve one of them. New `nnseg.schemas`
  generates the published schema and enforces it from one declaration; pydantic
  becomes a core dependency but stops at the wire. Options outside a task's
  schema are now refused at submit, which means `device`/`dtype`/`weights`/
  `batch_size`/`accumulate` are no longer settable remotely - deployment policy,
  not per-request knobs.

- **A content-addressed input store (nnseg).** Upload once, refer by digest
  thereafter: `GET/PUT /v1/inputs/{digest}`, `POST /v1/inputs` for a DICOM series
  (parts or a zip) as one tree, and a `{"kind":"input","sha256":...}` source. The
  server does the addressing - a declared digest is checked, never trusted. A
  blob's key is exactly the pre-existing upload identity, so no cached result
  moved. A tree's key is rooted in the sorted digests of its members, so it
  survives arrival order, filenames and zip metadata.

- **Results carry their content digest.** A finished job publishes `outputs`; the
  ETag is that digest and `If-None-Match` is honored (304), so a client stops
  re-downloading label volumes it already holds. `POST /v1/inputs?from_job=<id>`
  promotes a result into the input store, so one job's output can be another's
  input without the bytes routing through the client.

- **MONAI bundles can be multi-channel.** The engine takes a role->image mapping,
  orders channels by the bundle's own `channel_def`, and refuses inputs that do
  not share a grid - nnseg does not register images, and says so in the task's
  `behavior.alignment`. `brats_mri_segmentation` is curated as the first
  multi-input task. A region head (overlapping output channels, no `background`
  entry) is no longer read as a labelmap on either side.

- **Fixed: engine workers published results under a key nobody computes.** Every
  MONAI job recomputed forever - the worker's describe shim reported "unknown"
  weights for an engine whose identity is per task, so `publish_completion`
  re-keyed the result onto a key the API never computes. Its docstring already
  claimed the two "cannot drift"; there is now a test that checks it.

- **Modal:** a new `nnseg-inputs` Volume for the input store, mounted on the API
  function and every worker; the `jobs` volume is now `scratch`.

- **Engine / Ecosystem split (nnseg).** The two ideas that one class used to carry
  are now separate: an **ecosystem** is the user-facing *catalog* (`ts`, `moose`,
  `custom`, `fastsurfer`, `synthstrip`, keeping the `eco:task@version` grammar) and
  an **engine** is the *runtime* that runs it (`nnunetv2`, `fastsurfer`,
  `synthstrip`) - many ecosystems to one engine. A static registry
  (`nnseg.engines.registry`) is now the single source of truth for dispatch,
  enablement, knob forwarding, and the weights identity that keys the result cache;
  the engine ecosystems stop faking a `TaskSpec`, the per-engine describe shims are
  gone, `_spawn_worker` routes by grammar instead of hardcoded task prefixes, and
  the three Modal workers share one base class. Adding an engine is now a registry
  row plus a worker class that declares its image and compute. `/v1/tasks` reports
  `engine` per task and `/v1/version` lists the engines a deployment can run.
- **Renames that came with it** (nothing is deployed, so no aliases): the `native`
  ecosystem is now **`custom`**; `TaskSpec.source` is now **`lineage`** with value
  `nnunet` -> `nnunetv2` (freeing "source", which already meant *data source* on the
  wire); `WeightsStore(ecosystem=)` is now **`layout=`** with values
  `totalsegmentator`/`nnunet` -> `ts`/`nnunetv2` (it selects a weights tree, not a
  catalog); and `_is_native` is now `_uses_nnunet_preprocessing`, which is what it
  actually selects. Upstream names (`TOTALSEG_WEIGHTS_PATH`, `nnUNet_results`) are
  untouched.

- **All-GPU forward pipeline: reorient + resample on Metal.** `to_model_frame`
  gains `interpolation="auto"` (now the default): a per-axis Metal resampler
  (`resample_volume_mlx`) — factor-scaled, clamped anti-aliased cubic on
  *downsampling* axes (no aliasing of thin/high-contrast structure), linear on
  *upsampling/near-identity* axes (no cubic ringing — important for thick-slice
  CT's through-plane axis); ~0.5 s on a 418 M-voxel volume. And a GPU reorient
  (`reorient_array_mlx`) — permutation+flips derived from the direction cosines,
  bit-identical to `sitk.DICOMOrient` across RAS/LPS/SPL and arbitrary codes,
  replacing the ~5.5 s of CPU `DICOMOrient` memory-shuffle (forward 3.2 s + inverse
  2.3 s) with ~0.9 s on Metal. `restore` uses it for the inverse too. Net: chest
  fast-mode end-to-end **75 → 57.6 s**, output **bit-identical** to the prior SITK
  path. SITK now does only file IO + geometry. `"linear"/"bspline"/"nearest"`
  still route to SITK.
- **Fused Metal kernel for the logit restore (~100×).** The default linear
  `restore` was memory-gather-bound — 8 full-array corner fetches of the K
  logit channels, blend, then a separate argmax, materializing ~8× the
  K-channel output transiently. A single `mx.fast.metal_kernel` now does the
  whole inverse resample with one thread per *output* voxel: trilinear-
  interpolate all K channels inline (only the 8 source corners per channel)
  and reduce to one integer label on the fly. Nothing K-channel-sized is
  materialized, so there's no slab budget to tune (`peak_working_memory_mb`
  is ignored on this path). On ct.nii (512×512×165, 117ch) restore drops
  **24.7 s → 0.25 s (~98×)**; end-to-end `segment` **43 s → 11.5 s**. The
  pure-MLX slab path is retained as a fallback (`use_fused_kernel=False`, or
  automatic on kernel error). Region models get the same treatment via a
  fused threshold-paint kernel. Numerics: same separable blend op-order as
  the slab path → bit-identical on synthetic logits; on real smooth fields a
  handful of boundary voxels (~4e-5) flip on FMA-contraction rounding,
  negligible next to MLX↔PyTorch divergence. (Supersedes the earlier ~28%
  flattened-`mx.take` gather, an interim win on the now-fallback slab path.)
- **Fast nearest-neighbor inverse resample (path A), opt-in.** `restore` /
  `LoadedModel.segment` / `segment` gain `interpolation`/`output_interpolation`
  (`"linear"` default = logit interpolation, higher fidelity, like nnU-Net;
  `"nearest"` = argmax-at-model-spacing then NN-resample the label map, like TS).
  Profiling showed the default logit restore is memory-gather-bound (8-corner
  fetch of K=117 logit channels): on `ct.nii` (512×512×165) it's ~32 s; the NN
  path is **~0.4 s (≈75× faster)** with **98.5% agreement** (differs only at
  boundaries). So MLX end-to-end with `--resample nearest` ≈ ~7 s vs ~43 s
  (linear) / ~23 s (TS-MPS). CLIs: `nnmlx segment --resample linear|nearest`;
  `totalseg-mlx` maps TS's `--higher_order_resampling` (default NN like TS, `-ho`
  → logit interp).
- **Native weight download is live for TotalSegmentator.** TS v2 weights are public
  GitHub release zips, keyed per dataset id. A build-time generator
  (`scripts/refresh_ts_weights.py`) extracts the id→URL map from TS's
  `download_pretrained_weights` source into a shipped `data/ts_weights.json` (42
  datasets; 1 license-gated, flagged) — TS is never imported at runtime, same
  relationship as `ts_tasks.json`. The `totalsegmentator` store now has a default
  `fetch` that downloads the URL → verifies → unpacks, so `store.download(id)` works
  out of the box (verified end-to-end against GitHub). License-gated/unknown ids raise
  actionably. `nnunet`/`moose` have no default fetch (place locally or inject one).
- **CLIs auto-download by default (TS-like), opt-out.** `TotalSegmentator`/`totalseg-mlx`
  gain `--no-download`; `nnmlx segment` gains `--download/--no-download` (default on).
  Missing weights for the requested task are fetched before inference (`segment`'s new
  `required_weights_ids` resolves single/cascade/union ids). The *library* default stays
  explicit (no auto-download) — only the executables mimic TS. `download_archive` shows a
  tqdm byte-progress bar (auto-enabled on a TTY via `disable=None`, silent in pipes/tests).
- **`ModelStore.download(ids, *, force=False, build=False)` contract** — idempotent
  "ensure present" (fetch only what's missing; a no-op for present ids — the disk-layer
  twin of `load`'s read-through), with `force` to re-fetch. Returns the ids actually
  fetched. The fetch is an injectable `fetch(id, model_root)` seam; with none configured,
  a missing id raises actionably (pointing at the upstream downloader) instead of silently
  doing nothing. Added `verify_and_unpack(archive, sha256, dest)` (checks the archive's
  SHA-256 against the recipe's `weights_sha256` *before* unpacking — `.pth` is pickle, so
  this is a supply-chain gate, not just corruption detection; writes a `.verified` sidecar
  to avoid re-hashing) + `download_archive(url, dest)` + `sha256_file`. CLI: `nnmlx models
  download <ids> [--force]`. (The actual remote fetch / recipe URLs land with 0.11.)
- **HTTP client is now `httpx`** (was `requests`): `download_archive` and the torch-free
  range loader (`_torchfree/rangefile.py`, `load_pth_url`/`smart_load_url`) use httpx with
  `follow_redirects=True` (release assets 302 to a CDN). The `remote` extra is now just
  `httpx` (dropped `requests`/`remotezip` — `load_pth_url` reuses our `CachingRangeFile`).
- **Confirmed RAS is nnU-Net v2's universal canonical** (not just TS): the installed
  nnU-Net readers reorient inputs to RAS (`SimpleITKIO.read_images(orientation="RAS")`,
  `NibabelIO` to RAS) and back on write. So the RAS default is correct for all
  nnU-Net v2 ecosystems (TS, MOOSE, raw nnUNet); rationale documented in
  `preprocess.to_model_frame`.
- **Migrated `examples/` to the toolkit API** (the old ones imported removed
  symbols): `01_single_volume`, `02_batch_folder`, `03_logits_and_resolution`,
  `04_cascade_and_union`, `05_toolkit_namespaces`, plus a rewritten `examples/README`.
- **Expanded the `@slow` real-weights suite** (`test_real_weights.py`): organ-volume
  sanity, a non-default-trainer model (`Dataset117`), and skip-guarded region /
  MOOSE tests that run automatically once such weights are present. (Region,
  multi-fold, and MOOSE remain unvalidated on real data — no such weights downloaded.)

## [0.10.0] - 2026-05-31 — toolkit rearchitecture

Lands the composable toolkit API and removes the old hidden-state surface. Still
**pre-1.0** — breaking changes are expected, and 1.0 is gated on broader testing
(real-weights integration coverage, more tasks/ecosystems exercised).

### Fixed — left/right mirror in segmentation output (canonical orientation RAS, not LPS)

The inference canonical orientation was wrongly set to **LPS**, which mirrors the
volume left↔right vs **RAS** (nibabel's `as_closest_canonical`, what nnU-Net /
TotalSegmentator train and serve in). Since the network is not L/R-equivariant,
it saw a mirrored volume and produced **left/right-swapped labels** — structures
in the right *places* but with `left`/`right` reversed. Confirmed against the TS
mainline reference (`mlx-LEFT-lung` matched `ref-RIGHT-lung` at 0.93 Dice under
LPS; 0.97 vs `ref-LEFT` under RAS) and by anatomy (liver landed on the patient's
left). Default `reorient_to` is now `"RAS"` across `segment` / `LoadedModel` /
`preprocess.to_model_frame` / the CLI. This was a **pre-existing** port bug
(invisible to synthetic tests, which have no L/R semantics, and to old-vs-new
parity, which shared the same flip); added a real-weights `@slow` regression
(`test_real_weights.py`) asserting liver/spleen sit on the correct sides.

### Added — `nnmlx` CLI (Typer)

A command-line shell over the toolkit, so real-weights runs are one command:

```
uv run nnmlx segment total_fast ct.nii.gz seg.nii.gz
uv run nnmlx tasks list --modality CT
uv run nnmlx tasks show total
uv run nnmlx models list
```

Command groups: `segment` (run a named task → NIfTI), `tasks list`/`tasks show`
(catalog inspection), `models list`/`models loaded` (store inspection). Shared
`--ecosystem` / `--model-root` / `--max-memory-mb` on the top-level callback;
each command builds an explicit request-scoped `ModelStore` + `TaskCatalog` (no
global state). `typer` added to core deps; entry point `nnmlx`, also
`python -m nnunet_inference_mlx`.

### Added — `totalseg-mlx`, a TotalSegmentator-compatible CLI

A drop-in front end mirroring TotalSegmentator's `TotalSegmentator` argparse
(every flag parses), so existing TS command lines/scripts run on the MLX backend
unchanged:

```
totalseg-mlx -i ct.nii.gz -o segmentations            # one mask per class (TS default)
totalseg-mlx -i ct.nii.gz -o seg.nii.gz --ml          # single multilabel file
totalseg-mlx -i ct.nii.gz -o seg --fast -rs liver spleen -s
```

Supported: `-i/-o`, `-ot nifti`, `-ml`, `-f/--fast` + `-ff/--fastest` (→ our
`_fast`/`_fastest` tasks), `-ta/--task`, `-rs/--roi_subset`, `-rmb/--remove_small_blobs`,
`-s/--statistics` (volume mm³ + mean intensity → `statistics.json`), `-ss/--skip_saving`,
`-q/-v`, `--version`. Per-class output writes `{roi_name}.nii.gz` into the `-o`
directory, exactly like TS. Flags the MLX backend doesn't implement (radiomics,
nora, dicom output, body_seg/force_split, save_probabilities, license, …) are
accepted and ignored with a warning, so command lines don't break. The native,
non-TS interface remains `nnmlx`.

Entry points **`TotalSegmentator`** (a literal drop-in name — existing command
lines run verbatim under `uv run`) and `totalseg-mlx` (same front end; doesn't
shadow a real TS install). Console output **mimics TS**: citation line,
`Using 'fast' option...`, `Predicting...` / `Predicting part i of N ...` (per
union part), `  Predicted in Xs`, `Saving segmentations...` with a tqdm bar,
`  Saved in Xs`. To support that, `segment()` / the per-shape runners gained an
optional `progress` callback (invoked with short phase strings) so CLIs report
progress without the toolkit owning any console output.

### Added — output resolution control

`segment` / `LoadedModel.segment` / `postprocess.restore` gain output-resolution
knobs (mutually exclusive; default = the input grid):

- `--output-scaling S` — resolution multiplier (2 = finer/half-spacing, 0.5 = coarser).
- `--output-spacing MM` — absolute isotropic spacing.
- `--at-model-spacing` — the model's native training grid (no upsample back).

The labels are rendered **from the logits** at the requested grid (then argmax/
paint), not nearest-neighbor-resampled from a finished label map — higher
quality. The output header (spacing/shape/origin/direction) is recomputed over
the same physical extent, so it still overlays the input; `scaling=1` is a true
identity. Single-model tasks only for now (cascade/union raise). Downsampling
stays Nyquist-limited. SITK is now a core dependency (segmenting == image I/O),
so `uv run nnmlx segment …` works with no extra flags.

A composable toolkit API with **no hidden state**: three nouns + one verb —
`TaskCatalog` (name→recipe), `ModelStore` (id→model; read-through, bounded,
freeable), `segment` — over frozen value types (`Geometry`/`Volume`/
`Segmentation`/`Prediction`/`LabelSchema`/`RestorePlan`/`BuildOptions`) and
pure-fn stage namespaces (`preprocess`/`infer`/`postprocess`/`geometry`). Now the
package's public surface.

### Removed — the old hidden-state surface (breaking; Phase 5 cutover)

- **Module-global engine cache** (`engine_cache.py`: `cached_engine_from_*`,
  `get_cached_engine`, `clear_engine_cache`, `resolve_moose_config_folder`) →
  replaced by the owned, bounded `ModelStore`.
- **Module-global task registry + dispatcher** (`tasks.py`: `register_task`/
  `get_task`/`run_named_task`/`list_registered_tasks`/…) → replaced by the owned
  `TaskCatalog` (lookup) + `segment()` (dispatch). The recipe vocabulary
  (`TaskSpec`/`CascadeStep`/`UnionPart`/`AmbiguousTaskError`) is kept.
- **Old SITK orchestration** `workflow.py` (`run_workflow`/`run_label_union_workflow`/
  `Stage`/`ParallelStage`/`Bbox`/`compute_fg_bbox`/`crop_image`/`paste_segmentation`)
  and `resampling.predict_with_resampling` → replaced by `segment` composing
  `preprocess`/`infer`/`postprocess` + `geometry`.
- **Weights-layout discovery** (`WeightsLayout`/`discover_weights`/
  `register_weights_layout`) and `ModelBundle.from_folder`/`from_task` → folder
  reading now lives in `ModelData.read_folder`.
- Kept the low-level resampling primitives (`resample_image_to_target`,
  `inverse_resample_*`, `reorient`, `get_orientation`), the label primitives
  (`remap_labels`/`paint_union`/…), and `InferenceEngine` (now a private compute
  core). Verified end-to-end on real TotalSegmentator weights via the new API.

### Changed — forward resample now runs in float32 (behavior change)

The new pipeline casts the input to **float32 at read** (`sitk_to_volume`) and
resamples in float, matching nnU-Net v2's reference preprocessing. The legacy
`predict_with_resampling` resampled the raw int16 SITK image, which rounded
interpolated HU values to integers. On real int16 CT (TotalSegmentator
Dataset297, abdominal) this rounding accounted for **all** of the new-vs-old
divergence: 99.973% of voxels identical, differences confined to organ
boundaries. With both paths resampling in float the outputs are **bit-identical**
(verified end-to-end on real weights, same engine). Float is the intended
behavior going forward.

## [0.9.2] - 2026-05-27

### Added — source-aware engine resolution (MOOSE models)

`run_named_task` now picks an engine factory by the task's `source`:

- `ts` / `user` → integer nnU-Net dataset IDs via `cached_engine_from_task` (unchanged)
- `moose` → string folder names via the new **`cached_engine_from_moose_model(folder_name, models_dir=...)`**

MOOSE stores nnU-Netv2 models one folder per model (`Dataset123_Organs`) under a flat models dir — same inner `{trainer}__{plans}__{res}` config layout TS uses, only the outer identifier is a string. New **`resolve_moose_config_folder(folder_name, models_dir)`** maps the folder name to the config folder `cached_engine_from_folder` consumes. The MOOSE models dir resolves from `models_dir=` → `NNUNET_MLX_MOOSE_MODELS` / `MOOSE_MODELS` env vars → the installed `moosez` package, with a clear error if none resolve. `run_named_task` gains a `moose_models_dir=` parameter. Both new functions are exported.

### Added — nested-task cascade (`CascadeStep.crop_from_task`)

A cascade step can now reference another *registered task by name* instead of an inline `weights_id`:

```python
CascadeStep(crop_from_task="craniofacial_structures", crop_to_classes=(2, 7))
```

The dispatcher flattens the reference — recursively — into the `run_workflow` stage list: the referenced task (which may itself be `single` or `cascade`) runs, and its final stage carries the outer step's crop. `CascadeStep` validates exactly one of `weights_id` / `crop_from_task`.

This unblocks **TS `teeth`** (previously skipped by the generator). teeth crops from `craniofacial_structures`, which is itself a cascade — so it flattens three deep: `298 (rough total, crop→skull) → 115 (craniofacial, crop→teeth 2,7) → 113 (teeth)`. The TS registry now ships **51 tasks** (was 50; 23 single / 25 cascade / 3 label_union). The same mechanism covers MOOSE's two FOV-limited models (0.9.3).

### Added — `int | str` weights identifiers (MOOSE-ready)

A source audit of MOOSE v3.1.6 confirmed MOOSE identifies models by **string** (`"clin_ct_organs"` / `"Dataset123_Organs"`), not the integer dataset IDs TS/nnU-Net use. So the weights identifier is now the union type **`WeightsId = int | str`** (exported from the package root):

- `TaskSpec.single`, `CascadeStep.weights_id`, `UnionPart.weights_id` accept `int | str`.
- JSON (de)serialization preserves the incoming type — an integer stays an nnU-Net dataset ID, a string stays a MOOSE model identifier; TS integer IDs are *not* silently stringified.
- `run_named_task`'s default `engine_factory` resolves only integer IDs (via `cached_engine_from_task`); a string identifier without a custom factory raises a clear `NotImplementedError` rather than globbing for a bogus `Dataset` folder. Source-aware string resolution lands with MOOSE support (0.9.3).

6 new tests (`TestStringWeightsId`) cover string-id validation, type-preserving round-trip, and the factory guard. The full MOOSE scope — grounded in the v3.1.6 audit — is in `docs/post-0.8.2-roadmap.md`.

### Added — source-qualified task names (anticipating multi-system registries)

The registry now keys on `source:name` (e.g. `ts:total`, `moose:total`, `user:mytask`) so two model systems can ship a task with the same bare name without colliding. This is forward-prep for MOOSE support (0.9.3) — done now, while TS is the only source and there's nothing to migrate.

- **`TaskSpec.qualified_name`** property → `"source:name"`.
- **`get_task(name)`** accepts bare or qualified names. A bare name resolves when exactly one source defines it (the common case today); when multiple sources define it, it raises the new **`AmbiguousTaskError`** with the qualified alternatives. A qualified name (`"ts:total"`) always resolves directly.
- **`register_task`** keys on the qualified name — `ts:total` and `moose:total` coexist; re-registering the *same* qualified name still needs `overwrite=True`.
- **`unregister_task`** / **`run_named_task`** accept bare or qualified names with the same resolution rules.
- **`list_registered_tasks(*, source=None)`** returns sorted qualified keys; pass `source` to filter to one system. **`list_tasks_by_modality`** likewise returns qualified keys.
- **`TaskSpec.name`** may no longer contain `:` (reserved as the separator) — validated in `__post_init__`.

`AmbiguousTaskError` is exported from the package root.

### Added — `uv run`-native admin CLI for the registry (PEP 723)

`scripts/refresh_ts_registry.py` is now a two-subcommand admin tool — the maintainer-side counterpart to the user-facing `mlxseg` CLI (which never needs TotalSegmentator). It declares its `totalsegmentator` dependency via a PEP 723 inline `# /// script` block, so `uv run` provisions TS in an ephemeral environment on demand. TS (and its torch stack) never enters the package's own dependency set.

```bash
# Regenerate the shipped registry in place
uv run scripts/refresh_ts_registry.py generate --write

# Or emit to stdout for inspection
uv run scripts/refresh_ts_registry.py generate

# Verify the committed JSON is in sync with the generator
uv run scripts/refresh_ts_registry.py check
```

`check` exit codes are designed for CI automation (deferred): `0` in sync, `1` drift at the same TS version (regenerate), `2` committed file built against a different TS version than is currently resolved, `3` TS unimportable / file missing.

The generator output is now **deterministic** — the wall-clock `generated` timestamp was removed from `_meta` (it made every run diff). `_meta.ts_version` records what the data was built from; git history records when it changed. Two runs at the same TS version are byte-identical, so `check` only reports a real change.

Note: the "is the committed registry stale?" check is deliberately *not* a pytest — it requires provisioning the heavy TS/torch stack, which we keep out of the test environment. It lives in the admin CLI (`check`), to be wired into CI later. The unit suite validates the committed fixture's schema and content against our own code (`TestBuiltinRegistry`), which is the part that belongs there.

### Added — TS registry generator + 50 populated TS task entries

`scripts/refresh_ts_registry.py` extracts task definitions from a `totalsegmentator` distribution (provisioned via `uv run`, see above) and emits `src/nnunet_inference_mlx/data/ts_tasks.json`. The shipped JSON contains **51 TS tasks** (23 single, 25 cascade, 3 label-union) — including `teeth` via the nested-task cascade above — covering essentially the full TS v2.13.0 catalog.

### How the generator works

TS's task dispatch lives in `totalsegmentator.python_api.totalsegmentator()` as a flat `if task == "..." / elif` chain. Each branch is pure variable assignment:

```python
elif task == "lung_vessels":
    task_id = 117
    resample = [0.703125, 0.703125, 1.0]
    trainer = "nnUNetTrainerSkeletonRecall"
    crop = ["lung_upper_lobe_left", ...]
    robust_crop = True
```

The generator:

1. AST-locates the dispatch chain in the function source (just for the line span — no walking)
2. Dedents and `exec()`s it in a controlled namespace with `task` / `fast` / `fastest` set to each combination
3. Reads the resulting locals (`task_id`, `resample`, `trainer`, `crop`, ...)
4. Classifies by shape: list of IDs → label-union, crop=None → single, crop=names → cascade
5. Builds TaskSpec entries, resolving crop class names against `class_map["total"]` / `class_map["total_mr"]` / `class_map["body"]` as appropriate
6. Writes JSON to stdout

The "exec a code block in a controlled namespace" approach is more robust than AST walking — it tolerates TS syntax changes (the dispatch could become a match/case in a future TS version and our generator still works) and offloads Python's syntactic complexity to Python itself.

### What's covered

| Shape | Count | Examples |
|---|---|---|
| `single` | 23 | `body`, `body_mr`, `appendicular_bones`, `tissue_types`, `total_fast` (297), `total_fastest` (298), `total_mr_fast` (852), all the focused single-shot models |
| `cascade` | 24 | `lung_vessels`, `liver_vessels`, `liver_segments`, `cerebral_bleed`, `coronary_arteries`, `craniofacial_structures`, `kidney_cysts`, etc. — all the "rough total → focused" patterns |
| `label_union` | 3 | `total` (5 parts on CT, datasets 291-295), `total_mr` (2 parts on MR, datasets 850/851), `test` |

The `fast`/`fastest` flags on `total` and `total_mr` are expanded into separate registered names (`total_fast`, `total_fastest`, `total_mr_fast`, `total_mr_fastest`) — each is a distinct logical task with its own dataset ID.

### Skipped (with reasons)

- `total_v1`, `covid`, `appendicular_bones_auxiliary`, `face_mr_auxiliary`, `kidney_cysts_auxiliary` — present in TS's `class_map` but absent from the dispatch chain (legacy, deprecated, or auxiliary internal tasks).

(`teeth` was skipped in the first generator pass — its `crop_model` reference needed nested-task cascade support, now added above; it's included.)

### Usage

```bash
# Run inside an env with totalsegmentator installed
python scripts/refresh_ts_registry.py > src/nnunet_inference_mlx/data/ts_tasks.json
git diff src/nnunet_inference_mlx/data/ts_tasks.json
```

Now `run_named_task("total_fast", img)`, `run_named_task("lung_vessels", img)`, etc. work out-of-the-box (provided the underlying TS weights are downloaded; we don't manage that yet — that's the 0.11.0+ remote-download work).

### Tests

New `test_tasks.py` coverage: `TestBuiltinRegistry` (TS-catalog breadth + canonical pins), `TestCrossSourceConflicts` (qualified-name resolution + `AmbiguousTaskError`), `TestStringWeightsId` (MOOSE-style string IDs), `TestMooseEngineResolution` (folder resolution + source routing), and `TestNestedCascade` (crop_from_task flattening incl. the 3-deep `teeth` case). Total: **244 passing** (`-m "not slow"`), 1 skipped, 2 heavy benchmarks deselected by default. ~7s for the fast suite.

- `test_ts_tasks_present` — assert ≥ 40 tasks registered (catches generator regression)
- `test_canonical_ts_tasks_resolve` — verifies popular task names (`total`, `total_fast`, `body`, `lung_vessels`, …) are all registered with `source="ts"`
- `test_total_is_label_union_with_5_parts` — flagship task structurally is 5-part union with dataset IDs `{291, 292, 293, 294, 295}`
- `test_total_fast_is_single_model` — `total_fast` → dataset 297
- `test_lung_vessels_is_cascade` — 2-stage cascade ending at dataset 117

These assertions are pinned to TS v2.13.0. Future TS releases that change dataset IDs will fail these tests, surfacing the change at refresh time rather than silently.

### What's still TBD

- **CI workflow** to auto-refresh on a cron schedule + open a PR if `ts_tasks.json` diffs — not blocking; manual refresh (`uv run scripts/refresh_ts_registry.py generate --write`) is one command.
- **MOOSE registry generator + populated `moose_tasks.json`** — the engine resolution (`cached_engine_from_moose_model`) and nested cascade (`crop_from_task`) groundwork is now in place; remaining is `refresh_moose_registry.py` + a verification pass on a downloaded MOOSE model's `plans.json` (0.9.3).
- **MOOSE PET multi-channel** — only 2 models; deferred.
- **CLI (`mlxseg`)** — 0.10.0; consumes this registry.

## [0.9.1] - 2026-05-27

### Added — declarative task registry + `run_named_task` dispatcher

Names tasks. `mlxseg --task lung_vessels ...` (coming in 0.10.0 CLI) needs a registry; here it is.

The schema is informed by both TS (which has three pipeline shapes — single-model, cascade, label-union) and MOOSE's `moosez/constants.py` (which contributed modality-as-first-class, body coverage hints, and weight provisioning slots).

**New module `tasks.py`** with:

- **`TaskSpec`** dataclass — the descriptor:
  ```python
  TaskSpec(
      name: str,
      source: Literal["ts", "moose", "user"],
      modality: Literal["CT", "MR", "PET"],
      shape: Literal["single", "cascade", "label_union"],
      # Exactly one of these, matching `shape`:
      single: int | None = None,
      cascade: tuple[CascadeStep, ...] | None = None,
      union: tuple[UnionPart, ...] | None = None,
      # Informational:
      label_map: dict[int, str] = {},
      expected_coverage: str = "any",       # MOOSE-inspired
      weights_url: str | None = None,        # slot for 0.11.0+ remote weights
      weights_sha256: str | None = None,
  )
  ```
  Validated in `__post_init__`.

- **`CascadeStep`** / **`UnionPart`** — shape-specific sub-types.

- **Registry API**:
  - `register_task(spec, *, overwrite=False)`
  - `get_task(name)`
  - `unregister_task(name)`
  - `list_registered_tasks()`
  - `list_tasks_by_modality(modality)`

- **`run_named_task(name, image_sitk, *, folds=None, reorient_to="LPS", peak_working_memory_mb=None, verbose=False, engine_factory=None)`** — the dispatcher:
  - `shape="single"` → `predict_with_resampling`
  - `shape="cascade"` → `run_workflow` with `Stage` per `CascadeStep`
  - `shape="label_union"` → `run_label_union_workflow` with `ParallelStage` per `UnionPart`
  - `engine_factory` hook allows injecting custom engine construction (tests, non-standard weight locations).

### Added — JSON registry file

`src/nnunet_inference_mlx/data/ts_tasks.json` ships in the wheel. Loaded lazily on first registry access. Schema version 1; currently ships with an empty `tasks: []` — `scripts/refresh_ts_registry.py` (a generator that imports an installed TotalSegmentator distribution and emits this file) is planned for a follow-up patch.

Users can register custom tasks at runtime via `register_task(TaskSpec(...))` regardless of whether the shipped JSON has entries.

### Architecture notes

- The dispatcher is thin glue over 0.9.0's `predict_with_resampling`, `run_workflow`, and `run_label_union_workflow`. The registry adds *naming and persistence*; the dispatch logic itself is one branch per shape.
- The `engine_factory` injection point makes the dispatcher testable without weight files and gives users an extension point for non-standard weight locations (MOOSE-style remote downloads, custom WeightsLayout entries, etc.).
- The JSON storage is reviewable in PRs (vs hand-written Python dict), language-agnostic (other tools can consume it), and decoupled from our Python module — a downstream CLI or evaluator can read `ts_tasks.json` without importing our package.

### Tests

30 new tests in `test_tasks.py`. Total: **212 passing.**

Coverage:
- `TaskSpec` validation: shape↔data field consistency, allow-list enforcement (modality / source / shape), cascade min-length, union min-length, multiple-shape-fields rejection, missing-shape-field rejection
- JSON round-trip for all three shapes; int-key recovery in `label_remap` / `label_map`; default-value omission on disk
- Registry API: register / get / unregister / list / list-by-modality, duplicate-rejection, overwrite-with-explicit-flag, name-collision errors
- Dispatcher: per-shape backend routing, `weights_id` forwarding to factory, factory called once per distinct ID in order, unknown-task error, output geometry preservation
- Builtin registry loadability sanity check

### What's not in this release

- **`scripts/refresh_ts_registry.py`** — the generator that imports an installed TS and writes `ts_tasks.json`. Planned for 0.9.1.1 once verified against a real TS install. Until then, the shipped registry is empty and users register tasks at runtime.
- **CI cron** to auto-refresh against new TS releases — same dependency.
- **MOOSE entries / MOOSE-compat fields** — those are 0.9.2 work.
- **CLI (`mlxseg`)** — still 0.10.0; consumes this registry.

## [0.9.0] - 2026-05-26

### Added — multi-task label-union orchestrator

The TotalSegmentator full-mode pattern as a first-class primitive: run multiple independent task models against the same input, remap each to a shared unified label space, fold by paint priority. Previously buildable as ~25 lines of user code; now a documented, tested primitive.

**`ParallelStage`** dataclass — one task in the union:

```python
@dataclass
class ParallelStage:
    engine: InferenceEngine
    label_remap: dict[int, int]      # task-local ID → unified ID
    part_name: str | None = None     # optional, for logging
```

**`run_label_union_workflow(image, stages, *, reorient_to="LPS", ...)`** — orchestrator:

```python
from nnunet_inference_mlx import (
    cached_engine_from_task, ParallelStage, run_label_union_workflow,
)

stages = [
    ParallelStage(cached_engine_from_task(291), TS_REMAP_ORGANS,    "organs"),
    ParallelStage(cached_engine_from_task(292), TS_REMAP_VERTEBRAE, "vertebrae"),
    ParallelStage(cached_engine_from_task(293), TS_REMAP_CARDIAC,   "cardiac"),
    ParallelStage(cached_engine_from_task(294), TS_REMAP_MUSCLES,   "muscles"),
    ParallelStage(cached_engine_from_task(295), TS_REMAP_RIBS,      "ribs"),
]
seg = run_label_union_workflow(image, stages)
```

Semantics: each stage runs against the same input volume independently; labels are remapped from task-local to unified space; later stages overwrite earlier ones at overlapping voxels (list order = priority — matching the existing `_slab_resample_paint` convention).

Unlike `run_workflow` (sequential cascade with inter-stage cropping), there's no data dependency between stages — the orchestrator only adds a single LPS canonicalization at the boundary and a unified output buffer.

The orchestrator is intentionally thin glue — every step is a public top-level function (see below). Callers building non-standard variants (custom paint priority, logit-confidence merging, persistent intermediates) write the same recipe themselves.

### Added — top-level toolbox primitives

Four new public functions, exposed so the orchestrator recipe is fully decomposable:

- **`get_orientation(image_sitk) -> str`** — read the 3-letter DICOM orientation code (`"LPS"`, `"RAS"`, `"SAR"`, …) of a SITK image. Wraps the longwinded `DICOMOrientImageFilter_GetOrientationFromDirectionCosines`.
- **`reorient(image_sitk, code) -> sitk.Image`** — symmetric primitive that reorients to any DICOM code. No-op (returns the same object) when already in the requested orientation. Used to be inlined inside `predict_with_resampling` / `run_workflow`.
- **`remap_labels(seg, mapping)`** — vectorized LUT remap of integer labels. Source IDs not in the mapping become background. Auto-picks the smallest unsigned-int target dtype.
- **`paint_union(target, source)`** — overwrite `target` with `source` wherever `source != 0`. Same convention as `_slab_resample_paint`: list order is priority.

### Changed — `predict_with_resampling` kwarg rename: `reorient=` → `reorient_to=`

The previous parameter name shadowed the new top-level `reorient` function. Renamed to `reorient_to` (which also reads more naturally: "reorient *to* LPS"). Same default (`"LPS"`), same semantics.

**Migration:** rename any `reorient="LPS"` / `reorient=None` call to `reorient_to=...`.

### Refactor — `predict_with_resampling` and `run_workflow` use the new primitives

Both functions previously had the DICOMOrient logic inlined. They now call `get_orientation` / `reorient` directly, removing duplication. No behavior change.

### Tests

41 new tests across three files. Total: **182 passing.**

- `test_label_primitives.py` (20 tests) — `remap_labels` auto-dtype tiers, unmapped-drops-to-background, negative-ID rejection, shape preservation; `paint_union` overwrite-where-nonzero, zero-source transparency, priority-via-call-order, shape-mismatch error; a composition test that hand-rolls the union recipe (two tasks → remap each → paint into shared unified) to prove the toolbox is usable without the orchestrator
- `test_label_union_workflow.py` (10 tests) — empty-stages error, single-stage remap dispatch, geometry preservation, SAR orientation round-trip (the 0.8.1 chest-CT canary), `reorient_to=None` skip, two-stage paint priority (defining property of the orchestrator), disjoint-stages paint cohabit, output-dtype resolution across stages
- `test_canonical_orientation.py` (5 new tests) — `get_orientation` code detection, `reorient` direction change, no-op when already at target, round-trip geometry preservation

### What's not in this release

- **Declarative `TS_TASKS` registry + `run_named_task` dispatcher.** Originally planned for the same 0.9.0 milestone; split out to keep this release focused on the orchestrator + primitives. Likely lands as 0.9.1.
- **CLI (`mlxseg`)** — still 0.10.0, depends on the task registry.

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
- **`compute_fg_bbox(labels, *, classes=None, dilation_mm=0, spacing_zyx=None)`** — find FG bbox of a label volume, optionally restricted to specific classes and dilated by a physical margin. Returns `None` when no FG is found, signaling "skip cropping" to workflow callers.
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

- **`predict_sliding_window_streaming` dropped** in favor of the simpler `predict_sliding_window` kernel. The streaming variant's rolling-Z accumulator was optimizing peak memory that isn't where TS's pressure actually sits (the 5-models-in-one-process Metal cache, addressed separately by `engine.close()`). `SlidingWindowEngine.predict` now calls the non-streaming kernel; ~231 lines removed from `inference.py`. Test suite runs ~15% faster on the same volumes. No public-API impact — the variant was never exported.

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
