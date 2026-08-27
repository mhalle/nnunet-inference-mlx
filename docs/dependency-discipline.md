# Keeping the images lean — dependency & import discipline

**Requirements for preventing heavy dependencies (torch, CUDA, nnunetv2, surfa, …) from
weighing down `import nnseg` and the Modal images.** These are enforced by
`tests/test_nnseg_layering.py`; this document is the *why* and the *how*, so the rules are
maintained on purpose rather than rediscovered.

## Why it matters

A Modal cold start is dominated by **pulling the image** to a fresh worker, not by imports or
the model load. So a bloated image is a slow cold start for every user, and memory snapshots
**cannot** fix it (snapshots restore memory, not the image filesystem — measured: an api
snapshot gave 17.6 s restore ≈ 16.1 s creation, because the cost was the pull). The lever is
a **small image**. Concretely, dropping torch/CUDA/nnunetv2 from the api image took its cold
start from **16.1 s → 6.3 s**.

There's a second reason: nnseg is a **toolkit**. A lean consumer — a describe-only front-end,
a notebook that lists tasks, the serve API — should not pay for a multi-GB CUDA torch just to
`import nnseg`.

## Rule 1 — `import nnseg` must be torch-free

Importing the package, constructing a `Segmenter`, listing tasks, and describing a task must
pull **no torch**. Torch (and the rest of the inference stack) loads only when you actually
run inference.

- **Enforced by** `tests/test_nnseg_layering.py::test_importing_nnseg_is_torch_free` — a
  subprocess asserts `import nnseg` + the front-end surface (`Segmenter`, `ModelCache`,
  `TaskCatalog`, `io`) + `Segmenter(cpu).tasks()` + `describe()` leave `torch` out of
  `sys.modules`, and that `nnseg.segment` then *does* load torch.
- **Mechanism** (`src/nnseg/__init__.py`): the eager top-level surface is torch-free; the
  torch-pulling exports are loaded on first *attribute access* via a module `__getattr__`
  (PEP 562), listed in `_LAZY` (+ `_LAZY_SUBMODULES` for `backends`).
  - **Eager (torch-free):** `errors`, `io`, `job`, `progress`, `frame`, `grid`, `mapping`,
    `result`, `Segmenter`, `ModelCache`, `weights`, `tasks`, `reference`, `tables`.
  - **Lazy (torch):** `segment`, `TorchModel`, `resample_data`, `available_backends`,
    `resample_argmax`, `resample_paint`, `to_labels`, `ShuffleUp3d`, `swap_transposed`,
    and the `backends` subpackage.

If you add a new top-level export that pulls torch (or nnunetv2, or any heavy dep), put its
name in `_LAZY` — do **not** add an eager `from .<module> import …` for it.

## Rule 2 — heavy imports are call-time, never module top-level (on the light path)

In any module reachable by `import nnseg` without touching an inference symbol, heavy
dependencies must be imported **inside the function/method that uses them**, not at module
top level. This is already how `cache.py` (torch in `empty_cache`), `network.py` (nnunetv2 in
the predict methods), and `segmenter.py` (pipeline in `segment()`) are written.

- The top-level torch importers are exactly `network`, `pipeline`, `preprocess`, `resample`,
  `restore`, `shuffleup` — the modules that are *only* reached through a lazy export. Keep it
  that way; do not add a top-level `import torch`/`from torch` to any other module.
- The kernel layer (`grid`, `mapping`, `tables`, `restore`, `resample`, `reference`,
  `shuffleup`, `backends`) must depend on **torch + numpy only** (checked by
  `test_kernel_modules_depend_only_on_torch_and_numpy`). scipy is call-time only in
  `resample` (`test_scipy_is_only_a_call_time_dependency`).

## Rule 3 — the front-end (describe) path stays in torch-free modules

`describe()` powers the serve `/v1/tasks/{task}` endpoint, so it must be torch-free. It uses
`_resolve_spec` / `_uses_nnunet_preprocessing`, which live in **`tasks.py`** (torch-free), not in
`pipeline.py` (torch). They were moved there for exactly this reason — do not move task
resolution helpers back into a torch module, and route new describe-time helpers through
torch-free modules (`tasks`, `weights`, `values`, `errors`).

## Rule 4 — subpackages count (the gotcha)

The leak that broke Rule 1 the first time was the **`backends` subpackage**: its
`__init__.py` did a top-level `import torch` and eagerly imported the backend submodules
(each of which imports torch). A `grep 'import torch' src/nnseg/*.py` **misses subpackages**.
The only reliable check is at runtime:

```bash
uv run --no-project python -c "import sys, nnseg; print('torch loaded:', 'torch' in sys.modules)"
# must print: torch loaded: False
```

That is what the guard test runs in a subprocess. Trust it over any static grep.

## Rule 5 — per-image dependencies: install only the role's extras

Each Modal image installs **only the extras its role needs**, via
`uv_sync(extras=[...], frozen=False, extra_options="--no-sources-package nnunetv2")`:

| Image | Role | Extras | torch? |
|---|---|---|---|
| `api_image` | ASGI front-end (api/public) — describe + orchestrate + cache/publish | `serve, idc` | **no** |
| `image` (base) | nnU-Net GPU worker | `torch, serve, idc, cuda` | yes (CUDA) |
| `fs_image` | FastSurfer worker | `fastsurfer, idc` | yes (from the engine pkg) |
| `synthstrip_image` | SynthStrip worker | `synthstrip, idc, preview` | yes (from the engine pkg) |

Consequences to respect:
- **A dependency in `[project] dependencies` (core) weighs down *every* image.** Keep core
  minimal (`numpy, tqdm, typer, SimpleITK, pydantic`). This is why **mlx is an extra, not core** — the
  torch product installs mlx-free everywhere.
- **A dependency in an extra weighs down every image that installs that extra.** Put a dep in
  the *narrowest* extra that needs it (e.g. `triton` is in `cuda`, not `torch`; `matplotlib`
  — the preview renderer — is in `preview`, added only to worker images that render).
- The api image must never gain `torch`/`nnunetv2`/`triton`/`cuda`. If a new front-end
  feature seems to need torch, fix the import path (Rules 2–3) instead.

### uv_sync gotchas (all four are load-bearing)
- `frozen=False` — this repo gitignores `uv.lock`; the pyproject is the source of truth, so
  the image resolves at build.
- `--no-sources-package nnunetv2` — `uv sync` resolves the **whole** project's lock (every
  extra) before installing the selected ones, so it would hit the local
  `nnunetv2 = {path = "../upstream/nnUNet"}` source (absent in the build). Ignoring just that
  one source resolves nnunetv2 from PyPI; the engine git sources stay active. Do **not** use
  `--no-sources` (that kills the engine git sources too).
- `apt_install("git")` — every `uv_sync` image needs it, because resolving the full lock
  touches the engine packages' git sources.
- `--no-install-project` (implicit in Modal's `uv_sync`) — nnseg is **mounted**
  (`add_local_dir` + the `_pkg_dir()` `sys.path` shim), not installed, so a code edit does not
  rebuild the dependency layer.

## When core does grow — the pydantic precedent (2026-08-27)

Core gained one dependency since this document was written, and the reasoning is
the standard to hold the next one to.

`nnseg.schemas` declares each task's parameters as pydantic models, and one
declaration then produces three things: the JSON Schema `describe()` publishes,
the validation `POST /v1/jobs` enforces, and (through FastAPI response models)
the OpenAPI document. The alternative — a hand-written schema dict beside a
hand-written validator — is the same failure shape as a hand-maintained label
list: two copies of one truth that drift, and the drift is silent.

It was **measured before it was accepted**, not argued:

| | import | on disk |
|---|---|---|
| pydantic + pydantic-core | 27 ms | 7.8 MB |
| SimpleITK *(already core)* | 58 ms | 183 MB |
| numpy *(already core)* | 29 ms | — |

A twenty-third the size of a dependency core already requires, importing faster
than numpy, and **already present wherever the `serve` extra is installed**,
because FastAPI depends on it. So the marginal cost is confined to the engine
images, which carry torch.

Two conditions came with it, and both still hold:

1. **It stops at the wire.** `schemas.py` and `serve.py` only. It must not reach
   the value types (`Segmentation`, `Grid`, `LabelSchema`) or the kernel layer —
   those are frozen dataclasses on a hot path with no untrusted input, where
   validation is a cost with no reader. `tests/test_nnseg_layering.py` keeps the
   kernel torch+numpy-only, which also keeps pydantic out of it.
2. **It does not join the eager import path.** `import nnseg` still pulls neither
   torch nor pydantic (`test_importing_nnseg_is_torch_free` checks the first;
   the registry is imported lazily, which keeps the second).

The honest cost recorded for later: pydantic's v1→v2 migration was disruptive,
and this makes the project structurally exposed to their next major. `serve`
already bound us to pydantic v2 through FastAPI, so making it core widened the
blast radius without adding a new exposure — but it did widen it.

## Rule 6 — engine dependencies are self-describing

An engine's dependency tree comes from **its own package**, never re-listed in nnseg:
- `fastsurfer-lean` and `synthstrip-torch` (git sources in `[tool.uv.sources]`) declare their
  own deps; `uv` pulls them transitively.
- Adding an engine is: **one extra** (`fastsurfer = ["fastsurfer-lean"]`) + **one
  `[tool.uv.sources]` entry** (git + rev) + **the image's `extras` list**. The only nnseg-side
  deps you add are the serve-worker concerns the algorithm doesn't own (`scipy` cleanup via
  the extra, `obstore` via `idc`, `matplotlib` via `preview`).

## Rule 7 — weights are baked or volume-backed, never bundled

Model weights must not bloat the package or be committed to a repo:
- **nnU-Net / TS weights** live in the persistent `nnseg-weights` Modal Volume, provisioned
  once ever, then read locally.
- **FastSurfer / SynthStrip weights** are **baked at image build** (a `run_commands` step:
  `get_checkpoints` / `synthstrip_torch.fetch_weights()`), so cold containers never
  re-download them and the package repo stays weight-free.

## Conflicting extras

`torch` (pins `numpy>=2`) and `synthstrip` (pins `numpy<2`, for surfa) are declared
conflicting in `[tool.uv] conflicts`, so uv forks them into one universal lock. Therefore
`uv sync --all-extras` does **not** resolve; materialize an engine on its own with
`UV_PROJECT_ENVIRONMENT=.venvs/<engine> uv sync --extra <engine>`.

## Verification

```bash
# Rule 1 — the definitive runtime check
uv run --no-project python -c "import sys, nnseg; assert 'torch' not in sys.modules"

# All import-discipline rules (guard + AST layering + scipy call-time)
uv run --no-project pytest tests/test_nnseg_layering.py -q

# What an image actually installs (resolve the extras it uses)
uv run --no-project python -c "import sys, nnseg; nnseg.Segmenter(device='cpu').describe(nnseg.Segmenter(device='cpu').tasks()[0]); print('describe torch-free:', 'torch' not in sys.modules)"
```

## Checklist — before adding a dependency

1. **Is it needed at import, or only at call time?** Default to a call-time (method-level)
   import; a module on the light path must not import it at top level.
2. **Which role needs it?** Put it in the narrowest extra — never core unless every image and
   every consumer needs it.
3. **New top-level export that pulls a heavy dep?** Add its name to `_LAZY` in
   `__init__.py`; do not import it eagerly.
4. **New engine?** Its own package + extra + `[tool.uv.sources]` + the image's `extras` list.
   Weights baked or volume-backed.
5. **Run the guard:** `pytest tests/test_nnseg_layering.py`. If it needs a real `import nnseg`
   check for a new subpackage, the subprocess guard already covers it — add the new
   torch-free expectation there if you introduce a new front-end surface.
