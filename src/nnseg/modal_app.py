"""The nnseg REST protocol deployed on Modal - same contract, platform underneath.

    NNSEG_PROXY_AUTH=0 NNSEG_GPU=A10 nnseg modal deploy     # or: modal deploy <this file>

One `modal deploy` of this file gives a scale-to-zero deployment of the exact
contract `nnseg serve` speaks locally, so the client and the Slicer module cannot
tell the difference:

- an ASGI function on a cheap CPU container runs :func:`nnseg.serve.create_app` over
  a :class:`ModalExecutor` - stateless by construction, since `.spawn()` detaches the
  GPU work and all state lives in a `modal.Dict` (job metadata + progress snapshots)
  and a jobs Volume (uploads, label outputs);
- a GPU `Worker` class holds a warm ``Segmenter(cache_models=5)`` across jobs
  (``scaledown_window`` keeps it alive between a session's runs); weights
  self-provision into the shared ``nnseg-weights`` Volume on first use;
- where the LocalExecutor supplies a bounded FIFO, here **Modal is the queue**:
  spawn enqueues and the autoscaler drains up to ``NNSEG_MAX_CONTAINERS``
  (default 1: parallel requests queue and run serially on one warm worker - the
  economical posture, and every job after the first is warm; raise it at deploy
  time for cohort fan-out, up to the plan's GPU cap). ``accepting`` is always
  true - the backlog has no bound to enforce;
- progress writes are rate-limited (~4/s) because each Dict write is an RPC; the
  server's SSE endpoint reads them through its poll branch (``supports_push=False``);
- cancel is `FunctionCall.cancel()` - the container stops, billing stops;
- auth is Modal proxy auth (on by default; ``NNSEG_PROXY_AUTH=0`` to disable for a
  smoke test) - per-person tokens minted and revoked in the Modal dashboard, zero
  auth code here. The client sends them as Modal-Key / Modal-Secret headers.

Deploy-time configuration is by environment variable because Modal resolves
decorators at import: NNSEG_GPU (default L40S - wins `total` outright; A10 is the
economical fast-mode choice), NNSEG_APP_NAME, NNSEG_SCALEDOWN (seconds, default 120 -
conservative: a forgotten/left-up deploy idles at most ~2 min of GPU (~$0.07 on L40S)
before scaling down; raise it to keep a busy server warmer), NNSEG_PROXY_AUTH,
NNSEG_SNAPSHOT (memory snapshots, default ON - measured 2026-08-24: cold spawn->start
10-14 s -> 6.4-6.7 s, one 35 s snapshot-creation run per deploy).

The image mounts the *running* nnseg package (works from an editable checkout or an
installed wheel alike). TODO(release): switch to ``uv_pip_install("nnseg==<ver>")``
once published, so a deploy is pinned to a version instead of a working tree.
"""
import functools
import os
import sys
import threading
import time
from pathlib import Path

import modal

APP_NAME = os.environ.get("NNSEG_APP_NAME", "nnseg-serve")
GPU = os.environ.get("NNSEG_GPU", "L40S")
PROXY_AUTH = os.environ.get("NNSEG_PROXY_AUTH", "1") not in ("0", "false", "no")
SCALEDOWN = int(os.environ.get("NNSEG_SCALEDOWN", "120"))
GPU_SNAPSHOT = os.environ.get("NNSEG_GPU_SNAPSHOT", "0") not in ("0", "false", "no", "")
SNAPSHOT = (os.environ.get("NNSEG_SNAPSHOT", "1") not in ("0", "false", "no")) or GPU_SNAPSHOT
WARM_TASK = os.environ.get("NNSEG_WARM_TASK", "total_fast")
MAX_CONTAINERS = int(os.environ.get("NNSEG_MAX_CONTAINERS", "1"))
SHM_CACHE_GB = float(os.environ.get("NNSEG_SHM_CACHE_GB", "8"))
JOBS_TTL_H = float(os.environ.get("NNSEG_JOBS_TTL_H", "72"))
ARTIFACTS = set(filter(None, os.environ.get("NNSEG_ARTIFACTS",
                                            "preview,statistics").split(",")))
RESULTS_KEEP = int(os.environ.get("NNSEG_RESULTS_KEEP", "500"))
WEIGHTS_ROOT, SCRATCH_ROOT, CACHE_ROOT = "/weights", "/scratch", "/cache"
INPUTS_ROOT = "/inputs"
# Inputs get a bigger floor than fetched series: a re-fetchable IDC series
# costs a download when evicted, an uploaded volume is simply gone.
INPUTS_GB = float(os.environ.get("NNSEG_INPUTS_GB", "50"))
PUBLIC = os.environ.get("NNSEG_PUBLIC", "0") not in ("0", "false", "no", "")


def _engine_registry():
    """The engine registry, imported the same way the rest of nnseg is (mounted
    package, with the sys.path shim applied first)."""
    try:
        from nnseg.engines import registry
    except ImportError:
        sys.path.insert(0, "/root/pkg")
        from nnseg.engines import registry
    return registry


_engines = _engine_registry()
# Which engines this deployment runs. Snapshotted at import because Modal
# resolves the @app.cls decorators now; the registry reads the same env vars.
FASTSURFER = _engines.enabled("fastsurfer")
SYNTHSTRIP = _engines.enabled("synthstrip")
VOXTELL = _engines.enabled("voxtell")
MONAI = _engines.enabled("monai")


def _pkg_dir() -> Path:
    try:
        import nnseg
    except ImportError:                      # inside the container: the mounted copy
        sys.path.insert(0, "/root/pkg")
        import nnseg
    return Path(nnseg.__file__).parent


# Knobs read at RUNTIME inside the container must be forwarded into the image
# env at deploy time - a deploy-shell variable does not otherwise exist in the
# container (found the hard way: a TTL override that never took effect).
_RUNTIME_KNOBS = ("NNSEG_SHM_CACHE_GB", "NNSEG_JOBS_TTL_H", "NNSEG_RESULTS_KEEP",
                  "NNSEG_WARM_TASK", "NNSEG_ARTIFACTS",
                  # NNSEG_PUBLIC gates a module-level `if PUBLIC:` around the
                  # twin function. The deploy-time import registers it, but
                  # the CONTAINER re-imports this module - without the knob
                  # forwarded, its PUBLIC is False, the attribute never
                  # exists, and every request 303s while the runner crash-
                  # loops on AttributeError (hit live 2026-08-25).
                  "NNSEG_PUBLIC", *_engines.engine_env_vars())

# Base image (the ASGI api container + the nnU-Net GPU Worker). uv-NATIVE: the nnU-Net
# worker's deps come from pyproject extras - `torch` (torch/nnunetv2/scipy/scikit-image),
# `serve` (fastapi/uvicorn/python-multipart/matplotlib), `idc` (obstore), `cuda` (triton,
# the CUDA restore backend). nnunetv2 resolves from PyPI via --no-sources-package (the
# local ../upstream/nnUNet path source is dev-only and absent in the build). apt git: uv
# sync resolves the whole project's lock, which touches the engine git sources.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(extras=["torch", "serve", "idc", "cuda"], frozen=False,
             extra_options="--no-sources-package nnunetv2")
    .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/nnseg")
)

# FastSurfer engine image (built only when the deployment enables it). uv-NATIVE:
# `uv_sync` installs the project's deps for the `fastsurfer` + `idc` extras straight
# from pyproject (`--no-install-project`, so nnseg stays mounted, not installed) -
# the fastsurfer-lean git source + rev live ONLY in [tool.uv.sources], not here.
# `fastsurfer` pulls fastsurfer-lean (CNN inference deps incl. matplotlib, no
# monai/meshpy/torchio); `idc` pulls obstore (source fetch); core deps (SimpleITK
# etc.) come with the sync. frozen=False: this repo gitignores uv.lock (pyproject is
# the source of truth), so resolve at build.
fs_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")                       # uv needs git for the git source in pyproject
    # --no-sources-package nnunetv2: uv sync resolves the WHOLE project (every extra, to build
    # the lock) before installing the selected ones, so it would hit the local nnunetv2 path
    # source (file:///upstream/nnUNet, absent in the build) even though fastsurfer/idc don't
    # install nnunetv2. Ignoring just that one source resolves it from PyPI; the fastsurfer-lean
    # git source stays active.
    .uv_sync(extras=["fastsurfer", "idc"], frozen=False,
             extra_options="--no-sources-package nnunetv2")
    # Bake the ~67 MB VINN checkpoints into the image at BUILD (via FastSurfer's own
    # get_checkpoints, to the package-default paths) so cold containers never re-download
    # them from Zenodo/b2share - a reliability win (no runtime dependency on those hosts)
    # and it removes that slice from every cold start. At runtime get_checkpoints is then
    # a no-op. Placed before the nnseg mount so an nnseg edit doesn't bust this cache layer.
    .run_commands(
        "python -c \""
        "from FastSurferCNN import run_prediction as rp;"
        "from FastSurferCNN.utils.checkpoint import get_checkpoints,get_config_file,"
        "load_checkpoint_config_defaults as L;"
        "a=rp.make_parser().parse_args(['--t1','x','--sd','x']);"
        "get_checkpoints(a.ckpt_ax,a.ckpt_cor,a.ckpt_sag,"
        "urls=L('url',filename=get_config_file('FastSurferCNN')))"
        "\""
    )
    .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/nnseg")
)

# SynthStrip engine image (built only when enabled). uv-NATIVE, same shape as fs_image:
# `synthstrip` brings synthstrip-torch (from its git source in pyproject) + scipy (nnseg
# mask cleanup); `idc` brings obstore (fetch); `preview` brings matplotlib (serve-core
# preview - synthstrip-torch doesn't carry it, it's a serve-tier concern). numpy<2 comes
# from synthstrip-torch (surfa's reorient breaks on numpy 2.x). Weights fetch from MGH at
# first use (cached warm), like FastSurfer's checkpoints.
synthstrip_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")                       # uv needs git for the git source in pyproject
    .uv_sync(extras=["synthstrip", "idc", "preview"], frozen=False,
             extra_options="--no-sources-package nnunetv2")
    # Bake the 29 MB weights into the image at BUILD (to synthstrip-torch's default cache)
    # so cold containers don't re-download from MGH. Same rationale as FastSurfer above.
    .run_commands("python -c 'import synthstrip_torch; synthstrip_torch.fetch_weights()'")
    .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/nnseg")
)

# VoxTell engine image (built only when enabled). The `voxtell` extra brings the package
# and its own tree (torch<2.9, nnunetv2, transformers, huggingface_hub); `idc` brings
# obstore, `preview` matplotlib for the serve-core preview.
#
# Weights policy differs from the other engines, deliberately. The VoxTell checkpoint and
# the precomputed text-embedding bank are baked at BUILD (small, and they cover the common
# prompts with no text backbone at all). But a prompt outside that bank is embedded on the
# fly by Qwen3-Embedding-4B - ~8 GB, which would bloat the image and slow every cold pull,
# and the cold pull IS the cold start here. So HF_HOME points at the PERSISTENT weights
# volume instead: the backbone is fetched once ever and every later cold container reads it
# locally - the same treatment nnU-Net's weights already get.
voxtell_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(extras=["voxtell", "idc", "preview"], frozen=False,
             extra_options="--no-sources-package nnunetv2")
    # Bake the checkpoint into the image at a FIXED path (it is small) and address it by
    # VOXTELL_MODEL, so it stays findable after HF_HOME moves to the volume below.
    .run_commands(
        "python -c \""
        "import shutil;"
        "from voxtell.inference.predictor import download_voxtell_model as d;"
        "shutil.copytree(d(), '/opt/voxtell/model', dirs_exist_ok=True)\""
    )
    # The runtime caches - the small embedding bank, and the Qwen3 backbone that only a
    # prompt outside that bank needs - live on the PERSISTENT weights volume, so they are
    # fetched once ever and every later cold container reads them locally.
    .env({"VOXTELL_MODEL": "/opt/voxtell/model", "HF_HOME": f"{WEIGHTS_ROOT}/hf"})
    .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/nnseg")
)

# MONAI engine image (built only when enabled). The `monai` extra brings monai + torch;
# the curated bundles declare their own dependency set (itk, pytorch-ignite, einops, timm,
# ...) and the image carries the union - `tools/gen_monai_manifest.py` prints it, so the
# list is derived from the bundles rather than guessed. Weights are NOT baked: this is a
# catalog, so bundles install per task into the persistent weights volume (like nnU-Net and
# MOOSE), which is why this worker's _prepare/_ensure do real work.
monai_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(extras=["monai", "idc", "preview"], frozen=False,
             extra_options="--no-sources-package nnunetv2")
    .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/nnseg")
)

# Lean front-end image for the ASGI api/public functions. The api never runs inference -
# only catalog/describe + orchestration + cache/publish - and `import nnseg` + the whole
# describe path are torch-free (lazy inference imports), so this image carries NO torch /
# nnunetv2 / triton / CUDA: just serve-core (fastapi/uvicorn/matplotlib) + obstore + the
# core deps (numpy/SimpleITK). It cold-starts in a fraction of the worker image's time
# (the worker's ~16 s cold was the multi-GB image pull). The GPU work stays on the heavy
# worker image; the api just spawns it.
api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")                       # uv sync resolves the whole lock (engine git sources)
    .uv_sync(extras=["serve", "idc"], frozen=False,
             extra_options="--no-sources-package nnunetv2")
    .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/nnseg")
)

app = modal.App(APP_NAME, image=image)
weights_vol = modal.Volume.from_name("nnseg-weights", create_if_missing=True)
scratch_vol = modal.Volume.from_name(f"{APP_NAME}-scratch", create_if_missing=True)
# Uploaded inputs, addressed by their own bytes. A volume of its own rather than
# a prefix under the scratch one because the lifetimes are opposites: job
# directories churn and are deleted, while a preloaded input exists precisely to
# survive the scale-to-zero that makes preloading worth doing. Separate volumes
# are also what makes "inputs outlive results" expressible - an evicted result
# can be recomputed from its recipe, an evicted upload is simply gone.
inputs_vol = modal.Volume.from_name(f"{APP_NAME}-inputs", create_if_missing=True)
jobs_dict = modal.Dict.from_name(f"{APP_NAME}-jobs", create_if_missing=True)
cache_vol = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)


# -- marker operations -------------------------------------------------------
# The ownership rules for the two marker namespaces, at module level so unit
# tests can reach them (the closures they used to live in survived every
# mutation). The Dict has no CAS: each guard is get-then-op, which closes the
# always-loses cases; the residual one-RPC window is irreducible here.

def _install_inflight(key: str, jid: str) -> None:
    """Guarded install: never stomp a newer flight's marker. (A fresh submit
    installs directly - a submit is always the genuinely newest flight.)"""
    if jobs_dict.get(f"inflight:{key}") in (None, jid):
        jobs_dict[f"inflight:{key}"] = jid


def _release_inflight(key: str, jid: str) -> None:
    """Compare-and-delete: under duplicate flights the marker names the
    LATEST job, and this one may not be it - deleting unconditionally made
    probes 404 while the survivor still ran and left DELETE unable to
    cancel it."""
    if jobs_dict.get(f"inflight:{key}") == jid:
        try:
            del jobs_dict[f"inflight:{key}"]
        except Exception:
            pass


def _set_pending_marker(key: str, jid: str) -> None:
    """Refuse-if-present: a duplicate flight must not ACQUIRE ownership by
    stomping - it would then legally clear the marker while the sibling's
    overlap still renders, and probes would read a definitive 404 for
    artifacts that land seconds later."""
    if jobs_dict.get(f"artifacts:{key}") is None:
        jobs_dict[f"artifacts:{key}"] = {"state": "pending",
                                         "t": time.time(), "job": jid}


def _clear_pending_marker(key: str, jid: str) -> None:
    """Owner-only clear. A legacy marker without a job field (pre-ownership
    deploys) is treated as unowned and clearable by anyone - the sweep's
    rule."""
    m = jobs_dict.get(f"artifacts:{key}")
    if isinstance(m, dict) and m.get("job") not in (None, jid):
        return
    try:
        del jobs_dict[f"artifacts:{key}"]
    except Exception:
        pass


def _clear_own_artifacts_marker(jid: str, meta: dict) -> None:
    """Failure-path cleanup: drop this job's artifacts-pending marker (the
    overlap worker owns it on success). Without this a put that raised
    after set_pending left probes answering 202 until the sweep."""
    key = meta.get("cache_key")
    if key:
        _clear_pending_marker(key, jid)


def _prefetch_next(current_jid: str, stop, cache, read_ahead, vol_lock) -> None:
    """Best-effort CPU downloader, parallel to this GPU job: watch the shared
    jobs Dict for the oldest OTHER queued idc job and stage its series into the
    /dev/shm series cache. Scans every 2 s for the length of the run - a single
    scan at job start misses jobs whose submit lands moments later (the
    warm-chain case). With max_containers=1 the same container serves the next
    input, so the staging is always warm-handed. One-ahead only; the cache owns
    claims (atomic mkdir), commits (``.done`` marker) and LRU eviction, and a
    failed staging leaves nothing behind."""
    import threading

    def scan_once():
        cands = []
        for k in jobs_dict.keys():
            if ":" in str(k) or k == current_jid:
                continue                   # namespaced markers (inflight:/
                                           # artifacts:/cancel:) are not job
                                           # records; cancel: values are bare
                                           # floats and crashed this scan
            m = jobs_dict.get(k) or {}
            if m.get("state") != "queued" or m.get("kind") == "prepare":
                continue                       # prepare has no input to stage
            src = (m.get("source") or [{"kind": "upload"}])[0]
            kind = src.get("kind", "upload")
            ident = src.get("id") or src.get("crdc_series_uuid")
            if kind != "upload" and ident:
                cands.append((m.get("created", 0), kind, f"{kind}:{ident}", m["id"]))
            else:
                cands.append((m.get("created", 0), "upload", None, m["id"]))
        return min(cands)[1:] if cands else None

    def work():
        import shutil
        try:
            while not stop.is_set():
                nxt = scan_once()
                if nxt is None:
                    stop.wait(2.0)
                    continue
                kind, series, njid = nxt
                if kind != "upload":
                    if cache.staging(series) or read_ahead.has(series):
                        stop.wait(2.0)
                        continue
                    if not cache.has(series):
                        t_f = time.time()
                        if not cache.prefetch(series):
                            stop.wait(2.0)
                            continue
                        print(f"[prefetch] {series[:13]} staged in {time.time() - t_f:.1f}s "
                              f"(parallel to {current_jid})", flush=True)
                    t_r = time.time()
                    if read_ahead.fill(series, cache.path(series)):
                        print(f"[read-ahead] {series[:13]} read in {time.time() - t_r:.1f}s "
                              f"(parallel to {current_jid})", flush=True)
                    return                     # one-ahead only
                # upload: bytes already sit on the jobs volume - copy the file to
                # tmpfs under the lock (a reload racing save+commit could drop a
                # result; and reading a stable local copy sidesteps any question
                # of open handles across later reloads), then read it there.
                if read_ahead.has(njid):
                    stop.wait(2.0)
                    continue
                t_r = time.time()
                tmp = Path("/dev/shm") / f"preread_{njid}"
                try:
                    local = None
                    with vol_lock:
                        scratch_vol.reload()
                        srcs = list((Path(SCRATCH_ROOT) / njid).glob("input_*"))
                        if srcs:
                            tmp.mkdir(exist_ok=True)
                            local = tmp / srcs[0].name
                            shutil.copy2(srcs[0], local)
                    if local is None:
                        stop.wait(2.0)         # upload not visible yet; retry
                        continue
                    if read_ahead.fill(njid, local):
                        print(f"[read-ahead] upload {njid} read in {time.time() - t_r:.1f}s "
                              f"(parallel to {current_jid})", flush=True)
                        return                 # one-ahead only
                    stop.wait(2.0)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            print(f"[prefetch] failed: {e}", flush=True)

    threading.Thread(target=work, name="nnseg-prefetch", daemon=True).start()


def _purgeable(meta: dict, now: float, ttl_s: float) -> bool:
    """A job record may be purged when it is terminal and its ``finished``
    stamp is older than the TTL. Queued/running records are never purged by
    age - a stale active record is a symptom to surface, not tidy away."""
    if not isinstance(meta, dict):
        return True
    if meta.get("state") not in ("done", "failed", "cancelled"):
        return False
    return (now - float(meta.get("finished") or meta.get("created") or now)) > ttl_s


def _bound_jobs_store(current_jid: str) -> None:
    """The retention policy for the jobs store, run after every job: delete the
    finished job's own input upload (the bulk of the bytes - nothing reads an
    input after the job is terminal), purge terminal records + their
    directories past NNSEG_JOBS_TTL_H, and drop inflight markers whose job is
    gone. Keeps the jobs Dict listable and the jobs Volume bounded by traffic
    x TTL at ~result-size per job instead of ~input-size."""
    import shutil
    now, ttl_s = time.time(), JOBS_TTL_H * 3600.0
    try:
        jdir = Path(SCRATCH_ROOT) / current_jid
        for f in jdir.glob("input_*"):
            f.unlink(missing_ok=True)
        purged = []
        for k in list(jobs_dict.keys()):
            k = str(k)
            if k.startswith("inflight:"):
                continue
            m = jobs_dict.get(k)
            if k != current_jid and _purgeable(m, now, ttl_s):
                purged.append(k)
        for k in purged:
            shutil.rmtree(Path(SCRATCH_ROOT) / k, ignore_errors=True)
            try:
                del jobs_dict[k]
            except Exception:
                pass
        for k in list(jobs_dict.keys()):
            k = str(k)
            if k.startswith("inflight:"):
                jid = jobs_dict.get(k)         # markers hold the job id
                tgt = jobs_dict.get(jid) if isinstance(jid, str) else None
                if tgt is None or _purgeable(tgt, now, 0.0):
                    try:
                        del jobs_dict[k]
                    except Exception:
                        pass
            elif k.startswith("artifacts:"):   # a killed overlap thread leaves
                m = jobs_dict.get(k) or {}     # a stale pending marker behind
                if now - float(m.get("t") or 0) > 900:
                    try:
                        del jobs_dict[k]
                    except Exception:
                        pass
            elif k.startswith("cancel:"):
                if now - float(jobs_dict.get(k) or 0) > 900:
                    try:
                        del jobs_dict[k]
                    except Exception:
                        pass
        if purged:
            print(f"[purge] {len(purged)} finished jobs past {JOBS_TTL_H:g}h TTL", flush=True)
        scratch_vol.commit()
    except Exception as e:
        print(f"[purge] failed: {e}", flush=True)


_TERMINAL = ("done", "failed", "cancelled")


def _emit(jid: str, update: dict) -> None:
    """Merge an update into the job's Dict record, terminal-wins: once a
    record is terminal, only idempotent terminal re-writes land - a worker
    progress emit racing an API cancel can no longer resurrect the record to
    'running' (which wedged it forever: the purge never touches active
    states, so the inflight marker and 202 probes lived eternally). The
    read-modify-write is still not atomic - this closes the lost-CANCEL
    class, which is the one with an unbounded blast radius."""
    meta = jobs_dict.get(jid) or {}
    if meta.get("state") in _TERMINAL and update.get("state") not in _TERMINAL:
        return
    meta.update(update)
    jobs_dict[jid] = meta


_cls_extra = {"experimental_options": {"enable_gpu_snapshot": True}} if GPU_SNAPSHOT else {}


#: Serializes write+commit against reload on the inputs volume - a reload
#: between a write and its commit would discard the write.
_INPUTS_LOCK = threading.Lock()


def _content_store():
    """The input store, on its own Volume so it outlives every container.

    Volumes are not a POSIX-coherent shared filesystem: a write is published by
    commit() and someone else's write is seen by reload(). The single-writer
    claim inside SeriesCache is therefore not a true mutex across containers -
    but content addressing makes that harmless, because two writers of the same
    digest write identical bytes. What must hold is that a reader never sees a
    half-written entry, and the .done marker plus commit-after-write gives that:
    a reload lands on a committed version or the previous one.
    """
    from nnseg.content import ContentStore
    from nnseg.serve import SeriesCache

    def _no_fetch(key, entry):             # nothing is ever FETCHED into this one
        raise FileNotFoundError(f"{key} is not held by this server")

    cache = SeriesCache(Path(INPUTS_ROOT) / "content", _no_fetch,
                        budget_bytes=int(INPUTS_GB * (1 << 30)))
    return ContentStore(cache, commit=inputs_vol.commit, refresh=inputs_vol.reload,
                        lock=_INPUTS_LOCK)


def _reference(staged):
    """The one image artifacts render against.

    A multi-input job has no single input, but a preview and a statistics table
    do: they show the segmentation over an image, and the sensible choice is the
    task's FIRST declared channel - the reference the model's own channel order
    puts first.
    """
    if isinstance(staged, dict):
        return next(iter(staged.values()))
    return staged


def _execute_job(ctx, jid: str, source_tokens: dict | None = None) -> None:
    """The engine-agnostic job body shared by every worker: fetch/stage/
    read, then ctx._ensure + ctx._compute (the engine), then save +
    publish_completion + artifact overlap. Only _ensure/_compute/_prepare
    differ per engine; everything else - queue, cache, markers, prefetch,
    single-flight, cancel - is identical."""
    from dataclasses import asdict

    from nnseg.errors import Cancelled
    from nnseg.progress import CancelToken, Reporter
    meta = jobs_dict.get(jid)
    if meta is None or meta.get("state") == "cancelled":
        return
    jdir = Path(SCRATCH_ROOT) / jid
    last = {"t": 0.0}

    def on_progress(p):
        now = time.time()
        if now - last["t"] >= 0.25 or p.stage in ("restore", "finalize"):
            last["t"] = now
            if jobs_dict.contains(f"cancel:{jid}") if hasattr(jobs_dict, "contains")                         else jobs_dict.get(f"cancel:{jid}") is not None:
                token.cancel()         # cooperative: honored at the next check
            _emit(jid, {"progress": asdict(p)})

    token = CancelToken()
    started = time.time()
    pinned = []
    _emit(jid, {"state": "running", "started": started})
    prefetch_stop = threading.Event()
    _prefetch_next(jid, prefetch_stop, ctx.series_cache,
                   ctx.read_ahead, ctx._vol_lock)   # CPU downloader + pre-reader
    try:
        if meta.get("kind") == "prepare":
            rep = Reporter.of(on_progress, cancel=token)
            rep.stage("weights", meta["task"])
            result = ctx._prepare(meta["task"])
            _emit(jid, {"state": "done", "finished": time.time(), "result": result})
            return
        ctx._ensure(meta["task"])   # per-container weights provisioning (engine's own)
        entries = meta.get("source") or [{"kind": "upload"}]
        if len(entries) > 1:
            # A multi-input task. Everything needed is already on `source`: each
            # entry carries the CANONICAL role the wire bound it to, uploads were
            # written to the job dir under that role, and remote siblings fetch
            # exactly like a single-input job through the same pinned cache.
            rep = Reporter.of(on_progress, cancel=token)
            with ctx._vol_lock:
                scratch_vol.reload()
            staged = {}
            for entry in entries:
                role = entry.get("role") or "image"
                kind = entry.get("kind", "upload")
                if kind == "upload":
                    staged[role] = next(jdir.glob(f"input_{role}_*"))
                    continue
                if kind == "input":
                    staged[role] = ctx.content.resolve(
                        str(entry.get("id") or entry.get("sha256") or ""))
                    continue
                ident = str(entry.get("id") or entry.get("crdc_series_uuid") or "")
                key = f"{kind}:{ident}"
                ctx.series_cache.pin(key)
                pinned.append(key)
                rep.stage("fetch", f"{role} {ident[:8]}")
                t_f = time.time()
                staged[role] = ctx.series_cache.get_or_fetch(
                    key, check=rep.check,
                    credentials=(source_tokens or {}).get(kind))
                print(f"[fetch] {role} {ident[:13]} {time.time() - t_f:.1f}s", flush=True)
                rep.check()
            input_path = staged
        elif (kind := entries[0].get("kind", "upload")) == "input":
            # content this server already holds: nothing to fetch or copy
            input_path = ctx.content.resolve(str(entries[0].get("id") or ""))
        elif kind != "upload":
            src = entries[0]
            rep = Reporter.of(on_progress, cancel=token)
            ident = src.get("id") or src.get("crdc_series_uuid")
            key = f"{kind}:{ident}"
            ctx.series_cache.pin(key)
            pinned.append(key)
            if ctx.series_cache.has(key):
                how = "cached"
            elif ctx.series_cache.staging(key):
                how = "prefetched"
            else:
                how = "inline"
            rep.stage("fetch", ident[:13] if how == "inline" else how)
            t_f = time.time()
            input_path = ctx.series_cache.get_or_fetch(
                key, check=rep.check,
                credentials=(source_tokens or {}).get(kind))
            print(f"[fetch] {ident[:13]} {how} {time.time() - t_f:.1f}s", flush=True)
            rep.check()
            preread = ctx.read_ahead.pop(key)
            if preread is not None:
                rep.stage("read", "preread")
                print(f"[read] {ident[:13]} preread", flush=True)
                input_path = preread
        else:
            preread = ctx.read_ahead.pop(jid)
            if preread is not None:
                rep2 = Reporter.of(on_progress, cancel=token)
                rep2.stage("read", "preread")
                print(f"[read] upload {jid} preread", flush=True)
                input_path = preread
            else:
                with ctx._vol_lock:
                    scratch_vol.reload()
                input_path = next(jdir.glob("input_*"))
        from nnseg.serve import RESULT_NAME, ResultCache
        s = ctx._compute(input_path, meta, on_progress, token)
        with ctx._vol_lock:
            s.save(jdir / RESULT_NAME)
            scratch_vol.commit()
        from nnseg.serve import result_payload
        result = result_payload(s, jdir / RESULT_NAME)
        # The publication order (re-key, pair load, pending marker,
        # cache put, done, overlap start) lives in one place -
        # nnseg.serve.publish_completion; this side supplies the Dict
        # markers, the volume commit, and _emit. The marker landing
        # before the put also closes the last C8-class window here:
        # the entry becomes visible at commit, and the marker must
        # never trail it.
        from nnseg.serve import publish_completion

        def _migrate(old_key: str, new_key: str) -> None:
            _release_inflight(old_key, jid)
            _install_inflight(new_key, jid)
            meta["cache_key"] = new_key
            _emit(jid, {"cache_key": new_key})

        def _set_pending(key: str) -> None:
            _set_pending_marker(key, jid)

        def _clear_pending(key: str) -> None:
            _clear_pending_marker(key, jid)

        def _put(key: str) -> None:
            ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).put(
                key, jdir / RESULT_NAME, result,
                {"identity": meta.get("input_identity"), "task": meta["task"],
                 "options": meta.get("options"), "job": jid,
                 "computed": started})
            cache_vol.commit()

        def _mark_done() -> None:
            _emit(jid, {"state": "done", "finished": time.time(),
                        "result": result})

        def _start(pair, key: str) -> None:
            threading.Thread(target=ctx._artifact_worker,
                             args=(pair, key, jid, meta["task"]),
                             name="nnseg-artifacts", daemon=True).start()

        meta["cache_key"], _ = publish_completion(
            segmenter=ctx.seg, task=meta["task"],
            identity=tuple(meta.get("input_identity") or ()),
            options=meta.get("options") or {},
            cache_key=meta.get("cache_key"),
            labels_path=jdir / RESULT_NAME, input_image=_reference(input_path),
            artifacts=ARTIFACTS, cache_enabled=True,
            migrate_key=_migrate, set_pending=_set_pending,
            clear_pending=_clear_pending, put=_put,
            mark_done=_mark_done, start_worker=_start)
    except Cancelled:
        _emit(jid, {"state": "cancelled", "finished": time.time()})
        _clear_own_artifacts_marker(jid, meta)
    except Exception as e:               # noqa: BLE001 - reported to the client
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)                          # worker log for diagnosis
        _emit(jid, {"state": "failed", "finished": time.time(),
                    "error": f"{type(e).__name__}: {e}\n--- traceback ---\n{tb[-1600:]}"})
        _clear_own_artifacts_marker(jid, meta)   # a put that failed after
    finally:                                     # set_pending must not 202
        for key in pinned:                       # until the sweep
            ctx.series_cache.unpin(key)
        prefetch_stop.set()            # end the scan loop with the run
        with ctx._vol_lock:
            _bound_jobs_store(jid)
        if meta.get("cache_key"):
            _release_inflight(meta["cache_key"], jid)


class _WorkerBase:
    """Everything every worker does regardless of engine: the source registry +
    shared caches, the artifact overlap, and the job entry point.

    A plain (undecorated) base class - Modal collects ``@modal.enter`` /
    ``@modal.method`` across the MRO, so the three decorated workers below
    inherit these; the base itself must NOT be decorated or it becomes a
    ``modal.Cls`` instance that cannot be subclassed. What stays per-worker is
    exactly what differs: ``preload`` (its body IS the image's import set, and it
    runs pre-snapshot), the engine hooks, and the decorator's image/memory.
    """

    #: The registry engine this worker runs (a plain string: Modal harvests
    #: class attributes, so never hold the Engine object itself here).
    engine: str = _engines.NNUNETV2

    def _engine_setup(self) -> None:
        """Per-engine construction, after the shared setup. The nnU-Net worker
        builds a Segmenter; engine workers attach a describe-only shim."""
        self.seg = _EngineShim(self.engine)

    @modal.enter()
    def setup(self):
        """Post-restore, GPU attached: everything CUDA-adjacent lives here (unless
        the GPU snapshot already carries it)."""
        _pkg_dir()
        from nnseg.serve import ReadAhead, SeriesCache
        from nnseg.sources import registry
        self._sources = registry(None)

        def fetch_source(key, entry, credentials=None):
            prefix, ident = key.split(":", 1)
            if credentials is not None:
                return self._sources[prefix].fetch(ident, entry, credentials=credentials)
            return self._sources[prefix].fetch(ident, entry)

        self.series_cache = SeriesCache(Path("/dev/shm/series_cache"), fetch_source,
                                        budget_bytes=int(SHM_CACHE_GB * (1 << 30)))
        self.read_ahead = ReadAhead()
        self.content = _content_store()      # uploads referred to by digest
        self._vol_lock = threading.Lock()    # scan-thread reload vs save+commit
        # Never reset: under the GPU snapshot `preload` already ran and recorded
        # WARM_TASK here, and wiping it costs a redundant prepare + a multi-GB
        # weights_vol.commit() on the first job of every restored container.
        if not hasattr(self, "_ensured"):
            self._ensured = set()        # tasks whose weights this container verified
        if not hasattr(self, "seg"):
            self._engine_setup()

    def _artifact_worker(self, pair, cache_key: str, jid: str, task: str) -> None:
        """Post-done artifacts via the shared overlap body; this side's place
        is vol-locked add_artifact + tmpfs cleanup, and finish commits once,
        logs, and always deletes the pending marker."""
        from nnseg.serve import ResultCache, artifact_overlap
        cache = ResultCache(CACHE_ROOT, keep=RESULTS_KEEP)

        def _place(name: str, path) -> bool:
            with self._vol_lock:
                ok = cache.add_artifact(cache_key, name, path)
            Path(path).unlink(missing_ok=True)
            return ok

        def _finish(placed) -> None:
            try:
                if placed:
                    with self._vol_lock:
                        cache_vol.commit()
                print("[artifacts] overlap "
                      + " ".join(f"{n} {dt:.1f}s" for n, dt in placed)
                      + f" placed={[n for n, _ in placed]}", flush=True)
            except Exception as e:
                print(f"[artifacts] overlap failed: {e}", flush=True)
            finally:
                _clear_pending_marker(cache_key, jid)

        artifact_overlap(pair, task, ARTIFACTS,
                         preview_out=Path("/dev/shm") / f"preview_{jid}.png",
                         statistics_out=Path("/dev/shm") / f"stats_{jid}.json",
                         place=_place, finish=_finish)

    # -- engine hooks; _execute_job calls these. Engines that ship their weights
    # in their image have nothing to install, so these are the defaults.
    def _prepare(self, task: str) -> dict:
        return {"engine": self.engine, "task": task,
                "note": "weights ship with the engine image"}

    def _ensure(self, task: str) -> None:
        return None

    def _compute(self, input_path, meta, on_progress, token):
        raise NotImplementedError

    @modal.method()
    def run_job(self, jid: str, source_tokens: dict | None = None) -> None:
        _execute_job(self, jid, source_tokens)


@app.cls(gpu=GPU, timeout=3600, memory=32768, scaledown_window=SCALEDOWN,
         max_containers=MAX_CONTAINERS,
         volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
                  CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
         enable_memory_snapshot=SNAPSHOT, **_cls_extra)
class Worker(_WorkerBase):
    """The nnU-Net worker: runs every ecosystem whose engine is ``nnunetv2``
    (ts, moose, custom) through the Segmenter."""

    engine = _engines.NNUNETV2

    def _engine_setup(self):
        os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
        from nnseg import Segmenter
        self.seg = Segmenter(device="cuda", weights=WEIGHTS_ROOT, cache_models=5)

    _gpu_setup = _engine_setup          # legacy name used by the snapshot path

    @modal.enter(snap=SNAPSHOT)
    def preload(self):
        """The import bill, paid once per deploy: Modal snapshots memory after this
        and boots later cold containers from the snapshot. With classic snapshots
        CUDA must not be touched here - plain imports only. With the experimental
        GPU snapshot (NNSEG_GPU_SNAPSHOT=1) the CUDA state itself is captured, so
        the Segmenter is built and WARM_TASK's model loaded ONTO the GPU before the
        snapshot - a restored cold container then starts with a loaded model."""
        _pkg_dir()
        import nnunetv2  # noqa: F401
        import torch     # noqa: F401
        import nnseg     # noqa: F401 - pulls the pipeline import chain
        if GPU_SNAPSHOT:
            from nnseg.weights_fetch import ensure_task_weights
            self._engine_setup()
            ensure_task_weights(WARM_TASK, WEIGHTS_ROOT, progress=None)
            self.seg.warm(WARM_TASK)
            self._ensured = {WARM_TASK}

    # -- engine hooks (nnU-Net): unlike the image-baked engines, weights install
    # into the shared volume, so these do real work.
    def _prepare(self, task: str) -> dict:
        r = self.seg.prepare(task)
        weights_vol.commit()
        self._ensured.add(task)
        return r

    def _ensure(self, task: str) -> None:
        if task not in self._ensured:
            # Volume.commit scans the whole multi-GB weights tree, so ensure+
            # commit once per container, not per job. seg.prepare is catalog-
            # aware: ts, moose, custom all install through it.
            self.seg.prepare(task)
            weights_vol.commit()
            self._ensured.add(task)

    def _compute(self, input_path, meta, on_progress, token):
        return self.seg.segment(input_path, meta["task"], progress=on_progress,
                                cancel=token, **(meta.get("options") or {}))


class _EngineShim:
    """A Segmenter-shaped stand-in for engine workers, so ``publish_completion``'s
    re-key (weights_versions_of -> describe) reports the same weights identity the
    API-side describe does. Both read the registry, so they cannot drift - the
    divergence that once made every bare read 404 a cached result."""

    _catalog = None

    def __init__(self, engine: str):
        self._engine = engine

    def describe(self, task):
        identity = _engines.ENGINES[self._engine].weights_identity
        if identity is not None:
            return {"weights_installed": identity()}
        # An engine whose identity is PER TASK rather than constant - a CATALOG
        # like MONAI, where two bundles must not collide on one cached result.
        # `weights_identity=None` means "the ecosystem answers", so ask it, the
        # same way the API-side describe does. Returning [] here instead (which
        # this did until 2026-08-27) degrades weights_versions_of to "unknown",
        # and publish_completion then re-keys the finished result onto a key the
        # API never computes - so every job of that engine published into a slot
        # nothing would ever look up, and none of them ever hit the cache.
        cls = type(self)
        if cls._catalog is None:
            from nnseg.ecosystems import EcosystemCatalog
            cls._catalog = EcosystemCatalog(root=WEIGHTS_ROOT)
        try:
            info = cls._catalog.info(task) or {}
        except Exception:
            return {"weights_installed": []}
        return {"weights_installed": info.get("weights_installed") or []}

    def resolve_task(self, t):
        return t


if FASTSURFER:
    @app.cls(gpu=GPU, timeout=3600, memory=40960, scaledown_window=SCALEDOWN,
             max_containers=MAX_CONTAINERS, image=fs_image,
             volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
                  CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
             enable_memory_snapshot=SNAPSHOT, **_cls_extra)
    class FastSurferWorker(_WorkerBase):
        """The FastSurfer engine worker: the shared scheduler + serve-core from
        _WorkerBase, with FastSurfer's image and compute."""

        engine = "fastsurfer"

        @modal.enter(snap=SNAPSHOT)
        def preload(self):
            """Heavy imports paid once per deploy, before the memory snapshot; later
            cold containers restore from it. Stays per-worker because its body IS this
            image's import set. Classic snapshot => imports only, no CUDA; with
            NNSEG_GPU_SNAPSHOT the model is built onto the GPU so a restored container
            starts model-ready."""
            _pkg_dir()
            import torch  # noqa: F401
            import FastSurferCNN.run_prediction  # noqa: F401 - the CNN import graph
            import nnseg  # noqa: F401
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            if GPU_SNAPSHOT:
                from nnseg.engines import fastsurfer
                fastsurfer._get_runner("cuda", 8)

        def _compute(self, input_path, meta, on_progress, token):
            from nnseg.engines import fastsurfer
            # input_path is a SimpleITK image when read-ahead pre-read it
            # (memory-in, decode-once) or a path otherwise; segment() takes both
            # and writes no temp files (model is cached across jobs on this worker).
            return fastsurfer.segment(input_path, device="cuda")


if SYNTHSTRIP:
    @app.cls(gpu=GPU, timeout=3600, memory=32768, scaledown_window=SCALEDOWN,
             max_containers=MAX_CONTAINERS, image=synthstrip_image,
             volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
                  CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
             enable_memory_snapshot=SNAPSHOT, **_cls_extra)
    class SynthStripWorker(_WorkerBase):
        """The SynthStrip engine worker: the shared scheduler + serve-core, with
        the slim synthstrip image and the standalone synthstrip-torch package."""

        engine = "synthstrip"

        @modal.enter(snap=SNAPSHOT)
        def preload(self):
            """Heavy imports before the memory snapshot (see FastSurferWorker.preload)."""
            _pkg_dir()
            import torch  # noqa: F401
            import synthstrip_torch  # noqa: F401 - model class + torch
            import nnseg  # noqa: F401
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            if GPU_SNAPSHOT:
                from nnseg.engines import synthstrip
                synthstrip._get_model("cuda")

        def _compute(self, input_path, meta, on_progress, token):
            from nnseg.engines import synthstrip
            # input_path is a SimpleITK image (read-ahead memory-in) or a path;
            # segment() takes both and writes no temp files (model cached per worker).
            return synthstrip.segment(input_path, device="cuda")


#: engine name -> the worker class defined for it. Engine workers are defined
#: conditionally (Modal resolves the decorators at import, so an image is built
#: only where its engine is enabled), hence the lookup by name.
_WORKER_CLASSES = {_engines.NNUNETV2: "Worker",
                   "fastsurfer": "FastSurferWorker",
                   "synthstrip": "SynthStripWorker",
                   "voxtell": "VoxTellWorker",
                   "monai": "MonaiWorker"}
assert set(_WORKER_CLASSES) == set(_engines.ENGINES), (
    "every engine needs a worker class (and vice versa): "
    f"{sorted(_WORKER_CLASSES)} vs {sorted(_engines.ENGINES)}")


if VOXTELL:
    @app.cls(gpu=GPU, timeout=3600, memory=40960, scaledown_window=SCALEDOWN,
             max_containers=MAX_CONTAINERS, image=voxtell_image,
             volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
                  CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
             enable_memory_snapshot=SNAPSHOT, **_cls_extra)
    class VoxTellWorker(_WorkerBase):
        """The VoxTell engine worker: free-text prompts instead of a fixed task.

        The only worker whose compute reads the job's ``options`` for what to
        segment - ``{"prompts": [...]}`` - which is also what makes two prompt lists
        two different cache entries."""

        engine = "voxtell"

        @modal.enter(snap=SNAPSHOT)
        def preload(self):
            """Heavy imports before the memory snapshot (see FastSurferWorker.preload)."""
            _pkg_dir()
            import torch  # noqa: F401
            import voxtell.inference.predictor  # noqa: F401 - the model import graph
            import nnseg  # noqa: F401
            if GPU_SNAPSHOT:
                from nnseg.engines import voxtell
                voxtell._get_predictor("cuda")

        def _compute(self, input_path, meta, on_progress, token):
            from nnseg.engines import voxtell
            opts = dict(meta.get("options") or {})
            seg = voxtell.segment(input_path, opts.get("prompts"), device="cuda",
                                  progress=on_progress, cancel=token)
            # The text backbone and embedding bank land in HF_HOME on the weights
            # volume; commit once per container so the next cold start reads them
            # instead of re-downloading (the whole point of caching them there).
            if not getattr(self, "_hf_committed", False):
                try:
                    weights_vol.commit()
                    self._hf_committed = True
                except Exception as e:                  # never fail a finished job on this
                    print(f"[voxtell] weights volume commit failed: {e}", flush=True)
            return seg


if MONAI:
    @app.cls(gpu=GPU, timeout=3600, memory=40960, scaledown_window=SCALEDOWN,
             max_containers=MAX_CONTAINERS, image=monai_image,
             volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
                  CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
             enable_memory_snapshot=SNAPSHOT, **_cls_extra)
    class MonaiWorker(_WorkerBase):
        """The MONAI engine worker: a CATALOG of bundles, so unlike the other engine
        workers its _prepare/_ensure do real work - bundles install per task into the
        weights volume, exactly as the nnU-Net worker installs its models."""

        engine = "monai"

        @modal.enter(snap=SNAPSHOT)
        def preload(self):
            """Heavy imports before the memory snapshot (see FastSurferWorker.preload)."""
            _pkg_dir()
            import torch  # noqa: F401
            import monai  # noqa: F401 - eagerly loads transforms/networks/inferers...
            # ...but monai/__init__ EXCLUDES monai.bundle from that eager load
            # ("(^(monai.bundle))" in its exclude_pattern), and monai.bundle is the
            # only part this engine actually calls. Importing it here is what puts
            # it in the snapshot instead of on the first request after every restore.
            import monai.bundle  # noqa: F401
            import monai.transforms  # noqa: F401 - the chain every bundle composes
            import nnseg  # noqa: F401

        def _bundle_of(self, task: str) -> str:
            return str(task).partition(":")[2] or str(task)

        def _prepare(self, task: str) -> dict:
            from nnseg.ecosystems import MonaiEcosystem
            bundle = self._bundle_of(task)
            MonaiEcosystem().ensure(bundle, WEIGHTS_ROOT)
            weights_vol.commit()
            self._ensured.add(task)
            return {"engine": self.engine, "task": task, "bundle": bundle}

        def _ensure(self, task: str) -> None:
            if task not in self._ensured:
                self._prepare(task)

        def _compute(self, input_path, meta, on_progress, token):
            from nnseg.engines import monai_bundle
            return monai_bundle.segment(input_path, self._bundle_of(meta["task"]),
                                        root=WEIGHTS_ROOT, device="cuda",
                                        progress=on_progress, cancel=token)


def _worker_classes() -> dict:
    """engine name -> worker class, for the engines this deployment can run.

    Read from module globals on each call rather than frozen at import, so the
    enable flags stay patchable in tests and the map cannot drift from what was
    actually defined."""
    out = {}
    for engine, cls_name in _WORKER_CLASSES.items():
        cls = globals().get(cls_name)
        env = _engines.ENGINES[engine].enabled_env
        # the flag global mirrors the env var (NNSEG_FASTSURFER -> FASTSURFER)
        on = True if env is None else bool(globals().get(env[len("NNSEG_"):], False))
        if cls is not None and on:
            out[engine] = cls
    return out


def _spawn_worker(task: str, jid: str, source_tokens=None):
    """Dispatch to the worker for this task's engine.

    Routes on the *grammar* - every wire form is canonicalized to ``eco:task``
    before it gets here - through the engine registry, so a new engine needs no
    branch: one registry row plus a worker class. An ecosystem with no engine
    entry (every nnU-Net catalog) falls through to the default engine."""
    engine = _engines.engine_for_task(task).name
    workers = _worker_classes()
    if engine not in workers:
        env = _engines.ENGINES[engine].enabled_env
        raise RuntimeError(f"the {engine} engine is not enabled on this "
                           f"deployment (set {env}=1 at deploy)")
    return workers[engine]().run_job.spawn(jid, source_tokens=source_tokens)


class ModalExecutor:
    """The :func:`nnseg.serve.create_app` executor protocol over Modal primitives."""

    # One lock per API container: a scratch_vol.reload() here discards other
    # requests' uncommitted upload writes (the api function runs many inputs
    # concurrently), so every reload and every upload-write+commit serialize
    # through it. The worker has its own _vol_lock for the same reason.
    volume_guard = threading.Lock()

    @functools.cached_property
    def content(self):
        """Where a PUT /v1/inputs lands, and what a {"kind": "input"} source is
        checked against at submit. The same volume the workers read, so content
        stored here is resolvable there."""
        return _content_store()

    @property
    def sources(self):
        from nnseg.sources import registry
        return registry(None)

    def submit_prepare(self, jid, jdir, task):
        meta = {"id": jid, "task": task, "options": {}, "kind": "prepare",
                "state": "queued", "created": time.time(), "source": []}
        jobs_dict[jid] = meta
        _spawn_worker(task, jid)
        return meta

    def artifact_state(self, key: str) -> str:
        m = jobs_dict.get(f"artifacts:{key}")
        return "pending" if isinstance(m, dict) and m.get("state") == "pending" else "absent"

    def cache_list(self):
        from nnseg.serve import ResultCache
        try:
            cache_vol.reload()
        except Exception:
            pass
        return ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).list()

    supports_push = False                    # SSE uses the server's poll branch
    accepting = True                         # Modal's backlog is the queue

    def new_job_dir(self):
        import uuid
        jid = uuid.uuid4().hex[:12]
        d = Path(SCRATCH_ROOT) / jid
        d.mkdir(parents=True, exist_ok=True)
        return jid, d

    _weights_reload_lock = threading.Lock()
    _weights_reloaded_at = 0.0
    _wv_cache: dict = {}                   # task -> (versions, stamped at)

    def _fresh_weights_versions(self, task):
        """weights_versions_of, but stale-proof: an API container's mounted
        weights volume is frozen at container start, so after the worker
        first-installs a task this side kept deriving weights=["unknown"] -
        every probe missed the re-keyed entry and every Prefer'd GET
        recomputed, for the container's remaining lifetime. On "unknown",
        reload the volume and re-derive - THROTTLED to once per 30 s per
        container: "unknown" is also the honest permanent answer for weights
        nnseg did not install (TS-installed, hand-copied), and an unthrottled
        version reloaded a multi-GB volume on every HEAD probe forever."""
        from nnseg.serve import weights_versions_of
        cls = type(self)
        cached = cls._wv_cache.get(task)
        if cached is not None and time.time() - cached[1] < 30.0:
            return cached[0]               # the listing derives per ENTRY -
                                           # without this that is a describe()
                                           # volume walk per row
        wv = weights_versions_of(self.segmenter, task)
        if any("unknown" in str(v) for v in wv):
            with cls._weights_reload_lock:
                if time.time() - cls._weights_reloaded_at < 30.0:
                    return wv
                cls._weights_reloaded_at = time.time()
                try:
                    weights_vol.reload()
                except Exception:
                    cls._wv_cache[task] = (wv, time.time())
                    return wv
            wv = weights_versions_of(self.segmenter, task)
        cls._wv_cache[task] = (wv, time.time())
        return wv

    def resource_key(self, identity: str, task: str, opts=None) -> str:
        from nnseg.serve import result_key
        return result_key((identity,), task, opts or {},
                          self._fresh_weights_versions(task))

    def submit(self, jid, jdir, input_path, task, options, *, source=None,
               identity=(), no_cache: bool = False, source_tokens=None,
               inputs: tuple = ()):
        # `inputs` (the role -> local path binding) is accepted for signature
        # parity with LocalExecutor and deliberately not forwarded: this executor
        # is stateless by construction, and the worker rebuilds the binding from
        # `source` - each entry carries its canonical role, and uploads were
        # written into the job dir under that role. Sending server-local paths
        # through a Dict to another container would be sending it a lie.
        from nnseg.serve import result_key
        with self.volume_guard:
            scratch_vol.commit()                # make any upload visible to the worker
        key = None
        if identity:
            key = result_key(identity, task, options,
                             self._fresh_weights_versions(task))
            if not no_cache:
                hit = self.cache_get(key)
                if hit is not None:
                    meta = {"id": jid, "task": task, "options": options,
                            "input_identity": list(identity), "state": "done",
                            "cached": True, "created": time.time(),
                            "started": time.time(), "finished": time.time(),
                            "result": hit[1], "cache_path": str(hit[0])}
                    jobs_dict[jid] = meta
                    return meta
        meta = {"id": jid, "task": task, "options": options,
                "source": list(source or [{"kind": "upload"}]),
                "input_identity": list(identity), "cache_key": key,
                "state": "queued", "created": time.time()}
        jobs_dict[jid] = meta
        if key:
            jobs_dict[f"inflight:{key}"] = jid
        call = _spawn_worker(task, jid, source_tokens)
        _emit(jid, {"call_id": call.object_id})   # merge, never clobber worker emits
        return meta

    def cache_get(self, key):
        from nnseg.serve import ResultCache
        try:
            cache_vol.reload()
        except Exception:
            pass
        return ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).get(key)

    def find_inflight(self, key):
        jid = jobs_dict.get(f"inflight:{key}")
        if not jid:
            return None
        meta = jobs_dict.get(jid) or {}
        return jid if meta.get("state") in ("queued", "running") else None

    def cache_delete(self, key):
        from nnseg.serve import ResultCache
        try:
            cache_vol.reload()
        except Exception:
            pass
        deleted = ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).delete(key)
        if deleted:
            cache_vol.commit()
        return deleted

    def status_of(self, jid):
        meta = jobs_dict.get(jid)
        if meta is None:
            return None
        keys = ("id", "task", "state", "created", "started", "finished",
                "progress", "error", "input_identity", "cached",
                # the result handle and the options its URL form depends on -
                # serve's job route turns these into `key` + `links`
                "cache_key", "options")
        d = {k: meta.get(k) for k in keys if meta.get(k) is not None}
        if meta.get("state") == "done" and meta.get("result") is not None:
            d["result"] = meta["result"]
        return d

    def statuses(self):
        out = []
        try:
            keys = list(jobs_dict.keys())
        except Exception:                    # keys() availability varies by client version
            return out
        for jid in keys:
            if ":" in str(jid):              # inflight:/artifacts:/cancel: markers
                continue
            try:
                s = self.status_of(jid)
            except Exception:                # one bad row must not truncate the listing
                continue
            if s:
                out.append({k: s.get(k) for k in
                            ("id", "task", "state", "created", "started", "finished")})
        return out

    def cancel(self, jid):
        meta = jobs_dict.get(jid)
        if meta is None:
            return None, False
        state = meta.get("state")
        if state in ("queued", "running"):
            call_id = meta.get("call_id")
            if call_id:
                try:
                    modal.FunctionCall.from_id(call_id).cancel()
                except Exception:
                    pass
            jobs_dict[f"cancel:{jid}"] = time.time()
            _emit(jid, {"state": "cancelled", "finished": time.time()})
            return "cancelled", False
        import shutil
        try:
            del jobs_dict[jid]
        except Exception:
            pass
        shutil.rmtree(Path(SCRATCH_ROOT) / jid, ignore_errors=True)
        scratch_vol.commit()
        return state, True

    def result_file(self, jid):
        from nnseg.serve import RESULT_NAME
        meta = jobs_dict.get(jid)
        if meta is None:
            return None, None
        if meta.get("cache_path"):
            p = Path(meta["cache_path"])
            try:
                cache_vol.reload()
            except Exception:
                pass
            return meta["state"], (p if p.exists() else None)
        with self.volume_guard:
            scratch_vol.reload()
        p = Path(SCRATCH_ROOT) / jid / RESULT_NAME
        return meta["state"], (p if p.exists() else None)


@app.function(cpu=2.0, memory=2048, scaledown_window=300, image=api_image,
              volumes={SCRATCH_ROOT: scratch_vol, WEIGHTS_ROOT: weights_vol,
                       CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol})
@modal.concurrent(max_inputs=100)
@modal.asgi_app(requires_proxy_auth=PROXY_AUTH)
def api():
    _pkg_dir()
    os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
    from nnseg import Segmenter
    from nnseg.serve import create_app

    ex = ModalExecutor()
    # catalog/describe only - jobs run on the Worker; device string is cosmetic here
    ex.segmenter = Segmenter(device="cpu", weights=WEIGHTS_ROOT)
    return create_app(ex)


if PUBLIC:
    @app.function(cpu=1.0, memory=1024, scaledown_window=300, image=api_image,
                  volumes={CACHE_ROOT: cache_vol, WEIGHTS_ROOT: weights_vol})
    @modal.concurrent(max_inputs=100)
    @modal.asgi_app(requires_proxy_auth=False)
    def public():
        """The anonymous read-only twin (NNSEG_PUBLIC=1): cache hits only, no
        compute path in the function at all - it cannot spend GPU by
        construction. Shares the cache volume with the authed api."""
        _pkg_dir()
        os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
        from nnseg import Segmenter
        from nnseg.serve import (ResultCache, create_public_app, result_key,
                                 weights_versions_of)
        seg = Segmenter(device="cpu", weights=WEIGHTS_ROOT)
        cache = ResultCache(CACHE_ROOT, keep=RESULTS_KEEP)

        def key_fn(identity, task, opts=None):
            return result_key((identity,), task, opts or {},
                              weights_versions_of(seg, task))

        def get(key):
            try:
                cache_vol.reload()
            except Exception:
                pass
            return cache.get(key)

        def inflight(key):
            jid = jobs_dict.get(f"inflight:{key}")
            if not jid:
                return None
            meta = jobs_dict.get(jid) or {}
            if meta.get("state") not in ("queued", "running"):
                return None
            return {"progress": meta.get("progress")}

        def list_fn():
            try:
                cache_vol.reload()
            except Exception:
                pass
            return ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).list()

        return create_public_app(key_fn, get, seg.tasks, inflight=inflight,
                                 list_fn=list_fn, resolve_fn=seg.resolve_task)
