"""Run a ranked emit on Modal, for cases too large to do locally.

`ts:lung_vessels` is a cascade whose fine stage runs at NATIVE resolution, so on a 709x768x768
series it wants a K-channel field over hundreds of millions of voxels - hours on MPS, and a real
chance of exhausting a 17 GB laptop. On an L40S it is minutes.

Nothing is uploaded. The worker fetches the DICOM series straight from IDC by crdc_series_uuid,
which is a same-cloud transfer, and returns only the encoded arrays - a few megabytes against a
hundreds-of-megabytes input. That is the whole reason this is worth doing remotely rather than
shipping a volume around.

The image and the weights Volume are the deployed app's own, so the weights provisioned by any
previous run are reused rather than re-downloaded.

usage:
  uv run --no-project modal run tools/ranked_emit_modal.py \
      --identifier <crdc-series-uuid> --tasks ts:total,ts:body \
      --subject <name> --workdir <dir> [--source idc|openneuro]

`identifier` is whatever the chosen source door names an input by: a crdc_series_uuid for `idc`,
`ds<number>/<path>` for `openneuro`. Tasks are comma-separated and run in ONE call, so the fetch,
the container and the weights volume are paid for once rather than per task.

Then build the store from the downloaded directory exactly as for a local emit:
  uv run python tools/ranked_build_store.py <out> <store>.duckn <subject> last
"""
import os
import shutil
import sys
from pathlib import Path

import modal

PKG = Path(__file__).resolve().parent.parent / "src" / "nnseg"
TOOLS = Path(__file__).resolve().parent
WEIGHTS_ROOT = "/weights"

# Same recipe as src/nnseg/modal_app.py's base image: deps from pyproject extras, nnseg MOUNTED
# rather than installed so the running checkout is what executes. `idc` brings obstore for the
# fetch; `cuda` brings the Triton restore backend.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(extras=["torch", "idc", "cuda"], frozen=False,
             extra_options="--no-sources-package nnunetv2")
    .add_local_dir(str(PKG), remote_path="/root/pkg/nnseg")
    .add_local_file(str(TOOLS / "ranked_emit.py"), remote_path="/root/ranked_emit.py")
)

# FastSurfer needs its own image: it pins numpy/torch ranges that conflict with the torch
# extra, which is why engines get separate environments here at all. Recipe mirrors
# src/nnseg/modal_app.py's fs_image, INCLUDING baking the ~67 MB VINN checkpoints at build so
# cold containers never depend on Zenodo being up. Kept as a copy rather than an import,
# because importing modal_app would construct its App, Volumes and Dict as a side effect.
fs_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(extras=["fastsurfer", "idc"], frozen=False,
             extra_options="--no-sources-package nnunetv2")
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
    .add_local_dir(str(PKG), remote_path="/root/pkg/nnseg")
    .add_local_file(str(TOOLS / "ranked_emit_fastsurfer.py"),
                    remote_path="/root/ranked_emit_fastsurfer.py")
)

app = modal.App("nnseg-ranked-emit", image=image)
weights_vol = modal.Volume.from_name("nnseg-weights", create_if_missing=True)


def _fetch(source: str, identifier: str, dest):
    """Pull one input on the worker, by whichever nnseg source door names it.

    `idc` takes a crdc_series_uuid and yields a DICOM series directory; `openneuro` takes
    `ds<number>/<path>` off the public S3 bucket and yields a single file; `zenodo` takes
    `<recid>/<file>[!member]` and reads one member out of a remote zip by Range. All are
    cloud-to-cloud transfers, which is the whole reason the input is never uploaded from here.
    """
    from pathlib import Path as _P
    from nnseg import sources as _s
    dest = _P(dest)
    dest.mkdir(parents=True, exist_ok=True)      # IDCSource makes <dest>/series without parents
    door = {"idc": _s.IDCSource(), "openneuro": _s.openneuro_source(),
            "zenodo": _s.ZenodoSource()}[source]
    got = _P(door.fetch(identifier, dest))
    # An archive door lands a single NIfTI/NRRD member in a directory; the emitter wants
    # the file itself, and only a DICOM series is a directory.
    if got.is_dir():
        files = [p for p in got.iterdir() if p.is_file() and not p.name.startswith(".")]
        if len(files) == 1 and files[0].name.endswith((".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd")):
            got = files[0]
    return got


@app.function(gpu=["L40S", "A100-40GB", "A10G"], timeout=7200, volumes={WEIGHTS_ROOT: weights_vol})
def emit(identifier: str, tasks: list[str], depth: int = 6, clip: float = 8.0,
         envelope_mm=None, source: str = "idc") -> bytes:
    """Fetch once, run every task, return one gzipped tar with a directory per task.

    Many tasks per call rather than one, because everything except the network is shared: the
    DICOM fetch, the container start, the CUDA context, and the weights volume. Twelve separate
    calls would pay all of that twelve times.
    """
    import io
    import shutil
    import tarfile
    import tempfile
    import time

    sys.path.insert(0, "/root/pkg")
    sys.path.insert(0, "/root")
    os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT

    import torch
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)

    from nnseg.ecosystems import EcosystemCatalog

    work = Path(tempfile.mkdtemp())
    t = time.perf_counter()
    series = _fetch(source, identifier, work / "in")
    n = len(list(Path(series).iterdir())) if Path(series).is_dir() else 1
    if not n:
        raise RuntimeError(f"{source} returned nothing for {identifier!r} - wrong id, or it "
                           "is not reachable from this worker")
    print(f"fetched {n} file(s) in {time.perf_counter() - t:.0f}s", flush=True)

    import ranked_emit
    root = work / "out"
    root.mkdir()
    done, failed = [], []
    for task in tasks:
        short = task.split(":", 1)[-1]
        out = root / short
        t = time.perf_counter()
        try:
            EcosystemCatalog(root=WEIGHTS_ROOT).prepare(task)
            weights_vol.commit()
            ranked_emit.main(str(series), task, str(out), depth, clip,
                             "none" if envelope_mm is None else envelope_mm)
        except Exception as exc:                       # noqa: BLE001
            # one task failing must not lose the others - they are the expensive part
            print(f"!! {task}: {exc.__class__.__name__}: {exc}", flush=True)
            shutil.rmtree(out, ignore_errors=True)
            failed.append(task)
            continue
        if not (out / "meta.json").exists():
            print(f"!! {task}: no meta.json, dropping", flush=True)
            shutil.rmtree(out, ignore_errors=True)
            failed.append(task)
            continue
        done.append(short)
        print(f"== {task} in {time.perf_counter() - t:.0f}s", flush=True)

    if not done:
        raise RuntimeError(f"every task failed: {failed}")
    if failed:
        print(f"!! {len(failed)} of {len(tasks)} failed: {failed}", flush=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for d in sorted(root.iterdir()):
            tf.add(d, arcname=d.name)
    shutil.rmtree(work, ignore_errors=True)
    data = buf.getvalue()
    print(f"returning {len(data) / 1e6:.1f} MB (gzipped tar)", flush=True)
    return data


# A fallback list, not one type: a queued emit during an L40S capacity drought is worse than a
# slightly slower GPU, and FastSurfer's 2.5-D networks are far from needing an L40S anyway.
@app.function(gpu=["L40S", "A10G", "A100-40GB"], timeout=3600, image=fs_image)
def emit_brain(identifier: str, depth: int = 6, clip: float = 8.0,
               source: str = "openneuro") -> bytes:
    """The FastSurfer engine, on its own image. Returns a tar with one `brain/` directory.

    Separate from `emit` because the engine needs a different environment, not because the work
    differs - it still fetches on the worker and returns only the encoded arrays. `device=cuda`
    here: the MPS fallback the local path needs (no `max_unpool2d` on MPS) is irrelevant.
    """
    import io
    import shutil
    import tarfile
    import tempfile
    import time

    sys.path.insert(0, "/root/pkg")
    sys.path.insert(0, "/root")

    import torch
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)

    work = Path(tempfile.mkdtemp())
    t = time.perf_counter()
    got = _fetch(source, identifier, work / "in")
    print(f"fetched in {time.perf_counter() - t:.0f}s -> {got}", flush=True)

    out = work / "out" / "brain"
    import ranked_emit_fastsurfer
    ranked_emit_fastsurfer.main(str(got), str(out), depth, clip, "cuda")
    if not (out / "meta.json").exists():
        raise RuntimeError("fastsurfer emit produced no meta.json")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(out, arcname="brain")
    shutil.rmtree(work, ignore_errors=True)
    data = buf.getvalue()
    print(f"returning {len(data) / 1e6:.1f} MB", flush=True)
    return data


@app.local_entrypoint()
def main(identifier: str, tasks: str, subject: str, workdir: str,
         depth: int = 6, clip: float = 8.0, source: str = "idc"):
    """`tasks` is comma-separated (`ts:total,ts:body`); each lands in
    `<workdir>/ranked_<subject>_<task>`."""
    import io
    import tarfile

    want = [x.strip() for x in tasks.split(",") if x.strip()]
    work = Path(workdir)
    staging = work / f".partial_{subject}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    # `fastsurfer:brain` is a different engine on a different image; everything else is the
    # nnU-Net path. Dispatch here rather than making the caller know.
    if want == ["fastsurfer:brain"]:
        data = emit_brain.remote(identifier, depth, clip, source)
    else:
        data = emit.remote(identifier, want, depth, clip, None, source)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(staging)

    # Publish each task only if its own meta.json arrived. Staged then renamed, because the
    # builder reads the presence of meta.json as "this emit is finished" - a half-written
    # directory must never be visible under the real name.
    print()
    for d in sorted(staging.iterdir()):
        dest = work / f"ranked_{subject}_{d.name}"
        if not (d / "meta.json").exists():
            print(f"  {d.name:<16} no meta.json - not published")
            continue
        shutil.rmtree(dest, ignore_errors=True)
        d.rename(dest)
        mb = sum(f.stat().st_size for f in dest.iterdir()) / 1e6
        print(f"  {d.name:<16} -> {dest.name}  ({mb:.1f} MB)")
    shutil.rmtree(staging, ignore_errors=True)
    missing = [t for t in want
               if not (work / f"ranked_{subject}_{t.split(':')[-1]}").exists()]
    if missing:
        raise SystemExit(f"missing after publish: {missing}")
