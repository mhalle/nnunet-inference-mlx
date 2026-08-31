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
      --crdc <series-uuid> --task ts:lung_vessels --out ranked_<subject>_lung_vessels

Then build the store from the downloaded directory exactly as for a local emit:
  uv run python tools/ranked_build_store.py <out> <store>.duckn <subject> last
"""
import os
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

app = modal.App("nnseg-ranked-emit", image=image)
weights_vol = modal.Volume.from_name("nnseg-weights", create_if_missing=True)


@app.function(gpu="L40S", timeout=3600, volumes={WEIGHTS_ROOT: weights_vol})
def emit(crdc: str, task: str, depth: int = 6, clip: float = 8.0,
         envelope_mm=None) -> bytes:
    """Fetch, segment, encode; return the emit directory as a gzipped tar."""
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
    from nnseg.sources import IDCSource

    work = Path(tempfile.mkdtemp())
    t = time.perf_counter()
    # IDCSource.fetch makes <dest_dir>/series without parents=True, so dest_dir must exist
    (work / "in").mkdir(parents=True, exist_ok=True)
    series = IDCSource().fetch(crdc, work / "in")
    n = len(list(Path(series).iterdir()))
    print(f"fetched {n} instances in {time.perf_counter() - t:.0f}s", flush=True)

    t = time.perf_counter()
    EcosystemCatalog(root=WEIGHTS_ROOT).prepare(task)     # provision into the shared volume
    weights_vol.commit()
    print(f"weights ready in {time.perf_counter() - t:.0f}s", flush=True)

    out = work / "out"
    import ranked_emit
    ranked_emit.main(str(series), task, str(out), depth, clip,
                     "none" if envelope_mm is None else envelope_mm)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(out.iterdir()):
            tf.add(f, arcname=f.name)
    shutil.rmtree(work, ignore_errors=True)
    data = buf.getvalue()
    print(f"returning {len(data) / 1e6:.1f} MB (gzipped tar)", flush=True)
    return data


@app.local_entrypoint()
def main(crdc: str, task: str, out: str, depth: int = 6, clip: float = 8.0):
    import io
    import tarfile

    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    data = emit.remote(crdc, task, depth, clip, None)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(dest)
    got = sorted(p.name for p in dest.iterdir())
    print(f"\nunpacked {len(got)} files into {dest}:")
    for g in got:
        print(f"  {g}  {(dest / g).stat().st_size / 1e6:.1f} MB")
