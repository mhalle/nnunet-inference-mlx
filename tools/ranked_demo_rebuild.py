"""Regenerate the demo ranked stores through the current builder.

The recipe for one particular set of cases - unlike its siblings, this file is NOT generic. It
records which emit directory becomes which store, and which store needs padding. The generic
work is all in `ranked_build_store.py` and `ranked_align_parts.py`.

Rebuild rather than convert: the builder is the single place that decides layout (chunking,
sharding, the README), so a store that came out of it is by construction what a fresh run would
produce. Converting stores in place would leave the next build disagreeing with them.

`idc-torso1_total` takes an extra step: its five parts were computed on four different body
envelopes - the envelope is recomputed per model, because the body threshold depends on that
model's normalization - so it is built and then padded onto the shared model grid.

usage: uv run python tools/ranked_demo_rebuild.py [WORKDIR]

WORKDIR holds the `ranked_*` emit directories and receives `duckn_demo/`; it defaults to the
current directory. Nothing is read from the repo except the sibling tools.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
DEMO = WORK / "duckn_demo"

# (emit dir, store name, case, needs padding onto the shared model grid)
JOBS = [("ranked_idc_fast", "idc-torso1_total_fast.duckn", "idc-torso1", False),
        ("ranked_idc_total", "idc-torso1_total.duckn", "idc-torso1", True),
        ("ranked_abdo_fast", "CT_Abdo_total_fast.duckn", "CT_Abdo.nii.gz", False),
        ("ranked_abdo_total", "CT_Abdo_total.duckn", "CT_Abdo.nii.gz", False),
        ("ranked_brain", "ds000114_sub-01_brain.duckn", "ds000114_sub-01", False)]


def run(script, *args):
    subprocess.run([sys.executable, str(TOOLS / script), *args], check=True)


def count(d: Path):
    fs = [f for f in d.rglob("*") if f.is_file()]
    js = [f for f in fs if f.name == "zarr.json"]
    sz = np.array([f.stat().st_size for f in fs])
    return len(fs) - len(js), len(js), sz.sum(), int(np.ceil(sz / 4096).sum() * 4096)


print(f"workdir {WORK}")
for src, name, case, pad in JOBS:
    s = WORK / src
    if not s.exists():
        print(f"{src}: MISSING - skipped\n")
        continue
    dst = DEMO / name
    print(f"=== {name}  (from {src}){'  + align' if pad else ''}")
    if pad:
        tmp = DEMO / (name + ".unaligned")
        run("ranked_build_store.py", str(s), str(tmp), case)
        run("ranked_align_parts.py", str(tmp), str(dst))
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        run("ranked_build_store.py", str(s), str(dst), case)

print("=== provenance")
run("ranked_demo_provenance.py", str(DEMO))

print(f"\n{'store':<32}{'shards':>8}{'json':>6}{'logical':>10}{'on disk':>10}")
tf = 0
for _, name, _, _ in JOBS:
    d = DEMO / name
    if not d.exists():
        continue
    s_, j, lo, al = count(d)
    tf += s_ + j
    print(f"{name:<32}{s_:>8}{j:>6}{lo/1e6:>9.2f}M{al/1e6:>9.2f}M")
print(f"\ntotal files across all stores: {tf}")
