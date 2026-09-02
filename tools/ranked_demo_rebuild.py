"""Build the demo ranked stores, one directory per subject.

The recipe for one particular set of cases - unlike its siblings, this file is NOT generic. It
records which emit directory becomes which store. The generic work is all in
`ranked_build_store.py` and `ranked_align_parts.py`.

Layout is subject-major, so everything known about one subject sits together and the store name
is just the task:

    <store_dir>/<subject>/<task>.duckn

Subjects are named for the archive they came from, tasks for what the model does; neither name
encodes a body region, because a guessed region is the thing that goes stale and misleads.

Rebuild rather than convert: the builder is the single place that decides layout (chunking,
sharding, the README), so a store that came out of it is by construction what a fresh run would
produce.

`idc-torso1/total` takes an extra step: its five parts were computed on four different body
envelopes - the envelope is recomputed per model, because the body threshold depends on that
model's normalization - so it is built and then padded onto the shared model grid.

usage: uv run python tools/ranked_demo_rebuild.py [WORKDIR] [STORE_DIR]

WORKDIR holds the `ranked_*` emit directories and defaults to the current directory - it is
usually scratch, since the emit output is large and re-derivable. STORE_DIR receives the built
stores and defaults to `data/duckn_demo` in the repo, which is gitignored but inside Dropbox:
backed up without being versioned, since these are megabytes of rebuildable binary.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
DEMO = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else TOOLS.parent / "data" / "duckn_demo"

# Emit directories are `<WORKDIR>/ranked_<subject>_<task>`, so the table only names what is
# not derivable. `parts`: "last" keeps a cascade's fine stage only. `pad`: build then pad onto
# the shared model grid, needed where the parts were computed on different body envelopes.
CASCADES = {"lung_vessels", "liver_vessels", "liver_segments", "abdominal_muscles",
            "ventricle_parts", "pleural_pericard_effusion", "kidney_cysts", "lung_nodules",
            "heartchambers_highres", "coronary_arteries"}

CT_TASKS = ["total", "total_fast", "total_fastest", "body", "trunk_cavities",
            "vertebrae_body", "lung_vessels", "liver_vessels", "liver_segments"]

# Nothing needs padding any more. `idc-torso1/total` used to: its five parts were computed on
# four different body envelopes, so they had to be filled out onto a shared grid, which meant a
# store containing voxels the network never produced. Re-emitting with envelope=None removed
# both the step and the assertion - every part now IS the model grid. Kept as an empty set
# rather than deleted, because a store built from an envelope-cropped emit would need it again.
PAD: set[tuple[str, str]] = set()

JOBS = [(s, t, "last" if t in CASCADES else "all", (s, t) in PAD)
        for s in ("idc-torso1", "nlst-217076") for t in CT_TASKS]
JOBS.append(("ds000114_sub-01", "brain", "all", False))


def run(script, *args):
    subprocess.run([sys.executable, str(TOOLS / script), *args], check=True)


def count(d: Path):
    fs = [f for f in d.rglob("*") if f.is_file()]
    js = [f for f in fs if f.name == "zarr.json"]
    sz = np.array([f.stat().st_size for f in fs])
    return len(fs) - len(js), len(js), sz.sum()


print(f"workdir {WORK}\nstores  {DEMO}\n")
built = []
for subject, task, parts, pad in JOBS:
    src = f"ranked_{subject}_{task}"
    s = WORK / src
    # meta.json, not the directory: a remote emit creates its output directory before the run
    # lands in it, so an empty directory means "in flight", not "ready"
    if not (s / "meta.json").exists():
        print(f"{subject}/{task}: {src} not ready - skipped")
        continue
    dst = DEMO / subject / f"{task}.duckn"
    print(f"=== {subject}/{task}  (from {src}){'  + align' if pad else ''}")
    if pad:
        tmp = dst.with_suffix(".unaligned")
        run("ranked_build_store.py", str(s), str(tmp), subject, parts)
        run("ranked_align_parts.py", str(tmp), str(dst))
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        run("ranked_build_store.py", str(s), str(dst), subject, parts)
    built.append((subject, task, dst))

print("=== provenance")
run("ranked_demo_provenance.py", str(DEMO))

# Verify before reporting success. A build that "worked" has been wrong three times today -
# a dropped argument, degraded names, an index copied instead of rebuilt - and each was only
# visible against the spec. The cheap checks run always; --deep is a separate, slower pass.
print("=== verify")
run("ranked_verify.py", str(DEMO), "--all", "--quiet")

print(f"\n{'subject':<18}{'task':<16}{'shards':>8}{'json':>6}{'size':>10}")
tf = 0
for subject, task, d in built:
    s_, j, lo = count(d)
    tf += s_ + j
    print(f"{subject:<18}{task:<16}{s_:>8}{j:>6}{lo/1e6:>9.2f}M")
print(f"\ntotal files: {tf}")
