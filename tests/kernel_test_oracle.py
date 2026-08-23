"""Parity with the nnunet-inference-mlx fused kernel on the E3 abdo case.

Needs the E3 fixtures (bench/e3_cases.py) and LABELGRID_ORACLE=1; ~0.7 GB of
logits on the GPU, so run it through bench/_guard.py on the 16 GB machine.
"""
import os

import numpy as np
import pytest
import torch

import nnseg as lg
from nnseg import Mapping
from nnseg.backends import metal

E3_DIR = os.path.join(os.environ.get(
    "MEDSEG_SCRATCH",
    "/private/tmp/claude-501/-Users-halazar-Dropbox-development-medseg/e3eae47b-436c-4613-bbab-5e962500312c/scratchpad"),
    "e3")
LOGITS = os.path.join(E3_DIR, "logits_118x127x88x127_v2_f32.npy")
ORACLE = os.path.join(E3_DIR, "labels_abdo_native_mlx_fused.npy")


@pytest.mark.slow
def test_abdo_native_matches_mlx_fused():
    if os.environ.get("LABELGRID_ORACLE") != "1":
        pytest.skip("set LABELGRID_ORACLE=1 (run under bench/_guard.py)")
    if not (os.path.exists(LOGITS) and os.path.exists(ORACLE)):
        pytest.skip("E3 fixtures not present")
    if not metal.available():
        pytest.skip("no MPS")
    logits = torch.from_numpy(np.load(LOGITS, mmap_mode="r")[:]).to("mps")
    out_shape = (256, 178, 255)
    mapping = Mapping.spacing((1.49,) * 3, (3.0,) * 3)            # the MLX kernel's s2t = acq / target
    got = lg.to_labels(logits, out_shape, mapping, outside="clamp", coord_dtype=np.float32,
                       backend="metal").cpu().numpy()
    want = np.load(ORACLE, mmap_mode="r")
    mism = int((got != np.asarray(want)).sum())
    print(f"mismatch {mism} / {got.size} ({mism / got.size * 100:.6f} %), fp contract {metal.fp_contract()}")
    assert mism / got.size < 1e-4


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mlx_fused_oracle.npz")


def _fixture_cases():
    import json
    data = np.load(FIXTURE)
    cases = json.loads(str(data["cases"]))
    return [(name, c, data[f"{name}_logits"], data[f"{name}_labels"]) for name, c in cases.items()]


@pytest.mark.parametrize("coord_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["torch", "metal"])
def test_small_fixtures_match_mlx_fused_kernel(backend, coord_dtype):
    """Labels produced by nnunet-inference-mlx's fused Metal kernel
    (`inverse_resample_argmax`, oracle ref main@40ebe55) on two synthetic
    cases, generated 2026-08-22 and frozen here. Expected: identical."""
    if backend == "metal" and not metal.available():
        pytest.skip("no MPS")
    device = "mps" if backend == "metal" else "cpu"
    for name, c, logits, want in _fixture_cases():
        mapping = Mapping.spacing(c["acq"], c["target"])      # the MLX kernel's s2t = acq / target, clamped
        got = lg.to_labels(torch.from_numpy(logits).to(device), tuple(c["out"]), mapping, outside="clamp",
                           coord_dtype=coord_dtype, backend=backend).cpu().numpy()
        mism = int((got != want).sum())
        assert mism == 0, f"{name}/{backend}/{coord_dtype.__name__}: {mism} voxels differ from the MLX kernel"
