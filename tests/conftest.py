"""Shared fixtures for the kernel-layer tests: the device matrix, synthetic logits, and the
tie-aware comparison. Auto-loaded by pytest.
"""
import numpy as np
import pytest
import torch
from scipy.ndimage import uniform_filter

from nnseg import reference


def device_names():
    names = ["cpu"]
    if torch.backends.mps.is_available():
        names.append("mps")
    if torch.cuda.is_available():
        names.append("cuda")
    return names


@pytest.fixture(params=device_names())
def device(request):
    return torch.device(request.param)


def voronoi_logits(K=9, shape=(14, 18, 22), n_regions=20, seed=0, noise=0.3, smooth=True):
    """Small anatomically-shaped synthetic logits: Voronoi blobs, +4 inside / -4
    outside per class, box-smoothed so boundaries are soft, plus noise."""
    rng = np.random.default_rng(seed)
    z, y, x = shape
    seeds = rng.uniform(0, 1, size=(n_regions, 3)) * np.array([z, y, x])
    seed_label = rng.integers(1, K, n_regions)
    seed_label[: max(1, n_regions // 3)] = 0
    gz, gy, gx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    pts = np.stack([gz, gy, gx], -1).reshape(-1, 3).astype(np.float64)
    d = ((pts[:, None, :] - seeds[None]) ** 2).sum(-1)
    lab = seed_label[d.argmin(1)].reshape(shape)
    logits = np.where(lab[None] == np.arange(K)[:, None, None, None], 4.0, -4.0)
    if smooth:
        logits = uniform_filter(logits, size=(1, 3, 3, 3), mode="nearest")
        logits = uniform_filter(logits, size=(1, 3, 3, 3), mode="nearest")
    if noise:
        logits = logits + rng.normal(0, noise, logits.shape)
    return logits.astype(np.float32)


def assert_agree_up_to_ties(got, want, values, *, tol=1e-4, max_fraction=2e-3, what=""):
    """Backends compute in float32 and may round differently; every
    disagreement must be a genuine near-tie of the float64 reference."""
    got = np.asarray(got).astype(np.int64)
    want = np.asarray(want).astype(np.int64)
    assert got.shape == want.shape, (got.shape, want.shape)
    mism = got != want
    n = int(mism.sum())
    if n == 0:
        return 0
    m = reference.margins(values)
    bad = mism & (m > tol)
    assert not bad.any(), (f"{what}: {int(bad.sum())} of {n} mismatches are not ties "
                           f"(largest margin {m[mism].max():.3g})")
    assert mism.mean() <= max_fraction, f"{what}: {n} tie mismatches = {mism.mean():.2e} > {max_fraction}"
    return n
