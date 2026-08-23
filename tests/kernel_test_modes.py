"""LUT, paint compositing, region mode, output dtypes."""
import numpy as np
import pytest
import torch

import nnseg as lg
from nnseg import Mapping, build_tables, reference
from nnseg.backends import metal

from conftest import voronoi_logits


def _backends_for(device):
    return ["torch"] + (["metal"] if device.type == "mps" and metal.available() else [])


def _tie_free(mism, margin_list, tol=1e-4):
    """a mismatch is acceptable only where some contributing part had a near-tie"""
    near = np.zeros_like(mism)
    for m in margin_list:
        near |= m < tol
    return not (mism & ~near).any()


@pytest.mark.parametrize("backend_filter", [None])
def test_lut_and_paint_composite_two_parts(device, backend_filter):
    src_shape, out_shape = (10, 12, 14), (19, 23, 27)
    mapping = Mapping.center(out_shape, src_shape)
    part_a = voronoi_logits(K=5, shape=src_shape, seed=11)
    part_b = voronoi_logits(K=4, shape=src_shape, seed=22)
    lut_a = [0, 1, 2, 3, 4]
    lut_b = [0, 10, 11, 12]
    tables = build_tables(out_shape, src_shape, mapping)
    va, valid = reference.interpolate(part_a, tables)
    vb, _ = reference.interpolate(part_b, tables)
    want = np.zeros(out_shape, dtype=np.int64)
    reference.decide(va, valid, lut=lut_a, paint=True, out=want)
    reference.decide(vb, valid, lut=lut_b, paint=True, out=want)
    assert (want == 0).any() and (want >= 10).any() and ((want > 0) & (want < 10)).any()
    ta = torch.from_numpy(part_a).to(device)
    tb = torch.from_numpy(part_b).to(device)
    for backend in _backends_for(device):
        out = torch.zeros(out_shape, dtype=torch.uint8, device=device)
        r1 = lg.to_labels(ta, out_shape, mapping, lut=lut_a, paint=True, out=out, backend=backend)
        assert r1 is out
        lg.to_labels(tb, out_shape, mapping, lut=lut_b, paint=True, out=out, backend=backend)
        got = out.cpu().numpy().astype(np.int64)
        mism = got != want
        assert _tie_free(mism, [reference.margins(va), reference.margins(vb)]), f"{backend}: non-tie mismatches"
        assert mism.mean() < 2e-3


def test_paint_leaves_outside_untouched(device):
    src_shape, out_shape = (9, 11, 13), (24, 26, 30)
    mapping = Mapping((0.5, 0.5, 0.5), (-2.0, -1.0, -3.0))
    logits = torch.from_numpy(voronoi_logits(K=5, shape=src_shape)).to(device)
    tables = build_tables(out_shape, src_shape, mapping)
    _, valid = reference.interpolate(logits.cpu().numpy(), tables)
    assert not valid.all()
    for backend in _backends_for(device):
        out = torch.full(out_shape, 99, dtype=torch.uint8, device=device)
        lg.to_labels(logits, out_shape, mapping, paint=True, out=out, backend=backend)
        got = out.cpu().numpy()
        assert (got[~valid] == 99).all()
        assert (got[valid] != 99).any()


def test_regions_mode_matches_reference(device):
    src_shape, out_shape = (8, 10, 12), (15, 21, 25)
    mapping = Mapping.corner(out_shape, src_shape)
    rng = np.random.default_rng(5)
    # three overlapping sigmoid heads: smooth fields crossing 0
    base = voronoi_logits(K=3, shape=src_shape, seed=7, noise=0.0)
    logits = (base + rng.normal(0, 0.2, base.shape)).astype(np.float32)
    labels = [4, 2, 9]                      # paint order: 9 wins overlaps
    tables = build_tables(out_shape, src_shape, mapping)
    v, valid = reference.interpolate(logits, tables)
    want = reference.decide(v, valid, lut=labels, mode="regions", threshold=0.0, background=0)
    assert len(np.unique(want)) == 4
    near = (np.abs(v) < 1e-4).any(0)        # a value within tolerance of the threshold is a tie
    t = torch.from_numpy(logits).to(device)
    for backend in _backends_for(device):
        got = lg.resample_paint(t, out_shape, mapping, labels, threshold=0.0, backend=backend).cpu().numpy()
        mism = got != want
        assert not (mism & ~near).any(), f"{backend}: non-tie region mismatches"
        # paint mode: no-region voxels untouched
        out = torch.full(out_shape, 77, dtype=torch.uint8, device=device)
        lg.to_labels(t, out_shape, mapping, lut=labels, mode="regions", paint=True, out=out, backend=backend)
        got2 = out.cpu().numpy()
        hit = (v > 0.0).any(0)
        assert (got2[~hit & ~near] == 77).all()
        assert (got2[hit & ~near] == want[hit & ~near]).all()


def test_uint16_labels(device):
    src_shape, out_shape = (6, 7, 8), (11, 13, 15)
    mapping = Mapping.center(out_shape, src_shape)
    logits = torch.from_numpy(voronoi_logits(K=4, shape=src_shape, seed=3)).to(device)
    lut = [0, 300, 1000, 65535]
    tables = build_tables(out_shape, src_shape, mapping)
    v, valid = reference.interpolate(logits.cpu().numpy(), tables)
    want = reference.decide(v, valid, lut=lut)
    margins = reference.margins(v)
    for backend in _backends_for(device):
        got = lg.to_labels(logits, out_shape, mapping, lut=lut, backend=backend)
        assert got.dtype == torch.uint16
        g = got.cpu().numpy().astype(np.int64)
        assert not ((g != want) & (margins > 1e-4)).any()
        assert set(np.unique(g)) <= set(lut)


def test_background_value_and_argmax_alias(device):
    src_shape, out_shape = (6, 7, 8), (20, 20, 20)
    mapping = Mapping((0.5, 0.5, 0.5), (-3.0, -3.0, -3.0))
    logits = torch.from_numpy(voronoi_logits(K=4, shape=src_shape, seed=3)).to(device)
    tables = build_tables(out_shape, src_shape, mapping)
    _, valid = reference.interpolate(logits.cpu().numpy(), tables)
    for backend in _backends_for(device):
        got = lg.resample_argmax(logits, out_shape, mapping, background=7, backend=backend).cpu().numpy()
        assert (got[~valid] == 7).all()
        assert (got[valid] < 4).all()


def test_metal_rejects_bf16():
    if not metal.available():
        pytest.skip("no MPS")
    logits = torch.from_numpy(voronoi_logits(K=3, shape=(4, 5, 6))).to("mps").bfloat16()
    with pytest.raises(TypeError):
        lg.to_labels(logits, (8, 10, 12), Mapping.center((8, 10, 12), (4, 5, 6)), backend="metal")
    # the torch backend takes it (computes in fp32)
    lg.to_labels(logits, (8, 10, 12), Mapping.center((8, 10, 12), (4, 5, 6)), backend="torch")
