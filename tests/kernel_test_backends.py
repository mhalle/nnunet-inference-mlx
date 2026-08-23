"""Every backend against the float64 reference, and against each other."""
import numpy as np
import pytest
import torch

import nnseg as lg
from nnseg import Mapping, build_tables, reference
from nnseg.backends import metal

from conftest import assert_agree_up_to_ties, voronoi_logits


def _backends_for(device):
    from nnseg.backends import triton_gpu
    names = ["torch"]
    if device.type == "mps" and metal.available():
        names.append("metal")
    if device.type == "cuda" and triton_gpu.available():
        names.append("triton")
    return names


CASES = {
    # name: (src_shape, out_shape, mapping-maker, interp, outside)
    "upsample_iso": ((9, 11, 13), (18, 22, 26), lambda s, o: Mapping.spacing((1.5,) * 3, (3.0,) * 3), "linear", "background"),
    "mixed_up_down": ((12, 10, 14), (7, 25, 30), lambda s, o: Mapping.center(o, s), "linear", "background"),
    "nearest_z": ((12, 10, 14), (7, 25, 30), lambda s, o: Mapping.center(o, s), ("nearest", "linear", "linear"), "background"),
    "corner": ((9, 11, 13), (17, 14, 31), lambda s, o: Mapping.corner(o, s), "linear", "background"),
    "outside_bg": ((9, 11, 13), (24, 26, 30), lambda s, o: Mapping((0.5, 0.5, 0.5), (-2.0, -1.0, -3.0)), "linear", "background"),
    "outside_clamp": ((9, 11, 13), (24, 26, 30), lambda s, o: Mapping((0.5, 0.5, 0.5), (-2.0, -1.0, -3.0)), "linear", "clamp"),
    "all_nearest_down": ((16, 15, 14), (5, 6, 7), lambda s, o: Mapping.corner(o, s), "nearest", "background"),
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("logit_dtype", [torch.float32, torch.float16])
def test_backends_match_reference(device, name, logit_dtype):
    src_shape, out_shape, make, interp, outside = CASES[name]
    mapping = make(src_shape, out_shape)
    logits_np = voronoi_logits(K=7, shape=src_shape, seed=hash(name) % 1000)
    logits_t = torch.from_numpy(logits_np).to(device=device, dtype=logit_dtype)
    # the reference sees what the backend sees (fp16 storage)
    ref_input = logits_t.float().cpu().numpy()
    tables = build_tables(out_shape, src_shape, mapping, interp=interp, outside=outside)
    values, valid = reference.interpolate(ref_input, tables)
    want = reference.decide(values, valid, background=3)
    results = {}
    for backend in _backends_for(device):
        got = lg.to_labels(logits_t, out_shape, mapping, interp=interp, outside=outside, background=3, backend=backend)
        assert got.shape == out_shape and got.dtype == torch.uint8 and got.device == logits_t.device
        results[backend] = got.cpu().numpy()
        assert_agree_up_to_ties(results[backend], want, values, what=f"{backend}/{name}/{logit_dtype}")
        # outside voxels: exactly background, never a tie question
        np.testing.assert_array_equal(results[backend][~valid], 3)
    if "metal" in results:
        assert_agree_up_to_ties(results["metal"], results["torch"], values, what=f"metal-vs-torch/{name}")


def test_numpy_input_and_shape_errors():
    logits = voronoi_logits(K=4, shape=(5, 6, 7))
    out = lg.to_labels(logits, (9, 10, 11), Mapping.center((9, 10, 11), (5, 6, 7)), backend="torch")
    assert isinstance(out, torch.Tensor) and out.shape == (9, 10, 11)
    with pytest.raises(ValueError):
        lg.to_labels(logits[0], (9, 10, 11), Mapping.identity())
    with pytest.raises(ValueError):
        lg.to_labels(logits, (9, 10, 11), Mapping.identity(), lut=[1, 2, 3])
    with pytest.raises(ValueError):
        lg.to_labels(logits, (9, 10, 11), Mapping.identity(), mode="softmax")
    with pytest.raises(ValueError):
        lg.to_labels(logits, (9, 10, 11), Mapping.identity(), out=torch.zeros((1, 2, 3), dtype=torch.uint8))
    with pytest.raises(ValueError):
        lg.to_labels(logits, (9, 10, 11), Mapping.identity(), lut=[0, 1, 2, 300], out_dtype=torch.uint8)
    assert lg.to_labels(logits, (9, 10, 11), Mapping.identity(), lut=[0, 1, 2, 300]).dtype == torch.uint16


def test_slab_launches_cover_everything(device):
    if device.type != "mps" or not metal.available():
        pytest.skip("metal only")
    src_shape, out_shape = (9, 11, 13), (18, 22, 26)
    mapping = Mapping.spacing((1.5,) * 3, (3.0,) * 3)
    logits = torch.from_numpy(voronoi_logits(K=6, shape=src_shape)).to(device)
    whole = lg.to_labels(logits, out_shape, mapping, backend="metal").cpu().numpy()
    slabbed = lg.to_labels(logits, out_shape, mapping, backend="metal", slab_voxels=22 * 26 * 3 + 5).cpu().numpy()
    np.testing.assert_array_equal(whole, slabbed)


def test_metal_fp_contract_state():
    if not metal.available():
        pytest.skip("no MPS")
    assert metal.fp_contract() in ("off", "default")


def test_available_backends():
    """torch is always there; the fused backends appear exactly where their hardware does."""
    from nnseg.backends import triton_gpu
    names = lg.available_backends()
    assert "torch" in names
    assert ("metal" in names) == (metal.available())
    assert ("triton" in names) == (triton_gpu.available())
