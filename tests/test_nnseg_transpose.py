"""transpose_forward support, tested from nnU-Net's spec without an external
oracle: plans express spacing/patch in the transposed frame; nnseg keeps
everything outside the network canonical and permutes only around the net.
Oracle confirmation (e.g. moosez on the same case) is still pending - the
gate stays opt-in until then."""
import numpy as np
import pytest
import torch

from nnseg.network import canonical_spacing, inverse_perm


@pytest.mark.parametrize("fwd", [(0, 1, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0),
                                 (0, 2, 1), (1, 0, 2)])
def test_inverse_perm_round_trips(fwd):
    bwd = inverse_perm(fwd)
    assert tuple(fwd[b] for b in bwd) == (0, 1, 2)
    assert tuple(bwd[f] for f in fwd) == (0, 1, 2)


def test_canonical_spacing_maps_transposed_axes_home():
    # transposed axis k IS canonical axis fwd[k]; moose brain: fwd=(2, 0, 1)
    # means the model's first axis is canonical x, second is z, third is y.
    assert canonical_spacing((2.0, 1.0, 1.5), (2, 0, 1)) == (1.0, 1.5, 2.0)
    assert canonical_spacing((3.0, 0.5, 0.7), (0, 1, 2)) == (3.0, 0.5, 0.7)


@pytest.mark.parametrize("fwd", [(2, 0, 1), (1, 2, 0), (2, 1, 0)])
def test_permute_in_out_is_identity_around_any_pointwise_net(fwd):
    """The exact permutes predict_logits applies: x -> transposed frame ->
    (network) -> logits -> canonical. With a pointwise 'network' the round
    trip must be bit-identical, for every axis permutation."""
    bwd = inverse_perm(fwd)
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    xt = x.permute((0, *(f + 1 for f in fwd))).contiguous()
    back = xt.permute((0, *(b + 1 for b in bwd))).contiguous()
    assert torch.equal(back, x)
    # and an anisotropy-sensitive check: the transposed shape is the
    # canonical shape reordered by fwd, which is what the patch grid sees
    assert xt.shape[1:] == tuple(x.shape[1 + f] for f in fwd)


def test_spacing_and_shape_agree_through_the_permute():
    """The invariant that makes the whole scheme safe: after resampling to
    canonical_spacing and permuting by fwd, the per-axis physical spacing the
    network sees equals the plans spacing, axis for axis."""
    for fwd in [(2, 0, 1), (1, 2, 0), (0, 2, 1)]:
        plans_spacing = (2.0, 1.0, 1.5)
        canon = canonical_spacing(plans_spacing, fwd)
        seen = tuple(canon[f] for f in fwd)      # spacing of permuted axes
        assert seen == plans_spacing


def test_gate_message_names_the_opt_in(tmp_path):
    """The gate is re-armed pending oracle confirmation: the refusal must
    name allow_transpose so the caller knows the path exists."""
    import inspect

    from nnseg import network
    src = inspect.getsource(network)
    assert "pending confirmation" in src or "not yet confirmed" in src
    assert "allow_transpose=True" in src
