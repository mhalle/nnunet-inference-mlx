"""VoxTell engine: prompt handling, the axis-order contract, and label painting.

The model itself is never loaded - a fake predictor stands in - because what can go
silently wrong here is geometry and bookkeeping, not the network. Deliberately
non-cubic volumes: a transposed array is invisible on a cube.
"""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from nnseg.engines import voxtell as vt


class FakePredictor:
    """Records what it was handed and returns one mask per prompt.

    ``mask_fn(i, shape)`` builds prompt i's mask in VoxTell's own (X, Y, Z) frame.
    """

    def __init__(self, mask_fn=None):
        self.seen_shape = None
        self.seen_prompts = None
        self._mask_fn = mask_fn or (lambda i, shape: np.ones(shape, dtype=np.uint8))

    def predict_single_image(self, data, text_prompts, progress_callback=None, **kw):
        self.seen_shape, self.seen_prompts = data.shape, list(text_prompts)
        if progress_callback is not None:
            progress_callback(1, 1)
        return np.stack([self._mask_fn(i, data.shape) for i in range(len(text_prompts))])


#: SimpleITK's identity direction is LPS; this is RAS.
_RAS = (-1., 0., 0., 0., -1., 0., 0., 0., 1.)


def _image(shape_zyx=(4, 5, 6), spacing=(1.0, 2.0, 3.0), ras=False):
    """A SimpleITK image with distinct extents per axis, so an axis swap shows up.
    ``ras=True`` gives an already-RAS image, which isolates the transpose from the
    reorientation flips."""
    img = sitk.GetImageFromArray(np.zeros(shape_zyx, dtype=np.int16))
    img.SetSpacing(spacing)
    if ras:
        img.SetDirection(_RAS)
    return img


def _run(monkeypatch, prompts, mask_fn=None, image=None):
    fake = FakePredictor(mask_fn)
    monkeypatch.setattr(vt, "_get_predictor", lambda device: fake)
    seg = vt.segment(image if image is not None else _image(), prompts, device="cpu")
    return seg, fake


def test_prompts_must_be_present_and_non_empty():
    from nnseg.errors import InputError
    for bad in (None, [], "", ["  "], 42):
        with pytest.raises(InputError):
            vt.normalize_prompts(bad)
    assert vt.normalize_prompts("liver") == ["liver"]
    assert vt.normalize_prompts([" liver ", "spleen"]) == ["liver", "spleen"]


def test_the_model_gets_a_transposed_array_and_the_result_comes_back_on_the_input_grid(monkeypatch):
    """VoxTell wants nibabel order (X, Y, Z); nnseg holds SimpleITK (Z, Y, X). The
    engine must reverse the axes both ways - on a non-cubic volume a missed transpose
    changes the shape, which is exactly what this catches."""
    img = _image((4, 5, 6))                       # (Z, Y, X)
    seg, fake = _run(monkeypatch, ["liver"], image=img)
    assert fake.seen_shape == (6, 5, 4)           # handed over as (X, Y, Z)
    arr = sitk.GetArrayFromImage(seg.labels)
    assert arr.shape == (4, 5, 6)                 # and returned on the input's own grid
    assert seg.labels.GetSpacing() == img.GetSpacing()


def _slab(i, shape):                               # (X, Y, Z): the first two X planes
    m = np.zeros(shape, dtype=np.uint8)
    m[:2, :, :] = 1
    return m


def test_a_mask_lands_on_the_axis_it_was_set_on(monkeypatch):
    """Shape alone would not catch a swap of two equal-length axes, so place a slab on
    one axis in the model's frame and require it on the matching axis of the output.
    The input is already RAS, so this isolates the transpose from any flips."""
    seg, _ = _run(monkeypatch, ["liver"], mask_fn=_slab, image=_image((4, 5, 6), ras=True))
    arr = sitk.GetArrayFromImage(seg.labels)       # (Z, Y, X): X is the LAST axis
    assert arr[:, :, :2].all() and not arr[:, :, 2:].any()


def test_a_non_ras_input_is_reoriented_and_the_result_comes_back_in_its_own_frame(monkeypatch):
    """A LPS input is flipped into RAS for the model and flipped back afterwards, so a
    slab on RAS's first X planes belongs on the LAST X planes of the returned volume.
    That round-trip is what keeps left and right where the caller put them."""
    seg, _ = _run(monkeypatch, ["liver"], mask_fn=_slab, image=_image((4, 5, 6)))
    arr = sitk.GetArrayFromImage(seg.labels)
    assert arr[:, :, -2:].all() and not arr[:, :, :-2].any()


def test_labels_follow_prompt_order_and_later_prompts_win_on_overlap(monkeypatch):
    """Label i+1 is prompt i, the schema names it with the prompt text, and the mask
    painted later wins where two prompts overlap (the documented lossy part)."""
    def everything(i, shape):
        return np.ones(shape, dtype=np.uint8)      # both prompts cover the volume

    seg, _ = _run(monkeypatch, ["liver", "spleen"], mask_fn=everything)
    arr = sitk.GetArrayFromImage(seg.labels)
    assert set(np.unique(arr)) == {2}              # spleen (prompt 2) overwrote liver
    assert seg.schema.names == {1: "liver", 2: "spleen"}
    assert seg.provenance["prompts"] == ["liver", "spleen"]
    assert seg.provenance["overlap_voxels_overwritten"] == arr.size
    assert seg.provenance["empty_prompts"] == ["liver"]


def test_a_mask_shape_that_disagrees_with_the_input_is_refused(monkeypatch):
    """The self-check: if the model answers on a different grid than it was given,
    say so instead of writing a silently wrong segmentation."""
    from nnseg.errors import InputError

    def wrong(i, shape):
        return np.ones((shape[0] + 1, shape[1], shape[2]), dtype=np.uint8)

    with pytest.raises(InputError, match="not in the orientation it expects"):
        _run(monkeypatch, ["liver"], mask_fn=wrong)


def test_weights_identity_is_the_registry_s(monkeypatch):
    """One literal for the cache key, shared by the API describe and the worker re-key."""
    from nnseg.engines import registry as R
    assert vt.weights_installed() == R.ENGINES["voxtell"].weights_identity()
    assert vt.weights_installed() == [{"id": "voxtell", "version": "v1.1"}]


def test_prompts_change_the_result_cache_key():
    """The property the whole design rests on: prompts are options, options hash into
    the key, so two prompt lists cannot collide on one cached result."""
    from nnseg.serve import result_key
    key = lambda opts: result_key(("sha256:abc",), "voxtell:text", opts, ["voxtell=v1.1"])
    assert key({"prompts": ["liver"]}) != key({"prompts": ["spleen"]})
    assert key({"prompts": ["liver"]}) == key({"prompts": ["liver"]})
    assert key({"prompts": ["liver", "spleen"]}) != key({"prompts": ["spleen", "liver"]})


def test_catalog_lists_voxtell_without_a_fixed_label_set(monkeypatch):
    """voxtell:text resolves and describes, but carries no `structures` - what it
    segments is whatever the caller prompts for."""
    monkeypatch.setenv("NNSEG_VOXTELL", "1")
    from nnseg.ecosystems import EcosystemCatalog, VoxTellEcosystem
    cat = EcosystemCatalog([VoxTellEcosystem()], root=None)
    assert cat.resolve("voxtell:text")[2] == "voxtell:text"
    info = cat.info("voxtell:text")
    assert info["engine"] == "voxtell" and info["task_spec"] is False
    assert info["weights_installed"] == [{"id": "voxtell", "version": "v1.1"}]
    assert "structures" not in info
