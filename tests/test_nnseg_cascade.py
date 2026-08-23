"""Cascade task wiring (no models): the catalog resolves stages, weights_ids covers them,
and crop_from_task stages are flagged rather than crashing."""
import pytest

from nnseg.tasks import TaskCatalog, CascadeStep


def test_lung_vessels_is_a_two_stage_cascade():
    s = TaskCatalog("totalsegmentator").get("lung_vessels")
    assert s.shape == "cascade" and len(s.cascade) == 2
    coarse, fine = s.cascade
    assert coarse.weights_id == 297 and coarse.crop_to_classes == (10, 11, 12, 13, 14)
    assert fine.weights_id == 117 and fine.crop_to_classes == ()
    assert s.weights_ids == [297, 117]


def test_weights_ids_skips_crop_from_task_stages():
    s = TaskCatalog("totalsegmentator").get("teeth")
    # one stage crops from another task, not a model; it must not appear as a weights id
    assert all(isinstance(i, int) for i in s.weights_ids)
    assert any(st.crop_from_task for st in s.cascade)


def test_parts_directs_cascades_to_the_cascade_field():
    s = TaskCatalog("totalsegmentator").get("lung_vessels")
    with pytest.raises(NotImplementedError):
        s.parts



def test_segment_does_not_raise_on_spec_parts_for_a_cascade(tmp_path):
    """Regression: segment() reached spec.parts (which raises NotImplementedError for cascades)
    before the cascade branch. With an empty model root it must fail *later* - looking for
    weights - not with NotImplementedError."""
    import numpy as np
    import SimpleITK as sitk
    from nnseg import segment
    img = sitk.GetImageFromArray(np.zeros((8, 8, 8), np.int16))
    p = tmp_path / "v.nii.gz"
    sitk.WriteImage(img, str(p))
    with pytest.raises(Exception) as e:
        segment(str(p), "lung_vessels", device="cpu", model_root=str(tmp_path / "no_weights"))
    assert not isinstance(e.value, NotImplementedError), "segment hit spec.parts before the cascade branch"


def test_teeth_provisioning_recurses_through_crop_from_task(tmp_path):
    from nnseg.weights_fetch import ensure_task_weights
    # pre-place every model in the teeth chain so no network is touched: 113 (teeth),
    # 298 + 115 (craniofacial_structures)
    for did in (113, 298, 115):
        (tmp_path / f"Dataset{did}_x").mkdir()
    got = ensure_task_weights("teeth", tmp_path)
    names = {p.name for p in got}
    assert any("Dataset113" in n for n in names)          # teeth's own model
    assert any("Dataset298" in n for n in names) and any("Dataset115" in n for n in names)  # nested
