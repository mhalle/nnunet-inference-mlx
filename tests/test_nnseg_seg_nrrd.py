"""`.seg.nrrd` outputs carry names, extents, colors and provenance in the header.

The service's default artifact (decided 2026-08-24): a NIfTI labelmap loses the
segmentation's meaning; a seg.nrrd is self-contained and loads named in Slicer.
"""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from nnseg.grid import Grid
from nnseg.result import Segmentation
from nnseg.values import LabelSchema


def make_seg():
    arr = np.zeros((3, 4, 5), dtype=np.uint8)
    arr[0, :2, :2] = 1                        # spleen: z 0, y 0-1, x 0-1
    arr[1:3, 2:4, 3:5] = 2                    # liver:  z 1-2, y 2-3, x 3-4
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.7, 1.0))
    schema = LabelSchema(names={0: "background", 1: "spleen", 2: "liver"})
    return Segmentation(labels=img, schema=schema, grid=Grid(shape=(3, 4, 5)),
                        spec=None, provenance={"device": "test", "task": "unit"})


def test_seg_nrrd_header_carries_meaning(tmp_path):
    p = make_seg().save(tmp_path / "labels.seg.nrrd")
    back = sitk.ReadImage(str(p))
    md = {k: back.GetMetaData(k) for k in back.GetMetaDataKeys()}
    assert md["Segment0_Name"] == "spleen" and md["Segment0_LabelValue"] == "1"
    assert md["Segment1_Name"] == "liver" and md["Segment1_LabelValue"] == "2"
    assert md["Segment0_Extent"] == "0 1 0 1 0 0"          # x0 x1 y0 y1 z0 z1
    assert md["Segment1_Extent"] == "3 4 2 3 1 2"
    assert md["Segmentation_MasterRepresentation"] == "Binary labelmap"
    assert "device" in md["nnseg_provenance"]
    assert len(md["Segment0_Color"].split()) == 3
    assert np.array_equal(sitk.GetArrayFromImage(back),
                          sitk.GetArrayFromImage(make_seg().labels))


def test_nifti_stays_plain(tmp_path):
    p = make_seg().save(tmp_path / "labels.nii.gz")
    back = sitk.ReadImage(str(p))
    assert not any(k.startswith("Segment0") for k in back.GetMetaDataKeys())
