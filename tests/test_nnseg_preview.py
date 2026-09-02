"""The preview states its display convention and gets it from the geometry.

A laterality phantom: one blob on the patient's RIGHT (negative LPS world x),
nearer the FEET (inferior) and toward the BACK (posterior). Whatever the stored
axis order, a radiological panel must put that blob in the image's left half
(axial, coronal), at the bottom (coronal, sagittal: inferior), and - sagittal
viewed from the patient's left - in the right half (posterior). Neurological is
its mirror in x. Checked on the panel arrays, not by eye, and the markers must
say the same thing as the pixels.
"""
import numpy as np
import pytest
import SimpleITK as sitk

from nnseg import io as nio
from nnseg.preview import DISPLAY, DISPLAY_FRAMES, display_planes, render_preview


def _phantom(orientation: str):
    """A 40x50x60 mm volume stored in ``orientation`` whose single blob sits at
    patient right / posterior / inferior in WORLD terms (built in LPS, then
    DICOMOrient'ed, so the world position is the same for every storage order)."""
    arr = np.zeros((40, 50, 60), np.uint8)                       # (Z, Y, X) in LPS
    arr[4:12, 36:44, 6:14] = 1          # low z = inferior, high y = posterior, low x = RIGHT
    img = sitk.GetImageFromArray(arr)   # identity direction = LPS, 1 mm
    img.SetOrigin((-30.0, -25.0, -20.0))
    return sitk.DICOMOrient(img, orientation)


def _blob_center(l2d):
    r, c = np.argwhere(l2d == 1).T
    return r.mean() / l2d.shape[0], c.mean() / l2d.shape[1]     # fractions: (row, col)


@pytest.mark.parametrize("stored", ["LPS", "RAS", "LAS", "PIR", "SAL"])
def test_radiological_panels_put_the_patients_right_on_the_image_left(stored):
    seg = _phantom(stored)
    lab = sitk.GetArrayFromImage(seg)
    geo = nio.geometry_of(seg)
    panels = {name: (g, l, asp, mk) for name, g, l, asp, mk in
              display_planes(lab.astype(np.int16), lab, geo, display="radiological")}
    # axial: right on the image left, posterior at the bottom (origin='lower')
    row, col = _blob_center(panels["axial"][1])
    assert col < 0.5 and row < 0.5
    assert panels["axial"][3] == {"left": "R", "right": "L", "top": "A", "bottom": "P"}
    # coronal: right on the image left, inferior at the bottom
    row, col = _blob_center(panels["coronal"][1])
    assert col < 0.5 and row < 0.5
    assert panels["coronal"][3] == {"left": "R", "right": "L", "top": "S", "bottom": "I"}
    # sagittal viewed from the patient's left: posterior on the image right, inferior bottom
    row, col = _blob_center(panels["sagittal"][1])
    assert col > 0.5 and row < 0.5
    assert panels["sagittal"][3] == {"left": "A", "right": "P", "top": "S", "bottom": "I"}


def test_neurological_is_the_mirror_in_x_only():
    seg = _phantom("LAS")
    lab = sitk.GetArrayFromImage(seg)
    geo = nio.geometry_of(seg)
    rad = {n: (l, mk) for n, _, l, _, mk in display_planes(lab, lab, geo, display="radiological")}
    neu = {n: (l, mk) for n, _, l, _, mk in display_planes(lab, lab, geo, display="neurological")}
    for name in ("axial", "coronal"):
        assert np.array_equal(neu[name][0], rad[name][0][:, ::-1])
        assert neu[name][1]["left"] == "L" and rad[name][1]["left"] == "R"
    # sagittal flips its viewing side: anterior moves to the image right
    assert np.array_equal(neu["sagittal"][0], rad["sagittal"][0][:, ::-1])
    assert neu["sagittal"][1]["left"] == "P"


def test_panels_keep_physical_aspect_and_pick_the_slice_with_most_label():
    arr = np.zeros((10, 20, 30), np.uint8)
    arr[7, 5:15, 10:20] = 1                  # one labeled axial slice
    seg = sitk.GetImageFromArray(arr)
    seg.SetSpacing((0.5, 1.0, 3.0))          # x, y, z
    geo = nio.geometry_of(seg)
    (_, g, l, aspect, _), (_, _, lc, aspect_c, _), (_, _, ls, aspect_s, _) = display_planes(arr, arr, geo)
    assert l.sum() == 100 and lc.sum() == 10 and ls.sum() == 10
    assert aspect == pytest.approx(1.0 / 0.5)        # axial: row (y) / column (x)
    assert aspect_c == pytest.approx(3.0 / 0.5)      # coronal: z / x
    assert aspect_s == pytest.approx(3.0 / 1.0)      # sagittal: z / y


def test_default_convention_is_radiological_and_unknown_ones_are_loud():
    assert DISPLAY == "radiological" and set(DISPLAY_FRAMES) == {"radiological", "neurological"}
    seg = _phantom("LPS")
    with pytest.raises(KeyError):
        display_planes(sitk.GetArrayFromImage(seg), sitk.GetArrayFromImage(seg),
                       nio.geometry_of(seg), display="upside-down")


def test_render_preview_writes_a_png_with_the_convention_drawn(tmp_path):
    seg = _phantom("LAS")
    for i, (k, v) in enumerate({"Name": "blob", "LabelValue": "1", "Color": "0.9 0.3 0.2"}.items()):
        seg.SetMetaData(f"Segment0_{k}", v)
    p = tmp_path / "blob.seg.nrrd"
    sitk.WriteImage(seg, str(p))
    gray = sitk.Cast(seg, sitk.sitkInt16) * 0 - 1000
    gray.CopyInformation(seg)
    out = render_preview(gray, p, tmp_path / "preview.png", title="phantom")
    assert out is not None and out.stat().st_size > 3000
    # and the pixels agree with the phantom: the blob's red lands in the left half of the
    # axial panel (first third of the figure) - read back from the PNG, not the arrays
    import matplotlib.image as mpimg
    png = mpimg.imread(str(out))
    h, w = png.shape[:2]
    # the overlay is alpha 0.5 over a black (air) background, so "red" is a dim red
    red = (png[..., 0] > 0.25) & (png[..., 0] > 2 * png[..., 1]) & (png[..., 0] > 2 * png[..., 2])
    cols = np.argwhere(red[:, : w // 3])[:, 1]
    assert len(cols) > 20 and cols.mean() < (w // 3) / 2
