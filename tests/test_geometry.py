"""Tests for the Volume/Segmentation-native geometry namespace.

The crop origin-shift is validated against SITK's RegionOfInterest (the
proven ``workflow.crop_image``) as the oracle, including an oblique
direction matrix — the case where a naive origin shift would be wrong.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from nnunet_inference_mlx.geometry import Box, bbox_of_labels, crop, paste
from nnunet_inference_mlx.values import Geometry, LabelSchema, Segmentation, Volume

sitk = pytest.importorskip("SimpleITK")


def _volume(shape=(10, 12, 14), *, origin=(0.0, 0.0, 0.0),
            direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(2.0, 1.5, 1.0)):
    data = mx.arange(int(np.prod(shape)), dtype=mx.float32).reshape((*shape, 1))
    geom = Geometry(spacing_zyx=spacing, shape_zyx=shape,
                    origin_xyz=origin, direction_xyz=direction)
    return Volume(data=data, geometry=geom, channels=("CT",))


def _seg(labels_zyx):
    arr = np.asarray(labels_zyx)
    geom = Geometry(spacing_zyx=(2.0, 1.5, 1.0), shape_zyx=arr.shape)
    return Segmentation(data=mx.array(arr), geometry=geom,
                        schema=LabelSchema(names={0: "background", 1: "a", 2: "b"}))


class TestBox:
    def test_shape_and_slices(self):
        b = Box(1, 5, 2, 8, 0, 3)
        assert b.shape_zyx == (4, 6, 3)
        arr = np.zeros((10, 10, 10))
        assert arr[b.slices].shape == (4, 6, 3)

    def test_dilate_clamps(self):
        b = Box(2, 4, 2, 4, 2, 4).dilated(3, max_shape_zyx=(5, 5, 5))
        assert (b.z_start, b.z_end) == (0, 5)

    def test_compose(self):
        outer = Box(10, 20, 5, 15, 0, 8)
        inner = Box(1, 3, 2, 4, 0, 2)
        c = outer.compose(inner)
        assert (c.z_start, c.z_end) == (11, 13)
        assert (c.y_start, c.y_end) == (7, 9)


class TestBboxOfLabels:
    def test_none_when_empty(self):
        assert bbox_of_labels(_seg(np.zeros((6, 6, 6), np.uint8))) is None

    def test_finds_box(self):
        lab = np.zeros((8, 8, 8), np.uint8)
        lab[2:5, 1:4, 3:6] = 1
        b = bbox_of_labels(_seg(lab))
        assert (b.z_start, b.z_end, b.y_start, b.y_end, b.x_start, b.x_end) == \
            (2, 5, 1, 4, 3, 6)

    def test_class_filter(self):
        lab = np.zeros((8, 8, 8), np.uint8)
        lab[0:2, 0:2, 0:2] = 1
        lab[5:7, 5:7, 5:7] = 2
        b = bbox_of_labels(_seg(lab), classes=(2,))
        assert (b.z_start, b.z_end) == (5, 7)

    def test_out_of_range_class_returns_none(self):
        lab = np.zeros((6, 6, 6), np.uint8)
        lab[1:3, 1:3, 1:3] = 1
        assert bbox_of_labels(_seg(lab), classes=(9999,)) is None


class TestCropMatchesSITK:
    @pytest.mark.parametrize("direction", [
        (1, 0, 0, 0, 1, 0, 0, 0, 1),          # axis-aligned
        (-1, 0, 0, 0, -1, 0, 0, 0, 1),        # LPS-ish flips
        # oblique: 90° rotation in the XY plane
        (0, -1, 0, 1, 0, 0, 0, 0, 1),
    ])
    def test_crop_origin_matches_roi(self, direction):
        vol = _volume(origin=(7.0, -3.0, 11.0), direction=direction)
        box = Box(2, 7, 3, 9, 1, 10)

        cropped = crop(vol, box)

        # Oracle: SITK RegionOfInterest on the equivalent image.
        img = sitk.GetImageFromArray(np.asarray(vol.data[..., 0]))
        img.SetSpacing((vol.geometry.spacing_zyx[2], vol.geometry.spacing_zyx[1],
                        vol.geometry.spacing_zyx[0]))
        img.SetOrigin(vol.geometry.origin_xyz)
        img.SetDirection(vol.geometry.direction_xyz)
        roi = sitk.RegionOfInterestImageFilter()
        roi.SetIndex([box.x_start, box.y_start, box.z_start])
        roi.SetSize([box.x_end - box.x_start, box.y_end - box.y_start,
                     box.z_end - box.z_start])
        ref = roi.Execute(img)

        assert cropped.geometry.shape_zyx == box.shape_zyx
        np.testing.assert_allclose(cropped.geometry.origin_xyz, ref.GetOrigin(),
                                   atol=1e-5)
        # data content matches the ROI as well
        np.testing.assert_array_equal(np.asarray(cropped.data[..., 0]),
                                      sitk.GetArrayFromImage(ref))


class TestPaste:
    def test_paste_roundtrip(self):
        full = (16, 16, 16)
        canvas_geom = Geometry(spacing_zyx=(1, 1, 1), shape_zyx=full)
        box = Box(3, 8, 4, 10, 5, 9)
        patch_arr = np.ones(box.shape_zyx, np.uint8) * 2
        patch = Segmentation(data=mx.array(patch_arr),
                             geometry=Geometry(spacing_zyx=(1, 1, 1),
                                               shape_zyx=box.shape_zyx),
                             schema=LabelSchema(names={0: "bg", 2: "x"}))
        out = paste(patch, canvas_geom, box)
        arr = np.asarray(out.data)
        assert arr.shape == full
        assert (arr[box.slices] == 2).all()
        assert arr.sum() == 2 * np.prod(box.shape_zyx)

    def test_paste_shape_mismatch_raises(self):
        canvas_geom = Geometry(spacing_zyx=(1, 1, 1), shape_zyx=(8, 8, 8))
        patch = Segmentation(data=mx.zeros((2, 2, 2), dtype=mx.uint8),
                             geometry=Geometry(spacing_zyx=(1, 1, 1),
                                               shape_zyx=(2, 2, 2)),
                             schema=LabelSchema(names={0: "bg"}))
        with pytest.raises(ValueError, match="box shape"):
            paste(patch, canvas_geom, Box(0, 3, 0, 3, 0, 3))
