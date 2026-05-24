"""Tests for the workflow module — Bbox arithmetic, crop/paste round-trip,
and the run_workflow orchestrator.

Geometric primitives are pure-numpy / pure-SITK; the orchestrator runs on
a synthetic tiny engine built from random weights, so no real model
checkpoint is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_inference_mlx import (
    Bbox,
    Stage,
    compute_fg_bbox,
    crop_image,
    paste_segmentation,
    run_workflow,
)


# ---------------------------------------------------------------------------
# Bbox arithmetic
# ---------------------------------------------------------------------------


class TestBbox:
    def test_shape_zyx(self):
        b = Bbox(0, 10, 5, 20, 3, 7)
        assert b.shape_zyx == (10, 15, 4)

    def test_slices(self):
        b = Bbox(0, 10, 5, 20, 3, 7)
        assert b.slices == (slice(0, 10), slice(5, 20), slice(3, 7))

    def test_full(self):
        b = Bbox.full((5, 10, 15))
        assert b == Bbox(0, 5, 0, 10, 0, 15)
        assert b.shape_zyx == (5, 10, 15)

    def test_clamped(self):
        b = Bbox(-2, 100, 0, 50, 0, 50).clamped((10, 30, 30))
        assert b == Bbox(0, 10, 0, 30, 0, 30)

    def test_clamped_already_inside(self):
        b = Bbox(2, 8, 5, 25, 10, 20).clamped((50, 50, 50))
        assert b == Bbox(2, 8, 5, 25, 10, 20)

    def test_dilated_int(self):
        b = Bbox(5, 10, 5, 10, 5, 10).dilated(2)
        assert b == Bbox(3, 12, 3, 12, 3, 12)

    def test_dilated_tuple(self):
        b = Bbox(5, 10, 5, 10, 5, 10).dilated((1, 2, 3))
        assert b == Bbox(4, 11, 3, 12, 2, 13)

    def test_dilated_with_clamp(self):
        b = Bbox(0, 10, 0, 10, 0, 10).dilated(5, max_shape_zyx=(12, 12, 12))
        assert b == Bbox(0, 12, 0, 12, 0, 12)

    def test_compose(self):
        outer = Bbox(10, 30, 20, 50, 0, 40)
        inner = Bbox(2, 8, 5, 15, 3, 10)
        assert outer.compose(inner) == Bbox(12, 18, 25, 35, 3, 10)

    def test_compose_full_identity(self):
        """Composing with a full-sized inner bbox is the identity."""
        outer = Bbox(5, 15, 10, 30, 0, 20)
        inner = Bbox.full(outer.shape_zyx)
        # The inner spans 0..shape, so composing gives outer.start..outer.start+shape == outer
        assert outer.compose(inner) == outer

    def test_frozen(self):
        b = Bbox(0, 5, 0, 5, 0, 5)
        with pytest.raises(Exception):
            b.z_start = 1  # frozen dataclass


# ---------------------------------------------------------------------------
# compute_fg_bbox
# ---------------------------------------------------------------------------


class TestComputeFgBbox:
    def test_any_fg(self):
        labels = np.zeros((50, 50, 50), dtype=np.uint8)
        labels[10:30, 15:35, 20:40] = 1
        assert compute_fg_bbox(labels) == Bbox(10, 30, 15, 35, 20, 40)

    def test_class_filter(self):
        labels = np.zeros((50, 50, 50), dtype=np.uint8)
        labels[10:30, 15:35, 20:40] = 1
        labels[35:40, 5:10, 5:10] = 2
        # Restrict to class 2 only
        assert compute_fg_bbox(labels, classes=(2,)) == Bbox(35, 40, 5, 10, 5, 10)

    def test_class_filter_multiple(self):
        labels = np.zeros((30, 30, 30), dtype=np.uint8)
        labels[5:10, 5:10, 5:10] = 1
        labels[20:25, 20:25, 20:25] = 3
        labels[12:18, 12:18, 12:18] = 5
        # Classes 1 and 3 only — bbox spans both
        assert compute_fg_bbox(labels, classes=(1, 3)) == Bbox(5, 25, 5, 25, 5, 25)

    def test_empty_returns_none(self):
        labels = np.zeros((20, 20, 20), dtype=np.uint8)
        assert compute_fg_bbox(labels) is None

    def test_class_not_present_returns_none(self):
        labels = np.zeros((20, 20, 20), dtype=np.uint8)
        labels[5:10, 5:10, 5:10] = 1
        assert compute_fg_bbox(labels, classes=(99,)) is None

    def test_dilation_requires_spacing(self):
        labels = np.zeros((20, 20, 20), dtype=np.uint8)
        labels[5:10, 5:10, 5:10] = 1
        with pytest.raises(ValueError, match="spacing_zyx is required"):
            compute_fg_bbox(labels, dilation_mm=2.0)

    def test_dilation_applies(self):
        labels = np.zeros((30, 30, 30), dtype=np.uint8)
        labels[10:20, 10:20, 10:20] = 1
        b = compute_fg_bbox(
            labels, dilation_mm=4.0, spacing_zyx=(2.0, 2.0, 2.0),
        )
        # 4 mm / 2 mm = 2 voxels of dilation each direction
        assert b == Bbox(8, 22, 8, 22, 8, 22)

    def test_dilation_clamps_to_volume(self):
        labels = np.zeros((20, 20, 20), dtype=np.uint8)
        labels[2:18, 2:18, 2:18] = 1
        b = compute_fg_bbox(
            labels, dilation_mm=100.0, spacing_zyx=(1.0, 1.0, 1.0),
        )
        # Massive dilation clamped to volume edges
        assert b == Bbox(0, 20, 0, 20, 0, 20)


# ---------------------------------------------------------------------------
# crop_image + paste_segmentation
# ---------------------------------------------------------------------------


SimpleITK = pytest.importorskip("SimpleITK")


class TestCropImage:
    def _make_sitk(self, shape_zyx=(8, 6, 4)):
        arr = np.arange(np.prod(shape_zyx), dtype=np.float32).reshape(shape_zyx)
        img = SimpleITK.GetImageFromArray(arr)
        img.SetSpacing((1.5, 1.5, 2.0))   # X, Y, Z in SITK order
        img.SetOrigin((10.0, 20.0, 30.0))
        return img, arr

    def test_crop_size(self):
        img, _ = self._make_sitk()
        bbox = Bbox(2, 6, 1, 5, 1, 4)
        sub = crop_image(img, bbox)
        # SITK size is XYZ
        assert sub.GetSize() == (3, 4, 4)

    def test_crop_preserves_spacing(self):
        img, _ = self._make_sitk()
        sub = crop_image(img, Bbox(2, 6, 1, 5, 1, 4))
        assert sub.GetSpacing() == (1.5, 1.5, 2.0)

    def test_crop_shifts_origin_to_world_position(self):
        img, _ = self._make_sitk()
        bbox = Bbox(2, 6, 1, 5, 1, 4)
        sub = crop_image(img, bbox)
        # New origin = old + (start_voxel * spacing) per axis (XYZ)
        # X-start=1, Y-start=1, Z-start=2
        expected = (10.0 + 1 * 1.5, 20.0 + 1 * 1.5, 30.0 + 2 * 2.0)
        assert sub.GetOrigin() == expected

    def test_crop_pixel_values_match(self):
        img, arr = self._make_sitk()
        bbox = Bbox(2, 6, 1, 5, 1, 4)
        sub = crop_image(img, bbox)
        sub_arr = SimpleITK.GetArrayFromImage(sub)
        np.testing.assert_array_equal(sub_arr, arr[bbox.slices])


class TestPasteSegmentation:
    def test_basic_paste(self):
        small = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
        out = paste_segmentation(small, full_shape_zyx=(10, 10, 10),
                                  bbox=Bbox(2, 5, 3, 7, 1, 6))
        assert out.shape == (10, 10, 10)
        np.testing.assert_array_equal(out[2:5, 3:7, 1:6], small)

    def test_paste_fill(self):
        small = np.ones((2, 2, 2), dtype=np.uint8)
        out = paste_segmentation(small, full_shape_zyx=(5, 5, 5),
                                  bbox=Bbox(0, 2, 0, 2, 0, 2), fill=9)
        # Outside the bbox is fill=9
        assert out[3, 3, 3] == 9
        # Inside the bbox is small's data (1)
        assert out[0, 0, 0] == 1

    def test_paste_dtype_preserved(self):
        small = np.ones((2, 2, 2), dtype=np.uint16)
        out = paste_segmentation(small, (5, 5, 5), Bbox(0, 2, 0, 2, 0, 2))
        assert out.dtype == np.uint16

    def test_paste_shape_mismatch_raises(self):
        small = np.zeros((2, 2, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="does not match"):
            paste_segmentation(small, (5, 5, 5), Bbox(0, 3, 0, 3, 0, 3))


class TestCropPasteRoundTrip:
    def test_roundtrip_recovers_inside(self):
        img = SimpleITK.GetImageFromArray(
            np.arange(8 * 6 * 4, dtype=np.uint8).reshape(8, 6, 4),
        )
        img.SetSpacing((1.0, 1.0, 1.0))
        bbox = Bbox(2, 6, 1, 5, 1, 4)
        sub_arr = SimpleITK.GetArrayFromImage(crop_image(img, bbox))
        out = paste_segmentation(sub_arr, (8, 6, 4), bbox)
        original = SimpleITK.GetArrayFromImage(img)
        np.testing.assert_array_equal(out[bbox.slices], original[bbox.slices])

    def test_roundtrip_zero_outside(self):
        img = SimpleITK.GetImageFromArray(np.ones((8, 6, 4), dtype=np.uint8))
        img.SetSpacing((1.0, 1.0, 1.0))
        bbox = Bbox(2, 6, 1, 5, 1, 4)
        sub_arr = SimpleITK.GetArrayFromImage(crop_image(img, bbox))
        out = paste_segmentation(sub_arr, (8, 6, 4), bbox)
        # Top-left corner is outside the bbox, should be 0
        assert out[0, 0, 0] == 0


# ---------------------------------------------------------------------------
# run_workflow — orchestrator on a synthetic tiny engine
# ---------------------------------------------------------------------------


def _make_tiny_engine(num_classes: int = 3):
    """A real but tiny InferenceEngine built from random weights.

    Mirrors tests/test_engine.py's make_synthetic_bundle pattern.
    """
    import mlx.nn as nn
    from nnunet_inference_mlx import InferenceEngine, ModelBundle
    from nnunet_inference_mlx.plans import build_network_from_plans

    plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [32, 32, 32],
                "spacing": [1.5, 1.5, 1.5],
                "normalization_schemes": ["ZScoreNormalization"],
                "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "n_conv_per_stage_encoder": [2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2],
                "UNet_base_num_features": 8,
            }
        },
        "foreground_intensity_properties_per_channel": {},
    }
    # nnU-Net dataset.json convention: labels maps name -> int id.
    # The 'background' entry is required (id 0); foreground classes follow.
    dataset = {
        "labels": {
            "background": 0,
            **{f"class_{i}": i for i in range(1, num_classes)},
        },
        "channel_names": {"0": "CT"},
    }
    network = build_network_from_plans(
        plans, "3d_fullres", 1, num_classes, deep_supervision=False,
    )
    weights = dict(nn.utils.tree_flatten(network.parameters()))
    bundle = ModelBundle(
        plans=plans, dataset=dataset,
        fold_weights=[weights], metadata={}, fold_ids=(0,),
    )
    return InferenceEngine(bundle, verbose=False, progress=False)


def _make_sitk_volume(shape_zyx=(48, 48, 48), spacing=(1.5, 1.5, 1.5)):
    arr = np.random.randn(*shape_zyx).astype(np.float32)
    img = SimpleITK.GetImageFromArray(arr)
    # SITK spacing is XYZ
    img.SetSpacing(tuple(spacing[::-1]))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


@pytest.fixture(scope="module")
def tiny_engine():
    return _make_tiny_engine()


def test_run_workflow_empty_stages_raises(tiny_engine):
    img = _make_sitk_volume()
    with pytest.raises(ValueError, match="non-empty"):
        run_workflow(img, [])


def test_run_workflow_single_stage_preserves_geometry(tiny_engine):
    img = _make_sitk_volume(shape_zyx=(48, 48, 48))
    seg = run_workflow(img, [Stage(engine=tiny_engine)])
    assert seg.GetSize() == img.GetSize()
    assert seg.GetSpacing() == img.GetSpacing()
    assert seg.GetOrigin() == img.GetOrigin()


def test_run_workflow_with_unfindable_crop_class_passes_through(tiny_engine):
    """If crop_to_classes doesn't appear in stage 1's output, stage 2 still
    runs on the full input (no error, just no cropping)."""
    img = _make_sitk_volume()
    # Class 999 won't be in a K=3 output
    stages = [
        Stage(engine=tiny_engine, crop_to_classes=(999,)),
        Stage(engine=tiny_engine),
    ]
    seg = run_workflow(img, stages)
    assert seg.GetSize() == img.GetSize()
