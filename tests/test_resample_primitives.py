"""Tests for the 0.8.0 resampling additions:

- polymorphic input on ``inverse_resample_argmax`` (mx.array | np.ndarray)
- ``inverse_resample_paint`` (region-based scheme-aware inverse resample)
- ``resample_volume`` (numpy forward resample primitive)
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from nnunet_inference_mlx import (
    inverse_resample_argmax,
    inverse_resample_paint,
    resample_volume,
)


# ---------------------------------------------------------------------------
# inverse_resample_argmax: polymorphic input
# ---------------------------------------------------------------------------


def _make_logits(K=3, shape_zyx=(10, 10, 10), seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((K, *shape_zyx)).astype(np.float32)


class TestArgmaxPolymorphic:
    def test_accepts_mx_array(self):
        logits = mx.array(_make_logits(K=3, shape_zyx=(8, 8, 8)))
        seg = inverse_resample_argmax(
            logits, out_shape_zyx=(8, 8, 8),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
        )
        assert seg.shape == (8, 8, 8)
        assert seg.dtype == np.uint8

    def test_accepts_np_ndarray(self):
        logits_np = _make_logits(K=3, shape_zyx=(8, 8, 8))
        seg = inverse_resample_argmax(
            logits_np, out_shape_zyx=(8, 8, 8),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
        )
        assert seg.shape == (8, 8, 8)

    def test_mx_and_np_inputs_equivalent(self):
        logits_np = _make_logits(K=4, shape_zyx=(10, 10, 10), seed=7)
        logits_mx = mx.array(logits_np)
        common = dict(
            out_shape_zyx=(10, 10, 10),
            target_spacing_zyx=(1.5, 1.5, 1.5),
            acq_spacing_zyx=(1.5, 1.5, 1.5),
        )
        seg_from_np = inverse_resample_argmax(logits_np, **common)
        seg_from_mx = inverse_resample_argmax(logits_mx, **common)
        np.testing.assert_array_equal(seg_from_np, seg_from_mx)


# ---------------------------------------------------------------------------
# inverse_resample_paint: region-based scheme
# ---------------------------------------------------------------------------


class TestInverseResamplePaint:
    def test_basic_paint(self):
        """Each region with logit > 0 should paint its label value."""
        K = 3
        shape = (10, 10, 10)
        # Construct logits so each region is clearly above 0 in a specific
        # corner block, below 0 elsewhere.
        logits = np.full((K, *shape), -10.0, dtype=np.float32)
        logits[0, 0:5, :, :] = 5.0     # region 0 active in first half-Z
        logits[1, :, 0:5, :] = 5.0     # region 1 active in first half-Y
        logits[2, :, :, 0:5] = 5.0     # region 2 active in first half-X

        seg = inverse_resample_paint(
            logits,
            out_shape_zyx=shape,
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(1, 2, 3),
        )

        # Region 2 (label=3) paints last, wins at overlaps.
        # Voxel (0, 0, 0) is in all three regions → should be label 3.
        assert seg[0, 0, 0] == 3
        # Voxel (0, 0, 5) — only regions 0 and 1 active (not 2). Region 1 wins.
        assert seg[0, 0, 5] == 2
        # Voxel (5, 5, 5) — no region active → background (0).
        assert seg[5, 5, 5] == 0

    def test_threshold_default_at_zero(self):
        """Default threshold=0 (sigmoid > 0.5 ↔ logit > 0). A logit of 0.1
        should paint; a logit of -0.1 should not."""
        K = 1
        shape = (4, 4, 4)
        logits = np.zeros((K, *shape), dtype=np.float32)
        logits[0, 0, 0, 0] = 0.1
        logits[0, 1, 1, 1] = -0.1
        seg = inverse_resample_paint(
            logits,
            out_shape_zyx=shape,
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(7,),
        )
        assert seg[0, 0, 0] == 7
        assert seg[1, 1, 1] == 0

    def test_threshold_at_half_for_sigmoid_input(self):
        """If the caller passes post-sigmoid probabilities, threshold should
        be set to 0.5."""
        K = 1
        shape = (4, 4, 4)
        probs = np.full((K, *shape), 0.4, dtype=np.float32)
        probs[0, 0, 0, 0] = 0.6
        seg = inverse_resample_paint(
            probs,
            out_shape_zyx=shape,
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(5,),
            threshold=0.5,
        )
        assert seg[0, 0, 0] == 5
        # Other voxels are 0.4 (below 0.5) → background
        assert seg[1, 1, 1] == 0

    def test_paint_priority_overwrites(self):
        """Later regions in regions_class_order overwrite earlier ones."""
        # Both regions are everywhere active; later wins.
        K = 2
        shape = (4, 4, 4)
        logits = np.full((K, *shape), 5.0, dtype=np.float32)
        seg = inverse_resample_paint(
            logits,
            out_shape_zyx=shape,
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(11, 22),
        )
        # Region 1 (label 22) paints last → should win everywhere
        assert (seg == 22).all()

    def test_wrong_k_raises(self):
        """K channels in logits must match regions_class_order length."""
        logits = np.zeros((4, 4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="regions_class_order"):
            inverse_resample_paint(
                logits,
                out_shape_zyx=(4, 4, 4),
                target_spacing_zyx=(1.0, 1.0, 1.0),
                acq_spacing_zyx=(1.0, 1.0, 1.0),
                regions_class_order=(1, 2, 3),  # length 3, but K=4
            )

    def test_accepts_mx_array_input(self):
        logits = mx.array(np.full((2, 4, 4, 4), 5.0, dtype=np.float32))
        seg = inverse_resample_paint(
            logits,
            out_shape_zyx=(4, 4, 4),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(1, 2),
        )
        assert seg.shape == (4, 4, 4)

    def test_resample_to_different_acquisition_spacing(self):
        """Output shape should reflect requested acquisition spacing."""
        logits = np.full((2, 4, 4, 4), 5.0, dtype=np.float32)
        seg = inverse_resample_paint(
            logits,
            out_shape_zyx=(8, 8, 8),
            target_spacing_zyx=(2.0, 2.0, 2.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(1, 2),
        )
        assert seg.shape == (8, 8, 8)

    def test_dtype_auto_selects(self):
        """Output dtype is auto-picked from the max region label value."""
        # All labels fit in uint8
        seg_u8 = inverse_resample_paint(
            np.full((1, 4, 4, 4), 5.0, dtype=np.float32),
            out_shape_zyx=(4, 4, 4),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(200,),
        )
        assert seg_u8.dtype == np.uint8

        # A label > 255 forces uint16
        seg_u16 = inverse_resample_paint(
            np.full((1, 4, 4, 4), 5.0, dtype=np.float32),
            out_shape_zyx=(4, 4, 4),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(1000,),
        )
        assert seg_u16.dtype == np.uint16

    def test_dtype_override(self):
        seg = inverse_resample_paint(
            np.full((1, 4, 4, 4), 5.0, dtype=np.float32),
            out_shape_zyx=(4, 4, 4),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=(1,),
            out_dtype=np.int32,
        )
        assert seg.dtype == np.int32


# ---------------------------------------------------------------------------
# resample_volume (numpy forward resample primitive)
# ---------------------------------------------------------------------------


pytest.importorskip("scipy")


class TestResampleVolume:
    def test_identity_spacing(self):
        vol = np.random.randn(10, 12, 14).astype(np.float32)
        out = resample_volume(vol, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        assert out.shape == vol.shape
        np.testing.assert_allclose(out, vol, rtol=1e-5)

    def test_downsample(self):
        vol = np.random.randn(20, 20, 20).astype(np.float32)
        out = resample_volume(vol, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        # 2x coarser output → ~half size
        assert out.shape == (10, 10, 10)

    def test_upsample(self):
        vol = np.random.randn(10, 10, 10).astype(np.float32)
        out = resample_volume(vol, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
        # 2x finer output → ~double size
        assert out.shape == (20, 20, 20)

    def test_anisotropic_spacing(self):
        vol = np.random.randn(20, 20, 20).astype(np.float32)
        out = resample_volume(vol, (1.0, 1.0, 1.0), (2.0, 1.0, 0.5))
        assert out.shape == (10, 20, 40)

    @pytest.mark.parametrize("order", [0, 1, 3])
    def test_all_orders_run(self, order):
        vol = np.random.randn(8, 8, 8).astype(np.float32)
        out = resample_volume(vol, (1.0, 1.0, 1.0), (1.5, 1.5, 1.5), order=order)
        assert out.dtype == np.float32

    def test_integer_input_promoted_to_float32(self):
        vol = np.arange(8 * 8 * 8, dtype=np.int16).reshape(8, 8, 8)
        out = resample_volume(vol, (1.0, 1.0, 1.0), (1.5, 1.5, 1.5))
        assert out.dtype == np.float32

    def test_ndim_preserved(self):
        vol = np.random.randn(6, 6, 6).astype(np.float32)
        out = resample_volume(vol, (1.0, 1.0, 1.0), (0.5, 0.5, 0.5))
        assert out.ndim == 3
