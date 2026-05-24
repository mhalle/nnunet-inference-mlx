"""Tests for the resampling module.

Forward (SITK) and inverse (MLX path B) are exercised on synthetic data so
no real model weights are needed. ``predict_with_resampling`` is covered
indirectly via the workflow orchestrator tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from nnunet_inference_mlx import (
    inverse_resample_argmax,
    resample_image_to_target,
)

SimpleITK = pytest.importorskip("SimpleITK")


# ---------------------------------------------------------------------------
# Forward resample (SITK)
# ---------------------------------------------------------------------------


def _make_sitk(shape_zyx=(20, 30, 40), spacing_xyz=(1.0, 1.0, 1.0)):
    arr = np.random.randn(*shape_zyx).astype(np.float32)
    img = SimpleITK.GetImageFromArray(arr)
    img.SetSpacing(spacing_xyz)
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


class TestResampleImageToTarget:
    def test_identity_target_spacing(self):
        """Resampling to the input spacing should give back the same size."""
        img = _make_sitk(shape_zyx=(20, 30, 40), spacing_xyz=(1.5, 1.5, 1.5))
        # SITK spacing is XYZ; target_spacing_zyx is ZYX. 1.5 isotropic
        # is the same either way.
        out = resample_image_to_target(img, target_spacing_zyx=(1.5, 1.5, 1.5))
        assert out.GetSpacing() == img.GetSpacing()
        # Size may be off by 1 due to rounding; allow exact match for
        # identity case.
        assert out.GetSize() == img.GetSize()

    def test_downsample_to_coarser_spacing(self):
        """2x coarser target spacing → roughly half-size output."""
        img = _make_sitk(shape_zyx=(20, 30, 40), spacing_xyz=(1.0, 1.0, 1.0))
        out = resample_image_to_target(img, target_spacing_zyx=(2.0, 2.0, 2.0))
        # Output spacing matches request
        assert out.GetSpacing() == (2.0, 2.0, 2.0)
        # Output size is roughly half (10, 15, 20) ZYX -> (20, 15, 10) XYZ
        # nnU-Net rounds via target_size = round(in_size * in_spacing / out_spacing)
        assert out.GetSize()[0] in (20, 21)
        assert out.GetSize()[1] in (15, 16)
        assert out.GetSize()[2] in (10, 11)

    def test_upsample_to_finer_spacing(self):
        img = _make_sitk(shape_zyx=(10, 10, 10), spacing_xyz=(2.0, 2.0, 2.0))
        out = resample_image_to_target(img, target_spacing_zyx=(1.0, 1.0, 1.0))
        assert out.GetSpacing() == (1.0, 1.0, 1.0)
        # ~2x more voxels per axis
        assert out.GetSize()[0] in (19, 20, 21)

    def test_anisotropic_spacing(self):
        img = _make_sitk(shape_zyx=(20, 20, 20), spacing_xyz=(1.0, 1.0, 3.0))
        out = resample_image_to_target(img, target_spacing_zyx=(1.5, 1.5, 1.5))
        # Z spacing (XYZ index 2 in SITK) was 3.0 → 1.5, double
        # X/Y spacing 1.0 → 1.5, fewer voxels
        assert out.GetSpacing() == (1.5, 1.5, 1.5)

    def test_preserves_origin(self):
        img = _make_sitk(shape_zyx=(20, 20, 20))
        img.SetOrigin((10.0, 20.0, 30.0))
        out = resample_image_to_target(img, target_spacing_zyx=(2.0, 2.0, 2.0))
        # Origin is at the center of voxel (0,0,0), preserved by SITK Resample.
        assert out.GetOrigin() == (10.0, 20.0, 30.0)

    @pytest.mark.parametrize("interp", ["linear", "bspline", "nearest"])
    def test_all_interpolators_run(self, interp):
        img = _make_sitk(shape_zyx=(15, 15, 15))
        out = resample_image_to_target(
            img, target_spacing_zyx=(2.0, 2.0, 2.0), interpolation=interp,
        )
        assert out.GetSize()[0] > 0


# ---------------------------------------------------------------------------
# Inverse resample (path B, MLX)
# ---------------------------------------------------------------------------


def _make_logits(num_classes=4, shape_zyx=(16, 16, 16), seed=0):
    """Random K-channel logits as an MLX array, shape (K, Z, Y, X)."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((num_classes, *shape_zyx)).astype(np.float32)
    return mx.array(arr)


class TestInverseResampleArgmax:
    def test_identity_spacing(self):
        logits = _make_logits(num_classes=4, shape_zyx=(16, 16, 16))
        out = inverse_resample_argmax(
            logits,
            out_shape_zyx=(16, 16, 16),
            target_spacing_zyx=(1.5, 1.5, 1.5),
            acq_spacing_zyx=(1.5, 1.5, 1.5),
        )
        assert out.shape == (16, 16, 16)
        assert out.dtype == np.uint8
        # Identity case: same shape, same spacing => argmax should match
        # numpy argmax of the logits directly.
        expected = np.array(logits).argmax(axis=0).astype(np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_upsample_2x_acq_finer(self):
        """2x finer acquisition spacing → 2x more voxels per axis."""
        logits = _make_logits(num_classes=3, shape_zyx=(8, 8, 8))
        out = inverse_resample_argmax(
            logits,
            out_shape_zyx=(16, 16, 16),
            target_spacing_zyx=(2.0, 2.0, 2.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
        )
        assert out.shape == (16, 16, 16)
        assert set(np.unique(out).tolist()) <= {0, 1, 2}

    def test_downsample_acq_coarser(self):
        """Coarser acquisition → fewer voxels."""
        logits = _make_logits(num_classes=3, shape_zyx=(16, 16, 16))
        out = inverse_resample_argmax(
            logits,
            out_shape_zyx=(8, 8, 8),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(2.0, 2.0, 2.0),
        )
        assert out.shape == (8, 8, 8)

    def test_explicit_dtype(self):
        logits = _make_logits(num_classes=2)
        out = inverse_resample_argmax(
            logits,
            out_shape_zyx=(16, 16, 16),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            out_dtype=np.uint16,
        )
        assert out.dtype == np.uint16

    def test_slab_streaming_matches_full_materialize(self):
        """Tight memory budget forces slab streaming; the result should match
        the unconstrained full-materialize path bit-for-bit."""
        logits = _make_logits(num_classes=5, shape_zyx=(12, 12, 12), seed=42)
        # Generous budget — should fit in one slab
        full = inverse_resample_argmax(
            logits,
            out_shape_zyx=(20, 20, 20),
            target_spacing_zyx=(1.5, 1.5, 1.5),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            peak_working_memory_mb=2000,
        )
        # Tight budget — forces multiple slabs
        slabbed = inverse_resample_argmax(
            logits,
            out_shape_zyx=(20, 20, 20),
            target_spacing_zyx=(1.5, 1.5, 1.5),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            peak_working_memory_mb=1,  # force per-slab work
        )
        np.testing.assert_array_equal(full, slabbed)

    def test_anisotropic_acquisition(self):
        """Acquisition spacing differing per axis should still produce a
        sensible output shape."""
        logits = _make_logits(num_classes=3, shape_zyx=(10, 10, 10))
        out = inverse_resample_argmax(
            logits,
            out_shape_zyx=(20, 10, 5),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(0.5, 1.0, 2.0),
        )
        assert out.shape == (20, 10, 5)

    def test_cascade_option_runs(self):
        """Aggressive downsample with cascade_downsample=True should produce
        an output of the requested shape. We don't assert exact equivalence
        to single-step (they're not equivalent by design)."""
        logits = _make_logits(num_classes=3, shape_zyx=(32, 32, 32))
        out = inverse_resample_argmax(
            logits,
            out_shape_zyx=(4, 4, 4),
            target_spacing_zyx=(1.0, 1.0, 1.0),
            acq_spacing_zyx=(8.0, 8.0, 8.0),
            cascade_downsample=True,
        )
        assert out.shape == (4, 4, 4)
