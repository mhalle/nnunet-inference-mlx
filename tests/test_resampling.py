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


class TestMlxForwardResampler:
    """Per-axis Metal forward resampler: AA-cubic down, linear up.

    The headline guarantees: anti-aliased (no aliasing) when downsampling, and
    **no cubic ringing/overshoot** when upsampling — the thick-slice CT case,
    where the through-plane axis is upsampled to the model grid.
    """

    def test_shape_and_spacing(self):
        from nnunet_inference_mlx.resampling import resample_volume_mlx
        arr = np.random.randn(40, 60, 60).astype(np.float32)
        src = (1.0, 1.0, 1.0); tgt = (2.0, 1.5, 1.5)
        out_shape = tuple(max(1, round(arr.shape[i] * src[i] / tgt[i])) for i in range(3))
        out = np.asarray(resample_volume_mlx(arr, out_shape, src, tgt))
        assert out.shape == out_shape

    def test_no_ringing_on_upsampled_thick_axis(self):
        # Thick-slice CT: bone slab (+1000) amid air (-1000) along a 5 mm axis,
        # fine 0.7 mm in-plane, resampled to 1.5 mm iso. The thick axis is
        # UPSAMPLED (5->1.5) -> must use linear -> NO overshoot past [-1000,1000].
        # (Cubic would ring to ±~1140; that's the regression this guards.)
        from nnunet_inference_mlx.resampling import resample_volume_mlx
        v = np.full((40, 80, 80), -1000.0, np.float32)
        v[15:25] = 1000.0
        src = (5.0, 0.7, 0.7); tgt = (1.5, 1.5, 1.5)
        out_shape = tuple(max(1, round(v.shape[i] * src[i] / tgt[i])) for i in range(3))
        out = np.asarray(resample_volume_mlx(v, out_shape, src, tgt))
        assert out.max() <= 1000.5, f"overshoot above input max: {out.max()}"
        assert out.min() >= -1000.5, f"overshoot below input min: {out.min()}"

    def test_downsample_is_antialiased_not_just_2tap(self):
        # On a high-frequency pattern, AA-cubic downsampling should differ from a
        # naive 2-tap linear point-sample (it averages the footprint).
        from nnunet_inference_mlx.resampling import resample_volume_mlx, resample_image_to_target
        rng = np.random.default_rng(0)
        v = rng.standard_normal((64, 64, 64)).astype(np.float32)
        src = (1.0, 1.0, 1.0); tgt = (4.0, 4.0, 4.0)     # 4x downsample
        out_shape = (16, 16, 16)
        aa = np.asarray(resample_volume_mlx(v, out_shape, src, tgt))
        assert aa.shape == out_shape
        img = SimpleITK.GetImageFromArray(v); img.SetSpacing((1.0, 1.0, 1.0))
        lin = SimpleITK.GetArrayFromImage(resample_image_to_target(img, tgt, interpolation="linear"))
        assert not np.allclose(aa, lin)   # anti-aliased != naive linear

    def test_downsample_sharp_edges_no_out_of_range_ring(self):
        # Catmull-Rom rings (over/undershoots) at SHARP edges — random data
        # won't trigger it; a bone/air step will. The clamped-cubic output must
        # stay within the input value range (no sub-air / super-bone ringing).
        from nnunet_inference_mlx.resampling import resample_volume_mlx
        v = np.full((64, 64, 64), -1024.0, np.float32)
        v[20:44, 20:44, 20:44] = 1500.0          # sharp bone cube in air
        src = (1.0, 1.0, 1.0); tgt = (4.0, 4.0, 4.0)
        out = np.asarray(resample_volume_mlx(v, (16, 16, 16), src, tgt))
        assert out.min() >= -1024.0 - 1e-3, f"undershoot below air floor: {out.min()}"
        assert out.max() <= 1500.0 + 1e-3, f"overshoot above bone: {out.max()}"


class TestGpuReorient:
    """GPU reorient (transpose+flip) must be bit-identical to SITK DICOMOrient.

    This is the orientation path — the place a bug silently swaps left/right —
    so it's pinned bit-exact against SITK across a battery of input orientations
    and the two world targets (RAS=NIfTI, LPS=DICOM), incl. arbitrary codes.
    """

    @pytest.mark.parametrize("inp", ["RAS", "LPS", "SPL", "PIR", "AIL", "RPI"])
    @pytest.mark.parametrize("tgt", ["RAS", "LPS", "SPL"])
    def test_bit_exact_vs_sitk_dicomorient(self, inp, tgt):
        from nnunet_inference_mlx.imageio import geometry_from_sitk
        from nnunet_inference_mlx.resampling import reorient_array_mlx
        base = SimpleITK.GetImageFromArray(
            (np.arange(20 * 24 * 28).reshape(20, 24, 28) % 101).astype(np.float32))
        base.SetSpacing((0.7, 0.9, 1.3)); base.SetOrigin((11.0, -22.0, 33.0))
        src = SimpleITK.DICOMOrient(base, inp)
        ref = SimpleITK.DICOMOrient(src, tgt)
        g = geometry_from_sitk(src)
        out, geom = reorient_array_mlx(
            SimpleITK.GetArrayFromImage(src),
            direction_xyz=g.direction_xyz, spacing_zyx=g.spacing_zyx,
            origin_xyz=g.origin_xyz, target_code=tgt)
        np.testing.assert_array_equal(np.asarray(out), SimpleITK.GetArrayFromImage(ref))
        np.testing.assert_allclose(geom.spacing_zyx, tuple(reversed(ref.GetSpacing())), atol=1e-6)
        np.testing.assert_allclose(geom.origin_xyz, ref.GetOrigin(), atol=1e-4)
        np.testing.assert_allclose(geom.direction_xyz, ref.GetDirection(), atol=1e-6)


class TestFusedKernelEquivalence:
    """The fused Metal kernel (default) must agree with the pure-MLX slab path.

    The fused kernel does trilinear + argmax/paint inline (one thread per
    output voxel, no K-channel materialization). It uses the same separable
    blend op-order as the slab path, so on synthetic logits (no near-ties)
    results are bit-identical. On real smooth logit fields a handful of
    boundary voxels can flip where the Metal compiler's FMA contraction
    rounds differently than MLX's separate mul/add — measured at 18 voxels
    in 43M (4e-5) on a 512³/117-channel CT — so we assert near-exact
    agreement rather than bit-exact to stay robust across GPU architectures.
    """

    @pytest.mark.parametrize(
        "target,acq,out_shape",
        [
            ((1.5, 1.5, 1.5), (1.5, 1.5, 1.5), (16, 16, 16)),   # identity
            ((1.5, 1.5, 1.5), (1.0, 1.0, 1.0), (24, 24, 24)),   # upsample
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (8, 8, 8)),      # downsample
            ((1.0, 1.0, 1.0), (0.5, 1.0, 2.0), (32, 16, 8)),    # anisotropic
        ],
    )
    def test_fused_matches_slab_argmax(self, target, acq, out_shape):
        logits = _make_logits(num_classes=9, shape_zyx=(16, 16, 16), seed=7)
        kw = dict(out_shape_zyx=out_shape,
                  target_spacing_zyx=target, acq_spacing_zyx=acq)
        fused = inverse_resample_argmax(logits, use_fused_kernel=True, **kw)
        slab = inverse_resample_argmax(logits, use_fused_kernel=False, **kw)
        assert fused.shape == out_shape
        # Identity case must be exact (no interpolation, no FMA ambiguity).
        if target == acq:
            np.testing.assert_array_equal(fused, slab)
        agree = (fused == slab).mean()
        assert agree > 0.999, f"fused vs slab agreement {agree:.5f}"

    def test_fused_matches_slab_paint(self):
        from nnunet_inference_mlx.resampling import inverse_resample_paint

        logits = _make_logits(num_classes=3, shape_zyx=(12, 12, 12), seed=11)
        rco = (1, 2, 4)   # paint-priority label values
        kw = dict(
            out_shape_zyx=(20, 20, 20),
            target_spacing_zyx=(1.5, 1.5, 1.5),
            acq_spacing_zyx=(1.0, 1.0, 1.0),
            regions_class_order=rco,
            threshold=0.0,
        )
        fused = inverse_resample_paint(logits, use_fused_kernel=True, **kw)
        slab = inverse_resample_paint(logits, use_fused_kernel=False, **kw)
        assert set(np.unique(fused).tolist()) <= {0, *rco}
        np.testing.assert_array_equal(fused, slab)
