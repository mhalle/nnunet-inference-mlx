"""Tests for the core frozen value types (Phase 1 of the rearchitecture).

Pure values — no GPU, no IO. Verify construction, validation, derived
properties, the with_* copy methods, channel selection, schema parsing
(standard + region), and BuildOptions hashability.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from nnunet_inference_mlx.values import (
    BuildOptions,
    Geometry,
    LabelSchema,
    Prediction,
    Region,
    RestorePlan,
    Segmentation,
    Volume,
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_basic(self):
        g = Geometry(spacing_zyx=(1.0, 0.8, 0.8), shape_zyx=(40, 256, 256))
        assert g.spacing_zyx == (1.0, 0.8, 0.8)
        assert g.shape_zyx == (40, 256, 256)
        assert g.is_axis_aligned

    def test_physical_size(self):
        g = Geometry(spacing_zyx=(2.0, 1.0, 0.5), shape_zyx=(10, 20, 40))
        assert g.physical_size_zyx == (20.0, 20.0, 20.0)

    def test_non_identity_direction_not_axis_aligned(self):
        sar = (0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0)
        g = Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(8, 8, 8), direction_xyz=sar)
        assert not g.is_axis_aligned

    def test_with_spacing(self):
        g = Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(8, 8, 8))
        g2 = g.with_spacing((1.5, 1.5, 1.5))
        assert g2.spacing_zyx == (1.5, 1.5, 1.5)
        assert g2.shape_zyx == g.shape_zyx
        assert g.spacing_zyx == (1.0, 1.0, 1.0)  # original unchanged

    def test_hashable(self):
        g = Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(8, 8, 8))
        assert hash(g) == hash(Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(8, 8, 8)))
        assert len({g, g}) == 1

    def test_bad_lengths_rejected(self):
        with pytest.raises(ValueError, match="spacing_zyx"):
            Geometry(spacing_zyx=(1.0, 1.0), shape_zyx=(8, 8, 8))
        with pytest.raises(ValueError, match="direction"):
            Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(8, 8, 8),
                     direction_xyz=(1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# LabelSchema
# ---------------------------------------------------------------------------


class TestLabelSchema:
    def test_standard_from_dataset_json(self):
        ds = {"labels": {"background": 0, "liver": 1, "spleen": 2}}
        s = LabelSchema.from_dataset_json(ds)
        assert not s.is_region_model
        assert s.num_outputs == 3
        assert s.name_of(1) == "liver"
        assert s.name_of(99) == "label_99"

    def test_region_from_dataset_json(self):
        ds = {
            "labels": {"background": 0, "WT": [1, 2, 3], "TC": [1, 3], "ET": [3]},
            "regions_class_order": [2, 1, 3],
        }
        s = LabelSchema.from_dataset_json(ds)
        assert s.is_region_model
        assert s.paint_priority == (2, 1, 3)
        assert s.num_outputs == len(s.regions) == 3

    def test_region_without_order_raises(self):
        ds = {"labels": {"background": 0, "WT": [1, 2, 3]}}
        with pytest.raises(ValueError, match="regions_class_order"):
            LabelSchema.from_dataset_json(ds)

    def test_region_objects(self):
        r = Region(label_value=3, member_classes=(1, 2, 3))
        assert r.label_value == 3
        assert r.member_classes == (1, 2, 3)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def _geom(shape=(8, 8, 8)):
    return Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=shape)


class TestVolume:
    def test_single_channel(self):
        v = Volume(data=mx.zeros((8, 8, 8, 1)), geometry=_geom())
        assert v.num_channels == 1
        assert not v.is_multichannel
        assert v.shape_zyx == (8, 8, 8)

    def test_multichannel(self):
        v = Volume(data=mx.zeros((8, 8, 8, 2)), geometry=_geom(),
                   channels=("PET", "CT"))
        assert v.is_multichannel
        assert v.num_channels == 2

    def test_requires_4d(self):
        with pytest.raises(ValueError, match="4-D"):
            Volume(data=mx.zeros((8, 8, 8)), geometry=_geom())

    def test_spatial_shape_must_match_geometry(self):
        with pytest.raises(ValueError, match="spatial shape"):
            Volume(data=mx.zeros((4, 8, 8, 1)), geometry=_geom())

    def test_channel_count_must_match_names(self):
        with pytest.raises(ValueError, match="channel"):
            Volume(data=mx.zeros((8, 8, 8, 2)), geometry=_geom(), channels=("CT",))

    def test_select_channels(self):
        v = Volume(data=mx.broadcast_to(mx.arange(2).reshape(1, 1, 1, 2),
                                         (8, 8, 8, 2)),
                   geometry=_geom(), channels=("PET", "CT"))
        ct = v.select_channels(["CT"])
        assert ct.channels == ("CT",)
        assert ct.num_channels == 1
        assert float(ct.data[0, 0, 0, 0]) == 1.0

    def test_select_unknown_channel_raises(self):
        v = Volume(data=mx.zeros((8, 8, 8, 1)), geometry=_geom())
        with pytest.raises(KeyError):
            v.select_channels(["MR"])

    def test_with_data_keeps_geometry(self):
        v = Volume(data=mx.zeros((8, 8, 8, 1)), geometry=_geom())
        v2 = v.with_data(mx.ones((8, 8, 8, 1)))
        assert v2.geometry is v.geometry
        assert float(v2.data[0, 0, 0, 0]) == 1.0


# ---------------------------------------------------------------------------
# Segmentation / Prediction
# ---------------------------------------------------------------------------


class TestSegmentationPrediction:
    def test_segmentation(self):
        s = LabelSchema(names={0: "background", 1: "liver"})
        seg = Segmentation(data=mx.zeros((8, 8, 8), dtype=mx.uint8),
                           geometry=_geom(), schema=s)
        assert seg.geometry.shape_zyx == (8, 8, 8)

    def test_segmentation_requires_3d(self):
        s = LabelSchema(names={0: "background"})
        with pytest.raises(ValueError, match="3-D"):
            Segmentation(data=mx.zeros((8, 8, 8, 1)), geometry=_geom(), schema=s)

    def test_probabilities(self):
        s = LabelSchema(names={0: "bg", 1: "a", 2: "b"})
        p = Prediction(data=mx.zeros((3, 8, 8, 8)), geometry=_geom(),
                          schema=s, activation="softmax")
        assert p.num_classes == 3
        assert p.activation == "softmax"

    def test_probabilities_requires_4d(self):
        s = LabelSchema(names={0: "bg"})
        with pytest.raises(ValueError, match="4-D"):
            Prediction(data=mx.zeros((8, 8, 8)), geometry=_geom(), schema=s)


# ---------------------------------------------------------------------------
# RestorePlan / BuildOptions
# ---------------------------------------------------------------------------


class TestRestorePlanBuildOptions:
    def test_restore_plan_is_a_value(self):
        g = _geom()
        plan = RestorePlan(
            source_geometry=g, source_orientation="SAR",
            inference_geometry=g, inference_orientation="LPS",
            model_spacing_zyx=(1.5, 1.5, 1.5),
        )
        assert plan.source_geometry is g
        assert plan.inference_orientation == "LPS"
        assert plan.model_spacing_zyx == (1.5, 1.5, 1.5)

    def test_build_options_hashable(self):
        a = BuildOptions(folds=(0, 1), configuration="3d_fullres")
        b = BuildOptions(folds=[0, 1], configuration="3d_fullres")
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_build_options_folds_all(self):
        o = BuildOptions()
        assert o.folds == "all"
        assert hash(o)  # hashable

    def test_build_options_distinct_keys(self):
        assert BuildOptions(configuration="a") != BuildOptions(configuration="b")
        assert hash(BuildOptions(folds=(0,))) != hash(BuildOptions(folds=(0, 1)))

    def test_build_options_has_no_run_knobs(self):
        # step_size / use_mirroring are run-time, not build identity — they
        # must NOT be on BuildOptions (else they'd wrongly force rebuilds).
        o = BuildOptions()
        assert not hasattr(o, "step_size")
        assert not hasattr(o, "use_mirroring")
