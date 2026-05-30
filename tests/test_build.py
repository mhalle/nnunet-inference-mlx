"""Tests for build_model / LoadedModel (Phase 3 spine).

Builds a tiny synthetic ModelData (no disk, no real weights), compiles it
into a LoadedModel, and runs the full segment() path on a small Volume —
exercising build → forward-resample → infer → inverse-resample → restore,
reusing the proven engine/resampling internals.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from nnunet_inference_mlx.build import LoadedModel, build_model
from nnunet_inference_mlx.model_data import ModelData
from nnunet_inference_mlx.plans import build_network_from_plans
from nnunet_inference_mlx.values import Geometry, Prediction, Volume

sitk = pytest.importorskip("SimpleITK")


def _make_model_data(num_classes: int = 3) -> ModelData:
    plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [16, 16, 16],
                "spacing": [1.5, 1.5, 1.5],
                "normalization_schemes": ["ZScoreNormalization"],
                "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3]],
                "n_conv_per_stage_encoder": [2, 2],
                "n_conv_per_stage_decoder": [1],
                "UNet_base_num_features": 4,
            }
        },
        "foreground_intensity_properties_per_channel": {},
    }
    dataset = {
        "labels": {"background": 0, **{f"class_{i}": i for i in range(1, num_classes)}},
        "channel_names": {"0": "CT"},
    }
    net = build_network_from_plans(plans, "3d_fullres", 1, num_classes,
                                    deep_supervision=False)
    weights = dict(nn.utils.tree_flatten(net.parameters()))
    return ModelData(plans=plans, dataset=dataset, fold_weights=(weights,),
                     ecosystem="test", id=1)


def _volume(shape=(24, 24, 24)):
    return Volume(
        data=mx.random.normal((*shape, 1)),
        geometry=Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=shape),
        channels=("CT",),
    )


class TestBuildModel:
    def test_build_returns_loaded_model(self):
        m = build_model(_make_model_data())
        assert isinstance(m, LoadedModel)
        assert m.schema.num_outputs == 3
        assert m.config.id == 1

    def test_memory_mb_positive(self):
        m = build_model(_make_model_data())
        assert m.memory_mb > 0

    def test_multi_config_2d_first_resolves_3d_target_spacing(self):
        # TS part models list configs as ['2d','3d_lowres','3d_fullres',...] —
        # "2d" first, with a 2-element spacing. build_model must thread the
        # resolved configuration so engine.target_spacing is the 3D spacing,
        # not the first-config 2D one. (Found via real Dataset291 full run.)
        md = _make_model_data()
        cfg3d = md.plans["configurations"]["3d_fullres"]
        md.plans["configurations"] = {
            "2d": {**cfg3d, "patch_size": [16, 16], "spacing": [1.5, 1.5]},
            "3d_fullres": cfg3d,
        }
        assert next(iter(md.plans["configurations"])) == "2d"   # 2d is first
        m = build_model(md)
        assert m.config.config_name == "3d_fullres"
        ts = m._engine.target_spacing
        assert len(ts) == 3 and tuple(ts) == (1.5, 1.5, 1.5), ts

    def test_segment_returns_segmentation_in_input_geometry(self):
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        seg = m.segment(vol)
        assert seg.geometry.shape_zyx == (24, 24, 24)
        assert tuple(seg.data.shape) == (24, 24, 24)
        assert seg.schema.names == m.schema.names

    def test_segment_preserves_spacing(self):
        m = build_model(_make_model_data())
        vol = _volume((24, 24, 24))
        seg = m.segment(vol)
        assert seg.geometry.spacing_zyx == (1.0, 1.0, 1.0)

    def test_predict_returns_prediction_at_target_spacing(self):
        # logits are first-class: predict() stops at the model's native output
        m = build_model(_make_model_data())
        pred = m.predict(_volume((24, 24, 24)))      # input 1.0mm; target 1.5mm
        assert isinstance(pred, Prediction)
        assert pred.num_classes == 3
        assert pred.activation == "logits"           # single fold → raw logits
        assert pred.geometry.spacing_zyx == (1.5, 1.5, 1.5)
        assert pred.data.ndim == 4                   # (K, Z, Y, X)

    def test_close_releases_and_blocks_use(self):
        m = build_model(_make_model_data())
        m.close()
        with pytest.raises(RuntimeError, match="closed"):
            m.segment(_volume())

    def test_context_manager(self):
        with build_model(_make_model_data()) as m:
            seg = m.segment(_volume())
            assert seg.geometry.shape_zyx == (24, 24, 24)
        assert m._engine is None  # closed on exit

    def test_store_integration_uses_real_build(self, tmp_path):
        """A ModelStore with the real build (injected read returning our
        synthetic ModelData) loads a working LoadedModel and sizes it."""
        from nnunet_inference_mlx.store import ModelStore

        (tmp_path / "Dataset1_X" / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir(parents=True)
        md = _make_model_data()
        store = ModelStore("nnunet", model_root_dir=tmp_path,
                           read=lambda folder, **kw: md)  # real build
        m = store.load(1)
        assert isinstance(m, LoadedModel)
        assert store.loaded_mb > 0
        seg = m.segment(_volume())
        assert seg.geometry.shape_zyx == (24, 24, 24)
        store.unload_all()
        assert len(store) == 0
