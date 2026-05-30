"""Tests for segment() (Phase 4): named-task / pipeline dispatch on the new path.

Uses synthetic ModelData (real build) behind a ModelStore whose read returns
it for any id, and runs single / cascade / label_union recipes on a small
Volume — exercising the dispatch + the bridge to the proven workflows.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from nnunet_inference_mlx.catalog import TaskCatalog
from nnunet_inference_mlx.model_data import ModelData
from nnunet_inference_mlx.plans import build_network_from_plans
from nnunet_inference_mlx.segment import segment
from nnunet_inference_mlx.store import ModelStore
from nnunet_inference_mlx.tasks import CascadeStep, TaskSpec, UnionPart
from nnunet_inference_mlx.values import Geometry, Volume

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
                     ecosystem="ts", id=1)


def _volume(shape=(28, 28, 28)):
    return Volume(data=mx.random.normal((*shape, 1)),
                  geometry=Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=shape),
                  channels=("CT",))


def _store(tmp_path, ids):
    for i in ids:
        (tmp_path / f"Dataset{i}_X" / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir(parents=True)
    md = _make_model_data()
    return ModelStore("nnunet", model_root_dir=tmp_path,
                      read=lambda folder, **kw: md)   # real build


class TestSegmentSingle:
    def test_run_single_recipe(self, tmp_path):
        store = _store(tmp_path, [1])
        spec = TaskSpec(name="t", source="ts", modality="CT", shape="single", single=1)
        seg = segment(spec, _volume(), store=store)
        assert seg.geometry.shape_zyx == (28, 28, 28)
        assert tuple(seg.data.shape) == (28, 28, 28)

    def test_run_by_name_via_catalog(self, tmp_path):
        store = _store(tmp_path, [1])
        cat = TaskCatalog()
        cat.register(TaskSpec(name="mytask", source="ts", modality="CT",
                              shape="single", single=1))
        seg = segment("mytask", _volume(), store=store, catalog=cat)
        assert seg.geometry.shape_zyx == (28, 28, 28)


class TestSegmentCascade:
    def test_run_inline_cascade(self, tmp_path):
        store = _store(tmp_path, [1, 2])
        spec = TaskSpec(
            name="casc", source="ts", modality="CT", shape="cascade",
            cascade=(CascadeStep(weights_id=1, crop_to_classes=(1,)),
                     CascadeStep(weights_id=2)),
            label_map={1: "a", 2: "b"},
        )
        seg = segment(spec, _volume(), store=store)
        assert seg.geometry.shape_zyx == (28, 28, 28)

    def test_run_nested_cascade_via_catalog(self, tmp_path):
        store = _store(tmp_path, [1, 2, 3])
        cropper = TaskSpec(name="cropper", source="ts", modality="CT",
                           shape="cascade",
                           cascade=(CascadeStep(weights_id=1, crop_to_classes=(1,)),
                                    CascadeStep(weights_id=2)))
        target = TaskSpec(name="target", source="ts", modality="CT", shape="cascade",
                          cascade=(CascadeStep(crop_from_task="cropper",
                                               crop_to_classes=(1,)),
                                   CascadeStep(weights_id=3)),
                          label_map={1: "x"})
        cat = TaskCatalog()
        cat.register(cropper)
        cat.register(target)
        seg = segment(target, _volume(), store=store, catalog=cat)
        assert seg.geometry.shape_zyx == (28, 28, 28)


class TestSegmentUnion:
    def test_run_union(self, tmp_path):
        store = _store(tmp_path, [1, 2])
        spec = TaskSpec(
            name="uni", source="ts", modality="CT", shape="label_union",
            union=(UnionPart(weights_id=1, label_remap={1: 1, 2: 2}, name="p1"),
                   UnionPart(weights_id=2, label_remap={1: 3}, name="p2")),
            label_map={1: "x", 2: "y", 3: "z"},
        )
        seg = segment(spec, _volume(), store=store)
        assert seg.geometry.shape_zyx == (28, 28, 28)
        # union output uses the unified schema's names
        assert seg.schema.names == {1: "x", 2: "y", 3: "z"}
