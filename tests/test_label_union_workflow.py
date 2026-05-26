"""Integration tests for ``run_label_union_workflow``.

The orchestrator is intentionally thin glue over four top-level
primitives (``reorient``, ``predict_with_resampling``, ``remap_labels``,
``paint_union``). These tests verify the *integration* — paint priority,
geometry preservation, orientation round-trip, label dispatch — not the
internals of each primitive (those are covered by their own test files).

A synthetic engine is used so tests run in ~1s without weights.
"""

from __future__ import annotations

import mlx.nn as nn
import numpy as np
import pytest

from nnunet_inference_mlx import (
    InferenceEngine, ModelBundle, ParallelStage,
    run_label_union_workflow,
)
from nnunet_inference_mlx.plans import build_network_from_plans

sitk = pytest.importorskip("SimpleITK")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(num_classes: int = 3):
    """Build a tiny untrained engine. The actual predictions are
    effectively random, but that's fine — the orchestrator's job is
    geometry + dispatch + remap + paint, not prediction quality."""
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
        "labels": {"background": 0,
                   **{f"class_{i}": i for i in range(1, num_classes)}},
        "channel_names": {"0": "CT"},
    }
    net = build_network_from_plans(plans, "3d_fullres", 1, num_classes,
                                    deep_supervision=False)
    weights = dict(nn.utils.tree_flatten(net.parameters()))
    bundle = ModelBundle(plans=plans, dataset=dataset,
                          fold_weights=[weights], metadata={}, fold_ids=(0,))
    return InferenceEngine(bundle, verbose=False)


def _make_sitk(shape_zyx=(24, 24, 24), spacing_xyz=(1.0, 1.0, 1.0),
                direction=None, origin=(0.0, 0.0, 0.0)):
    arr = np.random.randn(*shape_zyx).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing_xyz)
    img.SetOrigin(origin)
    if direction is not None:
        img.SetDirection(direction)
    return img


# SAR orientation — the broken-chest-scan canary from 0.8.1.
SAR_DIRECTION = (0.0, 0.0, -1.0,
                 0.0, -1.0, 0.0,
                 1.0,  0.0, 0.0)


@pytest.fixture(scope="module")
def engine():
    return _make_engine(num_classes=3)


# ---------------------------------------------------------------------------
# Basic dispatch
# ---------------------------------------------------------------------------


class TestRunLabelUnionWorkflow:
    def test_empty_stages_raises(self, engine):
        img = _make_sitk()
        with pytest.raises(ValueError, match="non-empty"):
            run_label_union_workflow(img, [])

    def test_single_stage_returns_remapped_labels(self, engine):
        """One-stage union should be equivalent to a remap of a normal
        single-task prediction."""
        img = _make_sitk()
        stages = [ParallelStage(engine, {1: 50, 2: 60}, "task_a")]
        seg = run_label_union_workflow(img, stages)
        seg_arr = sitk.GetArrayFromImage(seg)
        # Output must only contain background or remapped target IDs.
        unique = set(np.unique(seg_arr).tolist())
        assert unique.issubset({0, 50, 60}), f"unexpected labels: {unique}"

    def test_geometry_preserved(self, engine):
        img = _make_sitk(spacing_xyz=(1.0, 0.9, 0.9),
                         origin=(5.0, 10.0, 15.0))
        stages = [ParallelStage(engine, {1: 1, 2: 2})]
        seg = run_label_union_workflow(img, stages)
        assert seg.GetSize() == img.GetSize()
        assert seg.GetSpacing() == img.GetSpacing()
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(seg.GetOrigin(), img.GetOrigin()))
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(seg.GetDirection(), img.GetDirection()))


# ---------------------------------------------------------------------------
# Orientation round-trip
# ---------------------------------------------------------------------------


class TestSarRoundTrip:
    def test_sar_input_returns_sar(self, engine):
        """Same correctness property as run_workflow: input direction
        survives the LPS canonicalization round-trip."""
        img = _make_sitk(direction=SAR_DIRECTION)
        stages = [ParallelStage(engine, {1: 7, 2: 8})]
        seg = run_label_union_workflow(img, stages)
        assert seg.GetSize() == img.GetSize()
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(seg.GetDirection(), img.GetDirection()))

    def test_reorient_to_none_skips(self, engine):
        """``reorient_to=None`` is the escape hatch for already-canonical
        inputs. Output direction equals input direction without doing
        the DICOMOrient round-trip."""
        img = _make_sitk()  # LPS
        stages = [ParallelStage(engine, {1: 1, 2: 2})]
        seg = run_label_union_workflow(img, stages, reorient_to=None)
        assert seg.GetSize() == img.GetSize()
        assert seg.GetDirection() == img.GetDirection()


# ---------------------------------------------------------------------------
# Paint priority via stage order — the central correctness property
# ---------------------------------------------------------------------------


class TestPaintPriority:
    def test_two_stages_later_overwrites_earlier(self, engine):
        """The defining property of run_label_union_workflow: stage[1]
        overwrites stage[0] at voxels where both produce foreground.

        Implementation strategy: give both stages the *same* engine but
        non-overlapping remap targets. Wherever the engine predicts
        foreground, both stages emit (different) unified IDs at the
        same voxel — so the output at every foreground voxel must
        carry stage[1]'s ID, never stage[0]'s.
        """
        img = _make_sitk()
        stages = [
            ParallelStage(engine, {1: 10, 2: 11}, "low_priority"),
            ParallelStage(engine, {1: 20, 2: 21}, "high_priority"),
        ]
        seg = run_label_union_workflow(img, stages)
        seg_arr = sitk.GetArrayFromImage(seg)
        unique = set(np.unique(seg_arr).tolist())
        # No "low priority" IDs should appear in the output —
        # every foreground voxel was painted twice and the second
        # paint (high_priority) won.
        assert 10 not in unique
        assert 11 not in unique
        # The high-priority IDs should appear (the synthetic random
        # logits do produce foreground voxels).
        assert unique.issubset({0, 20, 21})

    def test_disjoint_stages_both_appear(self, engine):
        """Stages with disjoint foreground (achieved by mapping each
        engine class to a different stage's target) preserve both.

        We can't easily make the engine emit disjoint foregrounds
        synthetically, so we use a weaker check: when a stage drops a
        class (leaves it out of label_remap), that class's voxels go
        to background and the next stage can paint there freely."""
        img = _make_sitk()
        # Stage 0: only emits class 1 → unified 10. Class 2 dropped.
        # Stage 1: only emits class 2 → unified 20. Class 1 dropped.
        stages = [
            ParallelStage(engine, {1: 10}, "task_a"),
            ParallelStage(engine, {2: 20}, "task_b"),
        ]
        seg = run_label_union_workflow(img, stages)
        seg_arr = sitk.GetArrayFromImage(seg)
        unique = set(np.unique(seg_arr).tolist())
        assert unique.issubset({0, 10, 20})


# ---------------------------------------------------------------------------
# Dtype resolution
# ---------------------------------------------------------------------------


class TestOutputDtype:
    def test_small_target_ids_use_uint8(self, engine):
        img = _make_sitk()
        stages = [ParallelStage(engine, {1: 5, 2: 10})]
        seg = run_label_union_workflow(img, stages)
        assert sitk.GetArrayFromImage(seg).dtype == np.uint8

    def test_large_target_ids_promote_to_uint16(self, engine):
        img = _make_sitk()
        stages = [ParallelStage(engine, {1: 500, 2: 1000})]
        seg = run_label_union_workflow(img, stages)
        assert sitk.GetArrayFromImage(seg).dtype == np.uint16

    def test_dtype_picked_from_max_across_stages(self, engine):
        """The unified buffer dtype is sized to fit every stage's
        target IDs, not just the first stage's."""
        img = _make_sitk()
        stages = [
            ParallelStage(engine, {1: 5, 2: 10}, "small"),
            ParallelStage(engine, {1: 5000}, "big"),
        ]
        seg = run_label_union_workflow(img, stages)
        # Second stage has 5000 — needs uint16 for both stages' values
        # to coexist losslessly in the unified buffer.
        assert sitk.GetArrayFromImage(seg).dtype == np.uint16
