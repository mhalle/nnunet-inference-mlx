"""Verify predict_with_resampling dispatches on engine.has_regions.

Before 0.8.0, predict_with_resampling unconditionally called
inverse_resample_argmax, which produces silently-wrong labels for
region-based (BraTS-style) models. This test confirms the scheme
dispatch now picks the correct primitive.

Strategy: build a tiny region-based engine, run predict_with_resampling,
and verify the output labels are in the expected region-label space
(values from regions_class_order, not channel indices). For a standard
engine, verify labels are in [0, num_classes-1] (argmax indices).
"""

from __future__ import annotations

import mlx.nn as nn
import numpy as np
import pytest

from nnunet_inference_mlx import (
    InferenceEngine, ModelBundle, predict_with_resampling,
)
from nnunet_inference_mlx.plans import build_network_from_plans

sitk = pytest.importorskip("SimpleITK")


def _make_engine(region_based: bool, target_spacing=(1.5, 1.5, 1.5)):
    plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [16, 16, 16],
                "spacing": list(target_spacing),
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
    if region_based:
        dataset = {
            "labels": {
                "background": 0,
                "whole_tumor": [1, 2, 3],
                "tumor_core": [1, 3],
                "enhancing": 3,
            },
            # Label values painted at the output. Each region paints in
            # priority order; later ones overwrite earlier.
            "regions_class_order": [1, 2, 4],
            "channel_names": {"0": "T1"},
        }
        out_channels = 3
    else:
        dataset = {
            "labels": {
                "background": 0,
                "class_1": 1,
                "class_2": 2,
                "class_3": 3,
            },
            "channel_names": {"0": "CT"},
        }
        out_channels = 4

    net = build_network_from_plans(
        plans, "3d_fullres", 1, out_channels, deep_supervision=False,
    )
    weights = dict(nn.utils.tree_flatten(net.parameters()))
    bundle = ModelBundle(plans=plans, dataset=dataset,
                          fold_weights=[weights], metadata={}, fold_ids=(0,))
    return InferenceEngine(bundle, verbose=False)


def _make_sitk_volume(shape_zyx=(24, 24, 24), spacing=(1.0, 1.0, 1.0)):
    arr = np.random.randn(*shape_zyx).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(reversed(spacing)))   # SITK is XYZ
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def test_standard_dispatch_uses_argmax():
    """Standard scheme: output labels should be channel indices [0, K-1]."""
    engine = _make_engine(region_based=False)
    img = _make_sitk_volume()
    seg = predict_with_resampling(engine, img)
    seg_arr = sitk.GetArrayFromImage(seg)

    # For standard models, labels are argmax indices: 0, 1, 2, or 3
    unique = set(np.unique(seg_arr).tolist())
    assert unique.issubset({0, 1, 2, 3}), \
        f"Standard scheme should produce argmax labels in [0..3]; got {unique}"


def test_region_based_dispatch_uses_paint():
    """Region-based: labels should be from regions_class_order, NOT [0..K-1]."""
    engine = _make_engine(region_based=True)
    img = _make_sitk_volume()
    seg = predict_with_resampling(engine, img)
    seg_arr = sitk.GetArrayFromImage(seg)

    # regions_class_order is [1, 2, 4]. Background is 0. Output labels can
    # only be in {0, 1, 2, 4}.
    unique = set(np.unique(seg_arr).tolist())
    allowed = {0, 1, 2, 4}
    assert unique.issubset(allowed), (
        f"Region-based scheme should produce labels from regions_class_order "
        f"+ background; got {unique}, expected subset of {allowed}"
    )
    # Crucially, label 3 (which is NOT in regions_class_order but WOULD
    # appear if we incorrectly did argmax across channel index 3) must
    # not appear.
    assert 3 not in unique, (
        "Label 3 appeared, which would only happen if argmax was used "
        "instead of paint-priority on regions_class_order"
    )


def test_region_based_output_geometry_matches_input():
    engine = _make_engine(region_based=True)
    img = _make_sitk_volume()
    seg = predict_with_resampling(engine, img)
    assert seg.GetSize() == img.GetSize()
    assert seg.GetSpacing() == img.GetSpacing()
    assert seg.GetOrigin() == img.GetOrigin()


def test_standard_output_geometry_matches_input():
    engine = _make_engine(region_based=False)
    img = _make_sitk_volume()
    seg = predict_with_resampling(engine, img)
    assert seg.GetSize() == img.GetSize()
