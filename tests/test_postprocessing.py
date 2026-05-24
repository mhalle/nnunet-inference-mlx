"""Tests for postprocessing.remove_small_components (cc3d backend)."""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_inference_mlx import remove_small_components

cc3d = pytest.importorskip("cc3d")


def _make_labels_with_blob_and_specks():
    """Build a label volume with one big blob of class 1, one big blob of
    class 2, and a few tiny specks of class 1.
    """
    labels = np.zeros((40, 40, 40), dtype=np.uint8)
    labels[5:25, 5:25, 5:25] = 1     # big label-1 cube: 8000 vox
    labels[28:38, 28:38, 28:38] = 2  # big label-2 cube: 1000 vox
    labels[0, 0, 0] = 1              # 1-vox speck
    labels[0, 0, 5] = 1              # 1-vox speck
    labels[39, 39, 39] = 2           # 1-vox speck of class 2
    return labels


def test_removes_small_specks_preserves_big_blobs():
    labels = _make_labels_with_blob_and_specks()
    out = remove_small_components(
        labels, spacing_zyx=(1.0, 1.0, 1.0), min_volume_mm3=10.0,
    )
    # Big blobs survive
    assert out[10, 10, 10] == 1
    assert out[30, 30, 30] == 2
    # Specks dropped
    assert out[0, 0, 0] == 0
    assert out[0, 0, 5] == 0
    assert out[39, 39, 39] == 0


def test_zero_threshold_is_noop():
    labels = _make_labels_with_blob_and_specks()
    out = remove_small_components(labels, spacing_zyx=(1.0, 1.0, 1.0),
                                   min_volume_mm3=0.0)
    np.testing.assert_array_equal(out, labels)


def test_negative_threshold_is_noop():
    labels = _make_labels_with_blob_and_specks()
    out = remove_small_components(labels, spacing_zyx=(1.0, 1.0, 1.0),
                                   min_volume_mm3=-5.0)
    np.testing.assert_array_equal(out, labels)


def test_preserves_input_when_not_in_place():
    labels = _make_labels_with_blob_and_specks()
    snapshot = labels.copy()
    _ = remove_small_components(
        labels, spacing_zyx=(1.0, 1.0, 1.0), min_volume_mm3=10.0,
        in_place=False,
    )
    np.testing.assert_array_equal(labels, snapshot)


def test_threshold_uses_physical_units():
    """At 2 mm spacing, 8 mm^3 = 1 voxel — so any 1-vox speck dropped
    only when min_volume_mm3 > 8."""
    labels = np.zeros((10, 10, 10), dtype=np.uint8)
    labels[1, 1, 1] = 1   # single voxel
    spacing = (2.0, 2.0, 2.0)  # 8 mm^3 / vox

    # Threshold of 5 mm^3 is below 8 — voxel survives
    out_kept = remove_small_components(labels, spacing, min_volume_mm3=5.0)
    assert out_kept[1, 1, 1] == 1

    # Threshold of 50 mm^3 — voxel dropped
    out_dropped = remove_small_components(labels, spacing, min_volume_mm3=50.0)
    assert out_dropped[1, 1, 1] == 0


def test_multilabel_respects_class_boundaries():
    """Two same-shape blobs of different classes should not be merged into
    one component just because they touch."""
    labels = np.zeros((20, 20, 20), dtype=np.uint8)
    labels[0:10, :, :] = 1
    labels[10:20, :, :] = 2
    out = remove_small_components(labels, (1.0, 1.0, 1.0), min_volume_mm3=10.0)
    # Both halves survive with their original IDs
    assert (out[0:10, :, :] == 1).all()
    assert (out[10:20, :, :] == 2).all()


def test_label_dtype_preserved():
    labels = np.zeros((10, 10, 10), dtype=np.uint16)
    labels[2:8, 2:8, 2:8] = 5
    out = remove_small_components(labels, (1.0, 1.0, 1.0), min_volume_mm3=10.0)
    assert out.dtype == np.uint16
