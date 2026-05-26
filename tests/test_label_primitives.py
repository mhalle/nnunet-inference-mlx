"""Unit tests for the top-level label-manipulation primitives.

These are the building blocks for the multi-task union pattern
(``run_label_union_workflow``, v0.9.0): caller-orchestrated remap +
paint operations decomposed into individually-importable, individually-
testable functions.

* ``remap_labels`` — vectorized lookup-table remap from one integer
  label space to another (e.g. task-local IDs → unified IDs).
* ``paint_union`` — overwrite-where-nonzero merge. Same convention as
  ``_slab_resample_paint``: caller controls priority via call order.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_inference_mlx import paint_union, remap_labels


# ---------------------------------------------------------------------------
# remap_labels
# ---------------------------------------------------------------------------


class TestRemapLabels:
    def test_basic_mapping(self):
        seg = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        out = remap_labels(seg, {1: 10, 2: 20})
        np.testing.assert_array_equal(out, [[0, 10, 20], [20, 10, 0]])

    def test_unmapped_drops_to_background(self):
        """IDs not in the mapping (and not background) become background.

        This is the "explicit drop" semantic — predictable, no silent
        carry-through of task-local IDs into the unified space.
        """
        seg = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        out = remap_labels(seg, {1: 100, 3: 300})
        np.testing.assert_array_equal(out, [0, 100, 0, 300, 0])

    def test_background_preserved(self):
        seg = np.zeros((3, 3), dtype=np.uint8)
        out = remap_labels(seg, {})
        np.testing.assert_array_equal(out, np.zeros((3, 3)))

    def test_custom_background(self):
        seg = np.array([0, 1, 5, 0], dtype=np.uint8)
        # Drop background sentinel "5" by remapping it onto background=255.
        out = remap_labels(seg, {1: 10}, background=255, out_dtype=np.uint8)
        # Note: source 0 maps to background=255 because it's not in mapping
        # and explicit drop sends "everything-not-in-mapping" to background.
        np.testing.assert_array_equal(out, [255, 10, 255, 255])

    def test_auto_dtype_uint8(self):
        out = remap_labels(np.array([1], dtype=np.uint8), {1: 200})
        assert out.dtype == np.uint8

    def test_auto_dtype_uint16(self):
        out = remap_labels(np.array([1], dtype=np.uint8), {1: 5000})
        assert out.dtype == np.uint16

    def test_auto_dtype_uint32(self):
        out = remap_labels(np.array([1], dtype=np.uint8), {1: 200_000})
        assert out.dtype == np.uint32

    def test_explicit_out_dtype(self):
        out = remap_labels(
            np.array([1], dtype=np.uint8), {1: 5},
            out_dtype="uint16",
        )
        assert out.dtype == np.uint16

    def test_rejects_negative_ids(self):
        with pytest.raises(ValueError, match="negative"):
            remap_labels(np.zeros(3, dtype=np.uint8), {1: -1})

    def test_rejects_non_array_input(self):
        with pytest.raises(TypeError, match="numpy"):
            remap_labels([0, 1, 2], {1: 5})  # type: ignore[arg-type]

    def test_shape_preserved(self):
        seg = np.random.randint(0, 5, size=(7, 8, 9), dtype=np.uint8)
        out = remap_labels(seg, {1: 10, 2: 20, 3: 30, 4: 40})
        assert out.shape == seg.shape

    def test_empty_array(self):
        seg = np.zeros((0,), dtype=np.uint8)
        out = remap_labels(seg, {1: 5})
        assert out.shape == (0,)


# ---------------------------------------------------------------------------
# paint_union
# ---------------------------------------------------------------------------


class TestPaintUnion:
    def test_paints_nonzero(self):
        target = np.zeros(5, dtype=np.uint8)
        source = np.array([0, 1, 0, 2, 0], dtype=np.uint8)
        out = paint_union(target, source)
        np.testing.assert_array_equal(out, [0, 1, 0, 2, 0])

    def test_zero_source_does_not_overwrite(self):
        """source==0 is transparent — target keeps its prior value."""
        target = np.array([5, 5, 5, 5], dtype=np.uint8)
        source = np.array([0, 0, 0, 0], dtype=np.uint8)
        out = paint_union(target, source)
        np.testing.assert_array_equal(out, [5, 5, 5, 5])

    def test_overwrite_at_overlap(self):
        """Non-zero source overwrites existing non-zero target — later wins."""
        target = np.array([1, 1, 1, 1], dtype=np.uint8)
        source = np.array([0, 2, 0, 2], dtype=np.uint8)
        out = paint_union(target, source)
        np.testing.assert_array_equal(out, [1, 2, 1, 2])

    def test_priority_via_call_order(self):
        """Two layers painted onto background in order: later wins overlap."""
        target = np.zeros(5, dtype=np.uint8)
        low_pri = np.array([1, 1, 1, 0, 0], dtype=np.uint8)
        high_pri = np.array([0, 0, 2, 2, 0], dtype=np.uint8)
        paint_union(target, low_pri)
        paint_union(target, high_pri)
        # Position 2 had both — high_pri painted last, so it wins.
        np.testing.assert_array_equal(target, [1, 1, 2, 2, 0])

    def test_in_place_returns_target(self):
        target = np.zeros(3, dtype=np.uint8)
        source = np.array([0, 1, 0], dtype=np.uint8)
        out = paint_union(target, source)
        # paint_union is documented as in-place + returns target.
        assert out is target
        np.testing.assert_array_equal(target, [0, 1, 0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            paint_union(np.zeros(5, dtype=np.uint8),
                        np.zeros(6, dtype=np.uint8))

    def test_3d_shape(self):
        target = np.zeros((4, 4, 4), dtype=np.uint16)
        source = np.zeros((4, 4, 4), dtype=np.uint16)
        source[1:3, 1:3, 1:3] = 7
        paint_union(target, source)
        assert target[2, 2, 2] == 7
        assert target[0, 0, 0] == 0

    def test_preserves_target_dtype(self):
        target = np.zeros(3, dtype=np.uint16)
        source = np.array([0, 1, 0], dtype=np.uint8)
        paint_union(target, source)
        assert target.dtype == np.uint16


# ---------------------------------------------------------------------------
# Composition: the multi-task union recipe in one place
# ---------------------------------------------------------------------------


class TestRemapPlusPaintRecipe:
    """Sanity check the canonical multi-task union pattern: each stage
    remaps its task-local labels into the unified space, then paints
    into the running union. List order = priority."""

    def test_two_task_union(self):
        # Task A: task-local labels 1=liver, 2=spleen
        seg_a = np.array([0, 1, 1, 2, 2, 0], dtype=np.uint8)
        # Task B: task-local labels 1=left_kidney, 2=right_kidney
        seg_b = np.array([0, 0, 1, 0, 2, 0], dtype=np.uint8)

        # Unified space: 10=liver, 11=spleen, 20=lkidney, 21=rkidney
        unified = np.zeros(seg_a.shape, dtype=np.uint8)
        paint_union(unified, remap_labels(seg_a, {1: 10, 2: 11}))
        paint_union(unified, remap_labels(seg_b, {1: 20, 2: 21}))

        # Position 2: A says liver(10), B says lkidney(20). B paints last → 20.
        # Position 4: A says spleen(11), B says rkidney(21). B paints last → 21.
        np.testing.assert_array_equal(unified, [0, 10, 20, 11, 21, 0])
