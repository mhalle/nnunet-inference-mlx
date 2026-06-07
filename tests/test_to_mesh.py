"""Phase D unit test for postprocess.to_mesh — the toolkit-API entrypoint.

The real end-to-end on TS-fast is exercised by examples/06_mesh_output.py;
this test just confirms the wiring (Prediction → Mesh) and the
region-model guard.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from nnunet_inference_mlx import (
    Geometry,
    LabelSchema,
    Mesh,
    Prediction,
    Region,
    postprocess,
)


def _prediction_single_voxel_cube() -> Prediction:
    labelmap = np.zeros((5, 5, 5), dtype=np.int32)
    labelmap[2, 2, 2] = 1
    K = 2
    logits = np.full((K, 5, 5, 5), -1.0, dtype=np.float32)
    for k in range(K):
        logits[k][labelmap == k] = 1.0
    return Prediction(
        data=mx.array(logits),
        geometry=Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(5, 5, 5)),
        schema=LabelSchema(names={0: "background", 1: "fg"}),
        activation="logits",
    )


def test_to_mesh_returns_mesh_at_same_grid():
    pred = _prediction_single_voxel_cube()
    mesh = postprocess.to_mesh(pred)
    assert isinstance(mesh, Mesh)
    assert mesh.geometry == pred.geometry
    assert mesh.schema is pred.schema
    # Same expected topology as the standalone surfacenets_logits cube test.
    assert mesh.num_points == 8
    assert mesh.num_quads == 6


def test_to_mesh_rejects_region_model():
    pred = Prediction(
        data=mx.zeros((3, 5, 5, 5), dtype=mx.float32),
        geometry=Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(5, 5, 5)),
        schema=LabelSchema(
            names={1: "WT", 2: "TC", 3: "ET"},
            regions=(
                Region(label_value=1, member_classes=(1, 2, 3)),
                Region(label_value=2, member_classes=(1, 3)),
                Region(label_value=3, member_classes=(3,)),
            ),
            paint_priority=(2, 1, 3),
        ),
        activation="sigmoid",
    )
    with pytest.raises(NotImplementedError, match="region-based"):
        postprocess.to_mesh(pred)
