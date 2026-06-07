"""Phase B tests for surfacenets_logits.

These exercise the algorithm on tiny synthetic logit volumes with known
expected topology + vertex positions. No VTK reference comparison — that
was the Phase 0 we deliberately cut. Validation against a real TS
sub-task is "open in Slicer" later (Phase D).
"""

from __future__ import annotations

import numpy as np

from nnunet_inference_mlx import (
    Geometry,
    LabelSchema,
    Mesh,
    surfacenets_logits,
)


def _geom(shape: tuple[int, int, int]) -> Geometry:
    return Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=shape)


def _schema_2c() -> LabelSchema:
    return LabelSchema(names={0: "background", 1: "fg"})


def _schema_3c() -> LabelSchema:
    return LabelSchema(names={0: "background", 1: "fg_a", 2: "fg_b"})


def _onehot(labelmap: np.ndarray, num_classes: int) -> np.ndarray:
    """(Z, Y, X) labelmap → (K, Z, Y, X) one-hot logits (1.0 / -1.0)."""
    K = num_classes
    Z, Y, X = labelmap.shape
    out = np.full((K, Z, Y, X), -1.0, dtype=np.float32)
    for k in range(K):
        out[k][labelmap == k] = 1.0
    return out


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------


class TestEmptyAndTrivial:
    def test_too_small_volume_returns_empty(self):
        # 1×1×1 has no cells — nothing to extract.
        labelmap = np.zeros((1, 1, 1), dtype=np.int32)
        m = surfacenets_logits(_onehot(labelmap, 1), _geom((1, 1, 1)), _schema_2c())
        assert m.is_empty

    def test_uniform_volume_returns_empty(self):
        labelmap = np.zeros((5, 5, 5), dtype=np.int32)
        m = surfacenets_logits(_onehot(labelmap, 2), _geom((5, 5, 5)), _schema_2c())
        assert m.is_empty

    def test_geometry_shape_mismatch_raises(self):
        labelmap = np.zeros((5, 5, 5), dtype=np.int32)
        logits = _onehot(labelmap, 2)
        try:
            surfacenets_logits(logits, _geom((4, 5, 5)), _schema_2c())
        except ValueError as e:
            assert "spatial shape" in str(e)
        else:
            assert False, "expected ValueError"


# ---------------------------------------------------------------------------
# Single-voxel cube — exact topology
# ---------------------------------------------------------------------------


class TestSingleVoxelCube:
    """A single label-1 voxel at the center of a 5×5×5 background volume.

    Expected SurfaceNets dual: 8 vertices (one per cell touching the
    center voxel), 6 quads (one per face of the cube), all BoundaryLabels
    = (1, 0) — foreground first per VTK rule (background in slot 1).
    The mesh is closed: V − E + F = 8 − 12 + 6 = 2.
    """

    def _build(self):
        labelmap = np.zeros((5, 5, 5), dtype=np.int32)
        labelmap[2, 2, 2] = 1
        return surfacenets_logits(_onehot(labelmap, 2), _geom((5, 5, 5)), _schema_2c())

    def test_vertex_count(self):
        m = self._build()
        assert m.num_points == 8

    def test_quad_count(self):
        m = self._build()
        assert m.num_quads == 6

    def test_all_boundary_labels_are_fg_bg(self):
        m = self._build()
        np.testing.assert_array_equal(m.boundary_labels, np.tile([[1, 0]], (6, 1)))

    def test_euler_characteristic(self):
        """V − E + F = 2 for a sphere-topology mesh."""
        m = self._build()
        V = m.num_points
        F = m.num_quads
        # Each quad has 4 edges; each edge of a closed manifold is shared
        # by exactly 2 quads. So distinct edges = 4F / 2 = 2F.
        edges = set()
        for q in m.quads:
            for k in range(4):
                a, b = int(q[k]), int(q[(k + 1) % 4])
                edges.add((min(a, b), max(a, b)))
        E = len(edges)
        assert E == 2 * F, f"expected closed mesh (E = 2F = {2*F}); got E={E}"
        assert V - E + F == 2, f"Euler characteristic = {V - E + F}, want 2"

    def test_vertices_inside_central_cells(self):
        """Each dual vertex should lie within one of the 8 cells adjacent
        to voxel (2, 2, 2). Each component of every vertex lies in [1, 3]."""
        m = self._build()
        assert (m.points >= 1.0).all() and (m.points <= 3.0).all()

    def test_each_vertex_pulled_toward_center(self):
        """The label-1 corner is at voxel (2, 2, 2). Each cell has one
        corner there and seven background corners — the dual vertex is the
        centroid of three sub-voxel edge crossings on those three edges,
        so its distance to (2, 2, 2) is the same for all 8 cells (by
        symmetry of the 8 corners around the central voxel)."""
        m = self._build()
        d = np.linalg.norm(m.points - np.array([2.0, 2.0, 2.0]), axis=1)
        assert np.allclose(d, d[0], atol=1e-5), f"distances: {d}"


# ---------------------------------------------------------------------------
# Half-volume — planar surface, normal direction, vertex positions
# ---------------------------------------------------------------------------


class TestHalfVolumeAlongX:
    """Volume split along X: voxels with x < 3 are background, x ≥ 3 are
    label 1. The boundary is the plane between x=2 and x=3 — vertices at
    sub-voxel x ≈ 2.5.

    On a 5×5×5 volume, the interior X-edges (z ∈ [1, 3], y ∈ [1, 3]) that
    cross the split contribute 3×3 = 9 quads on that plane.
    """

    def _build(self):
        labelmap = np.zeros((5, 5, 5), dtype=np.int32)
        labelmap[:, :, 3:] = 1
        return surfacenets_logits(_onehot(labelmap, 2), _geom((5, 5, 5)), _schema_2c())

    def test_some_quads_emitted(self):
        m = self._build()
        # 9 interior X-edges on the split plane.
        assert m.num_quads == 9

    def test_all_quads_have_fg_bg_boundary(self):
        m = self._build()
        np.testing.assert_array_equal(
            m.boundary_labels, np.tile([[1, 0]], (m.num_quads, 1))
        )

    def test_vertices_at_half_voxel_in_x(self):
        m = self._build()
        # All x-coords should be ≈ 2.5 (the sub-voxel split between voxels
        # 2 and 3). Other components are at integer positions (since only
        # the X-edges of each boundary cell are crossed for this split).
        np.testing.assert_allclose(m.points[:, 2], 2.5, atol=1e-5)

    def test_face_normal_points_into_background(self):
        """VTK convention: normal points Label0 → Label1.
        Label0 = fg = 1, Label1 = bg = 0. So the geometric normal points
        from the foreground (x ≥ 3) into the background (x < 3), i.e. -X.

        ``Mesh.points`` is stored in (Z, Y, X) component order — a
        left-handed permutation of (X, Y, Z), so ``np.cross`` on the
        stored components gives the geometric normal with sign flipped.
        Mirror the (Z, Y, X) → (X, Y, Z) swap that the VTK / glTF / PLY
        exporters apply before computing the cross product.
        """
        m = self._build()
        q = m.quads[0]
        p0 = m.points[q[0]][::-1]   # (Z, Y, X) → (X, Y, Z)
        p1 = m.points[q[1]][::-1]
        p2 = m.points[q[2]][::-1]
        n = np.cross(p1 - p0, p2 - p0)
        # X-component (index 0 in XYZ) should be strongly negative.
        assert n[0] < -0.1, f"normal X-component = {n[0]}; expected < 0"
        # Y and Z components should be ~0 (planar quad in YZ).
        assert abs(n[1]) < 1e-4 and abs(n[2]) < 1e-4


# ---------------------------------------------------------------------------
# Multi-class boundary — VTK Label0/Label1 sort
# ---------------------------------------------------------------------------


class TestMultiClassBoundaryLabels:
    """A volume split along X into three slabs of labels 1, 2, 0.

    The two interior split planes carry different BoundaryLabels:
      * x = midpoint between voxel x=1 and x=2: labels (1, 2) — both
        non-background → sort ascending → (1, 2)
      * x = midpoint between voxel x=3 and x=4: labels (2, 0) — background
        involved → background in slot 1 → (2, 0)
    """

    def _build(self):
        labelmap = np.zeros((5, 5, 6), dtype=np.int32)
        labelmap[:, :, :2] = 1     # x ∈ {0, 1} : label 1
        labelmap[:, :, 2:4] = 2    # x ∈ {2, 3} : label 2
        labelmap[:, :, 4:] = 0     # x ∈ {4, 5} : label 0 (background)
        return surfacenets_logits(_onehot(labelmap, 3), _geom((5, 5, 6)), _schema_3c())

    def test_two_label_pairs_present(self):
        m = self._build()
        pairs = {tuple(p) for p in m.boundary_labels.tolist()}
        assert pairs == {(1, 2), (2, 0)}

    def test_pair_1_2_sorted_ascending(self):
        m = self._build()
        # Every (1, 2) row: smaller label first, since neither is background.
        rows_12 = m.boundary_labels[
            (m.boundary_labels[:, 0] == 1) | (m.boundary_labels[:, 1] == 1)
        ]
        assert (rows_12 == np.array([1, 2])).all()

    def test_pair_2_0_background_last(self):
        m = self._build()
        # Every row involving label 0: 0 must be in slot 1.
        rows_with_bg = m.boundary_labels[
            (m.boundary_labels == 0).any(axis=1)
        ]
        assert (rows_with_bg[:, 1] == 0).all()
        assert (rows_with_bg[:, 0] != 0).all()


# ---------------------------------------------------------------------------
# Return shape is a Mesh value type
# ---------------------------------------------------------------------------


class TestReturnTypeIsMesh:
    def test_returns_mesh_with_geometry_and_schema(self):
        labelmap = np.zeros((5, 5, 5), dtype=np.int32)
        labelmap[2, 2, 2] = 1
        g, s = _geom((5, 5, 5)), _schema_2c()
        m = surfacenets_logits(_onehot(labelmap, 2), g, s)
        assert isinstance(m, Mesh)
        assert m.geometry == g
        assert m.schema is s
        assert m.points.dtype == np.float32
        assert m.quads.dtype == np.int32
        assert m.boundary_labels.dtype == np.int32
        assert m.normals is None
        assert m.stencils is None
