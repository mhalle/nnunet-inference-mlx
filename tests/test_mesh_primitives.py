"""Phase A smoke tests for the Mesh value type, mesh_concat, and the
npz serializer round-trip.

These are scaffolding tests: subsequent phases (cell sweep, multi-task
composite, VTK export) build on top.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_inference_mlx import (
    Geometry,
    LabelSchema,
    Mesh,
    mesh_concat,
    mesh_from_npz,
    mesh_to_npz,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _geom() -> Geometry:
    return Geometry(
        spacing_zyx=(1.5, 1.5, 1.5),
        shape_zyx=(64, 64, 64),
        origin_xyz=(0.0, 0.0, 0.0),
        direction_xyz=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _schema() -> LabelSchema:
    return LabelSchema(names={0: "background", 1: "spleen", 2: "kidney"})


def _single_quad_mesh(
    *,
    geometry: Geometry | None = None,
    schema: LabelSchema | None = None,
    label_pair: tuple[int, int] = (1, 0),
    with_normals: bool = False,
    with_stencils: bool = False,
) -> Mesh:
    """A trivial one-quad mesh for smoke tests."""
    g = geometry or _geom()
    s = schema or _schema()
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    quads = np.array([[0, 1, 2, 3]], dtype=np.int32)
    boundary_labels = np.array([label_pair], dtype=np.int32)
    normals = None
    if with_normals:
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1))
    stencils = None
    if with_stencils:
        # 4 points, each connected to its two cyclic neighbors.
        offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
        connectivity = np.array([1, 3, 0, 2, 1, 3, 0, 2], dtype=np.int32)
        stencils = (offsets, connectivity)
    return Mesh(
        points=points,
        quads=quads,
        boundary_labels=boundary_labels,
        geometry=g,
        schema=s,
        normals=normals,
        stencils=stencils,
    )


# ---------------------------------------------------------------------------
# Mesh value type
# ---------------------------------------------------------------------------


class TestMeshValueType:
    def test_empty_factory(self):
        g, s = _geom(), _schema()
        m = Mesh.empty(g, s)
        assert m.is_empty
        assert m.num_points == 0
        assert m.num_quads == 0
        assert m.points.shape == (0, 3)
        assert m.quads.shape == (0, 4)
        assert m.boundary_labels.shape == (0, 2)
        assert m.normals is None
        assert m.stencils is None
        assert m.geometry == g
        assert m.schema is s

    def test_empty_with_normals_and_stencils(self):
        m = Mesh.empty(_geom(), _schema(), with_normals=True, with_stencils=True)
        assert m.has_normals
        assert m.has_stencils
        assert m.normals.shape == (0, 3)
        assert m.stencils[0].shape == (1,)
        assert m.stencils[1].shape == (0,)

    def test_single_quad_well_formed(self):
        m = _single_quad_mesh()
        assert m.num_points == 4
        assert m.num_quads == 1
        assert not m.is_empty
        assert m.boundary_labels[0].tolist() == [1, 0]

    def test_bad_points_shape_rejected(self):
        with pytest.raises(ValueError, match="points must be"):
            Mesh(
                points=np.zeros((5,), dtype=np.float32),
                quads=np.zeros((0, 4), dtype=np.int32),
                boundary_labels=np.zeros((0, 2), dtype=np.int32),
                geometry=_geom(),
                schema=_schema(),
            )

    def test_bad_quads_shape_rejected(self):
        with pytest.raises(ValueError, match="quads must be"):
            Mesh(
                points=np.zeros((0, 3), dtype=np.float32),
                quads=np.zeros((1, 3), dtype=np.int32),
                boundary_labels=np.zeros((1, 2), dtype=np.int32),
                geometry=_geom(),
                schema=_schema(),
            )

    def test_quads_vs_labels_length_mismatch(self):
        with pytest.raises(ValueError, match="boundary_labels rows"):
            Mesh(
                points=np.zeros((4, 3), dtype=np.float32),
                quads=np.zeros((2, 4), dtype=np.int32),
                boundary_labels=np.zeros((3, 2), dtype=np.int32),
                geometry=_geom(),
                schema=_schema(),
            )

    def test_normals_shape_must_match_points(self):
        with pytest.raises(ValueError, match="normals shape"):
            Mesh(
                points=np.zeros((4, 3), dtype=np.float32),
                quads=np.zeros((0, 4), dtype=np.int32),
                boundary_labels=np.zeros((0, 2), dtype=np.int32),
                geometry=_geom(),
                schema=_schema(),
                normals=np.zeros((3, 3), dtype=np.float32),
            )

    def test_stencils_offsets_length(self):
        with pytest.raises(ValueError, match="offsets length"):
            Mesh(
                points=np.zeros((4, 3), dtype=np.float32),
                quads=np.zeros((0, 4), dtype=np.int32),
                boundary_labels=np.zeros((0, 2), dtype=np.int32),
                geometry=_geom(),
                schema=_schema(),
                stencils=(
                    np.array([0, 0, 0], dtype=np.int64),       # wrong length
                    np.zeros(0, dtype=np.int32),
                ),
            )


# ---------------------------------------------------------------------------
# mesh_concat
# ---------------------------------------------------------------------------


class TestMeshConcat:
    def test_empty_plus_empty(self):
        g, s = _geom(), _schema()
        m = mesh_concat(Mesh.empty(g, s), Mesh.empty(g, s))
        assert m.is_empty

    def test_empty_plus_single(self):
        g, s = _geom(), _schema()
        single = _single_quad_mesh(geometry=g, schema=s)
        out = mesh_concat(Mesh.empty(g, s), single)
        # Identity preserved (cheap path)
        assert out is single

    def test_single_plus_empty(self):
        g, s = _geom(), _schema()
        single = _single_quad_mesh(geometry=g, schema=s)
        out = mesh_concat(single, Mesh.empty(g, s))
        assert out is single

    def test_two_quads_offset_correctly(self):
        g, s = _geom(), _schema()
        a = _single_quad_mesh(geometry=g, schema=s, label_pair=(1, 0))
        b = _single_quad_mesh(geometry=g, schema=s, label_pair=(2, 0))
        out = mesh_concat(a, b)
        assert out.num_points == 8
        assert out.num_quads == 2
        # First quad keeps indices [0..3]; second quad's [0..3] shift to [4..7].
        np.testing.assert_array_equal(out.quads[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(out.quads[1], [4, 5, 6, 7])
        np.testing.assert_array_equal(out.boundary_labels, [[1, 0], [2, 0]])

    def test_geometry_mismatch_raises(self):
        s = _schema()
        a = _single_quad_mesh(geometry=_geom(), schema=s)
        g2 = Geometry(
            spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(64, 64, 64),
        )
        b = _single_quad_mesh(geometry=g2, schema=s)
        with pytest.raises(ValueError, match="geometry mismatch"):
            mesh_concat(a, b)

    def test_schema_identity_mismatch_raises(self):
        g = _geom()
        a = _single_quad_mesh(geometry=g, schema=_schema())
        b = _single_quad_mesh(geometry=g, schema=_schema())  # distinct object
        with pytest.raises(ValueError, match="schemas differ by identity"):
            mesh_concat(a, b)

    def test_normals_consistency_required(self):
        g, s = _geom(), _schema()
        a = _single_quad_mesh(geometry=g, schema=s, with_normals=True)
        b = _single_quad_mesh(geometry=g, schema=s, with_normals=False)
        with pytest.raises(ValueError, match="normals mismatch"):
            mesh_concat(a, b)

    def test_normals_concat(self):
        g, s = _geom(), _schema()
        a = _single_quad_mesh(geometry=g, schema=s, with_normals=True)
        b = _single_quad_mesh(geometry=g, schema=s, with_normals=True)
        out = mesh_concat(a, b)
        assert out.has_normals
        assert out.normals.shape == (8, 3)

    def test_stencils_concat_csr(self):
        g, s = _geom(), _schema()
        a = _single_quad_mesh(geometry=g, schema=s, with_stencils=True)
        b = _single_quad_mesh(geometry=g, schema=s, with_stencils=True)
        out = mesh_concat(a, b)
        assert out.has_stencils
        offsets, connectivity = out.stencils
        # 8 vertices → 9 offset entries
        assert offsets.shape == (9,)
        # All edges still represented; source connectivity offset by num_points(a)
        assert connectivity.shape == (16,)
        # Source's first vertex (id 0 in b) is id 4 in concat;
        # source's connectivity[0] (=1) becomes 5.
        assert int(connectivity[8]) == 5
        # Final tail offset equals total connectivity length
        assert int(offsets[-1]) == 16


# ---------------------------------------------------------------------------
# npz round-trip
# ---------------------------------------------------------------------------


class TestNpzRoundtrip:
    def test_empty_mesh(self, tmp_path):
        g, s = _geom(), _schema()
        m = Mesh.empty(g, s)
        out = mesh_from_npz(mesh_to_npz(m, tmp_path / "empty.npz"))
        assert out.is_empty
        assert out.geometry == g
        assert out.schema.names == s.names
        assert not out.has_normals
        assert not out.has_stencils

    def test_full_mesh(self, tmp_path):
        g, s = _geom(), _schema()
        m = _single_quad_mesh(
            geometry=g, schema=s, with_normals=True, with_stencils=True,
        )
        out = mesh_from_npz(mesh_to_npz(m, tmp_path / "full.npz"))
        np.testing.assert_array_equal(out.points, m.points)
        np.testing.assert_array_equal(out.quads, m.quads)
        np.testing.assert_array_equal(out.boundary_labels, m.boundary_labels)
        np.testing.assert_array_equal(out.normals, m.normals)
        np.testing.assert_array_equal(out.stencils[0], m.stencils[0])
        np.testing.assert_array_equal(out.stencils[1], m.stencils[1])
        assert out.geometry == m.geometry
        assert out.schema.names == m.schema.names

    def test_concat_round_trip(self, tmp_path):
        g, s = _geom(), _schema()
        a = _single_quad_mesh(geometry=g, schema=s, label_pair=(1, 0))
        b = _single_quad_mesh(geometry=g, schema=s, label_pair=(2, 0))
        concat = mesh_concat(a, b)
        out = mesh_from_npz(mesh_to_npz(concat, tmp_path / "concat.npz"))
        np.testing.assert_array_equal(out.points, concat.points)
        np.testing.assert_array_equal(out.quads, concat.quads)
        np.testing.assert_array_equal(out.boundary_labels, concat.boundary_labels)
