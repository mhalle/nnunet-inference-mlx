"""Phase C tests for mesh_to_vtk_polydata.

Skipped if VTK isn't available — VTK is a lazy/optional dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_inference_mlx import (
    Geometry,
    LabelSchema,
    Mesh,
    surfacenets_logits,
)

vtk = pytest.importorskip("vtk")
from nnunet_inference_mlx import mesh_to_vtk_polydata  # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy           # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _identity_geom(shape: tuple[int, int, int]) -> Geometry:
    return Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=shape)


def _cube_mesh() -> Mesh:
    """Single-voxel cube at (2, 2, 2) in a 5×5×5 background — same as
    the Phase B canonical case."""
    labelmap = np.zeros((5, 5, 5), dtype=np.int32)
    labelmap[2, 2, 2] = 1
    K = 2
    Z, Y, X = labelmap.shape
    logits = np.full((K, Z, Y, X), -1.0, dtype=np.float32)
    for k in range(K):
        logits[k][labelmap == k] = 1.0
    schema = LabelSchema(names={0: "background", 1: "fg"})
    return surfacenets_logits(logits, _identity_geom((5, 5, 5)), schema)


# ---------------------------------------------------------------------------
# Basic round-trip into vtkPolyData
# ---------------------------------------------------------------------------


class TestPolyDataConversion:
    def test_returns_vtk_polydata(self):
        pd = mesh_to_vtk_polydata(_cube_mesh())
        assert pd.IsA("vtkPolyData")

    def test_point_count_matches(self):
        m = _cube_mesh()
        pd = mesh_to_vtk_polydata(m)
        assert pd.GetNumberOfPoints() == m.num_points

    def test_cell_count_matches(self):
        m = _cube_mesh()
        pd = mesh_to_vtk_polydata(m)
        assert pd.GetNumberOfCells() == m.num_quads

    def test_all_cells_are_quads(self):
        m = _cube_mesh()
        pd = mesh_to_vtk_polydata(m)
        for i in range(pd.GetNumberOfCells()):
            assert pd.GetCell(i).GetNumberOfPoints() == 4

    def test_boundary_labels_array_present(self):
        m = _cube_mesh()
        pd = mesh_to_vtk_polydata(m)
        cd = pd.GetCellData()
        arr = cd.GetArray("BoundaryLabels")
        assert arr is not None
        assert arr.GetNumberOfComponents() == 2
        assert arr.GetNumberOfTuples() == m.num_quads
        # Every tuple should be (1, 0) on the single-voxel cube.
        for i in range(arr.GetNumberOfTuples()):
            assert tuple(arr.GetTuple(i)) == (1.0, 0.0)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


class TestCoordinateConversion:
    def test_identity_geometry_yields_xyz_swap(self):
        """Identity spacing / origin / direction → world(X, Y, Z) =
        stored(Z, Y, X) reversed."""
        m = _cube_mesh()
        pd = mesh_to_vtk_polydata(m)
        pts = vtk_to_numpy(pd.GetPoints().GetData())
        # The exported point order matches the mesh point order; just check
        # that exported XYZ == stored ZYX reversed.
        np.testing.assert_allclose(pts, m.points[:, ::-1], atol=1e-5)

    def test_nonidentity_origin_and_spacing(self):
        labelmap = np.zeros((5, 5, 5), dtype=np.int32)
        labelmap[2, 2, 2] = 1
        K = 2
        logits = np.full((K, 5, 5, 5), -1.0, dtype=np.float32)
        for k in range(K):
            logits[k][labelmap == k] = 1.0
        g = Geometry(
            spacing_zyx=(2.0, 1.5, 1.0),
            shape_zyx=(5, 5, 5),
            origin_xyz=(10.0, 20.0, 30.0),
        )
        schema = LabelSchema(names={0: "background", 1: "fg"})
        m = surfacenets_logits(logits, g, schema)
        pd = mesh_to_vtk_polydata(m)
        pts = vtk_to_numpy(pd.GetPoints().GetData())

        # Verify world position of a known stored point: expected =
        # origin + spacing_xyz * index_xyz.
        # Pick the first stored point and reconstruct.
        idx_zyx = m.points[0]
        spacing_zyx = np.array(g.spacing_zyx, dtype=np.float32)
        spacing_xyz = spacing_zyx[::-1]
        origin_xyz = np.array(g.origin_xyz, dtype=np.float32)
        expected = origin_xyz + spacing_xyz * idx_zyx[::-1]
        np.testing.assert_allclose(pts[0], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Topological sanity in VTK space
# ---------------------------------------------------------------------------


class TestPolyDataTopology:
    def test_polydata_is_closed(self):
        """vtkFeatureEdges should find no boundary edges on the cube mesh."""
        pd = mesh_to_vtk_polydata(_cube_mesh())
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(pd)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()
        n_boundary = fe.GetOutput().GetNumberOfCells()
        assert n_boundary == 0, f"expected closed surface, got {n_boundary} boundary edges"

    def test_face_normals_point_outward_into_background(self):
        """vtkPolyDataNormals on the cube mesh: every face normal should
        point away from the foreground voxel (2, 2, 2) — outward."""
        m = _cube_mesh()
        pd = mesh_to_vtk_polydata(m)
        pdn = vtk.vtkPolyDataNormals()
        pdn.SetInputData(pd)
        pdn.ComputeCellNormalsOn()
        pdn.ComputePointNormalsOff()
        # No consistency / splitting / flipping — trust the source winding.
        pdn.ConsistencyOff()
        pdn.SplittingOff()
        pdn.AutoOrientNormalsOff()
        pdn.Update()
        out = pdn.GetOutput()
        cell_normals = vtk_to_numpy(out.GetCellData().GetNormals())

        # Centroid of each face. The foreground voxel center is at world
        # (2, 2, 2) under identity geometry.
        fg_center = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        pts = vtk_to_numpy(out.GetPoints().GetData())
        for i in range(out.GetNumberOfCells()):
            cell = out.GetCell(i)
            ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
            centroid = pts[ids].mean(axis=0)
            outward = centroid - fg_center
            outward /= np.linalg.norm(outward)
            # Normal should point in roughly the same direction as outward.
            dot = float(np.dot(cell_normals[i], outward))
            assert dot > 0.5, f"face {i} normal {cell_normals[i]} not outward (dot={dot})"


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


class TestOptionalFields:
    def test_no_normals_by_default(self):
        pd = mesh_to_vtk_polydata(_cube_mesh())
        assert pd.GetPointData().GetNormals() is None

    def test_normals_emitted_when_present(self):
        m = _cube_mesh()
        # Synthesize a Mesh with normals (just normalize the points as a
        # placeholder direction — we're only testing the converter).
        norms = m.points / np.linalg.norm(m.points, axis=1, keepdims=True)
        m_with = Mesh(
            points=m.points, quads=m.quads, boundary_labels=m.boundary_labels,
            geometry=m.geometry, schema=m.schema,
            normals=norms.astype(np.float32),
        )
        pd = mesh_to_vtk_polydata(m_with)
        vn = pd.GetPointData().GetNormals()
        assert vn is not None
        assert vn.GetNumberOfComponents() == 3
        assert vn.GetNumberOfTuples() == m.num_points


# ---------------------------------------------------------------------------
# Empty mesh
# ---------------------------------------------------------------------------


class TestEmptyMesh:
    def test_empty_mesh_makes_empty_polydata(self):
        g = _identity_geom((5, 5, 5))
        s = LabelSchema(names={0: "background"})
        m = Mesh.empty(g, s)
        pd = mesh_to_vtk_polydata(m)
        assert pd.GetNumberOfPoints() == 0
        assert pd.GetNumberOfCells() == 0
