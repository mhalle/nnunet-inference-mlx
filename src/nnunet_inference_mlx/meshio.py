"""Mesh serializers — round-trip a :class:`Mesh` to disk or into vtkPolyData.

Parallel to :mod:`imageio` for :class:`Segmentation`: pure-numpy core,
no required external mesh library.

``mesh_to_npz`` / ``mesh_from_npz`` are the toolkit-canonical lossless
format, used for caching the result of an expensive ``to_mesh`` call and
reloading it later (or in a sibling process) without rerunning inference.

``mesh_to_vtk_polydata`` converts a Mesh into a ``vtkPolyData`` so the
result drops into any VTK-based pipeline (Slicer, vtkConstrainedSmoothing
Filter, vtkSurfaceNetsAtlas). VTK is a lazy import — installing this
package does not pull it in.

PLY / glTF / STL serializers are deferred until there's a concrete
consumer asking for them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from .values import Geometry, LabelSchema, Mesh, Region


PathLike = Union[str, Path]


def _geometry_to_dict(g: Geometry) -> dict:
    return {
        "spacing_zyx": list(g.spacing_zyx),
        "shape_zyx": list(g.shape_zyx),
        "origin_xyz": list(g.origin_xyz),
        "direction_xyz": list(g.direction_xyz),
    }


def _geometry_from_dict(d: dict) -> Geometry:
    return Geometry(
        spacing_zyx=tuple(d["spacing_zyx"]),
        shape_zyx=tuple(d["shape_zyx"]),
        origin_xyz=tuple(d["origin_xyz"]),
        direction_xyz=tuple(d["direction_xyz"]),
    )


def _schema_to_dict(s: LabelSchema) -> dict:
    return {
        "names": {str(k): v for k, v in s.names.items()},
        "regions": [
            {"label_value": r.label_value, "member_classes": list(r.member_classes)}
            for r in s.regions
        ],
        "paint_priority": list(s.paint_priority),
    }


def _schema_from_dict(d: dict) -> LabelSchema:
    names = {int(k): str(v) for k, v in d["names"].items()}
    regions = tuple(
        Region(
            label_value=int(r["label_value"]),
            member_classes=tuple(int(c) for c in r["member_classes"]),
        )
        for r in d.get("regions", [])
    )
    paint_priority = tuple(int(p) for p in d.get("paint_priority", []))
    return LabelSchema(names=names, regions=regions, paint_priority=paint_priority)


def mesh_to_npz(mesh: Mesh, path: PathLike) -> Path:
    """Serialize a :class:`Mesh` to a compressed npz file.

    Lossless round-trip via :func:`mesh_from_npz`. Layout:

      * ``points``, ``quads``, ``boundary_labels`` — required arrays
      * ``normals`` — present iff ``mesh.has_normals``
      * ``stencils_offsets``, ``stencils_connectivity`` — present iff
        ``mesh.has_stencils``
      * ``geometry_json``, ``schema_json`` — 0-D unicode arrays carrying
        the dataclass payloads as JSON

    Returns the written path for convenience.
    """
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "points": mesh.points,
        "quads": mesh.quads,
        "boundary_labels": mesh.boundary_labels,
        "geometry_json": np.asarray(json.dumps(_geometry_to_dict(mesh.geometry))),
        "schema_json": np.asarray(json.dumps(_schema_to_dict(mesh.schema))),
    }
    if mesh.has_normals:
        payload["normals"] = mesh.normals
    if mesh.has_stencils:
        offsets, connectivity = mesh.stencils
        payload["stencils_offsets"] = offsets
        payload["stencils_connectivity"] = connectivity
    np.savez_compressed(path, **payload)
    return path


def mesh_from_npz(path: PathLike) -> Mesh:
    """Load a :class:`Mesh` previously written by :func:`mesh_to_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        points = f["points"]
        quads = f["quads"]
        boundary_labels = f["boundary_labels"]
        geometry = _geometry_from_dict(json.loads(str(f["geometry_json"])))
        schema = _schema_from_dict(json.loads(str(f["schema_json"])))
        normals = f["normals"] if "normals" in f.files else None
        stencils: tuple[np.ndarray, np.ndarray] | None = None
        if "stencils_offsets" in f.files:
            stencils = (f["stencils_offsets"], f["stencils_connectivity"])
    return Mesh(
        points=points,
        quads=quads,
        boundary_labels=boundary_labels,
        geometry=geometry,
        schema=schema,
        normals=normals,
        stencils=stencils,
    )


# ---------------------------------------------------------------------------
# VTK PolyData converter (lazy VTK import)
# ---------------------------------------------------------------------------


def _points_to_world_xyz(
    points_zyx: np.ndarray, geometry: Geometry
) -> np.ndarray:
    """Map ``(N, 3)`` training-grid index coords in (Z, Y, X) order to
    ``(N, 3)`` world-mm coords in (X, Y, Z) order via the SITK affine.

    SITK convention: ``world = origin + direction @ (spacing * index)``,
    with ``index``, ``spacing``, ``direction`` and ``origin`` all in XYZ.
    Our mesh stores points in (Z, Y, X) — swap component order first.
    """
    points_xyz = np.ascontiguousarray(points_zyx[:, ::-1].astype(np.float32))
    spacing_xyz = np.asarray(geometry.spacing_zyx[::-1], dtype=np.float32)
    direction = np.asarray(geometry.direction_xyz, dtype=np.float32).reshape(3, 3)
    origin = np.asarray(geometry.origin_xyz, dtype=np.float32)
    scaled = points_xyz * spacing_xyz[None, :]
    return np.ascontiguousarray(scaled @ direction.T + origin[None, :])


def _normals_to_world_xyz(
    normals_zyx: np.ndarray, geometry: Geometry
) -> np.ndarray:
    """Rotate ``(N, 3)`` direction-only normals from (Z, Y, X) index to
    (X, Y, Z) world. Skips translation (normals are vectors) and spacing
    (we treat the normal as a direction, not a face area co-vector — fine
    for isotropic spacing, an approximation under anisotropic spacing
    that this MVP accepts)."""
    normals_xyz = np.ascontiguousarray(normals_zyx[:, ::-1].astype(np.float32))
    direction = np.asarray(geometry.direction_xyz, dtype=np.float32).reshape(3, 3)
    rotated = normals_xyz @ direction.T
    norms = np.linalg.norm(rotated, axis=1, keepdims=True)
    return np.ascontiguousarray(rotated / np.maximum(norms, np.float32(1e-30)))


def mesh_to_vtk_polydata(mesh: Mesh):
    """Convert a :class:`Mesh` to a ``vtkPolyData``.

    Coordinate conversion:
      * Points: (Z, Y, X) training-grid index → (X, Y, Z) world mm via
        ``geometry`` (origin + direction · diag(spacing) · index).
      * Normals (if present): rotation only by ``direction``, normalized.

    Cell-data array layout matches native ``vtkSurfaceNets3D`` output, so
    anything downstream that already speaks the SurfaceNets contract
    (``vtkSurfaceNetsAtlas``, ``vtkConstrainedSmoothingFilter``, Slicer's
    segmentation pipeline) consumes ours unchanged:

      * ``"BoundaryLabels"`` cell data, 2 components per face, in our
        :data:`Mesh.boundary_labels` order.
      * ``"Normals"`` point data, 3 components, only if the mesh carries
        them.

    Raises ``ImportError`` (with a helpful message) if the ``vtk`` package
    is not installed — VTK is a lazy/optional dependency of this package.
    """
    try:
        import vtk
        from vtk.util.numpy_support import (
            numpy_to_vtk,
            numpy_to_vtkIdTypeArray,
        )
    except ImportError as e:
        raise ImportError(
            "mesh_to_vtk_polydata requires the 'vtk' package; install with "
            "`pip install vtk` or add it to your project deps."
        ) from e

    pd = vtk.vtkPolyData()

    # Points in world-mm (X, Y, Z) order.
    points_world = _points_to_world_xyz(mesh.points, mesh.geometry)
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(points_world, deep=1))
    pd.SetPoints(vpts)

    # Quads as a vtkCellArray (VTK 9+ offsets + connectivity layout).
    M = mesh.num_quads
    if M > 0:
        offsets = np.arange(0, 4 * (M + 1), 4, dtype=np.int64)
        connectivity = mesh.quads.astype(np.int64, copy=False).ravel()
        cells = vtk.vtkCellArray()
        cells.SetData(
            numpy_to_vtkIdTypeArray(offsets, deep=1),
            numpy_to_vtkIdTypeArray(connectivity, deep=1),
        )
        pd.SetPolys(cells)

        # 2-component cell-data BoundaryLabels — vtkSurfaceNets3D convention.
        blabels = numpy_to_vtk(
            np.ascontiguousarray(mesh.boundary_labels.astype(np.int32, copy=False)),
            deep=1,
        )
        blabels.SetName("BoundaryLabels")
        # numpy_to_vtk on a (M, 2) array sets components automatically, but
        # be explicit for downstream consumers that inspect the metadata.
        blabels.SetNumberOfComponents(2)
        pd.GetCellData().AddArray(blabels)

    if mesh.has_normals:
        normals_world = _normals_to_world_xyz(mesh.normals, mesh.geometry)
        vnormals = numpy_to_vtk(normals_world, deep=1)
        vnormals.SetName("Normals")
        vnormals.SetNumberOfComponents(3)
        pd.GetPointData().SetNormals(vnormals)

    return pd


# ---------------------------------------------------------------------------
# Constrained Laplacian smoothing (lazy VTK)
# ---------------------------------------------------------------------------


def _world_xyz_to_grid_zyx(
    points_xyz: np.ndarray, geometry: Geometry,
) -> np.ndarray:
    """Inverse of :func:`_points_to_world_xyz`. Used after VTK smoothing
    to bring vertex positions back into the toolkit's (Z, Y, X) index
    coord convention."""
    direction = np.asarray(geometry.direction_xyz, dtype=np.float32).reshape(3, 3)
    origin = np.asarray(geometry.origin_xyz, dtype=np.float32)
    spacing_xyz = np.asarray(geometry.spacing_zyx[::-1], dtype=np.float32)
    centered = points_xyz - origin[None, :]
    rotated = centered @ np.linalg.inv(direction).T
    indices_xyz = rotated / spacing_xyz[None, :]
    return np.ascontiguousarray(indices_xyz[:, ::-1])


def mesh_smooth(
    mesh: Mesh,
    *,
    iterations: int = 2,
    relaxation: float = 0.1,
    constraint_voxels: float = 0.5,
) -> Mesh:
    """Apply ``vtkConstrainedSmoothingFilter`` to a Mesh's vertices.

    Returns a new :class:`Mesh` with smoothed vertex positions; quads,
    boundary labels, geometry, and schema are preserved unchanged.

    The smoother "borrows bandwidth" from neighboring cells — at saddle
    junctions and other locally-ambiguous configurations the cell-by-cell
    SurfaceNets dual vertex placement is intrinsically noisy (Nyquist:
    the boundary surface has sub-grid features the corner labels can't
    resolve), and constrained Laplacian relaxation pulls them toward
    the local consensus position from neighbors that aren't ambiguous.

    Defaults are deliberately *light* (2 iterations, relaxation 0.1,
    constraint distance = half a voxel): enough to remove visible
    junction noise without rounding off real anatomy. Tune up if needed.

    Requires the ``vtk`` package (lazy-imported, raises ``ImportError``
    with an install hint if missing).
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as e:
        raise ImportError(
            "mesh_smooth requires the 'vtk' package; install with "
            "`pip install vtk` or add it to your project deps."
        ) from e

    pd = mesh_to_vtk_polydata(mesh)

    sm = vtk.vtkConstrainedSmoothingFilter()
    sm.SetInputData(pd)
    sm.SetNumberOfIterations(int(iterations))
    sm.SetRelaxationFactor(float(relaxation))
    sm.SetConstraintStrategyToConstraintDistance()
    voxel_mm = float(min(mesh.geometry.spacing_zyx))
    sm.SetConstraintDistance(float(constraint_voxels) * voxel_mm)
    sm.Update()

    smoothed_pts_xyz = vtk_to_numpy(sm.GetOutput().GetPoints().GetData())
    new_points = _world_xyz_to_grid_zyx(
        smoothed_pts_xyz.astype(np.float32), mesh.geometry,
    )

    # Smoothing invalidates any per-vertex normals the input carried —
    # they were computed at the pre-smoothing positions and are no
    # longer aligned with the moved vertices. Renderers will recompute
    # mesh-averaged normals on the fly; if the caller wants them baked,
    # ``mesh_compute_normals`` does that explicitly.
    from dataclasses import replace
    return replace(mesh, points=new_points, normals=None)


def mesh_compute_normals(mesh: Mesh) -> Mesh:
    """Bake mesh-averaged per-vertex normals via ``vtkPolyDataNormals``.

    Returns a new :class:`Mesh` with ``normals`` set to the area-weighted
    average of the incident face normals for each vertex — the same
    operator that gives Phong/Gouraud-smooth shading in VTK and Slicer
    renderers. Use after :func:`mesh_smooth` if you need normals
    pre-baked into a file rather than computed at render time.

    Quads, boundary labels, and the geometry/schema are preserved
    unchanged; only ``normals`` changes.

    Requires the ``vtk`` package (lazy-imported).
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as e:
        raise ImportError(
            "mesh_compute_normals requires the 'vtk' package; install with "
            "`pip install vtk`."
        ) from e

    pd = mesh_to_vtk_polydata(mesh)
    pdn = vtk.vtkPolyDataNormals()
    pdn.SetInputData(pd)
    pdn.ComputePointNormalsOn()
    pdn.ComputeCellNormalsOff()
    pdn.SplittingOff()
    pdn.ConsistencyOff()
    pdn.AutoOrientNormalsOff()
    pdn.Update()

    n_xyz_world = vtk_to_numpy(pdn.GetOutput().GetPointData().GetNormals())
    # Rotate world (X, Y, Z) normals back into grid (Z, Y, X) component
    # order via the inverse of the geometry's direction matrix
    # (orthogonal — transpose is the inverse).
    direction = np.asarray(mesh.geometry.direction_xyz, dtype=np.float32).reshape(3, 3)
    rotated_back = n_xyz_world.astype(np.float32) @ direction
    new_normals = np.ascontiguousarray(rotated_back[:, ::-1])
    # Normalize defensively.
    mag = np.linalg.norm(new_normals, axis=1, keepdims=True)
    new_normals = new_normals / np.maximum(mag, np.float32(1e-10))

    from dataclasses import replace
    return replace(mesh, normals=new_normals.astype(np.float32))


def mesh_cleanup(
    mesh: Mesh,
    *,
    min_region_cells: int = 200,
    smooth_iters: int = 30,
    smooth_passband: float = 0.05,
) -> Mesh:
    """Mesh-side cleanup pass: drop tiny disconnected regions, then
    low-pass smooth.

    Two VTK filters in sequence:

      1. ``vtkPolyDataConnectivityFilter`` keeps only connected regions
         with at least ``min_region_cells`` faces. Drops floating
         disconnected debris and small "spike clusters" that survive
         the voxel-side `confidence_margin` / `drop_components_below_mm3`
         filters (those see the spike's voxels as face-connected to
         the main organ, so they aren't removed at the labelmap level —
         but the SurfaceNets mesh of the spike forms a small isolated
         protrusion that connectivity does separate from the main bulk).

      2. ``vtkWindowedSincPolyDataFilter`` is a volume-preserving low-
         pass filter that flattens high-frequency protrusions (residual
         pyramid spikes, single-voxel surface bumps) without shrinking
         the bulk of the mesh — unlike Laplacian smoothing, which
         shrinks. Aggressive (30 iter, passband 0.05) by default.

    Both filters preserve the ``BoundaryLabels`` cell-data and the quad
    topology. ``Normals`` are re-baked via :func:`mesh_compute_normals`
    after smoothing so shading stays consistent with the new positions.

    Recommended as a final pass on upsampled meshes from
    :func:`postprocess.to_mesh` with ``output_spacing_zyx`` / ``scale``
    upsamples — particularly for thin/curved structures like ribs or
    vessels, where 1-voxel spike clusters survive the per-voxel rules.

    Requires the ``vtk`` package (lazy-imported).
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as e:
        raise ImportError(
            "mesh_cleanup requires the 'vtk' package; install with "
            "`pip install vtk`."
        ) from e
    from dataclasses import replace

    pd = mesh_to_vtk_polydata(mesh)

    # ---- 1. Connectivity: identify and drop small isolated regions ---------
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(pd)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    sizes = vtk_to_numpy(conn.GetRegionSizes())
    keep_ids = [int(i) for i, s in enumerate(sizes) if int(s) >= min_region_cells]

    if not keep_ids:
        # Pathological: every region is below threshold. Return empty mesh.
        return Mesh.empty(mesh.geometry, mesh.schema, with_normals=mesh.has_normals)

    conn_keep = vtk.vtkPolyDataConnectivityFilter()
    conn_keep.SetInputData(pd)
    conn_keep.SetExtractionModeToSpecifiedRegions()
    for rid in keep_ids:
        conn_keep.AddSpecifiedRegion(rid)
    conn_keep.Update()

    # ---- 2. Clean: strip unreferenced points after region drop -------------
    cleanp = vtk.vtkCleanPolyData()
    cleanp.SetInputConnection(conn_keep.GetOutputPort())
    cleanp.PointMergingOn()
    cleanp.Update()

    # ---- 3. WindowedSinc: low-pass smooth (volume-preserving) --------------
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(cleanp.GetOutputPort())
    smooth.SetNumberOfIterations(int(smooth_iters))
    smooth.SetPassBand(float(smooth_passband))
    smooth.NonManifoldSmoothingOff()
    smooth.NormalizeCoordinatesOn()
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.Update()
    cleaned_pd = smooth.GetOutput()

    # ---- 4. Read back into a Mesh ------------------------------------------
    pts_xyz = vtk_to_numpy(cleaned_pd.GetPoints().GetData()).astype(np.float32)
    new_points = _world_xyz_to_grid_zyx(pts_xyz, mesh.geometry)

    polys = cleaned_pd.GetPolys()
    n_cells = cleaned_pd.GetNumberOfCells()
    if n_cells == 0:
        return Mesh.empty(mesh.geometry, mesh.schema, with_normals=mesh.has_normals)
    connectivity = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int32)
    new_quads = connectivity.reshape(n_cells, 4)

    cd = cleaned_pd.GetCellData()
    bl_vtk = cd.GetArray("BoundaryLabels")
    if bl_vtk is None:
        raise RuntimeError(
            "mesh_cleanup: BoundaryLabels cell-data lost during VTK pipeline; "
            "this should not happen with the standard filters."
        )
    new_bl = vtk_to_numpy(bl_vtk).reshape(n_cells, -1).astype(np.int32)

    new_mesh = replace(
        mesh,
        points=np.ascontiguousarray(new_points.astype(np.float32)),
        quads=np.ascontiguousarray(new_quads),
        boundary_labels=np.ascontiguousarray(new_bl),
        normals=None,  # smoothed positions invalidate prior normals
    )
    if mesh.has_normals:
        new_mesh = mesh_compute_normals(new_mesh)
    return new_mesh


__all__ = [
    "mesh_to_npz",
    "mesh_from_npz",
    "mesh_to_vtk_polydata",
    "mesh_smooth",
    "mesh_compute_normals",
    "mesh_cleanup",
]
