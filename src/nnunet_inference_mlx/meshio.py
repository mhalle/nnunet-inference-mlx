"""Mesh serializers — round-trip a :class:`Mesh` to disk.

Parallel to :mod:`imageio` for :class:`Segmentation`: pure-numpy in/out,
no required external mesh library. ``mesh_to_npz`` / ``mesh_from_npz`` are
the toolkit-canonical lossless format, used for caching the result of an
expensive ``to_mesh`` call and reloading it later (or in a sibling process)
without rerunning inference.

VTK PolyData / PLY / glTF / STL serializers are deferred until there's a
concrete consumer asking for them; this module's only obligation right now
is "save the value type and load it back identically."
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


__all__ = ["mesh_to_npz", "mesh_from_npz"]
