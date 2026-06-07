"""Logit-based surface extraction (Phase B MVP).

Reads a K-channel logit volume at training spacing and emits a SurfaceNets
dual mesh whose vertex positions come from edge-crossing interpolation in
the *continuous* logit field — not from a discretized label map.

No smoothing pass; no full case-table state machine; no non-manifold
detection. The MVP we agreed for testing whether logit-based surfaces are
viable, before investing in any of that machinery.

The algorithm in three lines:
  1. Argmax per voxel → label per voxel.
  2. For each grid edge whose endpoint labels differ, linearly interpolate
     ``logit_i - logit_j`` (the two dominant labels) to find t ∈ [0, 1]
     where it crosses zero.
  3. For each boundary cell (a cell whose 8 corners don't all share a
     label), emit one dual vertex at the centroid of its crossed-edge
     points. For each crossed grid edge, emit one quad connecting the
     dual vertices of the 4 cells incident to that edge, oriented so the
     face normal points Label0 → Label1 per the VTK BoundaryLabels rule.

Vertices live in *training-grid index coordinates* (fractional voxel
indices). World-mm conversion is the caller's responsibility — same
convention as :class:`Mesh.points` documents.
"""

from __future__ import annotations

import numpy as np

from .values import Geometry, LabelSchema, Mesh


def surfacenets_logits(
    logits: np.ndarray,
    geometry: Geometry,
    schema: LabelSchema,
) -> Mesh:
    """Extract a SurfaceNets dual mesh directly from K-channel logits.

    Parameters
    ----------
    logits :
        ``(K, Z, Y, X)`` float array at training spacing. Any float dtype
        is accepted; computation is promoted to float32.
    geometry :
        The training-grid geometry. Attached to the returned mesh.
    schema :
        Label name lookup. ``boundary_labels`` are in this schema's space.

    Returns
    -------
    Mesh
        Vertices in training-grid index coords (``(N, 3)`` in (Z, Y, X)
        order), one quad per crossed *interior* grid edge, with VTK's
        ``(Label0, Label1)`` convention (background last; else sorted
        ascending; normal points Label0 → Label1).

    Notes
    -----
    Volume-boundary closure: grid edges on the outermost voxel layer
    have fewer than 4 incident cells, so this MVP does not emit quads for
    them. Objects that don't touch the volume boundary produce closed
    surfaces; objects clipped by the volume have an open "ring" at the
    clip. Pad the input with a background border if you need closure.
    """
    if logits.ndim != 4:
        raise ValueError(f"logits must be 4-D (K, Z, Y, X); got ndim={logits.ndim}")
    if tuple(int(s) for s in logits.shape[1:]) != geometry.shape_zyx:
        raise ValueError(
            f"logits spatial shape {logits.shape[1:]} != geometry.shape_zyx "
            f"{geometry.shape_zyx}"
        )

    _, Z, Y, X = logits.shape
    if Z < 2 or Y < 2 or X < 2:
        return Mesh.empty(geometry, schema)

    logits = np.ascontiguousarray(logits, dtype=np.float32)
    labels = np.argmax(logits, axis=0).astype(np.int32)

    x_crossed, x_t = _edge_crossings(logits, labels, axis=2)
    y_crossed, y_t = _edge_crossings(logits, labels, axis=1)
    z_crossed, z_t = _edge_crossings(logits, labels, axis=0)

    cell_to_vertex, points = _cell_dual_vertices(
        x_crossed, x_t, y_crossed, y_t, z_crossed, z_t,
    )

    quads, boundary_labels = _emit_quads(
        x_crossed, y_crossed, z_crossed, cell_to_vertex, labels,
    )

    return Mesh(
        points=points,
        quads=quads.astype(np.int32, copy=False),
        boundary_labels=boundary_labels.astype(np.int32, copy=False),
        geometry=geometry,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Edge crossings (one axis at a time)
# ---------------------------------------------------------------------------


def _edge_crossings(
    logits: np.ndarray,
    labels: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For grid edges along ``axis`` (0=Z, 1=Y, 2=X), return:

      * ``crossed`` — bool array (one element shorter than ``labels`` on
        ``axis``), True where the endpoint dominant labels differ;
      * ``t`` — float32 array of the same shape, with the sub-voxel
        position in [0, 1] where ``logit_i - logit_j`` crosses zero (i at
        endpoint a, j at endpoint b).

    Computation is vectorized over the whole field; the t value is set to
    0 where the edge is not crossed (harmless because the cell sweep
    weighs t by ``crossed``).
    """
    sl_a = [slice(None)] * 3
    sl_b = [slice(None)] * 3
    sl_a[axis] = slice(None, -1)
    sl_b[axis] = slice(1, None)
    L0 = labels[tuple(sl_a)]
    L1 = labels[tuple(sl_b)]
    crossed = (L0 != L1)

    sl_a_log = [slice(None)] * 4
    sl_b_log = [slice(None)] * 4
    sl_a_log[axis + 1] = slice(None, -1)
    sl_b_log[axis + 1] = slice(1, None)
    logits_a = logits[tuple(sl_a_log)]
    logits_b = logits[tuple(sl_b_log)]

    L0_idx = L0.astype(np.intp)[None]
    L1_idx = L1.astype(np.intp)[None]
    logit_L0_a = np.take_along_axis(logits_a, L0_idx, axis=0)[0]
    logit_L1_a = np.take_along_axis(logits_a, L1_idx, axis=0)[0]
    logit_L0_b = np.take_along_axis(logits_b, L0_idx, axis=0)[0]
    logit_L1_b = np.take_along_axis(logits_b, L1_idx, axis=0)[0]

    d0 = logit_L0_a - logit_L1_a   # ≥ 0 at endpoint a (L0 dominant there)
    d1 = logit_L0_b - logit_L1_b   # ≤ 0 at endpoint b (L1 dominant there)
    denom = d0 - d1
    # Degenerate (both d0 and d1 effectively zero) → fall back to midpoint;
    # at real-world logit magnitudes this branch is never taken.
    safe_denom = np.where(denom > 1e-30, denom, 1.0)
    t = np.where(denom > 1e-30, d0 / safe_denom, 0.5).astype(np.float32)
    t = np.where(crossed, t, np.float32(0.0))
    return crossed, t


# ---------------------------------------------------------------------------
# Dual vertex per boundary cell
# ---------------------------------------------------------------------------


def _cell_dual_vertices(
    x_crossed: np.ndarray, x_t: np.ndarray,
    y_crossed: np.ndarray, y_t: np.ndarray,
    z_crossed: np.ndarray, z_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each cell, dual vertex = centroid of crossed-edge crossings.

    Returns
    -------
    cell_to_vertex :
        ``(Zm1, Ym1, Xm1) int64``. ``-1`` for non-boundary cells; for
        boundary cells, the compact vertex index into ``points``.
    points :
        ``(N, 3) float32``. Position in training-grid index coords,
        (Z, Y, X) component order.
    """
    Z = x_crossed.shape[0]
    Y = x_crossed.shape[1]
    Xm1 = x_crossed.shape[2]
    Zm1 = Z - 1
    Ym1 = Y - 1

    sum_pos = np.zeros((Zm1, Ym1, Xm1, 3), dtype=np.float32)
    count = np.zeros((Zm1, Ym1, Xm1), dtype=np.int32)

    # 4 X-edges per cell, indexed by (dz, dy) ∈ {0,1}²; t runs along X.
    for dz in (0, 1):
        for dy in (0, 1):
            c = x_crossed[dz:Zm1 + dz, dy:Ym1 + dy, :Xm1].astype(np.float32)
            t = x_t[dz:Zm1 + dz, dy:Ym1 + dy, :Xm1]
            sum_pos[..., 0] += c * np.float32(dz)
            sum_pos[..., 1] += c * np.float32(dy)
            sum_pos[..., 2] += c * t
            count += c.astype(np.int32)

    # 4 Y-edges per cell, indexed by (dz, dx); t runs along Y.
    for dz in (0, 1):
        for dx in (0, 1):
            c = y_crossed[dz:Zm1 + dz, :Ym1, dx:Xm1 + dx].astype(np.float32)
            t = y_t[dz:Zm1 + dz, :Ym1, dx:Xm1 + dx]
            sum_pos[..., 0] += c * np.float32(dz)
            sum_pos[..., 1] += c * t
            sum_pos[..., 2] += c * np.float32(dx)
            count += c.astype(np.int32)

    # 4 Z-edges per cell, indexed by (dy, dx); t runs along Z.
    for dy in (0, 1):
        for dx in (0, 1):
            c = z_crossed[:Zm1, dy:Ym1 + dy, dx:Xm1 + dx].astype(np.float32)
            t = z_t[:Zm1, dy:Ym1 + dy, dx:Xm1 + dx]
            sum_pos[..., 0] += c * t
            sum_pos[..., 1] += c * np.float32(dy)
            sum_pos[..., 2] += c * np.float32(dx)
            count += c.astype(np.int32)

    is_boundary = count > 0
    safe_count = np.maximum(count, 1).astype(np.float32)
    local_pos = sum_pos / safe_count[..., None]

    # Cell (a, b, c)'s base corner is at voxel (a, b, c); add it to local.
    z_grid = np.arange(Zm1, dtype=np.float32)[:, None, None]
    y_grid = np.arange(Ym1, dtype=np.float32)[None, :, None]
    x_grid = np.arange(Xm1, dtype=np.float32)[None, None, :]

    vertex_pos = np.empty((Zm1, Ym1, Xm1, 3), dtype=np.float32)
    vertex_pos[..., 0] = local_pos[..., 0] + z_grid
    vertex_pos[..., 1] = local_pos[..., 1] + y_grid
    vertex_pos[..., 2] = local_pos[..., 2] + x_grid

    n_boundary = int(is_boundary.sum())
    cell_to_vertex = np.full((Zm1, Ym1, Xm1), -1, dtype=np.int64)
    cell_to_vertex[is_boundary] = np.arange(n_boundary, dtype=np.int64)

    points = vertex_pos[is_boundary]
    return cell_to_vertex, points


# ---------------------------------------------------------------------------
# Quad emission
# ---------------------------------------------------------------------------


def _emit_quads(
    x_crossed: np.ndarray,
    y_crossed: np.ndarray,
    z_crossed: np.ndarray,
    cell_to_vertex: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit one quad per crossed *interior* grid edge.

    An edge is "interior" iff all 4 cells incident to it exist within the
    volume (i.e., the edge isn't on the outermost voxel layer along the
    two axes orthogonal to the edge).
    """
    quad_lists: list[np.ndarray] = []
    label_lists: list[np.ndarray] = []
    Z, Y, X = labels.shape

    # ----- X-edges. Interior: z ∈ [1, Z-2], y ∈ [1, Y-2] -----
    # x_crossed shape (Z, Y, X-1). Incident cells around an X-edge form a
    # 2×2 square in the Y-Z plane:
    #   A = (z-1, y-1, x)   C = (z-1, y, x)
    #   B = (z,   y-1, x)   D = (z,   y, x)
    # Viewed from +X (with +Y right, +Z up): A bottom-left, C bottom-right,
    # D top-right, B top-left. CCW from +X → +X normal: [A, C, D, B].
    if Z >= 3 and Y >= 3 and X >= 2:
        _append_axis(
            interior=x_crossed[1:Z - 1, 1:Y - 1, :],
            i_lbl=labels[1:Z - 1, 1:Y - 1, :-1],
            j_lbl=labels[1:Z - 1, 1:Y - 1, 1:],
            v_A=cell_to_vertex[0:Z - 2, 0:Y - 2, :X - 1],
            v_B=cell_to_vertex[1:Z - 1, 0:Y - 2, :X - 1],
            v_C=cell_to_vertex[0:Z - 2, 1:Y - 1, :X - 1],
            v_D=cell_to_vertex[1:Z - 1, 1:Y - 1, :X - 1],
            pos_winding=("A", "C", "D", "B"),
            quad_lists=quad_lists, label_lists=label_lists,
        )

    # ----- Y-edges. Interior: z ∈ [1, Z-2], x ∈ [1, X-2] -----
    # y_crossed shape (Z, Y-1, X). Incident cells:
    #   A = (z-1, y, x-1)   C = (z-1, y, x)
    #   B = (z,   y, x-1)   D = (z,   y, x)
    # Viewed from +Y the natural CCW order is [A, B, D, C] (+Y is the
    # middle axis of the right-handed XYZ frame, so the parity flips).
    if Z >= 3 and Y >= 2 and X >= 3:
        _append_axis(
            interior=y_crossed[1:Z - 1, :, 1:X - 1],
            i_lbl=labels[1:Z - 1, :-1, 1:X - 1],
            j_lbl=labels[1:Z - 1, 1:, 1:X - 1],
            v_A=cell_to_vertex[0:Z - 2, :Y - 1, 0:X - 2],
            v_B=cell_to_vertex[1:Z - 1, :Y - 1, 0:X - 2],
            v_C=cell_to_vertex[0:Z - 2, :Y - 1, 1:X - 1],
            v_D=cell_to_vertex[1:Z - 1, :Y - 1, 1:X - 1],
            pos_winding=("A", "B", "D", "C"),
            quad_lists=quad_lists, label_lists=label_lists,
        )

    # ----- Z-edges. Interior: y ∈ [1, Y-2], x ∈ [1, X-2] -----
    # z_crossed shape (Z-1, Y, X). Incident cells:
    #   A = (z, y-1, x-1)   C = (z, y-1, x)
    #   B = (z, y,   x-1)   D = (z, y,   x)
    if Z >= 2 and Y >= 3 and X >= 3:
        _append_axis(
            interior=z_crossed[:, 1:Y - 1, 1:X - 1],
            i_lbl=labels[:-1, 1:Y - 1, 1:X - 1],
            j_lbl=labels[1:, 1:Y - 1, 1:X - 1],
            v_A=cell_to_vertex[:Z - 1, 0:Y - 2, 0:X - 2],
            v_B=cell_to_vertex[:Z - 1, 1:Y - 1, 0:X - 2],
            v_C=cell_to_vertex[:Z - 1, 0:Y - 2, 1:X - 1],
            v_D=cell_to_vertex[:Z - 1, 1:Y - 1, 1:X - 1],
            pos_winding=("A", "C", "D", "B"),
            quad_lists=quad_lists, label_lists=label_lists,
        )

    if not quad_lists:
        return np.zeros((0, 4), dtype=np.int64), np.zeros((0, 2), dtype=np.int32)
    return np.concatenate(quad_lists, axis=0), np.concatenate(label_lists, axis=0)


def _append_axis(
    *,
    interior: np.ndarray,
    i_lbl: np.ndarray,
    j_lbl: np.ndarray,
    v_A: np.ndarray, v_B: np.ndarray, v_C: np.ndarray, v_D: np.ndarray,
    pos_winding: tuple[str, str, str, str],
    quad_lists: list[np.ndarray],
    label_lists: list[np.ndarray],
) -> None:
    """Mask interior crossings, assemble per-edge quads + BoundaryLabels.

    ``pos_winding`` is the four-letter vertex sequence that yields the
    +axis-direction face normal. The negative-direction winding is
    obtained by swapping winding slots 1 and 3 (same dual sense reflected).
    """
    if not interior.any():
        return
    mask = interior
    v_A_f = v_A[mask]
    v_B_f = v_B[mask]
    v_C_f = v_C[mask]
    v_D_f = v_D[mask]
    i_f = i_lbl[mask]
    j_f = j_lbl[mask]

    # VTK BoundaryLabels rule:
    #   - if background (0) is one of the pair, it goes in slot 1
    #   - else, sort ascending
    # Flip the natural winding when the rule puts j into slot 0 (so the
    # quad normal still points Label0 → Label1).
    swap_for_zero = (i_f == 0) & (j_f != 0)
    swap_for_sort = (i_f != 0) & (j_f != 0) & (j_f < i_f)
    flip = swap_for_zero | swap_for_sort

    label0 = np.where(flip, j_f, i_f)
    label1 = np.where(flip, i_f, j_f)

    lut = {"A": v_A_f, "B": v_B_f, "C": v_C_f, "D": v_D_f}
    q_pos = np.stack([lut[name] for name in pos_winding], axis=-1)
    neg_winding = (pos_winding[0], pos_winding[3], pos_winding[2], pos_winding[1])
    q_neg = np.stack([lut[name] for name in neg_winding], axis=-1)

    q = np.where(flip[:, None], q_neg, q_pos)
    quad_lists.append(q)
    label_lists.append(np.stack([label0, label1], axis=-1))


__all__ = ["surfacenets_logits"]
