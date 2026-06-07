"""Logit-based surface extraction.

Reads a K-channel logit volume at training spacing and emits a SurfaceNets
dual mesh whose vertex positions come from edge-crossing interpolation in
the *continuous* logit field — not from a discretized label map.

Algorithm in three lines:

  1. Argmax per voxel → label per voxel.
  2. For each grid edge whose endpoint labels differ, linearly interpolate
     ``logit_i - logit_j`` (the two dominant labels) to find t ∈ [0, 1]
     where it crosses zero.
  3. For each boundary cell, compute the connected components of crossed
     cell-edges (face-based connectivity, smaller-label-through-middle
     saddle disambiguation). Emit one dual vertex per component, placed
     at the centroid of that component's crossing points. For each
     crossed grid edge, emit one quad connecting the dual vertices of
     the 4 cells incident to that edge — selecting the component vertex
     that the corresponding cell-edge belongs to in each cell.

The component split is what fixes the non-manifold edges that a single-
vertex-per-cell rule generates at saddle cells (8 corners arranged so the
boundary surface inside the cell is two disjoint patches). VTK's
``vtkSurfaceNets3D`` does the same thing with its case-table machinery;
we do it via per-cell face-based connectivity with a saddle-resolution
rule that depends only on the face's labels (so two cells sharing a face
always agree about how to connect crossings on that face).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .values import Geometry, LabelSchema, Mesh


def surfacenets_logits(
    logits: np.ndarray,
    geometry: Geometry,
    schema: LabelSchema,
    *,
    project_to_surface: bool = False,
    emit_normals: bool = False,
    confidence_threshold: float = 0.0,
    drop_components_below_mm3: float = 0.0,
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
    project_to_surface :
        If True, after centroid placement do one Newton step toward the
        boundary level set ``logit_i - logit_j = 0`` for each *binary*
        cell vertex. Pushes the vertex onto the actual decision surface
        rather than the centroid of its crossings. Multi-component
        cells are left at the cell center (so the no-gap property at
        triple junctions is preserved). Off by default.
    emit_normals :
        If True, compute per-vertex normals from the logit-field
        gradient ``∇(logit_i - logit_j)`` at each vertex position
        (where ``(i, j)`` is that vertex's label pair). Independent of
        mesh discretization; usually visibly smoother than VTK's
        averaged-face-normal computation. Off by default.
    confidence_threshold :
        Logit-margin floor (in raw logit units) for treating an argmax
        decision as confident. Voxels whose top-1 vs top-2 logit margin
        falls below this AND whose 6-connected neighbors *unanimously*
        carry the same other label are relabeled to that neighbor label.
        Targeted fix for sub-Nyquist single-voxel "blob" artifacts —
        argmax flips driven by logit noise that the discretization
        amplifies into octahedron spikes on the mesh. 0.0 (default,
        no suppression) preserves the v8 behavior; values in the
        0.5–1.5 range typically clean up the noise floor without
        suppressing real thin features.
    drop_components_below_mm3 :
        Drop connected components of any label whose physical volume
        is below this threshold (in mm³). 26-connected, multi-label
        aware (same logic as
        :func:`postprocessing.remove_small_components`). Catches noise
        clusters too large for the confidence rule (which only handles
        fully isolated voxels). Requires the ``cc3d`` package; raises
        ``ImportError`` if invoked without it. Default ``0.0`` (off).
        TotalSegmentator's ``--remove_small_blobs`` uses ``200.0``.

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
    have fewer than 4 incident cells, so this implementation does not
    emit quads for them. Objects that don't touch the volume boundary
    produce closed surfaces; objects clipped by the volume have an open
    "ring" at the clip. Pad the input with a background border if you
    need closure.
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
    if confidence_threshold > 0.0:
        labels = _suppress_low_confidence_blobs(
            labels, logits, float(confidence_threshold),
        )
    if drop_components_below_mm3 > 0.0:
        from .postprocessing import remove_small_components
        labels = remove_small_components(
            labels, geometry.spacing_zyx,
            min_volume_mm3=float(drop_components_below_mm3),
            in_place=False,
        ).astype(np.int32, copy=False)

    x_crossed, x_t = _edge_crossings(logits, labels, axis=2)
    y_crossed, y_t = _edge_crossings(logits, labels, axis=1)
    z_crossed, z_t = _edge_crossings(logits, labels, axis=0)

    edge_comp, n_comp, comp_pairs = _cell_components(labels, logits)

    cell_to_vertex, points = _cell_dual_vertices(
        edge_comp, n_comp,
        x_t, y_t, z_t,
    )

    normals: np.ndarray | None = None
    if project_to_surface or emit_normals:
        points, normals = _gradient_refine(
            points, cell_to_vertex, n_comp, comp_pairs, logits,
            project=project_to_surface,
            emit_normals=emit_normals,
        )

    quads, boundary_labels = _emit_quads(
        x_crossed, y_crossed, z_crossed, cell_to_vertex, edge_comp, labels,
    )

    return Mesh(
        points=points,
        quads=quads.astype(np.int32, copy=False),
        boundary_labels=boundary_labels.astype(np.int32, copy=False),
        geometry=geometry,
        schema=schema,
        normals=normals,
    )


# ---------------------------------------------------------------------------
# Low-confidence single-voxel blob suppression
# ---------------------------------------------------------------------------


def _suppress_low_confidence_blobs(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Targeted fix for sub-Nyquist argmax-flip artifacts.

    A voxel V is *suppressed* (relabeled to the dominant neighbor's
    label) when both:

      1. ``margin(V) = top1_logit(V) − top2_logit(V) < threshold`` —
         the argmax decision at V is uncertain.
      2. **None** of V's 6 axis-aligned neighbors share V's label —
         V is a topologically isolated voxel.

    The "fully isolated" criterion is the safe one: it only catches
    voxels whose label has zero connectivity to its neighborhood
    (the classic 1-voxel argmax-flip blob), and never breaks
    elongated structures. A 1-voxel-wide rib has each voxel
    connected to one or two same-label neighbors along its length,
    so its voxels are not isolated and the rule leaves them alone.
    Trade-off: 2+ voxel noise clusters (where each voxel has ≥ 1
    same-label neighbor) are *not* suppressed by this rule.

    Relabel target: the majority label among V's 6 neighbors. If
    multiple labels tie, the smallest-numbered tied label is used —
    a deterministic choice that defaults to background (label 0)
    when bg is among the ties (the practically common case).
    """
    Z, Y, X = labels.shape
    if Z < 3 or Y < 3 or X < 3:
        return labels

    # Per-voxel logit margin: top-1 vs top-2 via mask-and-remax.
    winner_idx = labels.astype(np.intp)[None]
    winner_logit = np.take_along_axis(logits, winner_idx, axis=0)[0]
    masked = logits.copy()
    np.put_along_axis(masked, winner_idx, np.float32(-np.inf), axis=0)
    second_logit = masked.max(axis=0)
    del masked
    margin = winner_logit - second_logit

    # Interior block + 6 axis-aligned neighbors.
    center = labels[1:-1, 1:-1, 1:-1]
    nbrs = np.stack([
        labels[ :-2, 1:-1, 1:-1],
        labels[2:,   1:-1, 1:-1],
        labels[1:-1,  :-2, 1:-1],
        labels[1:-1, 2:,   1:-1],
        labels[1:-1, 1:-1,  :-2],
        labels[1:-1, 1:-1, 2:  ],
    ], axis=-1)                                     # (Zm2, Ym2, Xm2, 6)

    # Count how many neighbors share the center's label.
    n_same = (nbrs == center[..., None]).sum(axis=-1).astype(np.int8)
    low_confidence = margin[1:-1, 1:-1, 1:-1] < np.float32(threshold)
    is_isolated = (n_same == 0)

    suppress = low_confidence & is_isolated
    if not suppress.any():
        return labels

    # Majority label among the 6 neighbors. Use mode-via-sort:
    # sort the 6 neighbors per voxel, then find the longest run.
    nbrs_sorted = np.sort(nbrs, axis=-1)              # (Zm2, Ym2, Xm2, 6)
    # Run-length encoding: a new run starts wherever the sorted value
    # changes vs its predecessor; treat slot 0 as always a new run.
    new_run = np.concatenate([
        np.ones(nbrs_sorted.shape[:-1] + (1,), dtype=np.int8),
        (nbrs_sorted[..., 1:] != nbrs_sorted[..., :-1]).astype(np.int8),
    ], axis=-1)
    run_id = np.cumsum(new_run, axis=-1) - 1          # 0..k-1 per voxel
    # Count voxels per run (only 6 possible slot ids).
    counts = np.zeros(nbrs_sorted.shape[:-1] + (6,), dtype=np.int8)
    for slot in range(6):
        counts[..., slot] = (run_id == slot).sum(axis=-1)
    best_slot = counts.argmax(axis=-1)                # (Zm2, Ym2, Xm2)
    # The majority label is at sorted index = first occurrence of best_slot.
    # By construction that's the first slot whose run_id == best_slot;
    # find it via argmax-of-(run_id == best_slot).
    is_best = (run_id == best_slot[..., None])
    first_idx = is_best.argmax(axis=-1)               # (Zm2, Ym2, Xm2)
    majority = np.take_along_axis(
        nbrs_sorted, first_idx[..., None], axis=-1
    )[..., 0]

    new_labels = labels.copy()
    new_labels[1:-1, 1:-1, 1:-1] = np.where(suppress, majority, center)
    return new_labels


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

    d0 = logit_L0_a - logit_L1_a
    d1 = logit_L0_b - logit_L1_b
    denom = d0 - d1
    safe_denom = np.where(denom > 1e-30, denom, 1.0)
    t = np.where(denom > 1e-30, d0 / safe_denom, 0.5).astype(np.float32)
    t = np.where(crossed, t, np.float32(0.0))
    return crossed, t


# ---------------------------------------------------------------------------
# Cell connectivity: components of crossed cell-edges
# ---------------------------------------------------------------------------
#
# Corners of a cell are indexed 0..7 from (dz, dy, dx) ∈ {0,1}³ as
# ``index = dz*4 + dy*2 + dx``.
#
# Edges 0..11:
#   0..3   X-edges, indexed by (dz, dy):  (0,0), (0,1), (1,0), (1,1)
#   4..7   Y-edges, indexed by (dz, dx):  (0,0), (0,1), (1,0), (1,1)
#   8..11  Z-edges, indexed by (dy, dx):  (0,0), (0,1), (1,0), (1,1)


_EDGE_CORNERS = (
    (0, 1), (2, 3), (4, 5), (6, 7),     # X-edges 0..3
    (0, 2), (1, 3), (4, 6), (5, 7),     # Y-edges 4..7
    (0, 4), (1, 5), (2, 6), (3, 7),     # Z-edges 8..11
)


# For each of the 6 faces: cyclic ordering of (4 corners) and the (4 edges
# between consecutive corners). ``face_edges[k]`` is the edge between
# ``face_corners[k]`` and ``face_corners[(k + 1) % 4]``.
_FACES = (
    ((0, 1, 3, 2), (0, 5, 1, 4)),       # z=0 bottom
    ((4, 5, 7, 6), (2, 7, 3, 6)),       # z=1 top
    ((0, 1, 5, 4), (0, 9, 2, 8)),       # y=0 back
    ((2, 3, 7, 6), (1, 11, 3, 10)),     # y=1 front
    ((0, 2, 6, 4), (4, 10, 6, 8)),      # x=0 left
    ((1, 3, 7, 5), (5, 11, 7, 9)),      # x=1 right
)


@lru_cache(maxsize=8192)
def _cell_case(corner_labels: tuple, saddle_flips: int = 0) -> tuple:
    """For one cell configuration, return ``(n_components, edge_to_component)``.

    ``corner_labels`` is an 8-tuple of integer labels at the 8 cell
    corners (in canonical 0..7 order). ``saddle_flips`` is a 6-bit int
    where bit ``f`` set means "flip the saddle rule on face ``f``" —
    used by the logit-magnitude asymptotic decider, computed by the
    caller before this function runs. Returns a 12-tuple where element
    ``e`` is ``-1`` if cell-edge ``e`` is not crossed, else an integer
    in ``[0, n_components)`` identifying which boundary patch in the
    cell that crossing belongs to.

    Connectivity rule
    -----------------
    Two crossed edges are in the same component iff:

      1. They cross between the **same label pair**, *and*
      2. They are connected through some face of the cell (per-face
         connectivity below).

    Per-face connectivity (within a single label pair):
      * 2 same-pair crossings: connected.
      * 3 same-pair crossings: all connected.
      * 4 same-pair crossings: the only configuration that produces this
        is a 2-color saddle; the boundary is two segments. We default
        to "smaller-label-through-middle" (depends only on the face's
        labels, so two cells sharing the face always agree). When the
        caller's asymptotic decider says the logit field disagrees with
        the label-based rule, the corresponding bit in ``saddle_flips``
        is set and we use the flipped configuration instead.

    Cached: a real volume has at most a few thousand distinct cell
    configurations even when it has millions of cells. Adding the
    saddle_flips bitfield multiplies the cache space by up to 64 per
    label config, but in practice most configs have 0 saddle faces.
    """
    edge_pair: list[tuple] = [None] * 12  # type: ignore[list-item]
    crossed = [False] * 12
    for e, (a, b) in enumerate(_EDGE_CORNERS):
        la, lb = corner_labels[a], corner_labels[b]
        if la != lb:
            crossed[e] = True
            edge_pair[e] = (la, lb) if la < lb else (lb, la)
    if not any(crossed):
        return (0, (-1,) * 12, ())

    # Union-find on 12 edges.
    parent = list(range(12))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for face_idx, (face_corners, face_edges) in enumerate(_FACES):
        face_crossed = [e for e in face_edges if crossed[e]]
        if len(face_crossed) <= 1:
            continue

        # Group face crossings by label pair; connect only within a group.
        groups: dict[tuple, list[int]] = {}
        for e in face_crossed:
            groups.setdefault(edge_pair[e], []).append(e)

        for pair, es in groups.items():
            if len(es) <= 1:
                continue
            if len(es) <= 3:
                # Chain-connect — works for 2 (single union) and 3
                # (triangle = same component).
                for k in range(len(es) - 1):
                    union(es[k], es[k + 1])
                continue
            # len == 4: only possible if all 4 face edges cross between
            # the same pair, i.e. a 2-color saddle.
            L = tuple(corner_labels[c] for c in face_corners)
            if L[0] == L[2] and L[1] == L[3] and L[0] != L[1]:
                # Default rule: smaller label through the face middle.
                # If the asymptotic decider (encoded in saddle_flips)
                # contradicts the label rule for this face, flip.
                flip = bool((saddle_flips >> face_idx) & 1)
                smaller_through = (L[0] < L[1]) ^ flip
                if smaller_through:
                    union(face_edges[0], face_edges[1])
                    union(face_edges[2], face_edges[3])
                else:
                    union(face_edges[0], face_edges[3])
                    union(face_edges[1], face_edges[2])
            else:
                # Defensive: not a 2-color saddle (shouldn't reach here);
                # fall back to single component.
                for k in range(3):
                    union(es[k], es[k + 1])

    roots = sorted({find(e) for e in range(12) if crossed[e]})
    root_to_comp = {r: i for i, r in enumerate(roots)}
    edge_to_comp = tuple(
        root_to_comp[find(e)] if crossed[e] else -1
        for e in range(12)
    )
    # Per-component label pair — pick any edge in the component, take
    # its sorted (label_a, label_b). All edges in a component share
    # the same pair by construction.
    comp_pairs: list[tuple] = [None] * len(roots)   # type: ignore[list-item]
    for e in range(12):
        if not crossed[e]:
            continue
        k = root_to_comp[find(e)]
        if comp_pairs[k] is None:
            a, b = _EDGE_CORNERS[e]
            la, lb = corner_labels[a], corner_labels[b]
            comp_pairs[k] = (la, lb) if la < lb else (lb, la)
    return (len(roots), edge_to_comp, tuple(comp_pairs))


def _cell_components(
    labels: np.ndarray, logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cell component info for every cell in the volume.

    Returns
    -------
    edge_components :
        ``(Zm1, Ym1, Xm1, 12) int8`` — component ID per cell-edge, or
        ``-1`` for uncrossed edges (and for interior cells with no
        crossings at all).
    n_components :
        ``(Zm1, Ym1, Xm1) int8`` — number of components in each cell
        (0 for interior cells).

    Strategy:
      1. Stack the 8 corner labels per cell into ``(Zm1, Ym1, Xm1, 8)``.
      2. Compute the per-cell saddle-flip bitfield from the logit
         asymptotic decider (vectorized).
      3. Find unique ``(labels, saddle_flips)`` configurations across
         boundary cells, run :func:`_cell_case` once per unique config,
         scatter the result back.

    Step 3 keeps the Python work at "unique configs" rather than "all
    cells" even with the saddle-flip bitfield added — most label configs
    have 0 saddle faces, so the saddle bits only multiply cache space by
    a small factor.
    """
    # Stack 8 corner labels per cell.
    c000 = labels[:-1, :-1, :-1]
    c001 = labels[:-1, :-1, 1:]
    c010 = labels[:-1, 1:, :-1]
    c011 = labels[:-1, 1:, 1:]
    c100 = labels[1:, :-1, :-1]
    c101 = labels[1:, :-1, 1:]
    c110 = labels[1:, 1:, :-1]
    c111 = labels[1:, 1:, 1:]
    corners = np.stack(
        [c000, c001, c010, c011, c100, c101, c110, c111], axis=-1
    )

    Zm1, Ym1, Xm1 = corners.shape[:3]
    edge_components = np.full((Zm1, Ym1, Xm1, 12), -1, dtype=np.int8)
    n_components = np.zeros((Zm1, Ym1, Xm1), dtype=np.int8)

    # Boundary mask: at least one corner differs from corner-0.
    ref = corners[..., :1]
    is_boundary = (corners != ref).any(axis=-1)
    if not is_boundary.any():
        comp_pairs = np.zeros((Zm1, Ym1, Xm1, 1, 2), dtype=np.int32)
        return edge_components, n_components, comp_pairs

    # Saddle-flip bitfield (logit asymptotic decider).
    saddle_flips = _compute_saddle_flips(corners, logits, is_boundary)

    # Unique (label_config, saddle_flips) configurations.
    flat_corners = corners.reshape(-1, 8)
    flat_saddle = saddle_flips.reshape(-1)
    flat_boundary = is_boundary.reshape(-1)
    bc = flat_corners[flat_boundary]                       # (N_b, 8)
    bs = flat_saddle[flat_boundary]                        # (N_b,)
    combined = np.concatenate([bc, bs[:, None]], axis=1)   # (N_b, 9)
    unique_cfgs, inverse = np.unique(combined, axis=0, return_inverse=True)

    nb = unique_cfgs.shape[0]
    n_per_cfg = np.zeros(nb, dtype=np.int8)
    edge_per_cfg = np.full((nb, 12), -1, dtype=np.int8)
    # First pass: compute components per unique config; track the global
    # maximum component count so the per-cell pair array can be sized to
    # match what ``_cell_dual_vertices`` will allocate.
    case_results = []
    for i in range(nb):
        cfg = tuple(int(v) for v in unique_cfgs[i, :8])
        saddle = int(unique_cfgs[i, 8])
        n, et, pairs = _cell_case(cfg, saddle)
        n_per_cfg[i] = n
        edge_per_cfg[i] = et
        case_results.append(pairs)
    max_slots = max(1, int(n_per_cfg.max()))
    pairs_per_cfg = np.zeros((nb, max_slots, 2), dtype=np.int32)
    for i, pairs in enumerate(case_results):
        for k, (a, b) in enumerate(pairs):
            pairs_per_cfg[i, k, 0] = a
            pairs_per_cfg[i, k, 1] = b

    flat_n = np.zeros(flat_corners.shape[0], dtype=np.int8)
    flat_edge = np.full((flat_corners.shape[0], 12), -1, dtype=np.int8)
    flat_pairs = np.zeros(
        (flat_corners.shape[0], max_slots, 2), dtype=np.int32
    )
    flat_n[flat_boundary] = n_per_cfg[inverse]
    flat_edge[flat_boundary] = edge_per_cfg[inverse]
    flat_pairs[flat_boundary] = pairs_per_cfg[inverse]

    n_components = flat_n.reshape(Zm1, Ym1, Xm1)
    edge_components = flat_edge.reshape(Zm1, Ym1, Xm1, 12)
    comp_pairs = flat_pairs.reshape(Zm1, Ym1, Xm1, max_slots, 2)
    return edge_components, n_components, comp_pairs


def _compute_saddle_flips(
    corners: np.ndarray, logits: np.ndarray, boundary_mask: np.ndarray,
) -> np.ndarray:
    """Per-cell 6-bit bitfield; bit ``f`` set means the asymptotic
    decider disagrees with the label-rule saddle resolution on face ``f``.

    The asymptotic decider for a 2-color saddle face with labels A, B:
    take the bilinear value of ``logit_A - logit_B`` at the face center.
    By bilinear interpolation that's ``(1/4) * sum over 4 face corners
    of (logit_A[corner] - logit_B[corner])``; sign is what matters.

      * sum > 0 → A is connected through the face's middle (interior
        of the cell), so smaller-label connectivity holds iff A < B.
      * sum < 0 → B through the middle; smaller-label rule holds iff
        B < A, i.e. A > B.

    The label-only "smaller through middle" rule says A is through
    middle iff A < B. If the asymptotic and label rules agree, no flip.
    If they disagree, flip the rule on that face.

    Returns ``(Zm1, Ym1, Xm1) uint8`` (only the low 6 bits are used).
    """
    Zm1, Ym1, Xm1 = corners.shape[:3]
    saddle_flips = np.zeros((Zm1, Ym1, Xm1), dtype=np.uint8)

    for face_idx, (face_corners_cyclic, _) in enumerate(_FACES):
        c0_lbl = corners[..., face_corners_cyclic[0]]
        c1_lbl = corners[..., face_corners_cyclic[1]]
        c2_lbl = corners[..., face_corners_cyclic[2]]
        c3_lbl = corners[..., face_corners_cyclic[3]]

        # 2-color saddle: (A, B, A, B) around the face cycle.
        is_saddle = (
            (c0_lbl == c2_lbl) & (c1_lbl == c3_lbl) & (c0_lbl != c1_lbl)
            & boundary_mask
        )
        if not is_saddle.any():
            continue

        # Saddle cells are RARE (≪ 1% of the volume in real TS data);
        # gather logits only at those cells via flat fancy indexing
        # instead of slicing the whole K-channel volume per face corner.
        sz, sy, sx = np.nonzero(is_saddle)
        A_at_cell = c0_lbl[sz, sy, sx].astype(np.intp)
        B_at_cell = c1_lbl[sz, sy, sx].astype(np.intp)

        sum_diff = np.zeros(sz.shape[0], dtype=np.float32)
        for corner_idx in face_corners_cyclic:
            dz = (corner_idx >> 2) & 1
            dy = (corner_idx >> 1) & 1
            dx = corner_idx & 1
            zs, ys, xs = sz + dz, sy + dy, sx + dx
            sum_diff += logits[A_at_cell, zs, ys, xs] - logits[B_at_cell, zs, ys, xs]

        # Label rule: A through middle iff A < B.
        # Asymptotic:  A through middle iff sum_diff > 0.
        label_says_A_through = c0_lbl[sz, sy, sx] < c1_lbl[sz, sy, sx]
        asymp_says_A_through = sum_diff > 0
        disagree = label_says_A_through != asymp_says_A_through
        if disagree.any():
            fz, fy, fx = sz[disagree], sy[disagree], sx[disagree]
            saddle_flips[fz, fy, fx] |= np.uint8(1 << face_idx)
    return saddle_flips


# ---------------------------------------------------------------------------
# Dual vertices: one per (cell, component); all coincident in position
# ---------------------------------------------------------------------------


def _cell_dual_vertices(
    edge_comp: np.ndarray,                              # (Zm1, Ym1, Xm1, 12)
    n_comp: np.ndarray,                                 # (Zm1, Ym1, Xm1)
    x_t: np.ndarray, y_t: np.ndarray, z_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit one vertex per (cell, component); return the lookup and points.

    All components within a cell share the same *position* — the centroid
    of all crossings in the cell — but each gets its own vertex ID. This
    makes label-pair surfaces visually meet at triple lines while keeping
    the topology multi-material.

    Accumulation is per-cell (the same vectorized pattern as the v1
    single-vertex implementation); the multi-component aspect is just
    "how many vertex IDs to allocate per cell" and "broadcast the same
    position to all slots."

    Returns
    -------
    cell_to_vertex :
        ``(Zm1, Ym1, Xm1, MAX_COMP) int64`` — global vertex ID for each
        cell-component slot; ``-1`` where the slot is unused.
    points :
        ``(N_total, 3) float32`` — positions in training-grid index
        coords, in (Z, Y, X) component order.
    """
    Zm1, Ym1, Xm1 = n_comp.shape
    max_comp = int(n_comp.max()) if n_comp.size else 0
    if max_comp == 0:
        return (
            np.full((Zm1, Ym1, Xm1, 1), -1, dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
        )

    # Per-cell crossing-centroid accumulation (vectorized over the whole
    # volume — same shape as the v1 implementation, no per-component axis).
    sum_pos = np.zeros((Zm1, Ym1, Xm1, 3), dtype=np.float32)
    count = np.zeros((Zm1, Ym1, Xm1), dtype=np.int32)

    # 4 X-edges per cell, indexed by (dz, dy); t runs along X.
    for dz in (0, 1):
        for dy in (0, 1):
            c = x_t[dz:Zm1 + dz, dy:Ym1 + dy, :Xm1]                    # 0 if uncrossed
            crossed = (
                edge_comp[..., 0 + 2 * dz + dy] >= 0
            ).astype(np.float32)
            sum_pos[..., 0] += crossed * np.float32(dz)
            sum_pos[..., 1] += crossed * np.float32(dy)
            sum_pos[..., 2] += crossed * c
            count += crossed.astype(np.int32)

    # 4 Y-edges per cell, indexed by (dz, dx); t runs along Y.
    for dz in (0, 1):
        for dx in (0, 1):
            c = y_t[dz:Zm1 + dz, :Ym1, dx:Xm1 + dx]
            crossed = (
                edge_comp[..., 4 + 2 * dz + dx] >= 0
            ).astype(np.float32)
            sum_pos[..., 0] += crossed * np.float32(dz)
            sum_pos[..., 1] += crossed * c
            sum_pos[..., 2] += crossed * np.float32(dx)
            count += crossed.astype(np.int32)

    # 4 Z-edges per cell, indexed by (dy, dx); t runs along Z.
    for dy in (0, 1):
        for dx in (0, 1):
            c = z_t[:Zm1, dy:Ym1 + dy, dx:Xm1 + dx]
            crossed = (
                edge_comp[..., 8 + 2 * dy + dx] >= 0
            ).astype(np.float32)
            sum_pos[..., 0] += crossed * c
            sum_pos[..., 1] += crossed * np.float32(dy)
            sum_pos[..., 2] += crossed * np.float32(dx)
            count += crossed.astype(np.int32)

    safe_count = np.maximum(count, 1).astype(np.float32)
    local_pos = sum_pos / safe_count[..., None]                          # (Zm1, Ym1, Xm1, 3)

    # For multi-component cells (triple junctions, etc.) the centroid-of-
    # crossings is biased by the asymmetric distribution of crossings
    # across label pairs, which makes adjacent multi-component cells' verts
    # zig-zag along a triple line. Replacing the centroid with the cell
    # center for these cells turns the triple line into a straight-line
    # path through fixed grid positions — visually smooth, no bias.
    # Binary cells (the vast majority) keep their crossings centroid.
    multi = n_comp > 1
    if multi.any():
        local_pos[multi] = np.float32(0.5)

    # Add cell base position.
    z_grid = np.arange(Zm1, dtype=np.float32)[:, None, None]
    y_grid = np.arange(Ym1, dtype=np.float32)[None, :, None]
    x_grid = np.arange(Xm1, dtype=np.float32)[None, None, :]
    vertex_pos = np.empty((Zm1, Ym1, Xm1, 3), dtype=np.float32)
    vertex_pos[..., 0] = local_pos[..., 0] + z_grid
    vertex_pos[..., 1] = local_pos[..., 1] + y_grid
    vertex_pos[..., 2] = local_pos[..., 2] + x_grid

    # Allocate IDs to component slots used by each cell. The "used" mask
    # is ``slot < n_comp[..., None]``: cell (a, b, c) uses slots
    # ``0 .. n_comp[a, b, c] - 1``.
    slot = np.arange(max_comp, dtype=np.int8)[None, None, None, :]
    used = slot < n_comp[..., None]                                      # (Zm1, Ym1, Xm1, K)
    n_total = int(used.sum())
    cell_to_vertex = np.full((Zm1, Ym1, Xm1, max_comp), -1, dtype=np.int64)
    cell_to_vertex[used] = np.arange(n_total, dtype=np.int64)

    # Points: broadcast vertex_pos to every used slot. (Multiple slots in
    # the same cell read the same position — that's the coincidence rule.)
    pos_broadcast = np.broadcast_to(
        vertex_pos[:, :, :, None, :],
        (Zm1, Ym1, Xm1, max_comp, 3),
    )
    points = np.ascontiguousarray(pos_broadcast[used])                   # (N_total, 3)
    return cell_to_vertex, points


# ---------------------------------------------------------------------------
# Quad emission with per-cell component routing
# ---------------------------------------------------------------------------


def _emit_quads(
    x_crossed: np.ndarray,
    y_crossed: np.ndarray,
    z_crossed: np.ndarray,
    cell_to_vertex: np.ndarray,        # (Zm1, Ym1, Xm1, MAX_COMP)
    edge_comp: np.ndarray,             # (Zm1, Ym1, Xm1, 12)
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit one quad per crossed *interior* grid edge.

    For each grid edge, the 4 incident cells each have a *cell-edge index*
    in 0..11 that corresponds to the same grid edge. We look up the
    component for that cell-edge index, and use the matching vertex from
    that cell's ``cell_to_vertex`` slot.
    """
    quad_lists: list[np.ndarray] = []
    label_lists: list[np.ndarray] = []
    Z, Y, X = labels.shape

    # ----- X-edges. Interior z ∈ [1, Z-2], y ∈ [1, Y-2] -----
    # 4 incident cells with their cell-edge index for this grid X-edge:
    #   A=(z-1, y-1, x): cell-edge 3 (X-edge at top, dy=1)
    #   B=(z,   y-1, x): cell-edge 1 (X-edge at bottom, dy=1)
    #   C=(z-1, y,   x): cell-edge 2 (X-edge at top, dy=0)
    #   D=(z,   y,   x): cell-edge 0 (X-edge at bottom, dy=0)
    if Z >= 3 and Y >= 3 and X >= 2:
        _append_axis(
            interior=x_crossed[1:Z - 1, 1:Y - 1, :],
            i_lbl=labels[1:Z - 1, 1:Y - 1, :-1],
            j_lbl=labels[1:Z - 1, 1:Y - 1, 1:],
            cells=(
                ("A", 0, slice(0, Z - 2), slice(0, Y - 2), slice(0, X - 1), 3),
                ("B", 0, slice(1, Z - 1), slice(0, Y - 2), slice(0, X - 1), 1),
                ("C", 0, slice(0, Z - 2), slice(1, Y - 1), slice(0, X - 1), 2),
                ("D", 0, slice(1, Z - 1), slice(1, Y - 1), slice(0, X - 1), 0),
            ),
            cell_to_vertex=cell_to_vertex,
            edge_comp=edge_comp,
            pos_winding=("A", "C", "D", "B"),
            quad_lists=quad_lists, label_lists=label_lists,
        )

    # ----- Y-edges. Interior z ∈ [1, Z-2], x ∈ [1, X-2] -----
    # 4 incident cells and their cell-edge indices for this grid Y-edge:
    #   A=(z-1, y, x-1): cell-edge 7 (Y-edge at top, dx=1)
    #   B=(z,   y, x-1): cell-edge 5 (Y-edge at bottom, dx=1)
    #   C=(z-1, y, x):   cell-edge 6 (Y-edge at top, dx=0)
    #   D=(z,   y, x):   cell-edge 4 (Y-edge at bottom, dx=0)
    if Z >= 3 and Y >= 2 and X >= 3:
        _append_axis(
            interior=y_crossed[1:Z - 1, :, 1:X - 1],
            i_lbl=labels[1:Z - 1, :-1, 1:X - 1],
            j_lbl=labels[1:Z - 1, 1:, 1:X - 1],
            cells=(
                ("A", 0, slice(0, Z - 2), slice(0, Y - 1), slice(0, X - 2), 7),
                ("B", 0, slice(1, Z - 1), slice(0, Y - 1), slice(0, X - 2), 5),
                ("C", 0, slice(0, Z - 2), slice(0, Y - 1), slice(1, X - 1), 6),
                ("D", 0, slice(1, Z - 1), slice(0, Y - 1), slice(1, X - 1), 4),
            ),
            cell_to_vertex=cell_to_vertex,
            edge_comp=edge_comp,
            pos_winding=("A", "B", "D", "C"),
            quad_lists=quad_lists, label_lists=label_lists,
        )

    # ----- Z-edges. Interior y ∈ [1, Y-2], x ∈ [1, X-2] -----
    # 4 incident cells and their cell-edge indices for this grid Z-edge:
    #   A=(z, y-1, x-1): cell-edge 11 (Z-edge at dy=1, dx=1)
    #   B=(z, y,   x-1): cell-edge 9  (Z-edge at dy=0, dx=1)
    #   C=(z, y-1, x):   cell-edge 10 (Z-edge at dy=1, dx=0)
    #   D=(z, y,   x):   cell-edge 8  (Z-edge at dy=0, dx=0)
    if Z >= 2 and Y >= 3 and X >= 3:
        _append_axis(
            interior=z_crossed[:, 1:Y - 1, 1:X - 1],
            i_lbl=labels[:-1, 1:Y - 1, 1:X - 1],
            j_lbl=labels[1:, 1:Y - 1, 1:X - 1],
            cells=(
                ("A", 0, slice(0, Z - 1), slice(0, Y - 2), slice(0, X - 2), 11),
                ("B", 0, slice(0, Z - 1), slice(1, Y - 1), slice(0, X - 2), 9),
                ("C", 0, slice(0, Z - 1), slice(0, Y - 2), slice(1, X - 1), 10),
                ("D", 0, slice(0, Z - 1), slice(1, Y - 1), slice(1, X - 1), 8),
            ),
            cell_to_vertex=cell_to_vertex,
            edge_comp=edge_comp,
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
    cells: tuple,               # ((tag, _pad, sz, sy, sx, edge_idx), ...) for A, B, C, D
    cell_to_vertex: np.ndarray, # (Zm1, Ym1, Xm1, MAX_COMP)
    edge_comp: np.ndarray,      # (Zm1, Ym1, Xm1, 12)
    pos_winding: tuple,
    quad_lists: list[np.ndarray],
    label_lists: list[np.ndarray],
) -> None:
    """Mask interior crossings, look up per-cell component vertices, emit
    quads + BoundaryLabels.

    For each incident cell, the corresponding cell-edge index identifies
    which component of that cell the grid edge belongs to; we use the
    matching vertex from ``cell_to_vertex[cell, component]``.
    """
    if not interior.any():
        return
    mask = interior

    # For each of the 4 incident cells, look up the component vertex.
    v_lookup: dict[str, np.ndarray] = {}
    for tag, _pad, sz, sy, sx, edge_idx in cells:
        cell_ctv = cell_to_vertex[sz, sy, sx, :]            # (..., MAX_COMP)
        cell_ec = edge_comp[sz, sy, sx, edge_idx]           # (...,) component id, -1 if uncrossed
        # Crossed grid edges → all 4 incident cells must be boundary cells
        # on this edge → component IDs are ≥ 0 there. We still defensively
        # clip to 0 to keep indexing valid for masked-out cells.
        k = np.where(cell_ec >= 0, cell_ec, 0).astype(np.intp)
        vert = np.take_along_axis(cell_ctv, k[..., None], axis=-1)[..., 0]
        v_lookup[tag] = vert[mask]

    i_f = i_lbl[mask]
    j_f = j_lbl[mask]

    # VTK BoundaryLabels rule (background last; else sorted ascending).
    swap_for_zero = (i_f == 0) & (j_f != 0)
    swap_for_sort = (i_f != 0) & (j_f != 0) & (j_f < i_f)
    flip = swap_for_zero | swap_for_sort

    label0 = np.where(flip, j_f, i_f)
    label1 = np.where(flip, i_f, j_f)

    q_pos = np.stack([v_lookup[name] for name in pos_winding], axis=-1)
    neg_winding = (pos_winding[0], pos_winding[3], pos_winding[2], pos_winding[1])
    q_neg = np.stack([v_lookup[name] for name in neg_winding], axis=-1)

    q = np.where(flip[:, None], q_neg, q_pos)
    quad_lists.append(q)
    label_lists.append(np.stack([label0, label1], axis=-1))


# ---------------------------------------------------------------------------
# Gradient-refined vertex placement and field-gradient normals
# ---------------------------------------------------------------------------


def _gradient_refine(
    points: np.ndarray,                # (N, 3) global (Z, Y, X) grid coords
    cell_to_vertex: np.ndarray,        # (Zm1, Ym1, Xm1, MAX_COMP) int64
    n_comp: np.ndarray,                # (Zm1, Ym1, Xm1) int8
    comp_pairs: np.ndarray,            # (Zm1, Ym1, Xm1, MAX_COMP, 2) int32
    logits: np.ndarray,                # (K, Z, Y, X) float32
    *,
    project: bool,
    emit_normals: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Refine vertex positions and/or emit field-gradient normals.

    For each vertex we know:
      - its host cell ``(a, b, c)`` and component slot ``k``;
      - its label pair ``(i, j)`` from ``comp_pairs[a, b, c, k]``.

    The boundary surface for this vertex is the locus
    ``f(x) = logit_i(x) - logit_j(x) = 0``. We evaluate ``f`` and
    ``∇f`` at the vertex position via trilinear interp of the 8
    surrounding corner logits.

    ``project`` does one Newton step ``x ← x − f · ∇f / |∇f|²`` for
    *binary cells only* (single-component) — multi-component vertices
    are left at the cell center so the no-gap property at triple
    junctions is preserved.

    ``emit_normals`` returns ``normals[v] = ∇(L_{Label1} − L_{Label0})``
    normalized at the (possibly refined) vertex position. Matches the
    VTK convention ``normal → Label0 → Label1``: for ascending-sorted
    pair ``(p0, p1)`` and Label0 = p0 (the non-bg slot), the desired
    normal is ``∇(L_{p1} − L_{p0}) = −∇f``. For pairs with background
    in them, Label0 = non-bg, Label1 = 0; same algebra still gives
    ``−∇f``.
    """
    Zm1, Ym1, Xm1, MAX_COMP = cell_to_vertex.shape
    K, Z, Y, X = logits.shape
    n_total = int(points.shape[0])
    if n_total == 0:
        normals = np.zeros((0, 3), dtype=np.float32) if emit_normals else None
        return points, normals

    # Per-vertex info: cell (a, b, c), component k, label pair (i, j).
    used = (cell_to_vertex >= 0)
    z_arr, y_arr, x_arr, k_arr = np.nonzero(used)
    # Re-order so vertex IDs match np.arange order (which matches
    # C-order over `used` — np.nonzero already gives that order).
    vert_ids = cell_to_vertex[z_arr, y_arr, x_arr, k_arr]
    # Map global vertex ID → row in our per-vertex arrays.
    order = np.argsort(vert_ids, kind="stable")
    z_arr = z_arr[order]
    y_arr = y_arr[order]
    x_arr = x_arr[order]
    k_arr = k_arr[order]

    pair = comp_pairs[z_arr, y_arr, x_arr, k_arr]          # (N, 2)
    i_arr = pair[:, 0].astype(np.intp)
    j_arr = pair[:, 1].astype(np.intp)

    # Local position (u, v, w) ∈ [0, 1] within the cell.
    cell_base = np.stack([z_arr, y_arr, x_arr], axis=-1).astype(np.float32)
    local = points - cell_base                              # (N, 3) in (Z, Y, X)
    np.clip(local, np.float32(0.0), np.float32(1.0), out=local)

    # Gather f = logit_i - logit_j at the 8 cube corners for each vertex.
    f_corners = np.empty((8, n_total), dtype=np.float32)
    for corner_idx in range(8):
        dz = (corner_idx >> 2) & 1
        dy = (corner_idx >> 1) & 1
        dx = corner_idx & 1
        zs = z_arr + dz
        ys = y_arr + dy
        xs = x_arr + dx
        l_i = logits[i_arr, zs, ys, xs]
        l_j = logits[j_arr, zs, ys, xs]
        f_corners[corner_idx] = l_i - l_j

    if project:
        # Binary cells (single component) get a Newton step.
        n_at_cell = n_comp[z_arr, y_arr, x_arr]
        is_binary = (n_at_cell == 1)
        if is_binary.any():
            f_val = _trilinear_eval(f_corners, local)
            grad = _trilinear_grad(f_corners, local)        # (3, N)
            grad_sq = (grad ** 2).sum(axis=0)
            safe = np.maximum(grad_sq, np.float32(1e-10))
            step_scale = (-f_val / safe).astype(np.float32) # (N,) Newton: x ← x − f/|∇f|² · ∇f
            # Step in (Z, Y, X) order; clamp local position to [0, 1].
            step_local = step_scale[:, None] * grad.T       # (N, 3)
            new_local = np.where(is_binary[:, None], local + step_local, local)
            np.clip(new_local, np.float32(0.0), np.float32(1.0), out=new_local)
            local = new_local
            points = cell_base + local

    normals: np.ndarray | None = None
    if emit_normals:
        grad = _trilinear_grad(f_corners, local)            # (3, N)
        # The VTK BoundaryLabels rule puts background in slot 1; non-bg
        # pairs are sorted ascending. The "normal points Label0 → Label1"
        # convention then has different sign in terms of our sorted-pair
        # f = L_{pair[0]} − L_{pair[1]}:
        #   * pair (a, b) both non-zero: Label0=a, Label1=b → normal = −∇f
        #   * pair (0, b):               Label0=b, Label1=0 → normal = +∇f
        # i_arr is the smaller member of the sorted pair, so i_arr == 0
        # exactly captures the bg-involving case.
        sign = np.where(i_arr == 0, np.float32(1.0), np.float32(-1.0))
        n = (sign[:, None] * grad.T).astype(np.float32)
        mag = np.linalg.norm(n, axis=1, keepdims=True)
        mag = np.maximum(mag, np.float32(1e-10))
        normals = (n / mag).astype(np.float32)

    return np.ascontiguousarray(points.astype(np.float32)), normals


def _trilinear_eval(f_corners: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of ``f`` at local (u, v, w) ∈ [0, 1]³.

    ``f_corners`` is (8, N): one value per cube corner per vertex,
    indexed by corner id ``4*dz + 2*dy + dx``. ``local`` is (N, 3) with
    components in (Z, Y, X) order, i.e. ``local[:, 0] = u``,
    ``local[:, 1] = v``, ``local[:, 2] = w``.
    """
    u = local[:, 0]; v = local[:, 1]; w = local[:, 2]
    nu = 1.0 - u; nv = 1.0 - v; nw = 1.0 - w
    return (
        f_corners[0] * nu * nv * nw +
        f_corners[1] * nu * nv * w +
        f_corners[2] * nu * v * nw +
        f_corners[3] * nu * v * w +
        f_corners[4] * u * nv * nw +
        f_corners[5] * u * nv * w +
        f_corners[6] * u * v * nw +
        f_corners[7] * u * v * w
    )


def _trilinear_grad(f_corners: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Gradient ``∇f`` at local (u, v, w) — closed form for trilinear.

    Each component is bilinear in the other two coordinates of paired-
    corner differences:

        ∂f/∂u = Σ (f[4+i] − f[i]) · w_v(v) · w_w(w)
        ∂f/∂v = Σ (f[2+i] − f[i]) · w_u(u) · w_w(w)
        ∂f/∂w = Σ (f[1+i] − f[i]) · w_u(u) · w_v(v)

    Returns (3, N) in (u, v, w) = (Z, Y, X) order.
    """
    u = local[:, 0]; v = local[:, 1]; w = local[:, 2]
    nu = 1.0 - u; nv = 1.0 - v; nw = 1.0 - w
    dfdu = (
        (f_corners[4] - f_corners[0]) * nv * nw +
        (f_corners[5] - f_corners[1]) * nv * w +
        (f_corners[6] - f_corners[2]) * v * nw +
        (f_corners[7] - f_corners[3]) * v * w
    )
    dfdv = (
        (f_corners[2] - f_corners[0]) * nu * nw +
        (f_corners[3] - f_corners[1]) * nu * w +
        (f_corners[6] - f_corners[4]) * u * nw +
        (f_corners[7] - f_corners[5]) * u * w
    )
    dfdw = (
        (f_corners[1] - f_corners[0]) * nu * nv +
        (f_corners[3] - f_corners[2]) * nu * v +
        (f_corners[5] - f_corners[4]) * u * nv +
        (f_corners[7] - f_corners[6]) * u * v
    )
    return np.stack([dfdu, dfdv, dfdw], axis=0).astype(np.float32)


__all__ = ["surfacenets_logits"]
