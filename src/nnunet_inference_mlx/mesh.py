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
from typing import Union

import mlx.core as mx
import numpy as np

from .values import Geometry, LabelSchema, Mesh

LogitsLike = Union[np.ndarray, "mx.array"]


def surfacenets_logits(
    logits: LogitsLike,
    geometry: Geometry,
    schema: LabelSchema,
    *,
    project_to_surface: bool = False,
    emit_normals: bool = False,
    confidence_margin: float = 0.0,
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
    confidence_margin :
        Logit-margin floor for the *spike-voxel* edge filter. A voxel
        is classified as a spike when both:

          1. its top-1 vs top-2 logit margin is below ``confidence_margin``;
          2. none of its 6 axis-aligned neighbors share its label.

        Any grid edge incident to a spike voxel is dropped from the
        boundary topology, so the surrounding octahedron mesh never
        materializes. The voxel itself keeps its argmax (this knob
        does not relabel) and remains a corner of every adjacent cell;
        it just doesn't contribute crossings on its 6 outgoing edges.

        The dual condition (low margin **and** topological isolation)
        is what makes this safe on soft-tissue boundaries: organ
        surface voxels typically have many same-label neighbors and
        are protected even when their margin is low. The continuous
        logit info is consulted (via ``margin``) and the discrete
        topological neighborhood is consulted (via ``n_same == 0``);
        an edge is dropped only when both signals agree.

        Operates non-destructively — unlike ``confidence_threshold``,
        which relabels spike voxels to a neighbor majority — and so
        composes cleanly with the geometric refinements above. Values
        0.5–2.0 typically clean the chest noise floor without
        affecting anatomic surfaces; ``0.0`` (default) leaves the
        topology criterion as plain ``argmax labels differ``.
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

    # Maintain a single mx.array for GPU-side K-channel ops *and* a
    # numpy view for the remaining CPU paths. On Apple Silicon's
    # unified memory the two share storage — no data motion either way.
    if isinstance(logits, np.ndarray):
        logits_np = np.ascontiguousarray(logits, dtype=np.float32)
        logits_mx = mx.array(logits_np)
    else:
        if logits.dtype != mx.float32:
            logits = logits.astype(mx.float32)
        mx.eval(logits)
        logits_mx = logits
        logits_np = np.asarray(logits_mx)

    # Fused argmax + top-1/top-2 margin on the GPU. Margin is computed
    # unconditionally — the MLX cost is roughly the same as argmax alone,
    # and downstream consumers (confidence_threshold relabel, spike mask)
    # would otherwise recompute it.
    labels, margin_vol = _argmax_and_margin(logits_mx)

    if confidence_threshold > 0.0:
        labels = _suppress_low_confidence_blobs(
            labels, logits_np, float(confidence_threshold),
            precomputed_margin=margin_vol,
        )
    if drop_components_below_mm3 > 0.0:
        from .postprocessing import remove_small_components
        labels = remove_small_components(
            labels, geometry.spacing_zyx,
            min_volume_mm3=float(drop_components_below_mm3),
            in_place=False,
        ).astype(np.int32, copy=False)

    # Edge-level spike-suppression mask (low margin AND topologically
    # isolated voxel). Reuses the same margin we just computed.
    spike_mask: np.ndarray | None = None
    if confidence_margin > 0.0:
        spike_mask = _compute_spike_mask(
            labels, margin_vol, float(confidence_margin),
        )
    del margin_vol

    x_crossed, x_t = _edge_crossings(
        logits_mx, labels, axis=2, spike_mask=spike_mask,
    )
    y_crossed, y_t = _edge_crossings(
        logits_mx, labels, axis=1, spike_mask=spike_mask,
    )
    z_crossed, z_t = _edge_crossings(
        logits_mx, labels, axis=0, spike_mask=spike_mask,
    )

    edge_comp, n_comp, comp_pairs = _cell_components(
        labels, logits_np, x_crossed, y_crossed, z_crossed,
    )

    cell_to_vertex, points = _cell_dual_vertices(
        edge_comp, n_comp,
        x_t, y_t, z_t,
    )

    normals: np.ndarray | None = None
    if project_to_surface or emit_normals:
        points, normals = _gradient_refine(
            points, cell_to_vertex, n_comp, comp_pairs, logits_np,
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
# Per-voxel logit margin (shared between confidence_threshold and
# confidence_margin paths)
# ---------------------------------------------------------------------------


def _argmax_and_margin(logits: "LogitsLike") -> tuple[np.ndarray, np.ndarray]:
    """Fused argmax + top1−top2 margin in a single MLX pass via argpartition.

    Direct ``mx.argmax(axis=0)`` is surprisingly slow on M2 for wide K
    (~720 ms on chest at K=118) — it appears to use a sequential
    reduction kernel rather than a tree reduction. ``mx.argpartition``
    with ``kth=K-2`` runs the same K-channel reduction in ~250 ms and
    delivers the top-2 indices in one shot; we then extract top-1
    vs top-2 with two cheap take_along_axis gathers and a 2-element
    reduce. Total ~300 ms vs ~1.5 s for the equivalent numpy path.

    Returns ``(labels, margin)`` as numpy arrays (int32, float32) —
    materialization happens at the boundary so all downstream CPU
    code is unchanged.
    """
    logits_mx = mx.array(logits) if isinstance(logits, np.ndarray) else logits
    if logits_mx.dtype != mx.float32:
        logits_mx = logits_mx.astype(mx.float32)
    K = logits_mx.shape[0]
    # argpartition with kth=K-2 puts the 2 largest elements at the end
    # of axis 0 (unordered within those last two slots). Take_along_axis
    # to materialize the corresponding logit values, then a 2-element
    # argmax decides which of the two slots is top-1 vs top-2.
    part_idx = mx.argpartition(logits_mx, kth=K - 2, axis=0)[K - 2:]  # (2, Z, Y, X)
    part_vals = mx.take_along_axis(logits_mx, part_idx, axis=0)        # (2, Z, Y, X)
    which_is_top = mx.argmax(part_vals, axis=0)                        # (Z, Y, X), 0 or 1
    labels_mx = mx.take_along_axis(part_idx, which_is_top[None], axis=0)[0]
    top1 = mx.take_along_axis(part_vals, which_is_top[None], axis=0)[0]
    top2 = mx.take_along_axis(part_vals, (1 - which_is_top)[None], axis=0)[0]
    margin_mx = top1 - top2
    mx.eval(labels_mx, margin_mx)
    labels = np.asarray(labels_mx).astype(np.int32, copy=False)
    margin = np.asarray(margin_mx).astype(np.float32, copy=False)
    return labels, margin


def _compute_margin(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Per-voxel ``top1_logit − top2_logit`` (float32, shape ``labels.shape``).

    Standalone CPU fallback — used by tests and any caller that
    already holds ``labels`` and just wants margin. The pipeline path
    in :func:`surfacenets_logits` uses :func:`_argmax_and_margin`
    instead, which fuses both computations into a single GPU pass.
    """
    winner_idx = labels.astype(np.intp)[None]
    winner_logit = np.take_along_axis(logits, winner_idx, axis=0)[0]
    masked = logits.copy()
    np.put_along_axis(masked, winner_idx, np.float32(-np.inf), axis=0)
    second_logit = masked.max(axis=0)
    return (winner_logit - second_logit).astype(np.float32, copy=False)


def _compute_spike_mask(
    labels: np.ndarray, margin: np.ndarray, threshold: float,
) -> np.ndarray:
    """Per-voxel bool mask: True where the voxel is a low-confidence
    *and* topologically-isolated argmax flip — i.e. a "spike voxel"
    whose label has no 6-connected same-label support and whose
    top-1 vs top-2 logit margin sits below ``threshold``.

    Volume-boundary voxels (no full 6-neighbor support to evaluate)
    are always False — a spike at the volume edge can't be
    distinguished from a legitimate boundary-clipped object.

    This is the topology-criterion analog of the per-voxel relabel in
    :func:`_suppress_low_confidence_blobs`, but applied as an edge
    filter in the downstream pipeline (we never touch ``labels``).
    The argmax decision at the spike voxel is preserved; only the
    outgoing topology contribution is suppressed.
    """
    Z, Y, X = labels.shape
    spike = np.zeros(labels.shape, dtype=bool)
    if Z < 3 or Y < 3 or X < 3:
        return spike

    center = labels[1:-1, 1:-1, 1:-1]
    nbrs = np.stack([
        labels[ :-2, 1:-1, 1:-1],
        labels[2:,   1:-1, 1:-1],
        labels[1:-1,  :-2, 1:-1],
        labels[1:-1, 2:,   1:-1],
        labels[1:-1, 1:-1,  :-2],
        labels[1:-1, 1:-1, 2:  ],
    ], axis=-1)
    n_same = (nbrs == center[..., None]).sum(axis=-1)
    low_confidence = margin[1:-1, 1:-1, 1:-1] < np.float32(threshold)
    is_isolated = (n_same == 0)
    spike[1:-1, 1:-1, 1:-1] = low_confidence & is_isolated
    return spike


# ---------------------------------------------------------------------------
# Low-confidence single-voxel blob suppression
# ---------------------------------------------------------------------------


def _suppress_low_confidence_blobs(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float,
    *,
    precomputed_margin: np.ndarray | None = None,
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

    margin = (
        precomputed_margin if precomputed_margin is not None
        else _compute_margin(labels, logits)
    )

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
    logits_mx: "mx.array",
    labels: np.ndarray,
    axis: int,
    *,
    spike_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MLX implementation. For grid edges along ``axis`` (0=Z, 1=Y, 2=X), return:

      * ``crossed`` — bool array (one element shorter than ``labels`` on
        ``axis``), True where the endpoint dominant labels differ
        *and* (when ``spike_mask`` is provided) neither endpoint is a
        spike voxel;
      * ``t`` — float32 array of the same shape, with the sub-voxel
        position in [0, 1] where ``logit_i - logit_j`` crosses zero
        (i at endpoint a, j at endpoint b).

    Returns numpy arrays — the K-channel reduction (4 gathers across
    logits along axis 0) is the GPU win; the resulting per-voxel scalar
    fields are small and materialize back to numpy for the downstream
    CPU-side case-table dispatch.
    """
    sl_a = [slice(None)] * 3
    sl_b = [slice(None)] * 3
    sl_a[axis] = slice(None, -1)
    sl_b[axis] = slice(1, None)
    L0_np = labels[tuple(sl_a)]
    L1_np = labels[tuple(sl_b)]
    crossed = (L0_np != L1_np)
    if spike_mask is not None:
        # Drop edges incident to a spike voxel (low-confidence + isolated).
        spike_a = spike_mask[tuple(sl_a)]
        spike_b = spike_mask[tuple(sl_b)]
        crossed = crossed & ~(spike_a | spike_b)

    # MLX side: gather the 4 logit values per edge in a single graph.
    sl_a_log = [slice(None)] * 4
    sl_b_log = [slice(None)] * 4
    sl_a_log[axis + 1] = slice(None, -1)
    sl_b_log[axis + 1] = slice(1, None)
    logits_a = logits_mx[tuple(sl_a_log)]
    logits_b = logits_mx[tuple(sl_b_log)]

    L0_mx = mx.array(L0_np.astype(np.int32, copy=False))[None]
    L1_mx = mx.array(L1_np.astype(np.int32, copy=False))[None]

    logit_L0_a = mx.take_along_axis(logits_a, L0_mx, axis=0)[0]
    logit_L1_a = mx.take_along_axis(logits_a, L1_mx, axis=0)[0]
    logit_L0_b = mx.take_along_axis(logits_b, L0_mx, axis=0)[0]
    logit_L1_b = mx.take_along_axis(logits_b, L1_mx, axis=0)[0]

    d0 = logit_L0_a - logit_L1_a
    d1 = logit_L0_b - logit_L1_b
    denom = d0 - d1
    eps = mx.array(np.float32(1e-30))
    one = mx.array(np.float32(1.0))
    half = mx.array(np.float32(0.5))
    zero = mx.array(np.float32(0.0))
    safe_denom = mx.where(denom > eps, denom, one)
    t_mx = mx.where(denom > eps, d0 / safe_denom, half)
    # Mask t to zero outside crossed edges so downstream accumulators
    # don't need to gate.
    crossed_mx = mx.array(crossed)
    t_mx = mx.where(crossed_mx, t_mx, zero)
    mx.eval(t_mx)
    t = np.asarray(t_mx).astype(np.float32, copy=False)
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


@lru_cache(maxsize=131072)
def _cell_case(
    corner_labels: tuple, crossed_mask: int, saddle_flips: int = 0,
) -> tuple:
    """For one cell configuration, return ``(n_components, edge_to_component)``.

    ``corner_labels`` is an 8-tuple of integer labels at the 8 cell
    corners (in canonical 0..7 order). ``crossed_mask`` is a 12-bit int
    where bit ``e`` set means "cell-edge ``e`` is committed to a boundary
    crossing." In the absence of an edge-level confidence rule this is
    just ``(la != lb)`` per edge and is fully determined by
    ``corner_labels``; with ``confidence_margin > 0`` the caller may
    drop bits where the edge fails the margin test, and those edges no
    longer induce topology even if the corner labels differ.
    ``saddle_flips`` is a 6-bit int where bit ``f`` set means "flip the
    saddle rule on face ``f``" — used by the logit-magnitude asymptotic
    decider, computed by the caller. Returns a 12-tuple where element
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

    Cached: a real volume has at most a few tens of thousands of
    distinct cell configurations even when it has millions of cells.
    The cache key combines label config, crossed_mask, and saddle_flips;
    in the common no-confidence-margin path crossed_mask is determined
    by corner_labels so the effective cache space is unchanged.
    """
    edge_pair: list[tuple] = [None] * 12  # type: ignore[list-item]
    crossed = [False] * 12
    for e, (a, b) in enumerate(_EDGE_CORNERS):
        if (crossed_mask >> e) & 1:
            la, lb = corner_labels[a], corner_labels[b]
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
    labels: np.ndarray,
    logits: np.ndarray | None,
    x_crossed: np.ndarray,
    y_crossed: np.ndarray,
    z_crossed: np.ndarray,
    *,
    saddle_flips: np.ndarray | None = None,
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
      2. Pack the per-cell 12-bit ``crossed_mask`` from the three
         per-axis ``crossed`` arrays. A cell is a boundary cell iff its
         mask is nonzero — which is the principled definition under
         ``confidence_margin > 0`` (an all-zero mask cell has no
         committed topology even if its corner labels differ).
      3. Compute the per-cell saddle-flip bitfield from the logit
         asymptotic decider (vectorized), gated on all-four-face-edges
         being crossed (no saddle disambiguation if some face edges
         are confidence-suppressed).
      4. Find unique ``(corner_labels, crossed_mask, saddle_flips)``
         configurations across boundary cells, run :func:`_cell_case`
         once per unique config, scatter the result back.

    Step 4 keeps the Python work at "unique configs" rather than "all
    cells". With ``confidence_margin = 0`` the crossed_mask is fully
    determined by corner_labels so the unique-config count is unchanged;
    with ``confidence_margin > 0`` it grows by a small factor (the
    number of distinct partial-crossed patterns observed in the volume,
    typically ~2-3×).
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

    # Per-cell 12-bit crossed mask.
    cell_crossed_mask = _per_cell_crossed_mask(x_crossed, y_crossed, z_crossed)
    is_boundary = cell_crossed_mask != 0
    if not is_boundary.any():
        comp_pairs = np.zeros((Zm1, Ym1, Xm1, 1, 2), dtype=np.int32)
        return edge_components, n_components, comp_pairs

    # Saddle-flip bitfield (logit asymptotic decider). Caller may supply
    # a precomputed bitfield — useful for the slab-streaming target-grid
    # path where saddle face sums are accumulated inside the K-channel
    # slab loop so we don't have to revisit logits here.
    if saddle_flips is None:
        if logits is None:
            raise ValueError(
                "_cell_components: provide either `logits` or `saddle_flips`"
            )
        saddle_flips = _compute_saddle_flips(
            corners, logits, is_boundary, cell_crossed_mask,
        )

    # Unique (label_config, crossed_mask, saddle_flips) configurations.
    flat_corners = corners.reshape(-1, 8)
    flat_mask = cell_crossed_mask.reshape(-1)
    flat_saddle = saddle_flips.reshape(-1)
    flat_boundary = is_boundary.reshape(-1)
    bc = flat_corners[flat_boundary]                       # (N_b, 8)
    bm = flat_mask[flat_boundary].astype(np.int32)         # (N_b,) widened for stacking
    bs = flat_saddle[flat_boundary].astype(np.int32)       # (N_b,)
    combined = np.concatenate(
        [bc, bm[:, None], bs[:, None]], axis=1,
    )                                                       # (N_b, 10)
    unique_cfgs, inverse = np.unique(combined, axis=0, return_inverse=True)

    nb = unique_cfgs.shape[0]
    n_per_cfg = np.zeros(nb, dtype=np.int8)
    edge_per_cfg = np.full((nb, 12), -1, dtype=np.int8)
    case_results = []
    for i in range(nb):
        cfg = tuple(int(v) for v in unique_cfgs[i, :8])
        mask = int(unique_cfgs[i, 8])
        saddle = int(unique_cfgs[i, 9])
        n, et, pairs = _cell_case(cfg, mask, saddle)
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


def _per_cell_crossed_mask(
    x_crossed: np.ndarray,   # (Z,   Y,   X-1) bool
    y_crossed: np.ndarray,   # (Z,   Y-1, X)   bool
    z_crossed: np.ndarray,   # (Z-1, Y,   X)   bool
) -> np.ndarray:
    """Pack the per-cell 12-bit ``crossed`` mask.

    Bit layout matches :data:`_EDGE_CORNERS`:

      bits 0..3   X-edges, indexed by (dz, dy):  (0,0), (0,1), (1,0), (1,1)
      bits 4..7   Y-edges, indexed by (dz, dx):  (0,0), (0,1), (1,0), (1,1)
      bits 8..11  Z-edges, indexed by (dy, dx):  (0,0), (0,1), (1,0), (1,1)

    Returns ``(Z-1, Y-1, X-1) uint16``.
    """
    Zm1 = z_crossed.shape[0]
    Ym1 = y_crossed.shape[1]
    Xm1 = x_crossed.shape[2]
    mask = np.zeros((Zm1, Ym1, Xm1), dtype=np.uint16)
    # X-edges (4 per cell)
    mask |= x_crossed[:Zm1,     :Ym1,    :].astype(np.uint16) << 0
    mask |= x_crossed[:Zm1,     1:Ym1+1, :].astype(np.uint16) << 1
    mask |= x_crossed[1:Zm1+1,  :Ym1,    :].astype(np.uint16) << 2
    mask |= x_crossed[1:Zm1+1,  1:Ym1+1, :].astype(np.uint16) << 3
    # Y-edges (4 per cell)
    mask |= y_crossed[:Zm1,     :, :Xm1     ].astype(np.uint16) << 4
    mask |= y_crossed[:Zm1,     :, 1:Xm1+1  ].astype(np.uint16) << 5
    mask |= y_crossed[1:Zm1+1,  :, :Xm1     ].astype(np.uint16) << 6
    mask |= y_crossed[1:Zm1+1,  :, 1:Xm1+1  ].astype(np.uint16) << 7
    # Z-edges (4 per cell)
    mask |= z_crossed[:, :Ym1,    :Xm1     ].astype(np.uint16) << 8
    mask |= z_crossed[:, :Ym1,    1:Xm1+1  ].astype(np.uint16) << 9
    mask |= z_crossed[:, 1:Ym1+1, :Xm1     ].astype(np.uint16) << 10
    mask |= z_crossed[:, 1:Ym1+1, 1:Xm1+1  ].astype(np.uint16) << 11
    return mask


# Precomputed per-face bitmask of cell-edges that belong to that face;
# used by `_compute_saddle_flips` to test "all 4 face edges crossed."
_FACE_EDGE_BITS = tuple(
    sum(1 << e for e in face_edges) for _, face_edges in _FACES
)


def _compute_saddle_flips(
    corners: np.ndarray,
    logits: np.ndarray,
    boundary_mask: np.ndarray,
    cell_crossed_mask: np.ndarray,
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

    A face is only a true saddle if *all 4 of its cell-edges are
    crossed*. With ``confidence_margin > 0`` an apparent (A,B,A,B)
    label pattern can have one or more face edges confidence-suppressed,
    in which case the face is not a saddle and the flip bit must stay
    clear (else the case-table dispatch would consult a flip rule that
    doesn't apply).

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
        face_bits = np.uint16(_FACE_EDGE_BITS[face_idx])
        face_fully_crossed = (
            (cell_crossed_mask & face_bits) == face_bits
        )
        is_saddle = (
            (c0_lbl == c2_lbl) & (c1_lbl == c3_lbl) & (c0_lbl != c1_lbl)
            & boundary_mask & face_fully_crossed
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

    # Per-cell crossing-centroid accumulation. NOTE: a tested MLX
    # port of this 12-pass accumulation was 4.3× faster in isolation
    # (110 ms vs 480 ms on chest TS-fast) but a ~1 s end-to-end
    # regression when integrated — the MLX graph compile/eval costs
    # interleave badly with the other MLX stages (argmax, edge
    # crossings) and the variance grew. Kept on CPU until the rest
    # of the pipeline is unified into a single fused MLX graph
    # (Phase 3+ work).
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


# ---------------------------------------------------------------------------
# Slab-streaming surfacenets at a target grid (output spacing != source).
#
# Decomposes the surfacenets pipeline by what consumes K-channel data:
#
#   Pass 1  (slab-stream, K-channel work):
#     For each Z-slab of the target grid: trilinear-gather a K-channel
#     slab from source, run argmax+margin+edge_crossings+saddle_face_sums
#     on it, and write the small reduced state (labels, margin, per-axis
#     crossed/t, saddle_flips) into full-grid numpy buffers. K-channel
#     slab discarded between iterations → peak memory bounded by slab
#     budget, not by the (potentially huge) full upsampled volume.
#
#   Pass 2  (full-grid CPU work):
#     The case-table dispatch + cell dual-vertex placement use only the
#     small reduced buffers. Identical to the in-place pipeline.
#
#   Pass 3  (sparse vertex refine):
#     gradient_refine evaluates ∇f and f at vertex positions. Since the
#     upsampled K-channel volume is *defined* as a trilinear interp of
#     source, evaluating at a target-grid vertex maps trivially to
#     evaluating at the corresponding source coordinate via the same
#     trilinear over source corners. So gradient_refine reads directly
#     from the source K-channel logits (already resident) — never needs
#     the materialised upsampled volume.
#
# Result: bitwise-equivalent to the all-at-once version (when the
# all-at-once version would fit), but memory-bounded for any output
# size. Pattern mirrors inverse_resample_argmax for the labelmap path.
# ---------------------------------------------------------------------------


def _slab_stream_reduced(
    src_logits_mx: "mx.array",
    out_shape_zyx: tuple[int, int, int],
    src_spacing_zyx: tuple[float, float, float],
    out_spacing_zyx: tuple[float, float, float],
    *,
    peak_working_memory_mb: int = 1024,
) -> dict:
    """Pass 1: slab-stream K-channel work and fill full-grid reduced state.

    Returns a dict of full-grid numpy arrays:

      labels        (Z_o, Y_o, X_o)         int32
      margin        (Z_o, Y_o, X_o)         float32  (top1−top2 logit)
      x_crossed     (Z_o, Y_o, X_o-1)       bool
      x_t           (Z_o, Y_o, X_o-1)       float32  (sub-voxel position)
      y_crossed     (Z_o, Y_o-1, X_o)       bool
      y_t           (Z_o, Y_o-1, X_o)       float32
      z_crossed     (Z_o-1, Y_o, X_o)       bool
      z_t           (Z_o-1, Y_o, X_o)       float32
      saddle_flips  (Z_o-1, Y_o-1, X_o-1)   uint8  (6-bit per cell)

    The K-channel slab peaks at ~9× the slab K-channel size due to
    trilinear gather intermediates; ``peak_working_memory_mb`` sizes
    the slab to bound that peak.
    """
    from .resampling import (
        _precompute_trilinear_indices,
        _trilinear_from_indices_K,
    )

    K, Z_s, Y_s, X_s = src_logits_mx.shape
    Z_o, Y_o, X_o = out_shape_zyx

    # Slab size: bound peak K-channel + ~9× intermediates (8 corner
    # gathers in the trilinear blend + final blend held simultaneously).
    # Tried lowering this factor to 3× empirically — bigger slabs swapped
    # against system memory and ran 4× slower despite ~3× fewer iterations,
    # so the conservative 9× estimate stays. (MLX lazy eval doesn't fuse
    # away the corner intermediates on M2 in practice.)
    bytes_per_voxel = K * 4
    peak_factor = 9
    max_slab_voxels = (
        peak_working_memory_mb * 1024 * 1024 // (bytes_per_voxel * peak_factor)
    )
    plane = max(1, Y_o * X_o)
    slab_voxels = max(2, max_slab_voxels // plane)
    slab_voxels = min(slab_voxels, Z_o)

    s2t = tuple(out_spacing_zyx[i] / src_spacing_zyx[i] for i in range(3))
    y_coords = mx.arange(Y_o, dtype=mx.float32) * s2t[1]
    x_coords = mx.arange(X_o, dtype=mx.float32) * s2t[2]

    # Allocate full-grid reduced buffers.
    labels       = np.empty((Z_o, Y_o, X_o), dtype=np.int32)
    margin       = np.empty((Z_o, Y_o, X_o), dtype=np.float32)
    x_crossed    = np.empty((Z_o, Y_o, X_o - 1), dtype=bool)
    x_t_buf      = np.empty((Z_o, Y_o, X_o - 1), dtype=np.float32)
    y_crossed    = np.empty((Z_o, Y_o - 1, X_o), dtype=bool)
    y_t_buf      = np.empty((Z_o, Y_o - 1, X_o), dtype=np.float32)
    z_crossed    = np.empty((Z_o - 1, Y_o, X_o), dtype=bool)
    z_t_buf      = np.empty((Z_o - 1, Y_o, X_o), dtype=np.float32)
    saddle_flips = np.zeros((Z_o - 1, Y_o - 1, X_o - 1), dtype=np.uint8)

    z_lo = 0
    while z_lo < Z_o:
        z_hi = min(z_lo + slab_voxels - 1, Z_o - 1)  # inclusive last voxel
        slab_size = z_hi - z_lo + 1

        # Source Z range needed: cover output z in [z_lo, z_hi] with ±1 pad.
        z_global = mx.arange(z_lo, z_hi + 1, dtype=mx.float32) * s2t[0]
        z_lo_f = float(z_lo) * s2t[0]
        z_hi_f = float(z_hi) * s2t[0]
        zt_lo = max(0, int(z_lo_f) - 1)
        zt_hi = max(zt_lo + 1, min(Z_s, int(z_hi_f) + 2))
        slab_src = src_logits_mx[:, zt_lo:zt_hi]
        z_local = z_global - zt_lo

        # Trilinear K-channel slab at target coordinates.
        idx = _precompute_trilinear_indices(
            z_local, y_coords, x_coords, zt_hi - zt_lo, Y_s, X_s,
        )
        slab_K = _trilinear_from_indices_K(slab_src, *idx)
        mx.eval(slab_K)

        # Argmax + top-1/top-2 margin (one fused MLX pass).
        slab_labels, slab_margin = _argmax_and_margin(slab_K)

        # Edge crossings (no spike mask in pass 1; applied post-hoc).
        x_cr, x_tv = _edge_crossings(slab_K, slab_labels, axis=2)
        y_cr, y_tv = _edge_crossings(slab_K, slab_labels, axis=1)
        z_cr, z_tv = _edge_crossings(slab_K, slab_labels, axis=0)

        # Saddle face sums for cells fully inside the slab (z in [z_lo, z_hi-1]).
        if slab_size > 1:
            slab_cell_crossed_mask = _per_cell_crossed_mask(x_cr, y_cr, z_cr)
            slab_is_boundary = slab_cell_crossed_mask != 0
            if slab_is_boundary.any():
                slab_corners = np.stack([
                    slab_labels[:-1, :-1, :-1], slab_labels[:-1, :-1, 1:],
                    slab_labels[:-1, 1:, :-1],  slab_labels[:-1, 1:, 1:],
                    slab_labels[1:,  :-1, :-1], slab_labels[1:,  :-1, 1:],
                    slab_labels[1:,  1:, :-1],  slab_labels[1:,  1:, 1:],
                ], axis=-1)
                slab_K_np = np.asarray(slab_K)
                slab_saddle = _compute_saddle_flips(
                    slab_corners, slab_K_np, slab_is_boundary,
                    slab_cell_crossed_mask,
                )
            else:
                slab_saddle = np.zeros(
                    slab_cell_crossed_mask.shape, dtype=np.uint8,
                )

        # Write reduced state into full-grid buffers. Slab boundaries
        # overlap by 1 voxel; writes are idempotent (trilinear is
        # deterministic; argmax/margin/crossings produce identical
        # values from either side of the seam).
        labels[z_lo:z_hi + 1]    = slab_labels
        margin[z_lo:z_hi + 1]    = slab_margin
        x_crossed[z_lo:z_hi + 1] = x_cr
        x_t_buf[z_lo:z_hi + 1]   = x_tv
        y_crossed[z_lo:z_hi + 1] = y_cr
        y_t_buf[z_lo:z_hi + 1]   = y_tv
        if slab_size > 1:
            z_crossed[z_lo:z_hi]    = z_cr
            z_t_buf[z_lo:z_hi]      = z_tv
            saddle_flips[z_lo:z_hi] = slab_saddle

        # Drop slab K-channel before next iteration.
        del slab_K, slab_src

        if z_hi >= Z_o - 1:
            break
        z_lo = z_hi  # 1-voxel overlap with next slab

    return {
        "labels": labels, "margin": margin,
        "x_crossed": x_crossed, "x_t": x_t_buf,
        "y_crossed": y_crossed, "y_t": y_t_buf,
        "z_crossed": z_crossed, "z_t": z_t_buf,
        "saddle_flips": saddle_flips,
    }


def _apply_spike_mask_to_crossings(
    x_crossed: np.ndarray, y_crossed: np.ndarray, z_crossed: np.ndarray,
    spike_mask: np.ndarray,
) -> None:
    """Mask out crossings on edges incident to a spike voxel (in-place).

    Same semantics as the spike_mask gate in :func:`_edge_crossings`, but
    applied to fully-formed full-grid crossed buffers — used by the
    streaming path where Pass 1 doesn't yet have the labels/margin
    needed to compute the spike mask.
    """
    # X-edges: between (z, y, x) and (z, y, x+1).
    spike_x_a = spike_mask[:, :, :-1]
    spike_x_b = spike_mask[:, :, 1:]
    x_crossed &= ~(spike_x_a | spike_x_b)
    # Y-edges: between (z, y, x) and (z, y+1, x).
    spike_y_a = spike_mask[:, :-1, :]
    spike_y_b = spike_mask[:, 1:, :]
    y_crossed &= ~(spike_y_a | spike_y_b)
    # Z-edges: between (z, y, x) and (z+1, y, x).
    spike_z_a = spike_mask[:-1, :, :]
    spike_z_b = spike_mask[1:, :, :]
    z_crossed &= ~(spike_z_a | spike_z_b)


def _gradient_refine_at_source(
    points: np.ndarray,                # (N, 3) vertex positions in target-grid coords
    cell_to_vertex: np.ndarray,        # (Zm1, Ym1, Xm1, MAX_COMP)
    n_comp: np.ndarray,                # (Zm1, Ym1, Xm1) int8
    comp_pairs: np.ndarray,            # (Zm1, Ym1, Xm1, MAX_COMP, 2) int32
    src_logits_mx: "mx.array",         # (K, Z_s, Y_s, X_s) source logits
    src_spacing_zyx: tuple[float, float, float],
    out_spacing_zyx: tuple[float, float, float],
    *,
    project: bool,
    emit_normals: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pass 3: refine vertex positions and/or compute normals using *source* logits.

    The vertex positions live on the target grid, but ``f = L_i − L_j``
    on the target grid is by definition the trilinear interp of source
    ``L_i, L_j``. So we evaluate ``f`` and ``∇f`` directly over the 8
    surrounding source-grid corners — no need to materialise the upsampled
    K-channel volume.

    The Newton step writes target-grid coordinates back into ``points``;
    the gradient in target coords is the source-coord gradient scaled by
    ``s2t = out_spacing / src_spacing`` per axis (chain rule on a
    coordinate scale).
    """
    Zm1, Ym1, Xm1, MAX_COMP = cell_to_vertex.shape
    K, Z_s, Y_s, X_s = src_logits_mx.shape
    n_total = int(points.shape[0])
    if n_total == 0:
        normals = np.zeros((0, 3), dtype=np.float32) if emit_normals else None
        return points, normals

    # Map each vertex's target-grid position to source coords.
    s2t = np.array([
        out_spacing_zyx[0] / src_spacing_zyx[0],
        out_spacing_zyx[1] / src_spacing_zyx[1],
        out_spacing_zyx[2] / src_spacing_zyx[2],
    ], dtype=np.float32)
    src_pos = points * s2t[None, :]                        # (N, 3) in source-grid coords
    np.clip(src_pos[:, 0], 0.0, Z_s - 1.0001, out=src_pos[:, 0])
    np.clip(src_pos[:, 1], 0.0, Y_s - 1.0001, out=src_pos[:, 1])
    np.clip(src_pos[:, 2], 0.0, X_s - 1.0001, out=src_pos[:, 2])

    # Per-vertex (host cell, component slot, label pair).
    used = (cell_to_vertex >= 0)
    z_arr, y_arr, x_arr, k_arr = np.nonzero(used)
    vert_ids = cell_to_vertex[z_arr, y_arr, x_arr, k_arr]
    order = np.argsort(vert_ids, kind="stable")
    z_arr = z_arr[order]; y_arr = y_arr[order]
    x_arr = x_arr[order]; k_arr = k_arr[order]
    pair = comp_pairs[z_arr, y_arr, x_arr, k_arr]          # (N, 2)
    i_arr = pair[:, 0].astype(np.intp)
    j_arr = pair[:, 1].astype(np.intp)

    # 8 source-grid corner indices surrounding each vertex.
    src_z = src_pos[:, 0]; src_y = src_pos[:, 1]; src_x = src_pos[:, 2]
    z0 = np.floor(src_z).astype(np.intp); z1 = z0 + 1
    y0 = np.floor(src_y).astype(np.intp); y1 = y0 + 1
    x0 = np.floor(src_x).astype(np.intp); x1 = x0 + 1
    np.clip(z0, 0, Z_s - 1, out=z0); np.clip(z1, 0, Z_s - 1, out=z1)
    np.clip(y0, 0, Y_s - 1, out=y0); np.clip(y1, 0, Y_s - 1, out=y1)
    np.clip(x0, 0, X_s - 1, out=x0); np.clip(x1, 0, X_s - 1, out=x1)

    u = (src_z - z0.astype(np.float32))                    # local fractional source coords
    v = (src_y - y0.astype(np.float32))
    w = (src_x - x0.astype(np.float32))

    # Gather f = L_i − L_j at the 8 source corners for each vertex.
    # Use the source MLX array; do the gather there for speed, then to numpy.
    src_logits_np = np.asarray(src_logits_mx)
    f_corners = np.empty((8, n_total), dtype=np.float32)
    for corner_idx in range(8):
        dz = (corner_idx >> 2) & 1
        dy = (corner_idx >> 1) & 1
        dx = corner_idx & 1
        zz = z1 if dz else z0
        yy = y1 if dy else y0
        xx = x1 if dx else x0
        l_i = src_logits_np[i_arr, zz, yy, xx]
        l_j = src_logits_np[j_arr, zz, yy, xx]
        f_corners[corner_idx] = l_i - l_j

    # Use the existing trilinear primitives. Pass local source coords (u, v, w).
    local = np.stack([u, v, w], axis=-1)

    if project:
        n_at_cell = n_comp[z_arr, y_arr, x_arr]
        is_binary = (n_at_cell == 1)
        if is_binary.any():
            f_val = _trilinear_eval(f_corners, local)
            grad_src = _trilinear_grad(f_corners, local)    # (3, N) in source coords
            grad_sq = (grad_src ** 2).sum(axis=0)
            safe = np.maximum(grad_sq, np.float32(1e-10))
            # Newton step in source coords: src_pos ← src_pos − f · ∇f / |∇f|²
            step_src = (-f_val / safe).astype(np.float32)
            step_src_local = step_src[:, None] * grad_src.T  # (N, 3) source-coord step
            # Convert source-coord step back to target-coord step: scale by 1/s2t
            inv_s2t = np.float32(1.0) / s2t
            step_target = step_src_local * inv_s2t[None, :]
            new_points = np.where(is_binary[:, None], points + step_target, points)
            points = new_points

    normals: np.ndarray | None = None
    if emit_normals:
        # Re-evaluate gradient at (possibly refined) position.
        if project and is_binary.any():
            # Recompute local source coords after the step.
            src_pos_new = points * s2t[None, :]
            np.clip(src_pos_new[:, 0], 0.0, Z_s - 1.0001, out=src_pos_new[:, 0])
            np.clip(src_pos_new[:, 1], 0.0, Y_s - 1.0001, out=src_pos_new[:, 1])
            np.clip(src_pos_new[:, 2], 0.0, X_s - 1.0001, out=src_pos_new[:, 2])
            z0n = np.floor(src_pos_new[:, 0]).astype(np.intp)
            y0n = np.floor(src_pos_new[:, 1]).astype(np.intp)
            x0n = np.floor(src_pos_new[:, 2]).astype(np.intp)
            np.clip(z0n, 0, Z_s - 1, out=z0n)
            np.clip(y0n, 0, Y_s - 1, out=y0n)
            np.clip(x0n, 0, X_s - 1, out=x0n)
            # Re-gather (only the binary cells moved; just recompute all for simplicity)
            for corner_idx in range(8):
                dz = (corner_idx >> 2) & 1
                dy = (corner_idx >> 1) & 1
                dx = corner_idx & 1
                zz = z0n + dz; yy = y0n + dy; xx = x0n + dx
                np.clip(zz, 0, Z_s - 1, out=zz)
                np.clip(yy, 0, Y_s - 1, out=yy)
                np.clip(xx, 0, X_s - 1, out=xx)
                f_corners[corner_idx] = (
                    src_logits_np[i_arr, zz, yy, xx]
                    - src_logits_np[j_arr, zz, yy, xx]
                )
            un = src_pos_new[:, 0] - z0n.astype(np.float32)
            vn = src_pos_new[:, 1] - y0n.astype(np.float32)
            wn = src_pos_new[:, 2] - x0n.astype(np.float32)
            local = np.stack([un, vn, wn], axis=-1)
        grad_src = _trilinear_grad(f_corners, local)        # (3, N) in source coords
        # Normal lives in TARGET-grid coords. Convert source-grad to target-grad:
        #   ∂f/∂t_axis = ∂f/∂s_axis · ds_axis/dt_axis = ∂f/∂s_axis · s2t_axis
        grad_target = grad_src * s2t[:, None]
        # Sign convention matches in-place gradient_refine:
        #   pair (0, b)            : normal = +∇f
        #   pair (a, b) both non-0 : normal = −∇f
        sign = np.where(i_arr == 0, np.float32(1.0), np.float32(-1.0))
        n = (sign[:, None] * grad_target.T).astype(np.float32)
        mag = np.linalg.norm(n, axis=1, keepdims=True)
        mag = np.maximum(mag, np.float32(1e-10))
        normals = (n / mag).astype(np.float32)

    return np.ascontiguousarray(points.astype(np.float32)), normals


def surfacenets_logits_at_target(
    prediction,                         # Prediction (from values.py)
    target_geometry: Geometry,
    *,
    project_to_surface: bool = False,
    emit_normals: bool = False,
    confidence_margin: float = 0.0,
    confidence_threshold: float = 0.0,
    drop_components_below_mm3: float = 0.0,
    peak_working_memory_mb: int = 1024,
) -> Mesh:
    """Surfacenets at an arbitrary target grid, slab-streaming the K-channel work.

    Memory-bounded — no full upsampled K-channel volume is materialised.
    Bitwise-equivalent to ``surfacenets_logits`` called on the would-be
    fully-resampled prediction (when the latter would fit).
    """
    src_logits_mx = prediction.data
    src_spacing = tuple(float(s) for s in prediction.geometry.spacing_zyx)
    tgt_shape = tuple(int(n) for n in target_geometry.shape_zyx)
    tgt_spacing = tuple(float(s) for s in target_geometry.spacing_zyx)
    schema = prediction.schema

    Z_o, Y_o, X_o = tgt_shape
    if Z_o < 2 or Y_o < 2 or X_o < 2:
        return Mesh.empty(target_geometry, schema)

    # Pass 1: slab-stream the K-channel work → full-grid reduced state.
    reduced = _slab_stream_reduced(
        src_logits_mx, tgt_shape, src_spacing, tgt_spacing,
        peak_working_memory_mb=peak_working_memory_mb,
    )
    labels       = reduced["labels"]
    margin_vol   = reduced["margin"]
    x_crossed    = reduced["x_crossed"]
    x_t          = reduced["x_t"]
    y_crossed    = reduced["y_crossed"]
    y_t          = reduced["y_t"]
    z_crossed    = reduced["z_crossed"]
    z_t          = reduced["z_t"]
    saddle_flips = reduced["saddle_flips"]

    # Apply confidence_threshold (operates on full-grid labels + margin).
    if confidence_threshold > 0.0:
        # _suppress_low_confidence_blobs originally consumes K-channel logits to
        # recompute margin if not provided; we already have margin from pass 1.
        labels = _suppress_low_confidence_blobs(
            labels, None, float(confidence_threshold),
            precomputed_margin=margin_vol,
        )

    if drop_components_below_mm3 > 0.0:
        from .postprocessing import remove_small_components
        labels = remove_small_components(
            labels, tgt_spacing,
            min_volume_mm3=float(drop_components_below_mm3),
            in_place=False,
        ).astype(np.int32, copy=False)

    # If labels were modified (confidence_threshold relabel OR cc3d-style
    # small-component drop), the per-axis crossed booleans computed in
    # Pass 1 are now STALE. They reflect the original labels, not the
    # post-relabel ones, so changes (e.g., a spike voxel demoted to
    # background) wouldn't actually shrink the mesh — we'd still emit
    # quads at the old crossings.
    #
    # Recompute the boolean crossed masks from the new labels. The t
    # values stay stale for edges whose label *pair* changed (the
    # interpolation was done with the original L0/L1), but the topology
    # is now correct. For the spike-removal case this is the right
    # trade-off: surfaces around relabeled noise voxels go away cleanly,
    # at the cost of sub-voxel vertex-placement error on edges whose
    # pair shifted (typically negligible — small components are dropped
    # to background, t is irrelevant for newly-uncrossed edges).
    if confidence_threshold > 0.0 or drop_components_below_mm3 > 0.0:
        x_crossed = labels[:, :, :-1] != labels[:, :, 1:]
        y_crossed = labels[:, :-1, :] != labels[:, 1:, :]
        z_crossed = labels[:-1, :, :] != labels[1:, :, :]

    # Apply spike-mask gate to crossings post-hoc.
    if confidence_margin > 0.0:
        spike_mask = _compute_spike_mask(
            labels, margin_vol, float(confidence_margin),
        )
        _apply_spike_mask_to_crossings(x_crossed, y_crossed, z_crossed, spike_mask)
        # Recompute saddle_flips against the now-smaller crossed set.
        # (Saddle face sums computed in pass 1 are fine — they describe
        # the logit-bilinear direction; what changes is whether the cell
        # is a true 4-edges-crossed saddle. We just need the gate. Most
        # straightforward: rebuild saddle_flips from the masked crossings.)
        # For simplicity here, rebuild from the masked crossings via
        # the saddle_flips bitfield ANDed with all-4-face-edges-crossed.
        cell_crossed_mask = _per_cell_crossed_mask(x_crossed, y_crossed, z_crossed)
        for face_idx in range(6):
            bits = _FACE_EDGE_BITS[face_idx]
            face_all_crossed = (cell_crossed_mask & np.uint16(bits)) == np.uint16(bits)
            # Clear bit `face_idx` of saddle_flips where face is not fully crossed.
            keep_face = face_all_crossed
            saddle_flips = np.where(
                keep_face,
                saddle_flips,
                saddle_flips & np.uint8(~(1 << face_idx) & 0xFF),
            ).astype(np.uint8)

    # Pass 2: cell case-table dispatch + dual vertex placement.
    edge_comp, n_comp, comp_pairs = _cell_components(
        labels, None, x_crossed, y_crossed, z_crossed,
        saddle_flips=saddle_flips,
    )
    cell_to_vertex, points = _cell_dual_vertices(
        edge_comp, n_comp, x_t, y_t, z_t,
    )

    # Pass 3: vertex refine + normals using SOURCE logits (no upsampled volume).
    normals = None
    if project_to_surface or emit_normals:
        points, normals = _gradient_refine_at_source(
            points, cell_to_vertex, n_comp, comp_pairs,
            src_logits_mx, src_spacing, tgt_spacing,
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
        geometry=target_geometry,
        schema=schema,
        normals=normals,
    )


__all__ = ["surfacenets_logits", "surfacenets_logits_at_target"]
