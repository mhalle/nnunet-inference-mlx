"""Volume and surface area from the margin field, without rasterizing it.

Both quantities are integrals of the field rather than properties of a mask::

    V = int H(m) dx                  A = int delta(m) |grad m| dx

so they can be evaluated on the interpolant the store already defines. What that
buys is not speed - counting an existing labelmap is free - but an answer that does
not depend on where the geometry happened to land against the grid.

Measured against phantoms with closed-form truth (``nnseg.phantoms``, 1.5 mm):

    surface area   counting +39 to +54 %       field  -0.3 to +0.1 %
    volume         counting -10.3 to +4.2 %    field  -0.6 to -0.1 %

Counting's area does not converge - 4x refinement on a sphere leaves it at +50.4,
+50.9, +49.3, +50.9, +50.7 % - because a staircase genuinely has more surface than
the surface it stands for, by a factor that depends on how the body sits against the
axes. Face counting is not a coarse estimate of area; it is an estimate of something
else. Volume counting is closer but erratic, and saws by 1.1 % of a sphere under pure
sub-voxel translation where the field moves by 0.00 %.

The method is one sweep over the cells BETWEEN voxel centers. A cell whose eight
corners agree in sign is full or empty and contributes exactly; a cell that straddles
gets the plane its corner values imply, and the volume and area of a plane cutting a
box are elementary. Area is the volume expression differentiated in the level - the
co-area formula written for a box - so the two cannot disagree about where the surface
is. Only straddling cells do any work: 5-29 % of a real organ's bounding box, which is
the point of not rasterizing.

What this does NOT fix, and what would:

*Sharp creases read about 4.5 % low in area.* Any smooth interpolant rounds a corner
(measured on a cube; volume is unaffected at -0.28 %, the same as every other body).
Anatomy has few true edges, but a structure with a real crease will under-report - and
subdividing makes it WORSE, not better, since the finer interpolant rounds more.

*Accuracy stops at the interpolant, not the quadrature.* Subdividing the cells moves a
sphere from -0.282 % to -0.281 % for 67x the time. A C0 trilinear field is
second-order and no rule inside its cells can beat the cells; the way past it is a
smoother interpolant (cubic B-spline), not a finer rule. Adaptive subdivision would be
wasted work and is deliberately absent.

*Quantization biases area low, not just noisily.* See :func:`volume_area`.

*Area is scale-dependent by nature.* It is a first-derivative functional, so it
depends on the grid it was measured on in a way volume does not - a ripple at fixed
amplitude moves area 5.6 % across spatial frequencies while volume moves 0.5 %. An
area must always be reported with its spacing. Volume may be compared across grids;
area may not.
"""
from __future__ import annotations

import numpy as np

__all__ = ["counted_volume_area", "volume_area"]

_CORNERS = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
# psi = c + n . (p - 1/2) at the eight corners. Orthogonal, so the unweighted solve is
# the face-mean difference the fast path uses - the weighted fit below generalizes it.
_X = np.array([[1.0, i - 0.5, j - 0.5, k - 0.5] for i, j, k in _CORNERS])
_XTX = _X.T @ _X                                  # diag(8, 2, 2, 2)
_RIDGE = 0.02                                     # conditioning only; see _censored_fit


def _cube_cut(n_abs, alpha):
    """``(inside fraction, section area / |n|)`` for ``n . x = alpha`` in a unit cube.

    The general form divides by ``6abc``, which goes to pieces exactly where a medical
    image spends most of its surface: a face perpendicular to an axis has two vanishing
    components, and differencing cubes over a denominator of 1e-18 returns noise scaled
    by the field's units. The degenerate cases are limits, not singularities - a plane
    through one axis cuts a slab, through two a prism - so they are taken as limits, and
    the branch is on the SORTED components, i.e. on how flat the plane is rather than on
    which axis it happens to face.
    """
    a = -np.sort(-np.asarray(n_abs), axis=0)          # descending: A >= B >= C >= 0
    A, B, C = a[0], a[1], a[2]
    S = A + B + C
    alpha = np.clip(alpha, 0.0, S)
    tiny = np.finfo(np.float64).tiny
    tol = 1e-6                                        # relative to A, the largest

    def safe(x):
        return np.where(x > tiny, x, 1.0)

    v1 = alpha / safe(A)                              # slab
    d1 = np.where((alpha > 0) & (alpha < A), 1.0 / safe(A), 0.0)

    t2 = ((alpha, 1), (alpha - A, -1), (alpha - B, -1), (alpha - A - B, 1))
    v2 = sum(s * np.maximum(t, 0.0) ** 2 for t, s in t2) / safe(2.0 * A * B)
    d2 = sum(s * np.maximum(t, 0.0) for t, s in t2) / safe(A * B)

    t3 = ((alpha, 1), (alpha - A, -1), (alpha - B, -1), (alpha - C, -1),
          (alpha - A - B, 1), (alpha - A - C, 1), (alpha - B - C, 1), (alpha - S, -1))
    v3 = sum(s * np.maximum(t, 0.0) ** 3 for t, s in t3) / safe(6.0 * A * B * C)
    d3 = sum(s * np.maximum(t, 0.0) ** 2 for t, s in t3) / safe(2.0 * A * B * C)

    flat2, flat1 = C <= tol * A, B <= tol * A
    v = np.where(flat1, v1, np.where(flat2, v2, v3))
    d = np.where(flat1, d1, np.where(flat2, d2, d3))
    return np.clip(v, 0.0, 1.0), np.maximum(d, 0.0)


def _censored_fit(psi, clip, steps=2):
    """Plane fit that reads a saturated corner as an inequality, not a measurement.

    A cell's corners reach half a cell diagonal from the surface. Wherever the margin
    climbs faster than ``clip`` over that distance the far corners are AT the clip, and
    the plain fit averages them in as if they were values - which flattens the plane and
    loses area. Real TotalSegmentator margins run 3-7 logits/mm against a 2.6 mm
    diagonal at 1.5 mm, so 30-95 % of straddling cells have such a corner and the loss
    is 2-5 % of the surface. This is the single largest error in the module.

    Dropping those corners is not enough on its own: ``|psi| >= clip`` is information,
    and a body with flat faces (a box, at a narrow band) has whole corner planes
    censored, where an unconstrained fit rotates the normal freely and does far worse
    than doing nothing. So the drop is one step of a censored regression - fit without
    them, then reactivate any whose prediction lands back inside the band or on the
    wrong side, which is exactly the constraint they carry. Two steps suffice; measured
    against phantom truth over the band-to-diagonal range the real data spans, this
    takes smooth bodies from -5.2 % to -1.0 % at the worst ratio and -3.6 % to -0.4 % in
    the middle, and leaves a cube bit-for-bit where it was.
    """
    censored = np.abs(psi) >= clip - 1e-3
    w = (~censored).astype(np.float64)
    theta = None
    for _ in range(steps):
        M = np.einsum('in,ia,ib->nab', w, _X, _X) + _RIDGE * _XTX
        rhs = np.einsum('in,ia,in->na', w, _X, psi) + _RIDGE * (_X.T @ psi).T
        theta = np.linalg.solve(M, rhs[..., None])[..., 0]
        pred = np.einsum('ia,na->in', _X, theta)
        binding = censored & ((np.abs(pred) < clip) | (np.sign(pred) != np.sign(psi)))
        w = np.where(censored, binding.astype(np.float64), 1.0)
    return theta[:, 0], theta[:, 1:].T


def _sweep(block, sp, clip=None):
    """``(volume, area)`` over the cells of one contiguous block, in mm."""
    cell = float(np.prod(sp))
    stack = np.stack([block[i:block.shape[0] - 1 + i,
                            j:block.shape[1] - 1 + j,
                            k:block.shape[2] - 1 + k] for i, j, k in _CORNERS])
    n_in = (stack > 0).sum(0)
    volume = float((n_in == 8).sum()) * cell
    straddle = (n_in > 0) & (n_in < 8)
    if not straddle.any():
        return volume, 0.0

    psi = [-c[straddle] for c in stack]               # inside is psi < 0: a lower set
    if clip is None:
        n = np.stack([sum(p for p, cn in zip(psi, _CORNERS) if cn[ax] == 1) / 4.0
                      - sum(p for p, cn in zip(psi, _CORNERS) if cn[ax] == 0) / 4.0
                      for ax in range(3)])            # psi per cell, index space
        centre = sum(psi) / 8.0
    else:
        centre, n = _censored_fit(np.stack(psi).astype(np.float64), float(clip))
    a = np.abs(n)
    alpha = 0.5 * a.sum(0) - centre                   # the sign folding collapses to this
    frac, dfrac = _cube_cut(a, alpha)
    grad = np.linalg.norm(n / sp[:, None], axis=0)    # physical |grad psi|
    return volume + float(frac.sum()) * cell, float((grad * dfrac).sum()) * cell


def volume_area(margin, spacing, *, clip=None, slab: int = 16) -> tuple[float, float]:
    """``(volume_mm3, area_mm2)`` for ``{margin > 0}``, integrating the interpolant.

    ``margin`` is a signed field sampled at voxel CENTERS - :func:`nnseg.ranked.margin`
    for one class, or :func:`nnseg.ranked.decode_groups` for a union. Zero is the
    surface and the magnitude only has to be monotone through it, so clip and the uint8
    support cost the VOLUME nothing measurable (<= 0.03 % on every phantom).

    The CLIP is not free for area, and the uint8 quantization has nothing to do with it
    (measured: clipping alone reproduces the whole loss to three decimals - round-to-
    nearest is noise, and noise on a surface adds area rather than removing it). Pass
    ``clip`` to correct it; see :func:`_censored_fit` for what goes wrong and why the
    obvious repair is worse than the disease.

    The integration domain is the box the voxel centers span, which is half a voxel
    inside the array. A structure touching the array edge is therefore cut - real ones
    sit inside an envelope, and the phantoms refuse to be sampled that close.

    **Pass ``clip``** - ``code.meta["clip"]`` - whenever the field came out of a store.
    Without it the saturated corners of straddling cells are read as values rather than
    as bounds, and area comes out 1-5 % low on real margin gradients (see
    :func:`_censored_fit`). It is not the default only because a field that was never
    clipped has no such corners and must not have them invented.

    Cropped to the structure's bounding box and streamed in z-slabs, because the
    obvious dense version allocates eight float64 copies of the volume: 3.3 GB for a
    473 x 333 x 333 part, which is how the first version of this died.
    """
    m = np.asarray(margin)
    sp = np.asarray(spacing, dtype=np.float64)
    if m.ndim != 3:
        raise ValueError(f"margin must be (Z, Y, X); got shape {m.shape}")
    if sp.shape != (3,) or (sp <= 0).any():
        raise ValueError(f"spacing must be 3 positive values (Z, Y, X); got {spacing!r}")

    inside = m > 0
    if not inside.any():
        return 0.0, 0.0
    # every straddling cell has an inside corner, so one voxel of margin catches them all
    idx = [np.flatnonzero(inside.any(axis=tuple(j for j in range(3) if j != ax)))
           for ax in range(3)]
    lo = [max(int(i[0]) - 1, 0) for i in idx]
    hi = [min(int(i[-1]) + 2, m.shape[ax]) for ax, i in enumerate(idx)]
    box = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64, copy=False)
    if any(n < 2 for n in box.shape):
        return 0.0, 0.0

    volume = area = 0.0
    for z0 in range(0, box.shape[0] - 1, max(1, int(slab))):
        z1 = min(z0 + max(1, int(slab)) + 1, box.shape[0])
        v, a = _sweep(box[z0:z1], sp, clip)
        volume += v
        area += a
    return volume, area


def counted_volume_area(margin, spacing) -> tuple[float, float]:
    """``(volume_mm3, area_mm2)`` the way a rasterized labelmap gives them up.

    Kept next to :func:`volume_area` rather than left implicit in a caller, because the
    two are meant to be reported together until one has been shown to be better on a
    real cohort - and because the area here is a trap worth naming: counting the faces
    between inside and outside overstates a smooth surface by about half, and does not
    improve when the grid does.
    """
    sp = np.asarray(spacing, dtype=np.float64)
    inside = np.asarray(margin) > 0
    volume = float(inside.sum()) * float(np.prod(sp))
    area = sum(float(np.count_nonzero(np.diff(inside.astype(np.int8), axis=ax)))
               * float(np.prod(sp) / sp[ax]) for ax in range(3))
    return volume, area
