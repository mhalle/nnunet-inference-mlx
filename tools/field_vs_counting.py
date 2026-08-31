"""Measure volume and surface area from the margin field, and score it against counting.

The claim under test is that a structure's volume and area are integrals of the ranked
field - ``V = int H(m)`` and ``A = int delta(m) |grad m|`` - and can therefore be
evaluated on the interpolant directly, without ever rasterizing it to a mask. The
baseline is what `nnseg.statistics` does today: count voxels whose winner is `c`, and
(for area) count the faces between them and everything else.

The integrator here is a single sweep over the cells BETWEEN voxel centers. A cell whose
eight corners agree in sign is full or empty and contributes exactly; a cell that
straddles gets the plane through it that the corner values imply, and the volume and
area of a plane cutting a box are elementary::

    V(a) = sum_S (-1)^|S| max(a - sum S, 0)^3 / (6 abc)      inside fraction
    A(a) = |grad| * sum_S (-1)^|S| max(a - sum S, 0)^2 / (2 abc)

over subsets S of the three normal components, which is the Scardovelli-Zaleski form.
Both come from one expression - the second is the first's derivative in the level, which
is the co-area formula written for a box - so the two numbers cannot disagree about where
the surface is. No mesher, no dependency, and only the straddling cells do any work:
about 1.5 % of a volume at 1.5 mm, which is the whole point of not rasterizing.

`refine` subdivides by trilinear-upsampling the field, which is exactly subdividing the
cells of the same interpolant. Measured, it buys NOTHING: a sphere at 1.5 mm sits at
-0.282 % of its volume at refine 1 and -0.281 % at refine 4, for 67x the time. That is
the answer to "would better quadrature help" - no, because the error is not quadrature
error. A C0 trilinear interpolant is second-order and no rule inside its cells can be
better than the cells are; the way past it is a smoother interpolant (cubic B-spline),
not a finer one. It is left in as the diagnostic that establishes this, defaulted off.

usage: uv run python tools/field_vs_counting.py [--quick]
"""
import sys

import numpy as np
import torch

from nnseg import phantoms as ph
from nnseg import ranked
from nnseg.grid import Grid

CORNERS = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]


def _cube_cut(n_abs, alpha):
    """``(inside fraction, section area / |n|)`` for ``n . x = alpha`` in a unit cube,
    ``n`` positive. The second is the first's derivative in ``alpha`` - area and volume
    are the same question at two orders, which is the co-area formula written for a box.

    The general form divides by ``6abc``, so it goes to pieces exactly where a medical
    image spends most of its surface: a face perpendicular to an axis has two vanishing
    components, and differencing cubes over a denominator of ``1e-18`` returns noise
    scaled by the field's units - which is why the first version of this got the sphere
    right and the box off by 80 %. The degenerate cases are limits, not singularities, so
    they are taken as limits: a plane through only one axis cuts a slab, a plane through
    two cuts a prism, and both have elementary answers. Branch on the SORTED components
    so the test is on how flat the plane is, not on which axis it happens to face.
    """
    a = -np.sort(-np.asarray(n_abs), axis=0)          # descending: A >= B >= C >= 0
    A, B, C = a[0], a[1], a[2]
    S = A + B + C
    alpha = np.clip(alpha, 0.0, S)
    tiny = np.finfo(np.float64).tiny
    tol = 1e-6                                        # relative to A, the largest

    def safe(x):
        return np.where(x > tiny, x, 1.0)

    # 1D: the plane faces one axis; the cut is a slab
    v1 = alpha / safe(A)
    d1 = np.where((alpha > 0) & (alpha < A), 1.0 / safe(A), 0.0)

    # 2D: one component negligible; the cut is a prism, constant along the third axis
    t2 = ((alpha, 1), (alpha - A, -1), (alpha - B, -1), (alpha - A - B, 1))
    v2 = sum(sg * np.maximum(t, 0.0) ** 2 for t, sg in t2) / safe(2.0 * A * B)
    d2 = sum(sg * np.maximum(t, 0.0) for t, sg in t2) / safe(A * B)

    t3 = ((alpha, 1), (alpha - A, -1), (alpha - B, -1), (alpha - C, -1),
          (alpha - A - B, 1), (alpha - A - C, 1), (alpha - B - C, 1), (alpha - S, -1))
    v3 = sum(sg * np.maximum(t, 0.0) ** 3 for t, sg in t3) / safe(6.0 * A * B * C)
    d3 = sum(sg * np.maximum(t, 0.0) ** 2 for t, sg in t3) / safe(2.0 * A * B * C)

    flat2, flat1 = C <= tol * A, B <= tol * A
    v = np.where(flat1, v1, np.where(flat2, v2, v3))
    d = np.where(flat1, d1, np.where(flat2, d2, d3))
    return np.clip(v, 0.0, 1.0), np.maximum(d, 0.0)


def measure_field(margin, spacing, *, refine: int = 1):
    """``(volume_mm3, area_mm2)`` for ``{margin > 0}`` by integrating the interpolant.

    ``margin`` is sampled at voxel CENTERS, so the integration domain is the box those
    centers span - which is the right domain, and is not the same as the union of the
    voxels. A body that reaches the array edge would be clipped by that; the phantoms
    refuse to be sampled that close, and real structures are inside an envelope.
    """
    m = np.asarray(margin, dtype=np.float64)
    sp = np.asarray(spacing, dtype=np.float64)
    if refine > 1:
        t = torch.from_numpy(m)[None, None]
        size = tuple(int((n - 1) * refine + 1) for n in m.shape)
        m = torch.nn.functional.interpolate(t, size=size, mode="trilinear",
                                            align_corners=True)[0, 0].numpy()
        sp = sp / refine

    cell = float(sp.prod())
    corners = [m[i:m.shape[0] - 1 + i, j:m.shape[1] - 1 + j, k:m.shape[2] - 1 + k]
               for i, j, k in CORNERS]
    stack = np.stack(corners)
    inside = stack > 0
    n_in = inside.sum(0)
    straddle = (n_in > 0) & (n_in < 8)

    volume = float((n_in == 8).sum()) * cell
    if not straddle.any():
        return volume, 0.0

    # psi = -margin, so "inside" is psi < 0 and the cut is a lower set
    psi = [-c[straddle] for c in stack]
    axis_n = []
    for ax in range(3):                       # mean over the far face minus the near face
        far = [p for p, cn in zip(psi, CORNERS) if cn[ax] == 1]
        near = [p for p, cn in zip(psi, CORNERS) if cn[ax] == 0]
        axis_n.append(sum(far) / 4.0 - sum(near) / 4.0)
    n = np.stack(axis_n)                      # psi per cell, index space
    centre = sum(psi) / 8.0

    a = np.abs(n)
    alpha = 0.5 * a.sum(0) - centre           # the sign folding collapses to exactly this
    frac, dfrac = _cube_cut(a, alpha)
    grad = np.linalg.norm(n / sp[:, None], axis=0)        # physical |grad psi|

    volume += float(frac.sum()) * cell
    area = float((grad * dfrac).sum()) * cell
    return volume, area


def measure_counting(margin, spacing):
    """``(volume_mm3, area_mm2)`` the way a rasterized labelmap gives them up: count the
    voxels, then count the faces between inside and outside. The volume is what
    `nnseg.statistics` computes today; the area is the usual companion, and is the one
    that does not converge - a staircase has more surface than the surface it stands for,
    by a factor that depends on how the body happens to sit against the axes."""
    sp = np.asarray(spacing, dtype=np.float64)
    inside = np.asarray(margin) > 0
    volume = float(inside.sum()) * float(sp.prod())
    area = 0.0
    for ax in range(3):
        face = float(sp.prod() / sp[ax])
        d = np.diff(inside.astype(np.int8), axis=ax)
        area += float(np.count_nonzero(d)) * face
    return volume, area


# -- the evaluation -------------------------------------------------------------

def _grid(extent_mm, spacing):
    n = int(round(extent_mm / spacing))
    return Grid(shape=(n,) * 3, spacing=(spacing,) * 3, origin=(-(n - 1) * spacing / 2,) * 3)


def _row(label, got, truth):
    v, a = got
    return (f"  {label:<22s} {v:11.1f} {100*(v/truth[0]-1):+7.2f} %   "
            f"{a:11.1f} {100*(a/truth[1]-1):+7.2f} %")


def bodies():
    return [("sphere r=20", ph.sphere(20.0)),
            ("ellipsoid 12/18/26", ph.ellipsoid((12.0, 18.0, 26.0))),
            ("torus 26/8", ph.torus(26.0, 8.0)),
            ("shell 20/16", ph.shell(20.0, 16.0)),
            ("box 14^3", ph.box((14.0, 14.0, 14.0))),
            ("rounded box 10+6", ph.rounded_box((10.0, 10.0, 10.0), 6.0)),
            ("star r=20 m=4", ph.star(20.0, 0.15, 4))]


def accuracy(spacing=1.5, refine=1):
    print(f"\n=== accuracy at {spacing} mm (refine={refine}) "
          f"{'=' * 30}\n{'':24s}{'volume mm3':>13s}{'err':>9s}     {'area mm2':>13s}{'err':>9s}")
    for label, b in bodies():
        p = ph.Phantom((b,))
        g = _grid(96.0, spacing)
        truth = (b.volume_mm3, b.area_mm2)
        m = ph.margins(p, g)[1]
        print(f"\n{label}   truth {truth[0]:.1f} mm3 / {truth[1]:.1f} mm2")
        print(_row("counting", measure_counting(m, g.spacing), truth))
        print(_row("field (analytic)", measure_field(m, g.spacing, refine=refine), truth))
        code = ranked.encode(ph.logits(p, g), depth=2)
        print(_row("field (through store)",
                   measure_field(ranked.margin(code, 1), g.spacing, refine=refine), truth))


def stability(spacing=1.5, refine=1):
    """The invariance tests: nothing about the geometry changes, so nothing about the
    measurement should either. Truth is constant across every row."""
    b = ph.sphere(20.0)
    p, g = ph.Phantom((b,)), _grid(96.0, spacing)
    print(f"\n=== stability at {spacing} mm {'=' * 46}")
    for name, kwargs in (("sub-voxel shift", [{"offset_mm": (k * spacing / 8,) * 3} for k in range(8)]),
                         ("rotation", [{"rotation": ph.rotation_zyx(t, t / 2, t / 3)}
                                       for t in np.linspace(0, np.pi / 2, 8)])):
        cv, ca, fv, fa = [], [], [], []
        for kw in kwargs:
            m = ph.margins(p, g, **kw)[1]
            v, a = measure_counting(m, g.spacing); cv.append(v); ca.append(a)
            v, a = measure_field(m, g.spacing, refine=refine); fv.append(v); fa.append(a)
        for what, cs, fs, t in (("volume", cv, fv, b.volume_mm3), ("area", ca, fa, b.area_mm2)):
            sp_c, sp_f = np.ptp(cs) / t, np.ptp(fs) / t
            print(f"  {name:<16s} {what:<7s} counting spread {100*sp_c:6.2f} %  "
                  f"bias {100*(np.mean(cs)/t-1):+6.2f} %   |   field spread {100*sp_f:6.2f} %  "
                  f"bias {100*(np.mean(fs)/t-1):+6.2f} %")


def convergence(refine=1):
    print(f"\n=== convergence (sphere r=20, refine={refine}) {'=' * 33}")
    b = ph.sphere(20.0); p = ph.Phantom((b,))
    print(f"  {'spacing':>8s} {'count vol':>10s} {'field vol':>10s}   "
          f"{'count area':>10s} {'field area':>10s}")
    for spacing in (3.0, 2.0, 1.5, 1.0, 0.75):
        g = _grid(96.0, spacing)
        m = ph.margins(p, g)[1]
        cv, ca = measure_counting(m, g.spacing)
        fv, fa = measure_field(m, g.spacing, refine=refine)
        print(f"  {spacing:8.2f} {100*(cv/b.volume_mm3-1):+9.2f} % {100*(fv/b.volume_mm3-1):+9.2f} %   "
              f"{100*(ca/b.area_mm2-1):+9.2f} % {100*(fa/b.area_mm2-1):+9.2f} %")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    accuracy()
    if not quick:
        stability()
        convergence()
