"""Analytic phantoms: geometry with known volume and area, expressed as logits.

Measuring a structure from the ranked field (rather than from a rasterized labelmap)
needs a reference the raster cannot supply - a surface whose volume and area are known
in closed form, sampled on a grid we choose. A segmentation model cannot provide that:
run it on a synthetic image and the ground truth evaporates, because the network's
decision surface is not the surface that was drawn. So the geometry is turned into
logits directly. Nothing downstream can tell the difference: :func:`ranked.encode`
takes ``(K, Z, Y, X)`` floats and never asks where they came from.

The construction is one line. Give every class a signed distance ``d_c`` (negative
inside), background included, and set ``l_c = -(gradient / 2) * d_c``. Classes are
disjoint and background is their complement, so exactly one ``d_c`` is negative
anywhere and ``argmax`` is exactly the intended labelmap. At ``c``'s surface the
runner-up is whatever is on the other side, whose distance is ``-d_c`` there, so

    margin_c = l_c - max_{j != c} l_j = -gradient * d_c

- the margin field *is* the signed distance, scaled - and ``|grad margin| = gradient``
exactly. That is the point of the halving: both sides of a boundary move, so the
margin climbs at twice the rate of either logit, and naming the parameter after the
margin means the co-area integrand has a known value with no factor to remember.

Background must be a real class with a real distance, not a flat zero. A flat zero
loses to any body's interior but *beats* every other body there, so it captures the
runner-up slot throughout - internal walls stop competing, ``margin`` measures every
structure against background, and a partition phantom silently stops containing the
triple junctions it exists to provide. :class:`Phantom` therefore carries a
``background_sdf``; for disjoint bodies it defaults to the union's, ``-min_c d_c``,
which is only correct BECAUSE they are disjoint. Abutting bodies (:func:`sectors`)
must supply their own, since a shared wall is not a boundary of the union.

Two entry points, and the difference between them is the point:

:func:`margins` returns the analytic field, bypassing the encoder - for unit-testing
quadrature, where quantization would only be noise.

:func:`logits` returns what ``encode`` consumes - for testing the stored form, where
``clip`` truncation, ``depth`` truncation, the uint8 support and the margin/deficit
gauge distinction are all live and are exactly the things that could be wrong.

A body's ``sdf`` need not be a true distance function; only its zero set must be
exact. Real logits are not distance functions either, so the approximate ones here
(ellipsoid, star) are the more representative test, not the weaker one.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Body", "DEFAULT_GRADIENT", "Phantom", "box", "ellipsoid", "labels",
           "logits", "margins", "rotation_zyx", "rounded_box", "sample_sdf", "sectors",
           "shell", "sphere", "star", "torus"]

DEFAULT_GRADIENT = 4.0    # |grad margin| in logits per mm: clip 8 spans ~2 mm


@dataclass(frozen=True)
class Body:
    """One closed surface, with the truth it exists to supply.

    ``sdf`` maps points ``(..., 3)`` in mm - axis order (Z, Y, X), matching
    :class:`~nnseg.grid.Grid` - to a signed value, negative inside. ``bound_mm`` is a
    bounding-sphere radius about the origin; :func:`logits` refuses to sample a body
    the grid would clip, because a truncated body silently invalidates every number
    in this module.
    """

    name: str
    sdf: Callable[[np.ndarray], np.ndarray]
    volume_mm3: float
    area_mm2: float
    bound_mm: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    param: Callable | None = None
    param_u_periodic: bool = False

    def quadrature_truth(self) -> tuple[float, float]:
        """``(volume, area)`` recomputed from the parametrization, when there is one.

        The closed forms above are the truth; this is the second opinion that catches a
        transcription error in them. Where no elementary form exists (ellipsoid area,
        star) the two are the same number and the check is on the rule's convergence
        instead - see the tests.
        """
        if self.param is None:
            raise ValueError(f"{self.name!r} has no parametrization to integrate; its "
                             f"truth is elementary and is checked by limits instead.")
        return _surface_truth(self.param, u_periodic=self.param_u_periodic)


@dataclass(frozen=True)
class Phantom:
    """Bodies plus channel 0, which is always background.

    ``bodies`` must be pairwise disjoint. A partition of a region (:func:`sectors`)
    satisfies that too - the sectors share walls but no interior - and is the case
    worth testing, since the shared wall is where ``margin`` is kinked and where the
    per-class volumes stop summing to the whole. Such a phantom must set
    ``background_sdf``; see :meth:`background`.
    """

    bodies: tuple[Body, ...]
    name: str = "phantom"
    background_sdf: Callable[[np.ndarray], np.ndarray] | None = None

    def background(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance for channel 0, negative outside every body.

        The default is the union's, ``-min_c d_c``, which holds only while the bodies
        are disjoint AND separated - it reads a shared wall as a boundary of the union,
        which it is not. Anything that abuts passes its own.
        """
        if self.background_sdf is not None:
            return np.asarray(self.background_sdf(pts), dtype=np.float64)
        return -np.min([b.sdf(pts) for b in self.bodies], axis=0)

    @property
    def n_classes(self) -> int:
        return len(self.bodies) + 1

    @property
    def channel_names(self) -> list[str]:
        return ["background"] + [b.name for b in self.bodies]

    def truth(self) -> dict[str, dict[str, float]]:
        """Per-body volume and area in mm. Areas of touching bodies each count the
        shared wall, so ``sum(area)`` double-counts internal interfaces; volumes do
        not, and ``sum(volume)`` is the closure invariant worth asserting."""
        return {b.name: {"volume_mm3": b.volume_mm3, "area_mm2": b.area_mm2}
                for b in self.bodies}


# -- shapes ---------------------------------------------------------------------
# Points arrive as (..., 3) in (Z, Y, X). Bodies are built at the origin and placed
# by `center`; `sample_sdf` adds a global offset on top, so one body can be swept
# through sub-voxel translations without being rebuilt.

def _p(pts, center) -> np.ndarray:
    return np.asarray(pts, dtype=np.float64) - np.asarray(center, dtype=np.float64)


def sphere(radius: float, *, center=(0.0, 0.0, 0.0), name="sphere") -> Body:
    """Exact SDF, exact truth, constant curvature. The baseline everything else is
    read against."""
    R = float(radius)
    def param(u, v):
        return R * torch.stack([torch.sin(u) * torch.cos(v),
                                torch.sin(u) * torch.sin(v),
                                torch.cos(u) * torch.ones_like(v)])

    return Body(name, lambda q: np.linalg.norm(_p(q, center), axis=-1) - R,
                4.0 / 3.0 * math.pi * R ** 3, 4.0 * math.pi * R ** 2,
                R + float(np.linalg.norm(center)), tuple(center), param)


def ellipsoid(semi_axes, *, center=(0.0, 0.0, 0.0), name="ellipsoid") -> Body:
    """Volume in closed form; area by spectral quadrature (no elementary form exists).

    The SDF is the standard gradient-normalized bound: the zero set is exact, the
    gradient is only approximately unit. That is the realistic case - a network's
    margin is not a distance function either - and it is why area must not be
    inferred from ``|grad margin|`` alone.
    """
    a = np.asarray(semi_axes, dtype=np.float64)          # (Z, Y, X) semi-axes
    if a.shape != (3,) or (a <= 0).any():
        raise ValueError(f"semi_axes must be 3 positive values (Z, Y, X); got {semi_axes!r}")

    def sdf(q):
        p = _p(q, center)
        k0 = np.linalg.norm(p / a, axis=-1)
        k1 = np.linalg.norm(p / (a * a), axis=-1)
        return np.where(k0 == 0, -a.min(), (k0 - 1.0) * k0 / np.maximum(k1, 1e-300))

    def param(u, v):                                     # ordinary XYZ: V and A do not care
        return torch.stack([a[2] * torch.sin(u) * torch.cos(v),
                            a[1] * torch.sin(u) * torch.sin(v),
                            a[0] * torch.cos(u)])

    _, area = _surface_truth(param)
    return Body(name, sdf, 4.0 / 3.0 * math.pi * float(a.prod()), area,
                float(a.max()) + float(np.linalg.norm(center)), tuple(center), param)


def torus(major: float, minor: float, *, center=(0.0, 0.0, 0.0), name="torus") -> Body:
    """Exact SDF, exact truth, genus 1. The hole is what makes it worth having: a
    cell sweep that mishandles topology still gets a sphere right."""
    R, r = float(major), float(minor)
    if not 0 < r < R:
        raise ValueError(f"need 0 < minor < major; got minor={r}, major={R}")

    def sdf(q):
        p = _p(q, center)
        radial = np.hypot(p[..., 2], p[..., 1]) - R      # about the Z axis
        return np.hypot(radial, p[..., 0]) - r

    def param(u, v):                                     # both directions periodic
        return torch.stack([(R + r * torch.cos(v)) * torch.cos(u),
                            (R + r * torch.cos(v)) * torch.sin(u),
                            r * torch.sin(v)])

    return Body(name, sdf, 2.0 * math.pi ** 2 * R * r * r, 4.0 * math.pi ** 2 * R * r,
                R + r + float(np.linalg.norm(center)), tuple(center), param,
                param_u_periodic=True)


def shell(outer: float, inner: float, *, center=(0.0, 0.0, 0.0), name="shell") -> Body:
    """A wall of controllable thickness - the regime where voxel counting fails and
    where cortex, vessel wall and bowel wall live. Drive ``outer - inner`` below the
    spacing to find where a measurement stops meaning anything."""
    Ro, Ri = float(outer), float(inner)
    if not 0 <= Ri < Ro:
        raise ValueError(f"need 0 <= inner < outer; got inner={Ri}, outer={Ro}")

    def sdf(q):
        rho = np.linalg.norm(_p(q, center), axis=-1)
        return np.maximum(rho - Ro, Ri - rho)

    return Body(name, sdf, 4.0 / 3.0 * math.pi * (Ro ** 3 - Ri ** 3),
                4.0 * math.pi * (Ro ** 2 + Ri ** 2),      # both surfaces bound the body
                Ro + float(np.linalg.norm(center)), tuple(center))


def box(half_extents, *, center=(0.0, 0.0, 0.0), name="box") -> Body:
    """Exact SDF, exact truth, and curvature that is a distribution rather than a
    function. Adversarial on purpose: every smooth interpolant rounds the corners, so
    area comes out low by an amount worth measuring instead of discovering later.
    Sample it rotated as well - face counting is wrong by ~sqrt(2) on a 45-degree cube,
    and a field method that is not should be able to prove it."""
    h = np.asarray(half_extents, dtype=np.float64)
    if h.shape != (3,) or (h <= 0).any():
        raise ValueError(f"half_extents must be 3 positive values (Z, Y, X); got {half_extents!r}")

    def sdf(q):
        d = np.abs(_p(q, center)) - h
        return (np.linalg.norm(np.maximum(d, 0.0), axis=-1)
                + np.minimum(d.max(axis=-1), 0.0))

    return Body(name, sdf, float(8.0 * h.prod()),
                float(8.0 * (h[0] * h[1] + h[1] * h[2] + h[2] * h[0])),
                float(np.linalg.norm(h)) + float(np.linalg.norm(center)), tuple(center))


def rounded_box(half_extents, radius: float, *, center=(0.0, 0.0, 0.0),
                name="rounded_box") -> Body:
    """A box of half-extents ``h`` dilated by ``radius`` - so the outer half-extents
    are ``h + radius``, and both truths are elementary (slabs + quarter-cylinders +
    eighth-spheres). One continuous knob from box to sphere: as ``radius -> 0`` it is
    :func:`box`, as ``h -> 0`` it is :func:`sphere`, and in between it sweeps curvature
    without ever leaving closed form."""
    h = np.asarray(half_extents, dtype=np.float64)
    R = float(radius)
    if h.shape != (3,) or (h < 0).any() or R <= 0:
        raise ValueError(f"need 3 non-negative half_extents and positive radius; "
                         f"got {half_extents!r}, {radius!r}")
    inner = box(h, center=center)
    pq_qr_rp = float(h[0] * h[1] + h[1] * h[2] + h[2] * h[0])
    vol = (8.0 * float(h.prod()) + 8.0 * R * pq_qr_rp
           + 2.0 * math.pi * R * R * float(h.sum()) + 4.0 / 3.0 * math.pi * R ** 3)
    area = (8.0 * pq_qr_rp + 4.0 * math.pi * R * float(h.sum()) + 4.0 * math.pi * R * R)
    return Body(name, lambda q: inner.sdf(q) - R, vol, area,
                float(np.linalg.norm(h)) + R + float(np.linalg.norm(center)), tuple(center))


def star(radius: float, amplitude: float, mode: int, *, center=(0.0, 0.0, 0.0),
         name="star") -> Body:
    """A wiggly star body: ``r = R (1 + eps Re[((x+iy)/rho)^m])``, smooth on the sphere.

    The one phantom whose *point* is that area and volume behave differently.
    ``amplitude`` and ``mode`` set the ripple's height and spatial frequency
    independently, so a sweep in ``mode`` at fixed ``amplitude`` changes the area a
    lot and the volume barely at all - which is the scale dependence of area, made
    measurable rather than argued about. Push ``mode`` until the ripple is near the
    grid's Nyquist and the area estimate should visibly stop converging.

    Truth by Gauss-Legendre x trapezoid on the parametrization: spectrally accurate
    for an analytic integrand, and the test asserts its own convergence.
    """
    R, eps, m = float(radius), float(amplitude), int(mode)
    if not 0 <= eps < 1:
        raise ValueError(f"amplitude must be in [0, 1) to keep the body star-shaped; got {eps}")
    if m < 1:
        raise ValueError(f"mode must be >= 1 to stay smooth at the poles; got {m}")

    def sdf(q):
        p = _p(q, center)
        rho = np.linalg.norm(p, axis=-1)
        w = (p[..., 2] + 1j * p[..., 1]) / np.maximum(rho, 1e-300)   # (x + iy) / rho
        f = np.where(rho == 0, 0.0, np.real(w ** m))
        return rho - R * (1.0 + eps * f)

    def param(u, v):
        r = R * (1.0 + eps * torch.sin(u) ** m * torch.cos(m * v))
        return torch.stack([r * torch.sin(u) * torch.cos(v),
                            r * torch.sin(u) * torch.sin(v),
                            r * torch.cos(u)])

    vol, area = _surface_truth(param)
    return Body(name, sdf, vol, area, R * (1.0 + eps) + float(np.linalg.norm(center)),
                tuple(center), param)


def sectors(n: int, radius: float, *, center=(0.0, 0.0, 0.0), name="sector") -> Phantom:
    """A ball cut into ``n`` wedges about the Z axis: the triple-junction phantom.

    Every wall is shared, and the axis is a junction line where ``margin`` is kinked
    and where per-class volumes computed from separate margin fields stop being a
    partition. ``sum(volume)`` is exactly the ball, which is the closure invariant;
    the residual it leaves is the error budget for that kink.
    """
    R = float(radius)
    if n < 2:
        raise ValueError(f"need at least 2 sectors; got {n}")
    ball = 4.0 / 3.0 * math.pi * R ** 3
    # cap share + two half-discs of radius R; correct at n = 2, where the half-ball's
    # single flat face is counted once from each side of the same wedge
    area = 4.0 * math.pi * R ** 2 / n + math.pi * R ** 2
    bodies = []
    for k in range(n):
        a, b = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        na = np.array([-math.sin(a), math.cos(a)])       # inward across the low wall
        nb = np.array([-math.sin(b), math.cos(b)])

        def sdf(q, na=na, nb=nb):
            p = _p(q, center)
            xy = np.stack([p[..., 2], p[..., 1]], axis=-1)     # (x, y)
            return np.maximum(np.linalg.norm(p, axis=-1) - R,
                              np.maximum(-(xy @ na), xy @ nb))

        bodies.append(Body(f"{name}{k}", sdf, ball / n, area,
                           R + float(np.linalg.norm(center)), tuple(center)))
    # the union of the sectors IS the ball; the walls between them are not its boundary
    return Phantom(tuple(bodies), name=f"{n}-sectors",
                   background_sdf=lambda q: R - np.linalg.norm(_p(q, center), axis=-1))


def rotation_zyx(az: float, ay: float, ax: float) -> np.ndarray:
    """Rotation matrix for points in (Z, Y, X) order, radians, applied Z then Y then X.

    :class:`~nnseg.grid.Grid` carries no direction cosines - orientation belongs to the
    caller's ``Frame`` - so orientation dependence is tested by rotating the *body*
    inside an axis-aligned grid, which probes the same thing and keeps the truth fixed.
    """
    cz, sz, cy, sy, cx, sx = (math.cos(az), math.sin(az), math.cos(ay),
                              math.sin(ay), math.cos(ax), math.sin(ax))
    Rz = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]], dtype=np.float64)
    return Rx @ Ry @ Rz


# -- sampling -------------------------------------------------------------------

def _points(grid, offset_mm, rotation) -> np.ndarray:
    """Voxel-center positions in mm, (Z, Y, X, 3), pulled back into the body frame."""
    idx = np.stack(np.meshgrid(*[np.arange(s, dtype=np.float64) for s in grid.shape],
                               indexing="ij"), axis=-1)
    pts = grid.index_to_mm(idx)
    if offset_mm is not None:
        pts = pts - np.asarray(offset_mm, dtype=np.float64)
    if rotation is not None:
        pts = pts @ np.asarray(rotation, dtype=np.float64)      # inverse of R @ p
    return pts


def sample_sdf(body: Body, grid, *, offset_mm=None, rotation=None) -> np.ndarray:
    """One body's signed distance on ``grid``, in mm, negative inside.

    ``offset_mm`` translates the body (sweep it over one voxel in eighths to see the
    sawtooth that voxel counting has and a field measure should not). ``rotation`` is
    a 3x3 matrix from :func:`rotation_zyx`; both leave the truth untouched, which is
    what makes them free invariance tests.
    """
    return body.sdf(_points(grid, offset_mm, rotation)).astype(np.float64)


def _check_fits(phantom: Phantom, grid, offset_mm) -> None:
    lo, hi = grid.extent_mm
    off = np.zeros(3) if offset_mm is None else np.asarray(offset_mm, dtype=np.float64)
    for b in phantom.bodies:
        c = np.asarray(b.center, dtype=np.float64) + off
        # bound_mm already includes |center|, so measure the ball about the offset origin
        if (c - (b.bound_mm - np.linalg.norm(b.center)) < lo).any() or \
           (c + (b.bound_mm - np.linalg.norm(b.center)) > hi).any():
            raise ValueError(
                f"body {b.name!r} (bound {b.bound_mm:.3f} mm about {tuple(c)}) leaves the "
                f"grid extent {tuple(lo)}..{tuple(hi)}. A clipped body makes every number "
                f"in Phantom.truth() wrong, so this is refused rather than sampled.")


def logits(phantom: Phantom, grid, *, gradient: float = DEFAULT_GRADIENT,
           offset_mm=None, rotation=None, noise: float = 0.0, seed: int = 0,
           dtype=torch.float32) -> torch.Tensor:
    """``(K, Z, Y, X)`` logits for ``phantom`` on ``grid`` - what ``encode`` consumes.

    Channel 0 is background, channel ``c + 1`` is body ``c``, and every one of them is
    ``-(gradient / 2) * d``. ``gradient`` is ``|grad margin|`` in logits per mm - the
    margin, not the logit, because that is the field a measurement reads and the halving
    is otherwise a factor waiting to be forgotten. Size it so the +/-clip band spans a
    voxel or two, matching how sharp a trained network actually is.

    It is also a free test: scaling every logit difference by one factor cannot move a
    zero level set, so a volume that depends on ``gradient`` is reporting a quantization
    artifact rather than a geometry.

    ``noise`` adds band-limited perturbation (smoothed Gaussian, in logits) - the tier
    that shows the asymmetry between the two measurements. Volume integrates an
    indicator and shrugs it off; area is a first-derivative functional and does not.
    """
    _check_fits(phantom, grid, offset_mm)
    pts = _points(grid, offset_mm, rotation)
    half = float(gradient) / 2.0
    planes = [-half * phantom.background(pts)]
    for b in phantom.bodies:
        planes.append(-half * b.sdf(pts).astype(np.float64))
    out = np.stack(planes)
    if noise:
        out = out + float(noise) * _band_limited(out.shape, seed)
    return torch.from_numpy(out).to(dtype)


def _band_limited(shape, seed: int, *, scale: int = 4) -> np.ndarray:
    """Unit-variance noise with no structure below ``scale`` voxels, numpy + torch only.

    Drawn on a grid coarsened by ``scale`` and interpolated back up, which is what
    band-limited means rather than an approximation of it - and it keeps this module a
    kernel leaf, with no scipy at module level or at call time.
    """
    rng = np.random.default_rng(seed)
    coarse = [max(2, -(-s // scale)) for s in shape[1:]]
    n = torch.from_numpy(rng.standard_normal((1, shape[0], *coarse)))
    n = torch.nn.functional.interpolate(n, size=tuple(shape[1:]), mode="trilinear",
                                        align_corners=False)[0].numpy()
    return n / max(float(n.std()), 1e-12)


def margins(phantom: Phantom, grid, *, gradient: float = DEFAULT_GRADIENT,
            offset_mm=None, rotation=None, clip: float | None = None) -> np.ndarray:
    """``(K, Z, Y, X)`` analytic margins - ``ranked.margin`` without the round trip.

    Computed from the definition, ``l_c - max_{j != c} l_j``, rather than from the
    identity it satisfies - so a phantom whose bodies are not really disjoint, or whose
    background distance is wrong, shows up as ``margins`` disagreeing with
    ``-gradient * sdf`` instead of quietly agreeing with a broken construction.

    Pass ``clip`` to saturate exactly as the stored form does, which is what a
    comparison against a decoded code should be held to.
    """
    lg = logits(phantom, grid, gradient=gradient, offset_mm=offset_mm,
                rotation=rotation, dtype=torch.float64).numpy()
    out = np.empty_like(lg)
    for c in range(lg.shape[0]):
        others = np.delete(lg, c, axis=0)
        out[c] = lg[c] - others.max(axis=0)
    return out if clip is None else np.clip(out, -float(clip), float(clip))


def labels(phantom: Phantom, grid, *, offset_mm=None, rotation=None) -> np.ndarray:
    """The intended labelmap: 0 background, ``c + 1`` inside body ``c``.

    By construction this is ``logits().argmax(0)`` exactly, which is what makes it a
    check on the encoder rather than a restatement of it.
    """
    pts = _points(grid, offset_mm, rotation)
    out = np.zeros(tuple(grid.shape), np.int64)
    for c, b in enumerate(phantom.bodies):
        out[b.sdf(pts) < 0] = c + 1
    return out


# -- truth by quadrature --------------------------------------------------------

def _surface_truth(param, *, nu: int = 256, nv: int = 256,
                   u_periodic: bool = False) -> tuple[float, float]:
    """``(volume, area)`` for a closed surface ``X(u, v)``, to ~1e-12.

    Gauss-Legendre over ``u`` in ``[0, pi]`` (or trapezoid when the direction is
    periodic, as on a torus) and trapezoid over ``v`` in ``[0, 2pi)``, which is
    spectrally accurate for an analytic periodic integrand. Area from
    ``|X_u x X_v|`` and volume from the divergence theorem ``V = (1/3) int X . n dS``
    - deliberately the *same* quadrature, so a parametrization error shows up as the
    two disagreeing rather than as a plausible wrong pair.

    Partials come from autograd: ``param`` is elementwise in ``u`` and ``v``, so one
    ``grad`` of the sum recovers the whole derivative field, and no shape's calculus
    has to be done by hand to add it here.
    """
    if u_periodic:
        u = np.arange(nu) * 2 * math.pi / nu
        wu = np.full(nu, 2 * math.pi / nu)
    else:
        x, w = np.polynomial.legendre.leggauss(nu)
        u, wu = 0.5 * math.pi * (x + 1.0), 0.5 * math.pi * w
    v = np.arange(nv) * 2 * math.pi / nv
    wv = np.full(nv, 2 * math.pi / nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    W = torch.from_numpy(wu[:, None] * wv[None, :])

    ut = torch.tensor(U, dtype=torch.float64, requires_grad=True)
    vt = torch.tensor(V, dtype=torch.float64, requires_grad=True)
    X = param(ut, vt)
    Xu = torch.stack([torch.autograd.grad(X[i].sum(), ut, retain_graph=True)[0]
                      for i in range(3)])
    Xv = torch.stack([torch.autograd.grad(X[i].sum(), vt, retain_graph=True)[0]
                      for i in range(3)])
    n = torch.cross(Xu, Xv, dim=0)
    with torch.no_grad():
        area = float((W * n.norm(dim=0)).sum())
        vol = abs(float((W * (X * n).sum(0)).sum()) / 3.0)   # orientation-agnostic
    return vol, area
