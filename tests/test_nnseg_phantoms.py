"""The phantoms, held to their own claims.

Nothing here measures the ranked field - the point is the opposite. A phantom is only
useful as a reference if its truth is right and its logits really do encode the
geometry that truth describes, so these tests check the reference itself: closed forms
against an independent quadrature, the construction's margin identity, and what
survives the encoder. Whatever measures volume and area from the field is graded
against this; this is graded against calculus.

No weights and no files - all synthetic, so it runs in the fast suite.
"""
import math
import unittest

import numpy as np

from nnseg import phantoms as ph
from nnseg import ranked
from nnseg.grid import Grid


def _grid(n=48, spacing=1.0):
    """Cube centered on the origin, so a body at the origin is centered in the grid."""
    half = (n - 1) * spacing / 2.0
    return Grid(shape=(n, n, n), spacing=(spacing,) * 3, origin=(-half,) * 3)


class TestTruth(unittest.TestCase):
    """The closed forms, against the parametric quadrature that does not share their
    algebra. Where a closed form exists this catches a transcription error in it; where
    one does not (ellipsoid area, star) the quadrature IS the truth, so what is checked
    is the rule's convergence instead."""

    def test_closed_forms_match_the_parametric_quadrature(self):
        for body in (ph.sphere(10.0), ph.torus(9.0, 3.0), ph.ellipsoid((3.0, 5.0, 8.0)),
                     ph.star(10.0, 0.15, 3)):
            with self.subTest(body.name):
                vol, area = body.quadrature_truth()
                self.assertAlmostEqual(vol / body.volume_mm3, 1.0, places=10)
                self.assertAlmostEqual(area / body.area_mm2, 1.0, places=10)

    def test_the_quadrature_has_converged_at_the_default_rule(self):
        """Doubling the rule must not move the answer, or the 'truth' is a resolution."""
        coarse = ph._surface_truth(ph.star(10.0, 0.2, 5).param, nu=128, nv=128)
        fine = ph._surface_truth(ph.star(10.0, 0.2, 5).param, nu=512, nv=512)
        for c, f in zip(coarse, fine):
            self.assertAlmostEqual(c / f, 1.0, places=11)

    def test_rounded_box_reaches_the_box_and_the_sphere(self):
        """The two limits pin the Steiner decomposition from both ends: no radius is a
        box, no box is a sphere. Between them the same formula sweeps curvature."""
        rb, bx = ph.rounded_box((5.0, 5.0, 5.0), 1e-7), ph.box((5.0, 5.0, 5.0))
        self.assertAlmostEqual(rb.volume_mm3 / bx.volume_mm3, 1.0, places=6)
        self.assertAlmostEqual(rb.area_mm2 / bx.area_mm2, 1.0, places=6)
        rb, sp = ph.rounded_box((1e-9, 1e-9, 1e-9), 4.0), ph.sphere(4.0)
        self.assertAlmostEqual(rb.volume_mm3 / sp.volume_mm3, 1.0, places=8)
        self.assertAlmostEqual(rb.area_mm2 / sp.area_mm2, 1.0, places=8)

    def test_sector_volumes_close_on_the_ball(self):
        """The closure invariant. Per-class volumes measured from separate margin fields
        will NOT close exactly - the residual is the error budget for the kink at the
        junction line - so the truth they are compared against has to close exactly."""
        for n in (2, 3, 5, 8):
            with self.subTest(n=n):
                sec = ph.sectors(n, 10.0)
                total = sum(b.volume_mm3 for b in sec.bodies)
                self.assertAlmostEqual(total / (4 / 3 * math.pi * 1000.0), 1.0, places=12)


class TestConstruction(unittest.TestCase):
    """The identity the whole module rests on: margin is the signed distance, scaled."""

    def test_argmax_of_the_logits_is_the_intended_labelmap(self):
        g, p = _grid(), ph.Phantom((ph.sphere(8.0), ph.torus(16.0, 3.0)))
        np.testing.assert_array_equal(ph.logits(p, g).argmax(0).numpy(), ph.labels(p, g))

    def test_the_zero_level_set_is_exactly_the_bodys_surface(self):
        """The one requirement that holds for every phantom without qualification:
        ``margin_c > 0`` exactly where ``d_c < 0``. Everything a measurement does is
        downstream of the surface being where the geometry says it is."""
        g = _grid(n=80, spacing=0.5)
        cases = {"disjoint": ph.Phantom((ph.sphere(6.0), ph.torus(14.0, 2.0))),
                 "partition": ph.sectors(4, 8.0)}
        for label, p in cases.items():
            with self.subTest(label):
                m = ph.margins(p, g)
                for c, b in enumerate(p.bodies, start=1):
                    np.testing.assert_array_equal(m[c] > 0, ph.sample_sdf(b, g) < 0)

    def test_margin_is_the_distance_scaled_by_the_gradient(self):
        """The identity the module advertises, checked against `margins`, which computes
        `l_c - max_{j != c} l_j` instead of assuming it. Exact everywhere for a body with
        an exact SDF, because the default background distance is then exactly `-d`."""
        g = _grid(n=80, spacing=0.5)
        for body in (ph.sphere(8.0), ph.torus(14.0, 3.0), ph.box((4.0, 6.0, 3.0)),
                     ph.shell(8.0, 5.0)):
            with self.subTest(body.name):
                p = ph.Phantom((body,))
                np.testing.assert_allclose(ph.margins(p, g, gradient=3.0)[1],
                                           -3.0 * ph.sample_sdf(body, g), atol=1e-9)

    def test_a_composite_sdf_holds_the_identity_only_near_its_own_surface(self):
        """A sector's SDF is `max` over its constraints: the zero set is exact, the
        distance is not. Past the surface's reach - the distance to the nearest edge or
        junction - the neighbour's own nearest constraint is a different wall, so `d_nb`
        stops being `-d_c` and the identity goes with it. Measured here it survives to
        0.75 mm off a rim and a junction line and breaks by 1.8 logits at 1.5 mm.

        Which is the honest scope of the whole construction: `margin = -gradient * d` is
        a property OF THE BOUNDARY. A level-set measurement only reads there, so this
        costs nothing - but anything that assumes `|grad margin|` is uniform across a
        whole band is assuming something no phantom here promises and no network does
        either."""
        g, sec = _grid(spacing=0.5), ph.sectors(4, 8.0)
        d = ph.sample_sdf(sec.bodies[0], g)
        got, want = ph.margins(sec, g, gradient=3.0)[1], -3.0 * d
        pts = ph._points(g, None, None)
        rho, axis = np.linalg.norm(pts, axis=-1), np.hypot(pts[..., 2], pts[..., 1])
        reach = (np.abs(d) < 0.75) & (rho < 7.0) & (axis > 2.0)
        self.assertGreater(reach.sum(), 500, "no voxels within reach of the surface")
        np.testing.assert_allclose(got[reach], want[reach], atol=1e-9)
        self.assertGreater(np.abs(got - want)[np.abs(d) < 1.5].max(), 1.0,
                           "the rim and the junction should break it; if they stopped, say so")

    def test_flat_background_would_swallow_the_runner_up_slot(self):
        """Why channel 0 needs a real distance. With background pinned at logit 0 it
        loses to a body's interior but beats every OTHER body there, so it takes the
        runner-up slot throughout the ball and the walls between sectors stop competing
        - the phantom keeps its labelmap and quietly loses its triple junction. Here the
        neighbour must win that slot, which is the thing a flat zero destroys."""
        g, sec = _grid(spacing=0.5), ph.sectors(4, 8.0)
        lg = ph.logits(sec, g, gradient=3.0).numpy()
        pts = ph._points(g, None, None)
        d0 = ph.sample_sdf(sec.bodies[0], g)
        deep = (np.abs(d0) < 0.6) & (np.linalg.norm(pts, axis=-1) < 6.0) & \
               (np.hypot(pts[..., 2], pts[..., 1]) > 3.0)      # on a wall, deep inside
        self.assertGreater(deep.sum(), 100, "no wall voxels selected")
        runner_up = np.argsort(-lg, axis=0)[1][deep]
        self.assertEqual((runner_up == 0).sum(), 0,
                         "background must not be the runner-up on an internal wall")

    def test_the_gradient_does_not_move_any_zero_level_set(self):
        """It rescales every logit difference by the same factor, so it cannot. This is
        the free invariance test on the measurement side too: a volume that depends on
        the gradient is reporting a quantization artifact, not a geometry."""
        g, p = _grid(), ph.Phantom((ph.sphere(8.0), ph.box((4.0, 6.0, 3.0), center=(0, 0, 16))))
        ref = ph.logits(p, g, gradient=1.0).argmax(0).numpy()
        for grad in (0.5, 4.0, 20.0):
            with self.subTest(gradient=grad):
                np.testing.assert_array_equal(
                    ph.logits(p, g, gradient=grad).argmax(0).numpy(), ref)

    def test_a_body_that_leaves_the_grid_is_refused(self):
        """A clipped body makes every number in truth() wrong while still producing a
        perfectly plausible field, so this fails loudly instead of quietly."""
        with self.assertRaisesRegex(ValueError, "leaves the grid extent"):
            ph.logits(ph.Phantom((ph.sphere(40.0),)), _grid(n=32))


class TestThroughTheEncoder(unittest.TestCase):
    """What survives the stored form. These are the ones that can actually fail: clip,
    depth, the uint8 support and the margin/deficit gauge are all live here."""

    def test_decoded_margin_matches_the_analytic_field_to_half_a_quantum(self):
        g, p, grad = _grid(), ph.Phantom((ph.sphere(8.0),)), 4.0
        code = ranked.encode(ph.logits(p, g, gradient=grad), depth=2)
        got = ranked.margin(code, 1)
        want = ph.margins(p, g, gradient=grad, clip=ranked.CLIP)[1]
        band = np.abs(want) < ranked.CLIP - 2.0          # away from the saturated floor
        err = np.abs(got - want)[band].max()
        self.assertLess(err, 0.5 * ranked.CLIP / ranked.SUPPORT_MAX * 1.01,
                        "round-to-nearest should cost at most half a quantum")

    def test_the_decoded_winner_is_still_the_intended_labelmap(self):
        g, p = _grid(), ph.Phantom((ph.sphere(8.0), ph.torus(16.0, 3.0)))
        code = ranked.encode(ph.logits(p, g), depth=3)
        np.testing.assert_array_equal(code.ranks[0].astype(np.int64) - 1, ph.labels(p, g))

    def test_depth_truncation_floors_live_classes_at_a_high_order_junction(self):
        """Eight sectors meet on the axis, so nine classes compete there and a depth-6
        store keeps six. The three it drops are NOT safely far behind - measured here
        they are within about a logit of winning - so they decode to the `-clip` floor
        and their surfaces are unrecoverable in a tube ~2.5 mm wide around the junction
        line, about 6 % of each class's near-surface voxels on this grid.

        That is where the stored form runs out, not a bug in it - and an eight-fold
        junction LINE is non-generic, so it is a stress case rather than a forecast. In
        general position at most four regions meet in 3D, and then only at isolated
        points (surfaces codim 1, triple curves codim 2, quadruple points codim 3); any
        perturbation splits an eight-fold line into a network of quadruple points.
        Measured on a real TotalSegmentator store (idc-torso1, 52 M voxels/part), depth
        6 saturates at 0.024 % of voxels and the deepest class it keeps there is already
        a median 7.59 logits behind - on the 8.0 clip. So `clip` binds before `depth`
        does, raising depth would add planes that are sentinel everywhere, and the
        surfaces themselves never leave ranks 0 and 1: at most two classes sit within a
        logit of the winner at 99.997 % of voxels, three at 0.003 %, four at none.

        Outside the tube recovery here is exact to the quantization, which is the other
        half of the claim and the half that holds in production.
        """
        g, sec, grad = _grid(spacing=0.5), ph.sectors(8, 8.0), 2.0
        code = ranked.encode(ph.logits(sec, g, gradient=grad), depth=6)
        want = ph.margins(sec, g, gradient=grad)
        axis = np.hypot(*[ph._points(g, None, None)[..., i] for i in (2, 1)])
        self.assertTrue((code.ranks[1:] == 0).any(), "depth too generous to truncate")

        quantum = ranked.CLIP / ranked.SUPPORT_MAX
        for c in range(1, sec.n_classes):
            with self.subTest(channel=c):
                got, near = ranked.margin(code, c), np.abs(want[c]) < 2.0
                far = near & (axis > 3.0)
                self.assertLess(np.abs(got - want[c])[far].max(), quantum,
                                "away from the junction, recovery is quantization-limited")
                tube = near & (axis < 1.0)
                self.assertGreater(np.abs(got - want[c])[tube].max(), 1.0,
                                   "the junction tube is where depth stops being free")

class TestWhatTheRasterCannotDo(unittest.TestCase):
    """Characterization, not aspiration. These record the baseline a field measurement
    has to beat, in the two regimes where the raster is known to be wrong. If a change
    ever makes one of them pass trivially, the phantom stopped exercising the problem."""

    def test_voxel_counting_saws_under_sub_voxel_translation(self):
        """The same sphere, shifted by a fraction of a voxel, counts a different number
        of voxels - a spread of ~2 % of its volume with no geometry changing at all.
        A field measure integrates a continuous level set and should be flat here; this
        is the number to quote it against."""
        g, p = _grid(n=40), ph.Phantom((ph.sphere(6.0),))
        truth = p.bodies[0].volume_mm3
        vols = np.array([(ph.labels(p, g, offset_mm=(k / 8,) * 3) == 1).sum()
                         for k in range(8)], dtype=float)
        spread = (vols.max() - vols.min()) / truth
        self.assertGreater(spread, 0.01, "the sawtooth is the point of this phantom")

    def test_area_is_scale_dependent_where_volume_is_not(self):
        """One amplitude, four spatial frequencies. Area climbs with the frequency and
        volume barely moves, because area is a first-derivative functional and volume
        integrates an indicator. Any area this system reports has to name the scale it
        was measured at; volume does not."""
        bodies = [ph.star(10.0, 0.12, m) for m in (2, 4, 6, 8)]
        vols = np.array([b.volume_mm3 for b in bodies])
        areas = np.array([b.area_mm2 for b in bodies])
        self.assertTrue(np.all(np.diff(areas) > 0), "area must grow with frequency")
        self.assertGreater(np.ptp(areas) / areas.mean(), 5 * np.ptp(vols) / vols.mean())

    def test_noise_perturbs_the_surface_without_moving_the_volume_much(self):
        """The same asymmetry, now in the logits rather than the geometry: band-limited
        noise ripples the level set. The labelmap barely changes; a surface drawn
        through it changes a lot more."""
        g, p = _grid(), ph.Phantom((ph.sphere(8.0),))
        clean = ph.logits(p, g).argmax(0).numpy() == 1
        dirty = ph.logits(p, g, noise=1.0).argmax(0).numpy() == 1
        dv = abs(int(dirty.sum()) - int(clean.sum())) / clean.sum()
        self.assertLess(dv, 0.02, "noise this mild should not move the volume")
        self.assertTrue((clean != dirty).any(), "noise this mild should move the surface")


if __name__ == "__main__":
    unittest.main()
