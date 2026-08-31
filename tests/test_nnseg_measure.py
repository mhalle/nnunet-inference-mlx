"""Volume and area from the field, against geometry that knows its own answers.

Every number here is checked against `nnseg.phantoms`, whose truth is closed form or
spectral quadrature rather than another measurement. The counting baseline is measured
in the same breath, because the case for integrating the field is comparative: it is
not that counting is imprecise, it is that its area estimates a different quantity and
its volume depends on where the body happens to sit against the grid.

No weights and no files - all synthetic, so it runs in the fast suite.
"""
import math
import unittest

import numpy as np

from nnseg import measure, phantoms as ph, ranked
from nnseg.grid import Grid


def _grid(extent_mm=96.0, spacing=1.5):
    n = int(round(extent_mm / spacing))
    return Grid(shape=(n,) * 3, spacing=(spacing,) * 3, origin=(-(n - 1) * spacing / 2,) * 3)


def _field(body, grid, **kw):
    return ph.margins(ph.Phantom((body,)), grid, **kw)[1]


SMOOTH = [ph.sphere(20.0), ph.ellipsoid((12.0, 18.0, 26.0)), ph.torus(26.0, 8.0),
          ph.shell(20.0, 16.0), ph.rounded_box((10.0, 10.0, 10.0), 6.0),
          ph.star(20.0, 0.15, 4)]


class TestCubeCut(unittest.TestCase):
    """The plane-cut primitive, against sections whose area is known by hand. The
    degenerate branches are the ones that matter - a face perpendicular to an axis is
    where a medical image spends most of its surface, and it is where the general
    `1/(6abc)` form returns noise."""

    def _one(self, n, alpha):
        n = np.asarray(n, dtype=np.float64)[:, None]
        v, d = measure._cube_cut(np.abs(n), np.array([float(alpha)]))
        return float(v[0]), float(np.linalg.norm(n)) * float(d[0])

    def test_the_diagonal_section_is_a_hexagon(self):
        v, a = self._one([1 / 3, 1 / 3, 1 / 3], 0.5)
        self.assertAlmostEqual(v, 0.5, places=12)
        self.assertAlmostEqual(a, 3 * math.sqrt(3) / 4, places=12)

    def test_a_face_on_plane_cuts_a_unit_square(self):
        v, a = self._one([1.0, 0.0, 0.0], 0.25)
        self.assertAlmostEqual(v, 0.25, places=12)
        self.assertAlmostEqual(a, 1.0, places=12)

    def test_an_edge_on_plane_cuts_a_root_two_rectangle(self):
        v, a = self._one([0.5, 0.5, 0.0], 0.5)
        self.assertAlmostEqual(v, 0.5, places=12)
        self.assertAlmostEqual(a, math.sqrt(2.0), places=12)

    def test_the_degenerate_branches_are_continuous_with_the_general_one(self):
        """Approaching a flat axis must not jump: the branch is a limit being taken,
        not a different answer being substituted."""
        for eps in (1e-4, 1e-5):
            with self.subTest(eps=eps):
                near = self._one([0.6, 0.4 - eps, eps], 0.5)
                flat = self._one([0.6, 0.4, 0.0], 0.5)
                self.assertAlmostEqual(near[0], flat[0], places=3)
                self.assertAlmostEqual(near[1], flat[1], places=3)


class TestAccuracy(unittest.TestCase):
    def test_smooth_bodies_come_out_within_a_percent(self):
        g = _grid()
        for b in SMOOTH:
            with self.subTest(b.name):
                v, a = measure.volume_area(_field(b, g), g.spacing)
                self.assertLess(abs(v / b.volume_mm3 - 1), 0.01)
                self.assertLess(abs(a / b.area_mm2 - 1), 0.01)

    def test_anisotropic_spacing_measures_the_same_geometry(self):
        """The body does not change when the sampling does. Index space and mm are
        different spaces and the area element is not a single scale factor, so this is
        where a missing spacing term would show."""
        b = ph.sphere(20.0)
        for spacing in ((1.5, 1.5, 1.5), (3.0, 1.0, 1.0), (1.0, 1.0, 3.0)):
            with self.subTest(spacing=spacing):
                n = [int(round(96.0 / s)) for s in spacing]
                g = Grid(shape=tuple(n), spacing=spacing,
                         origin=tuple(-(k - 1) * s / 2 for k, s in zip(n, spacing)))
                v, a = measure.volume_area(_field(b, g), g.spacing)
                self.assertLess(abs(v / b.volume_mm3 - 1), 0.02)
                self.assertLess(abs(a / b.area_mm2 - 1), 0.03)

    def test_the_stored_form_is_free_for_volume_and_biases_area_low(self):
        """The asymmetry, measured. Clip and the uint8 support cost the volume nothing;
        they cost area 0.3-1.1 %, and always in the same direction, because quantizing a
        field is a low-pass filter and area is a first-derivative functional. An area
        read out of a store is a slight under-estimate by construction - still an order
        of magnitude better than counting, but not a neutral measurement."""
        g = _grid()
        losses = []
        for b in (ph.sphere(20.0), ph.torus(26.0, 8.0), ph.star(20.0, 0.15, 4)):
            with self.subTest(b.name):
                p = ph.Phantom((b,))
                exact = measure.volume_area(ph.margins(p, g)[1], g.spacing)
                code = ranked.encode(ph.logits(p, g), depth=2)
                stored = measure.volume_area(ranked.margin(code, 1), g.spacing)
                self.assertLess(abs(stored[0] / exact[0] - 1), 0.001, "volume is free")
                self.assertLess(abs(stored[1] / exact[1] - 1), 0.015)
                losses.append(stored[1] / exact[1] - 1)
        self.assertTrue(all(x < 0 for x in losses),
                        f"quantization should only ever remove area; got {losses}")

    def test_a_sharp_crease_reads_low_in_area_but_not_in_volume(self):
        """The known bias, pinned rather than left to be discovered. A trilinear field
        cannot represent an edge, so a cube's area comes out a few percent short - while
        its volume is as good as any smooth body's, because rounding a corner moves a
        second-order amount of volume and a first-order amount of surface."""
        g, b = _grid(), ph.box((14.0, 14.0, 14.0))
        v, a = measure.volume_area(_field(b, g), g.spacing)
        self.assertLess(abs(v / b.volume_mm3 - 1), 0.01)
        self.assertTrue(-0.08 < a / b.area_mm2 - 1 < -0.02,
                        f"expected a few percent low, got {100*(a/b.area_mm2-1):+.2f} %")


class TestAgainstCounting(unittest.TestCase):
    """The comparison the module exists to win. Truth is identical in every row."""

    def test_counted_area_overstates_a_smooth_surface_by_about_half(self):
        g = _grid()
        for b in SMOOTH:
            with self.subTest(b.name):
                ca = measure.counted_volume_area(_field(b, g), g.spacing)[1]
                self.assertGreater(ca / b.area_mm2 - 1, 0.15,
                                   "face counting is not a coarse area, it is another quantity")

    def test_counted_area_does_not_converge_but_the_field_does(self):
        """Four grid refinements. The field's error falls off; counting's sits where it
        was. This is the whole argument in one table, so it is asserted rather than
        described."""
        b = ph.sphere(20.0)
        cerr, ferr = [], []
        for spacing in (3.0, 2.0, 1.5, 1.0):
            g = _grid(96.0, spacing)
            m = _field(b, g)
            cerr.append(measure.counted_volume_area(m, g.spacing)[1] / b.area_mm2 - 1)
            ferr.append(measure.volume_area(m, g.spacing)[1] / b.area_mm2 - 1)
        self.assertGreater(min(cerr), 0.30, "counting stays badly high at every spacing")
        self.assertLess(np.ptp(cerr), 0.10, "and does not trend toward the truth")
        self.assertLess(abs(ferr[-1]), abs(ferr[0]) / 4, "the field converges")

    def test_counting_saws_under_sub_voxel_translation_and_the_field_does_not(self):
        """Nothing about the geometry changes, so nothing about the measurement should.
        Counting's answer depends on where the body sits against the grid; the field's
        does not, which is the property that makes a longitudinal comparison mean
        something."""
        g, b = _grid(), ph.sphere(20.0)
        cv, fv, ca, fa = [], [], [], []
        for k in range(8):
            m = _field(b, g, offset_mm=(k * 1.5 / 8,) * 3)
            v, a = measure.counted_volume_area(m, g.spacing); cv.append(v); ca.append(a)
            v, a = measure.volume_area(m, g.spacing); fv.append(v); fa.append(a)
        self.assertGreater(np.ptp(cv) / b.volume_mm3, 0.005)
        self.assertLess(np.ptp(fv) / b.volume_mm3, 0.0005)
        self.assertLess(np.ptp(fa) / b.area_mm2, 0.005)


class TestMechanics(unittest.TestCase):
    def test_the_slab_size_is_not_part_of_the_answer(self):
        """It bounds memory - the dense form wants eight float64 copies of the volume -
        and must not do anything else."""
        g = _grid()
        m = _field(ph.torus(26.0, 8.0), g)
        ref = measure.volume_area(m, g.spacing, slab=1)
        for slab in (3, 16, 10_000):
            with self.subTest(slab=slab):
                got = measure.volume_area(m, g.spacing, slab=slab)
                self.assertAlmostEqual(got[0], ref[0], places=6)
                self.assertAlmostEqual(got[1], ref[1], places=6)

    def test_an_absent_structure_measures_zero_rather_than_failing(self):
        self.assertEqual(measure.volume_area(np.full((8, 8, 8), -8.0), (1.0, 1.0, 1.0)),
                         (0.0, 0.0))
        self.assertEqual(measure.counted_volume_area(np.full((8, 8, 8), -8.0),
                                                     (1.0, 1.0, 1.0)), (0.0, 0.0))

    def test_a_float32_field_measures_the_same_as_a_float64_one(self):
        """`ranked.margin` returns float32; the accumulation is float64 regardless."""
        g = _grid()
        m = _field(ph.sphere(20.0), g)
        a = measure.volume_area(m.astype(np.float32), g.spacing)
        b = measure.volume_area(m.astype(np.float64), g.spacing)
        self.assertAlmostEqual(a[0] / b[0], 1.0, places=6)
        self.assertAlmostEqual(a[1] / b[1], 1.0, places=6)

    def test_bad_shapes_and_spacings_are_refused(self):
        with self.assertRaisesRegex(ValueError, r"\(Z, Y, X\)"):
            measure.volume_area(np.zeros((4, 4)), (1.0, 1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "3 positive values"):
            measure.volume_area(np.zeros((4, 4, 4)), (1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()


class TestCensoredCorners(unittest.TestCase):
    """A clipped field's straddling cells have corners AT the clip, and those are bounds
    rather than values. Reading them as values is the largest error in the module, and
    it gets worse the faster the margin climbs - which is set by the network, not by us.
    These sweep the band-to-cell-diagonal ratio across the range measured on real
    TotalSegmentator output (0.43 to 1.02) rather than testing one convenient point.
    """

    CLIP = 8.0
    DIAG = 1.5 * math.sqrt(3.0)

    def _clipped(self, body, grid, ratio):
        """A field whose +/-clip band spans ``ratio`` cell diagonals."""
        m = ph.margins(ph.Phantom((body,)), grid, gradient=self.CLIP / (ratio * self.DIAG))[1]
        return np.clip(m, -self.CLIP, self.CLIP)

    def test_ignoring_the_clip_loses_area_and_passing_it_recovers_most(self):
        g = _grid()
        for ratio in (0.4, 0.6, 0.8):
            for b in (ph.sphere(20.0), ph.torus(26.0, 8.0), ph.star(20.0, 0.15, 4)):
                with self.subTest(ratio=ratio, body=b.name):
                    m = self._clipped(b, g, ratio)
                    naive = measure.volume_area(m, g.spacing)[1] / b.area_mm2 - 1
                    fixed = measure.volume_area(m, g.spacing, clip=self.CLIP)[1] / b.area_mm2 - 1
                    self.assertLess(fixed, 0.0, "the correction must not overshoot")
                    self.assertLess(abs(fixed), abs(naive) + 1e-9)
                    self.assertLess(abs(fixed), 0.012)

    def test_the_loss_is_worst_where_the_band_is_narrowest(self):
        """The dose-response that identifies the cause. If this ever stops holding, the
        error being corrected is not the one described."""
        g, b = _grid(), ph.sphere(20.0)
        naive = [abs(measure.volume_area(self._clipped(b, g, r), g.spacing)[1] / b.area_mm2 - 1)
                 for r in (0.4, 0.6, 0.8, 1.2)]
        self.assertTrue(all(x > y for x, y in zip(naive, naive[1:])), naive)
        self.assertGreater(naive[0], 0.03, "a narrow band should cost several percent")
        self.assertLess(naive[-1], 0.01, "a wide one should cost nearly nothing")

    def test_a_flat_faced_body_is_not_made_worse(self):
        """The reason the censored corners are reactivated rather than dropped. A cube at
        a narrow band has whole corner planes at the clip; an unconstrained fit rotates
        the normal freely and does far worse than doing nothing. The inequality they
        carry is what prevents that, so this must come out unchanged, not merely close."""
        g, b = _grid(), ph.box((14.0, 14.0, 14.0))
        for ratio in (0.4, 0.6, 1.0):
            with self.subTest(ratio=ratio):
                m = self._clipped(b, g, ratio)
                naive = measure.volume_area(m, g.spacing)[1]
                fixed = measure.volume_area(m, g.spacing, clip=self.CLIP)[1]
                self.assertAlmostEqual(fixed / naive, 1.0, places=6)

    def test_an_unclipped_field_is_left_alone(self):
        """No corner is at the clip, so there is nothing to reactivate and the two paths
        must agree - otherwise the correction is inventing censoring where there is none."""
        g, b = _grid(), ph.sphere(20.0)
        m = ph.margins(ph.Phantom((b,)), g)[1]
        a = measure.volume_area(m, g.spacing)
        c = measure.volume_area(m, g.spacing, clip=1e6)
        self.assertAlmostEqual(c[0] / a[0], 1.0, places=6)
        self.assertAlmostEqual(c[1] / a[1], 1.0, places=4)

    def test_volume_is_barely_moved_by_any_of_this(self):
        """Clipping cost the volume nothing to begin with - the misplacement is symmetric
        across the surface and cancels - so the correction must not disturb it either."""
        g, b = _grid(), ph.sphere(20.0)
        m = self._clipped(b, g, 0.5)
        naive = measure.volume_area(m, g.spacing)[0]
        fixed = measure.volume_area(m, g.spacing, clip=self.CLIP)[0]
        for got in (naive, fixed):
            self.assertLess(abs(got / b.volume_mm3 - 1), 0.01)
