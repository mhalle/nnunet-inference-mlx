"""The ranked encoder: what it stores, what it recovers, and what zero means.

No weights and no files - synthetic logits, so these run in the fast suite.
"""
import unittest

import numpy as np
import torch

from nnseg import ranked


def _logits(K=8, shape=(6, 10, 12), seed=0):
    """Blobby, spatially correlated logits - a caricature of a segmentation, so the
    encoder is exercised on confident interiors AND near-tie boundaries."""
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape],
                                indexing="ij")
    out = []
    for k in range(K):
        c = torch.rand(3, generator=g) * torch.tensor(shape, dtype=torch.float32)
        d = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2).sqrt()
        out.append(6.0 - 0.9 * d)
    return torch.stack(out)


class TestEncode(unittest.TestCase):
    def test_rank_zero_is_the_argmax_and_ranks_store_class_plus_one(self):
        lg = _logits()
        code = ranked.encode(lg, depth=3)
        want = lg.argmax(0).numpy()
        np.testing.assert_array_equal(code.ranks[0].astype(np.int64) - 1, want)
        self.assertEqual(code.meta["rank_sentinel"], 0)

    def test_the_winners_rank_plane_never_carries_the_sentinel(self):
        """Every voxel has a winner, so plane 0 is always a real class - which is what makes
        `ranks[0] - 1` a labelmap with no special cases."""
        code = ranked.encode(_logits(), depth=4)
        self.assertTrue((code.ranks[0] != 0).all())

    def test_zero_means_absent_in_both_arrays(self):
        """The fill-value contract: an unwritten block decodes to 'nothing here'. A class
        masked out of the rank plane must also read as absent through `margin`."""
        code = ranked.encode(_logits(K=12), depth=4, clip=2.0)
        masked = code.ranks[1:] == 0
        self.assertTrue(masked.any(), "clip too generous to exercise the sentinel")
        self.assertTrue((code.support[masked[: code.support.shape[0]]] == 0).all(),
                        "a sentinel rank must coincide with zero support")

    def test_margin_is_positive_inside_negative_outside_and_zero_on_the_boundary(self):
        lg = _logits()
        code = ranked.encode(lg, depth=4)
        winner = lg.argmax(0).numpy()
        for c in range(int(lg.shape[0])):
            m = ranked.margin(code, c)
            inside = winner == c
            if inside.any():
                self.assertTrue((m[inside] >= 0).all(), f"class {c} negative where it won")
            if (~inside).any():
                self.assertTrue((m[~inside] <= 0).all(), f"class {c} positive where it lost")

    def test_margin_recovers_the_true_logit_gap_within_quantization(self):
        lg = _logits().double()
        code = ranked.encode(lg.float(), depth=6, clip=8.0)
        top2 = torch.topk(lg, 2, dim=0).values
        truth = (top2[0] - top2[1]).numpy()                     # the winner's true margin
        got = np.stack([ranked.margin(code, c) for c in range(int(lg.shape[0]))]).max(0)
        near = truth < 8.0 * 0.98                               # away from the clip
        self.assertLess(np.abs(got[near] - truth[near]).max(), 8.0 / 255,
                        "margins must land within one quantization step")

    def test_the_winners_margin_is_the_gap_to_the_runner_up_not_its_complement(self):
        """A confident winner must decode to a LARGE margin. Reading the stored support as
        the gap itself (rather than counting up from the clip) inverts this and still looks
        plausible - non-negative inside, zero outside - so it is pinned down here."""
        lg = torch.zeros(3, 1, 1, 1)
        lg[0] = 6.0                                             # winner leads by 6 logits
        m = ranked.margin(ranked.encode(lg, depth=3, clip=8.0), 0)
        self.assertAlmostEqual(float(m.ravel()[0]), 6.0, places=1)

    def test_probabilities_round_trip_against_a_direct_softmax(self):
        lg = _logits(K=10)
        code = ranked.encode(lg, depth=10)                       # exhaustive: tail is exact
        self.assertTrue(code.meta["exhaustive"])
        self.assertIsNone(code.tail)
        ids, p = ranked.probabilities(code)
        truth = torch.softmax(lg.double(), 0).numpy()
        live = ids >= 0                                          # -1 marks an absent class
        got = np.take_along_axis(truth, np.where(live, ids, 0), axis=0)
        # the floor is the support quantization: half a step is clip/(2*255) logits, which
        # for a near-certain winner is ~1.6 % of its probability
        self.assertLess(np.abs(p[live] - got[live]).max(), 0.02)

    def test_absent_classes_report_no_probability_and_an_unusable_id(self):
        """-1 would index the LAST class under take_along_axis, quietly crediting this
        voxel's mass to whichever class sorts last, so p must be 0 wherever the id is -1."""
        code = ranked.encode(_logits(K=12), depth=6, clip=1.0)   # tight clip: plenty absent
        ids, p = ranked.probabilities(code)
        self.assertTrue((ids < 0).any(), "clip too generous to exercise the sentinel")
        self.assertTrue((p[ids < 0] == 0).all())
        self.assertTrue((p[ids >= 0] > 0).all())

    def test_truncated_probabilities_use_the_tail_to_renormalize(self):
        lg = _logits(K=12)
        code = ranked.encode(lg, depth=3)
        self.assertFalse(code.meta["exhaustive"])
        self.assertIsNotNone(code.tail)
        ids, p = ranked.probabilities(code)
        live = ids >= 0
        truth = np.take_along_axis(torch.softmax(lg.double(), 0).numpy(),
                                   np.where(live, ids, 0), axis=0)
        self.assertLess(np.abs(p[live] - truth[live]).max(), 0.02)

    def test_deeper_truncation_shrinks_the_tail(self):
        lg = _logits(K=12)
        tails = [ranked.encode(lg, depth=d).meta["max_tail"] for d in (2, 3, 5)]
        self.assertGreater(tails[0], tails[-1])
        self.assertEqual(sorted(tails, reverse=True), tails)

    def test_rank_dtype_follows_the_class_count(self):
        """class + 1 must fit; above 254 classes the byte silently would not, so the dtype is
        chosen from K and declared rather than hard-coded."""
        self.assertEqual(ranked.encode(_logits(K=8), depth=2).meta["rank_dtype"], "uint8")
        self.assertEqual(ranked._rank_dtype(300), np.uint16)

    def test_slabbing_does_not_change_the_result(self):
        lg = _logits(shape=(9, 8, 8))
        a = ranked.encode(lg, depth=4, slab=2)
        b = ranked.encode(lg, depth=4, slab=1000)
        np.testing.assert_array_equal(a.ranks, b.ranks)
        np.testing.assert_array_equal(a.support, b.support)

    def test_depth_is_capped_at_the_class_count(self):
        code = ranked.encode(_logits(K=4), depth=99)
        self.assertEqual(code.meta["depth"], 4)
        self.assertTrue(code.meta["exhaustive"])

    def test_it_refuses_input_that_is_not_logits(self):
        with self.assertRaises(ValueError):
            ranked.encode(torch.zeros(4, 5, 6))                  # missing an axis
        with self.assertRaises(TypeError):
            ranked.encode(torch.zeros(2, 3, 4, 5, dtype=torch.int32))


class TestRegions(unittest.TestCase):
    """Sigmoid heads: no winner, no normalizer, and channels that may overlap."""

    def test_overlapping_regions_are_all_kept(self):
        lg = torch.full((3, 4, 5, 6), 2.0)                        # every region present at once
        code = ranked.encode_regions(lg)
        for c in range(3):
            self.assertTrue((ranked.margin(code, c) > 0).all())

    def test_the_threshold_is_folded_in_so_zero_is_the_boundary(self):
        lg = torch.full((2, 3, 3, 3), 1.5)
        at = ranked.encode_regions(lg, threshold=1.5)
        self.assertEqual(float(ranked.margin(at, 0).max()), 0.0,
                         "a logit exactly at the threshold must encode as margin 0 exactly")
        above = ranked.encode_regions(lg, threshold=0.0)
        self.assertGreater(ranked.margin(above, 0).min(), 0)

    def test_regions_decode_as_independent_sigmoids(self):
        lg = torch.tensor([2.0, -1.0]).reshape(2, 1, 1, 1).expand(2, 2, 2, 2).contiguous()
        _, p = ranked.probabilities(ranked.encode_regions(lg))
        self.assertAlmostEqual(float(p[0].mean()), 1 / (1 + np.exp(-2.0)), places=2)
        self.assertAlmostEqual(float(p[1].mean()), 1 / (1 + np.exp(1.0)), places=2)
        self.assertEqual(int(ranked.encode_regions(torch.zeros(1, 1, 1, 1))
                             .support.ravel()[0]), ranked.ZERO_LEVEL)
        self.assertGreater(float(p.sum(0).mean()), 0.0)           # nothing normalizes these

    def test_regions_carry_no_ranks_or_tail(self):
        code = ranked.encode_regions(torch.zeros(3, 2, 2, 2))
        self.assertIsNone(code.ranks)
        self.assertIsNone(code.tail)
        self.assertEqual(code.meta["mode"], "regions")


if __name__ == "__main__":
    unittest.main()


class TestGauge(unittest.TestCase):
    """Which decoded field may be interpolated before an argmax.

    `deficit` is the logits shifted by a per-voxel constant shared by all channels, so it is
    restore-equivalent. `margin` adds the winner's lead to one channel only - fine at voxel
    centers, wrong once a stencil mixes voxels with different winners. Nearest-neighbour
    restore cannot see the difference, which is exactly how it stays hidden.
    """

    def _sample(self, stack, n=20000, seed=0):
        import torch.nn.functional as F
        g = torch.Generator().manual_seed(seed)
        pts = (torch.rand(n, 3, generator=g) * 2 - 1).view(1, 1, 1, n, 3)
        return F.grid_sample(stack[None], pts, mode="bilinear",
                             align_corners=True)[0, :, 0, 0].argmax(0)

    def test_deficit_is_the_logits_up_to_a_per_voxel_constant(self):
        lg = _logits(K=6)
        code = ranked.encode(lg, depth=6, clip=40.0)          # exhaustive + generous clip
        d = np.stack([ranked.deficit(code, c) for c in range(6)])
        shift = lg.numpy() - d                                 # must be constant across c
        np.testing.assert_allclose(shift.max(0), shift.min(0), atol=0.2)

    def test_interpolating_deficit_reproduces_interpolating_logits(self):
        lg = _logits(K=6)
        code = ranked.encode(lg, depth=6, clip=40.0)
        d = torch.from_numpy(np.stack([ranked.deficit(code, c) for c in range(6)]))
        agree = float((self._sample(d) == self._sample(lg)).float().mean())
        self.assertGreater(agree, 0.995, "deficit must be restore-equivalent to the logits")

    def test_margin_is_not_restore_equivalent_and_that_is_expected(self):
        """Pinned so nobody 'simplifies' the restore path back onto margin: the two agree at
        voxel centers, so only an interpolating comparison catches the swap."""
        lg = _logits(K=6)
        code = ranked.encode(lg, depth=6, clip=40.0)
        m = torch.from_numpy(np.stack([ranked.margin(code, c) for c in range(6)]))
        d = torch.from_numpy(np.stack([ranked.deficit(code, c) for c in range(6)]))
        np.testing.assert_array_equal(m.argmax(0).numpy(), d.argmax(0).numpy())   # centers agree
        self.assertLess(float((self._sample(m) == self._sample(lg)).float().mean()),
                        float((self._sample(d) == self._sample(lg)).float().mean()))

    def test_margin_keeps_the_winners_lead_and_deficit_zeroes_it(self):
        lg = torch.zeros(3, 1, 1, 1); lg[0] = 5.0
        code = ranked.encode(lg, depth=3, clip=8.0)
        self.assertAlmostEqual(float(ranked.margin(code, 0).ravel()[0]), 5.0, places=1)
        self.assertEqual(float(ranked.deficit(code, 0).ravel()[0]), 0.0)
        self.assertAlmostEqual(float(ranked.deficit(code, 1).ravel()[0]), -5.0, places=1)
