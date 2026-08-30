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
    centers, wrong once a stencil mixes voxels with different winners. Nearest-neighbor
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


class TestEmit(unittest.TestCase):
    """The shared hook every engine hands its distribution through."""

    def test_emit_encodes_stamps_and_sinks(self):
        got = []
        spec = ranked.RankedSpec(sink=lambda part, code: got.append((part, code)),
                                 depth=4, clip=6.0)
        code = ranked.emit(spec, "organs", _logits(K=8), engine="nnunetv2", task="ts:total")
        self.assertEqual(len(got), 1)
        part, sunk = got[0]
        self.assertEqual(part, "organs")
        self.assertIs(sunk, code)
        self.assertEqual(code.meta["engine"], "nnunetv2")
        self.assertEqual(code.meta["task"], "ts:total")
        self.assertEqual(code.meta["depth"], 4)          # the spec's, not the default
        self.assertEqual(code.meta["clip"], 6.0)

    def test_emit_is_a_noop_without_a_spec(self):
        """So a call site can be unconditional rather than guarded - the reason this
        returns None instead of raising."""
        self.assertIsNone(ranked.emit(None, "organs", _logits(K=4)))

    def test_emit_does_not_lose_the_encoders_own_meta(self):
        spec = ranked.RankedSpec(sink=lambda part, code: None, depth=3)
        code = ranked.emit(spec, "p", _logits(K=5), part="p")
        for key in ("mode", "classes", "shape", "support_max"):
            self.assertIn(key, code.meta, f"caller meta must not displace {key}")

    def test_sink_key_is_a_string_even_when_the_caller_passes_an_index(self):
        got = []
        spec = ranked.RankedSpec(sink=lambda part, code: got.append(part))
        ranked.emit(spec, 3, _logits(K=4))
        self.assertEqual(got, ["3"])

    def test_emit_accepts_the_nnunet_call_sites_full_keyword_set(self):
        """Mirrors pipeline.emit_probabilities' call. That path needs weights so it is not
        in the fast suite, and it is the one that stamps `part` INTO meta - which collided
        with the sink key until the first three params were made positional-only. Without
        this the collision only shows up on a GPU run."""
        got = []
        spec = ranked.RankedSpec(sink=lambda p, c: got.append((p, c)), depth=3)
        code = ranked.emit(
            spec, "organs", _logits(K=6),
            part="organs", task="ts:total", nnseg="0.2.0", engine="nnunetv2",
            spacing_zyx=[1.5, 1.5, 1.5], envelope_lo=[0, 0, 0], model_grid=[4, 4, 4],
            labels=[0, 1, 2, 3, 4, 5], convention="corner", reoriented_to_ras=True,
            input_orientation="RAS", frame={"source": None})
        self.assertEqual(got[0][0], "organs")
        self.assertEqual(code.meta["part"], "organs")
        self.assertEqual(code.meta["engine"], "nnunetv2")

    def test_the_first_three_parameters_stay_positional_only(self):
        """Pinned structurally, not just by the call above: dropping the `/` would let a
        caller's `part=` or `logits=` in meta silently become the parameter."""
        import inspect
        kinds = [p.kind for p in inspect.signature(ranked.emit).parameters.values()]
        self.assertEqual(kinds[:3], [inspect.Parameter.POSITIONAL_ONLY] * 3)
        self.assertEqual(kinds[3], inspect.Parameter.VAR_KEYWORD)


class TestTieBreaking(unittest.TestCase):
    """Ties must resolve the same way everywhere, and the same way argmax does."""

    def test_ranks0_is_exactly_argmax_when_the_top_two_tie(self):
        lg = torch.zeros(5, 1, 1, 1)
        lg[3] = lg[1] = 7.0                      # classes 1 and 3 tied for the win
        code = ranked.encode(lg, depth=4, clip=8.0)
        self.assertEqual(int(code.ranks[0].ravel()[0]) - 1, int(lg.argmax(0).ravel()[0]))
        self.assertEqual(int(code.ranks[0].ravel()[0]) - 1, 1)     # the LOWER index

    def test_a_tie_deeper_down_also_orders_by_index(self):
        """The consequential case: at the depth boundary a tie decides which class is KEPT,
        so the loser decodes to -clip instead of -gap."""
        lg = torch.zeros(6, 1, 1, 1)
        lg[0] = 9.0
        lg[4] = lg[2] = 3.0                      # tied for second
        code = ranked.encode(lg, depth=3, clip=8.0)
        self.assertEqual([int(v) - 1 for v in code.ranks[:2].ravel()], [0, 2])

    def test_a_run_of_three_ties_sorts_fully_not_pairwise(self):
        """Bubble rather than one adjacent pass - a single sweep would leave 4,2,3."""
        lg = torch.zeros(6, 1, 1, 1)
        lg[4] = lg[2] = lg[3] = 5.0
        code = ranked.encode(lg, depth=3, clip=8.0)
        self.assertEqual([int(v) - 1 for v in code.ranks.ravel()], [2, 3, 4])

    def test_ties_do_not_disturb_a_strict_ordering(self):
        lg = torch.tensor([1.0, 9.0, 5.0, 3.0]).view(4, 1, 1, 1)
        code = ranked.encode(lg, depth=4, clip=40.0)
        self.assertEqual([int(v) - 1 for v in code.ranks.ravel()], [1, 2, 3, 0])

    def test_encoding_is_deterministic_over_tie_heavy_fp16(self):
        """fp16 quantized hard so classes collide constantly - the condition that made
        device and host disagree at 1.2 % of voxels before this."""
        g = torch.Generator().manual_seed(3)
        lg = ((torch.rand(9, 6, 7, 8, generator=g) * 6).round() / 2).half()
        a = ranked.encode(lg, depth=5, clip=8.0)
        b = ranked.encode(lg.clone(), depth=5, clip=8.0)
        np.testing.assert_array_equal(a.ranks, b.ranks)
        np.testing.assert_array_equal(a.support, b.support)
        np.testing.assert_array_equal(a.ranks[0] - 1, lg.float().argmax(0).numpy())

    def test_a_tie_straddling_the_depth_boundary_is_a_known_residual(self):
        """Pinned as a LIMIT, not a guarantee: _settle_ties orders what topk selected and
        cannot change which it selected, so the class kept at the last slot is still topk's
        call. Measured at 16 voxels in 11.5 M on real data, worst margin 1.40 logits on the
        least-significant plane. If this ever starts failing because selection became
        deterministic, delete the test - do not 'fix' it."""
        lg = torch.zeros(5, 1, 1, 1)
        lg[0] = 9.0
        lg[2] = lg[4] = 2.0                      # tied for the LAST kept slot at depth 2
        code = ranked.encode(lg, depth=2, clip=8.0)
        kept = int(code.ranks[1].ravel()[0]) - 1
        self.assertIn(kept, (2, 4))              # either is a correct answer to a tie
        self.assertEqual(int(code.ranks[0].ravel()[0]) - 1, 0)   # the winner is not in doubt


class TestDecodeGroups(unittest.TestCase):
    """The inverse of encode, on whatever device the consumer runs on."""

    def test_a_group_of_one_reproduces_the_numpy_margin(self):
        lg = _logits(K=8)
        code = ranked.encode(lg, depth=6, clip=8.0)
        got = ranked.decode_groups(code, [[c] for c in range(8)]).numpy()
        for c in range(8):
            np.testing.assert_allclose(got[c], ranked.margin(code, c), atol=1e-5,
                                       err_msg=f"channel {c}")

    def test_a_union_is_not_the_max_of_member_margins(self):
        """The whole reason this exists: max() gets the sign right and underestimates the
        magnitude, so it would put a surface at every internal boundary."""
        lg = _logits(K=6)
        code = ranked.encode(lg, depth=6, clip=8.0)
        union = ranked.decode_groups(code, [[0, 1, 2]])[0].numpy()
        naive = np.maximum.reduce([ranked.margin(code, c) for c in (0, 1, 2)])
        np.testing.assert_array_equal(union > 0, naive > 0)          # same mask
        self.assertGreater(float((union - naive).max()), 0.5)        # bigger magnitude
        self.assertGreaterEqual(float((union - naive).min()), -1e-5)  # never smaller

    def test_the_union_mask_equals_the_or_of_its_member_labels(self):
        """Against the stored LABEL, not the sign of each member's margin. A lead smaller
        than the support quantum (clip/255, so 0.031 logits at the default clip) decodes to
        zero, so `margin(c) > 0` is not a reliable "c won" test near a boundary - and any
        residual disagreement must sit exactly there. §8.4 measures the same effect from the
        other side, at 99.9992 % rather than 100."""
        lg = _logits(K=7)
        code = ranked.encode(lg, depth=7, clip=8.0)     # the default the store ships with
        members = [1, 3, 5]
        union = ranked.decode_groups(code, [members])[0].numpy()
        won = np.isin(code.ranks[0].astype(np.int32) - 1, members)
        off = (union > 0) != won
        self.assertTrue(bool((code.support[0][off] == ranked.SUPPORT_MAX).all()),
                        "disagreement outside the dead zone of the support quantum")
        self.assertLess(float(off.mean()), 0.01)

    def test_internal_boundaries_vanish(self):
        """Rendering members separately puts a surface between them; the union is one
        manifold. Count zero crossings along an axis."""
        lg = _logits(K=5)
        code = ranked.encode(lg, depth=5, clip=40.0)
        a, b = ranked.margin(code, 1), ranked.margin(code, 2)
        union = ranked.decode_groups(code, [[1, 2]])[0].numpy()
        internal = ((a > 0) & (np.roll(b, -1, axis=0) > 0))[:-1]
        self.assertTrue(internal.any(), "test needs an internal boundary to be meaningful")
        crossed = internal & ((union > 0) != (np.roll(union, -1, axis=0) > 0))[:-1]
        self.assertEqual(int(crossed.sum()), 0)

    def test_quantize_reserves_zero_and_round_trips_within_one_level(self):
        """Sign agreement is NOT the contract - a margin below one level quantizes onto 128
        either way. What is promised: 0 stays the absent sentinel, 128 is the boundary, and
        the value survives to within a level."""
        lg = _logits(K=5)
        code = ranked.encode(lg, depth=5, clip=8.0)
        f = ranked.decode_groups(code, [[0]])[0]
        q = ranked.decode_groups(code, [[0]], quantize=True)[0]
        self.assertEqual(q.dtype, torch.uint8)
        self.assertGreaterEqual(int(q.min()), 1)
        back = (q.float() - 128.0) / (255.0 - 128.0) * 8.0
        self.assertLess(float((back - f.clamp(-8.0, 8.0)).abs().max()), 8.0 / 127.0)

    def test_regions_union_is_max_over_members(self):
        lg = _logits(K=4)
        code = ranked.encode_regions(lg, clip=8.0, threshold=0.0)
        got = ranked.decode_groups(code, [[0, 2]])[0].numpy()
        want = np.maximum(ranked.margin(code, 0), ranked.margin(code, 2))
        np.testing.assert_allclose(got, want, atol=1e-5)

    def test_decodes_onto_a_device_when_one_is_available(self):
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else None)
        if dev is None:
            self.skipTest("no accelerator")
        code = ranked.encode(_logits(K=6), depth=5, clip=8.0)
        out = ranked.decode_groups(code, [[0, 1], [2]], device=dev)
        self.assertEqual(out.device.type, dev)
        np.testing.assert_allclose(out.cpu().numpy(),
                                   ranked.decode_groups(code, [[0, 1], [2]]).numpy(),
                                   atol=1e-4)

    def test_a_group_field_matches_the_TRUTH_not_just_the_other_decoder(self):
        """Every other test here is self-consistency - decode_groups against margin(), both
        read out of the same code. This one computes m_S from the uncompressed logits.

        False positives are structurally impossible: ranks[0] is the exact argmax, so the
        store always agrees on who won, and a member can only fail to be claimed. Losses are
        confined to the quantization dead zone. On the real organs part this is Dice 0.9996
        to 0.9999 with FP 0 for every group."""
        lg = _logits(K=9)
        code = ranked.encode(lg, depth=7, clip=8.0)
        members = [1, 3, 5]
        mask = torch.zeros(9, dtype=torch.bool)
        mask[members] = True
        truth = lg[mask].amax(0) - lg[~mask].amax(0)
        got = ranked.decode_groups(code, [members])[0]

        self.assertEqual(int(((got > 0) & (truth <= 0)).sum()), 0, "must never over-claim")
        band = truth.abs() < 8.0 * 0.98
        err = (got - truth)[band].abs()
        self.assertLess(float(err.quantile(0.95)), 8.0 / 255 * 1.5,
                        "in-band error must stay within the support quantum")

    def test_depth_preserves_the_mask_but_costs_field_MAGNITUDE(self):
        """Measured on real data: depth 3 and depth 6 give byte-identical group masks, while
        the liver's p95 margin error goes 0.0149 -> 0.2969. So depth is a rendering decision,
        not a labelmap one - a member below the cut reads as absent (-clip) rather than at
        its true level."""
        lg = _logits(K=10)
        members = [0, 2, 4]
        mask = torch.zeros(10, dtype=torch.bool)
        mask[members] = True
        truth = lg[mask].amax(0) - lg[~mask].amax(0)
        deep = ranked.decode_groups(ranked.encode(lg, depth=8, clip=8.0), [members])[0]
        shallow = ranked.decode_groups(ranked.encode(lg, depth=2, clip=8.0), [members])[0]
        np.testing.assert_array_equal((deep > 0).numpy(), (shallow > 0).numpy())
        band = truth.abs() < 8.0 * 0.98
        self.assertLessEqual(float((deep - truth)[band].abs().max()),
                             float((shallow - truth)[band].abs().max()))

    def test_a_resident_code_decodes_without_re_uploading(self):
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else None)
        if dev is None:
            self.skipTest("no accelerator")
        code = ranked.encode(_logits(K=6), depth=5, clip=8.0)
        res = ranked.to_device(code, dev)
        self.assertEqual(res.support.device.type, dev)
        self.assertIsInstance(code.support, np.ndarray)      # the original is not consumed
        np.testing.assert_allclose(ranked.decode_groups(res, [[0, 1]]).cpu().numpy(),
                                   ranked.decode_groups(code, [[0, 1]]).numpy(), atol=1e-4)

    def test_a_resident_code_decodes_where_it_lives_by_default(self):
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else None)
        if dev is None:
            self.skipTest("no accelerator")
        res = ranked.to_device(ranked.encode(_logits(K=5), depth=4), dev)
        self.assertEqual(ranked.decode_groups(res, [[0]]).device.type, dev)

    def test_the_numpy_readers_refuse_a_resident_code_clearly(self):
        """Rather than indexing a tensor with numpy semantics and returning nonsense."""
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else None)
        if dev is None:
            self.skipTest("no accelerator")
        res = ranked.to_device(ranked.encode(_logits(K=5), depth=4), dev)
        for fn in (lambda: ranked.margin(res, 0), lambda: ranked.deficit(res, 0),
                   lambda: ranked.probabilities(res)):
            with self.assertRaises(TypeError) as cm:
                fn()
            self.assertIn("decode_groups", str(cm.exception))
