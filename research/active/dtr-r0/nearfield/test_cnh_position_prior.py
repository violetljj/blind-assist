"""Focused integrity checks for the consumed-v3 position-prior diagnostic."""
import unittest
import numpy as np
import cnh_position_prior_analysis as a
import cnh_position_prior_readout as r


class PositionPriorTests(unittest.TestCase):
    def packed(self, negative=3.1, positive=3.13):
        scores = {arm: np.full((2, 6), negative) for arm in a.ARMS}
        for arm in scores:
            scores[arm][0, 0] = positive
        y = np.zeros((2, 6), int)
        y[0, 0] = 1
        y[1, 0] = 1
        w = np.full((2, 6), np.nan)
        w[:, 0] = (1.2, .8)
        scores['G4local'].fill(-np.inf)
        return a.pack([dict(unit=128, y=y, w=w, strata=np.full((2, 6), 'tiny'), s=scores)], (0, 2, 4))

    def test_joint_grid_includes_exact_baseline(self):
        p = self.packed()
        self.assertAlmostEqual(a.select(p, 'G0', .1)[0], 3.125)
        threshold = a.select(p, 'G4', .1)
        self.assertEqual(threshold, (3.125, np.inf))
        result = a.evaluate(p, 'G4', threshold)
        self.assertEqual(result['tiny']['timely']['numerator'], 1)
        self.assertEqual(result['all']['false_alert']['numerator'], 0)
        self.assertNotIn('false_alert', result['tiny'])

    def test_first_alert_and_unreachable_finite_threshold(self):
        p = self.packed(13., 14.)
        self.assertEqual(a.select(p, 'G0', .1), (np.inf,))
        p['w'][0, :2] = (.8, 1.2)
        p['s']['G0'][0, :2] = 14.
        result = a.evaluate(p, 'G0', (13.5,))
        self.assertEqual(result['tiny']['timely']['numerator'], 0)
        self.assertEqual(result['tiny']['late']['numerator'], 1)

    def test_low_footprint_and_background_exclusion(self):
        oid = np.full((2, 8, 8, 256), 100, int)
        oid[:, 2, 3, :8] = 1
        dist = np.full(oid.shape, 1.4)
        objects = [dict(id=1, category='LOW'), dict(id=100, category='BACKGROUND')]
        masks, counts = r.oracle_masks(oid, dist, objects, 128, 0, [np.array([[0, 0]])])
        self.assertEqual(counts, [1, 1])
        self.assertEqual(int(masks['G1'].sum()), 32)
        self.assertTrue(masks['G1'][:, 2, 3].all())
        self.assertFalse(masks['G1'][:, 0, 0].any())
        self.assertFalse((masks['G2'] & ~masks['G1']).any())
        self.assertFalse((masks['G1'] & ~masks['G3d']).any())

    def test_temporal_branch_independent_of_oracle_and_future(self):
        rng = np.random.default_rng(12)
        hist = rng.normal(0, 2, (3, 8, 8, 16))
        ambient, bias = np.ones((3, 8, 8)), np.zeros((8, 8, 16))
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        empty = {arm: np.zeros_like(hist, bool) for arm in r.ARMS[1:-1]}
        full = {arm: np.ones_like(hist, bool) for arm in r.ARMS[1:-1]}
        first = r.sequence_scores(hist, ambient, bias, poses, poses, empty)
        second = r.sequence_scores(hist, ambient, bias, poses, poses, full)
        np.testing.assert_array_equal(first['G4local'], second['G4local'])
        self.assertTrue(np.isneginf(first['G1']).all())
        np.testing.assert_allclose(second['G1'], second['G0'], atol=1e-6)
        hist[2] += 100
        future = r.sequence_scores(hist, ambient, bias, poses, poses, full)
        np.testing.assert_array_equal(second['G4local'][:2], future['G4local'][:2])


if __name__ == '__main__':
    unittest.main()
