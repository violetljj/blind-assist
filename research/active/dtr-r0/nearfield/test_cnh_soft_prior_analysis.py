"""Synthetic joint-selection tests, independent brute force for first-hit counts."""
import unittest
import numpy as np
import cnh_soft_prior_analysis as soft


def brute(p, score, thresholds):
    false, timely = [], []
    for threshold in thresholds:
        f, t = 0, 0
        for i in range(len(score)):
            alert = score[i] >= threshold
            pos = np.flatnonzero(p['y'][i])
            if not len(pos):
                f += int(alert.any())
            elif p['near'][i]:
                hits = pos[alert[pos]]
                t += int(bool(len(hits) and p['w'][i, hits[0]] >= 1.))
        false.append(f)
        timely.append(t)
    return np.array(false), np.array(timely)


class SoftPriorTests(unittest.TestCase):
    def test_score_parity_tolerates_rounding_but_not_mask_change(self):
        reference = np.array([-np.inf, 1., 5.])
        observed = np.array([-np.inf, 1.+1e-7, 5.+1e-6])
        self.assertLess(soft.score_parity(reference, observed)['max_rel'], 1e-5)
        with self.assertRaises(ValueError):
            soft.score_parity(reference, np.array([-50., 1., 5.]))
        with self.assertRaises(ValueError):
            soft.score_parity(reference, np.array([-np.inf, 1.1, 5.]))

    def test_all_threshold_interval_counts_match_brute(self):
        rng = np.random.default_rng(812)
        y = rng.random((40, 9)) < .6
        y[:10] = False
        w = rng.uniform(.3, 2., y.shape)
        p = dict(y=y, w=w, empty=~y.any(1), near=(y & (w <= 1.)).any(1))
        scores = rng.choice([-np.inf, 0., 1., 1.025, 2., 3., 5., 12., 20.], size=y.shape)
        actual = soft.threshold_counts(p, scores)
        expected = brute(p, scores, soft.GRID)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_nonmonotonic_witness_requires_all_thresholds(self):
        p = dict(y=np.array([[False, False], [True, True]]),
                 w=np.array([[np.nan, np.nan], [.8, 1.3]]),
                 empty=np.array([True, False]), near=np.array([False, True]))
        score = np.array([[1., 0.], [1.5, 2.5]])
        picked = soft.select_joint(p, [score.copy() for _ in soft.DELTAS])
        for row in picked.values():
            self.assertEqual(row['delta'], 0.)
            self.assertEqual(row['threshold'], 2.5)
            self.assertEqual(row['calib_near_timely'], 1)
            self.assertEqual(row['calib_false_alerts'], 0)

    def test_ties_prefer_fewer_false_then_smaller_delta(self):
        p = dict(y=np.array([[False, False], [True, True]]),
                 w=np.array([[np.nan, np.nan], [1.3, .8]]),
                 empty=np.array([True, False]), near=np.array([False, True]))
        base = np.array([[1., 0.], [2., 0.]])
        scores = [base.copy() for _ in soft.DELTAS]
        # Larger delta can achieve same timely count at larger threshold; smaller delta wins.
        scores[-1] = base+3
        for row in soft.select_joint(p, scores).values():
            self.assertEqual(row['delta'], 0.)
            self.assertEqual(row['threshold'], 2.)

    def test_disabled_threshold_and_nonfinite_rejection(self):
        p = dict(y=np.array([[False], [True]]), w=np.array([[np.nan], [.8]]),
                 empty=np.array([True, False]), near=np.array([False, True]))
        score = np.array([[20.], [1.]])
        for row in soft.select_joint(p, [score]*8).values():
            self.assertTrue(np.isinf(row['threshold']))
            self.assertEqual(row['delta'], 0.)
        with self.assertRaises(ValueError):
            soft.threshold_counts(p, np.array([[np.nan], [1.]]))

    def test_gain_denominator_fixed_and_fa_flag_separate(self):
        results = {}
        for g in ('HEAD', 'BODY'):
            for b in ('0.05', '0.10', '0.20'):
                for policy, timely, fa in [('G0', .4, .1), ('G1', .6, .1),
                    ('G1c__COMBO_MODERATE__A80', .51, .14), ('G1c__COMBO_MODERATE__A90', .49, .1)]:
                    results[f'{g}/{policy}/{b}'] = {'eval': {'tiny': {'timely': {'rate': timely}}, 'all': {'false_alert': {'rate': fa}}}}
        _, gates = soft.result_checks(results)
        self.assertTrue(gates['HEAD']['passes_any_auc_at_most_090'])
        self.assertTrue(gates['HEAD']['tested'][0]['false_alert_exceeds_G0_by_2pp'])
        self.assertAlmostEqual(results['HEAD/G1c__COMBO_MODERATE__A80/0.10']['retention_of_fixed_hard_G1_gain'], .55)


if __name__ == '__main__':
    unittest.main()
