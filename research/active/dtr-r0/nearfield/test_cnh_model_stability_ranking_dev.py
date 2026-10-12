import unittest
import numpy as np
from cnh_model_stability_ranking_dev import _rank_summary, ranking_metrics


class RankingTests(unittest.TestCase):
    def test_ties_get_half_auc_but_no_fractional_operating_point(self):
        tied = _rank_summary(np.array([1., 1.]), np.array([1., 1.]), .05)
        self.assertEqual(tied['auc'], .5)
        self.assertEqual(tied['partial_step_area'], 0.)
        result = _rank_summary(np.array([3., 2.]), np.array([2., 1.]), .5)
        self.assertEqual(result['auc'], .875)
        self.assertEqual(result['partial_step_area'], .25)
        self.assertEqual(result['partial_step_area_normalized'], .5)

    def test_window_boundaries_early_deadlines_and_unknown(self):
        x = np.full((6, 13), np.nan)
        x[0, 2:4] = [2, 3]
        x[0, :2] = 1000  # Warmup must not improve the peak.
        x[0, 4:] = 1000  # Future after deadline must not improve it either.
        x[4, 2:12] = 3
        result = ranking_metrics(x, [1, 1, 1, 0, 0, 1], [0, 0, 0, 0, 1, 0],
                                 [3, 1, -1, -1, -1, 12], mask=[1, 1, 1, 1, 1, 0])
        self.assertEqual(result['contact_events'], 3)
        self.assertEqual(result['positive_events'], 1)
        self.assertEqual(result['early_deadline_events'], 1)
        self.assertEqual(result['missing_deadline_events'], 1)
        self.assertEqual(result['unevaluable_contact_events'], 2)
        self.assertEqual(result['control_intervals'], 10)
        self.assertEqual(result['auc'], .5)
        self.assertEqual(result['partial_step_area'], 0.)

    def test_increasing_transform_preserves_ranking(self):
        rng = np.random.default_rng(7)
        scores = rng.integers(-3, 4, (8, 13)).astype(float)
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
        args = labels, ~labels, np.array([2, 5, 10, 12, -1, -1, -1, -1])
        left = ranking_metrics(scores, *args, cap=.3)
        right = ranking_metrics(scores*7+20, *args, cap=.3)
        self.assertEqual(left, right)
        for key in left:
            self.assertNotIn('threshold', key)

    def test_missing_class_and_selected_nonfinite(self):
        x = np.zeros((2, 13))
        result = ranking_metrics(x, [1, 0], [0, 0], [2, -1])
        self.assertIsNone(result['auc'])
        self.assertIsNone(result['partial_step_area_normalized'])
        self.assertEqual(result['positive_events'], 1)
        x[1, 2] = np.nan
        with self.assertRaisesRegex(ValueError, 'control window'):
            ranking_metrics(x, [1, 0], [0, 1], [2, -1])


if __name__ == '__main__':
    unittest.main()
