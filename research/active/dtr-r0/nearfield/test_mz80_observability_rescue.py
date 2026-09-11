import unittest

import numpy as np

from mz80_observability_rescue import conditioned_rescue, observability_unknown, oracle_curve


class Mz80ObservabilityRescueTests(unittest.TestCase):
    def test_ideal_profile_has_no_unknown_query(self):
        np.testing.assert_array_equal(observability_unknown(4.0, 4.0), np.zeros(4, bool))

    def test_short_cap_marks_only_uncovered_queries(self):
        np.testing.assert_array_equal(observability_unknown(2.45, 1.95), [False, True, False, True])
        np.testing.assert_array_equal(observability_unknown(1.55, 1.40), np.ones(4, bool))

    def test_conditioned_rescue_preserves_tof_and_blocks_known_rgb(self):
        tof = np.array([[1, -1, -1, -1]], float)
        rgb = np.ones((1, 4))
        alert = conditioned_rescue(tof, rgb, np.array([False, False, True, True]), 0.0)
        np.testing.assert_array_equal(alert, [[True, False, True, True]])

    def test_oracle_budget_never_exceeds_fp_limit(self):
        tof = -np.ones((4, 4))
        rgb = np.arange(16, dtype=float).reshape(4, 4)
        truth = np.zeros((4, 4), bool)
        truth[3, 3] = True
        known = np.ones((4, 4), bool)
        curve = oracle_curve(tof, rgb, np.ones(4, bool), truth, known, budgets=(0, 1, 5))
        for budget in (0, 1, 5):
            self.assertLessEqual(curve[str(budget)]['added_fp'], budget)
        self.assertEqual(curve['0']['rescued_tp'], 1)


if __name__ == '__main__':
    unittest.main()
