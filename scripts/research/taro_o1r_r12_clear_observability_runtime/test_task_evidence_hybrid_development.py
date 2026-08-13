import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_hybrid_development as subject


class HybridDevelopmentTest(unittest.TestCase):
    def test_unit_interval_constant_is_zero(self) -> None:
        np.testing.assert_array_equal(np.zeros(3), subject._unit_interval(np.ones(3)))

    def test_admission_requires_each_source(self) -> None:
        passing = {
            "parent_macro": {"ranker": 12.0, "passive": 10.0, "generic": 11.0},
            "strict_win_parent_count": 4,
        }
        failing = {
            "parent_macro": {"ranker": 10.5, "passive": 10.0, "generic": 11.0},
            "strict_win_parent_count": 8,
        }
        self.assertTrue(subject.policy_is_admissible({"TUM_RGBD": passing, "BONN_RGBD_DYNAMIC": passing}))
        self.assertFalse(subject.policy_is_admissible({"TUM_RGBD": passing, "BONN_RGBD_DYNAMIC": failing}))


if __name__ == "__main__":
    unittest.main()
