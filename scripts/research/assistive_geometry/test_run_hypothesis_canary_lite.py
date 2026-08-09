from __future__ import annotations

import unittest

import numpy as np

from scripts.research.assistive_geometry.run_hypothesis_canary_lite import (
    discrete_hazard_cdf,
    interval_censored_nll,
    minimum_exchangeable_units,
    profile_conditioned_clearance,
    run_canary,
    soft_widest_forward_path_capacity,
    widest_forward_path_capacity,
)


class AssistiveGeometryHypothesisCanaryTest(unittest.TestCase):
    def test_hazard_representation_is_nested_and_trains_right_censoring(self) -> None:
        _, _, occupancy = discrete_hazard_cdf([0.5, -0.5, 1.0, -1.0])
        self.assertTrue(np.all(np.diff(occupancy) >= 0.0))
        self.assertLess(
            interval_censored_nll([-3.0] * 4, None),
            interval_censored_nll([1.0] * 4, None),
        )
        self.assertTrue(np.isfinite(interval_censored_nll([0.5, -0.5, 1.0], 1)))

    def test_profile_query_is_conservative_under_wider_body(self) -> None:
        obstacles = np.asarray([[0.2, 1.5], [0.5, 0.8], [-0.8, 0.6]])
        clearance, censored = profile_conditioned_clearance(
            obstacles,
            [0.1, 0.25, 0.6, 0.9],
            maximum_range_m=3.0,
        )
        self.assertTrue(np.all(np.diff(clearance) <= 0.0))
        self.assertEqual([True, False, False, False], censored.tolist())

    def test_widest_path_separates_equal_band_aggregates(self) -> None:
        blocked = np.ones((6, 6), dtype=np.float64)
        blocked[3, :] = 0.0
        routed = np.ones((6, 6), dtype=np.float64)
        routed[1, 0:2] = 0.0
        routed[3, 2:4] = 0.0
        routed[5, 4:6] = 0.0
        blocked_bands = [np.mean(part) for part in np.split(blocked, 3, axis=1)]
        routed_bands = [np.mean(part) for part in np.split(routed, 3, axis=1)]
        self.assertTrue(np.allclose(blocked_bands, routed_bands))
        self.assertEqual(0.0, widest_forward_path_capacity(blocked))
        self.assertEqual(1.0, widest_forward_path_capacity(routed))
        self.assertLess(
            abs(soft_widest_forward_path_capacity(routed, 0.01) - 1.0),
            abs(soft_widest_forward_path_capacity(routed, 0.20) - 1.0),
        )

    def test_full_canary_preserves_claim_boundaries_and_shift_counterexample(self) -> None:
        result = run_canary()
        self.assertEqual("PASS", result["status"])
        self.assertFalse(result["development_outcome_access"])
        self.assertFalse(result["confirmation_outcome_access"])
        self.assertFalse(result["model_or_checkpoint_access"])
        conformal = result["hypotheses"]["H4_ONE_SIDED_CONFORMAL_CLEARANCE"]
        self.assertTrue(conformal["iid_bound_observed"])
        self.assertTrue(conformal["distribution_shift_breaks_iid_observation"])
        self.assertLessEqual(
            conformal["conformal_false_clear_all_known"],
            conformal["conformal_miscoverage"],
        )
        feasibility = conformal["cluster_crc_feasibility"]
        self.assertEqual(12, minimum_exchangeable_units(0.08))
        self.assertEqual(19, minimum_exchangeable_units(0.05))
        self.assertFalse(feasibility["target_0_08_feasible_with_current_four_parents"])
        self.assertEqual(0.20, feasibility["current_best_case_finite_sample_term"])


if __name__ == "__main__":
    unittest.main()
