import math
import unittest

from scripts.research.hftf.deployment.depthart.depthart_task_preserving_d2_task_head_canary import (
    TaskHeadPolicy,
    compose_band,
    monotone_occupancy_probabilities,
    run_canary,
)


class TaskHeadCanaryTest(unittest.TestCase):
    def test_monotone_projection_never_reclears_later_horizon(self) -> None:
        probabilities = monotone_occupancy_probabilities((4.0, -4.0, -4.0))
        self.assertTrue(all(left <= right for left, right in zip(probabilities, probabilities[1:])))

    def test_hard_evidence_overrides_confident_logits(self) -> None:
        result = compose_band(
            occupancy_logits=(4.0, 4.0, 4.0),
            known_probability=1.0,
            raw_clearance_m=0.5,
            residual_logit=0.0,
            valid_depth_fraction=0.8,
            ground_support_fraction=0.2,
            band_support_points=0,
            ground_plane_available=True,
        )
        self.assertEqual(result["states"], ["UNKNOWN_GROUND"] * 3)

    def test_residual_is_bounded(self) -> None:
        policy = TaskHeadPolicy(maximum_clearance_residual_m=0.5)
        result = compose_band(
            occupancy_logits=(-4.0, -4.0, -4.0),
            known_probability=1.0,
            raw_clearance_m=1.0,
            residual_logit=100.0,
            valid_depth_fraction=0.8,
            ground_support_fraction=0.2,
            band_support_points=100,
            ground_plane_available=True,
            policy=policy,
        )
        self.assertTrue(math.isclose(result["clearance_m"], 1.5, abs_tol=1e-12))

    def test_frozen_canary_passes(self) -> None:
        result = run_canary()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
