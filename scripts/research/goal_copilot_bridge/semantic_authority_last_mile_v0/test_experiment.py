import unittest

from .experiment import KINDS, build_cohort, run_experiment


class SemanticAuthorityLastMileV0Test(unittest.TestCase):
    def test_cohort_is_balanced_and_deterministic(self) -> None:
        first = build_cohort(seed=7, per_kind=3)
        second = build_cohort(seed=7, per_kind=3)
        self.assertEqual(first, second)
        self.assertEqual({kind: sum(row.kind == kind for row in first) for kind in KINDS}, {kind: 3 for kind in KINDS})

    def test_geometry_arm_changes_approach_not_identity(self) -> None:
        report = run_experiment(seed=240824, per_kind=4)
        self.assertEqual(report["identity_contract"], "EXACT_SEMANTIC_AUTHORITY_FIXED_GEOMETRY_CANNOT_REBIND")
        self.assertEqual(report["cohort"]["episode_count"], 12)
        baseline = report["metrics"]["bbox_center_scale"]
        challenger = report["metrics"]["sage_lm"]
        self.assertGreater(challenger["direction_accuracy"], baseline["direction_accuracy"])
        self.assertGreater(challenger["target_front_arrival_rate"], baseline["target_front_arrival_rate"])
        self.assertLess(challenger["median_endpoint_lateral_error_m"], baseline["median_endpoint_lateral_error_m"])
        self.assertLess(challenger["premature_arrival_count"], baseline["premature_arrival_count"])
        self.assertGreater(challenger["completion_precision"], 0.85)
        self.assertEqual(challenger["movement_steps_while_lost"], 0)


if __name__ == "__main__":
    unittest.main()
