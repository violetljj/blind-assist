from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_d2_calibration import core


def cohort(rows: list[tuple[str, str]]) -> dict:
    return {
        "claim_ceiling": "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY",
        "episodes": [{"evaluator_episode": {
            "goal_spec": {"target_name": parent}, "goal_reference_resolution": resolution,
        }} for parent, resolution in rows],
    }


class CalibrationCoreTest(unittest.TestCase):
    def test_conformal_quantile_uses_parent_level_finite_sample_rank(self) -> None:
        self.assertEqual(0.9, core.conformal_quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.1))
        with self.assertRaisesRegex(core.CalibrationError, "independent calibration parents"):
            core.conformal_quantile([0.1] * 8, 0.1)

    def test_action_semantics_do_not_confuse_set_with_ambiguity(self) -> None:
        self.assertEqual("COMMIT", core.calibrated_action("RESOLVABLE", ["b"]))
        self.assertEqual("SET", core.calibrated_action("RESOLVABLE", ["b", "c"]))
        self.assertEqual("AMBIGUOUS", core.calibrated_action("REFERENT_AMBIGUOUS", []))
        self.assertEqual("ABSTAIN", core.calibrated_action("INSUFFICIENT", []))
        with self.assertRaises(core.CalibrationError):
            core.calibrated_action("REFERENT_AMBIGUOUS", ["b", "c"])

    def test_runtime_feature_leakage_fails_closed(self) -> None:
        core.validate_runtime_features({"observed_feature_families": ["PLACE_IDENTITY"]})
        with self.assertRaisesRegex(core.CalibrationError, "leakage"):
            core.validate_runtime_features({"observed_feature_families": [], "goal_reference_resolution": "UNIQUE"})

    def test_frontdoor_counts_parents_not_frames(self) -> None:
        report = core.audit_data_frontdoor([cohort([
            *(('a', 'UNIQUE') for _ in range(20)),
            *(('b', 'AMBIGUOUS') for _ in range(20)),
        ])])
        self.assertEqual(1, report["parent_counts"]["UNIQUE"])
        self.assertEqual(1, report["parent_counts"]["AMBIGUOUS"])
        self.assertFalse(report["calibrator_fit_authorized"])


if __name__ == "__main__":
    unittest.main()
