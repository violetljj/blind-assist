import copy
import unittest

import evaluate_public_silver_retrospective_close_stress as evaluator


def review() -> dict:
    return {
        "schema": evaluator.REVIEW_SCHEMA,
        "decision": evaluator.REVIEW_DECISION,
        "chronology_attestation": {
            "original_frame_order_used": True,
            "hard_cut_observed": False,
        },
        "policy_frozen_before_detector_scoring": True,
        "risk_window": {"frame_indices": [1, 2]},
        "clear_window": {"frame_indices": [3, 4]},
        "limitations": {
            "new_source": False,
            "prospective": False,
            "independent_from_prior_source_review": False,
            "eligible_for_r712_positive_source_gate": False,
            "training_authorized": False,
            "calibration_authorized": False,
            "blind_evaluation_authorized": False,
            "android_integration_authorized": False,
            "production_authorized": False,
        },
    }


class RetrospectiveCloseStressTest(unittest.TestCase):
    def test_decreasing_score_closes_event(self) -> None:
        result = evaluator.score_change(risk_score=0.3, clear_score=0.1)
        self.assertTrue(result["correct"])
        self.assertEqual(result["predicted_transition"], "close_event")

    def test_equal_score_abstains(self) -> None:
        result = evaluator.score_change(risk_score=0.2, clear_score=0.2)
        self.assertFalse(result["correct"])
        self.assertEqual(result["predicted_transition"], "abstain")

    def test_review_requires_frozen_pre_score_policy(self) -> None:
        value = review()
        value["policy_frozen_before_detector_scoring"] = False
        with self.assertRaisesRegex(ValueError, "before detector"):
            evaluator.validate_review(value)

    def test_review_rejects_hard_cut(self) -> None:
        value = review()
        value["chronology_attestation"]["hard_cut_observed"] = True
        with self.assertRaisesRegex(ValueError, "Hard-cut|hard-cut"):
            evaluator.validate_review(value)

    def test_review_cannot_claim_prospective_source(self) -> None:
        value = review()
        value["limitations"]["prospective"] = True
        with self.assertRaisesRegex(ValueError, "overstates"):
            evaluator.validate_review(value)

    def test_risk_window_must_precede_clear_window(self) -> None:
        value = review()
        value["clear_window"]["frame_indices"] = [2, 3]
        with self.assertRaisesRegex(ValueError, "precede"):
            evaluator.validate_review(value)


if __name__ == "__main__":
    unittest.main()
