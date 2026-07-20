import unittest

import audit_public_video_temporal_risk_profile_pair_error as subject


class TemporalRiskProfilePairErrorAuditTest(unittest.TestCase):
    def test_detects_teacher_separation_and_safe_lateral_threshold_failure(self) -> None:
        checks = subject.pair_checks(
            negative_model_score=0.80,
            positive_model_score=0.88,
            negative_teacher_score=1.0 / 9.0,
            positive_teacher_score=1.0 / 3.0,
            threshold=0.68,
        )
        self.assertTrue(checks["offline_teacher_orders_pair"])
        self.assertTrue(checks["causal_head_orders_pair"])
        self.assertTrue(checks["positive_head_score_passes_threshold"])
        self.assertTrue(checks["safe_lateral_head_score_fails_threshold"])

    def test_rejects_reversed_teacher_order(self) -> None:
        checks = subject.pair_checks(0.5, 0.7, 0.4, 0.2, 0.68)
        self.assertFalse(checks["offline_teacher_orders_pair"])


if __name__ == "__main__":
    unittest.main()
