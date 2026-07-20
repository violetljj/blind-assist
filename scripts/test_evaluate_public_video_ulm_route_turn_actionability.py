import unittest

import evaluate_public_video_ulm_route_turn_actionability as subject


class UlmRouteTurnActionabilityTest(unittest.TestCase):
    def test_selects_route_turn_rejection(self) -> None:
        review = {
            "source": {"continuous_ego_pedestrian_capture": True},
            "true_radial_safe_lateral_negative": {
                "rejected_frozen_candidates": [{"candidate_id": "c01", "reason": "real barrier and actual route turn"}]
            },
        }
        row = subject.select_route_turn_rejection(review, "c01", "actual route turn")
        self.assertEqual("c01", row["candidate_id"])

    def test_rejects_reason_drift(self) -> None:
        review = {
            "source": {"continuous_ego_pedestrian_capture": True},
            "true_radial_safe_lateral_negative": {
                "rejected_frozen_candidates": [{"candidate_id": "c01", "reason": "detector confusion"}]
            },
        }
        with self.assertRaisesRegex(ValueError, "reason mismatch"):
            subject.select_route_turn_rejection(review, "c01", "actual route turn")

    def test_selects_exact_frozen_candidate(self) -> None:
        candidates = {"sources": [{
            "source_id": "source",
            "events": [{
                "event_entry_timestamp_ms": 1000,
                "last_active_timestamp_ms": 3000,
                "radial_approach_passed": True,
            }],
        }]}
        row = subject.select_candidate(candidates, "source", 1000, 3000)
        self.assertTrue(row["radial_approach_passed"])


if __name__ == "__main__":
    unittest.main()
