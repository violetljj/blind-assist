import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_public_video_marker_radial_approach_review as subject


class MarkerApproachReviewTest(unittest.TestCase):
    def test_negative_window_passes_without_overlap(self) -> None:
        candidates = {"sources": [{"source_id": "s", "events": []}]}
        review = {"sources": [{
            "source_id": "s", "evaluation_role": "prospective_negative_control",
            "review_window_ms": [1000, 2000],
            "visual_finding": {"pedestrian_corridor_risk_present": False, "should_open_risk_event": False},
        }]}
        rows, counts = subject.score_reviewed_sources(candidates, review)
        self.assertTrue(rows[0]["negative_control_passed"])
        self.assertEqual(1, counts["passed_negative"])

    def test_overlapping_candidate_fails_negative(self) -> None:
        candidates = {"sources": [{"source_id": "s", "events": [
            {"event_entry_timestamp_ms": 1200, "last_active_timestamp_ms": 1800}
        ]}]}
        review = {"sources": [{
            "source_id": "s", "evaluation_role": "prospective_negative_control",
            "review_window_ms": [1000, 2000],
            "visual_finding": {"pedestrian_corridor_risk_present": False, "should_open_risk_event": False},
        }]}
        rows, counts = subject.score_reviewed_sources(candidates, review)
        self.assertFalse(rows[0]["negative_control_passed"])
        self.assertEqual(0, counts["passed_negative"])

    def test_context_only_receives_no_gate_credit(self) -> None:
        candidates = {"sources": [{"source_id": "s", "events": []}]}
        review = {"sources": [{"source_id": "s", "evaluation_role": "context_only", "reason": "not scoped"}]}
        rows, counts = subject.score_reviewed_sources(candidates, review)
        self.assertFalse(rows[0]["gate_credit"])
        self.assertEqual(1, counts["context_only"])


if __name__ == "__main__":
    unittest.main()
