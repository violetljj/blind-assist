import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_public_video_marker_radial_lifecycle_positive as subject


class MarkerRadialLifecyclePositiveTest(unittest.TestCase):
    def review(self) -> dict:
        return {
            "visual_risk_present_window_ms": [671000, 735000],
            "stable_route_clear_window_ms": [782000, 810000],
            "latest_acceptable_open_timestamp_ms": 675000,
        }

    def candidate(self) -> dict:
        return {
            "event_entry_timestamp_ms": 671000,
            "last_active_timestamp_ms": 692000,
            "radial_approach_passed": True,
        }

    def test_early_clear_preserves_false_clear_failure(self) -> None:
        base = [{
            "event_entry_timestamp_ms": 671000,
            "last_active_timestamp_ms": 692000,
            "confirmed_clear_timestamp_ms": 697000,
        }]
        result = subject.score_positive_source([self.candidate()], base, self.review())
        self.assertFalse(result["prospective_positive_passed"])
        self.assertEqual(38000, result["false_clear_gap_ms"])
        self.assertFalse(result["checks"]["event_not_cleared_before_visual_risk_ends"])

    def test_conservative_clear_after_risk_passes(self) -> None:
        base = [{
            "event_entry_timestamp_ms": 671000,
            "last_active_timestamp_ms": 692000,
            "confirmed_clear_timestamp_ms": 778000,
        }]
        result = subject.score_positive_source([self.candidate()], base, self.review())
        self.assertTrue(result["prospective_positive_passed"])
        self.assertEqual(0, result["false_clear_gap_ms"])

    def test_reopen_before_stable_clear_fails_same_event_once(self) -> None:
        base = [{
            "event_entry_timestamp_ms": 671000,
            "last_active_timestamp_ms": 692000,
            "confirmed_clear_timestamp_ms": 778000,
        }]
        second = {
            "event_entry_timestamp_ms": 701000,
            "last_active_timestamp_ms": 773000,
            "radial_approach_passed": True,
        }
        result = subject.score_positive_source([self.candidate(), second], base, self.review())
        self.assertFalse(result["prospective_positive_passed"])
        self.assertFalse(result["checks"]["same_event_not_reopened"])


if __name__ == "__main__":
    unittest.main()
