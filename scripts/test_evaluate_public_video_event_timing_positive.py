import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_public_video_event_timing_positive as subject


REVIEW = {
    "material_risk_onset_ms": 10000,
    "latest_useful_reminder_ms": 13000,
    "reviewed_risk_end_ms": 15000,
    "stable_post_clear_window_ms": [17000, 22000],
}


class PublicVideoEventTimingPositiveTest(unittest.TestCase):
    def test_fixed_early_warning_lower_bound_is_inclusive(self) -> None:
        checks = subject.timing_checks(
            reminder_timestamp_ms=7000,
            route_delta=0.1,
            event={"confirmed_clear_timestamp_ms": 22000},
            reminder_count=1,
            review=REVIEW,
            maximum_early_warning_lead_ms=3000,
        )
        self.assertTrue(all(checks.values()))

    def test_reminder_before_fixed_band_fails(self) -> None:
        checks = subject.timing_checks(
            reminder_timestamp_ms=6999,
            route_delta=0.1,
            event={"confirmed_clear_timestamp_ms": 22000},
            reminder_count=1,
            review=REVIEW,
            maximum_early_warning_lead_ms=3000,
        )
        self.assertFalse(checks["reminder_inside_fixed_early_warning_band"])

    def test_route_veto_and_repeated_reminder_fail(self) -> None:
        checks = subject.timing_checks(
            reminder_timestamp_ms=10000,
            route_delta=0.0,
            event={"confirmed_clear_timestamp_ms": 18000},
            reminder_count=2,
            review=REVIEW,
            maximum_early_warning_lead_ms=3000,
        )
        self.assertFalse(checks["route_relation_supports_entry"])
        self.assertFalse(checks["same_visual_episode_reminder_once"])


if __name__ == "__main__":
    unittest.main()
