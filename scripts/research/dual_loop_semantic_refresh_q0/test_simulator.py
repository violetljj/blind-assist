from __future__ import annotations

import unittest

from .simulator import (
    Frame,
    FullRatePolicy,
    FixedIntervalPolicy,
    FeedbackState,
    MaxAgePolicy,
    MissingFastFeatureStream,
    divergence_levels,
    simulate_arm,
)


def risk(level: str = "HIGH", proximity: str = "NEAR") -> dict:
    return {
        "level": level,
        "direction": "CENTER",
        "proximity": proximity,
        "evidence_state": "SUPPORTED_TARGET_EVIDENCE",
        "source_detection": None if level == "NONE" else {
            "class_id": 0,
            "label": "person",
            "source": "OBJECT_DETECTOR",
            "confidence": 0.9,
            "left": 10.0,
            "top": 10.0,
            "right": 30.0,
            "bottom": 60.0,
            "frame_width": 100,
            "frame_height": 100,
        },
    }


def frame(index: int, stable_risk: dict, feedback: bool) -> Frame:
    return Frame(
        session_id="s",
        frame_id=f"{index:03d}",
        timestamp_ns=1_000_000_000 + index * 50_000_000,
        detector_output_sha256=f"hash-{index}",
        detections=(),
        stable_risk=stable_risk,
        reference_feedback_triggered=feedback,
        reference_risk_event=None,
    )


class SimulatorTest(unittest.TestCase):
    def test_full_rate_reproduces_reference_feedback(self) -> None:
        frames = [frame(0, risk(), True), frame(1, risk(), False)]
        result = simulate_arm(frames, FullRatePolicy())
        self.assertEqual([True, False], [row["candidate_feedback_triggered"] for row in result.outputs])

    def test_fixed_interval_has_independent_cache_and_refreshes_by_time(self) -> None:
        frames = [frame(0, risk(), True), frame(1, risk("NONE", "FAR"), False), frame(2, risk(), True)]
        result = simulate_arm(
            frames,
            FixedIntervalPolicy(interval_ns=100_000_000, name="FIXED"),
        )
        self.assertEqual([True, False, True], [row["detector_ran"] for row in result.outputs])
        self.assertEqual("000", result.outputs[1]["cache_source_frame_id"])
        self.assertEqual("002", result.outputs[2]["cache_source_frame_id"])

    def test_feedback_state_does_not_share_cooldown_between_arms(self) -> None:
        state = FeedbackState()
        state.reset_for_session("s", 1_000_000_000)
        self.assertTrue(state.update(risk(), 1_000_000_000))
        self.assertFalse(state.update(risk(), 1_100_000_000))
        other = FeedbackState()
        other.reset_for_session("s", 1_000_000_000)
        self.assertTrue(other.update(risk(), 1_000_000_000))

    def test_divergence_is_reported_at_three_levels(self) -> None:
        cached = risk("NONE", "FAR")
        reference = risk("HIGH", "NEAR")
        result = divergence_levels(cached, reference, True, False)
        self.assertTrue(result["level1_any"])
        self.assertTrue(result["level2_any"])
        self.assertTrue(result["level3_any"])
        self.assertTrue(result["level3"]["event_active_changed"])

    def test_track_failure_policy_fails_closed_without_fast_features(self) -> None:
        with self.assertRaises(MissingFastFeatureStream):
            simulate_arm(
                [frame(0, risk(), True)],
                MaxAgePolicy(max_age_ns=100_000_000, name="AGE_TRACK_FAILURE"),
            )


if __name__ == "__main__":
    unittest.main()
