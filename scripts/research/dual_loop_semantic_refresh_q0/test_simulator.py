from __future__ import annotations

import unittest

from .simulator import (
    Frame,
    FullRatePolicy,
    FixedIntervalPolicy,
    FeedbackState,
    MaxAgePolicy,
    MissingFastFeatureStream,
    PROPAGATION_MODE,
    constrained_operating_point,
    divergence_levels,
    evaluate_episode_alignment,
    raw_nondominated_set,
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
        self.assertEqual(PROPAGATION_MODE, result.propagation_mode)
        self.assertEqual([True, False, True], [row["detector_ran"] for row in result.outputs])
        self.assertEqual("000", result.outputs[1]["cache_source_frame_id"])
        self.assertEqual("002", result.outputs[2]["cache_source_frame_id"])

    def test_active_event_id_increments_after_clear(self) -> None:
        result = simulate_arm(
            [
                frame(0, risk(), False),
                frame(1, risk("NONE", "FAR"), False),
                frame(2, risk(), False),
            ],
            FullRatePolicy(),
        )
        first_id = result.outputs[0]["candidate_event"]["active_event_id"]
        clear_id = result.outputs[1]["candidate_event"]["active_event_id"]
        second_id = result.outputs[2]["candidate_event"]["active_event_id"]
        self.assertIsNotNone(first_id)
        self.assertIsNone(clear_id)
        self.assertIsNotNone(second_id)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(3, result.outputs[2]["candidate_event"]["next_event_sequence_number"])

    def test_episode_alignment_reports_iou_delay_feedback_and_stale(self) -> None:
        candidate_rows = [
            {
                "session_id": "s",
                "frame_id": "000",
                "source_capture_timestamp_ns": 1_000_000_000,
                "risk": risk(),
                "feedback_triggered": True,
                "cache_age_ns": 0,
            },
            {
                "session_id": "s",
                "frame_id": "001",
                "source_capture_timestamp_ns": 1_050_000_000,
                "risk": risk(),
                "feedback_triggered": False,
                "cache_age_ns": 50_000_000,
            },
            {
                "session_id": "s",
                "frame_id": "002",
                "source_capture_timestamp_ns": 1_100_000_000,
                "risk": risk("NONE", "FAR"),
                "feedback_triggered": False,
                "cache_age_ns": 0,
            },
        ]
        reference_rows = [
            {
                **row,
                "feedback_triggered": index == 1,
                "cache_age_ns": 0,
            }
            for index, row in enumerate(candidate_rows)
        ]
        result = evaluate_episode_alignment(candidate_rows, reference_rows)
        self.assertEqual(1, result["matched_episode_count"])
        self.assertEqual(1.0, result["temporal_iou"]["mean"])
        self.assertEqual(0.0, result["onset_delay_ms"]["max"])
        self.assertEqual(1, result["feedback"]["matched_count"])
        self.assertEqual(50.0, result["candidate_longest_stale_duration_ms"])

    def test_constrained_selection_is_not_raw_pareto_front(self) -> None:
        def metrics(
            policy: str,
            calls: int,
            level3: int,
            recall: float,
            iou: float,
            onset_p95: float,
        ) -> dict:
            return {
                "status": "VALID",
                "policy": policy,
                "detector_call_count": calls,
                "divergence": {"level3_event_or_feedback_frame_count": level3},
                "truth_item_metrics": {"reference_event_missed_count": 0},
                "episode_alignment": {
                    "reference_episode_unmatched_count": 0,
                    "reference_episode_match_recall": recall,
                    "temporal_iou": {"mean": iou},
                    "absolute_onset_delay_ms": {"p95": onset_p95},
                    "risk_signature_match_rate": 1.0,
                },
            }

        fast = metrics("FAST", 10, 100, 0.96, 0.81, 149.0)
        accurate = metrics("ACCURATE", 20, 0, 1.0, 1.0, 0.0)
        self.assertEqual(["ACCURATE", "FAST"], raw_nondominated_set([fast, accurate]))
        selection = constrained_operating_point([fast, accurate])
        self.assertEqual(["ACCURATE", "FAST"], selection["admissible_set"])
        self.assertEqual("FAST", selection["best_operating_point"])

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
