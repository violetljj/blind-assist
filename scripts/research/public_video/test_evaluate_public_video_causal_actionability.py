import unittest

import evaluate_public_video_causal_actionability as subject


class CausalActionabilityTest(unittest.TestCase):
    def test_consecutive_endpoint_requires_contiguous_seconds(self) -> None:
        frames = [
            {"timestamp_ms": 0, "trace_intrusion_score": 1.0},
            {"timestamp_ms": 2000, "trace_intrusion_score": 1.0},
        ]
        self.assertIsNone(subject.consecutive_endpoint(frames, lambda value: value >= 1 / 3, 2))

    def test_intervention_then_route_clear(self) -> None:
        event = self._event([0.0, 1.0, 1.0, 0.0, 0.0], role="true_radial_safe_lateral_negative")
        result = subject.classify_event(event, self._policy())
        self.assertEqual("intervention_then_route_clear", result["actionability_class"])
        self.assertTrue(result["eventual_safe_label_conflicts_with_causal_intervention"])
        self.assertEqual(2000, result["intervention_timestamp_ms"])
        self.assertEqual(4000, result["route_clear_timestamp_ms"])

    def test_persistent_intervention(self) -> None:
        event = self._event([0.0, 1.0, 1.0, 1.0, 0.0], role="prospective_positive_event")
        result = subject.classify_event(event, self._policy())
        self.assertEqual("persistent_intervention", result["actionability_class"])
        self.assertIsNone(result["route_clear_timestamp_ms"])

    def test_context_only(self) -> None:
        event = self._event([0.0, 0.5, 0.0, 0.0, 0.0], role="true_radial_safe_lateral_negative")
        result = subject.classify_event(event, self._policy())
        self.assertEqual("context_only", result["actionability_class"])
        self.assertFalse(result["eventual_safe_label_conflicts_with_causal_intervention"])

    @staticmethod
    def _policy() -> dict:
        return {
            "intervention_threshold": 1 / 3,
            "intervention_consecutive_one_second_samples": 2,
            "route_clear_threshold": 1 / 3,
            "route_clear_consecutive_one_second_samples": 2,
        }

    @staticmethod
    def _event(scores: list[float], role: str) -> dict:
        return {
            "source_id": "source",
            "role": role,
            "event_entry_timestamp_ms": 0,
            "event_last_active_timestamp_ms": (len(scores) - 1) * 1000,
            "tier_1_context_notice_timestamp_ms": 0,
            "lifecycle_clear_timestamp_ms": len(scores) * 1000,
            "causal_trace": {
                "frames": [
                    {"timestamp_ms": index * 1000, "trace_intrusion_score": score}
                    for index, score in enumerate(scores)
                ]
            },
        }


if __name__ == "__main__":
    unittest.main()
