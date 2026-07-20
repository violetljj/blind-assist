import unittest

import audit_public_video_actionability_relabels as subject


class ActionabilityRelabelAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "intervention_threshold": 1 / 3,
            "intervention_consecutive_one_second_samples": 2,
            "route_clear_threshold": 1 / 3,
            "route_clear_consecutive_one_second_samples": 2,
        }

    @staticmethod
    def frames(scores: list[float | None]) -> list[dict]:
        return [
            {"timestamp_ms": index * 1000, "trace_intrusion_score": score}
            for index, score in enumerate(scores)
        ]

    def test_context_only(self) -> None:
        result = subject.replay(self.frames([0.0, 1.0, 0.0]), self.policy)
        self.assertEqual("context_only", result["actionability_class"])

    def test_intervention_then_clear_and_reentry(self) -> None:
        result = subject.replay(self.frames([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]), self.policy)
        self.assertEqual("intervention_then_route_clear", result["actionability_class"])
        self.assertEqual(2, result["intervention_episode_count"])

    def test_persistent_intervention(self) -> None:
        result = subject.replay(self.frames([0.0, 1.0, 1.0, 0.0]), self.policy)
        self.assertEqual("persistent_intervention", result["actionability_class"])

    def test_gap_breaks_consecutive_run(self) -> None:
        frames = [
            {"timestamp_ms": 0, "trace_intrusion_score": 1.0},
            {"timestamp_ms": 2000, "trace_intrusion_score": 1.0},
        ]
        result = subject.replay(frames, self.policy)
        self.assertEqual("context_only", result["actionability_class"])

    def test_invalid_score_is_reported_and_breaks_run(self) -> None:
        result = subject.replay(self.frames([1.0, None, 1.0]), self.policy)
        self.assertEqual("context_only", result["actionability_class"])
        self.assertEqual(1, result["invalid_causal_score_count"])


if __name__ == "__main__":
    unittest.main()
