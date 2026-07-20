import unittest

import evaluate_public_video_explicit_route_intent_oracle as subject


class ExplicitRouteIntentOracleTest(unittest.TestCase):
    def test_open_requires_consecutive_one_second_samples(self) -> None:
        frames = [
            {"timestamp_ms": 0, "trace_intrusion_score": 1.0},
            {"timestamp_ms": 2000, "trace_intrusion_score": 1.0},
        ]
        self.assertIsNone(subject.first_consecutive_open(frames, 1.0 / 3.0))

    def test_open_reports_second_sample(self) -> None:
        frames = [
            {"timestamp_ms": 0, "trace_intrusion_score": 0.0},
            {"timestamp_ms": 1000, "trace_intrusion_score": 1.0 / 3.0},
            {"timestamp_ms": 2000, "trace_intrusion_score": 2.0 / 3.0},
        ]
        self.assertEqual(2000, subject.first_consecutive_open(frames, 1.0 / 3.0))

    def test_gap_or_inactive_resets_consecutive_state(self) -> None:
        frames = [
            {"timestamp_ms": 0, "trace_intrusion_score": 1.0},
            {"timestamp_ms": 1000, "trace_intrusion_score": 0.0},
            {"timestamp_ms": 2000, "trace_intrusion_score": 1.0},
        ]
        self.assertIsNone(subject.first_consecutive_open(frames, 1.0 / 3.0))


if __name__ == "__main__":
    unittest.main()
