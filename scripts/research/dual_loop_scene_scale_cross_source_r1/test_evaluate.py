import unittest

from .evaluate import score_interval


class EvaluateTest(unittest.TestCase):
    def test_interval_is_inclusive_and_uses_named_branch(self) -> None:
        rows = [
            {
                "frame_id": index,
                "source_capture_timestamp_ns": index * 100_000_000,
                "baseline_feedback_triggered": index in (10, 20),
                "candidate_feedback_triggered": index == 20,
            }
            for index in range(30)
        ]
        baseline = score_interval(
            rows, 1.0, 2.0, "baseline_feedback_triggered"
        )
        candidate = score_interval(
            rows, 1.0, 2.0, "candidate_feedback_triggered"
        )
        self.assertEqual(2, baseline["trigger_count"])
        self.assertEqual(10, baseline["first_trigger_frame_id"])
        self.assertEqual(1, candidate["trigger_count"])
        self.assertEqual(20, candidate["first_trigger_frame_id"])


if __name__ == "__main__":
    unittest.main()
