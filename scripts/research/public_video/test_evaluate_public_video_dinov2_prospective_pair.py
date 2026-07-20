import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_public_video_dinov2_prospective_pair as evaluator


class DinoV2ProspectivePairEvaluatorTest(unittest.TestCase):
    def test_window_evaluation_orders_open_and_close(self):
        samples = [{"timestamp_ms": index * 1000, "vector": [value, 0.0]} for index, value in enumerate([0, 0, 0, 1, 1, 1, 0, 0, 0])]
        review = {"pre_risk_clear_window_ms": [0, 3000], "risk_present_window_ms": [3000, 6000], "stable_post_clear_window_ms": [6000, 9000]}
        result = evaluator.evaluate_windows(samples, review, np.array([1.0, 0.0]), minimum_samples=3)
        self.assertTrue(result["open_ordered"])
        self.assertTrue(result["close_ordered"])

    def test_rejects_overlapping_windows(self):
        samples = [{"timestamp_ms": index * 1000, "vector": [0.0]} for index in range(9)]
        review = {"pre_risk_clear_window_ms": [0, 4000], "risk_present_window_ms": [3000, 6000], "stable_post_clear_window_ms": [6000, 9000]}
        with self.assertRaisesRegex(ValueError, "ordered"):
            evaluator.evaluate_windows(samples, review, np.array([1.0]), minimum_samples=3)

    def test_rejects_too_few_samples(self):
        samples = [{"timestamp_ms": index * 1000, "vector": [0.0]} for index in range(6)]
        review = {"pre_risk_clear_window_ms": [0, 2000], "risk_present_window_ms": [2000, 4000], "stable_post_clear_window_ms": [4000, 6000]}
        with self.assertRaisesRegex(ValueError, "too few"):
            evaluator.evaluate_windows(samples, review, np.array([1.0]), minimum_samples=3)


if __name__ == "__main__":
    unittest.main()
