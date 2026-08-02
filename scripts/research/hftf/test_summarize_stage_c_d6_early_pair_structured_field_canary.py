import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_stage_c_d6_early_pair_structured_field_canary import (
    summarize_values,
)


class EarlyPairStructuredFieldSummaryTest(unittest.TestCase):
    def test_summarize_values_preserves_sign_counts(self):
        result = summarize_values([-0.2, 0.0, 0.1, 0.3])

        self.assertEqual(4, result["count"])
        self.assertAlmostEqual(0.05, result["mean"])
        self.assertAlmostEqual(0.05, result["median"])
        self.assertEqual(2, result["positive_count"])
        self.assertEqual(1, result["zero_count"])
        self.assertEqual(1, result["negative_count"])


if __name__ == "__main__":
    unittest.main()
