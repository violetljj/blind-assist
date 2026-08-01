from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_stage_c_f0_1_train_dev_corpus import (
    _add_label,
    _empty_summary,
    _validate_label,
)


class StageCF01CorpusValidationTest(unittest.TestCase):
    def test_nullable_label_round_trip(self) -> None:
        known = np.zeros((2, 6, 6), dtype=int)
        risk = np.full((2, 6, 6), None, dtype=object)
        known[0, 0, 0] = 1
        risk[0, 0, 0] = 1
        parsed_known, parsed_risk = _validate_label(
            {
                "known_target": known.tolist(),
                "risk_target_nullable": risk.tolist(),
            }
        )
        self.assertTrue(parsed_known[0, 0, 0])
        self.assertEqual(1, parsed_risk[0, 0, 0])

    def test_unknown_numeric_safe_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric iff"):
            _validate_label(
                {
                    "known_target": np.zeros((2, 6, 6), dtype=int).tolist(),
                    "risk_target_nullable": np.zeros(
                        (2, 6, 6), dtype=int
                    ).tolist(),
                }
            )

    def test_independent_aggregate_counts(self) -> None:
        known = np.zeros((2, 6, 6), dtype=bool)
        risk = np.zeros((2, 6, 6), dtype=np.uint8)
        known[0, 0, 0] = True
        risk[0, 0, 0] = 1
        summary = _empty_summary()
        _add_label(summary, "future", known, risk)
        self.assertEqual(1, summary["future"]["body"]["known"])
        self.assertEqual(1, summary["future"]["body"]["positive_known"])
        self.assertEqual(35, summary["future"]["body"]["unknown"])
        self.assertEqual(0, summary["future"]["head"]["known"])


if __name__ == "__main__":
    unittest.main()
