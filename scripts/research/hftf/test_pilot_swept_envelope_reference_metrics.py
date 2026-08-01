from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_swept_envelope_reference_metrics import (
    _confusion,
    _paired_correctness,
    _pixel_lattices_disjoint,
    _summarize_counts,
)


class SweptEnvelopeReferenceMetricsTest(unittest.TestCase):
    def test_frozen_pixel_lattices_are_disjoint(self) -> None:
        self.assertTrue(_pixel_lattices_disjoint(8, 4, 4, 2))
        self.assertFalse(_pixel_lattices_disjoint(8, 4, 4, 0))

    def test_confusion_respects_known_mask(self) -> None:
        prediction = np.asarray([True, True, False, False, True])
        reference = np.asarray([True, False, True, False, False])
        known = np.asarray([True, True, True, True, False])
        self.assertEqual(
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
            _confusion(prediction, reference, known),
        )

    def test_paired_correctness_is_directional(self) -> None:
        candidate = np.asarray([True, False, True, False])
        baseline = np.asarray([False, True, True, False])
        reference = np.asarray([True, True, False, False])
        known = np.ones(4, dtype=bool)
        self.assertEqual(
            {
                "candidate_only_correct": 1,
                "baseline_only_correct": 1,
            },
            _paired_correctness(
                candidate, baseline, reference, known
            ),
        )

    def test_summary_is_json_serializable(self) -> None:
        summary = _summarize_counts(
            {
                "tp": 3,
                "fp": 1,
                "fn": 1,
                "tn": 5,
                "candidate_only_correct": 2,
                "baseline_only_correct": 1,
            }
        )
        self.assertAlmostEqual(0.75, summary["f1"])
        self.assertAlmostEqual(0.8, summary["accuracy"])
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
