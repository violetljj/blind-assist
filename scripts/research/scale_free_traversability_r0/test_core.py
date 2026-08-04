from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from core import (
    BAND_SCORE_PERCENTILE,
    BANDS,
    CAUSAL_WINDOW,
    MINIMUM_MARGIN_LOG_UNITS,
    MINIMUM_VALID_FRACTION,
    MINIMUM_WINNER_COUNT,
    ROW_BASELINE_PERCENTILE,
    decide_relative_open,
    score_relative_intrusion,
)


class ScaleFreeTraversabilityR0Test(unittest.TestCase):
    def test_committed_protocol_matches_implementation(self) -> None:
        protocol_path = (
            Path(__file__).resolve().parents[3]
            / "docs/research/hftf/SCALE_FREE_TRAVERSABILITY_R0_PROTOCOL_2026-08-04.json"
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        operator = protocol["operator"]
        decision = protocol["decision"]
        self.assertEqual(operator["bands_x"], {name: list(bounds) for name, bounds in BANDS.items()})
        self.assertEqual(operator["minimum_valid_fraction"], MINIMUM_VALID_FRACTION)
        self.assertEqual(operator["row_baseline_percentile"], ROW_BASELINE_PERCENTILE)
        self.assertEqual(operator["band_score_percentile"], BAND_SCORE_PERCENTILE)
        self.assertEqual(decision["causal_window"], CAUSAL_WINDOW)
        self.assertEqual(decision["minimum_margin_log_units"], MINIMUM_MARGIN_LOG_UNITS)
        self.assertEqual(decision["minimum_winner_count"], MINIMUM_WINNER_COUNT)

    def test_scores_are_invariant_to_global_depth_scale(self) -> None:
        depth = np.linspace(1.0, 4.0, 120 * 160, dtype=np.float64).reshape(120, 160)
        first = score_relative_intrusion(depth)
        second = score_relative_intrusion(depth * 7.25)
        self.assertEqual("VALID", first["status"])
        self.assertEqual("VALID", second["status"])
        for band in first["scores"]:
            self.assertAlmostEqual(first["scores"][band], second["scores"][band], places=12)

    def test_near_center_intrusion_makes_center_score_larger(self) -> None:
        depth = np.full((120, 160), 4.0, dtype=np.float64)
        depth[36:108, 64:96] = 1.0
        result = score_relative_intrusion(depth)
        self.assertEqual("VALID", result["status"])
        self.assertGreater(result["scores"]["center"], result["scores"]["left"])
        self.assertGreater(result["scores"]["center"], result["scores"]["right"])

    def test_decision_requires_warmup_and_stable_winner(self) -> None:
        row = {"left": 0.10, "center": 0.30, "right": 0.25}
        self.assertEqual("UNKNOWN_WARMUP", decide_relative_open([row] * 4)["reason"])
        result = decide_relative_open([row] * 5)
        self.assertEqual("RELATIVELY_OPEN_LEFT", result["label"])
        self.assertEqual(5, result["winner_count"])

    def test_decision_abstains_when_margin_is_small(self) -> None:
        row = {"left": 0.10, "center": 0.15, "right": 0.30}
        self.assertEqual("AMBIGUOUS", decide_relative_open([row] * 5)["label"])


if __name__ == "__main__":
    unittest.main()
