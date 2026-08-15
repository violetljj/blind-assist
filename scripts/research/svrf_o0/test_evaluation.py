from __future__ import annotations

import unittest

from scripts.research.svrf_o0.evaluation import CandidateRow, TruthRow, evaluate_o0


class SvrfO0EvaluationTest(unittest.TestCase):
    def fixture(self, *, buy_gain_with_unknown: bool = False):
        truth = []
        candidates = []
        for parent in range(8):
            source = f"source-{parent % 2}"
            for frame in range(6):
                for region, relative_risk in (("left", 0.1), ("center", 0.9), ("right", 0.5)):
                    approach = "APPROACHING" if relative_risk >= 0.8 else "STABLE" if relative_risk >= 0.4 else "RECEDING"
                    row = TruthRow(
                        source,
                        f"parent-{parent}",
                        f"sequence-{parent}",
                        f"frame-{frame}",
                        region,
                        approach,
                        relative_risk,
                        relative_risk >= 0.8,
                        frame * 0.1,
                        0.0 if relative_risk >= 0.8 else None,
                    )
                    truth.append(row)
                    for arm in ("A0", "A1", "A2", "A3", "N0", "N1", "N2", "N3"):
                        valid = not (buy_gain_with_unknown and arm == "A3" and frame >= 3)
                        if arm == "A3":
                            score = 0.9 if relative_risk >= 0.8 else 0.0 if relative_risk >= 0.4 else -0.8
                            risk = 0.9 if relative_risk >= 0.8 else 0.4 if relative_risk >= 0.4 else 0.1
                        elif arm in {"A0", "A1", "A2"}:
                            score = 0.8 if relative_risk >= 0.4 else -0.5
                            risk = 0.4 if relative_risk >= 0.4 else 0.1
                        else:
                            score = 0.0
                            risk = 0.5
                        candidates.append(CandidateRow(arm, row.parent_id, row.sequence_id, row.frame_id, region, "VALID_RELATIVE_RISK" if valid else "UNKNOWN_QUALITY", score if valid else None, risk if valid else None))
        return candidates, truth

    def test_a3_passes_only_with_matched_gain_and_degraded_negative_controls(self) -> None:
        result = evaluate_o0(*self.fixture())
        self.assertTrue(result["passed"])
        self.assertEqual(result["best_single_arm"], "A0")
        self.assertEqual(result["matched_parent_effect"]["improved"], 8)
        self.assertEqual(result["matched_source_effect"]["improved"], 2)
        self.assertEqual(result["arms"]["A3"]["parent_macro"]["approaching_auroc"], 1.0)
        self.assertAlmostEqual(result["arms"]["A3"]["parent_macro"]["time_to_detection_mean_seconds"], 0.1)

    def test_unknown_cannot_buy_the_a3_gain(self) -> None:
        result = evaluate_o0(*self.fixture(buy_gain_with_unknown=True))
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["matched_coverage_not_bought_by_unknown"])


if __name__ == "__main__":
    unittest.main()
