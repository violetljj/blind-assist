from __future__ import annotations

import unittest
from dataclasses import replace

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
                        source_id=source,
                        parent_id=f"parent-{parent}",
                        sequence_id=f"sequence-{parent}",
                        frame_id=f"frame-{frame}",
                        region_id=region,
                        approach_class=approach,
                        relative_risk=relative_risk,
                        high_risk=relative_risk >= 0.8,
                        time_seconds=frame * 0.1,
                        high_risk_onset_seconds=0.0 if relative_risk >= 0.8 else None,
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
        self.assertTrue(result["evaluable"])
        self.assertEqual(result["metric_support"]["parents_with_all_three_approach_classes"], 8)
        self.assertEqual(result["arms"]["A3"]["parent_macro"]["approaching_auroc"], 1.0)
        self.assertAlmostEqual(result["arms"]["A3"]["parent_macro"]["time_to_detection_mean_seconds"], 0.1)

    def test_unknown_cannot_buy_the_a3_gain(self) -> None:
        result = evaluate_o0(*self.fixture(buy_gain_with_unknown=True))
        self.assertFalse(result["passed"])
        self.assertTrue(result["evaluable"])
        self.assertFalse(result["gates"]["matched_coverage_not_bought_by_unknown"])

    def test_truth_unknown_remains_in_all_locked_identity_denominators(self) -> None:
        candidates, truth = self.fixture()
        unknown = TruthRow(
            source_id="source-0",
            parent_id="parent-0",
            sequence_id="sequence-0",
            frame_id="frame-unknown",
            region_id="center",
            truth_status="UNKNOWN_ALIGNMENT_SUPPORT",
            unknown_reason="LIDAR_RGB_ALIGNMENT_UNSUPPORTED",
        )
        truth.append(unknown)
        for arm in ("A0", "A1", "A2", "A3", "N0", "N1", "N2", "N3"):
            candidates.append(
                CandidateRow(
                    arm,
                    unknown.parent_id,
                    unknown.sequence_id,
                    unknown.frame_id,
                    unknown.region_id,
                    "VALID_RELATIVE_RISK",
                    0.0,
                    0.0,
                )
            )
        result = evaluate_o0(candidates, truth)
        parent = result["arms"]["A3"]["by_parent"]["parent-0"]
        self.assertEqual(parent["rows"], 19)
        self.assertEqual(parent["truth_labelable_rows"], 18)
        self.assertEqual(parent["candidate_valid_rows"], 19)
        self.assertEqual(parent["joint_evaluable_rows"], 18)
        self.assertAlmostEqual(parent["truth_labelability_coverage"], 18 / 19)
        self.assertEqual(parent["coverage"], parent["joint_evaluable_coverage"])
        overall = result["arms"]["A3"]["overall_all_locked_identities"]
        self.assertEqual(overall["rows"], 145)
        self.assertAlmostEqual(overall["truth_labelability_coverage"], 144 / 145)
        self.assertAlmostEqual(overall["joint_evaluable_coverage"], 144 / 145)
        self.assertEqual(result["truth_unknown_reasons"], {"LIDAR_RGB_ALIGNMENT_UNSUPPORTED": 1})

    def test_missing_parent_core_support_is_explicitly_not_evaluable(self) -> None:
        candidates, truth = self.fixture()
        truth = [
            replace(row, high_risk=False)
            if row.parent_id == "parent-0" and row.high_risk is True
            else row
            for row in truth
        ]
        result = evaluate_o0(candidates, truth)
        self.assertFalse(result["evaluable"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "SVRF_O0_NOT_EVALUABLE_LOCKED_COHORT")
        self.assertIsNone(result["arms"]["A3"]["parent_macro"]["false_clear_rate"])
        self.assertEqual(result["metric_support"]["parents_with_high_risk_rows"], 7)

    def test_unknown_truth_requires_reason_and_cannot_carry_labels(self) -> None:
        row = TruthRow(
            source_id="source",
            parent_id="parent",
            sequence_id="sequence",
            frame_id="frame",
            region_id="region",
            truth_status="UNKNOWN_ALIGNMENT_SUPPORT",
        )
        with self.assertRaisesRegex(ValueError, "explicit reason"):
            row.validate()
        with self.assertRaisesRegex(ValueError, "cannot carry evaluator labels"):
            replace(row, unknown_reason="missing", relative_risk=0.5).validate()


if __name__ == "__main__":
    unittest.main()
