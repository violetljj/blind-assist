from __future__ import annotations

import copy
import unittest

from scripts.research.taro_o1r_r11_abstention_runtime import phase_b_metrics as metrics


def row(query_id: str, state: str, index: int) -> dict:
    return {"query_id": query_id, "grid_index": index, "state": state}


def frame(identity: tuple[str, str], token: str, truth_states: list[str]) -> tuple[dict, dict, dict]:
    baseline_states = ["OCCUPIED_OBSERVED" if state == "OCCUPIED_OBSERVED" else "UNKNOWN" for state in truth_states]
    candidate_states = list(baseline_states)
    common = {
        "parent_id": identity[0],
        "video_id": identity[1],
        "physical_frame_id": f"{identity[1]}:{token}",
    }
    baseline = {**common, "query_results": [row(f"q{token}-{i}", state, i) for i, state in enumerate(baseline_states)]}
    candidate = {**common, "query_results": [row(f"q{token}-{i}", state, i) for i, state in enumerate(candidate_states)]}
    label = {**common, "query_labels": [row(f"q{token}-{i}", state, i) for i, state in enumerate(truth_states)]}
    return baseline, candidate, label


def passing_fixture() -> tuple[list[tuple[str, str]], list[dict], list[dict], list[dict]]:
    identities = [(f"p{i:02d}", f"v{i:02d}") for i in range(24)]
    baselines: list[dict] = []
    candidates: list[dict] = []
    labels: list[dict] = []
    for index, identity in enumerate(identities[:12]):
        for repeat in range(2):
            values = frame(identity, f"o{index:02d}{repeat}", ["OCCUPIED_OBSERVED"] * 9)
            for target, value in zip((baselines, candidates, labels), values, strict=True):
                target.append(value)
    for index, identity in enumerate(identities[12:]):
        values = frame(identity, f"c{index:02d}", ["CLEAR_OBSERVED", "CLEAR_OBSERVED"] + ["UNKNOWN"] * 7)
        for target, value in zip((baselines, candidates, labels), values, strict=True):
            target.append(value)
    return identities, baselines, candidates, labels


class R11PhaseBMetricsTests(unittest.TestCase):
    def test_perfect_dual_class_fixture_passes_all_frozen_gates(self) -> None:
        summary = metrics.summarize(*passing_fixture())
        self.assertEqual(summary["terminal"], metrics.PASS_TERMINAL)
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["scientifically_evaluable"])
        self.assertEqual(summary["evaluability"]["definite_occupied_query_count"], 216)
        self.assertEqual(summary["evaluability"]["physical_frames_with_definite_clear"], 12)
        self.assertEqual(summary["evaluability"]["definite_clear_query_count"], 24)
        self.assertTrue(all(row["passed"] for row in summary["gates"].values()))

    def test_missing_one_clear_frame_is_not_evaluable_before_effect_gates(self) -> None:
        fixture = passing_fixture()
        for collection in fixture[1:]:
            collection.pop()
        summary = metrics.summarize(*fixture)
        self.assertEqual(summary["terminal"], metrics.NOT_EVALUABLE_TERMINAL)
        self.assertFalse(summary["scientifically_evaluable"])
        self.assertEqual(summary["evaluability"]["physical_frames_with_definite_clear"], 11)

    def test_r11_recall_loss_fails_fixed_gate(self) -> None:
        identities, baselines, candidates, labels = passing_fixture()
        for candidate in candidates[:3]:
            candidate["query_results"][0]["state"] = "UNKNOWN"
        summary = metrics.summarize(identities, baselines, candidates, labels)
        self.assertEqual(summary["terminal"], metrics.FAIL_TERMINAL)
        self.assertFalse(summary["gates"]["micro_occupied_recall_loss_vs_r7"]["passed"])

    def test_clear_false_positive_fails_query_and_frame_specificity(self) -> None:
        identities, baselines, candidates, labels = passing_fixture()
        for baseline, candidate in zip(baselines[-3:], candidates[-3:], strict=True):
            baseline["query_results"][0]["state"] = "OCCUPIED_OBSERVED"
            candidate["query_results"][0]["state"] = "OCCUPIED_OBSERVED"
        summary = metrics.summarize(identities, baselines, candidates, labels)
        self.assertEqual(summary["terminal"], metrics.FAIL_TERMINAL)
        self.assertFalse(summary["gates"]["clear_frame_specificity"]["passed"])

    def test_abstention_effect_is_reported_but_not_required_for_absolute_pass(self) -> None:
        identities, baselines, candidates, labels = passing_fixture()
        baselines[-1]["query_results"][0]["state"] = "OCCUPIED_OBSERVED"
        baselines[-2]["query_results"][0]["state"] = "OCCUPIED_OBSERVED"
        summary = metrics.summarize(identities, baselines, candidates, labels)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["abstention_effect"]["abstained_definite_clear_frames"], 2)
        self.assertTrue(summary["abstention_effect"]["effect_evaluable"])

    def test_unknown_truth_never_enters_precision_or_clear_denominators(self) -> None:
        identities, baselines, candidates, labels = passing_fixture()
        baselines[-1]["query_results"][-1]["state"] = "OCCUPIED_OBSERVED"
        candidates[-1]["query_results"][-1]["state"] = "OCCUPIED_OBSERVED"
        summary = metrics.summarize(identities, baselines, candidates, labels)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["r11_occupied_predictions_on_truth_unknown"], 1)
        self.assertFalse(summary["unknown_is_negative"])

    def test_r11_occupied_must_be_r7_subset(self) -> None:
        identities, baselines, candidates, labels = passing_fixture()
        candidates[-1]["query_results"][0]["state"] = "OCCUPIED_OBSERVED"
        with self.assertRaises(metrics.R11PhaseBMetricsError):
            metrics.summarize(identities, baselines, candidates, labels)


if __name__ == "__main__":
    unittest.main()
