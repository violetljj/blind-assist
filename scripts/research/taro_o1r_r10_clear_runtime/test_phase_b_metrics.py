from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r10_clear_runtime import phase_b_metrics as metrics


def _feature(query_id: str, occupied: bool) -> dict:
    return {
        "query_id": query_id,
        "r6_state": "UNKNOWN",
        "query_receipt": {"present": True},
        "occupied_hits": [[[False, False, occupied]]],
    }


def _records(*, clear_false_occupied: int = 0, clear_limit: int | None = None, unknown_occupied: int = 0):
    identities = [(f"p{i}", f"v{i}") for i in range(8)]
    sources = []
    labels = []
    clear_seen = 0
    clear_fp_seen = 0
    unknown_occ_seen = 0

    def append_frame(parent: str, frame_index: int, states: list[str]) -> None:
        nonlocal clear_seen, clear_fp_seen, unknown_occ_seen
        features = []
        query_labels = []
        for index, state in enumerate(states):
            query_id = f"{parent}:{frame_index}:{index}"
            occupied = state == "OCCUPIED_OBSERVED"
            if state == "CLEAR_OBSERVED":
                clear_seen += 1
                if clear_fp_seen < clear_false_occupied:
                    occupied = True
                    clear_fp_seen += 1
            if state == "UNKNOWN" and unknown_occ_seen < unknown_occupied:
                occupied = True
                unknown_occ_seen += 1
            features.append(_feature(query_id, occupied))
            query_labels.append({"query_id": query_id, "state": state})
        physical = f"{parent}:f{frame_index}"
        sources.append({"parent_id": parent, "physical_frame_id": physical, "query_features": features})
        labels.append({"physical_frame_id": physical, "query_labels": query_labels})

    for parent_index in range(6):
        append_frame(f"p{parent_index}", 0, ["OCCUPIED_OBSERVED"] * 9)
        append_frame(f"p{parent_index}", 1, ["OCCUPIED_OBSERVED"] * 9)
    clear_distribution = [3, 3, 3, 3, 1, 1, 1, 1]
    if clear_limit is not None:
        remaining = clear_limit
        clear_distribution = []
        for count in [3, 3, 3, 3, 1, 1, 1, 1]:
            selected = min(count, remaining)
            clear_distribution.append(selected)
            remaining -= selected
    for parent_index, clear_count in enumerate(clear_distribution):
        states = ["CLEAR_OBSERVED"] * clear_count + ["UNKNOWN"] * (9 - clear_count)
        append_frame(f"p{parent_index}", 2, states)
    return identities, sources, labels


class PhaseBMetricsTests(unittest.TestCase):
    def test_exact_evaluable_pass_includes_clear_wilson_gate(self) -> None:
        result = metrics.summarize(*_records())
        self.assertTrue(result["passed"])
        self.assertEqual(result["terminal"], metrics.PASS_TERMINAL)
        self.assertGreaterEqual(result["evaluability"]["definite_clear_query_count"], 12)
        self.assertTrue(result["gates"]["one_sided_95_wilson_clear_specificity_lower_bound"]["passed"])

    def test_one_below_clear_floor_is_not_evaluable(self) -> None:
        result = metrics.summarize(*_records(clear_limit=11))
        self.assertFalse(result["scientifically_evaluable"])
        self.assertEqual(result["terminal"], metrics.NOT_EVALUABLE_TERMINAL)

    def test_clear_false_occupied_fails_specificity(self) -> None:
        result = metrics.summarize(*_records(clear_false_occupied=2))
        self.assertTrue(result["scientifically_evaluable"])
        self.assertFalse(result["gates"]["clear_specificity_on_definite_clear"]["passed"])
        self.assertEqual(result["terminal"], metrics.FAIL_TERMINAL)

    def test_truth_unknown_predictions_do_not_enter_precision_denominator(self) -> None:
        result = metrics.summarize(*_records(unknown_occupied=20))
        self.assertEqual(result["occupied_predictions_on_truth_unknown"], 20)
        self.assertEqual(result["gates"]["occupied_precision_on_definite_labels"]["value"], 1.0)
        self.assertFalse(result["unknown_is_negative"])


if __name__ == "__main__":
    unittest.main()
