import unittest

from compare_f1a_label_reviews import ReviewError
from finalize_f1a_label_repair import (
    canonical_disagreement_id,
    evaluate_gate,
    validate_cross_item_consistency,
)


class FinalizeF1aLabelRepairTest(unittest.TestCase):
    def test_canonicalizes_adjudicator_shorthand_identity(self) -> None:
        self.assertEqual("DISAGREEMENT-007", canonical_disagreement_id("D007"))
        self.assertEqual(
            "DISAGREEMENT-007", canonical_disagreement_id("DISAGREEMENT-007")
        )

    def test_ready_gate_counts_session_roles_and_categories(self) -> None:
        manifest = {
            "inputs": [
                {"input_id": "a", "role": "DECISION", "parent_capture_id": "a"},
                {"input_id": "b", "role": "DECISION", "parent_capture_id": "b"},
                {"input_id": "c", "role": "DEVELOPMENT", "parent_capture_id": "c"},
            ]
        }
        records = []
        for index in range(6):
            input_id = "a" if index < 3 else "b"
            records.append(
                {
                    "item_kind": "positive_event",
                    "input_id": input_id,
                    "session_id": input_id,
                    "role": "DECISION",
                }
            )
        categories = ["A", "A", "B", "B", "C", "C", "D", "D", "A", "B", "C", "D"]
        for index, category in enumerate(categories):
            input_id = "a" if index % 2 == 0 else "b"
            records.append(
                {
                    "item_kind": "negative_window",
                    "input_id": input_id,
                    "session_id": input_id,
                    "role": "DECISION",
                    "negative_type": category,
                }
            )
        spec = {
            "ready_gate": {
                "independent_capture_sessions_min": 3,
                "positive_events_min": 6,
                "positive_sessions_min": 2,
                "negative_windows_min": 12,
                "negative_categories_min": 4,
                "negative_windows_per_category_min": 2,
                "development_sessions": 1,
                "decision_sessions": 2,
            }
        }
        result = evaluate_gate(records, spec=spec, manifest=manifest)
        self.assertEqual("READY", result["terminal"])
        self.assertTrue(all(result["checks"].values()))

    def test_gate_holds_when_one_decision_has_no_positive(self) -> None:
        manifest = {
            "inputs": [
                {"input_id": "a", "role": "DECISION", "parent_capture_id": "a"},
                {"input_id": "b", "role": "DECISION", "parent_capture_id": "b"},
                {"input_id": "c", "role": "DEVELOPMENT", "parent_capture_id": "c"},
            ]
        }
        spec = {
            "ready_gate": {
                "independent_capture_sessions_min": 3,
                "positive_events_min": 1,
                "positive_sessions_min": 1,
                "negative_windows_min": 0,
                "negative_categories_min": 0,
                "negative_windows_per_category_min": 2,
                "development_sessions": 1,
                "decision_sessions": 2,
            }
        }
        records = [
            {
                "item_kind": "positive_event",
                "input_id": "a",
                "session_id": "a",
                "role": "DECISION",
            }
        ]
        result = evaluate_gate(records, spec=spec, manifest=manifest)
        self.assertEqual("HOLD_DATA", result["terminal"])
        self.assertFalse(result["checks"]["each_decision_has_positive"])

    def test_rejects_overlapping_natural_items(self) -> None:
        items = [
            {
                "item_kind": "positive_event",
                "input_id": "a",
                "onset_interval_seconds": {"start": 10.0, "end": 10.5},
                "end_or_clear_interval_seconds": {"start": 15.0, "end": 16.0},
            },
            {
                "item_kind": "negative_window",
                "input_id": "a",
                "window_interval_seconds": {"start": 14.0, "end": 18.0},
            },
        ]
        with self.assertRaises(ReviewError):
            validate_cross_item_consistency(items)


if __name__ == "__main__":
    unittest.main()
