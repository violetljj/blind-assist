import unittest

from finalize_f1a_negative_supplement import evaluate_combined_gate


class FinalizeF1aNegativeSupplementTest(unittest.TestCase):
    def test_supplement_can_close_four_category_gate_without_changing_decisions(self) -> None:
        records = []
        for session in ("d1", "d2"):
            records.extend(
                [
                    {
                        "item_kind": "positive_event",
                        "input_id": session,
                        "session_id": session,
                        "parent_capture_id": session,
                        "role": "DECISION",
                    }
                    for _ in range(3)
                ]
            )
            records.append(
                {
                    "item_kind": "negative_window",
                    "negative_type": "A",
                    "input_id": session,
                    "session_id": session,
                    "parent_capture_id": session,
                    "role": "DECISION",
                }
            )
        for index, category in enumerate(
            ["A", "B", "B", "C", "C", "D", "D", "A", "B", "C"]
        ):
            records.append(
                {
                    "item_kind": "negative_window",
                    "negative_type": category,
                    "input_id": "dev" if index < 9 else "supp",
                    "session_id": "dev" if index < 9 else "supp",
                    "parent_capture_id": "dev" if index < 9 else "supp",
                    "role": "DEVELOPMENT" if index < 9 else "DEVELOPMENT_SUPPLEMENT",
                }
            )
        result = evaluate_combined_gate(records)
        self.assertEqual("READY", result["terminal"])
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
