import unittest

from materialize_conservative_adjudication import _build_rows
from pipeline import ContractError


class ConservativeAdjudicationTest(unittest.TestCase):
    def _input(self, geometry_bucket: str = "NOT_EVALUABLE") -> dict[str, object]:
        roles = {
            role: {
                "event_bucket": "NOT_EVALUABLE",
                "decision": "NOT_EVALUABLE",
                "model_output_visible": False,
            }
            for role in (
                "RGB_REVIEWER_A",
                "RGB_REVIEWER_B",
                "RGB_REVIEWER_C",
                "GEOMETRY_EVIDENCE_REVIEWER",
                "COUNTEREXAMPLE_REVIEWER",
            )
        }
        roles["GEOMETRY_EVIDENCE_REVIEWER"]["event_bucket"] = geometry_bucket
        return {
            "batch_id": "batch-a",
            "candidate_id": "candidate-a",
            "event_id": "event-a",
            "adjudication_input_id": "input-a",
            "dataset_id": "EgoWalk",
            "model_output_visible": False,
            "review_records": roles,
        }

    def test_builds_only_fail_closed_terminal(self) -> None:
        rows = _build_rows([self._input()], batch_id="batch-a")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["adjudication_decision"], "NOT_EVALUABLE")
        self.assertEqual(rows[0]["admission_status"], "NOT_ADMITTED")
        self.assertIsNone(rows[0]["phase_intervals"])

    def test_refuses_geometry_gate_that_is_not_closed(self) -> None:
        with self.assertRaises(ContractError):
            _build_rows([self._input("NORMAL_WALKABLE_NEGATIVE")], batch_id="batch-a")


if __name__ == "__main__":
    unittest.main()
