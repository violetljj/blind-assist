from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from normalize_adjudication_terminals import normalize
from pipeline import ContractError


def _row(decision: str, *, admission_status: str = "NOT_ADMITTED", event_bucket: str = "NOT_EVALUABLE") -> dict[str, object]:
    return {
        "schema": "hftf_d7_public_real_completed_adjudication_v1",
        "record_kind": "COMPLETED_ADJUDICATION",
        "candidate_id": "d7cand-test",
        "event_id": "d7event-test",
        "adjudication_decision": decision,
        "admission_status": admission_status,
        "event_bucket": event_bucket,
        "model_output_visible": False,
    }


class NormalizeAdjudicationTerminalsTest(unittest.TestCase):
    def test_legacy_final_fields_are_canonicalized_without_changing_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final.jsonl"
            legacy = _row("NOT_EVALUABLE")
            legacy["decision"] = legacy.pop("adjudication_decision")
            legacy["schema"] = "hftf_d7_public_real_final_adjudication_v1"
            legacy["record_kind"] = "FINAL_ADJUDICATION"
            path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

            result = normalize(path, expected_count=1, canonicalize_legacy_final=True)

            self.assertEqual(result["decision_aliases"], 1)
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["adjudication_decision"], "NOT_EVALUABLE")
            self.assertNotIn("decision", row)
            self.assertEqual(row["schema"], "hftf_d7_public_real_completed_adjudication_v1")
            self.assertEqual(row["record_kind"], "COMPLETED_ADJUDICATION")

    def test_safe_not_admit_is_rebound_to_frozen_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final.jsonl"
            path.write_text(json.dumps(_row("NOT_ADMIT")) + "\n", encoding="utf-8")

            result = normalize(path, expected_count=1)

            self.assertEqual(result["rebound_not_admit"], 1)
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["adjudication_decision"], "NOT_EVALUABLE")

    def test_unsafe_not_admit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final.jsonl"
            path.write_text(json.dumps(_row("NOT_ADMIT", event_bucket="NORMAL_WALKABLE_NEGATIVE")) + "\n", encoding="utf-8")

            with self.assertRaises(ContractError):
                normalize(path, expected_count=1)

    def test_legacy_not_admitted_decision_is_rebound_only_for_unevaluable_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final.jsonl"
            legacy = _row("NOT_ADMITTED", admission_status=None)
            path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

            result = normalize(path, expected_count=1)

            self.assertEqual(result["rebound_not_admitted_decision"], 1)
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["adjudication_decision"], "NOT_EVALUABLE")
            self.assertEqual(row["admission_status"], "NOT_ADMITTED")

    def test_unsafe_not_admitted_decision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final.jsonl"
            legacy = _row("NOT_ADMITTED", admission_status=None, event_bucket="NORMAL_WALKABLE_NEGATIVE")
            path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

            with self.assertRaises(ContractError):
                normalize(path, expected_count=1)


if __name__ == "__main__":
    unittest.main()
