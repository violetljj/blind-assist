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


if __name__ == "__main__":
    unittest.main()
