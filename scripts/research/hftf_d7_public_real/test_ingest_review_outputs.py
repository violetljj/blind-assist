from __future__ import annotations

import unittest

from ingest_review_outputs import _phase_valid, _validate_review_row
from pipeline import ContractError


class IngestReviewOutputsTest(unittest.TestCase):
    def test_phase_contract_is_required_for_support(self) -> None:
        self.assertFalse(_phase_valid("NORMAL_WALKABLE_NEGATIVE", None))
        self.assertTrue(_phase_valid(
            "NORMAL_WALKABLE_NEGATIVE",
            {"continuous_negative_interval": {"start_timestamp_ns": 1, "end_timestamp_ns": 2}},
        ))

    def test_completed_review_firewall_and_role_contract(self) -> None:
        row = {
            "schema": "hftf_d7_public_real_completed_review_v1",
            "record_kind": "COMPLETED_REVIEW",
            "review_role": "GEOMETRY_EVIDENCE_REVIEWER",
            "batch_id": "b",
            "candidate_id": "c",
            "review_completed": True,
            "decision": "NOT_EVALUABLE",
            "event_bucket": "NOT_EVALUABLE",
            "phase_intervals": None,
            "model_output_visible": False,
            "source_native_geometry_only": True,
            "counterexample_search_completed": False,
        }
        _validate_review_row(row, role="GEOMETRY_EVIDENCE_REVIEWER", batch_id="b")
        row["model_output_visible"] = True
        with self.assertRaises(ContractError):
            _validate_review_row(row, role="GEOMETRY_EVIDENCE_REVIEWER", batch_id="b")


if __name__ == "__main__":
    unittest.main()
