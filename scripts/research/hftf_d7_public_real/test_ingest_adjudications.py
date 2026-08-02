from __future__ import annotations

import unittest

from ingest_adjudications import _phase_valid, _validate_row
from pipeline import ContractError


class IngestAdjudicationsTest(unittest.TestCase):
    def test_phase_contract(self) -> None:
        self.assertFalse(_phase_valid("BOUNDARY_LEVEL_CHANGE_POSITIVE", None))
        self.assertTrue(_phase_valid(
            "BOUNDARY_LEVEL_CHANGE_POSITIVE",
            {
                "pre_interval": {"start_timestamp_ns": 1, "end_timestamp_ns": 2},
                "alertable_interval": {"start_timestamp_ns": 2, "end_timestamp_ns": 3},
                "passed_clearance_interval": {"start_timestamp_ns": 3, "end_timestamp_ns": 4},
            },
        ))

    def test_not_evaluable_terminal_is_valid_but_admit_without_phase_is_not(self) -> None:
        event = {"candidate_id": "c", "event_id": "e"}
        terminal = {
            "schema": "hftf_d7_public_real_completed_adjudication_v1",
            "record_kind": "COMPLETED_ADJUDICATION",
            "batch_id": "b",
            "candidate_id": "c",
            "event_id": "e",
            "adjudication_decision": "NOT_EVALUABLE",
            "admission_status": "NOT_ADMITTED",
            "event_bucket": "NOT_EVALUABLE",
            "phase_intervals": None,
            "model_output_visible": False,
        }
        _validate_row(terminal, batch_id="b", event_by_candidate={"c": event})
        admit = dict(terminal)
        admit.update({
            "adjudication_decision": "ADMIT",
            "admission_status": "ADMITTED",
            "event_bucket": "NORMAL_WALKABLE_NEGATIVE",
        })
        with self.assertRaises(ContractError):
            _validate_row(admit, batch_id="b", event_by_candidate={"c": event})


if __name__ == "__main__":
    unittest.main()
