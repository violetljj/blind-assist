from __future__ import annotations

import unittest
from pathlib import Path

from merge_thor_magni_candidate_intake import ContractError, _id_set_from_rows, _new_event, _validate_candidate


class MergeThorMagniCandidateIntakeTest(unittest.TestCase):
    def _candidate(self) -> dict[str, object]:
        return {
            "schema": "hftf_d7_public_real_candidate_v1",
            "candidate_id": "d7cand-thor-test",
            "parent_event_id": "d7parent-thor-test",
            "dataset_id": "THOR-MAGNI",
            "source_session_id": "d7sess-thor-test",
            "ancestry_group": "d7anc-thor-test",
            "candidate_selection": "MODEL_BLIND_UNIFORM_QTM_100HZ_SOURCE_COVERAGE",
            "model_output_visible_to_selector": False,
            "native_geometry_used_for_selection": False,
            "event_bucket": "NOT_EVALUABLE",
            "truth_status": "NOT_EVALUABLE",
            "timestamp_semantics": "SOURCE_SYNCHRONIZED_QTM_TIME",
            "start_timestamp_ns": 10,
            "end_timestamp_ns": 20,
            "frame_ids": ["d7frm-thor-test"],
            "rgb_uri": "raw/thor-test/video.mp4",
            "source_license": "CC-BY-4.0",
            "source_hash": "hash",
            "source_metadata": {"source_sync_contract": "paper"},
        }

    def test_validation_accepts_source_synchronized_model_blind_candidate(self) -> None:
        candidate = self._candidate()
        _validate_candidate(candidate, session_id="d7sess-thor-test", ancestry="d7anc-thor-test")

    def test_event_preserves_source_time_and_only_assigns_terminal_shell(self) -> None:
        event = _new_event(self._candidate(), reason="REVIEW_PENDING", root=Path("F:\\d7"))
        self.assertEqual(event["start_timestamp_ns"], 10)
        self.assertEqual(event["end_timestamp_ns"], 20)
        self.assertEqual(event["event_bucket"], "NOT_EVALUABLE")
        self.assertFalse(event["review_model_output_visible"])
        self.assertIsNone(event["rgb_local_path"])

    def test_duplicate_intake_ids_fail_closed(self) -> None:
        with self.assertRaises(ContractError):
            _id_set_from_rows([{"candidate_id": "same"}, {"candidate_id": "same"}], "candidate_id")


if __name__ == "__main__":
    unittest.main()
