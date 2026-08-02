from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from merge_sanpo_candidate_intake import _new_event_row, _new_review_row


class MergeSanpoCandidateIntakeTest(unittest.TestCase):
    def _candidate(self) -> dict[str, object]:
        return {
            "candidate_id": "candidate-a",
            "parent_event_id": "parent-a",
            "dataset_id": "SANPO-Real",
            "source_session_id": "session-a",
            "ancestry_group": "ancestry-a",
            "frame_ids": ["frame-a"],
            "source_license": "CC-BY-4.0",
            "source_hash": "sha256:source",
            "source_metadata": {"timestamp_semantics": "DERIVED_RELATIVE_NOMINAL"},
        }

    def test_new_event_is_not_evaluable_and_model_blind(self) -> None:
        row = _new_event_row(self._candidate(), reason="REVIEW_PENDING")
        self.assertEqual(row["event_bucket"], "NOT_EVALUABLE")
        self.assertEqual(row["truth_status"], "NOT_EVALUABLE")
        self.assertFalse(row["candidate_selection_model_visible"])
        self.assertFalse(row["review_model_output_visible"])
        self.assertEqual(row["frame_ids"], ["frame-a"])

    def test_geometry_and_counterexample_roles_keep_distinct_firewalls(self) -> None:
        geometry = _new_review_row(self._candidate(), role="GEOMETRY_EVIDENCE_REVIEWER", event_id="parent-a", reason="REVIEW_PENDING")
        counterexample = _new_review_row(self._candidate(), role="COUNTEREXAMPLE_REVIEWER", event_id="parent-a", reason="REVIEW_PENDING")
        self.assertTrue(geometry["source_native_geometry_only"])
        self.assertFalse(geometry["counterexample_search_required"])
        self.assertFalse(counterexample["source_native_geometry_only"])
        self.assertTrue(counterexample["counterexample_search_required"])


if __name__ == "__main__":
    unittest.main()
