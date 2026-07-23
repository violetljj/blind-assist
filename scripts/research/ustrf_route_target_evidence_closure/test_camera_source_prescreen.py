from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import validate_camera_source_prescreen as validator


class CameraSourcePrescreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.roster_path = (
            cls.repo
            / "configs/ustrf_route_target_evidence_closure_r1_camera_source_prescreen_0327.json"
        )
        cls.roster = json.loads(cls.roster_path.read_text(encoding="utf-8"))
        cls.execution_path = (
            cls.repo
            / "configs/ustrf_route_target_evidence_closure_r1_camera_source_prescreen_0327_execution.json"
        )
        cls.freeze_receipt_path = (
            cls.repo
            / "artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/"
            "camera-source-prescreen-0327-execution-freeze-receipt-r1.json"
        )

    def test_exact_predecode_roster_passes(self) -> None:
        result = validator.validate_roster(self.repo, self.roster)
        self.assertEqual(result["decision"], "VALID_PREDECODE_REJECT_ONLY_CANARY_ROSTER")
        self.assertEqual(len(result["selected_canaries"]), 2)
        self.assertFalse(result["candidate_outputs_executed"])

    def test_event_canary_selection_drift_fails_closed(self) -> None:
        changed = copy.deepcopy(self.roster)
        changed["selected_canaries"][0]["sequence_id"] = changed["selected_canaries"][1]["sequence_id"]
        with self.assertRaisesRegex(RuntimeError, "event_capacity canary selection drifted"):
            validator.validate_roster(self.repo, changed)

    def test_candidate_output_exposure_fails_closed(self) -> None:
        changed = copy.deepcopy(self.roster)
        changed["candidate_outputs_executed"] = True
        with self.assertRaisesRegex(RuntimeError, "candidate outputs exposed"):
            validator.validate_roster(self.repo, changed)

    def test_scaled_gate_drift_fails_closed(self) -> None:
        changed = copy.deepcopy(self.roster)
        changed["scaled_reject_only_gates"]["minimum_accepted_positive_events"] = 1
        with self.assertRaisesRegex(RuntimeError, "scaled reject-only gate drifted"):
            validator.validate_roster(self.repo, changed)

    def test_prior_sequence_decode_fails_closed(self) -> None:
        original = validator.load_json

        def load_with_decode(path: Path):
            payload = original(path)
            if payload.get("schema") and "remote" in str(path) and "0327" in str(path):
                payload = copy.deepcopy(payload)
                payload["sequence_content_decoded"] = True
            return payload

        with patch.object(validator, "load_json", side_effect=load_with_decode):
            with self.assertRaisesRegex(RuntimeError, "content decoded before roster freeze"):
                validator.validate_roster(self.repo, self.roster)

    def test_exact_predecode_truth_execution_freeze_passes(self) -> None:
        result = validator.validate_execution(
            self.repo,
            self.execution_path,
            self.freeze_receipt_path,
        )
        self.assertEqual(result["decision"], "VALID_PREDECODE_CANARY_TRUTH_EXECUTION_FREEZE")


if __name__ == "__main__":
    unittest.main()
