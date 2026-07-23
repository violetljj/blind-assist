from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import validate_navwareset_source_prescreen as validator


class NavWareSetSourcePrescreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config_path = (
            cls.repo / "configs/ustrf_route_target_evidence_closure_r1_navwareset_source_prescreen.json"
        )
        cls.receipt_path = (
            cls.repo
            / "artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/"
            "navwareset-source-prescreen-freeze-receipt-r1.json"
        )
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def test_exact_predecode_two_stage_prescreen_passes(self) -> None:
        result = validator.validate(self.repo, self.config_path, self.receipt_path)
        self.assertEqual(
            result["decision"],
            "VALID_PREDECODE_TWO_STAGE_REJECT_ONLY_NAVWARESET_PRESCREEN",
        )
        self.assertFalse(result["candidate_outputs_executed"])

    def test_stage_b_gate_drift_fails_closed(self) -> None:
        original = validator.load_json

        def load_with_drift(path: Path):
            payload = original(path)
            if path.resolve() == self.config_path.resolve():
                payload = copy.deepcopy(payload)
                payload["stage_b_full_lifecycle_canary"]["scaled_reject_only_gates"][
                    "minimum_accepted_critical_events"
                ] = 0
            return payload

        with patch.object(validator, "load_json", side_effect=load_with_drift):
            with self.assertRaisesRegex(RuntimeError, "scaled reject-only gate drifted"):
                validator.validate(self.repo, self.config_path, self.receipt_path)

    def test_predecode_payload_exposure_fails_closed(self) -> None:
        original = validator.load_json

        def load_with_exposure(path: Path):
            payload = original(path)
            if path.resolve() == self.receipt_path.resolve():
                payload = copy.deepcopy(payload)
                payload["payload_bytes_downloaded_before_freeze"] = 1
            return payload

        with patch.object(validator, "load_json", side_effect=load_with_exposure):
            with self.assertRaisesRegex(RuntimeError, "payload downloaded before freeze"):
                validator.validate(self.repo, self.config_path, self.receipt_path)


if __name__ == "__main__":
    unittest.main()
