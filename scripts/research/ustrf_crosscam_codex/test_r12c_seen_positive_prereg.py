from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import validate_r12c_seen_positive_prereg as subject


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "configs/ustrf_crosscam_seen_positive_r12c_prereg_v1.json"


class R12cSeenPositivePreregTest(unittest.TestCase):
    def test_bangkok_replacement_closes_sixth_slot_without_r13(self) -> None:
        result = subject.validate(CONTRACT)
        self.assertTrue(result["candidate_qualified_as_non_r13_seen_positive"])
        self.assertEqual(2, result["alertable_robust_inside_anchor_count"])
        self.assertEqual(6, result["eligible_positive_count_after_validation"])
        self.assertFalse(result["r13_slot_consumed"])
        self.assertFalse(result["authorization"]["london_768_candidate_execution_authorized"])

    def test_boundary_sensitive_328_is_not_used_as_an_alertable_anchor(self) -> None:
        result = subject.validate(CONTRACT)
        boundary = next(row for row in result["anchor_results"] if row["timestamp_ms"] == 328000)
        self.assertEqual("pre_alert_boundary_sensitive_not_gate_eligible", boundary["role"])
        self.assertEqual("uncertain_boundary", boundary["robust_relation"])
        alertable = [row for row in result["anchor_results"] if row["role"] == "alertable_positive"]
        self.assertEqual(["inside", "inside"], [row["robust_relation"] for row in alertable])

    def test_r13_access_or_polygon_rescue_fails_closed(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        original_load_json = subject.load_json

        def loader_with_contract(replacement: dict):
            return lambda path: replacement if Path(path).resolve() == CONTRACT.resolve() else original_load_json(path)

        modified = copy.deepcopy(contract)
        modified["non_r13_guard"]["r13_slot_consumed"] = True
        with patch.object(subject, "load_json", side_effect=loader_with_contract(modified)):
            with self.assertRaisesRegex(ValueError, "non-R1.3 guard"):
                subject.validate(CONTRACT)
        modified = copy.deepcopy(contract)
        modified["geometry_contract"]["route_polygon_may_be_moved_after_detector_result"] = True
        with patch.object(subject, "load_json", side_effect=loader_with_contract(modified)):
            with self.assertRaisesRegex(ValueError, "polygon rescue"):
                subject.validate(CONTRACT)


if __name__ == "__main__":
    unittest.main()
