import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_video_event_timing_contract as subject


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "public_video_event_timing_contract_r750.json"
TEMPLATE = ROOT / "configs" / "public_video_event_timing_review_template_r750.json"


class PublicVideoEventTimingContractTest(unittest.TestCase):
    def test_frozen_contract_loads(self) -> None:
        value, meta = subject.load_contract(CONTRACT)
        self.assertEqual(3000, value["review_timing_fields"]["maximum_early_warning_lead_ms"])
        self.assertEqual(64, len(meta["sha256"]))

    def test_japan_failure_cannot_be_reclassified(self) -> None:
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        changed = copy.deepcopy(value)
        changed["prospective_isolation"]["japan_source_id_forbidden_for_acceptance"] = ""
        with self.assertRaisesRegex(ValueError, "Japan"):
            subject.validate_contract(changed)

    def test_true_radial_safe_lateral_gate_cannot_be_removed(self) -> None:
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        changed = copy.deepcopy(value)
        changed["acceptance"]["negative_must_be_vetoed_by_frozen_route_relation"] = False
        with self.assertRaisesRegex(ValueError, "acceptance gate"):
            subject.validate_contract(changed)

    def test_review_template_exposes_both_roles_and_no_authorization(self) -> None:
        value = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        subject.validate_review_template(value)
        self.assertTrue(all(flag is False for flag in value["authorization"].values()))


if __name__ == "__main__":
    unittest.main()
