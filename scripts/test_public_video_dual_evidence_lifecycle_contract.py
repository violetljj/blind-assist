import copy
import unittest
from pathlib import Path

import public_video_dual_evidence_lifecycle_contract as contract
import run_public_silver_frozen_feature_probe as common


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "public_video_dual_evidence_lifecycle_contract_r717.json"
)


class DualEvidenceLifecycleContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = common.load_json(CONTRACT_PATH)

    def test_frozen_contract_validates(self) -> None:
        self.assertEqual(contract.validate_contract(copy.deepcopy(self.value))["contract_id"], "r717_dual_evidence_lifecycle_prospective_v1")

    def test_margin_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.value)
        value["lifecycle_contract"]["strong_normalized_change_margin"] = 0.04
        with self.assertRaisesRegex(ValueError, "margin"):
            contract.validate_contract(value)

    def test_cold_start_cannot_be_clear(self) -> None:
        value = copy.deepcopy(self.value)
        value["lifecycle_contract"]["cold_start_state"] = "clear"
        with self.assertRaisesRegex(ValueError, "cold start"):
            contract.validate_contract(value)

    def test_semantic_groups_cannot_drift(self) -> None:
        value = copy.deepcopy(self.value)
        value["feature_contract"]["semantic_exit_channel"]["selected_groups"]["surface_material"].append("road")
        with self.assertRaisesRegex(ValueError, "groups"):
            contract.validate_contract(value)

    def test_feature_report_must_precede_review(self) -> None:
        value = copy.deepcopy(self.value)
        value["review_protocol"]["full_feature_report_frozen_before_visual_review"] = False
        with self.assertRaisesRegex(ValueError, "chronology"):
            contract.validate_contract(value)

    def test_source_lineage_cannot_be_disabled(self) -> None:
        value = copy.deepcopy(self.value)
        value["source_eligibility"]["source_inventory_eligibility_required"] = False
        with self.assertRaisesRegex(ValueError, "eligibility"):
            contract.validate_contract(value)

    def test_authorization_cannot_be_enabled(self) -> None:
        value = copy.deepcopy(self.value)
        value["authorization"]["android_runtime_change_authorized"] = True
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            contract.validate_contract(value)


if __name__ == "__main__":
    unittest.main()
