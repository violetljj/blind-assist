from __future__ import annotations

import copy
import unittest

from scripts.research.assistive_geometry_data_upgrade import validate_due_sanpo_prescreen_result as result_validator


class SanpoStaticPrescreenResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = result_validator.load_json(result_validator.RESULT_PATH)

    def test_governed_result_replays(self) -> None:
        validated = result_validator.validate_result(self.result)
        self.assertEqual("VALID", validated["status"])
        self.assertEqual(
            {"SANPO_REAL": "PARTIAL", "SANPO_SYNTHETIC": "PARTIAL"},
            validated["decisions"],
        )

    def test_decision_upgrade_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["source_results"]["SANPO_SYNTHETIC"]["decision"] = "PRESCREEN_ADMIT"
        with self.assertRaisesRegex(result_validator.ResultError, "source result replay mismatch"):
            result_validator.validate_result(mutated)

    def test_source_support_upgrade_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["source_results"]["SANPO_REAL"]["source_data_support_established"] = True
        with self.assertRaisesRegex(result_validator.ResultError, "source result replay mismatch"):
            result_validator.validate_result(mutated)

    def test_hard_rejection_drift_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["source_results"]["SANPO_REAL"]["hard_rejection_reasons"] = ["INVENTED"]
        with self.assertRaisesRegex(result_validator.ResultError, "source result replay mismatch"):
            result_validator.validate_result(mutated)

    def test_payload_authority_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["execution_disclosure"]["payload_download_or_open"] = True
        with self.assertRaisesRegex(result_validator.ResultError, "execution disclosure drift"):
            result_validator.validate_result(mutated)

    def test_successor_drift_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["unique_successor"] = "RUN_PAYLOAD_AUDIT"
        with self.assertRaisesRegex(result_validator.ResultError, "successor drift"):
            result_validator.validate_result(mutated)

    def test_result_field_injection_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["payload_audit_passed"] = True
        with self.assertRaisesRegex(result_validator.ResultError, "result field set drift"):
            result_validator.validate_result(mutated)


if __name__ == "__main__":
    unittest.main()
