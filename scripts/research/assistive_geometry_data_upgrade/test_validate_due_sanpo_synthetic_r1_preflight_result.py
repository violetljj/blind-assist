from __future__ import annotations

import copy
import unittest

from scripts.research.assistive_geometry_data_upgrade import (
    validate_due_sanpo_synthetic_r1_preflight_result as validator,
)


class SanpoSyntheticR1PreflightResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.load_json(validator.RESULT_PATH)

    def test_governed_result_validates(self) -> None:
        self.assertEqual("VALID", validator.validate_result(self.result)["status"])

    def test_missing_object_identity_is_fixed(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["observed_failure"]["object_name"] = "guessed/replacement.json"
        with self.assertRaisesRegex(validator.ResultError, "observed failure drift"):
            validator.validate_result(mutated)

    def test_not_evaluable_cannot_be_upgraded(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["decision"] = "PASS"
        with self.assertRaisesRegex(validator.ResultError, "decision drift"):
            validator.validate_result(mutated)

    def test_frame_body_access_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["execution_disclosure"]["frame_body_requested_or_read"] = True
        with self.assertRaisesRegex(validator.ResultError, "execution disclosure drift"):
            validator.validate_result(mutated)

    def test_inventory_cannot_fill_capability_count(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["capability_counts"]["oracle_depth_factor_frames"] = 1
        with self.assertRaisesRegex(validator.ResultError, "capability count drift"):
            validator.validate_result(mutated)

    def test_source_support_authority_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["authority"]["source_data_support_established"] = True
        with self.assertRaisesRegex(validator.ResultError, "authority drift"):
            validator.validate_result(mutated)

    def test_artifact_receipt_path_or_hash_drift_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["artifact_receipts"][0]["path"] = "outside/attempt.json"
        with self.assertRaisesRegex(validator.ResultError, "artifact receipt drift"):
            validator.validate_result(mutated)

    def test_body_canary_successor_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["unique_successor"] = "RUN_BODY_CANARY"
        with self.assertRaisesRegex(validator.ResultError, "successor drift"):
            validator.validate_result(mutated)


if __name__ == "__main__":
    unittest.main()
