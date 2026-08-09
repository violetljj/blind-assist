import copy
import json
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d1_contract import (
    validate,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
PROTOCOL = json.loads((
    REPO_ROOT
    / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_PROTOCOL_2026-08-09.json"
).read_text(encoding="utf-8"))
ROSTER = json.loads((
    REPO_ROOT
    / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D1_ARKIT_DEVELOPMENT_ROSTER_LOCK_2026-08-09.json"
).read_text(encoding="utf-8"))
B0 = json.loads((
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.json"
).read_text(encoding="utf-8"))
R2 = json.loads((
    REPO_ROOT
    / "docs/research/hftf/DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json"
).read_text(encoding="utf-8"))


class ValidateDepthArtTaskPreservingD1ContractTest(unittest.TestCase):
    def test_frozen_contract_and_metadata_roster_are_valid_but_unactivated(self) -> None:
        receipt = validate(PROTOCOL, ROSTER, B0, R2)
        self.assertEqual(
            receipt["status"],
            "CONTRACT_AND_METADATA_ROSTER_VALID_EXECUTION_NOT_AUTHORIZED",
        )
        self.assertFalse(receipt["execution_authorized"])
        self.assertEqual(receipt["outcome_access"], "NONE")

    def test_rejects_gate_drift_from_r2(self) -> None:
        protocol = copy.deepcopy(PROTOCOL)
        protocol["quality_contract"]["gates"]["false_clear_all_known_max"] = 0.09
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate(protocol, ROSTER, B0, R2)

    def test_rejects_non_training_identity(self) -> None:
        roster = copy.deepcopy(ROSTER)
        roster["primary"][0]["fold"] = "Validation"
        with self.assertRaisesRegex(ValueError, "Training identities"):
            validate(PROTOCOL, roster, B0, R2)

    def test_rejects_duplicate_visit(self) -> None:
        roster = copy.deepcopy(ROSTER)
        roster["reserve"][0]["visit_id"] = roster["primary"][0]["visit_id"]
        with self.assertRaisesRegex(ValueError, "visit identities overlap"):
            validate(PROTOCOL, roster, B0, R2)

    def test_rejects_any_outcome_access(self) -> None:
        roster = copy.deepcopy(ROSTER)
        roster["invariants"]["outcome_access"] = "OPENED"
        with self.assertRaisesRegex(ValueError, "outcome access"):
            validate(PROTOCOL, roster, B0, R2)


if __name__ == "__main__":
    unittest.main()
