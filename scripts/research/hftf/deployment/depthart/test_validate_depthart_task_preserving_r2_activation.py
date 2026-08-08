import copy
import json
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_r2_activation import (
    validate,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/research/hftf/DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json"
)


class ValidateDepthArtTaskPreservingR2ActivationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def manifest(self) -> dict:
        return {
            "schema": "blindassist_depthart_task_preserving_deployment_r2_activation_manifest_v1",
            "protocol_id": "DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2",
            "status": "PREPARED_NOT_ACTIVATED",
            "execution_authorized": False,
            "outcome_access": "NONE",
            "implementation": {
                "candidate_count": 1,
                "candidate_representation_family": "FP16",
                "reference_checkpoint_sha256": "A" * 64,
                "reference_graph_sha256": "B" * 64,
                "candidate_graph_sha256": "C" * 64,
                "task_postprocess_sha256": "D" * 64,
                "runtime_config_sha256": "E" * 64,
                "task_postprocess_identical": True,
            },
            "cohort": {
                "cohort_id": "FRESH_R2_COHORT",
                "role": "SEALED_UNSEEN",
                "provenance": "independent RGB-D source",
                "manifest_sha256": "F" * 64,
                "parent_ids": ["parent-1", "parent-2"],
                "session_ids": ["session-1", "session-2"],
                "excluded_cohort_ids": [
                    "DEPTHART_R0_CONSUMED_120_FRAME_TUM",
                    "DEPTHART_G4D_SYNTHETIC_CANARY",
                ],
                "outcome_files_registered_not_opened": True,
            },
            "gates": copy.deepcopy(self.protocol["gates"]),
        }

    def test_valid_manifest_remains_unactivated(self) -> None:
        receipt = validate(self.protocol, self.manifest())
        self.assertEqual(
            receipt["status"], "PRE_OUTCOME_CONTRACT_VALID_EXECUTION_NOT_ACTIVATED"
        )
        self.assertFalse(receipt["execution_authorized"])
        self.assertTrue(receipt["checks"]["strict_g4d_terminal_immutable"])

    def test_rejects_outcome_access(self) -> None:
        manifest = self.manifest()
        manifest["outcome_access"] = "OPENED"
        with self.assertRaisesRegex(ValueError, "outcome access"):
            validate(self.protocol, manifest)

    def test_rejects_gate_drift(self) -> None:
        manifest = self.manifest()
        manifest["gates"]["false_clear_all_known_max"] = 0.09
        with self.assertRaisesRegex(ValueError, "gates differ"):
            validate(self.protocol, manifest)

    def test_rejects_missing_consumed_cohort_exclusion(self) -> None:
        manifest = self.manifest()
        manifest["cohort"]["excluded_cohort_ids"] = ["DEPTHART_G4D_SYNTHETIC_CANARY"]
        with self.assertRaisesRegex(ValueError, "prior cohorts"):
            validate(self.protocol, manifest)


if __name__ == "__main__":
    unittest.main()
