from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    enumerate_trials,
    sha256_file,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0.receipt import (
    source_manifest,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_validation_r0.receipt import (
    CANDIDATE_LOCK_SHA256,
    DEVELOPMENT_RECEIPT_SHA256,
    RECEIPT_SCHEMA_VERSION,
    validation_control_manifest,
)
from scripts.research.egomotion_compensated_looming.run_observable_support_recovery_sealed_validation_r0 import (
    OUTPUT_ROOT,
    SEALED_VALIDATION_SEEDS,
    sealed_validation_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parents[1]


class SealedValidationBoundaryR0Test(unittest.TestCase):
    def test_sealed_inventory_is_exact_and_complete(self) -> None:
        protocol = sealed_validation_protocol()
        trials = enumerate_trials(protocol)
        self.assertEqual(SEALED_VALIDATION_SEEDS, tuple(range(3000, 3020)))
        self.assertEqual(len(trials), 2520)
        self.assertEqual(len({trial.trial_id for trial in trials}), 2520)
        self.assertEqual(sum(trial.split == "clean" for trial in trials), 1680)
        self.assertEqual(sum(trial.split == "stress" for trial in trials), 840)

    def test_candidate_and_development_receipts_are_frozen(self) -> None:
        candidate_lock = (
            MODULE_ROOT
            / "rcle_observable_support_r0"
            / "RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json"
        )
        development_receipt = (
            REPO_ROOT
            / "artifacts.local"
            / "evidence"
            / "rcle_observable_support_recovery_r0"
            / "development_gate_r0"
            / "receipt.json"
        )
        self.assertEqual(sha256_file(candidate_lock), CANDIDATE_LOCK_SHA256)
        self.assertEqual(
            sha256_file(development_receipt), DEVELOPMENT_RECEIPT_SHA256
        )

    def test_candidate_source_manifest_still_matches_implementation_lock(
        self,
    ) -> None:
        lock_path = (
            MODULE_ROOT
            / "rcle_observable_support_r0"
            / "RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json"
        )
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock["source_manifest"], source_manifest(REPO_ROOT))

    def test_output_and_receipt_schema_are_fixed(self) -> None:
        expected = (
            REPO_ROOT
            / "artifacts.local"
            / "evidence"
            / "rcle_observable_support_recovery_r0"
            / "sealed_validation_gate_r0"
        )
        self.assertEqual(OUTPUT_ROOT, expected)
        schema_path = (
            MODULE_ROOT
            / "schemas"
            / "rcle_observable_support_recovery_sealed_validation_receipt_r0.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            RECEIPT_SCHEMA_VERSION,
        )

    def test_runner_exposes_no_seed_role_or_output_override(self) -> None:
        runner = (
            MODULE_ROOT
            / "run_observable_support_recovery_sealed_validation_r0.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--seed", runner)
        self.assertNotIn("--output-root", runner)
        self.assertNotIn("--dataset-root", runner)
        self.assertNotIn("--role", runner)

    def test_validation_control_manifest_is_bounded(self) -> None:
        manifest = validation_control_manifest(REPO_ROOT)
        self.assertEqual(len(manifest), 6)
        self.assertTrue(
            all(
                "rcle_observable_support_r0/" not in path
                for path in manifest
            )
        )


if __name__ == "__main__":
    unittest.main()
