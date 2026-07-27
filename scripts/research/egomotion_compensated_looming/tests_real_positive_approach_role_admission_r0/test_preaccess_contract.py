from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r0 import (
    bootstrap_claim,
    producer,
)


ROOT = Path(__file__).resolve().parents[4]
CONTRACT = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0_CONTRACT_2026-07-27.json"
)


class PreAccessContractTest(unittest.TestCase):
    def test_single_candidate_and_legal_terminals(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["source_selection"]["candidate_count"], 1)
        self.assertTrue(contract["source_selection"]["no_replacement"])
        self.assertEqual(
            contract["legal_terminals"],
            [
                "REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID",
                "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID",
            ],
        )

    def test_source_descriptor_hash_is_bound(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        descriptor = contract["source_selection"]["source_descriptor_canonical_json"]
        observed = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
        self.assertEqual(
            observed,
            contract["source_selection"]["source_descriptor_sha256"],
        )
        self.assertEqual(observed, bootstrap_claim.SOURCE_DESCRIPTOR_SHA256)

    def test_rgb_and_algorithm_access_are_forbidden(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertFalse(contract["access_contract"]["algorithm_outcome_access"])
        forbidden = " ".join(contract["access_contract"]["forbidden_payload_reads"])
        self.assertIn("RGB image pixels", forbidden)
        self.assertIn("RCLE algorithm output", forbidden)

    def test_claim_is_exclusive_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / "claim.json"
            first = bootstrap_claim.create_claim(CONTRACT, claim)
            self.assertEqual(
                first["contract_sha256"],
                bootstrap_claim.sha256_file(CONTRACT),
            )
            with self.assertRaises(FileExistsError):
                bootstrap_claim.create_claim(CONTRACT, claim)

    def test_quaternion_identity_rotation(self) -> None:
        observed = producer.quaternion_rotation_wxyz(
            {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        )
        self.assertTrue((observed == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]).all())

    def test_source_code_does_not_import_rgb_algorithm(self) -> None:
        module_root = Path(producer.__file__).parent
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in module_root.glob("*.py")
        )
        self.assertNotIn("rgb_algorithm_canary_r0", source)
        self.assertNotIn("dataset_classical", source)


if __name__ == "__main__":
    unittest.main()
