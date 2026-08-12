#!/usr/bin/env python3
"""Tests for the R2 F1 supervision contract validator."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from validate_ag_r2_f1_supervision_contract import (  # noqa: E402
    DEFAULT_CONTRACT,
    identity_digest,
    validate_document,
    validate_path,
)


class SupervisionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))

    def test_frozen_contract_passes_all_static_gates(self) -> None:
        result = validate_path(DEFAULT_CONTRACT)
        self.assertTrue(result["passed"])
        self.assertEqual(result["parent_count"], 13)
        self.assertEqual(result["role_parent_counts"], {"FIT": 9, "CHECKPOINT_SELECTION": 2, "TRAIN_CANARY": 2})
        self.assertEqual(result["gate_count"], 15)

    def test_parent_assignment_hashes_are_identity_only(self) -> None:
        for row in self.document["cohort_contract"]["parents"]:
            self.assertEqual(row["assignment_sha256"], identity_digest(row["parent_id"]))

    def test_duplicate_parent_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["cohort_contract"]["parents"][1]["parent_id"] = mutated["cohort_contract"]["parents"][0]["parent_id"]
        with self.assertRaisesRegex(ValueError, "duplicate parent"):
            validate_document(mutated, verify_bindings=False)

    def test_held_orientation_drift_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["cohort_contract"]["parents"][10]["orientation"] = "LANDSCAPE_IDENTITY"
        with self.assertRaisesRegex(ValueError, "orientation drift"):
            validate_document(mutated, verify_bindings=False)

    def test_teacher_provenance_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["provenance_contract"]["teacher_provenance_admitted"] = True
        with self.assertRaisesRegex(ValueError, "provenance contract drift"):
            validate_document(mutated, verify_bindings=False)

    def test_proxy_sigma_target_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["uncertainty_contract"]["direct_uncertainty_proxy_regression_forbidden"] = False
        with self.assertRaisesRegex(ValueError, "uncertainty contract drift"):
            validate_document(mutated, verify_bindings=False)

    def test_binding_sha_drift_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["bindings"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binding SHA drift"):
            validate_document(mutated)


if __name__ == "__main__":
    unittest.main()
