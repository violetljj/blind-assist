"""Executable invariants and misleading-PASS canaries for admission v1."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_android_research_preview_admission.py"
FIXTURE = ROOT / "schemas" / "fixtures" / "android-admission-v1" / "pass.json"
SPEC = importlib.util.spec_from_file_location("admission_validator", VALIDATOR_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class AdmissionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.receipt = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def assert_rejected(self, receipt: dict) -> None:
        with self.assertRaises(validator.ValidationError):
            validator.validate(receipt)

    def test_complete_pass_fixture(self) -> None:
        validator.validate(self.receipt)

    def test_rejects_missing_measurement_claiming_pass(self) -> None:
        self.receipt["observed"]["android"]["latency_p95_ms"] = None
        self.assert_rejected(self.receipt)

    def test_rejects_backend_mismatch_claiming_pass(self) -> None:
        self.receipt["observed"]["android"]["observed_backend"] = "GPU"
        self.assert_rejected(self.receipt)

    def test_rejects_fallback_claiming_pass(self) -> None:
        self.receipt["observed"]["android"]["fallback_observed"] = True
        self.receipt["observed"]["android"]["fallback_backend"] = "CPU"
        self.assert_rejected(self.receipt)

    def test_rejects_parity_failure_claiming_pass(self) -> None:
        self.receipt["observed"]["android"]["reference_parity"]["measured_max_abs_error"] = 0.02
        self.assert_rejected(self.receipt)

    def test_rejects_threshold_failure_claiming_pass(self) -> None:
        self.receipt["evidence"]["pooled"]["metrics"]["false_clear"] = 0.06
        self.assert_rejected(self.receipt)

    def test_rejects_pooled_denominator_not_equal_to_records(self) -> None:
        self.receipt["evidence"]["pooled"]["denominator"] = 19
        self.assert_rejected(self.receipt)

    def test_fail_precedes_simultaneous_unknown(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["evidence"]["complete"] = False
        receipt["evidence"]["parent_session_disjoint"] = False
        receipt["admission"].update(
            decision="FAIL",
            authorized_scope="NONE",
            reason_codes=["PARENT_SESSION_OVERLAP"],
        )
        validator.validate(receipt)

    def test_rejects_nondeterministic_reason_order(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["observed"]["identity"]["model_sha256"] = "7" * 64
        receipt["evidence"]["parent_session_disjoint"] = False
        receipt["admission"].update(
            decision="FAIL",
            authorized_scope="NONE",
            reason_codes=["PARENT_SESSION_OVERLAP", "MODEL_SHA256_MISMATCH"],
        )
        self.assert_rejected(receipt)


if __name__ == "__main__":
    unittest.main()

