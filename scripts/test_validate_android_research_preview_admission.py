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

    def test_schema_rejects_missing_and_extra_surface(self) -> None:
        for path in (("run_id",), ("contract", "frozen_at"), ("contract", "candidate_id")):
            receipt = copy.deepcopy(self.receipt)
            target = receipt
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
            with self.subTest(path=path):
                self.assert_rejected(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["unexpected"] = True
        self.assert_rejected(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["observed"]["android"]["unexpected"] = True
        self.assert_rejected(receipt)

    def test_schema_rejects_invalid_freeze_time_and_boolean_denominator(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["contract"]["frozen_at"] = "not-a-date-time"
        self.assert_rejected(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["contract"]["frozen_at"] = "2026-08-12T06:52:00"
        self.assert_rejected(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["evidence"]["parent_sessions"][0]["denominator"] = True
        self.assert_rejected(receipt)

    def test_reference_runtime_is_frozen_and_compared(self) -> None:
        mismatch = copy.deepcopy(self.receipt)
        mismatch["observed"]["android"]["reference_parity"]["runtime_sha256"] = "7" * 64
        decision, reasons = validator._android_decision(mismatch)
        self.assertEqual(validator.FAIL, decision)
        self.assertIn("REFERENCE_RUNTIME_SHA256_MISMATCH", reasons)
        self.assert_rejected(mismatch)

        missing = copy.deepcopy(self.receipt)
        missing["observed"]["android"]["reference_parity"]["runtime_sha256"] = None
        decision, reasons = validator._android_decision(missing)
        self.assertEqual(validator.UNKNOWN, decision)
        self.assertIn("MISSING_REFERENCE_PARITY_EVIDENCE", reasons)
        self.assert_rejected(missing)

    def test_each_android_threshold_is_derived(self) -> None:
        violations = {
            "cold_start_ms": 501.0,
            "warm_start_ms": 151.0,
            "latency_p50_ms": 101.0,
            "latency_p95_ms": 1_000_000_000.0,
            "peak_memory_mb": 301.0,
            "thermal_window_seconds": 299.0,
        }
        for metric, value in violations.items():
            receipt = copy.deepcopy(self.receipt)
            receipt["observed"]["android"][metric] = value
            decision, reasons = validator._android_decision(receipt)
            with self.subTest(metric=metric):
                self.assertEqual(validator.FAIL, decision)
                self.assertIn(f"{metric.upper()}_THRESHOLD", reasons)
                self.assert_rejected(receipt)

    def test_quality_decision_is_derived_from_frozen_thresholds(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["contract"]["thresholds"]["values"]["false_clear"] = 0.001
        decision, reasons = validator._quality_decision(receipt)
        self.assertEqual(validator.FAIL, decision)
        self.assertEqual(["FALSE_CLEAR_THRESHOLD"], reasons)
        self.assert_rejected(receipt)

    def test_fallback_state_must_be_internally_consistent(self) -> None:
        false_with_backend = copy.deepcopy(self.receipt)
        false_with_backend["observed"]["android"]["fallback_backend"] = "GPU"
        decision, reasons = validator._android_decision(false_with_backend)
        self.assertEqual(validator.FAIL, decision)
        self.assertIn("FALLBACK_STATE_INCONSISTENT", reasons)
        self.assert_rejected(false_with_backend)

        true_without_backend = copy.deepcopy(self.receipt)
        true_without_backend["observed"]["android"]["fallback_observed"] = True
        decision, reasons = validator._android_decision(true_without_backend)
        self.assertEqual(validator.FAIL, decision)
        self.assertIn("BACKEND_FALLBACK", reasons)
        self.assertIn("FALLBACK_STATE_INCONSISTENT", reasons)
        self.assert_rejected(true_without_backend)

    def test_parent_session_metric_must_equal_sum_over_denominator(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["evidence"]["parent_sessions"][0]["metrics"]["false_clear"] = 0.99
        self.assert_rejected(receipt)

    def test_pooled_metrics_are_recomputed_from_parent_session_sums(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["evidence"]["pooled"]["metrics"]["false_clear"] = 0.99
        self.assert_rejected(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["evidence"]["parent_sessions"][0]["metric_sums"]["false_clear"] = 0.2
        self.assert_rejected(receipt)
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
