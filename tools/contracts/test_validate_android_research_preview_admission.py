"""Executable invariants and misleading-PASS canaries for admission v1."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VALIDATOR_PATH = ROOT / "validate_android_research_preview_admission.py"
FIXTURE = ROOT / "fixtures" / "android-admission-v1" / "pass.json"
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
        self.receipt["evidence"]["pooled"]["denominators"]["false_clear"] = 199
        self.assert_rejected(self.receipt)

    def test_fail_precedes_simultaneous_unknown(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["evidence"]["complete"] = False
        receipt["observed"]["android"]["observed_backend"] = "GPU"
        receipt["android_feasibility"].update(decision="FAIL", reason_codes=["BACKEND_MISMATCH"])
        receipt["admission"].update(
            decision="FAIL",
            authorized_scope="NONE",
            reason_codes=["BACKEND_MISMATCH"],
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
        receipt["evidence"]["parent_sessions"][0]["denominators"]["false_clear"] = True
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
        receipt["evidence"]["parent_sessions"][0]["metric_sums"]["false_clear"] = 2
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

    def rebind(self, r):
        for name, field in (("thresholds", "values"), ("roster", "members")):
            b = r["contract"][name]
            content = {"version": b["version"], field: b[field]}
            if name == "roster":
                content["roles"] = b["roles"]
            b["sha256"] = validator.canonical_sha256(content)
        r["evidence"]["frozen_contract_sha256"] = validator.canonical_sha256(r["contract"])

    def test_all_terminal_fixtures(self):
        expected = {"pass": "PASS", "missing-denominator": "UNKNOWN", "zero-denominator": "UNKNOWN",
                    "nonfinite-aggregate": "UNKNOWN", "backend-mismatch": "FAIL", "fallback": "FAIL",
                    "identity-mismatch": "FAIL", "parity-failure": "FAIL", "incomplete-evidence": "UNKNOWN",
                    "combined-fail-unknown": "FAIL"}
        for name, decision in expected.items():
            with self.subTest(name=name):
                r = json.loads((FIXTURE.parent / (name + ".json")).read_text())
                self.assertEqual(r["admission"]["decision"], decision)
                validator.validate(r)

    def test_rate_domains_even_with_consistent_sums(self):
        for name in ("false_clear", "false_block", "known_coverage", "transition_consistency"):
            r = copy.deepcopy(self.receipt)
            for row in r["evidence"]["parent_sessions"]:
                row["metrics"][name] = 100
                row["metric_sums"][name] = 10000
            r["evidence"]["pooled"]["metrics"][name] = 100
            with self.subTest(metric=name):
                self.assert_rejected(r)

    def test_rate_threshold_cannot_exceed_one_even_rehashed(self):
        self.receipt["contract"]["thresholds"]["values"]["false_clear"] = 100
        self.rebind(self.receipt)
        self.assert_rejected(self.receipt)

    def test_changed_threshold_without_rehash_rejected(self):
        self.receipt["contract"]["thresholds"]["values"]["false_clear"] = .06
        self.assert_rejected(self.receipt)

    def test_full_contract_digest_detects_replaced_identity(self):
        self.receipt["contract"]["identity"]["model_sha256"] = "f" * 64
        self.receipt["observed"]["identity"]["model_sha256"] = "f" * 64
        self.assert_rejected(self.receipt)

    def test_requested_backend_input_shape_and_reset_are_compared(self):
        for kind in ("requested_backend", "input_shape", "state_reset_sha256"):
            r = copy.deepcopy(self.receipt)
            if kind == "requested_backend":
                r["observed"]["android"][kind] = "GPU"
            else:
                r["observed"]["identity"][kind] = [1, 3, 224, 224] if kind == "input_shape" else "f" * 64
            with self.subTest(kind=kind):
                self.assert_rejected(r)

    def test_authority_reason_is_fixed(self):
        self.receipt["product_authority"]["reason_codes"] = ["PRODUCT_AUTHORITY_GRANTED"]
        self.assert_rejected(self.receipt)

    def test_outcome_time_cannot_precede_freeze_claiming_pass(self):
        self.receipt["evidence"]["outcome_opened_at"] = "2026-08-11T00:00:00Z"
        self.assert_rejected(self.receipt)
        self.receipt["evidence"]["opened_after_contract_freeze"] = False
        self.assert_rejected(self.receipt)
        self.receipt["admission"].update(decision="FAIL", authorized_scope="NONE", reason_codes=["PRE_FREEZE_OUTCOME_ACCESS"])
        validator.validate(self.receipt)

    def test_parent_overlap_across_roles_is_derived(self):
        r = self.receipt
        r["contract"]["roster"]["members"].append(dict(parent_id="synthetic-parent-a", session_id="training-session", data_role="training"))
        self.rebind(r)
        self.assert_rejected(r)
        r["evidence"]["parent_session_disjoint"] = False
        r["admission"].update(decision="FAIL", authorized_scope="NONE", reason_codes=["PARENT_SESSION_OVERLAP"])
        validator.validate(r)

    def test_session_cannot_be_shared_between_parents(self):
        self.receipt["contract"]["roster"]["members"].append(dict(parent_id="other", session_id="synthetic-session-a", data_role="development"))
        self.rebind(self.receipt)
        self.assert_rejected(self.receipt)

    def test_performance_order_and_unbacked_pooled_values_rejected(self):
        self.receipt["observed"]["android"]["latency_p50_ms"] = 130
        self.assert_rejected(self.receipt)

    def test_per_session_failure_not_hidden_by_pooled_percentile(self):
        r = self.receipt
        first, second = r["evidence"]["performance_sessions"]
        first["samples"]["latency_ms"] = [200]
        second["samples"]["latency_ms"] = [80] * 100
        for row in (first, second):
            row["summary"] = validator.performance_summary(row["samples"])
        r["observed"]["android"].update(latency_p50_ms=80, latency_p95_ms=80)
        validator._validate_records(r)
        decision, reasons = validator._android_decision(r)
        self.assertEqual(decision, "FAIL")
        self.assertIn("SESSION_LATENCY_P95_MS_THRESHOLD", reasons)
        self.assert_rejected(r)

    def test_missing_session_cannot_pass(self):
        self.receipt["evidence"]["performance_sessions"].pop()
        self.assert_rejected(self.receipt)

    def test_metric_specific_denominators_are_independent(self):
        r = self.receipt
        for row in r["evidence"]["parent_sessions"]:
            row["denominators"]["false_clear"] *= 2
            row["metric_sums"]["false_clear"] *= 2
        r["evidence"]["pooled"]["denominators"]["false_clear"] *= 2
        validator.validate(r)

    def test_nonfinite_values_rejected_not_coerced_to_pass(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            r = copy.deepcopy(self.receipt)
            r["evidence"]["parent_sessions"][0]["metric_sums"]["false_clear"] = value
            with self.subTest(value=value):
                self.assert_rejected(r)

    def test_fractional_event_count_rejected(self):
        r = self.receipt
        row = r["evidence"]["parent_sessions"][0]
        row["metric_sums"]["false_clear"] = .5
        row["metrics"]["false_clear"] = .005
        r["evidence"]["pooled"]["metrics"]["false_clear"] = .0125
        self.assert_rejected(r)

    def test_schema_boolean_const_rejects_integer(self):
        self.receipt["admission"]["terminal"] = 1
        self.assert_rejected(self.receipt)

    def test_oversized_integer_rejected_cleanly(self):
        self.receipt["observed"]["android"]["peak_memory_mb"] = 10 ** 1000
        self.assert_rejected(self.receipt)


if __name__ == "__main__":
    unittest.main()
