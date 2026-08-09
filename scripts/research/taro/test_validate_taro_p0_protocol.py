#!/usr/bin/env python3
"""Mutation tests for the non-execution TARO P0 protocol validator."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.taro.validate_taro_p0_protocol import validate_runtime_absence, validate_static_contract


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = REPO_ROOT / "docs/research/taro/TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK_2026-08-10.json"
SCHEMA_BUNDLE = REPO_ROOT / "docs/research/taro/TARO_P0_SCHEMA_BUNDLE_2026-08-10.json"
FIXTURES = REPO_ROOT / "docs/research/taro/TARO_P0_ANALYTIC_FIXTURE_SPEC_2026-08-10.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TaroP0ProtocolValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load(PROTOCOL)
        cls.schema_bundle = load(SCHEMA_BUNDLE)
        cls.fixtures = load(FIXTURES)

    def validate(
        self,
        *,
        protocol: dict | None = None,
        schema_bundle: dict | None = None,
        fixtures: dict | None = None,
    ) -> list[str]:
        return validate_static_contract(
            copy.deepcopy(self.protocol if protocol is None else protocol),
            copy.deepcopy(self.schema_bundle if schema_bundle is None else schema_bundle),
            copy.deepcopy(self.fixtures if fixtures is None else fixtures),
        )

    def test_frozen_contract_is_valid(self) -> None:
        self.assertEqual(self.validate(), [])

    def test_o0m_execution_authority_drift_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["execution_authority"]["o0m_execution"] = True
        self.assertIn("EXECUTION_AUTHORITY_EXCEEDED", self.validate(protocol=protocol))

    def test_task_query_schema_is_mandatory(self) -> None:
        schema_bundle = copy.deepcopy(self.schema_bundle)
        del schema_bundle["$defs"]["TaroTaskQuery"]
        self.assertIn("TARGET_SCHEMA_DEFS", self.validate(schema_bundle=schema_bundle))

    def test_prior_or_regularizer_cannot_manufacture_rank(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["identifiability_rule"]["measurement_matrix"] = "Whitened Hessian including prior and regularizer."
        errors = self.validate(fixtures=fixtures)
        self.assertIn("MEASUREMENT_ONLY_INFORMATION", errors)
        self.assertIn("PRIOR_REGULARIZER_FIREWALL", errors)

    def test_k_corruption_cannot_become_factorial_arm(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["factorial_contract"]["arms"].append("K")
        errors = self.validate(fixtures=fixtures)
        self.assertIn("FACTORIAL_ARMS", errors)
        self.assertIn("K_MIXED_IN_FACTORIAL", errors)

    def test_common_support_oracle_mode_is_mandatory(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["factorial_contract"]["oracle_modes"] = ["FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY"]
        self.assertIn("ORACLE_MODES", self.validate(fixtures=fixtures))

    def test_missing_anchor_cannot_become_clear(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        case = next(item for item in fixtures["identifiability_cases"] if item["id"] == "missing_anchor_unknown")
        case["expected"]["query_identifiable"] = True
        case["expected"]["query_state"] = "CLEAR_OBSERVED"
        self.assertIn("DEGENERATE_NOT_UNKNOWN:missing_anchor_unknown", self.validate(fixtures=fixtures))

    def test_rank_deficient_query_invariant_positive_control_is_mandatory(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        case = next(item for item in fixtures["identifiability_cases"] if item["id"] == "full_state_underdetermined_query_identifiable_clear")
        case["expected"]["query_state"] = "UNKNOWN"
        self.assertIn("QUERY_INVARIANT_POSITIVE_CONTROL", self.validate(fixtures=fixtures))

    def test_body_motion_candidate_must_be_rejected(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        case = next(item for item in fixtures["action_filter_cases"] if item["id"] == "step_sideways_forbidden")
        case["expected_allowed"] = True
        self.assertIn("BODY_MOTION_FILTER", self.validate(fixtures=fixtures))

    def test_o0r_cannot_be_marked_evaluable(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["o0r_admission"]["current_terminal"] = "READY"
        protocol["o0r_admission"]["execution_authority"] = True
        errors = self.validate(protocol=protocol)
        self.assertIn("O0R_CURRENT_TERMINAL", errors)
        self.assertIn("O0R_AUTHORITY", errors)

    def test_factor_specific_arms_cannot_select_post_outcome(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["factorial_contract"]["factor_specific_arms_are_diagnostic"] = False
        self.assertIn("FACTOR_ARM_SELECTION_FIREWALL", self.validate(protocol=protocol))

    def test_future_source_timestamp_is_rejected(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        frame = fixtures["schema_examples"]["TaroFrameReceipt"]
        frame["max_source_timestamp_ns"] = frame["sensor_timestamp_ns"] + 1
        self.assertIn("TIMESTAMP_CEILING", self.validate(fixtures=fixtures))

    def test_identifiability_truth_is_recomputed_from_measurements(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        case = next(item for item in fixtures["identifiability_cases"] if item["id"] == "full_state_underdetermined_query_identifiable_clear")
        case["measurement_jacobian_whitened"] = [[0.0] * 4 for _ in range(4)]
        case["query_jacobian_branches_m"] = [[0.3, 0.0, 0.0, 0.0]]
        self.assertIn(f"IDENTIFIABILITY_TRUTH:{case['id']}", self.validate(fixtures=fixtures))

    def test_nonfinite_factor_truth_is_rejected(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        case = fixtures["factor_oracle_mechanics_cases"][0]
        case["truth_clearance_m"] = float("nan")
        self.assertIn(f"FACTOR_TRUTH:{case['id']}", self.validate(fixtures=fixtures))

    def test_anchor_outcome_overlap_and_duplicate_evidence_are_rejected(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        frame = fixtures["schema_examples"]["TaroFrameReceipt"]
        frame["metric_anchor"]["shared_with_outcome"] = True
        frame["sparse_tracks"].append(copy.deepcopy(frame["sparse_tracks"][0]))
        errors = self.validate(fixtures=fixtures)
        self.assertIn("ANCHOR_INDEPENDENCE_FIREWALL", errors)
        self.assertIn("DUPLICATE_EVIDENCE_ID", errors)

    def test_contradictory_posterior_is_rejected(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        posterior = fixtures["schema_examples"]["TaroFactorPosterior"]
        posterior["query_identifiable"] = False
        self.assertIn("POSTERIOR_CONTRADICTORY_IDENTIFIABILITY", self.validate(fixtures=fixtures))

    def test_empty_gate_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["o0m_gates"][0]["condition"] = ""
        self.assertIn("O0M_GATE_EMPTY:O0M_G01_BINDING_AND_INTEGRITY:condition", self.validate(protocol=protocol))

    def test_unsupported_schema_keyword_fails_closed(self) -> None:
        schema_bundle = copy.deepcopy(self.schema_bundle)
        schema_bundle["$defs"]["FiniteNumber"]["oneOf"] = [{"type": "number"}]
        errors = self.validate(schema_bundle=schema_bundle)
        self.assertTrue(any(error.startswith("UNSUPPORTED_SCHEMA_KEYWORD:") for error in errors))

    def test_per_arm_receipt_hash_is_recomputed(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        case = fixtures["factor_oracle_mechanics_cases"][0]
        key = "NONE|VALUE_ONLY_COMMON_SUPPORT"
        case["arm_receipts"][key] = "0" * len(case["arm_receipts"][key])
        self.assertIn(f"FACTOR_ARM_RECEIPTS:{case['id']}", self.validate(fixtures=fixtures))

    def test_nonempty_gate_body_drift_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["o0m_gates"][0]["condition"] = "TBD"
        self.assertIn("O0M_GATE_CONTRACT_DRIFT", self.validate(protocol=protocol))

    def test_causal_factor_label_is_derived_from_numeric_error(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        case = next(item for item in fixtures["factor_oracle_mechanics_cases"] if item["id"] == "scale_only_causal_patch")
        case["causal_factors"] = ["SUPPORT"]
        self.assertIn(f"CAUSAL_FACTOR_TRUTH:{case['id']}", self.validate(fixtures=fixtures))

    def test_posterior_covariance_must_be_symmetric_psd(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["schema_examples"]["TaroFactorPosterior"]["state_covariance"][0] = -1.0
        self.assertIn("POSTERIOR_COVARIANCE_NOT_SYMMETRIC_PSD", self.validate(fixtures=fixtures))

    def test_body_motion_candidate_cannot_be_allowed(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        candidate = fixtures["schema_examples"]["TaroObservationCandidate"]
        candidate["requires_body_motion"] = True
        candidate["allowed"] = True
        self.assertIn("CANDIDATE_BODY_MOTION_FILTER", self.validate(fixtures=fixtures))

    def test_realized_candidate_requires_receipt_and_baseline(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        candidate = fixtures["schema_examples"]["TaroObservationCandidate"]
        candidate["realized"] = True
        self.assertIn("CANDIDATE_REALIZED_RECEIPT", self.validate(fixtures=fixtures))

    def test_candidate_future_timestamp_is_rejected(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        candidate = fixtures["schema_examples"]["TaroObservationCandidate"]
        candidate["max_source_timestamp_ns"] += 1
        self.assertIn("CANDIDATE_TIMESTAMP_FIREWALL", self.validate(fixtures=fixtures))

    def test_renamed_runtime_file_is_rejected_by_module_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            module = Path(temporary) / "scripts/research/taro"
            module.mkdir(parents=True)
            (module / "renamed_solver.py").write_text("pass\n", encoding="utf-8")
            self.assertIn(
                "UNAUTHORIZED_P0_MODULE_FILE:scripts/research/taro/renamed_solver.py",
                validate_runtime_absence(Path(temporary)),
            )

    def test_task_query_identity_must_match_frame_receipt(self) -> None:
        for field in ("frame_id", "body_profile_id", "path_id"):
            with self.subTest(field=field):
                fixtures = copy.deepcopy(self.fixtures)
                fixtures["schema_examples"]["TaroTaskQuery"][field] = "wrong"
                self.assertIn("TASK_QUERY_FRAME_BINDING", self.validate(fixtures=fixtures))

    def test_posterior_identity_must_match_frame_and_query(self) -> None:
        for field in ("frame_id", "query_id"):
            with self.subTest(field=field):
                fixtures = copy.deepcopy(self.fixtures)
                fixtures["schema_examples"]["TaroFactorPosterior"][field] = "wrong"
                self.assertIn("POSTERIOR_QUERY_BINDING", self.validate(fixtures=fixtures))

    def test_posterior_future_timestamp_is_rejected(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["schema_examples"]["TaroFactorPosterior"]["max_source_timestamp_ns"] += 1
        self.assertIn("POSTERIOR_TIMESTAMP_FIREWALL", self.validate(fixtures=fixtures))

    def test_posterior_factor_and_provenance_are_bound(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        posterior = fixtures["schema_examples"]["TaroFactorPosterior"]
        posterior["factor_reference"]["content_sha256"] = "0" * 64
        posterior["input_provenance"] = ["other"]
        errors = self.validate(fixtures=fixtures)
        self.assertIn("POSTERIOR_FACTOR_BINDING", errors)
        self.assertIn("POSTERIOR_PROVENANCE", errors)

    def test_disallowed_candidate_requires_non_none_reason(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        candidate = fixtures["schema_examples"]["TaroObservationCandidate"]
        candidate["allowed"] = False
        candidate["filter_reason"] = "NONE"
        self.assertIn("CANDIDATE_DISALLOWED_REASON", self.validate(fixtures=fixtures))

    def test_disallowed_candidate_cannot_be_realized(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        candidate = fixtures["schema_examples"]["TaroObservationCandidate"]
        candidate.update(
            allowed=False,
            filter_reason="BODY_MOTION_FORBIDDEN",
            realized=True,
            actual_receipt_frame_id="future-frame",
            realized_baseline_m=0.03,
        )
        self.assertIn("CANDIDATE_REALIZED_RECEIPT", self.validate(fixtures=fixtures))

    def test_frame_causal_watermark_dominates_all_dependents(self) -> None:
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["schema_examples"]["TaroFrameReceipt"]["max_source_timestamp_ns"] -= 1
        errors = self.validate(fixtures=fixtures)
        self.assertIn("ANCHOR_TIMESTAMP_FIREWALL", errors)
        self.assertIn("POSTERIOR_TIMESTAMP_FIREWALL", errors)
        self.assertIn("CANDIDATE_TIMESTAMP_FIREWALL", errors)


if __name__ == "__main__":
    unittest.main()
