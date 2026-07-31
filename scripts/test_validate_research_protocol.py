#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import validate_research_protocol as target


POLICY = json.loads(target.DEFAULT_POLICY.read_text(encoding="utf-8"))
V3_POLICY = json.loads(target.V3_POLICY.read_text(encoding="utf-8"))
V3_FILE_SHA256 = "6b82e2b131da41763311a62730300dc7936006090c9bcf712142de98faba9613"


def base_protocol(stage: str = "DISCOVERY") -> dict:
    claim = POLICY["stages"][stage]["allowed_claims"][0]
    level = POLICY["stages"][stage]["minimum_freeze_level"]
    constraints = [
        {
            "id": "source-integrity",
            "class": "INVARIANT",
            "description": "Source identity is retained.",
            "failure_scope": "ITEM",
        }
    ]
    if stage in {"CONFIRMATION", "DEPLOYMENT"}:
        constraints.append(
            {
                "id": "effect-gate",
                "class": "GATE",
                "description": "Frozen effect threshold.",
                "failure_scope": "IMPLEMENTATION_VERSION",
                "metric": "median_effect",
                "operator": "GTE",
                "threshold": 0.05,
                "unit": "s^-1",
                "rationale": "Separates an explicit positive condition.",
                "calibration_source": "Independent synthetic calibration.",
                "sensitivity_plan": "Report 0.04 and 0.06 sensitivity.",
                "revision_policy": "New version only after outcome access.",
            }
        )
    partition = {
        "id": "data-a",
        "source_identity": "source-a",
        "content_identity": "content-a",
        "identity_basis": "Fixture manifest and content digest.",
        "independence_group": "group-a",
        "ancestry": [],
        "role": stage,
        "outcome_access": "NONE",
        "result_access_state": (
            "SEALED_UNSEEN"
            if stage in {"CONFIRMATION", "DEPLOYMENT"}
            else "CONTENT_INSPECTED"
        ),
        "observation_unit": "CAPTURE_SESSION",
        "split_basis": "SESSION_LEVEL_PREASSIGNMENT",
        "research_track": (
            "SEALED_EVALUATION"
            if stage == "CONFIRMATION"
            else (
                "EXTERNAL_TRANSFER"
                if stage == "DEPLOYMENT"
                else "CAPABILITY_DISCOVERY"
            )
        ),
        "reuse_policy": "ROLE_BOUND",
    }
    if stage in {"CONFIRMATION", "DEPLOYMENT"}:
        partition["identity_manifest_ref"] = (
            "scripts/fixtures/research_governance/"
            "confirmation_identity_manifest.json"
        )
        partition["identity_sha256"] = (
            "77efbf2bb7ab2481b3465ccbaeecafeff"
            "47ba63154a33b2d7b73a9aba04bf68d"
        )
        partition["disjointness_evidence"] = (
            "Frozen manifest proves separation from development ancestry."
        )
    return {
        "schema_version": target.PROTOCOL_SCHEMA,
        "profile": POLICY["profile_selection_rules"]["default_by_stage"][stage],
        "governance_policy_id": POLICY["policy_id"],
        "governance_policy_sha256": target._policy_digest(POLICY),
        "protocol_id": f"TEST_{stage}",
        "version": "R0",
        "stage": stage,
        "question": "What does this test establish?",
        "claims_allowed": [claim],
        "data_partitions": [partition],
        "constraints": constraints,
        "freeze": {
            "level": level,
            "outcome_access_started": False,
            "amendment_mode": "IN_PLACE_BEFORE_OUTCOME",
        },
        "result_model": {
            "execution_validity": "NOT_RUN",
            "scientific_outcome": "NOT_RUN",
            "invalid_execution_effect": "CLOSE_EVIDENCE_VERSION_ONLY",
            "terminal_scope": "ITEM",
        },
        "successor_policy": {
            "new_version_allowed": True,
            "preserve_previous_evidence": True,
        },
        "experiment_design": {
            "search_strategy": "SINGLE_VARIABLE_COUNTERFACTUAL",
            "minimal_discriminating_experiment": "One deterministic fixture.",
            "resource_budget": "One fixture and one replay.",
            "stop_conditions": "Stop this candidate if the fixture falsifies it.",
        },
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "theoretical_or_empirical_basis": "A direct source-native measure is closer to the causal variable.",
                "causal_difference": "Tests one direct mechanism.",
                "expected_information_gain": "Separates two explanations.",
                "minimal_test": "One deterministic counterfactual fixture.",
                "evaluation_metric": "Direct-measure separation between the two conditions.",
                "falsifier": "The direct measurement disagrees.",
                "cost": "LOW",
                "resource_budget": "One fixture replay.",
                "stop_condition": "Stop H1 if the fixture falsifies the direct measure.",
                "selection_reason": "Highest information per unit cost.",
            }
        ],
    }


class ProgressiveResearchGovernanceTest(unittest.TestCase):
    def test_current_research_entries_share_forward_r4_scope(self) -> None:
        expected_markers = {
            "docs/SANPO_CURRENT_STATUS.md": (
                "FORWARD_GOVERNANCE: THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
                "DEFAULT_NEW_WORK_LANE: THESIS_DEVELOPMENT",
                "DEVELOPMENT_REQUIRES_PRODUCTION_PROMOTION_GATES: false",
                "PRODUCTION_PROMOTION_REQUIRES_EXPLICIT_SCOPE: true",
                "HISTORICAL_TERMINALS_IMMUTABLE: true",
            ),
            "docs/SANPO_TRAINING_PROTOCOL.md": (
                "FORWARD_GOVERNANCE: THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
                "DEFAULT_NEW_WORK_LANE: THESIS_DEVELOPMENT",
                "DEVELOPMENT_REQUIRES_FRESH_HOLDOUT: false",
                "DEVELOPMENT_REQUIRES_INT8_OR_DEVICE_EVENT_GATE: false",
                "PRODUCTION_PROMOTION_REQUIRES_EXPLICIT_SCOPE: true",
            ),
            "docs/SANPO_CANDIDATE_PROMOTION_GATES.md": (
                "FORWARD_GOVERNANCE: THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
                "SCOPE: PRODUCTION_PROMOTION_ONLY",
                "DEVELOPMENT_REQUIRES_THIS_GATE_CHAIN: false",
                "ALGORITHM_SELECTION_BENCHMARK_IS_DEVICE_EVENT_GATE: false",
                "PLATFORM_ENGINEERING_BENCHMARK_IS_DEVICE_EVENT_GATE: false",
            ),
            "docs/research/dual-loop/README.md": (
                "FORWARD_GOVERNANCE: THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
                "DEFAULT_NEW_WORK_LANE: THESIS_DEVELOPMENT",
                "DEVELOPMENT_REQUIRES_LEGACY_FORMAL_GATES: false",
                "HISTORICAL_TERMINALS_IMMUTABLE: true",
            ),
            "docs/research/rcle/README.md": (
                "FORWARD_GOVERNANCE: THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
                "DEFAULT_NEW_WORK_LANE_IF_RESUMED: THESIS_DEVELOPMENT",
                "NEW_DEVELOPMENT_REQUIRES_LEGACY_ONE_SHOT_AUTHORITY: false",
                "HISTORICAL_TERMINALS_IMMUTABLE: true",
            ),
            "docs/research/ustrf-sc/README.md": (
                "FORWARD_GOVERNANCE: THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
                "DEFAULT_NEW_WORK_LANE_IF_REOPENED: THESIS_DEVELOPMENT",
                "NEW_DEVELOPMENT_REQUIRES_LEGACY_FORMAL_GATES: false",
                "HISTORICAL_TERMINALS_IMMUTABLE: true",
            ),
        }
        for relative_path, markers in expected_markers.items():
            with self.subTest(path=relative_path):
                text = (target.REPO_ROOT / relative_path).read_text(encoding="utf-8")
                for marker in markers:
                    self.assertIn(marker, text)
        navigation_markers = {
            "README.md": (
                "THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
                "THESIS_DEVELOPMENT",
                "PRODUCTION_PROMOTION",
                "docs/research/dual-loop/README.md",
            ),
            "docs/README.md": (
                "THESIS_DEVELOPMENT / PRODUCTION_PROMOTION",
                "PRODUCTION_PROMOTION",
                "默认采用 R4",
            ),
            "scripts/README.md": (
                "THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
                "双环 current 入口",
                "THESIS_DEVELOPMENT",
                "PRODUCTION_PROMOTION",
            ),
        }
        for relative_path, markers in navigation_markers.items():
            with self.subTest(path=relative_path):
                text = (target.REPO_ROOT / relative_path).read_text(encoding="utf-8")
                for marker in markers:
                    self.assertIn(marker, text)
        sanpo_scoped_workflows = (
            "docs/SANPO_SEQUENCE_EVALSET.md",
            "docs/SANPO_SEGMENTATION_CANDIDATE.md",
            "docs/SANPO_TRAVERSABILITY_BASELINE.md",
            "docs/SANPO_V3_REGRESSION_DATASET.md",
            "docs/SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md",
            "docs/PUBLIC_VIDEO_GPT_SILVER_LABEL_PROTOCOL.md",
        )
        for relative_path in sanpo_scoped_workflows:
            with self.subTest(path=relative_path):
                text = (target.REPO_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("执行范围：", text)
                self.assertIn("R4", text)
        sanpo_current = (
            target.REPO_ROOT / "docs/SANPO_CURRENT_STATUS.md"
        ).read_text(encoding="utf-8")
        ustrf_current = (
            target.REPO_ROOT / "docs/research/ustrf-sc/README.md"
        ).read_text(encoding="utf-8")
        self.assertNotIn("RCLE 当前研究主线", sanpo_current)
        self.assertNotIn("当前 BlindAssist 研究主线已切换到 [RCLE]", ustrf_current)

    def test_current_policy_is_thesis_first_r4(self) -> None:
        self.assertEqual(
            "THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
            POLICY["policy_id"],
        )
        self.assertEqual([], target.validate_policy(POLICY).errors)

    def test_development_defaults_are_reversible_and_lightweight(self) -> None:
        efficiency = POLICY["efficiency_policy"]
        canary = POLICY["execution_profiles"]["CANARY_LITE"]
        development = POLICY["execution_profiles"]["DEVELOPMENT_STANDARD"]
        self.assertFalse(efficiency["discovery_fresh_holdout_required"])
        self.assertTrue(
            efficiency["early_algorithm_work_prefers_development_or_consumed_data"]
        )
        self.assertFalse(
            efficiency["thesis_prototype_uses_product_safety_certification_by_default"]
        )
        self.assertFalse(canary["fresh_holdout_required"])
        self.assertTrue(
            canary[
                "small_mapping_and_decoder_require_complete_synthetic_canary_first"
            ]
        )
        self.assertTrue(development["rerunnable"])
        self.assertTrue(development["versioned_operational_repair_allowed"])
        self.assertTrue(development["development_truth_may_be_reused"])
        self.assertTrue(development["early_runtime_and_device_benchmark_allowed"])
        self.assertFalse(
            development["formal_candidate_utility_required_before_device_benchmark"]
        )
        self.assertFalse(development["one_shot_default"])
        self.assertFalse(development["full_hash_chain_default"])
        self.assertFalse(development["full_independent_recompute_default"])
        self.assertFalse(development["per_file_sha_freeze_default"])
        self.assertTrue(development["teacher_visible_output_each_round"])

    def test_device_benchmarks_are_split_by_decision_role(self) -> None:
        policy = POLICY["device_benchmark_policy"]
        self.assertEqual(
            [
                "ALGORITHM_SELECTION_BENCHMARK",
                "PLATFORM_ENGINEERING_BENCHMARK",
            ],
            policy["benchmark_types"],
        )
        self.assertFalse(
            policy["formal_model_selection_must_precede_all_device_benchmarks"]
        )
        self.assertFalse(policy["benchmarks_create_confirmation_authority"])
        self.assertTrue(
            policy["algorithm_selection_benchmark"]["candidate_ranking_authority"]
        )
        self.assertFalse(
            policy["platform_engineering_benchmark"]["candidate_ranking_authority"]
        )
        self.assertTrue(
            policy["platform_engineering_benchmark"][
                "proxy_or_synthetic_workload_allowed"
            ]
        )

    def test_confirmation_requires_explicit_activation(self) -> None:
        confirmation = POLICY["execution_profiles"]["CONFIRMATION_STRICT"]
        self.assertTrue(confirmation["explicit_user_activation_required"])
        self.assertFalse(
            confirmation["same_evidence_version_rerunnable_after_outcome_access"]
        )
        self.assertEqual(
            "FIX_AND_RERUN_NEW_EVIDENCE_VERSION_SAME_DATA_ALLOWED_WITH_INCIDENT_LOG",
            confirmation["technical_failure_before_claim_metrics"],
        )

    def test_thesis_first_requirements_cannot_be_silently_reversed(self) -> None:
        weakened = copy.deepcopy(POLICY)
        weakened["execution_profiles"]["DEVELOPMENT_STANDARD"][
            "full_hash_chain_default"
        ] = True
        result = target.validate_policy(weakened)
        self.assertIn("POLICY_THESIS_FIRST_DEVELOPMENT", result.errors)

        weakened = copy.deepcopy(POLICY)
        weakened["execution_profiles"]["CONFIRMATION_STRICT"][
            "explicit_user_activation_required"
        ] = False
        result = target.validate_policy(weakened)
        self.assertIn("POLICY_THESIS_FIRST_CONFIRMATION", result.errors)

        weakened = copy.deepcopy(POLICY)
        weakened["device_benchmark_policy"][
            "formal_model_selection_must_precede_all_device_benchmarks"
        ] = True
        result = target.validate_policy(weakened)
        self.assertIn("POLICY_THESIS_FIRST_DEVICE_BENCHMARKS", result.errors)

    def test_historical_r3_policy_remains_immutable_and_resolvable(self) -> None:
        self.assertEqual(
            V3_FILE_SHA256,
            hashlib.sha256(target.V3_POLICY.read_bytes()).hexdigest(),
        )
        self.assertEqual([], target.validate_policy(V3_POLICY).errors)
        self.assertEqual(
            target.V3_POLICY,
            target.canonical_policy_path("RISK_TIERED_RESEARCH_GOVERNANCE_R3"),
        )
        self.assertEqual(
            target.DEFAULT_POLICY,
            target.canonical_policy_path("THESIS_FIRST_RESEARCH_GOVERNANCE_R4"),
        )

    def test_current_policy_requires_stage_appropriate_profile(self) -> None:
        protocol = base_protocol("DEVELOPMENT")
        protocol["profile"] = "CANARY_LITE"
        result = target.validate_document(protocol, POLICY)
        self.assertIn("EXECUTION_PROFILE_BELOW_STAGE", result.errors)

    def test_lower_stage_may_escalate_profile_with_rationale(self) -> None:
        protocol = base_protocol("CANARY")
        protocol["profile"] = "DEVELOPMENT_STANDARD"
        protocol["profile_escalation_rationale"] = (
            "Claim-critical identity risk requires implementation-level replay."
        )
        result = target.validate_document(protocol, POLICY)
        self.assertFalse(result.errors)

    def test_lower_stage_profile_escalation_needs_named_risk(self) -> None:
        protocol = base_protocol("CANARY")
        protocol["profile"] = "DEVELOPMENT_STANDARD"
        result = target.validate_document(protocol, POLICY)
        self.assertIn("EXECUTION_PROFILE_ESCALATION_RATIONALE", result.errors)

    def test_operational_invalid_may_use_lightweight_incident(self) -> None:
        protocol = base_protocol("CANARY")
        protocol["result_model"] = {
            "execution_validity": "INVALID",
            "scientific_outcome": "NOT_EVALUABLE_DUE_TO_EXECUTION",
            "invalid_execution_effect": "CLOSE_EVIDENCE_VERSION_ONLY",
            "terminal_scope": "EVIDENCE_VERSION",
        }
        protocol["failure_record_mode"] = "LIGHTWEIGHT_OPERATIONAL_INCIDENT"
        protocol["operational_incident"] = {
            "failure_class": "DEPENDENCY_OR_ENVIRONMENT",
            "observation": "The runner stopped before outcome access.",
            "impact_scope": "EVIDENCE_VERSION",
            "scientific_outcome_accessed": False,
            "prevention_or_existing_guard": "Add a dependency preflight.",
        }
        protocol["round_summary"] = {
            key: "NONE" for key in POLICY["required_round_summary_fields"]
        }
        result = target.validate_document(protocol, POLICY)
        self.assertFalse(result.errors)

    def test_scientific_failure_cannot_use_lightweight_incident(self) -> None:
        protocol = base_protocol("CANARY")
        protocol["result_model"] = {
            "execution_validity": "VALID",
            "scientific_outcome": "MECHANISM_DIRECTION_NOT_SUPPORTED",
            "invalid_execution_effect": "CLOSE_EVIDENCE_VERSION_ONLY",
            "terminal_scope": "IMPLEMENTATION_VERSION",
        }
        protocol["failure_record_mode"] = "LIGHTWEIGHT_OPERATIONAL_INCIDENT"
        protocol["operational_incident"] = {
            "failure_class": "SCIENTIFIC_RESULT",
            "observation": "The mechanism was not supported.",
            "impact_scope": "IMPLEMENTATION_VERSION",
            "scientific_outcome_accessed": False,
            "prevention_or_existing_guard": "NONE",
        }
        protocol["round_summary"] = {
            key: "NONE" for key in POLICY["required_round_summary_fields"]
        }
        result = target.validate_document(protocol, POLICY)
        self.assertIn(
            "LIGHTWEIGHT_INCIDENT_REQUIRES_OPERATIONAL_INVALID", result.errors
        )

    def test_discovery_numeric_gap_is_warning_not_blocker(self) -> None:
        protocol = base_protocol()
        protocol["constraints"].append(
            {
                "id": "exploratory-number",
                "class": "DIAGNOSTIC",
                "description": "Exploratory continuous value.",
                "failure_scope": "ITEM",
                "value": 5.0,
            }
        )
        result = target.validate_document(protocol, POLICY)
        self.assertFalse(result.errors)
        self.assertTrue(any(code.startswith("NUMERIC_JUSTIFICATION") for code in result.warnings))

    def test_confirmation_numeric_gate_requires_rationale(self) -> None:
        protocol = base_protocol("CONFIRMATION")
        del protocol["constraints"][1]["calibration_source"]
        result = target.validate_document(protocol, POLICY)
        self.assertIn(
            "NUMERIC_JUSTIFICATION:effect-gate:calibration_source", result.errors
        )

    def test_development_data_cannot_be_repackaged_as_confirmation(self) -> None:
        protocol = base_protocol("CONFIRMATION")
        protocol["data_partitions"].append(
            {
                "id": "data-b",
                "source_identity": "source-b-alias",
                "content_identity": "content-a",
                "identity_basis": "Same fixture under a second label.",
                "independence_group": "group-a",
                "ancestry": [],
                "role": "DEVELOPMENT",
                "outcome_access": "FULL",
                "result_access_state": "TUNED_ON",
                "observation_unit": "CAPTURE_SESSION",
                "split_basis": "SESSION_LEVEL_PREASSIGNMENT",
                "research_track": "DEVELOPMENT_DIAGNOSTIC",
                "reuse_policy": "CANARY_ONLY",
            }
        )
        result = target.validate_document(protocol, POLICY)
        self.assertIn("DATA_CONTENT_LEAKAGE:content-a", result.errors)
        self.assertIn("DATA_INDEPENDENCE_LEAKAGE:group-a", result.errors)

    def test_post_outcome_change_requires_new_version(self) -> None:
        protocol = base_protocol()
        protocol["freeze"]["outcome_access_started"] = True
        result = target.validate_document(protocol, POLICY)
        self.assertIn("POST_OUTCOME_REQUIRES_NEW_VERSION", result.errors)

    def test_content_inspected_same_source_new_session_can_confirm(self) -> None:
        protocol = base_protocol("CONFIRMATION")
        protocol["data_partitions"].append(
            {
                "id": "development-session",
                "source_identity": "source-a",
                "content_identity": "content-development-session",
                "identity_basis": "Distinct capture-session manifest.",
                "independence_group": "session-development",
                "ancestry": ["source-a-device"],
                "role": "DEVELOPMENT",
                "outcome_access": "FULL",
                "result_access_state": "TUNED_ON",
                "observation_unit": "CAPTURE_SESSION",
                "split_basis": "SESSION_LEVEL_PREASSIGNMENT",
                "research_track": "DEVELOPMENT_DIAGNOSTIC",
                "reuse_policy": "DEVELOPMENT_ONLY",
            }
        )
        protocol["data_partitions"][0]["source_identity"] = "source-a"
        protocol["data_partitions"][0][
            "result_access_state"
        ] = "CONTENT_INSPECTED"
        result = target.validate_document(protocol, POLICY)
        self.assertFalse(result.errors)

    def test_output_inspected_confirmation_is_rejected(self) -> None:
        protocol = base_protocol("CONFIRMATION")
        protocol["data_partitions"][0][
            "result_access_state"
        ] = "OUTPUT_INSPECTED"
        result = target.validate_document(protocol, POLICY)
        self.assertIn(
            "CONFIRMATION_RESULT_ACCESS_CONTAMINATED:0", result.errors
        )

    def test_random_clip_split_is_not_independent(self) -> None:
        protocol = base_protocol("CONFIRMATION")
        protocol["data_partitions"][0][
            "split_basis"
        ] = "RANDOM_CLIP_FROM_SAME_SEQUENCE"
        result = target.validate_document(protocol, POLICY)
        self.assertIn("NONINDEPENDENT_SPLIT_BASIS:0", result.errors)

    def test_cartesian_sweep_requires_information_rationale(self) -> None:
        protocol = base_protocol()
        protocol["experiment_design"] = {"search_strategy": "CARTESIAN_SWEEP"}
        result = target.validate_document(protocol, POLICY)
        self.assertIn("CARTESIAN_SWEEP_JUSTIFICATION_REQUIRED", result.errors)
        self.assertIn("CARTESIAN_SWEEP_MAX_TRIALS_REQUIRED", result.errors)

    def test_reopening_failure_requires_material_change(self) -> None:
        protocol = base_protocol()
        protocol["reopens_prior_failure"] = True
        result = target.validate_document(protocol, POLICY)
        self.assertIn("PRIOR_FAILURE_ID_REQUIRED", result.errors)
        self.assertIn("PRIOR_FAILURE_DIFFERENCE_REQUIRED", result.errors)
        self.assertIn("MATERIAL_CHANGE_REQUIRED", result.errors)

    def test_failure_requires_learning_and_narrow_scope(self) -> None:
        protocol = base_protocol()
        protocol["result_model"]["execution_validity"] = "INVALID"
        protocol["result_model"][
            "scientific_outcome"
        ] = POLICY["invalid_execution_scientific_outcome"]
        protocol["result_model"]["terminal_scope"] = "RESEARCH_QUESTION"
        result = target.validate_document(protocol, POLICY)
        self.assertIn("INVALID_EXECUTION_SCOPE_TOO_BROAD", result.errors)
        self.assertIn("FAILURE_LEARNING_REQUIRED", result.errors)
        self.assertIn("ROUND_SUMMARY_REQUIRED", result.errors)

    def test_successful_completed_round_requires_summary(self) -> None:
        protocol = base_protocol()
        protocol["result_model"]["execution_validity"] = "VALID"
        protocol["result_model"]["scientific_outcome"] = POLICY["stages"]["DISCOVERY"][
            "allowed_claims"
        ][0]
        result = target.validate_document(protocol, POLICY)
        self.assertIn("ROUND_SUMMARY_REQUIRED", result.errors)

    def test_b1a_closure_overlay_keeps_question_open(self) -> None:
        record = {
            "schema_version": target.CLOSURE_SCHEMA,
            "governance_policy_id": POLICY["policy_id"],
            "governance_policy_sha256": target._policy_digest(POLICY),
            "record_id": "TEST-CLOSURE-R1",
            "created_on": "2026-07-26",
            "scientific_question": {"id": "RCLE-Q1", "state": "OPEN"},
            "protocol_version": {"id": "RCLE-B1-R5", "state": "CLOSED_INVALID"},
            "evidence_instance": {
                "id": "RCLE-B1A-R5-CANONICAL-1",
                "artifact_integrity": "INVALID",
                "execution_state": "CONSUMED_CLOSED",
                "scientific_inference_allowed": False,
                "failure_class": "SERIALIZATION_CONTRACT",
                "run_claim_sha256": "0" * 64,
                "ledger_sha256": "1" * 64,
                "receipt_sha256": "2" * 64,
            },
            "closure_effects": [
                {
                    "target_type": "evidence_instance",
                    "target_id": "RCLE-B1A-R5-CANONICAL-1",
                    "basis": "serialization_mismatch",
                },
                {
                    "target_type": "dependency_branch",
                    "target_id": "RCLE-B1B-R5",
                    "basis": "explicit_version_dependency",
                },
            ],
            "dependency_edges": [
                {
                    "from_id": "RCLE-B1B-R5",
                    "depends_on_id": "RCLE-B1-R5",
                }
            ],
            "authority_ceiling": "DIAGNOSTIC",
            "recovery": {
                "same_version_rerun_allowed": False,
                "new_version_may_be_proposed": True,
            },
            "failure_learning": {
                "failure_class": "SERIALIZATION_CONTRACT",
                "observation": "Blank grids used different key sets.",
                "inference": "The evidence instance is not replay-identical.",
                "alternative_explanations": ["Scientific outcomes were not tested."],
                "constraint_challenges": ["Terminal scope was too broad."],
                "next_hypotheses": ["Use a staged discovery protocol."],
                "reuse_candidates": ["REGRESSION_FIXTURE", "SOURCE_CHARACTERIZATION"],
                "information_gain": "Exact blank-grid parity must be tested.",
            },
            "round_summary": {
                "new_facts_and_evidence": ["Blank-grid parity was incomplete."],
                "weakened_or_rejected_hypotheses": ["One-shot readiness was weakened."],
                "unresolved_questions": ["RCLE mechanism remains untested."],
                "reusable_assets": ["REGRESSION_FIXTURE"],
                "next_high_information_experiments": ["Progressive discovery."],
                "governance_changes_needed": ["Use narrow closure scope."],
            },
        }
        result = target.validate_document(record, POLICY)
        self.assertFalse(result.errors)

    def test_not_run_cannot_predeclare_scientific_result(self) -> None:
        protocol = base_protocol()
        protocol["result_model"]["scientific_outcome"] = "DATA_CHARACTERIZED"
        result = target.validate_document(protocol, POLICY)
        self.assertIn("NOT_RUN_CANNOT_ASSERT_OUTCOME", result.errors)

    def test_confirmation_requires_stable_partition_identity(self) -> None:
        protocol = base_protocol("CONFIRMATION")
        del protocol["data_partitions"][0]["content_identity"]
        del protocol["data_partitions"][0]["independence_group"]
        protocol["data_partitions"][0]["identity_manifest_ref"] = (
            "scripts/fixtures/research_governance/does-not-exist.json"
        )
        result = target.validate_document(protocol, POLICY)
        self.assertIn("DATA_PARTITION_FIELD:0:content_identity", result.errors)
        self.assertIn("DATA_PARTITION_FIELD:0:independence_group", result.errors)
        self.assertIn("DATA_IDENTITY_MANIFEST:0_REF_NOT_FOUND", result.errors)

    def test_text_only_confirmation_gate_is_rejected(self) -> None:
        protocol = base_protocol("CONFIRMATION")
        gate = protocol["constraints"][1]
        for field_name in ("metric", "operator", "threshold", "unit"):
            gate.pop(field_name)
        result = target.validate_document(protocol, POLICY)
        self.assertIn("GATE_FIELD:effect-gate:metric", result.errors)
        self.assertIn("GATE_FIELD:effect-gate:threshold", result.errors)

    def test_malicious_invalid_closure_is_rejected(self) -> None:
        record = {
            "schema_version": target.CLOSURE_SCHEMA,
            "record_id": "MALICIOUS-CLOSURE",
            "created_on": "2026-07-26",
            "scientific_question": {"id": "Q1", "state": "CLOSED"},
            "protocol_version": {"id": "", "state": "CLOSED_INVALID"},
            "evidence_instance": {
                "id": "E1",
                "artifact_integrity": "CORRUPT",
                "execution_state": "CONSUMED_CLOSED",
                "scientific_inference_allowed": True,
                "failure_class": "SERIALIZATION_CONTRACT",
            },
            "closure_effects": [
                {
                    "target_type": "scientific_question",
                    "target_id": "Q1",
                    "basis": "SERIALIZATION_MISMATCH",
                }
            ],
            "authority_ceiling": "CONFIRMATION",
            "recovery": {
                "same_version_rerun_allowed": False,
                "new_version_may_be_proposed": True,
            },
            "failure_learning": {
                "failure_class": "SERIALIZATION_CONTRACT",
                "observation": "Mismatch.",
                "inference": "No scientific inference.",
                "alternative_explanations": ["Untested."],
                "constraint_challenges": ["Scope."],
                "next_hypotheses": ["Retry differently."],
                "reuse_candidates": ["REGRESSION_FIXTURE"],
                "information_gain": "Parity matters.",
            },
            "round_summary": {
                "new_facts_and_evidence": ["Mismatch."],
                "weakened_or_rejected_hypotheses": ["Readiness."],
                "unresolved_questions": ["Mechanism."],
                "reusable_assets": ["Fixture."],
                "next_high_information_experiments": ["Parity test."],
                "governance_changes_needed": ["Narrow scope."],
            },
        }
        result = target.validate_document(record, POLICY)
        self.assertIn("ARTIFACT_INTEGRITY", result.errors)
        self.assertIn("PROTOCOL_VERSION_ID", result.errors)
        self.assertIn("ILLEGAL_QUESTION_CLOSURE_BASIS", result.errors)
        self.assertIn("INDEPENDENT_RETIREMENT_DECISION_REQUIRED", result.errors)
        self.assertTrue(
            any(code.startswith("EVIDENCE_SHA256:") for code in result.errors)
        )

    def test_weakened_policy_is_rejected(self) -> None:
        weakened = json.loads(json.dumps(POLICY))
        weakened["hard_rules"][
            "canary_or_development_data_must_not_be_confirmation_data"
        ] = False
        result = target.validate_document(base_protocol(), weakened)
        self.assertIn(
            "POLICY_HARD_RULE_DISABLED:canary_or_development_data_must_not_be_confirmation_data",
            result.errors,
        )

    def test_unknown_policy_enum_cannot_expand_validator_semantics(self) -> None:
        expanded = json.loads(json.dumps(POLICY))
        expanded["artifact_integrity_states"].append("CORRUPT")
        result = target.validate_document(base_protocol(), expanded)
        self.assertIn("POLICY_ENUM:artifact_integrity_states", result.errors)

    def test_contract_is_bound_to_exact_policy_revision(self) -> None:
        protocol = base_protocol()
        protocol["governance_policy_sha256"] = "0" * 64
        result = target.validate_document(protocol, POLICY)
        self.assertIn("GOVERNANCE_POLICY_SHA256", result.errors)

    def test_retired_question_requires_bound_independent_decision(self) -> None:
        path = (
            target.REPO_ROOT
            / "docs"
            / "research"
            / "rcle"
            / "RCLE_PHASE_B_BONN_B1A_CLOSURE_SCOPE_2026-07-26.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record["scientific_question"]["state"] = "RETIRED"
        record["closure_effects"].append(
            {
                "target_type": "scientific_question",
                "target_id": record["scientific_question"]["id"],
                "basis": "execution_contract_failure",
            }
        )
        result = target.validate_document(record, POLICY)
        self.assertIn("ILLEGAL_QUESTION_CLOSURE_BASIS", result.errors)
        self.assertIn("INDEPENDENT_RETIREMENT_DECISION_REQUIRED", result.errors)

    def test_consumed_claim_cannot_be_declared_unconsumed_and_rerun(self) -> None:
        path = (
            target.REPO_ROOT
            / "docs"
            / "research"
            / "rcle"
            / "RCLE_PHASE_B_BONN_B1A_CLOSURE_SCOPE_2026-07-26.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record["recovery"].update(
            {
                "same_version_rerun_allowed": True,
                "claim_unconsumed": True,
                "same_version_rerun_justification": "Pretend the claim was unused.",
            }
        )
        result = target.validate_document(record, POLICY)
        self.assertIn("CONSUMED_CLAIM_CANNOT_RERUN_SAME_VERSION", result.errors)
        self.assertIn("CONSUMED_CLAIM_CANNOT_BE_UNCONSUMED", result.errors)

    def test_nonexistent_evidence_cannot_retire_question(self) -> None:
        path = (
            target.REPO_ROOT
            / "docs"
            / "research"
            / "rcle"
            / "RCLE_PHASE_B_BONN_B1A_CLOSURE_SCOPE_2026-07-26.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record["scientific_question"]["state"] = "RETIRED"
        record["closure_effects"].append(
            {
                "target_type": "scientific_question",
                "target_id": record["scientific_question"]["id"],
                "basis": "scope_retired_by_decision",
            }
        )
        record["independent_retirement_decision"] = {
            "decision_id": "FAKE-RETIREMENT",
            "basis": "Two claimed independent failures.",
            "independent_evidence_ids": ["FAKE-E1", "FAKE-E2"],
        }
        record["independent_evidence_registry"] = [
            {
                "id": "FAKE-E1",
                "protocol_id": "FAKE-P1",
                "source_identity": "FAKE-S1",
                "independence_group": "FAKE-G1",
                "ref_type": "LOCAL_JSON",
                "evidence_ref": "docs/research/rcle/does-not-exist-e1.json",
                "content_sha256": "1" * 64,
            },
            {
                "id": "FAKE-E2",
                "protocol_id": "FAKE-P2",
                "source_identity": "FAKE-S2",
                "independence_group": "FAKE-G2",
                "ref_type": "LOCAL_JSON",
                "evidence_ref": "docs/research/rcle/does-not-exist-e2.json",
                "content_sha256": "2" * 64,
            },
        ]
        result = target.validate_document(record, POLICY)
        self.assertIn("INDEPENDENT_EVIDENCE:0_REF_NOT_FOUND", result.errors)
        self.assertIn("INDEPENDENT_EVIDENCE:1_REF_NOT_FOUND", result.errors)

    def test_cli_returns_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "protocol.json"
            path.write_text(
                json.dumps(base_protocol(), ensure_ascii=False), encoding="utf-8"
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
            result = target.validate_document(loaded, POLICY)
            self.assertEqual("VALID", result.payload(str(path))["status"])


if __name__ == "__main__":
    unittest.main()
