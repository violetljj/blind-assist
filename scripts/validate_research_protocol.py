#!/usr/bin/env python3
"""Validate progressive research protocols and closure-scope overlays.

The validator deliberately separates hard governance errors from method warnings:
early research stages may continue with disclosed warnings, while confirmation
cannot silently omit threshold rationale or reuse development data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "configs" / "research_governance_v4.json"
V3_POLICY = REPO_ROOT / "configs" / "research_governance_v3.json"
R2_POLICY = REPO_ROOT / "configs" / "research_governance_v2.json"
LEGACY_POLICY = REPO_ROOT / "configs" / "research_governance_v1.json"
PROTOCOL_SCHEMA = "blindassist.research_protocol.v1"
CLOSURE_SCHEMA = "blindassist.research_closure_scope.v1"
POLICY_SCHEMA = "blindassist.research_governance.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

REQUIRED_POLICY_STAGES = {
    "DISCOVERY",
    "CANARY",
    "DEVELOPMENT",
    "CONFIRMATION",
    "DEPLOYMENT",
}
REQUIRED_HARD_RULES = {
    "canary_or_development_data_must_not_be_confirmation_data",
    "post_outcome_amendment_requires_new_version",
    "confirmation_requires_at_least_one_gate",
    "confirmation_numeric_gates_require_full_justification",
    "invalid_execution_does_not_close_research_question_by_default",
    "legacy_evidence_is_immutable",
    "failure_requires_learning_record",
    "blind_cartesian_search_requires_explicit_justification",
    "rules_may_be_challenged_but_not_silently_bypassed",
    "failed_assets_may_be_reused_only_with_explicit_new_role",
}
DATA_DRIVEN_HARD_RULES = {
    "content_inspection_does_not_burn_algorithm_outcome",
    "same_source_independent_session_may_be_confirmation",
    "random_frame_or_clip_split_is_not_independent",
    "capability_map_is_not_an_execution_gate",
    "one_source_need_not_answer_every_question",
    "external_transfer_is_separate_from_ordinary_holdout",
}
DATA_DRIVEN_POLICY_IDS = {
    "DATA_CAPABILITY_DRIVEN_RESEARCH_GOVERNANCE_R2",
    "RISK_TIERED_RESEARCH_GOVERNANCE_R3",
    "THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
}
PROFILE_POLICY_IDS = {
    "RISK_TIERED_RESEARCH_GOVERNANCE_R3",
    "THESIS_FIRST_RESEARCH_GOVERNANCE_R4",
}
THESIS_FIRST_POLICY_ID = "THESIS_FIRST_RESEARCH_GOVERNANCE_R4"
STAGE_KERNEL = {
    "DISCOVERY": (
        "F0",
        False,
        {"CANDIDATE_FOUND", "CANDIDATE_NOT_FOUND", "DATA_CHARACTERIZED"},
    ),
    "CANARY": (
        "F1",
        False,
        {
            "MECHANISM_DIRECTION_SUPPORTED",
            "MECHANISM_DIRECTION_NOT_SUPPORTED",
            "IMPLEMENTATION_DEBUGGED",
            "NOT_EVALUABLE",
        },
    ),
    "DEVELOPMENT": (
        "F1",
        False,
        {
            "IMPLEMENTATION_READY_FOR_CONFIRMATION",
            "IMPLEMENTATION_NOT_READY",
            "NOT_EVALUABLE",
        },
    ),
    "CONFIRMATION": (
        "F2",
        True,
        {"CONFIRM_PASS", "CONFIRM_FAIL", "NOT_EVALUABLE"},
    ),
    "DEPLOYMENT": (
        "F3",
        True,
        {"DEPLOYMENT_GATE_PASS", "DEPLOYMENT_GATE_FAIL", "NOT_EVALUABLE"},
    ),
}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, code: str) -> None:
        if code not in self.errors:
            self.errors.append(code)

    def warn(self, code: str) -> None:
        if code not in self.warnings:
            self.warnings.append(code)

    def payload(self, subject: str) -> dict[str, Any]:
        return {
            "status": "VALID" if not self.errors else "INVALID",
            "subject": subject,
            "errors": sorted(self.errors),
            "warnings": sorted(self.warnings),
        }


def _object(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _load_bound_repo_json(
    reference: Any,
    expected_sha256: Any,
    result: ValidationResult,
    code_prefix: str,
) -> dict[str, Any] | None:
    if not _nonempty_text(reference):
        result.error(f"{code_prefix}_REF")
        return None
    relative = Path(reference)
    if relative.is_absolute():
        result.error(f"{code_prefix}_REF_SCOPE")
        return None
    root = REPO_ROOT.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        result.error(f"{code_prefix}_REF_SCOPE")
        return None
    if not resolved.is_file():
        result.error(f"{code_prefix}_REF_NOT_FOUND")
        return None
    content = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
        result.error(f"{code_prefix}_SHA256_MISMATCH")
        return None
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        result.error(f"{code_prefix}_JSON")
        return None
    if not isinstance(value, dict):
        result.error(f"{code_prefix}_OBJECT")
        return None
    return value


def _sha256_text(value: str) -> bool:
    return bool(SHA256_RE.fullmatch(value))


def _policy_digest(policy: dict[str, Any]) -> str:
    canonical = json.dumps(
        policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _nonempty_text_list(value: Any, minimum: int = 1) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= minimum
        and all(_nonempty_text(item) for item in value)
    )


def _merge(target: ValidationResult, source: ValidationResult) -> None:
    for code in source.errors:
        target.error(code)
    for code in source.warnings:
        target.warn(code)


def validate_policy(policy: dict[str, Any]) -> ValidationResult:
    """Validate the minimum integrity kernel before using a revisable policy.

    Policies may evolve, but a caller cannot silently remove the protections that
    make a protocol auditable and still use this validator as an authority gate.
    """

    result = ValidationResult()
    if policy.get("schema_version") != POLICY_SCHEMA:
        result.error("POLICY_SCHEMA_VERSION")
    if not _nonempty_text(policy.get("policy_id")):
        result.error("POLICY_ID")
    if not DATE_RE.fullmatch(str(policy.get("adopted_on", ""))):
        result.error("POLICY_ADOPTED_ON")
    if policy.get("policy_is_revisable") is not True:
        result.error("POLICY_MUST_BE_REVISABLE")
    if policy.get("user_guidance_is_revisable") is not True:
        result.error("USER_GUIDANCE_MUST_BE_REVISABLE")
    priorities = policy.get("project_objective_priority")
    if not _nonempty_text_list(priorities) or "INFORMATION_GAIN_PER_UNIT_COST" not in priorities:
        result.error("POLICY_EFFICIENCY_PRIORITY")
    efficiency = _object(policy.get("efficiency_policy"))
    if efficiency is None:
        result.error("POLICY_EFFICIENCY")
    else:
        if efficiency.get("default") != "MINIMUM_SUFFICIENT_RIGOR":
            result.error("POLICY_EFFICIENCY_DEFAULT")
        for name in (
            "allow_reversible_shortcuts",
            "allow_parallel_independent_work",
            "allow_lightweight_posthoc_recording_in_discovery",
        ):
            if efficiency.get(name) is not True:
                result.error(f"POLICY_EFFICIENCY_DISABLED:{name}")
        if not _nonempty_text_list(efficiency.get("forbidden_shortcuts")):
            result.error("POLICY_FORBIDDEN_SHORTCUTS")
        if not _nonempty_text(efficiency.get("escalation_rule")):
            result.error("POLICY_EFFICIENCY_ESCALATION")

    stages = _object(policy.get("stages"))
    if stages is None or not REQUIRED_POLICY_STAGES.issubset(stages):
        result.error("POLICY_STAGES")
    else:
        for stage in REQUIRED_POLICY_STAGES:
            stage_policy = _object(stages.get(stage))
            if stage_policy is None:
                result.error(f"POLICY_STAGE_OBJECT:{stage}")
                continue
            minimum_level, confirmation_authority, allowed_kernel = STAGE_KERNEL[stage]
            if stage_policy.get("minimum_freeze_level") != minimum_level:
                result.error(f"POLICY_STAGE_FREEZE:{stage}")
            claims = stage_policy.get("allowed_claims")
            if not _nonempty_text_list(claims) or not set(claims).issubset(
                allowed_kernel
            ):
                result.error(f"POLICY_STAGE_CLAIMS:{stage}")
            if stage_policy.get("confirmation_authority") is not confirmation_authority:
                result.error(f"POLICY_STAGE_AUTHORITY:{stage}")

    required_lists = {
        "constraint_classes": {
            "INVARIANT",
            "GATE",
            "GUARDRAIL",
            "DIAGNOSTIC",
            "ASSUMPTION",
        },
        "data_roles": REQUIRED_POLICY_STAGES,
        "outcome_access_levels": {"NONE", "METADATA_ONLY", "GEOMETRY_ONLY", "FULL"},
        "failure_scopes": {
            "ITEM",
            "WINDOW",
            "SEQUENCE",
            "BRANCH",
            "IMPLEMENTATION_VERSION",
            "EVIDENCE_VERSION",
            "RESEARCH_QUESTION",
            "PRODUCT",
        },
        "amendment_modes": {"IN_PLACE_BEFORE_OUTCOME", "NEW_VERSION_ONLY"},
        "execution_validity_values": {"VALID", "INVALID", "NOT_RUN"},
        "constraint_operators": {"GT", "GTE", "LT", "LTE", "EQ", "NEQ", "RANGE_INCLUSIVE"},
        "closure_target_types": {
            "scientific_question",
            "protocol_version",
            "evidence_instance",
            "dependency_branch",
        },
        "scientific_question_states": {"OPEN", "CLOSED", "RETIRED"},
        "protocol_version_states": {
            "OPEN",
            "CLOSED_VALID",
            "CLOSED_INVALID",
            "SUPERSEDED",
        },
        "artifact_integrity_states": {"VALID", "INVALID"},
        "closure_execution_states": {"NOT_RUN", "COMPLETED", "CONSUMED_CLOSED"},
        "authority_ceilings": {
            "NONE",
            "DIAGNOSTIC",
            "DISCOVERY",
            "CANARY",
            "DEVELOPMENT",
            "CONFIRMATION",
            "DEPLOYMENT",
        },
        "question_closure_bases": {
            "independent_confirmatory_failure",
            "theoretical_refutation",
            "scope_retired_by_decision",
        },
    }
    for name, required in required_lists.items():
        value = policy.get(name)
        if (
            not _nonempty_text_list(value)
            or len(value) != len(set(value))
            or set(value) != required
        ):
            result.error(f"POLICY_ENUM:{name}")
    if policy.get("freeze_levels") != ["F0", "F1", "F2", "F3"]:
        result.error("POLICY_ENUM:freeze_levels")

    required_field_sets = {
        "minimum_numeric_constraint_fields": {
            "unit",
            "rationale",
            "calibration_source",
            "sensitivity_plan",
            "revision_policy",
        },
        "required_failure_learning_fields": {
            "failure_class",
            "observation",
            "inference",
            "alternative_explanations",
            "constraint_challenges",
            "next_hypotheses",
            "reuse_candidates",
            "information_gain",
        },
        "required_hypothesis_fields": {
            "hypothesis_id",
            "theoretical_or_empirical_basis",
            "causal_difference",
            "expected_information_gain",
            "minimal_test",
            "evaluation_metric",
            "falsifier",
            "cost",
            "resource_budget",
            "stop_condition",
            "selection_reason",
        },
        "required_experiment_design_fields": {
            "search_strategy",
            "minimal_discriminating_experiment",
            "resource_budget",
            "stop_conditions",
        },
        "required_round_summary_fields": {
            "new_facts_and_evidence",
            "weakened_or_rejected_hypotheses",
            "unresolved_questions",
            "reusable_assets",
            "next_high_information_experiments",
            "governance_changes_needed",
        },
        "material_change_dimensions": {
            "DATA",
            "INPUT_SIGNAL",
            "COMPENSATION",
            "SYSTEM_ROLE",
            "EVALUATION_TARGET",
            "DEPLOYMENT_CONDITION",
        },
        "required_data_partition_fields": {
            "source_identity",
            "content_identity",
            "identity_basis",
            "independence_group",
            "ancestry",
            "reuse_policy",
        },
        "required_evidence_hash_fields": {
            "run_claim_sha256",
            "ledger_sha256",
            "receipt_sha256",
        },
    }
    for name, required in required_field_sets.items():
        value = policy.get(name)
        if not _nonempty_text_list(value) or not required.issubset(set(value)):
            result.error(f"POLICY_REQUIRED_LIST:{name}")

    hard_rules = _object(policy.get("hard_rules"))
    if hard_rules is None:
        result.error("POLICY_HARD_RULES")
    else:
        for name in REQUIRED_HARD_RULES:
            if hard_rules.get(name) is not True:
                result.error(f"POLICY_HARD_RULE_DISABLED:{name}")
        if policy.get("policy_id") in DATA_DRIVEN_POLICY_IDS:
            for name in DATA_DRIVEN_HARD_RULES:
                if hard_rules.get(name) is not True:
                    result.error(f"POLICY_HARD_RULE_DISABLED:{name}")

    if policy.get("policy_id") in DATA_DRIVEN_POLICY_IDS:
        exact_lists = {
            "result_access_states": {
                "CONTENT_INSPECTED",
                "OUTPUT_INSPECTED",
                "TUNED_ON",
                "SEALED_UNSEEN",
            },
            "independence_units": {
                "PERSON",
                "CAPTURE_SESSION",
                "ROUTE",
                "SEQUENCE",
            },
            "capability_map_columns": {
                "dataset_id",
                "sequence_id",
                "scene_motion",
                "available_modalities",
                "observation_unit",
                "access_cost",
                "outcome_access_state",
                "assigned_role",
                "claim_ceiling",
                "notes",
            },
        }
        for name, expected in exact_lists.items():
            value = policy.get(name)
            if (
                not _nonempty_text_list(value)
                or len(value) != len(set(value))
                or set(value) != expected
            ):
                result.error(f"POLICY_ENUM:{name}")
        tracks = _object(policy.get("research_tracks"))
        expected_tracks = {
            "CAPABILITY_DISCOVERY",
            "DEVELOPMENT_DIAGNOSTIC",
            "SEALED_EVALUATION",
            "EXTERNAL_TRANSFER",
        }
        if tracks is None or set(tracks) != expected_tracks:
            result.error("POLICY_RESEARCH_TRACKS")
        else:
            for name, track in tracks.items():
                if not isinstance(track, dict):
                    result.error(f"POLICY_RESEARCH_TRACK:{name}")
                    continue
                stages_value = track.get("allowed_stages")
                if (
                    not _nonempty_text_list(stages_value)
                    or not set(stages_value).issubset(REQUIRED_POLICY_STAGES)
                ):
                    result.error(f"POLICY_RESEARCH_TRACK_STAGES:{name}")
                if not _nonempty_text(track.get("purpose")):
                    result.error(f"POLICY_RESEARCH_TRACK_PURPOSE:{name}")

    if policy.get("policy_id") in PROFILE_POLICY_IDS:
        profiles = _object(policy.get("execution_profiles"))
        expected_profiles = {
            "CANARY_LITE",
            "DEVELOPMENT_STANDARD",
            "CONFIRMATION_STRICT",
        }
        if profiles is None or set(profiles) != expected_profiles:
            result.error("POLICY_EXECUTION_PROFILES")
        else:
            for profile_name, profile in profiles.items():
                if not isinstance(profile, dict):
                    result.error(f"POLICY_EXECUTION_PROFILE:{profile_name}")
                    continue
                if not _nonempty_text_list(profile.get("default_stages")):
                    result.error(f"POLICY_EXECUTION_PROFILE_STAGES:{profile_name}")
                if not _nonempty_text_list(profile.get("allowed_freeze_levels")):
                    result.error(f"POLICY_EXECUTION_PROFILE_FREEZE:{profile_name}")
                if not _nonempty_text(profile.get("review_mode")):
                    result.error(f"POLICY_EXECUTION_PROFILE_REVIEW:{profile_name}")
                if not _nonempty_text_list(profile.get("minimum_artifacts")):
                    result.error(f"POLICY_EXECUTION_PROFILE_ARTIFACTS:{profile_name}")
        selection = _object(policy.get("profile_selection_rules"))
        defaults = _object(selection.get("default_by_stage")) if selection else None
        if defaults is None or set(defaults) != REQUIRED_POLICY_STAGES:
            result.error("POLICY_PROFILE_STAGE_DEFAULTS")
        elif any(value not in expected_profiles for value in defaults.values()):
            result.error("POLICY_PROFILE_STAGE_DEFAULT_VALUE")
        if set(policy.get("failure_record_modes", [])) != {
            "FULL_FAILURE_LEARNING",
            "LIGHTWEIGHT_OPERATIONAL_INCIDENT",
        }:
            result.error("POLICY_FAILURE_RECORD_MODES")
    if policy.get("policy_id") == THESIS_FIRST_POLICY_ID:
        efficiency = _object(policy.get("efficiency_policy"))
        if (
            efficiency is None
            or efficiency.get("default_research_mode")
            != "REVERSIBLE_DEVELOPMENT_UNLESS_FINAL_CONFIRMATION_IS_EXPLICITLY_ACTIVATED"
            or efficiency.get("final_confirmation_requires_explicit_user_activation")
            is not True
            or efficiency.get("allow_early_runtime_and_device_benchmark_in_development")
            is not True
        ):
            result.error("POLICY_THESIS_FIRST_EFFICIENCY")
        profiles = _object(policy.get("execution_profiles"))
        development = _object(profiles.get("DEVELOPMENT_STANDARD")) if profiles else None
        if (
            development is None
            or development.get("rerunnable") is not True
            or development.get("versioned_operational_repair_allowed") is not True
            or development.get("development_truth_may_be_reused") is not True
            or development.get("early_runtime_and_device_benchmark_allowed") is not True
            or development.get("full_hash_chain_default") is not False
            or development.get("full_independent_recompute_default") is not False
            or development.get("per_file_sha_freeze_default") is not False
            or development.get("teacher_visible_output_each_round") is not True
        ):
            result.error("POLICY_THESIS_FIRST_DEVELOPMENT")
        confirmation = _object(profiles.get("CONFIRMATION_STRICT")) if profiles else None
        if (
            confirmation is None
            or confirmation.get("explicit_user_activation_required") is not True
            or confirmation.get("same_evidence_version_rerunnable_after_outcome_access")
            is not False
            or confirmation.get("technical_failure_before_claim_metrics")
            != "FIX_AND_RERUN_NEW_EVIDENCE_VERSION_SAME_DATA_ALLOWED_WITH_INCIDENT_LOG"
            or confirmation.get("outcome_informed_algorithm_change")
            != "SAME_DATA_DEVELOPMENT_ONLY_NEW_CONFIRMATION_DATA_REQUIRED"
        ):
            result.error("POLICY_THESIS_FIRST_CONFIRMATION")
        selection = _object(policy.get("profile_selection_rules"))
        if (
            selection is None
            or selection.get("final_confirmation_requires_explicit_user_activation")
            is not True
            or selection.get("development_escalation_to_confirmation_is_never_automatic")
            is not True
            or selection.get(
                "device_benchmark_is_development_engineering_evidence_not_confirmation"
            )
            is not True
        ):
            result.error("POLICY_THESIS_FIRST_SELECTION")
        hard_rules = _object(policy.get("hard_rules"))
        for name in (
            "development_repair_and_rerun_allowed",
            "device_benchmark_may_precede_model_selection",
            "confirmation_only_after_explicit_user_activation",
            "operational_failure_does_not_close_candidate_or_research_question",
        ):
            if hard_rules is None or hard_rules.get(name) is not True:
                result.error(f"POLICY_THESIS_FIRST_HARD_RULE:{name}")

    expected_values = {
        "default_invalid_execution_effect": "CLOSE_EVIDENCE_VERSION_ONLY",
        "not_run_scientific_outcome": "NOT_RUN",
        "invalid_execution_scientific_outcome": "NOT_EVALUABLE_DUE_TO_EXECUTION",
    }
    for name, expected in expected_values.items():
        if policy.get(name) != expected:
            result.error(f"POLICY_VALUE:{name}")
    return result


def _validate_hypotheses(
    contract: dict[str, Any], policy: dict[str, Any], result: ValidationResult
) -> None:
    hypotheses = contract.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        result.error("HYPOTHESES_REQUIRED")
        return
    required = policy["required_hypothesis_fields"]
    seen: set[str] = set()
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            result.error(f"HYPOTHESIS_OBJECT:{index}")
            continue
        for field_name in required:
            if not _nonempty_text(hypothesis.get(field_name)):
                result.error(f"HYPOTHESIS_FIELD:{index}:{field_name}")
        hypothesis_id = hypothesis.get("hypothesis_id")
        if _nonempty_text(hypothesis_id):
            if hypothesis_id in seen:
                result.error(f"HYPOTHESIS_DUPLICATE:{hypothesis_id}")
            seen.add(hypothesis_id)

    design = _object(contract.get("experiment_design"))
    if design is None:
        result.error("EXPERIMENT_DESIGN_REQUIRED")
        design = {}
    for field_name in policy["required_experiment_design_fields"]:
        if not _nonempty_text(design.get(field_name)):
            result.error(f"EXPERIMENT_DESIGN_FIELD:{field_name}")
    if design.get("search_strategy") == "CARTESIAN_SWEEP":
        if not _nonempty_text(design.get("sweep_justification")):
            result.error("CARTESIAN_SWEEP_JUSTIFICATION_REQUIRED")
        max_trials = design.get("max_trials")
        if not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials <= 0:
            result.error("CARTESIAN_SWEEP_MAX_TRIALS_REQUIRED")

    if contract.get("reopens_prior_failure") is True:
        if not _nonempty_text(contract.get("prior_failure_id")):
            result.error("PRIOR_FAILURE_ID_REQUIRED")
        if not _nonempty_text(contract.get("difference_from_previous")):
            result.error("PRIOR_FAILURE_DIFFERENCE_REQUIRED")
        dimensions = contract.get("material_change_dimensions")
        if not isinstance(dimensions, list) or not dimensions:
            result.error("MATERIAL_CHANGE_REQUIRED")
        else:
            for dimension in dimensions:
                if dimension not in policy["material_change_dimensions"]:
                    result.error(f"MATERIAL_CHANGE_DIMENSION:{dimension}")


def _validate_failure_learning(
    contract: dict[str, Any], policy: dict[str, Any], result: ValidationResult
) -> None:
    result_model = _object(contract.get("result_model")) or {}
    execution = result_model.get("execution_validity")
    outcome = result_model.get("scientific_outcome")
    failureish = execution == "INVALID" or (
        isinstance(outcome, str)
        and any(
            token in outcome
            for token in ("FAIL", "NOT_EVALUABLE", "NOT_SUPPORTED", "NOT_READY", "NOT_FOUND")
        )
    )
    if execution != "NOT_RUN":
        _validate_round_summary(contract, policy, result)
    if not failureish:
        return
    if (
        policy.get("policy_id") in PROFILE_POLICY_IDS
        and contract.get("failure_record_mode")
        == "LIGHTWEIGHT_OPERATIONAL_INCIDENT"
    ):
        if execution != "INVALID" or outcome != "NOT_EVALUABLE_DUE_TO_EXECUTION":
            result.error("LIGHTWEIGHT_INCIDENT_REQUIRES_OPERATIONAL_INVALID")
            return
        incident = _object(contract.get("operational_incident"))
        if incident is None:
            result.error("OPERATIONAL_INCIDENT_REQUIRED")
            return
        for field_name in (
            "failure_class",
            "observation",
            "impact_scope",
            "prevention_or_existing_guard",
        ):
            if not _nonempty_text(incident.get(field_name)):
                result.error(f"OPERATIONAL_INCIDENT_FIELD:{field_name}")
        if incident.get("scientific_outcome_accessed") is not False:
            result.error("OPERATIONAL_INCIDENT_SCIENTIFIC_OUTCOME_ACCESSED")
        return
    learning = _object(contract.get("failure_learning"))
    if learning is None:
        result.error("FAILURE_LEARNING_REQUIRED")
        return
    for field_name in policy["required_failure_learning_fields"]:
        value = learning.get(field_name)
        if field_name in {
            "alternative_explanations",
            "constraint_challenges",
            "next_hypotheses",
            "reuse_candidates",
        }:
            if not isinstance(value, list):
                result.error(f"FAILURE_LEARNING_LIST:{field_name}")
        elif not _nonempty_text(value):
            result.error(f"FAILURE_LEARNING_FIELD:{field_name}")


def _validate_round_summary(
    document: dict[str, Any], policy: dict[str, Any], result: ValidationResult
) -> None:
    summary = _object(document.get("round_summary"))
    if summary is None:
        result.error("ROUND_SUMMARY_REQUIRED")
        return
    for field_name in policy["required_round_summary_fields"]:
        value = summary.get(field_name)
        if isinstance(value, list):
            if not value:
                result.error(f"ROUND_SUMMARY_FIELD:{field_name}")
        elif not _nonempty_text(value):
            result.error(f"ROUND_SUMMARY_FIELD:{field_name}")


def _validate_independent_evidence_registry(
    document: dict[str, Any],
    evidence_ids: Any,
    result: ValidationResult,
) -> None:
    valid_ids = _nonempty_text_list(evidence_ids, minimum=2)
    if valid_ids and len(evidence_ids) != len(set(evidence_ids)):
        valid_ids = False
    if not valid_ids:
        result.error("INDEPENDENT_RETIREMENT_EVIDENCE")

    registry = document.get("independent_evidence_registry")
    registry_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(registry, list) or not registry:
        result.error("INDEPENDENT_EVIDENCE_REGISTRY_REQUIRED")
    else:
        for index, item in enumerate(registry):
            if not isinstance(item, dict):
                result.error(f"INDEPENDENT_EVIDENCE_OBJECT:{index}")
                continue
            evidence_id = item.get("id")
            if not _nonempty_text(evidence_id):
                result.error(f"INDEPENDENT_EVIDENCE_ID:{index}")
                continue
            if evidence_id in registry_by_id:
                result.error(f"INDEPENDENT_EVIDENCE_DUPLICATE:{evidence_id}")
            registry_by_id[evidence_id] = item
            for field_name in (
                "source_identity",
                "protocol_id",
                "independence_group",
            ):
                if not _nonempty_text(item.get(field_name)):
                    result.error(f"INDEPENDENT_EVIDENCE_FIELD:{index}:{field_name}")
            if not isinstance(item.get("content_sha256"), str) or not _sha256_text(
                item["content_sha256"]
            ):
                result.error(f"INDEPENDENT_EVIDENCE_SHA256:{index}")
            if item.get("ref_type") != "LOCAL_JSON":
                result.error(f"INDEPENDENT_EVIDENCE_REF_TYPE:{index}")
                continue
            bound = _load_bound_repo_json(
                item.get("evidence_ref"),
                item.get("content_sha256"),
                result,
                f"INDEPENDENT_EVIDENCE:{index}",
            )
            if bound is not None:
                if (
                    bound.get("schema_version")
                    != "blindassist.research_evidence_reference.v1"
                ):
                    result.error(f"INDEPENDENT_EVIDENCE_SCHEMA:{index}")
                expected_fields = {
                    "evidence_id": item.get("id"),
                    "protocol_id": item.get("protocol_id"),
                    "source_identity": item.get("source_identity"),
                    "independence_group": item.get("independence_group"),
                }
                for field_name, expected in expected_fields.items():
                    if bound.get(field_name) != expected:
                        result.error(
                            f"INDEPENDENT_EVIDENCE_BOUND_FIELD:{index}:{field_name}"
                        )

    if valid_ids:
        selected = [
            registry_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in registry_by_id
        ]
        if len(selected) != len(evidence_ids):
            result.error("INDEPENDENT_RETIREMENT_EVIDENCE_UNBOUND")
        groups = [item.get("independence_group") for item in selected]
        if len(groups) != len(set(groups)):
            result.error("INDEPENDENT_RETIREMENT_GROUPS_NOT_UNIQUE")


def validate_protocol(
    contract: dict[str, Any], policy: dict[str, Any]
) -> ValidationResult:
    result = ValidationResult()
    for name in (
        "protocol_id",
        "version",
        "stage",
        "question",
        "claims_allowed",
        "data_partitions",
        "constraints",
        "freeze",
        "result_model",
        "successor_policy",
    ):
        if name not in contract:
            result.error(f"REQUIRED:{name}")

    stage = contract.get("stage")
    stages = policy["stages"]
    if stage not in stages:
        result.error("STAGE")
        return result
    stage_policy = stages[stage]

    selected_profile: dict[str, Any] | None = None
    if policy.get("policy_id") in PROFILE_POLICY_IDS:
        profile_name = contract.get("profile")
        profiles = policy["execution_profiles"]
        profile = _object(profiles.get(profile_name))
        if profile is None:
            result.error("EXECUTION_PROFILE")
        else:
            selected_profile = profile
            default_profile = policy["profile_selection_rules"]["default_by_stage"][stage]
            profile_rank = {
                "CANARY_LITE": 0,
                "DEVELOPMENT_STANDARD": 1,
                "CONFIRMATION_STRICT": 2,
            }
            if profile_rank[profile_name] < profile_rank[default_profile]:
                result.error("EXECUTION_PROFILE_BELOW_STAGE")
            if stage not in profile["default_stages"] and not _nonempty_text(
                contract.get("profile_escalation_rationale")
            ):
                result.error("EXECUTION_PROFILE_ESCALATION_RATIONALE")

    claims = contract.get("claims_allowed")
    if not isinstance(claims, list) or not claims:
        result.error("CLAIMS_ALLOWED")
    else:
        allowed = set(stage_policy["allowed_claims"])
        for claim in claims:
            if claim not in allowed:
                result.error(f"CLAIM_EXCEEDS_STAGE:{claim}")

    freeze = _object(contract.get("freeze"))
    if freeze is None:
        result.error("FREEZE_OBJECT")
    else:
        level = freeze.get("level")
        levels = policy["freeze_levels"]
        if level not in levels:
            result.error("FREEZE_LEVEL")
        elif levels.index(level) < levels.index(stage_policy["minimum_freeze_level"]):
            result.error("FREEZE_LEVEL_BELOW_STAGE")
        elif (
            selected_profile is not None
            and level not in selected_profile["allowed_freeze_levels"]
        ):
            result.error("FREEZE_LEVEL_OUTSIDE_PROFILE")
        amendment = freeze.get("amendment_mode")
        if amendment not in policy["amendment_modes"]:
            result.error("AMENDMENT_MODE")
        if freeze.get("outcome_access_started") is True and amendment != "NEW_VERSION_ONLY":
            result.error("POST_OUTCOME_REQUIRES_NEW_VERSION")

    partitions = contract.get("data_partitions")
    seen_ids: set[str] = set()
    identities: dict[str, set[str]] = {}
    independence_groups: dict[str, set[str]] = {}
    if not isinstance(partitions, list) or not partitions:
        result.error("DATA_PARTITIONS")
    else:
        for index, partition in enumerate(partitions):
            if not isinstance(partition, dict):
                result.error(f"DATA_PARTITION_OBJECT:{index}")
                continue
            partition_id = partition.get("id")
            role = partition.get("role")
            access = partition.get("outcome_access")
            if not _nonempty_text(partition_id):
                result.error(f"DATA_PARTITION_ID:{index}")
            elif partition_id in seen_ids:
                result.error(f"DATA_PARTITION_DUPLICATE:{partition_id}")
            else:
                seen_ids.add(partition_id)
            if role not in policy["data_roles"]:
                result.error(f"DATA_ROLE:{index}")
            if access not in policy["outcome_access_levels"]:
                result.error(f"OUTCOME_ACCESS:{index}")
            if policy.get("policy_id") in DATA_DRIVEN_POLICY_IDS:
                result_access = partition.get("result_access_state")
                observation_unit = partition.get("observation_unit")
                split_basis = partition.get("split_basis")
                if result_access not in policy["result_access_states"]:
                    result.error(f"RESULT_ACCESS_STATE:{index}")
                if observation_unit not in policy["independence_units"]:
                    result.error(f"OBSERVATION_UNIT:{index}")
                if not _nonempty_text(split_basis):
                    result.error(f"SPLIT_BASIS:{index}")
                elif split_basis in {
                    "RANDOM_FRAME",
                    "RANDOM_CLIP_FROM_SAME_SEQUENCE",
                }:
                    result.error(f"NONINDEPENDENT_SPLIT_BASIS:{index}")
                if role in {"CONFIRMATION", "DEPLOYMENT"}:
                    if result_access not in {
                        "CONTENT_INSPECTED",
                        "SEALED_UNSEEN",
                    }:
                        result.error(
                            f"CONFIRMATION_RESULT_ACCESS_CONTAMINATED:{index}"
                        )
                    if (
                        role == "CONFIRMATION"
                        and partition.get("research_track")
                        not in {"SEALED_EVALUATION", "EXTERNAL_TRANSFER"}
                    ):
                        result.error(f"CONFIRMATION_RESEARCH_TRACK:{index}")
                elif result_access == "SEALED_UNSEEN":
                    result.warn(f"SEALED_DATA_ASSIGNED_NONCONFIRMATION:{index}")
                track = partition.get("research_track")
                if track not in policy["research_tracks"]:
                    result.error(f"RESEARCH_TRACK:{index}")
            for field_name in policy["required_data_partition_fields"]:
                value = partition.get(field_name)
                if field_name == "ancestry":
                    if not isinstance(value, list) or not all(
                        _nonempty_text(item) for item in value
                    ):
                        result.error(f"DATA_PARTITION_FIELD:{index}:{field_name}")
                elif not _nonempty_text(value):
                    result.error(f"DATA_PARTITION_FIELD:{index}:{field_name}")
            content_identity = partition.get("content_identity")
            independence_group = partition.get("independence_group")
            ancestry = partition.get("ancestry")
            if role in {"CONFIRMATION", "DEPLOYMENT"}:
                identity_sha256 = partition.get("identity_sha256")
                identity_manifest = _load_bound_repo_json(
                    partition.get("identity_manifest_ref"),
                    identity_sha256,
                    result,
                    f"DATA_IDENTITY_MANIFEST:{index}",
                )
                if identity_manifest is not None:
                    if (
                        identity_manifest.get("schema_version")
                        != "blindassist.data_identity_manifest.v1"
                    ):
                        result.error(f"DATA_IDENTITY_MANIFEST_SCHEMA:{index}")
                    if identity_manifest.get("protocol_id") != contract.get("protocol_id"):
                        result.error(f"DATA_IDENTITY_MANIFEST_PROTOCOL:{index}")
                    for field_name in (
                        "source_identity",
                        "content_identity",
                        "independence_group",
                        "ancestry",
                    ):
                        if identity_manifest.get(field_name) != partition.get(field_name):
                            result.error(
                                f"DATA_IDENTITY_MANIFEST_FIELD:{index}:{field_name}"
                            )
                if not _nonempty_text(partition.get("disjointness_evidence")):
                    result.error(f"DATA_DISJOINTNESS_EVIDENCE:{index}")
            if _nonempty_text(content_identity) and role in policy["data_roles"]:
                identities.setdefault(content_identity, set()).add(role)
            if _nonempty_text(independence_group) and role in policy["data_roles"]:
                independence_groups.setdefault(independence_group, set()).add(role)
                if isinstance(ancestry, list):
                    for ancestor in ancestry:
                        if _nonempty_text(ancestor):
                            independence_groups.setdefault(ancestor, set()).add(role)
        for identity, roles in identities.items():
            if "CONFIRMATION" in roles and roles.intersection(
                {"DISCOVERY", "CANARY", "DEVELOPMENT"}
            ):
                result.error(f"DATA_CONTENT_LEAKAGE:{identity}")
        for group, roles in independence_groups.items():
            if "CONFIRMATION" in roles and roles.intersection(
                {"DISCOVERY", "CANARY", "DEVELOPMENT"}
            ):
                result.error(f"DATA_INDEPENDENCE_LEAKAGE:{group}")

    constraints = contract.get("constraints")
    gates = 0
    if not isinstance(constraints, list) or not constraints:
        result.error("CONSTRAINTS")
    else:
        seen_constraints: set[str] = set()
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                result.error(f"CONSTRAINT_OBJECT:{index}")
                continue
            constraint_id = constraint.get("id")
            constraint_class = constraint.get("class")
            if not _nonempty_text(constraint_id):
                result.error(f"CONSTRAINT_ID:{index}")
            elif constraint_id in seen_constraints:
                result.error(f"CONSTRAINT_DUPLICATE:{constraint_id}")
            else:
                seen_constraints.add(constraint_id)
            if constraint_class not in policy["constraint_classes"]:
                result.error(f"CONSTRAINT_CLASS:{index}")
            if constraint_class == "GATE":
                gates += 1
                for field_name in ("metric", "operator", "threshold", "unit"):
                    value = constraint.get(field_name)
                    if field_name == "threshold":
                        operator = constraint.get("operator")
                        valid_threshold = _finite_number(value) or (
                            operator == "RANGE_INCLUSIVE"
                            and isinstance(value, list)
                            and len(value) == 2
                            and all(_finite_number(item) for item in value)
                        )
                        if not valid_threshold:
                            result.error(
                                f"GATE_FIELD:{constraint_id or index}:{field_name}"
                            )
                    elif not _nonempty_text(value):
                        result.error(f"GATE_FIELD:{constraint_id or index}:{field_name}")
                if constraint.get("operator") not in policy["constraint_operators"]:
                    result.error(f"GATE_OPERATOR:{constraint_id or index}")
            if not _nonempty_text(constraint.get("description")):
                result.error(f"CONSTRAINT_DESCRIPTION:{index}")
            if constraint.get("failure_scope") not in policy["failure_scopes"]:
                result.error(f"FAILURE_SCOPE:{index}")
            numeric_value = constraint.get("threshold", constraint.get("value"))
            has_numeric_value = _finite_number(numeric_value) or (
                isinstance(numeric_value, list)
                and len(numeric_value) == 2
                and all(_finite_number(item) for item in numeric_value)
            )
            if has_numeric_value:
                missing = [
                    name
                    for name in policy["minimum_numeric_constraint_fields"]
                    if not _nonempty_text(constraint.get(name))
                ]
                for name in missing:
                    code = f"NUMERIC_JUSTIFICATION:{constraint_id or index}:{name}"
                    if stage in {"CONFIRMATION", "DEPLOYMENT"} and constraint_class == "GATE":
                        result.error(code)
                    else:
                        result.warn(code)
    if stage in {"CONFIRMATION", "DEPLOYMENT"} and gates == 0:
        result.error("CONFIRMATION_GATE_REQUIRED")

    result_model = _object(contract.get("result_model"))
    if result_model is None:
        result.error("RESULT_MODEL")
    else:
        execution = result_model.get("execution_validity")
        outcome = result_model.get("scientific_outcome")
        terminal_scope = result_model.get("terminal_scope")
        if execution not in policy["execution_validity_values"]:
            result.error("EXECUTION_VALIDITY")
        if execution == "NOT_RUN":
            if outcome != policy["not_run_scientific_outcome"]:
                result.error("NOT_RUN_CANNOT_ASSERT_OUTCOME")
        elif execution == "INVALID":
            if outcome != policy["invalid_execution_scientific_outcome"]:
                result.error("INVALID_EXECUTION_CANNOT_ASSERT_OUTCOME")
        elif outcome not in stage_policy["allowed_claims"]:
            result.error("SCIENTIFIC_OUTCOME")
        if terminal_scope not in policy["failure_scopes"]:
            result.error("TERMINAL_SCOPE")
        invalid_effect = result_model.get("invalid_execution_effect")
        if invalid_effect != policy["default_invalid_execution_effect"]:
            result.error("INVALID_EXECUTION_EFFECT")
        if execution == "INVALID" and terminal_scope in {"RESEARCH_QUESTION", "PRODUCT"}:
            result.error("INVALID_EXECUTION_SCOPE_TOO_BROAD")
        if terminal_scope == "RESEARCH_QUESTION":
            retirement = _object(contract.get("question_retirement"))
            if retirement is None:
                result.error("QUESTION_RETIREMENT_DECISION_REQUIRED")
            else:
                for field_name in ("decision_id", "basis"):
                    if not _nonempty_text(retirement.get(field_name)):
                        result.error(f"QUESTION_RETIREMENT_FIELD:{field_name}")
                _validate_independent_evidence_registry(
                    contract,
                    retirement.get("independent_evidence_ids"),
                    result,
                )
                if not _nonempty_text_list(retirement.get("alternative_explanations")):
                    result.error("QUESTION_RETIREMENT_ALTERNATIVES")
        if terminal_scope == "PRODUCT" and stage != "DEPLOYMENT":
            result.error("PRODUCT_SCOPE_REQUIRES_DEPLOYMENT")

    successor = _object(contract.get("successor_policy"))
    if successor is None:
        result.error("SUCCESSOR_POLICY")
    else:
        if successor.get("preserve_previous_evidence") is not True:
            result.error("PREVIOUS_EVIDENCE_MUST_BE_PRESERVED")
        if not isinstance(successor.get("new_version_allowed"), bool):
            result.error("NEW_VERSION_POLICY")

    _validate_hypotheses(contract, policy, result)
    _validate_failure_learning(contract, policy, result)
    return result


def validate_closure_scope(
    record: dict[str, Any], policy: dict[str, Any]
) -> ValidationResult:
    result = ValidationResult()
    if not _nonempty_text(record.get("record_id")):
        result.error("RECORD_ID")
    if not DATE_RE.fullmatch(str(record.get("created_on", ""))):
        result.error("CREATED_ON")
    question = _object(record.get("scientific_question"))
    protocol = _object(record.get("protocol_version"))
    evidence = _object(record.get("evidence_instance"))
    recovery = _object(record.get("recovery"))
    effects = record.get("closure_effects")
    if question is None:
        result.error("QUESTION_OBJECT")
    if protocol is None:
        result.error("PROTOCOL_VERSION_OBJECT")
    if evidence is None:
        result.error("EVIDENCE_INSTANCE_OBJECT")
    if recovery is None:
        result.error("RECOVERY_OBJECT")
    if not isinstance(effects, list) or not effects:
        result.error("CLOSURE_EFFECTS")

    references: dict[str, str] = {}
    if question is not None:
        if not _nonempty_text(question.get("id")):
            result.error("QUESTION_ID")
        else:
            references["scientific_question"] = question["id"]
        if question.get("state") not in policy["scientific_question_states"]:
            result.error("QUESTION_STATE")
    if protocol is not None:
        if not _nonempty_text(protocol.get("id")):
            result.error("PROTOCOL_VERSION_ID")
        else:
            references["protocol_version"] = protocol["id"]
        if protocol.get("state") not in policy["protocol_version_states"]:
            result.error("PROTOCOL_VERSION_STATE")
    if evidence is not None:
        if not _nonempty_text(evidence.get("id")):
            result.error("EVIDENCE_INSTANCE_ID")
        else:
            references["evidence_instance"] = evidence["id"]
        artifact_integrity = evidence.get("artifact_integrity")
        if artifact_integrity not in policy["artifact_integrity_states"]:
            result.error("ARTIFACT_INTEGRITY")
        if evidence.get("execution_state") not in policy["closure_execution_states"]:
            result.error("CLOSURE_EXECUTION_STATE")
        if not isinstance(evidence.get("scientific_inference_allowed"), bool):
            result.error("SCIENTIFIC_INFERENCE_ALLOWED")
        if artifact_integrity == "INVALID":
            if evidence.get("scientific_inference_allowed") is not False:
                result.error("INVALID_EVIDENCE_SCIENTIFIC_INFERENCE")
        if not _nonempty_text(evidence.get("failure_class")):
            result.error("EVIDENCE_FAILURE_CLASS")
        for field_name in policy["required_evidence_hash_fields"]:
            value = evidence.get(field_name)
            if not isinstance(value, str) or not _sha256_text(value):
                result.error(f"EVIDENCE_SHA256:{field_name}")

    authority = record.get("authority_ceiling")
    if authority not in policy["authority_ceilings"]:
        result.error("AUTHORITY_CEILING")

    dependency_effect_ids: set[str] = set()
    if isinstance(effects, list):
        for index, effect in enumerate(effects):
            if not isinstance(effect, dict):
                result.error(f"CLOSURE_EFFECT_OBJECT:{index}")
                continue
            if not all(
                _nonempty_text(effect.get(name))
                for name in ("target_type", "target_id", "basis")
            ):
                result.error(f"CLOSURE_EFFECT_FIELDS:{index}")
                continue
            target_type = effect.get("target_type")
            target_id = effect.get("target_id")
            if target_type not in policy["closure_target_types"]:
                result.error(f"CLOSURE_TARGET_TYPE:{index}")
                continue
            if target_type in references and target_id != references[target_type]:
                result.error(f"CLOSURE_TARGET_REFERENCE:{index}")
            if target_type == "dependency_branch":
                dependency_effect_ids.add(target_id)
            basis = str(effect.get("basis")).strip().lower()
            if target_type == "scientific_question":
                if basis not in policy["question_closure_bases"]:
                    result.error("ILLEGAL_QUESTION_CLOSURE_BASIS")
                if question is not None and question.get("state") not in {
                    "CLOSED",
                    "RETIRED",
                }:
                    result.error("QUESTION_CLOSURE_STATE_CONTRADICTION")

    edges = record.get("dependency_edges")
    edge_pairs: set[tuple[str, str]] = set()
    if dependency_effect_ids:
        if not isinstance(edges, list) or not edges:
            result.error("DEPENDENCY_EDGES_REQUIRED")
        else:
            for index, edge in enumerate(edges):
                if not isinstance(edge, dict) or not all(
                    _nonempty_text(edge.get(name))
                    for name in ("from_id", "depends_on_id")
                ):
                    result.error(f"DEPENDENCY_EDGE_FIELDS:{index}")
                    continue
                pair = (edge["from_id"], edge["depends_on_id"])
                if pair[0] == pair[1]:
                    result.error(f"DEPENDENCY_SELF_CYCLE:{index}")
                edge_pairs.add(pair)
            protocol_id = references.get("protocol_version")
            for target_id in dependency_effect_ids:
                if not any(
                    from_id == target_id
                    and (depends_on_id == protocol_id or depends_on_id in references.values())
                    for from_id, depends_on_id in edge_pairs
                ):
                    result.error(f"DEPENDENCY_EFFECT_UNPROVEN:{target_id}")
            adjacency: dict[str, set[str]] = {}
            for from_id, depends_on_id in edge_pairs:
                adjacency.setdefault(from_id, set()).add(depends_on_id)
            visiting: set[str] = set()
            visited: set[str] = set()

            def has_cycle(node: str) -> bool:
                if node in visiting:
                    return True
                if node in visited:
                    return False
                visiting.add(node)
                if any(has_cycle(parent) for parent in adjacency.get(node, set())):
                    return True
                visiting.remove(node)
                visited.add(node)
                return False

            if any(has_cycle(node) for node in list(adjacency)):
                result.error("DEPENDENCY_GRAPH_CYCLE")

    invalid_closure = (
        evidence is not None
        and evidence.get("artifact_integrity") == "INVALID"
        and protocol is not None
        and protocol.get("state") == "CLOSED_INVALID"
    )
    if recovery is not None:
        same_version = recovery.get("same_version_rerun_allowed")
        new_version = recovery.get("new_version_may_be_proposed")
        if not isinstance(same_version, bool):
            result.error("SAME_VERSION_RERUN_POLICY")
        if not isinstance(new_version, bool):
            result.error("NEW_VERSION_PROPOSAL_POLICY")
        if invalid_closure and same_version is True:
            if recovery.get("claim_unconsumed") is not True:
                result.error("SAME_VERSION_RERUN_REQUIRES_UNCONSUMED_CLAIM")
            if not _nonempty_text(recovery.get("same_version_rerun_justification")):
                result.error("SAME_VERSION_RERUN_JUSTIFICATION")
        if (
            evidence is not None
            and evidence.get("execution_state") == "CONSUMED_CLOSED"
        ):
            if same_version is not False:
                result.error("CONSUMED_CLAIM_CANNOT_RERUN_SAME_VERSION")
            if recovery.get("claim_unconsumed") is True:
                result.error("CONSUMED_CLAIM_CANNOT_BE_UNCONSUMED")

    if question is not None and question.get("state") in {"CLOSED", "RETIRED"}:
        retirement = _object(record.get("independent_retirement_decision"))
        if retirement is None:
            result.error("INDEPENDENT_RETIREMENT_DECISION_REQUIRED")
        else:
            if not _nonempty_text(retirement.get("decision_id")):
                result.error("INDEPENDENT_RETIREMENT_DECISION_ID")
            if not _nonempty_text(retirement.get("basis")):
                result.error("INDEPENDENT_RETIREMENT_BASIS")
            _validate_independent_evidence_registry(
                record,
                retirement.get("independent_evidence_ids"),
                result,
            )

    learning = _object(record.get("failure_learning"))
    if learning is None:
        result.error("FAILURE_LEARNING_REQUIRED")
    else:
        for field_name in policy["required_failure_learning_fields"]:
            value = learning.get(field_name)
            if field_name in {
                "alternative_explanations",
                "constraint_challenges",
                "next_hypotheses",
                "reuse_candidates",
            }:
                if not isinstance(value, list):
                    result.error(f"FAILURE_LEARNING_LIST:{field_name}")
            elif not _nonempty_text(value):
                result.error(f"FAILURE_LEARNING_FIELD:{field_name}")
    _validate_round_summary(record, policy, result)
    return result


def validate_document(document: dict[str, Any], policy: dict[str, Any]) -> ValidationResult:
    policy_result = validate_policy(policy)
    if policy_result.errors:
        return policy_result
    binding = ValidationResult()
    if document.get("governance_policy_id") != policy.get("policy_id"):
        binding.error("GOVERNANCE_POLICY_ID")
    if document.get("governance_policy_sha256") != _policy_digest(policy):
        binding.error("GOVERNANCE_POLICY_SHA256")
    schema = document.get("schema_version")
    if schema == PROTOCOL_SCHEMA:
        result = validate_protocol(document, policy)
    elif schema == CLOSURE_SCHEMA:
        result = validate_closure_scope(document, policy)
    else:
        result = ValidationResult()
        result.error("SCHEMA_VERSION")
    _merge(result, binding)
    return result


def canonical_policy_path(governance_policy_id: Any) -> Path:
    if governance_policy_id == "PROGRESSIVE_RESEARCH_GOVERNANCE_R1":
        return LEGACY_POLICY
    if governance_policy_id == "DATA_CAPABILITY_DRIVEN_RESEARCH_GOVERNANCE_R2":
        return R2_POLICY
    if governance_policy_id == "RISK_TIERED_RESEARCH_GOVERNANCE_R3":
        return V3_POLICY
    return DEFAULT_POLICY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    try:
        contract = _load_object(args.contract.resolve())
        if args.policy is not None:
            policy_path = args.policy.resolve()
        else:
            policy_path = canonical_policy_path(contract.get("governance_policy_id"))
        policy = _load_object(policy_path.resolve())
        result = validate_document(contract, policy)
        payload = result.payload(str(args.contract))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        payload = {
            "status": "INVALID",
            "subject": str(args.contract),
            "errors": [f"LOAD:{type(error).__name__}:{error}"],
            "warnings": [],
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
