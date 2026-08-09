#!/usr/bin/env python3
"""Static validator for the Assistive Geometry R2 F1-P protocol lock.

This validator has no model, data materialization, training, checkpoint, or
optimizer behavior. It verifies only the frozen interface and governance
contract needed before a separate F1 execution lock could ever be proposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_TRAIN_ONLY_FACTOR_LEARNABILITY_PROTOCOL_LOCK"
FACTOR_SCHEMA_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_SCHEMA_2026-08-10"
F0_TERMINAL = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PASS"
DCA_TERMINAL = "AG_DCA_R0_COMPLETE_THREE_HYPOTHESES_NOT_SUPPORTED"
SUCCESSOR_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK"

EXPECTED_PREDICTION_FIELDS = {
    "depth_shape_positive_hw",
    "log_metric_scale_m_scalar",
    "depth_log_sigma_hw",
    "depth_valid_probability_hw",
    "metric_scale_valid",
    "support_probability_hw",
    "support_plane_normal_camera_xyz",
    "camera_height_m",
    "support_residual_sigma_m",
    "support_valid",
    "obstacle_evidence_probability_hw",
    "boundary_probability_hw",
    "boundary_localization_sigma_px_hw",
    "evidence_valid_hw",
}
EXPECTED_LOSSES = {
    "depth_shape_log_charbonnier",
    "metric_scale_log_huber",
    "depth_heteroscedastic_nll",
    "depth_validity_brier",
    "support_probability_brier",
    "support_plane_angular",
    "camera_height_log_huber",
    "support_residual_heteroscedastic_nll",
    "support_validity_brier",
    "obstacle_evidence_brier",
    "boundary_probability_brier",
    "boundary_localization_heteroscedastic_nll",
    "evidence_validity_brier",
}
EXPECTED_ROLE_MINIMUMS = {"FIT": 8, "CHECKPOINT_SELECTION": 2, "TRAIN_CANARY": 2}
REQUIRED_FACTOR_FAMILIES = {"depth_scale", "support_surface", "obstacle_boundary_evidence"}
EXPECTED_CAPABILITY_BINDINGS = {
    "metric_depth_source": "oracle_depth_factor",
    "support_source": "oracle_support_factor",
    "crisp_obstacle_source": "oracle_obstacle_factor",
    "depth_uncertainty_direct_truth": "r2_depth_uncertainty_truth_materialized",
    "support_uncertainty_direct_truth": "r2_support_uncertainty_truth_materialized",
    "continuous_boundary_truth": "r2_obstacle_boundary_truth_materialized",
    "complete_factor_schema_truth": "r2_complete_factor_schema_truth",
}
REQUIRED_KILL_GATES = {
    "F1_K01_SCHEMA_OR_SHORTCUT_INVALID",
    "F1_K02_SUPERVISION_FRONTDOOR_UNSATISFIED",
    "F1_K03_DEPTH_SCALE_NOT_LEARNABLE",
    "F1_K04_SUPPORT_SURFACE_NOT_LEARNABLE",
    "F1_K05_BOUNDARY_EVIDENCE_NOT_LEARNABLE",
    "F1_K06_UNCERTAINTY_NOT_CALIBRATED",
    "F1_K07_CHECKPOINT_SELECTION_INVALID",
    "F1_K08_ANY_DOWNSTREAM_RESCUE_ATTEMPT",
}
PROHIBITED_RUNTIME_PATHS = (
    "scripts/research/assistive_geometry/train_geometry_r2_f1.py",
    "scripts/research/assistive_geometry/geometry_r2_factor_model.py",
    "scripts/research/assistive_geometry/materialize_geometry_r2_f1_labels.py",
    "artifacts.local/models/assistive-geometry-r2-f1",
    "artifacts.local/work/assistive-geometry-r2-f1",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def error_if(errors: list[str], condition: bool, code: str) -> None:
    if condition:
        errors.append(code)


def validate_static_contract(
    protocol: dict[str, Any],
    factor_schema: dict[str, Any],
    dca_result: dict[str, Any],
    f0_result: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    error_if(errors, protocol.get("protocol_id") != PROTOCOL_ID, "PROTOCOL_ID")
    status = protocol.get("status")
    error_if(errors, not isinstance(status, list) or "F1_EXECUTION_NOT_AUTHORIZED" not in status, "STATUS_AUTHORITY")
    error_if(errors, factor_schema.get("schema_id") != FACTOR_SCHEMA_ID, "FACTOR_SCHEMA_ID")
    error_if(errors, f0_result.get("terminal") != F0_TERMINAL or f0_result.get("passed") is not True, "F0_PREDECESSOR")
    error_if(errors, dca_result.get("terminal") != DCA_TERMINAL, "DCA_PREDECESSOR")

    prediction_fields = factor_schema.get("prediction_fields")
    names = {
        item.get("name")
        for item in prediction_fields
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    } if isinstance(prediction_fields, list) else set()
    error_if(errors, names != EXPECTED_PREDICTION_FIELDS, "PREDICTION_FIELD_SET")
    groups = {
        item.get("group")
        for item in prediction_fields
        if isinstance(item, dict)
    } if isinstance(prediction_fields, list) else set()
    error_if(errors, groups != REQUIRED_FACTOR_FAMILIES, "PREDICTION_FACTOR_GROUPS")
    forbidden = set(factor_schema.get("forbidden_prediction_or_supervision_fields", []))
    error_if(errors, not {"clearance", "occupancy", "blocked", "final_state", "task_confidence"} <= forbidden, "FORBIDDEN_TASK_FIELDS")
    error_if(errors, bool(names & forbidden), "TASK_SHORTCUT_IN_FACTOR_FIELDS")
    container = factor_schema.get("prediction_container", {})
    error_if(errors, container.get("additional_factor_groups_allowed") is not False, "ADDITIONAL_FACTOR_GROUPS")

    unknown = factor_schema.get("unknown_and_mask_contract", {})
    error_if(errors, unknown.get("unknown_is_negative") is not False, "UNKNOWN_AS_NEGATIVE")
    error_if(errors, unknown.get("invalid_value_may_supervise_only_its_validity_head") is not True, "INVALID_VALUE_SCOPE")
    error_if(errors, unknown.get("entire_factor_family_without_valid_support") != "F1_KILL_NOT_SILENT_SKIP", "UNSUPERVISED_FACTOR_POLICY")
    uncertainty = factor_schema.get("uncertainty_supervision", {})
    error_if(errors, uncertainty.get("allowed_mechanism") != "HETEROSCEDASTIC_PROPER_SCORE_AGAINST_VALID_FACTOR_RESIDUALS", "UNCERTAINTY_MECHANISM")
    error_if(errors, uncertainty.get("zero_or_constant_sigma_pseudo_truth_forbidden") is not True, "SIGMA_PSEUDO_TRUTH")

    authority = protocol.get("execution_authority", {})
    error_if(errors, authority.get("protocol_lock") is not True or authority.get("static_validation") is not True, "PROTOCOL_LOCK_AUTHORITY")
    forbidden_authorities = {
        "data_materialization",
        "label_materialization",
        "model_definition",
        "trainer_creation",
        "optimizer_step",
        "checkpoint_creation",
        "f1_execution",
        "f2_development",
        "teacher_distillation",
        "temporal",
        "mobile_device",
        "calibration",
        "confirmation",
    }
    error_if(errors, any(authority.get(name) is not False for name in forbidden_authorities), "EXECUTION_AUTHORITY_EXCEEDED")

    availability = protocol.get("factor_supervision_availability_matrix")
    summary = dca_result.get("capability_summary", {})
    if not isinstance(availability, list):
        errors.append("AVAILABILITY_MATRIX")
    else:
        by_id = {item.get("id"): item for item in availability if isinstance(item, dict)}
        error_if(errors, set(by_id) != set(EXPECTED_CAPABILITY_BINDINGS), "AVAILABILITY_MATRIX_IDS")
        for matrix_id, capability_id in EXPECTED_CAPABILITY_BINDINGS.items():
            item = by_id.get(matrix_id, {})
            observed = summary.get(capability_id, {})
            for field in ("frames", "parents", "portrait", "landscape"):
                error_if(errors, item.get(field) != observed.get(field), f"AVAILABILITY_DRIFT:{matrix_id}:{field}")
            error_if(errors, item.get("dca_capability_id") != capability_id, f"AVAILABILITY_BINDING:{matrix_id}")
    blockers = protocol.get("current_execution_blockers")
    error_if(errors, not isinstance(blockers, list) or len(blockers) < 4, "CURRENT_BLOCKERS")
    error_if(errors, protocol.get("current_f1_execution_admission") != "FAIL_NOT_AUTHORIZED", "CURRENT_ADMISSION")

    loss_contract = protocol.get("loss_contract", {})
    components = loss_contract.get("components")
    loss_names = {
        item.get("id") for item in components if isinstance(item, dict)
    } if isinstance(components, list) else set()
    error_if(errors, loss_names != EXPECTED_LOSSES, "LOSS_COMPONENT_SET")
    loss_families = {
        item.get("factor_family") for item in components if isinstance(item, dict)
    } if isinstance(components, list) else set()
    error_if(errors, loss_families != REQUIRED_FACTOR_FAMILIES, "LOSS_FACTOR_COVERAGE")
    if isinstance(components, list):
        for item in components:
            if not isinstance(item, dict):
                errors.append("LOSS_COMPONENT_OBJECT")
                continue
            loss_id = item.get("id", "UNKNOWN")
            error_if(errors, not item.get("validity_field"), f"LOSS_VALIDITY:{loss_id}")
            error_if(errors, item.get("reported_independently") is not True, f"LOSS_REPORTING:{loss_id}")
            error_if(errors, item.get("optimizer_weight_policy") != "EQUAL_AFTER_FIT_ONLY_BASELINE_NORMALIZATION", f"LOSS_WEIGHT:{loss_id}")
            targets = set(item.get("supervision_fields", []))
            error_if(errors, bool(targets & forbidden), f"LOSS_TASK_TARGET:{loss_id}")
    error_if(errors, loss_contract.get("aggregate_loss_is_checkpoint_metric") is not False, "AGGREGATE_CHECKPOINT_METRIC")
    error_if(errors, loss_contract.get("downstream_reducer_loss_allowed") is not False, "DOWNSTREAM_REDUCER_LOSS")

    split = protocol.get("train_internal_role_contract", {})
    role_minimums = split.get("minimum_parent_counts")
    error_if(errors, role_minimums != EXPECTED_ROLE_MINIMUMS, "TRAIN_ROLE_MINIMUMS")
    error_if(errors, split.get("parent_disjoint") is not True, "TRAIN_ROLE_DISJOINTNESS")
    error_if(errors, split.get("assignment_before_payload_or_outcome") is not True, "TRAIN_ROLE_PREASSIGNMENT")
    error_if(errors, split.get("development_calibration_confirmation_forbidden") is not True, "PROTECTED_ROLE_FIREWALL")

    checkpoint = protocol.get("checkpoint_selection_contract", {})
    error_if(errors, checkpoint.get("uses_only_role") != "CHECKPOINT_SELECTION", "CHECKPOINT_ROLE")
    error_if(errors, checkpoint.get("aggregate_training_loss_forbidden") is not True, "CHECKPOINT_AGGREGATE_LOSS")
    error_if(errors, checkpoint.get("reducer_or_task_metric_forbidden") is not True, "CHECKPOINT_TASK_METRIC")
    error_if(errors, checkpoint.get("tie_break") != "EARLIEST_STEP_THEN_LEXICOGRAPHIC_SHA256", "CHECKPOINT_TIE_BREAK")
    error_if(errors, checkpoint.get("candidate_schedule_frozen") is not False, "CHECKPOINT_SCHEDULE_MUST_REMAIN_UNRESOLVED")

    gates = protocol.get("f1_kill_gates")
    gate_ids = {item.get("id") for item in gates if isinstance(item, dict)} if isinstance(gates, list) else set()
    error_if(errors, gate_ids != REQUIRED_KILL_GATES, "KILL_GATE_SET")
    error_if(errors, protocol.get("f1_success_rule", {}).get("all_factor_families_conjunctive") is not True, "FACTOR_SUCCESS_CONJUNCTION")
    error_if(errors, protocol.get("f1_success_rule", {}).get("reducer_task_metric_may_rescue") is not False, "DOWNSTREAM_RESCUE")
    error_if(errors, protocol.get("f2_admission", {}).get("execution_authority") is not False, "F2_AUTHORITY")

    successor = protocol.get("unique_successor", {})
    error_if(errors, successor.get("id") != SUCCESSOR_ID, "SUCCESSOR_ID")
    error_if(errors, successor.get("execution_authority") is not False, "SUCCESSOR_AUTHORITY")
    return errors


def validate_bindings(protocol: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    bindings = protocol.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        return ["BINDINGS"]
    for item in bindings:
        if not isinstance(item, dict):
            errors.append("BINDING_OBJECT")
            continue
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append("BINDING_FIELDS")
            continue
        path = repo_root / relative
        if not path.is_file():
            errors.append(f"BINDING_ABSENT:{relative}")
        elif sha256_file(path) != expected.upper():
            errors.append(f"BINDING_SHA:{relative}")
    return errors


def validate_runtime_absence(repo_root: Path) -> list[str]:
    return [f"PROHIBITED_RUNTIME_PRESENT:{relative}" for relative in PROHIBITED_RUNTIME_PATHS if (repo_root / relative).exists()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--dca-result", type=Path, required=True)
    parser.add_argument("--f0-result", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol = load_json(args.protocol.resolve())
    schema = load_json(args.schema.resolve())
    dca = load_json(args.dca_result.resolve())
    f0 = load_json(args.f0_result.resolve())
    errors = validate_static_contract(protocol, schema, dca, f0)
    errors.extend(validate_bindings(protocol, repo_root))
    errors.extend(validate_runtime_absence(repo_root))
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_protocol_static_validation_v1",
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "f1_execution_authority": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
