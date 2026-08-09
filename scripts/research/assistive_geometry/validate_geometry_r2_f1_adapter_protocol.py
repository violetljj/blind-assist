#!/usr/bin/env python3
"""Validate the non-execution R2 F1 FactorTensorAdapter protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT_IMPORT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_IMPORT))

from scripts.research.assistive_geometry.audit_geometry_r2_f1_adapter_gap import (
    READY_FOR_CANARY_TERMINAL,
    REDUCER_REQUIRED_FIELDS,
    REQUIRED_ADAPTER_OPERATIONS,
    audit_contract,
)


PROTOCOL_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK"
FIXTURE_SCHEMA = "blindassist_assistive_geometry_r2_f1_factortensor_adapter_fixture_v1"
SUCCESSOR_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_IMPLEMENTATION_AND_SYNTHETIC_CANARY"
SEMANTIC_CORE_SHA256 = "26730AE2ADD63545C121AF923B6B73CFEBF1B740F033BB280F5308EDAD08C244"
FIXTURE_CANONICAL_SHA256 = "B1466E20612D292DE9F5858434664F51C0A5BF9E7A32A1B0A6B7E410E3FA543E"

SEMANTIC_FIELDS = (
    "adapter_schema",
    "frame_contract",
    "numeric_contract",
    "prediction_field_consumers",
    "reducer_field_producers",
    "operations",
    "mutation_canary_gates",
    "authority",
    "execution_authority",
    "unique_successor",
    "claims_allowed",
    "claims_forbidden",
    "constraints",
    "experiment_design",
    "hypotheses",
    "successor_policy",
    "result_model",
    "claim_ceiling",
)

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

EXPECTED_REDUCER_FIELDS = {
    "schema",
    "frame_id",
    "factor_identity",
    *(
        f"{group}.{field}"
        for group, fields in REDUCER_REQUIRED_FIELDS.items()
        for field in fields
    ),
}

EXPECTED_CASE_IDS = (
    "nominal_landscape_single_component",
    "portrait_equivalent_single_component",
    "local_component_depth_missing",
    "high_depth_uncertainty_monotone",
    "support_invalid_fail_closed",
    "geometry_receipt_identity_mismatch",
    "two_components_canonical_left_then_right",
    "bridge_pixel_merges_components",
)

EXPECTED_BINDINGS = {
    "FROZEN_F1_FACTOR_SCHEMA": "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_SCHEMA_2026-08-10.json",
    "BYTE_FROZEN_F0_REDUCER": "scripts/research/assistive_geometry/geometry_r2_reducer.py",
    "ADAPTER_GAP_AUDIT": "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_GAP_AUDIT_2026-08-10.json",
    "SYNTHETIC_FIXTURE": "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_FIXTURE_2026-08-10.json",
    "STATIC_VALIDATOR": "scripts/research/assistive_geometry/validate_geometry_r2_f1_adapter_protocol.py",
    "MUTATION_TESTS": "scripts/research/assistive_geometry/test_validate_geometry_r2_f1_adapter_protocol.py",
}

PROHIBITED_PATHS = (
    "scripts/research/assistive_geometry/factor_tensor_adapter.py",
    "scripts/research/assistive_geometry/run_factor_tensor_adapter_canary.py",
    "artifacts.local/evidence/assistive-geometry/r2-f1-factortensor-adapter-r0",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def semantic_core(protocol: dict[str, Any]) -> dict[str, Any]:
    return {field: protocol.get(field) for field in SEMANTIC_FIELDS}


def _matrix_shape(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        return None
    widths = {len(row) for row in value}
    if len(widths) != 1 or 0 in widths:
        return None
    return len(value), widths.pop()


def _scan_forbidden(value: Any, forbidden: set[str], path: str = "input") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in forbidden:
                errors.append(f"FORBIDDEN_INPUT_KEY:{path}.{key}")
            errors.extend(_scan_forbidden(child, forbidden, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_scan_forbidden(child, forbidden, f"{path}[{index}]"))
    return errors


def validate_bindings(protocol: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    bindings = protocol.get("bindings")
    if not isinstance(bindings, list):
        return ["BINDINGS_NOT_LIST"]
    roles = [item.get("role") for item in bindings if isinstance(item, dict)]
    if len(roles) != len(set(roles)):
        errors.append("BINDING_ROLE_DUPLICATE")
    if set(roles) != set(EXPECTED_BINDINGS):
        errors.append("BINDING_ROLE_SET")
    for binding in bindings:
        if not isinstance(binding, dict):
            errors.append("BINDING_NOT_OBJECT")
            continue
        role = binding.get("role")
        path_text = binding.get("path")
        if role not in EXPECTED_BINDINGS or path_text != EXPECTED_BINDINGS.get(role):
            errors.append(f"BINDING_PATH:{role}")
            continue
        path = repo_root / str(path_text)
        if not path.is_file():
            errors.append(f"BINDING_ABSENT:{role}")
        elif binding.get("sha256") != sha256_file(path):
            errors.append(f"BINDING_SHA:{role}")
    return errors


def validate_fixture(fixture: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if canonical_sha256(fixture) != FIXTURE_CANONICAL_SHA256:
        errors.append("FIXTURE_CANONICAL_DRIFT")
    if fixture.get("schema") != FIXTURE_SCHEMA:
        errors.append("FIXTURE_SCHEMA")
    if fixture.get("patch_semantics") != "RECURSIVE_OBJECT_MERGE_ARRAYS_REPLACE":
        errors.append("FIXTURE_PATCH_SEMANTICS")
    numeric = fixture.get("numeric_contract", {})
    if numeric.get("probability_threshold") != 0.5 or numeric.get("component_connectivity") != 8:
        errors.append("FIXTURE_NUMERIC_CONTRACT")
    base = fixture.get("base_input")
    if not isinstance(base, dict):
        return errors + ["FIXTURE_BASE_INPUT"]
    prediction = base.get("prediction", {})
    geometry = base.get("geometry_receipt", {})
    if prediction.get("camera_geometry_receipt_sha256") != geometry.get("content_sha256"):
        errors.append("FIXTURE_BASE_RECEIPT_IDENTITY")
    tensor_hw = tuple(geometry.get("tensor_hw", []))
    matrix_paths = (
        prediction.get("depth_scale", {}).get("depth_shape_positive_hw"),
        prediction.get("depth_scale", {}).get("depth_log_sigma_hw"),
        prediction.get("depth_scale", {}).get("depth_valid_probability_hw"),
        prediction.get("support_surface", {}).get("support_probability_hw"),
        prediction.get("obstacle_boundary_evidence", {}).get("obstacle_evidence_probability_hw"),
        prediction.get("obstacle_boundary_evidence", {}).get("boundary_probability_hw"),
        prediction.get("obstacle_boundary_evidence", {}).get("boundary_localization_sigma_px_hw"),
        prediction.get("obstacle_boundary_evidence", {}).get("evidence_valid_hw"),
    )
    if any(_matrix_shape(value) != tensor_hw for value in matrix_paths):
        errors.append("FIXTURE_BASE_TENSOR_SHAPE")
    forbidden = {
        "clearance",
        "clearance_m",
        "direct_clearance",
        "occupancy",
        "occupancy_logit",
        "free",
        "blocked",
        "risk",
        "risk_score",
        "task_confidence",
        "final_state",
        "unknown_logit",
        "ttc",
        "future_clearance",
    }
    errors.extend(_scan_forbidden(base, forbidden))
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        return errors + ["FIXTURE_CASES_NOT_LIST"]
    ids = tuple(item.get("id") for item in cases if isinstance(item, dict))
    if ids != EXPECTED_CASE_IDS:
        errors.append("FIXTURE_CASE_IDS")
    if len(ids) != len(set(ids)):
        errors.append("FIXTURE_CASE_ID_DUPLICATE")
    case_map = {item.get("id"): item for item in cases if isinstance(item, dict)}
    if case_map.get("local_component_depth_missing", {}).get("expected", {}).get("forbidden_state") != "OCCUPIED_OBSERVED":
        errors.append("FIXTURE_LOCAL_MISSING_FAIL_CLOSED")
    if case_map.get("support_invalid_fail_closed", {}).get("expected", {}).get("all_unknown") is not True:
        errors.append("FIXTURE_SUPPORT_FAIL_CLOSED")
    if case_map.get("geometry_receipt_identity_mismatch", {}).get("expected", {}).get("all_unknown") is not True:
        errors.append("FIXTURE_RECEIPT_FAIL_CLOSED")
    if case_map.get("high_depth_uncertainty_monotone", {}).get("expected", {}).get("occupancy_strength_relation_to_nominal") != "NOT_STRONGER":
        errors.append("FIXTURE_UNCERTAINTY_MONOTONICITY")
    parity = [item for item in cases if isinstance(item, dict) and item.get("expected", {}).get("parity_group") == "central_obstacle"]
    if len(parity) != 2:
        errors.append("FIXTURE_ORIENTATION_PARITY_PAIR")
    for item in cases:
        if isinstance(item, dict):
            errors.extend(_scan_forbidden(item.get("patch", {}), forbidden, f"case.{item.get('id')}.patch"))
    return errors


def validate_protocol(protocol: dict[str, Any], factor_schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("protocol_id") != PROTOCOL_ID:
        errors.append("PROTOCOL_ID")
    if protocol.get("profile") != "CANARY_LITE" or protocol.get("stage") != "CANARY":
        errors.append("PROFILE_OR_STAGE")
    if canonical_sha256(semantic_core(protocol)) != SEMANTIC_CORE_SHA256:
        errors.append("SEMANTIC_CORE_DRIFT")
    operations = protocol.get("operations")
    operation_ids = [item.get("id") for item in operations if isinstance(item, dict)] if isinstance(operations, list) else []
    if set(operation_ids) != REQUIRED_ADAPTER_OPERATIONS or len(operation_ids) != len(REQUIRED_ADAPTER_OPERATIONS):
        errors.append("OPERATION_SET")
    for item in operations if isinstance(operations, list) else []:
        if not isinstance(item, dict) or not all(item.get(field) for field in ("id", "consumes", "emits", "rule", "failure")):
            errors.append("OPERATION_FIELDS")
            break
    prediction_fields = {
        item.get("name")
        for item in factor_schema.get("prediction_fields", [])
        if isinstance(item, dict)
    }
    if prediction_fields != EXPECTED_PREDICTION_FIELDS:
        errors.append("F1_PREDICTION_SCHEMA_DRIFT")
    consumers = protocol.get("prediction_field_consumers")
    if not isinstance(consumers, dict) or set(consumers) != EXPECTED_PREDICTION_FIELDS or any(not value for value in consumers.values()):
        errors.append("PREDICTION_FIELD_COVERAGE")
    producers = protocol.get("reducer_field_producers")
    if not isinstance(producers, dict) or set(producers) != EXPECTED_REDUCER_FIELDS or any(value not in REQUIRED_ADAPTER_OPERATIONS for value in producers.values()):
        errors.append("REDUCER_FIELD_COVERAGE")
    audit = audit_contract(factor_schema, protocol)
    if audit.get("terminal") != READY_FOR_CANARY_TERMINAL or audit.get("adapter_static_contract_complete") is not True:
        errors.append("GAP_AUDIT_NOT_CLOSED")
    authority = protocol.get("authority", {})
    expected_authority = {
        "outside_learned_graph": True,
        "deterministic": True,
        "trainable_parameters": 0,
        "final_task_shortcut_allowed": False,
        "execution_authority": False,
    }
    if authority != expected_authority:
        errors.append("ADAPTER_AUTHORITY")
    execution = protocol.get("execution_authority", {})
    if not isinstance(execution, dict) or execution.get("protocol_lock") is not True or execution.get("static_validation") is not True or any(value is not False for key, value in execution.items() if key not in {"protocol_lock", "static_validation"}):
        errors.append("EXECUTION_AUTHORITY")
    gates = protocol.get("mutation_canary_gates")
    gate_ids = [item.get("id") for item in gates if isinstance(item, dict)] if isinstance(gates, list) else []
    if gate_ids != [f"A{index:02d}_{suffix}" for index, suffix in enumerate(("FIELD_COVERAGE", "DETERMINISTIC_REPLAY", "SCALE_UNCERTAINTY_MONOTONE", "SUPPORT_FAIL_CLOSED", "COMPONENT_SEMANTICS", "ORIENTATION_EQUIVARIANCE", "LOCAL_MISSING_DEPTH", "NO_UNCERTAINTY_STRENGTHENING", "GRAPH_AND_SHORTCUT_FIREWALL", "F0_BYTE_IDENTITY"), 1)]:
        errors.append("GATE_SET")
    successor = protocol.get("unique_successor", {})
    policy = protocol.get("successor_policy", {})
    if successor.get("id") != SUCCESSOR_ID or successor.get("execution_authority") is not False:
        errors.append("UNIQUE_SUCCESSOR")
    if policy.get("unique_successor") != SUCCESSOR_ID or policy.get("successor_currently_authorized") is not False:
        errors.append("SUCCESSOR_POLICY")
    if protocol.get("result_model", {}).get("not_run") != "R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_FROZEN_CANARY_NOT_RUN":
        errors.append("RESULT_MODEL")
    return errors


def validate_absence(repo_root: Path) -> list[str]:
    return [f"PROHIBITED_ARTIFACT_PRESENT:{path}" for path in PROHIBITED_PATHS if (repo_root / path).exists()]


def validate_all(
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    factor_schema: dict[str, Any],
    repo_root: Path,
    *,
    check_bindings: bool = True,
    check_absence: bool = True,
) -> list[str]:
    errors = validate_protocol(protocol, factor_schema) + validate_fixture(fixture)
    if check_bindings:
        errors.extend(validate_bindings(protocol, repo_root))
    if check_absence:
        errors.extend(validate_absence(repo_root))
    return sorted(set(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--factor-schema", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol = load_json(args.contract.resolve())
    fixture = load_json(args.fixture.resolve())
    factor_schema = load_json(args.factor_schema.resolve())
    errors = validate_all(protocol, fixture, factor_schema, repo_root)
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_adapter_protocol_validation_v1",
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "operation_count": len(protocol.get("operations", [])),
        "fixture_case_count": len(fixture.get("cases", [])),
        "scientific_status": "NOT_RUN",
        "adapter_implementation_authority": False,
        "f1_execution_authority": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
