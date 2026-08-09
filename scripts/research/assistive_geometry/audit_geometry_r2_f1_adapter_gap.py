#!/usr/bin/env python3
"""Audit the frozen F1 factor schema against the byte-frozen F0 reducer input.

This is a static interface audit. It does not implement a tensor adapter,
execute the reducer, materialize labels, define a model, or authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "blindassist_assistive_geometry_r2_f1_adapter_gap_audit_v1"
BLOCKED_TERMINAL = "R2_F1_EXECUTION_BLOCKED_FACTORTENSOR_ADAPTER_ABSENT"
READY_FOR_CANARY_TERMINAL = "R2_F1_ADAPTER_STATIC_CONTRACT_COMPLETE_CANARY_NOT_RUN"
SUCCESSOR_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_SCHEMA_AND_MUTATION_CANARY_LOCK"

REQUIRED_ADAPTER_OPERATIONS = {
    "input_geometry_receipt",
    "metric_depth_from_shape_and_scale",
    "scale_m_from_log_metric_scale",
    "scale_sigma_m_from_calibrated_scale_distribution",
    "support_normal_camera_mapping",
    "support_normal_sigma_rad",
    "support_camera_height_mapping",
    "support_height_sigma_m",
    "support_residual_sigma_m",
    "dense_evidence_to_obstacle_components",
    "obstacle_forward_depth_interval",
    "obstacle_lateral_metric_interval",
    "boundary_sigma_px_to_metric",
    "evidence_probability_aggregation",
    "evidence_sigma_calibration",
    "local_missing_depth_to_depth_valid",
    "canonical_component_order",
}

REDUCER_REQUIRED_FIELDS = {
    "input_geometry": {
        "k_valid",
        "transform_valid",
        "gravity_valid",
        "gravity_up_camera",
        "orientation",
    },
    "depth_scale": {"valid", "scale_m", "scale_sigma_m"},
    "support": {
        "valid",
        "normal_camera",
        "normal_sigma_rad",
        "camera_height_m",
        "height_sigma_m",
        "residual_sigma_m",
    },
    "boundary": {"valid", "coverage", "obstacles"},
    "obstacle": {
        "depth_valid",
        "depth_shape_forward",
        "depth_shape_sigma",
        "lateral_center_m",
        "lateral_half_width_m",
        "boundary_sigma_m",
        "evidence_probability",
        "evidence_sigma",
    },
}

GAP_GROUPS = {
    "scale_uncertainty": {
        "reducer_fields": ["depth_scale.scale_sigma_m"],
        "reason": "depth_log_sigma_hw is a dense log-depth residual scale and has no frozen deterministic calibration/aggregation to the reducer's scalar linear-meter scale_sigma_m",
        "required_operations": ["scale_sigma_m_from_calibrated_scale_distribution"],
    },
    "support_uncertainty": {
        "reducer_fields": ["support.normal_sigma_rad", "support.height_sigma_m"],
        "reason": "the F1 schema has support_residual_sigma_m but no normal-angle or camera-height uncertainty and no frozen derivation",
        "required_operations": ["support_normal_sigma_rad", "support_height_sigma_m"],
    },
    "dense_to_obstacle_list": {
        "reducer_fields": [
            "boundary.obstacles",
            "obstacle.depth_valid",
            "obstacle.depth_shape_forward",
            "obstacle.depth_shape_sigma",
            "obstacle.lateral_center_m",
            "obstacle.lateral_half_width_m",
            "obstacle.boundary_sigma_m",
            "obstacle.evidence_probability",
            "obstacle.evidence_sigma",
        ],
        "reason": "dense depth/boundary/evidence maps have no frozen component extraction, metric reprojection, interval aggregation, split/merge policy, evidence uncertainty or canonical order",
        "required_operations": [
            "dense_evidence_to_obstacle_components",
            "obstacle_forward_depth_interval",
            "obstacle_lateral_metric_interval",
            "boundary_sigma_px_to_metric",
            "evidence_probability_aggregation",
            "evidence_sigma_calibration",
            "local_missing_depth_to_depth_valid",
            "canonical_component_order",
        ],
    },
    "camera_and_frame_binding": {
        "reducer_fields": [
            "input_geometry.k_valid",
            "input_geometry.transform_valid",
            "input_geometry.gravity_valid",
            "input_geometry.gravity_up_camera",
            "input_geometry.orientation",
            "boundary.coverage",
        ],
        "reason": "the F1 schema binds a receipt SHA but does not define the adapter's exact K/transform/gravity validation or boundary-coverage computation",
        "required_operations": ["input_geometry_receipt"],
    },
}


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


def audit_contract(factor_schema: dict[str, Any], adapter_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    prediction_fields = {
        item.get("name")
        for item in factor_schema.get("prediction_fields", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    graph_boundary = factor_schema.get("graph_boundary", {})
    adapter_present = isinstance(adapter_contract, dict)
    operations = {
        item.get("id")
        for item in adapter_contract.get("operations", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if adapter_present else set()
    contract_flags = adapter_contract.get("authority", {}) if adapter_present else {}
    missing_operations = sorted(REQUIRED_ADAPTER_OPERATIONS - operations)
    authority_valid = (
        adapter_present
        and contract_flags.get("outside_learned_graph") is True
        and contract_flags.get("deterministic") is True
        and contract_flags.get("trainable_parameters") == 0
        and contract_flags.get("final_task_shortcut_allowed") is False
        and contract_flags.get("execution_authority") is False
    )
    static_complete = adapter_present and not missing_operations and authority_valid
    return {
        "schema": RESULT_SCHEMA,
        "terminal": READY_FOR_CANARY_TERMINAL if static_complete else BLOCKED_TERMINAL,
        "adapter_contract_present": adapter_present,
        "adapter_static_contract_complete": static_complete,
        "prediction_field_count": len(prediction_fields),
        "f1_graph_declares_reducer_outside": graph_boundary.get("geometry_r2_reducer_outside_learned_graph") is True,
        "reducer_required_fields": {key: sorted(value) for key, value in REDUCER_REQUIRED_FIELDS.items()},
        "required_adapter_operations": sorted(REQUIRED_ADAPTER_OPERATIONS),
        "missing_adapter_operations": missing_operations,
        "adapter_authority_valid": authority_valid,
        "gap_groups": GAP_GROUPS,
        "current_execution_authority": False,
        "unique_successor": {"id": SUCCESSOR_ID, "execution_authority": False},
        "claim_ceiling": "Static F1-schema to F0-reducer interface compatibility only. No adapter mechanics, mutation canary, factor learnability, training or task evidence.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-schema", type=Path, required=True)
    parser.add_argument("--adapter-contract", type=Path)
    parser.add_argument("--f0-reducer", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    factor_schema = load_json(args.factor_schema.resolve())
    adapter = load_json(args.adapter_contract.resolve()) if args.adapter_contract else None
    result = audit_contract(factor_schema, adapter)
    result["bindings"] = {
        "factor_schema_sha256": sha256_file(args.factor_schema.resolve()),
        "f0_reducer_sha256": sha256_file(args.f0_reducer.resolve()),
        "adapter_contract_sha256": sha256_file(args.adapter_contract.resolve()) if args.adapter_contract else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
