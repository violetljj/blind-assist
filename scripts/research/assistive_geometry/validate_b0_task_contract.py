#!/usr/bin/env python3
"""Validate the pre-outcome Assistive Geometry B0 task contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "blindassist.assistive_geometry.b0_task_contract.v1"
EXPECTED_SUCCESSOR = "BLINDASSIST_ASSISTIVE_GEOMETRY_B0_INPUT_DATA_PREFLIGHT"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_contract(payload: dict[str, Any]) -> None:
    require(payload.get("schema_version") == EXPECTED_SCHEMA, "schema drift")
    statuses = set(payload.get("status", []))
    require("EXECUTION_NOT_AUTHORIZED" in statuses, "B0 must not authorize execution")
    authority = payload.get("authority", {})
    for key in (
        "student_training",
        "candidate_outcome_access",
        "depthart_d1_activation",
        "r2_activation",
        "default_app_change",
    ):
        require(authority.get(key) is False, f"authority must stay false: {key}")

    geometry = payload.get("input_geometry", {})
    require(geometry.get("requested_buffer_wh") == [640, 480], "CameraX request drift")
    resize = geometry.get("resize", {})
    require(resize.get("fixed_tensor_nchw") == [1, 3, 608, 448], "tensor shape drift")
    require(resize.get("padding") == "NONE", "padding is not frozen to NONE")
    require(geometry.get("extra_crop") == "FORBIDDEN", "extra crop must remain forbidden")
    require(geometry.get("dynamic_transformed_k_input_required") is True, "dynamic K required")

    state = payload.get("geometry_state", {})
    require(state.get("horizons_m") == [1.0, 1.5, 2.0], "horizon drift")
    body = state.get("research_body_profile", {})
    require(abs(float(body.get("total_half_width_m", -1)) - 0.42) < 1e-9, "body profile drift")
    require(state.get("states") == ["CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"], "state drift")

    unknown = payload.get("unknown_contract", {})
    require(unknown.get("unknown_is_negative") is False, "UNKNOWN must not become negative")
    require(
        unknown.get("deterministic_invalid_cannot_be_overridden_by_confidence") is True,
        "confidence must not override deterministic invalidity",
    )

    roles = payload.get("truth_roles", {})
    require(
        roles.get("teacher_outputs") == "distillation_only_not_real_world_safety_truth",
        "teacher authority drift",
    )
    firewall = payload.get("data_firewall", {})
    require(firewall.get("consumed_120_frame_cohort_forbidden") is True, "consumed cohort reopened")
    require(firewall.get("existing_arkitscenes_r2_roster_forbidden") is True, "R2 roster reused")

    blockers = payload.get("unresolved_blockers", [])
    require(len(blockers) >= 5, "unresolved blockers were silently removed")
    require(payload.get("next_successor") == EXPECTED_SUCCESSOR, "successor drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(payload)
    print(json.dumps({"status": "B0_TASK_CONTRACT_VALID", "contract": str(args.contract)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
