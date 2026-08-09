#!/usr/bin/env python3
"""Validate the pre-outcome D1 contract and metadata-only Development roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = (
    "blindassist_depthart_task_preserving_d1_fixed_mixed_development_protocol_v1"
)
ROSTER_SCHEMA = (
    "blindassist_depthart_task_preserving_d1_arkit_development_roster_lock_v1"
)
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_SCREEN"
PRIOR_TERMINAL = (
    "CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED"
)
SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _validate_binding(repo: Path, binding: dict[str, Any], name: str) -> None:
    _require(isinstance(binding, dict), f"binding {name} must be an object")
    path = repo / str(binding.get("path", ""))
    _require(path.is_file(), f"binding {name} path missing")
    _require(path.stat().st_size == binding.get("bytes"), f"binding {name} byte drift")
    _require(
        isinstance(binding.get("sha256"), str)
        and SHA256_PATTERN.fullmatch(binding["sha256"]) is not None,
        f"binding {name} SHA-256 invalid",
    )
    _require(_sha256(path) == binding["sha256"], f"binding {name} SHA-256 drift")


def validate(
    protocol: dict[str, Any],
    roster: dict[str, Any],
    b0: dict[str, Any],
    r2: dict[str, Any],
) -> dict[str, Any]:
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema mismatch")
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id mismatch")
    _require(protocol.get("immutable_prior_terminal") == PRIOR_TERMINAL,
             "strict G4-D terminal changed")
    _require(
        protocol.get("status")
        == "CONTRACT_AND_METADATA_ROSTER_FROZEN_EXECUTION_NOT_ACTIVATED_NO_OUTCOME_ACCESSED",
        "protocol is not in the frozen pre-outcome state",
    )
    _require(protocol.get("current_execution_authorized") is False,
             "protocol cannot authorize execution")
    _require(protocol.get("current_outcome_access") == "NONE",
             "protocol outcome access must remain NONE")

    input_geometry = protocol.get("input_geometry")
    b0_input = b0.get("input_geometry")
    _require(isinstance(input_geometry, dict) and isinstance(b0_input, dict),
             "input geometry missing")
    _require(input_geometry.get("requested_buffer_wh") == b0_input.get("requested_buffer_wh"),
             "CameraX buffer geometry drift")
    _require(input_geometry.get("nominal_display_wh") == b0_input["resize"].get("nominal_display_wh"),
             "display geometry drift")
    _require(input_geometry.get("tensor_nchw") == b0_input["resize"].get("fixed_tensor_nchw"),
             "product tensor geometry drift")
    _require(input_geometry.get("extra_crop") == "FORBIDDEN", "extra crop must be forbidden")
    _require(input_geometry.get("padding") == "NONE", "padding must remain NONE")
    _require(input_geometry.get("intrinsics_transform_order") == [
        "source crop_rect",
        "clockwise rotation to display upright",
        "independent final sx_sy full-FOV resize",
    ], "intrinsics transform order drift")

    postprocess = protocol.get("task_postprocess")
    geometry_state = b0.get("geometry_state")
    _require(isinstance(postprocess, dict) and isinstance(geometry_state, dict),
             "task postprocess missing")
    expected_bands = {
        key: geometry_state["bands"][key] for key in ("left", "center", "right")
    }
    _require(postprocess.get("bands") == expected_bands, "band contract drift")
    _require(postprocess.get("horizons_m") == geometry_state.get("horizons_m"),
             "horizon contract drift")
    body = geometry_state.get("research_body_profile", {})
    for key in ("body_half_width_m", "lateral_margin_m", "total_half_width_m"):
        _require(postprocess.get(key) == body.get(key), f"{key} drift")
    _require(postprocess.get("clearance_quantile") == geometry_state.get("clearance_quantile"),
             "clearance quantile drift")
    _require(postprocess.get("unknown_is_negative") is False,
             "UNKNOWN must never be negative")
    _require(protocol["quality_contract"].get("required_cells_per_frame") == 9,
             "all three bands and horizons are required")
    _require(protocol["quality_contract"].get("gates") == r2.get("gates"),
             "D1 gates must exactly match the frozen R2 quality gates")

    _require(roster.get("schema") == ROSTER_SCHEMA, "roster schema mismatch")
    _require(roster.get("protocol_id") == PROTOCOL_ID, "roster protocol id mismatch")
    _require(
        roster.get("status")
        == "METADATA_ROSTER_8_PRIMARY_8_RESERVE_LOCKED_MEDIA_UNOPENED_DOWNLOAD_NOT_AUTHORIZED",
        "roster is not metadata-only and unopened",
    )
    primary = roster.get("primary")
    reserve = roster.get("reserve")
    _require(isinstance(primary, list) and len(primary) == 8, "primary roster must contain 8 rows")
    _require(isinstance(reserve, list) and len(reserve) == 8, "reserve roster must contain 8 rows")
    selected = primary + reserve
    _require(all(row.get("fold") == "Training" for row in selected),
             "D1 roster must use Training identities")
    parent_ids = [row.get("visit_id") for row in selected]
    session_ids = [row.get("video_id") for row in selected]
    _require(all(isinstance(value, str) and value for value in parent_ids + session_ids),
             "roster contains an invalid identity")
    _require(len(parent_ids) == len(set(parent_ids)), "roster visit identities overlap")
    _require(len(session_ids) == len(set(session_ids)), "roster video identities overlap")
    invariants = roster.get("invariants")
    _require(isinstance(invariants, dict), "roster invariants missing")
    _require(invariants.get("selection_overlap_with_exclusion_snapshot") == 0,
             "roster overlaps a frozen prior identity")
    for key in (
        "media_body_bytes_read",
        "depth_or_rgb_opened",
        "truth_or_model_outputs_read",
        "download_authorized",
        "replacement_by_outcome_allowed",
    ):
        _require(invariants.get(key) is False, f"roster invariant {key} must be false")
    _require(invariants.get("outcome_access") == "NONE",
             "roster outcome access must remain NONE")

    return {
        "schema": "blindassist_depthart_task_preserving_d1_contract_validation_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "CONTRACT_AND_METADATA_ROSTER_VALID_EXECUTION_NOT_AUTHORIZED",
        "checks": {
            "strict_g4d_terminal_immutable": True,
            "product_portrait_geometry_bound_to_b0": True,
            "dynamic_intrinsics_and_unknown_fail_closed": True,
            "three_bands_by_three_horizons_frozen": True,
            "d1_quality_gates_equal_r2": True,
            "metadata_primary_and_reserve_unique": True,
            "prior_route_identity_overlap_zero": True,
            "media_truth_and_model_outcomes_unopened": True,
        },
        "execution_authorized": False,
        "outcome_access": "NONE",
        "next_action": (
            "LICENSE_SCOPE_EXTENSION_THEN_LABEL_BLIND_PORTRAIT_POSE_RGBD_MEDIA_PREFLIGHT"
        ),
        "authority": "PRE_OUTCOME_CONTRACT_AND_METADATA_ROSTER_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = _load(args.protocol)
    for name, binding in protocol.get("bindings", {}).items():
        _validate_binding(args.repo, binding, name)
    roster = _load(args.roster)
    _validate_binding(args.repo, roster.get("planner", {}), "roster_planner")
    b0 = _load(args.repo / protocol["bindings"]["b0_task_contract"]["path"])
    r2 = _load(args.repo / protocol["bindings"]["r2_quality_contract"]["path"])
    receipt = validate(protocol, roster, b0, r2)
    receipt["identities"] = {
        "protocol_sha256": _sha256(args.protocol),
        "roster_sha256": _sha256(args.roster),
    }
    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
