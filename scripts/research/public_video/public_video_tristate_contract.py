#!/usr/bin/env python3
"""Validation helpers for the frozen prospective public-video lifecycle contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_tristate_lifecycle_contract_v1"
WORKZONE_MARKER_ADDITIONS = frozenset({
    "barricade",
    "cone",
    "construction worker",
    "traffic cone",
})
SELECTED_GROUPS = ("surface_material", "barrier_structure")
REQUIRED_FALSE_AUTHORIZATIONS = {
    "training_execution_authorized",
    "human_event_truth_present",
    "calibration_authorized",
    "blind_evaluation_authorized",
    "android_runtime_change_authorized",
    "production_model_replacement_authorized",
}


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unexpected tri-state lifecycle contract schema")
    if not isinstance(contract.get("contract_id"), str) or not contract["contract_id"]:
        raise ValueError("contract_id is missing")
    model = contract.get("model")
    scan = contract.get("scan")
    lifecycle = contract.get("lifecycle")
    eligibility = contract.get("source_eligibility")
    acceptance = contract.get("acceptance")
    authorization = contract.get("authorization")
    if not all(isinstance(value, dict) for value in (
        model, scan, lifecycle, eligibility, acceptance, authorization
    )):
        raise ValueError("contract sections are incomplete")
    if model.get("text_prompt_used") is not False:
        raise ValueError("prospective contract must remain prompt-free")
    weights_sha256 = model.get("weights_sha256")
    if not isinstance(weights_sha256, str) or len(weights_sha256) != 64:
        raise ValueError("contract weights SHA256 is invalid")
    if scan.get("workzone_marker_additions") != sorted(WORKZONE_MARKER_ADDITIONS):
        raise ValueError("work-zone marker additions differ from the frozen set")
    if scan.get("baseline_groups") != list(SELECTED_GROUPS):
        raise ValueError("scan baseline groups differ from the frozen set")
    if lifecycle.get("selected_groups") != list(SELECTED_GROUPS):
        raise ValueError("lifecycle selected groups differ from the frozen set")
    for key in ("sample_interval_ms", "image_size"):
        if not isinstance(scan.get(key), int) or scan[key] <= 0:
            raise ValueError(f"scan {key} must be a positive integer")
    if not isinstance(scan.get("confidence"), (int, float)) or not 0 < scan["confidence"] < 1:
        raise ValueError("scan confidence must be between zero and one")
    for key in ("entry_window_samples", "entry_min_active_samples", "clear_absent_samples"):
        if not isinstance(lifecycle.get(key), int) or lifecycle[key] <= 0:
            raise ValueError(f"lifecycle {key} must be a positive integer")
    if lifecycle["entry_min_active_samples"] > lifecycle["entry_window_samples"]:
        raise ValueError("lifecycle entry threshold exceeds its window")
    if any(eligibility.get(key) is not expected for key, expected in {
        "item_level_reusable_license_required": True,
        "original_temporal_order_required": True,
        "continuous_capture_required": True,
        "hard_cut_or_montage_allowed": False,
        "prospective_source_must_not_have_influenced_r78_parameters": True,
    }.items()):
        raise ValueError("source eligibility has been weakened")
    if acceptance.get("exactly_one_exit_interval_required") is not True:
        raise ValueError("exactly-one interval acceptance must remain enabled")
    if acceptance.get("gpt_visual_reference_must_be_contained") is not True:
        raise ValueError("visual reference containment must remain enabled")
    if acceptance.get("post_clear_single_frame_reopen_allowed") is not False:
        raise ValueError("post-clear single-frame reopening must remain forbidden")
    if not 0 <= float(acceptance.get("minimum_risk_present_active_fraction", -1)) <= 1:
        raise ValueError("risk active-fraction threshold is invalid")
    if not 0 <= float(acceptance.get("maximum_stable_clear_active_fraction", -1)) <= 1:
        raise ValueError("clear active-fraction threshold is invalid")
    if any(authorization.get(key) is not False for key in REQUIRED_FALSE_AUTHORIZATIONS):
        raise ValueError("prospective contract contains an unauthorized promotion flag")
    return contract


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve()
    mil.reject_independent_direction(resolved)
    sidecar = Path(str(resolved) + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"contract or sidecar is missing: {resolved}")
    expected = sidecar.read_text(encoding="ascii").strip().lower()
    actual = common.sha256_file(resolved)
    if expected != actual:
        raise ValueError(f"contract sidecar mismatch: {resolved}")
    return validate_contract(common.load_json(resolved)), {
        "path": str(resolved),
        "sha256": actual,
    }


def validate_scan_binding(
    contract: dict[str, Any],
    *,
    weights_sha256: str,
    sample_interval_ms: int,
    image_size: int,
    confidence: float,
    require_nearfield_corridor: bool,
    include_workzone_markers: bool,
) -> None:
    scan = contract["scan"]
    if weights_sha256 != contract["model"]["weights_sha256"]:
        raise ValueError("weights do not match the frozen prospective contract")
    if sample_interval_ms != scan["sample_interval_ms"]:
        raise ValueError("sample interval does not match the frozen prospective contract")
    if image_size != scan["image_size"]:
        raise ValueError("image size does not match the frozen prospective contract")
    if abs(confidence - float(scan["confidence"])) > 1e-12:
        raise ValueError("confidence does not match the frozen prospective contract")
    if require_nearfield_corridor is not bool(scan["require_nearfield_corridor"]):
        raise ValueError("near-field setting does not match the frozen prospective contract")
    if not include_workzone_markers:
        raise ValueError("frozen prospective contract requires work-zone marker additions")
