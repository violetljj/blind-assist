"""Validation helpers for the frozen r7.17 dual-evidence lifecycle contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_public_silver_dual_evidence_lifecycle_fusion as fusion
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_dual_evidence_lifecycle_contract_v1"
EXPECTED_STATES = ["clear", "risk", "uncertain"]
EXPECTED_WINDOWS = ["pre_risk_clear", "risk_present", "stable_post_clear"]
EXPECTED_GROUPS = {
    "surface_material": [
        "dirt", "dirt field", "dirt road", "dirt track",
        "earth", "sand", "sand bar", "sand box",
    ],
    "barrier_structure": [
        "barrier", "construction site", "furniture", "obstacle course",
    ],
}
REQUIRED_FALSE_AUTHORIZATIONS = {
    "human_event_truth_present",
    "training_execution_authorized",
    "calibration_authorized",
    "blind_evaluation_authorized",
    "android_runtime_change_authorized",
    "production_model_replacement_authorized",
}


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise ValueError(f"{label} SHA256 is invalid")
    return value.lower()


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unexpected dual-evidence lifecycle contract schema")
    if not isinstance(contract.get("contract_id"), str) or not contract["contract_id"]:
        raise ValueError("contract_id is missing")
    sections = (
        "derivation_evidence", "feature_contract", "lifecycle_contract",
        "review_protocol", "source_eligibility", "acceptance", "authorization",
    )
    if any(not isinstance(contract.get(key), dict) for key in sections):
        raise ValueError("dual-evidence contract sections are incomplete")

    derivation = contract["derivation_evidence"]
    _sha256(derivation.get("r716_report_sha256"), "r7.16 report")
    if any(derivation.get(key) is not expected for key, expected in {
        "r716_was_post_hoc": True,
        "margin_selected_before_new_source_review": True,
        "new_source_visual_content_opened_before_contract_freeze": False,
    }.items()):
        raise ValueError("derivation chronology or post-hoc disclosure was weakened")

    features = contract["feature_contract"]
    if features.get("sample_interval_ms") != 1000:
        raise ValueError("prospective feature sample interval must remain 1000 ms")
    dynamic = features.get("dynamic_channel")
    static = features.get("static_channel")
    semantic = features.get("semantic_exit_channel")
    if not all(isinstance(value, dict) for value in (dynamic, static, semantic)):
        raise ValueError("feature channel contracts are incomplete")
    _sha256(dynamic.get("weights_sha256"), "dynamic weights")
    if dynamic.get("image_size") != 320 or float(dynamic.get("confidence", -1)) != 0.15:
        raise ValueError("dynamic channel inference contract drifted")
    if static.get("motion_size") != 320 or static.get("minimum_reliable_transitions_per_window") != 2:
        raise ValueError("static channel reliability contract drifted")
    _sha256(semantic.get("weights_sha256"), "semantic weights")
    if (
        semantic.get("image_size") != 640
        or float(semantic.get("confidence", -1)) != 0.01
        or semantic.get("text_prompt_used") is not False
    ):
        raise ValueError("semantic channel inference contract drifted")
    if semantic.get("selected_groups") != EXPECTED_GROUPS:
        raise ValueError("semantic exit groups differ from the frozen set")
    if any(semantic.get(key) is not True for key in (
        "required_risk_window_detection",
        "required_post_clear_absence",
        "required_post_clear_dynamic_peak_below_risk_peak",
    )) or semantic.get("maximum_risk_to_clear_gap_ms") != 5000:
        raise ValueError("semantic exit requirements were weakened")

    lifecycle = contract["lifecycle_contract"]
    if lifecycle.get("states") != EXPECTED_STATES:
        raise ValueError("lifecycle states differ from the frozen set")
    if abs(float(lifecycle.get("strong_normalized_change_margin", -1)) - fusion.STRONG_MARGIN) > 1e-12:
        raise ValueError("strong relative-change margin drifted")
    if lifecycle.get("cold_start_state") != "uncertain":
        raise ValueError("cold start must remain uncertain")
    if lifecycle.get("missing_or_conflicting_evidence_state") != "uncertain":
        raise ValueError("missing or conflicting evidence must remain uncertain")
    if lifecycle.get("absolute_scene_threshold_used") is not False or lifecycle.get("learned_parameters") != 0:
        raise ValueError("prospective lifecycle contract must remain threshold-relative and unlearned")

    review = contract["review_protocol"]
    if review.get("required_windows") != EXPECTED_WINDOWS:
        raise ValueError("required review windows differ from the frozen set")
    if (
        review.get("full_feature_report_frozen_before_visual_review") is not True
        or review.get("original_temporal_order_required") is not True
        or review.get("hard_cut_or_montage_allowed") is not False
        or review.get("reviewer_may_not_edit_feature_values") is not True
        or review.get("minimum_pre_risk_clear_samples") != 3
        or review.get("minimum_risk_present_samples") != 3
        or review.get("minimum_stable_post_clear_samples") != 6
        or review.get("post_clear_confirmation_samples") != 3
    ):
        raise ValueError("review chronology or sample requirements were weakened")

    eligibility = contract["source_eligibility"]
    if any(eligibility.get(key) is not expected for key, expected in {
        "item_level_reusable_license_required": True,
        "continuous_capture_required": True,
        "original_temporal_order_required": True,
        "hard_cut_or_montage_allowed": False,
        "source_inventory_eligibility_required": True,
        "source_must_not_have_influenced_r714_r715_r716": True,
        "pedestrian_forward_view_reported_separately": True,
    }.items()):
        raise ValueError("prospective source eligibility was weakened")

    acceptance = contract["acceptance"]
    if any(acceptance.get(key) is not True for key in (
        "strong_open_required", "close_required",
        "weak_close_requires_exact_semantic_corroboration",
        "stable_post_clear_must_not_reopen",
        "cold_start_control_must_remain_uncertain",
        "semantic_absence_only_control_must_remain_uncertain",
        "conflicting_rise_and_exit_control_must_remain_uncertain",
        "visual_boundary_must_be_contained",
        "prospective_source_lineage_required",
    )):
        raise ValueError("prospective acceptance checks were weakened")
    if any(contract["authorization"].get(key) is not False for key in REQUIRED_FALSE_AUTHORIZATIONS):
        raise ValueError("prospective contract contains an unauthorized promotion flag")
    return contract


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve()
    mil.reject_independent_direction(resolved)
    sidecar = Path(str(resolved) + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"contract or sidecar is missing: {resolved}")
    actual = common.sha256_file(resolved)
    if sidecar.read_text(encoding="ascii").strip().lower() != actual:
        raise ValueError(f"contract sidecar mismatch: {resolved}")
    return validate_contract(common.load_json(resolved)), {
        "path": str(resolved),
        "sha256": actual,
    }
