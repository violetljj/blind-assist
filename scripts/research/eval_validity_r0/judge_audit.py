"""Fail-closed primitive-first judge audit for evaluator validity.

The reviewer records observable primitive relations only.  Actionability,
clearance and knownness are derived by a frozen machine rule after the review
packet is submitted.  The audit also separates the current-system oracle chain
from an oracle's native information-ceiling measurement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .common import PROTOCOL_ID, read_json, sha256_file, sha256_json


JUDGE_CONTRACT_SCHEMA = "blindassist.eval_validity_r0.judge_contract.v1"
EVENT_LEDGER_SCHEMA = "blindassist.eval_validity_r0.judge_event_ledger.v2"
REVIEW_MAP_SCHEMA = "blindassist.eval_validity_r0.judge_review_map.v1"
REVIEW_SCHEMA = "blindassist.eval_validity_r0.judge_review.v6"
PAIR_SCHEMA = "blindassist.eval_validity_r0.judge_counterfactual_pairs.v5"
ORACLE_SCHEMA = "blindassist.eval_validity_r0.judge_oracle_manifest.v3"
REPORT_SCHEMA = "blindassist.eval_validity_r0.judge_audit_report.v6"

SCENARIO_CATEGORIES = (
    "known_front_obstacle",
    "unknown_object",
    "roadside_nonblocking",
    "head_on_approach",
    "lateral_crossing",
    "camera_motion_only",
    "wide_corridor",
    "insufficient_evidence",
)
DISCOVERY_ARMS = (
    "source_mask",
    "random_continuous_rgb",
    "motion_temporal_change",
    "metadata_only_normal",
)
INDEPENDENT_DISCOVERY_ARMS = DISCOVERY_ARMS[1:]
PHYSICAL_CONDITIONS = {"BLOCKING_PATH", "NON_BLOCKING_PATH", "UNKNOWN"}
PRIMITIVE_FIELDS = (
    "visibility",
    "path_relation",
    "motion_relation",
    "phase",
    "route_certainty",
    "evidence_quality",
)
PRIMITIVE_VALUES = {
    "visibility": {"EVALUABLE", "NOT_EVALUABLE"},
    "path_relation": {"BLOCKING_PATH", "NON_BLOCKING_PATH", "AMBIGUOUS"},
    "motion_relation": {"APPROACHING", "LATERAL_PASS", "RECEDING", "STATIC_OR_UNCLEAR"},
    "phase": {"BEFORE_INTRUSION", "CURRENT_INTRUSION", "PASSED_CLEAR", "UNKNOWN"},
    "route_certainty": {"SINGLE_PLAUSIBLE_ROUTE", "MULTIPLE_PLAUSIBLE_ROUTES", "UNKNOWN"},
    "evidence_quality": {"CLEAR", "BLUR", "OCCLUSION", "CAMERA_ROTATION", "INSUFFICIENT"},
}
PRIMITIVE_POLICY_VERSION = "primitive_observability_v4"
VISIBILITY_POLICY_VERSION = "visibility_observability_v2"
PATH_RELATION_POLICY_VERSION = "path_relation_observability_v2"
ROUTE_CERTAINTY_POLICY_VERSION = "route_certainty_observability_v2"
EVIDENCE_QUALITY_POLICY_VERSION = "evidence_quality_observability_v2"
PHASE_POLICY_VERSION = "phase_observability_v2"
VISIBILITY_EVIDENCE_WINDOW = "CURRENT_RGB_FRAME_ONLY"
GEOMETRIC_EVIDENCE_WINDOW = "CURRENT_RGB_FRAME_ONLY"
CAUSAL_TEMPORAL_EVIDENCE_WINDOW = "CURRENT_PLUS_PAST_PREFIX"
RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW = "FULL_EVENT_RGB"
PRIMITIVE_OBSERVATION_POLICY = {
    "version": PRIMITIVE_POLICY_VERSION,
    "route_anchor": {
        "definition": "The route is the pedestrian-support surface currently carrying the camera and its visible forward continuation.",
        "anchor_rule": "Use the surface directly under the camera and connected forward; do not infer the route from object identity, a reminder policy or a parallel/background walkway.",
        "included_surfaces": ["SIDEWALK", "PATH", "TRAIL", "OTHER_CURRENT_WALKABLE_SUPPORT_SURFACE"],
        "excluded_surfaces": ["VEHICLE_LANE_NOT_CURRENTLY_OCCUPIED_BY_CAMERA", "PARALLEL_OR_BACKGROUND_WALKWAY_NOT_CONNECTED_TO_CURRENT_SURFACE", "OBJECT_SIDE_DETOUR_NOT_VISIBLE_AS_A_CONTINUATION"],
        "branch_rule": "Count a branch only when two or more materially distinct walkable continuations of this current surface are visibly connected.",
    },
    "field_evidence_windows": {
        "visibility": VISIBILITY_EVIDENCE_WINDOW,
        "path_relation": GEOMETRIC_EVIDENCE_WINDOW,
        "route_certainty": GEOMETRIC_EVIDENCE_WINDOW,
        "evidence_quality": GEOMETRIC_EVIDENCE_WINDOW,
        "motion_relation": CAUSAL_TEMPORAL_EVIDENCE_WINDOW,
        "phase": CAUSAL_TEMPORAL_EVIDENCE_WINDOW,
        "retrospective_motion_relation": RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
        "retrospective_phase": RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
    },
    "visibility": {
        "version": VISIBILITY_POLICY_VERSION,
        "operational_question": "Is the forward route scene observable in the current RGB frame?",
        "evaluable_if": "CURRENT_FRAME_CONTAINS_LOCALIZED_ROUTE_OR_SCENE_REGION",
        "not_evaluable_if": [
            "NO_LOCALIZED_ROUTE_OR_SCENE_REGION",
            "ROUTE_SCENE_OUT_OF_FRAME",
            "ROUTE_SCENE_FULLY_OCCLUDED",
        ],
        "object_identity_required": False,
        "obstacle_presence_required": False,
        "current_frame_only": True,
        "independent_of": [
            "path_relation",
            "motion_relation",
            "phase",
            "route_certainty",
            "evidence_quality",
            "actionability",
        ],
        "blur_rotation_handling": "Record blur, occlusion or camera rotation in evidence_quality; do not automatically convert it to NOT_EVALUABLE.",
        "causal_evidence_window": VISIBILITY_EVIDENCE_WINDOW,
        "retrospective_evidence_window": VISIBILITY_EVIDENCE_WINDOW,
    },
    "path_relation": {
        "version": PATH_RELATION_POLICY_VERSION,
        "operational_question": "Does a localized physical region or edge occupy the current route corridor in this frame?",
        "blocking_if": [
            "LOCALIZED_REGION_OR_EDGE_INTERSECTS_CURRENT_ROUTE_CORRIDOR",
            "VISIBLE_REGION_CLOSURE_LEAVES_NO_CONTINUOUS_PASSABLE_WIDTH_IN_CURRENT_ROUTE",
        ],
        "non_blocking_if": [
            "CURRENT_ROUTE_CORRIDOR_IS_VISIBLE_AND_CLEAR",
            "LOCALIZED_REGION_IS_OUTSIDE_CURRENT_ROUTE_CORRIDOR",
            "A_CONTINUOUS_PASSABLE_ROUTE_AROUND_REGION_IS_VISIBLE",
        ],
        "ambiguous_if": [
            "CURRENT_ROUTE_ANCHOR_OR_REGION_RELATION_CANNOT_BE_LOCALIZED",
            "RELEVANT_ROUTE_WIDTH_OR_EDGE_IS_HIDDEN",
            "TWO_SUPPORT_SURFACES_CANNOT_BE_DISTINGUISHED_AS_CURRENT_ROUTE",
        ],
        "object_identity_required": False,
        "motion_or_actionability_required": False,
        "current_frame_only": True,
        "not_ambiguous_because": [
            "OBJECT_CLASS_IS_UNKNOWN",
            "REMINDER_DECISION_IS_UNCERTAIN",
            "NO_SEPARATE_TARGET_IS_PRESENT_WHEN_THE_CURRENT_ROUTE_CORRIDOR_IS_CLEAR",
        ],
        "evidence_window": GEOMETRIC_EVIDENCE_WINDOW,
    },
    "route_certainty": {
        "version": ROUTE_CERTAINTY_POLICY_VERSION,
        "operational_question": "How many plausible forward continuations of the current support surface are visible?",
        "single_if": ["ONE_CONTINUOUS_FORWARD_CONTINUATION_OF_CURRENT_SUPPORT_SURFACE_IS_IDENTIFIABLE"],
        "multiple_if": ["TWO_OR_MORE_MATERIALLY_DISTINCT_WALKABLE_CONTINUATIONS_OF_CURRENT_SURFACE_ARE_VISIBLE"],
        "unknown_if": ["CURRENT_SUPPORT_SURFACE_OR_FORWARD_CONTINUATION_CANNOT_BE_LOCALIZED"],
        "current_frame_only": True,
        "not_multiple_because": [
            "VEHICLE_LANE_IS_VISIBLE",
            "PARALLEL_OR_BACKGROUND_WALKWAY_IS_VISIBLE",
            "A_SIDE_DETOUR_IS_ONLY_HYPOTHETICAL_OR_NOT_CONNECTED",
        ],
        "evidence_window": GEOMETRIC_EVIDENCE_WINDOW,
    },
    "evidence_quality": {
        "version": EVIDENCE_QUALITY_POLICY_VERSION,
        "operational_question": "Does the current RGB frame provide enough visual quality to interpret the route and candidate physical relations?",
        "clear_if": "ROUTE_ANCHOR_AND_RELEVANT_BOUNDARIES_ARE_INTERPRETABLE; MINOR_SOFTNESS_OR_NORMAL_SHADOWS_REMAIN_CLEAR",
        "blur_if": "MOTION_OR_FOCUS_BLUR_MATERIALLY_OBSCURES_THE_ROUTE_OR_RELEVANT_BOUNDARY; MINOR_SOFTNESS_DOES_NOT_COUNT",
        "occlusion_if": "A_FOREGROUND_OR_SCENE_OCCLUDER_MATERIALLY_HIDES_THE_ROUTE_OR_RELEVANT_BOUNDARY",
        "camera_rotation_if": "GLOBAL_CAMERA_ROTATION_PREVENTS_A_STABLE_ROUTE_ORIENTATION_FOR_THE_RELATION",
        "insufficient_if": "NO_USABLE_ROUTE_ANCHOR_REMAINS_AFTER_THE_ABOVE_CHECKS",
        "classification_precedence": ["INSUFFICIENT", "CAMERA_ROTATION", "OCCLUSION", "BLUR", "CLEAR"],
        "current_frame_only": True,
        "not_a_catch_all_for": ["ACTIONABILITY_UNCERTAINTY", "UNKNOWN_OBJECT_IDENTITY", "REVIEWER_DISAGREEMENT"],
        "evidence_window": GEOMETRIC_EVIDENCE_WINDOW,
    },
    "phase": {
        "version": PHASE_POLICY_VERSION,
        "operational_question": "What is the current route-occupancy phase relative to the observed allowed temporal prefix?",
        "before_if": "CURRENT_PATH_IS_NON_BLOCKING_AND_NO_EARLIER_BLOCKING_PATH_IS_OBSERVED_IN_THE_ALLOWED_PREFIX",
        "current_if": "CURRENT_PATH_IS_BLOCKING",
        "passed_if": "CURRENT_PATH_IS_NON_BLOCKING_AND_AN_EARLIER_BLOCKING_PATH_IS_OBSERVED_IN_THE_ALLOWED_PREFIX",
        "unknown_if": [
            "CURRENT_PATH_RELATION_IS_AMBIGUOUS_OR_UNAVAILABLE",
            "ROUTE_ANCHOR_OR_EVIDENCE_QUALITY_DOES_NOT_SUPPORT_THE_PATH_RELATION",
            "ALLOWED_PREFIX_CANNOT_ESTABLISH_THE_CURRENT_ROUTE_OCCUPANCY_STATE",
        ],
        "causal_rule": "Use current frame plus past prefix only; do not use future frames to call a current clear frame PASSED_CLEAR or to infer a future intrusion.",
        "retrospective_rule": "Full-event view may use earlier and later frames to describe the event sequence, but does not change current-only visibility/path/route/evidence fields.",
        "not_actionability_derived": True,
        "evidence_window_causal": CAUSAL_TEMPORAL_EVIDENCE_WINDOW,
        "evidence_window_retrospective": RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
    },
    "temporal_fields_causal_evidence_window": CAUSAL_TEMPORAL_EVIDENCE_WINDOW,
    "temporal_fields_retrospective_evidence_window": RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
}
DERIVED_LABELS = {"YES", "NO", "UNKNOWN"}
KNOWNNESS_VALUES = {"KNOWN", "UNKNOWN"}
ORACLE_ARMS = ("truth_mask_adapter", "truth_depth", "truth_geometry", "truth_trajectory")
SYSTEM_ARMS = ("current_yolo", *ORACLE_ARMS)
ORACLE_METRICS = (
    "positive_hits",
    "critical_misses",
    "false_alert_events",
    "cleared_positives",
    "response_delay_frames",
)
ORACLE_OPPORTUNITY_FIELDS = (
    "eligible_event_ids",
    "eligible_for_native_task",
    "eligible_for_system_chain",
    "required_inputs",
    "expected_improvement_dimension",
    "not_evaluable_reason",
)
DEFAULT_METADATA_FORBIDDEN_TOKENS = (
    "blocking",
    "parallel",
    "negative",
    "approach",
    "curb",
    "unknown-object",
    "roadside",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, where: str, *, integer: bool = False) -> float | int:
    if integer:
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{where}: expected non-negative integer")
        return value
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{where}: expected number")
    number = float(value)
    _require(number >= 0 and number == number and number != float("inf"), f"{where}: expected finite non-negative number")
    return number


def _contract(value: dict[str, Any]) -> dict[str, Any]:
    _require(value.get("schema_version") == JUDGE_CONTRACT_SCHEMA, "judge contract: schema mismatch")
    _require(value.get("protocol_id") == PROTOCOL_ID, "judge contract: protocol mismatch")
    _require(value.get("mode") in {"CALIBRATION_BURNED", "FORMAL_FROZEN"}, "judge contract: invalid mode")
    _require(value.get("cohort_role") in {"CALIBRATION_BURNED", "FORMAL_AUDIT"}, "judge contract: invalid cohort role")
    expected_role = "CALIBRATION_BURNED" if value["mode"] == "CALIBRATION_BURNED" else "FORMAL_AUDIT"
    _require(value["cohort_role"] == expected_role, "judge contract: mode/cohort role mismatch")
    for field in ("minimum_events", "maximum_events", "minimum_contiguous_frames", "minimum_event_duration_ms"):
        _number(value.get(field), f"judge contract: {field}", integer=True)
    _require(0 < value["minimum_events"] <= value["maximum_events"], "judge contract: invalid event range")
    for field in (
        "minimum_pre_alert_duration_ms",
        "minimum_active_duration_ms",
        "minimum_passed_clear_duration_ms",
    ):
        _number(value.get(field), f"judge contract: {field}", integer=True)
    coverage = value.get("required_coverage_min_counts")
    sessions = value.get("required_coverage_min_source_sessions")
    _require(isinstance(coverage, dict) and isinstance(sessions, dict), "judge contract: missing coverage minimums")
    for category in SCENARIO_CATEGORIES:
        _number(coverage.get(category), f"judge contract: coverage {category}", integer=True)
        _number(sessions.get(category), f"judge contract: source sessions {category}", integer=True)
    _number(value.get("maximum_event_share_per_session"), "judge contract: maximum session share")
    _require(0 < value["maximum_event_share_per_session"] <= 1, "judge contract: invalid session share")
    _number(value.get("yolo_box_similarity_threshold"), "judge contract: box similarity")
    _require(0 <= value["yolo_box_similarity_threshold"] <= 1, "judge contract: invalid box similarity")
    for field in (
        "minimum_counterfactual_pairs",
        "maximum_counterfactual_pairs",
        "boundary_tolerance_frames",
    ):
        _number(value.get(field), f"judge contract: {field}", integer=True)
    _require(value["minimum_counterfactual_pairs"] <= value["maximum_counterfactual_pairs"], "judge contract: invalid pair range")
    for field in (
        "minimum_pair_distance_scale_similarity",
        "minimum_pair_position_similarity",
        "minimum_pair_visibility_similarity",
        "minimum_event_consistency",
        "minimum_boundary_consistency",
        "minimum_primitive_consistency",
        "minimum_primitive_field_consistency",
        "minimum_derived_actionability_consistency",
        "minimum_derived_clearance_consistency",
        "minimum_primitive_to_derived_determinism",
        "maximum_unknown_event_rate",
    ):
        _number(value.get(field), f"judge contract: {field}")
        _require(0 <= value[field] <= 1, f"judge contract: invalid {field}")
    tokens = value.get("metadata_blind_forbidden_tokens", DEFAULT_METADATA_FORBIDDEN_TOKENS)
    _require(isinstance(tokens, list) and all(isinstance(item, str) and item for item in tokens), "judge contract: invalid metadata tokens")
    policy = value.get("derived_actionability_policy")
    _require(isinstance(policy, dict), "judge contract: missing derived actionability policy")
    _require(policy.get("reviewer_may_submit_action_labels") is False, "judge contract: reviewer action labels must be machine-derived")
    primitive_policy = value.get("primitive_observation_policy")
    _require(isinstance(primitive_policy, dict), "judge contract: missing primitive observation policy")
    _require(primitive_policy.get("version") == PRIMITIVE_POLICY_VERSION, "judge contract: primitive observation policy version mismatch")
    _require(primitive_policy.get("route_anchor") == PRIMITIVE_OBSERVATION_POLICY["route_anchor"], "judge contract: route anchor definition mismatch")
    _require(primitive_policy.get("field_evidence_windows") == PRIMITIVE_OBSERVATION_POLICY["field_evidence_windows"], "judge contract: field evidence windows mismatch")
    visibility_policy = primitive_policy.get("visibility")
    _require(isinstance(visibility_policy, dict), "judge contract: missing visibility policy")
    _require(visibility_policy.get("version") == VISIBILITY_POLICY_VERSION, "judge contract: visibility policy version mismatch")
    _require(visibility_policy.get("operational_question") == PRIMITIVE_OBSERVATION_POLICY["visibility"]["operational_question"], "judge contract: visibility operational question mismatch")
    _require(visibility_policy.get("evaluable_if") == PRIMITIVE_OBSERVATION_POLICY["visibility"]["evaluable_if"], "judge contract: visibility evaluable rule mismatch")
    _require(visibility_policy.get("not_evaluable_if") == PRIMITIVE_OBSERVATION_POLICY["visibility"]["not_evaluable_if"], "judge contract: visibility non-evaluable rule mismatch")
    _require(visibility_policy.get("object_identity_required") is False and visibility_policy.get("obstacle_presence_required") is False, "judge contract: visibility must not require object identity or obstacle presence")
    _require(visibility_policy.get("current_frame_only") is True, "judge contract: visibility must be current-frame-only")
    _require(visibility_policy.get("independent_of") == PRIMITIVE_OBSERVATION_POLICY["visibility"]["independent_of"], "judge contract: visibility dependency boundary mismatch")
    _require(visibility_policy.get("blur_rotation_handling") == PRIMITIVE_OBSERVATION_POLICY["visibility"]["blur_rotation_handling"], "judge contract: visibility quality boundary mismatch")
    _require(visibility_policy.get("causal_evidence_window") == VISIBILITY_EVIDENCE_WINDOW and visibility_policy.get("retrospective_evidence_window") == VISIBILITY_EVIDENCE_WINDOW, "judge contract: visibility evidence window mismatch")
    for field in ("path_relation", "route_certainty", "evidence_quality"):
        _require(primitive_policy.get(field) == PRIMITIVE_OBSERVATION_POLICY[field], f"judge contract: {field} observation policy mismatch")
    _require(primitive_policy.get("phase") == PRIMITIVE_OBSERVATION_POLICY["phase"], "judge contract: phase observation policy mismatch")
    _require(primitive_policy.get("temporal_fields_causal_evidence_window") == CAUSAL_TEMPORAL_EVIDENCE_WINDOW, "judge contract: causal temporal evidence window mismatch")
    _require(primitive_policy.get("temporal_fields_retrospective_evidence_window") == RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW, "judge contract: retrospective temporal evidence window mismatch")
    pair_policy = value.get("counterfactual_pair_policy")
    _require(isinstance(pair_policy, dict), "judge contract: missing counterfactual pair policy")
    _require(pair_policy.get("selection_stage") == "AFTER_REVIEWS_SEALED", "judge contract: pair selection must follow sealed reviews")
    _require(pair_policy.get("yolo_role") == "SELECTION_ONLY", "judge contract: pair YOLO role must be selection-only")
    _require(pair_policy.get("yolo_visible_to_reviewers") is False and pair_policy.get("yolo_used_for_truth") is False, "judge contract: invalid pair YOLO visibility policy")
    _require(pair_policy.get("primitive_labels_visible_to_pair_builder") is False and pair_policy.get("derived_labels_visible_to_pair_builder") is False, "judge contract: pair builder cannot see labels")
    _require(pair_policy.get("selection_fields") == ["yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity", "selection_time_slot"], "judge contract: pair selection fields mismatch")
    _require(pair_policy.get("selection_time_slot_source") in {"frame_index_or_timestamp", "fixed_sampling_slot"}, "judge contract: invalid selection time-slot source")
    _require(pair_policy.get("reviewed_event_phase_field") == "reviewed_event_phase" and pair_policy.get("reviewed_motion_relation_field") == "reviewed_motion_relation", "judge contract: reviewer-derived selection fields are not explicitly separated")
    _require(pair_policy.get("reviewed_event_phase_visible_to_pair_builder") is False and pair_policy.get("reviewed_motion_relation_visible_to_pair_builder") is False, "judge contract: reviewer-derived fields must be hidden from pair builder")
    _require(isinstance(pair_policy.get("ordering_rule"), str) and pair_policy["ordering_rule"], "judge contract: missing deterministic pair ordering rule")
    _require(pair_policy.get("below_minimum_terminal") == "NOT_EVALUABLE", "judge contract: pair below-minimum terminal mismatch")
    discovery_policy = value.get("discovery_arm_policy")
    _require(isinstance(discovery_policy, dict), "judge contract: missing discovery arm policy")
    _require(discovery_policy.get("allowed_arms") == list(DISCOVERY_ARMS), "judge contract: discovery arm allowlist mismatch")
    _require(discovery_policy.get("source_mask_arm") == "source_mask", "judge contract: source-mask discovery arm mismatch")
    _require(discovery_policy.get("independent_arms") == list(INDEPENDENT_DISCOVERY_ARMS), "judge contract: independent discovery arm allowlist mismatch")
    for field in ("minimum_distinct_arms_formal", "minimum_independent_arms_formal", "minimum_distinct_arms_calibration"):
        _number(discovery_policy.get(field), f"judge contract: {field}", integer=True)
    _require(discovery_policy["minimum_distinct_arms_formal"] >= 2, "judge contract: formal cohort needs multiple discovery arms")
    _require(discovery_policy["minimum_independent_arms_formal"] >= 1, "judge contract: formal cohort needs an independent discovery arm")
    native_gates = value.get("native_information_ceiling_gates")
    _require(isinstance(native_gates, dict), "judge contract: missing native information gates")
    for arm in ORACLE_ARMS:
        _require(isinstance(native_gates.get(arm), dict) and native_gates[arm], f"judge contract: missing native gate {arm}")
        for metric, threshold in native_gates[arm].items():
            _number(threshold, f"judge contract: native gate {arm}/{metric}")
            _require(0 <= threshold <= 1, f"judge contract: native gate outside [0,1] {arm}/{metric}")
    _require(value.get("oracle_opportunity_required_fields") == list(ORACLE_OPPORTUNITY_FIELDS), "judge contract: oracle opportunity fields mismatch")
    _require(isinstance(value.get("shared_execution"), dict), "judge contract: missing shared execution")
    retrospective_policy = value.get("retrospective_comparison_policy")
    _require(isinstance(retrospective_policy, dict), "judge contract: missing retrospective comparison policy")
    _require(retrospective_policy.get("adjudicates_causal_truth") is False and retrospective_policy.get("gated") is False, "judge contract: retrospective view cannot adjudicate causal truth")
    return value


def _validate_registry(registry: dict[str, Any] | None) -> set[str]:
    if registry is None:
        return set()
    sessions = registry.get("excluded_source_sessions")
    _require(isinstance(sessions, list) and all(isinstance(item, str) for item in sessions), "exclusion registry: invalid sessions")
    return set(sessions)


def _validate_frame_indices(value: Any, where: str, minimum: int) -> list[int]:
    _require(isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value), f"{where}: invalid frames")
    frames = sorted(value)
    _require(len(frames) == len(set(frames)), f"{where}: duplicate frames")
    _require(len(frames) >= minimum and frames and frames[-1] - frames[0] + 1 == len(frames), f"{where}: not a contiguous minimum window")
    return frames


def _load_events(ledger: dict[str, Any], contract: dict[str, Any], excluded: set[str]) -> dict[str, dict[str, Any]]:
    _require(ledger.get("schema_version") == EVENT_LEDGER_SCHEMA, "event ledger: schema mismatch")
    _require(ledger.get("protocol_id") == PROTOCOL_ID, "event ledger: protocol mismatch")
    _require(ledger.get("cohort_role") == contract["cohort_role"], "event ledger: cohort role mismatch")
    items = ledger.get("items")
    _require(isinstance(items, list), "event ledger: items missing")
    events: dict[str, dict[str, Any]] = {}
    sessions: set[str] = set()
    for index, item in enumerate(items):
        where = f"event ledger item {index}"
        _require(isinstance(item, dict), f"{where}: expected object")
        event_id, session_id = item.get("event_id"), item.get("source_session_id")
        _require(isinstance(event_id, str) and event_id, f"{where}: invalid event_id")
        _require(isinstance(session_id, str) and session_id, f"{where}: invalid source_session_id")
        discovery_arm = item.get("discovery_arm")
        _require(discovery_arm in DISCOVERY_ARMS, f"{where}: invalid discovery_arm")
        _require(event_id not in events and session_id not in sessions, f"{where}: event/session is not independent")
        _require(session_id not in excluded, f"{where}: excluded source session")
        frames = _validate_frame_indices(item.get("frame_indices"), where, contract["minimum_contiguous_frames"])
        timestamps = item.get("frame_timestamps_ms")
        _require(isinstance(timestamps, list) and len(timestamps) == len(frames), f"{where}: timestamp coverage mismatch")
        _require(all(isinstance(value, int) and not isinstance(value, bool) for value in timestamps), f"{where}: invalid timestamps")
        _require(all(left < right for left, right in zip(timestamps, timestamps[1:])), f"{where}: timestamps not increasing")
        _require(timestamps[-1] - timestamps[0] >= contract["minimum_event_duration_ms"], f"{where}: event duration below time gate")
        categories = item.get("coverage")
        coverage_status = item.get("coverage_status", "FROZEN")
        _require(coverage_status in {"FROZEN", "UNCLASSIFIED_PILOT_PENDING"}, f"{where}: invalid coverage_status")
        if contract["mode"] == "CALIBRATION_BURNED" and coverage_status == "UNCLASSIFIED_PILOT_PENDING":
            _require(isinstance(categories, list) and not categories, f"{where}: unclassified pilot coverage must remain empty")
        else:
            _require(isinstance(categories, list) and categories and set(categories) <= set(SCENARIO_CATEGORIES), f"{where}: invalid coverage")
        _require("insufficient_evidence" not in categories or len(categories) == 1, f"{where}: UNKNOWN category cannot fill another category")
        condition = item.get("physical_condition")
        _require(condition in PHYSICAL_CONDITIONS, f"{where}: invalid physical condition")
        provenance = item.get("label_provenance")
        _require(isinstance(provenance, dict), f"{where}: missing label provenance")
        for field in ("truth_constructible_without_yolo", "yolo_boxes_used", "oracle_outputs_used", "model_outputs_visible"):
            _require(isinstance(provenance.get(field), bool), f"{where}: invalid provenance {field}")
        evidence = item.get("evidence_sufficiency", "SUFFICIENT")
        _require(evidence in {"SUFFICIENT", "INSUFFICIENT", "UNKNOWN"}, f"{where}: invalid evidence sufficiency")
        events[event_id] = {**item, "discovery_arm": discovery_arm, "frame_indices": frames, "frame_timestamps_ms": timestamps, "coverage": sorted(set(categories)), "coverage_status": coverage_status}
        sessions.add(session_id)
    _require(events, "event ledger: empty")
    return events


def _opaque_id(value: Any, forbidden_tokens: Iterable[str], where: str) -> str:
    _require(isinstance(value, str) and value, f"{where}: invalid opaque review id")
    lowered = value.lower()
    _require(not any(token.lower() in lowered for token in forbidden_tokens), f"{where}: review id leaks semantic metadata")
    return value


def _load_review_map(value: dict[str, Any], events: dict[str, dict[str, Any]], contract: dict[str, Any]) -> dict[str, str]:
    _require(value.get("schema_version") == REVIEW_MAP_SCHEMA, "review map: schema mismatch")
    _require(value.get("protocol_id") == PROTOCOL_ID, "review map: protocol mismatch")
    items = value.get("items")
    _require(isinstance(items, list), "review map: items missing")
    result: dict[str, str] = {}
    for index, item in enumerate(items):
        where = f"review map item {index}"
        _require(isinstance(item, dict), f"{where}: expected object")
        opaque = _opaque_id(item.get("review_item_id"), contract["metadata_blind_forbidden_tokens"], where)
        event_id = item.get("parent_event_id")
        _require(event_id in events and opaque not in result, f"{where}: invalid/duplicate parent mapping")
        result[opaque] = event_id
    _require(set(result.values()) == set(events), "review map: event coverage mismatch")
    return result


def _derive_observation(observation: dict[str, Any]) -> dict[str, str]:
    unresolved = (
        observation["visibility"] != "EVALUABLE"
        or observation["path_relation"] == "AMBIGUOUS"
        or observation["phase"] == "UNKNOWN"
        or observation["route_certainty"] != "SINGLE_PLAUSIBLE_ROUTE"
        or observation["evidence_quality"] != "CLEAR"
    )
    if unresolved:
        actionability = "UNKNOWN"
    elif observation["path_relation"] == "BLOCKING_PATH" and observation["phase"] == "CURRENT_INTRUSION":
        actionability = "YES"
    elif observation["path_relation"] == "NON_BLOCKING_PATH" or observation["phase"] in {"BEFORE_INTRUSION", "PASSED_CLEAR"}:
        actionability = "NO"
    else:
        actionability = "UNKNOWN"
    knownness = "KNOWN" if not unresolved else "UNKNOWN"
    return {"actionability": actionability, "knownness": knownness}


def _phase_durations(event: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, int]:
    """Return the longest contiguous timestamp span for each declared phase."""
    timestamps = dict(zip(event["frame_indices"], event["frame_timestamps_ms"]))
    by_frame = {observation["frame_index"]: observation for observation in observations}
    phases = ("BEFORE_INTRUSION", "CURRENT_INTRUSION", "PASSED_CLEAR")
    durations: dict[str, int] = {}
    for phase in phases:
        longest = 0
        run_start: int | None = None
        previous_frame: int | None = None
        for frame in event["frame_indices"]:
            if by_frame[frame]["phase"] != phase:
                run_start = None
                previous_frame = None
                continue
            if run_start is None or previous_frame is None or frame != previous_frame + 1:
                run_start = timestamps[frame]
            longest = max(longest, timestamps[frame] - run_start)
            previous_frame = frame
        durations[phase] = longest
    return durations


def _phase_ordered(observations: list[dict[str, Any]]) -> bool:
    order = {"BEFORE_INTRUSION": 0, "CURRENT_INTRUSION": 1, "PASSED_CLEAR": 2}
    sequence = [order[observation["phase"]] for observation in observations if observation["phase"] in order]
    return all(left <= right for left, right in zip(sequence, sequence[1:]))


def _derive_event_summary(event: dict[str, Any], observations: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(observations, key=lambda item: item["frame_index"])
    derived = [{**observation, **_derive_observation(observation)} for observation in ordered]
    yes_frames = [item["frame_index"] for item in derived if item["actionability"] == "YES"]
    unknown_present = any(item["actionability"] == "UNKNOWN" for item in derived)
    event_actionability = "UNKNOWN" if unknown_present else ("YES" if yes_frames else "NO")
    onset = min(yes_frames) if yes_frames else None
    passed_frames = [item["frame_index"] for item in derived if onset is not None and item["frame_index"] > onset and item["phase"] == "PASSED_CLEAR"]
    clear_frame = min(passed_frames) if passed_frames else None
    if onset is None:
        event_cleared = "UNKNOWN" if unknown_present else "NO"
    elif unknown_present:
        event_cleared = "UNKNOWN"
    else:
        event_cleared = "YES" if clear_frame is not None else "NO"
    event_knownness = "KNOWN" if all(item["knownness"] == "KNOWN" for item in derived) else "UNKNOWN"
    durations = _phase_durations(event, ordered)
    time_gate = {
        "event_duration_ms": event["frame_timestamps_ms"][-1] - event["frame_timestamps_ms"][0],
        "phase_durations_ms": durations,
        "phase_ordered": _phase_ordered(ordered),
        "passed": event["frame_timestamps_ms"][-1] - event["frame_timestamps_ms"][0] >= contract["minimum_event_duration_ms"],
    }
    if yes_frames:
        time_gate["passed"] = time_gate["passed"] and time_gate["phase_ordered"] and durations["BEFORE_INTRUSION"] >= contract["minimum_pre_alert_duration_ms"] and durations["CURRENT_INTRUSION"] >= contract["minimum_active_duration_ms"] and durations["PASSED_CLEAR"] >= contract["minimum_passed_clear_duration_ms"]
    return {
        "event_reminder_now": event_actionability,
        "event_cleared": event_cleared,
        "knownness": event_knownness,
        "alert_onset_frame": onset,
        "clear_frame": clear_frame,
        "observations": derived,
        "time_gate": time_gate,
    }


def _validate_observation(observation: Any, event: dict[str, Any], where: str) -> dict[str, Any]:
    _require(isinstance(observation, dict), f"{where}: expected object")
    allowed = {"frame_index", *PRIMITIVE_FIELDS}
    _require(set(observation) == allowed, f"{where}: direct action or metadata field is forbidden")
    frame = observation.get("frame_index")
    _require(frame in event["frame_indices"], f"{where}: frame outside event")
    for field in PRIMITIVE_FIELDS:
        _require(observation.get(field) in PRIMITIVE_VALUES[field], f"{where}: invalid primitive {field}")
    return observation


def _review_items(
    reviews: Iterable[dict[str, Any]],
    events: dict[str, dict[str, Any]],
    review_map: dict[str, str],
    contract: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_role: dict[str, dict[str, dict[str, Any]]] = {}
    required_false = (
        "other_review_visible_before_submission",
        "model_output_visible",
        "candidate_metadata_visible",
        "selection_reason_visible",
        "semantic_bucket_visible",
        "source_session_visible",
    )
    for review in reviews:
        _require(review.get("schema_version") == REVIEW_SCHEMA, "review: schema mismatch")
        _require(review.get("protocol_id") == PROTOCOL_ID, "review: protocol mismatch")
        role, view = review.get("reviewer_role"), review.get("view")
        _require(isinstance(role, str) and role, "review: invalid reviewer role")
        _require(view in {"CAUSAL", "RETROSPECTIVE"}, f"review {role}: invalid view")
        if view == "CAUSAL":
            _require(role.startswith("CAUSAL_"), f"review {role}: causal role prefix mismatch")
            _require(review.get("future_frame_access") is False, f"review {role}: causal review has future-frame access")
        else:
            _require(role.startswith("RETROSPECTIVE_"), f"review {role}: retrospective role prefix mismatch")
            _require(review.get("future_frame_access") is True, f"review {role}: retrospective view is not hindsight")
        _require(review.get("sealed_before_pair_selection") is True, f"review {role}: review was not sealed before pair selection")
        _require(review.get("isolated_context") is True and review.get("metadata_blind") is True, f"review {role}: not isolated/metadata blind")
        _require(review.get("primitive_policy_version") == PRIMITIVE_POLICY_VERSION, f"review {role}: primitive policy version mismatch")
        _require(review.get("visibility_evidence_window") == VISIBILITY_EVIDENCE_WINDOW, f"review {role}: visibility evidence window mismatch")
        _require(review.get("field_evidence_windows") == PRIMITIVE_OBSERVATION_POLICY["field_evidence_windows"], f"review {role}: field evidence windows mismatch")
        expected_temporal_window = CAUSAL_TEMPORAL_EVIDENCE_WINDOW if view == "CAUSAL" else RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW
        _require(review.get("temporal_fields_evidence_window") == expected_temporal_window, f"review {role}: temporal evidence window mismatch")
        for field in required_false:
            _require(review.get(field) is False, f"review {role}: {field} must be false")
        _require(role not in by_role, f"review: duplicate role {role}")
        items = review.get("items")
        _require(isinstance(items, list), f"review {role}: items missing")
        indexed: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(items):
            where = f"review {role} item {index}"
            _require(isinstance(item, dict) and set(item) == {"review_item_id", "primitive_observations"}, f"{where}: direct action or metadata field is forbidden")
            opaque = _opaque_id(item.get("review_item_id"), contract["metadata_blind_forbidden_tokens"], where)
            event_id = review_map.get(opaque)
            _require(event_id in events and opaque not in indexed, f"{where}: invalid/duplicate review item")
            observations = item.get("primitive_observations")
            _require(isinstance(observations, list) and len(observations) == len(events[event_id]["frame_indices"]), f"{where}: primitive frame coverage mismatch")
            checked = [_validate_observation(value, events[event_id], f"{where} frame {obs_index}") for obs_index, value in enumerate(observations)]
            _require({value["frame_index"] for value in checked} == set(events[event_id]["frame_indices"]), f"{where}: primitive frame identity mismatch")
            indexed[event_id] = {"review_item_id": opaque, "summary": _derive_event_summary(events[event_id], checked, contract)}
        _require(set(indexed) == set(events), f"review {role}: event coverage mismatch")
        by_role[role] = indexed
    return by_role


def _causal_roles(by_role: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    return [role for role in by_role if role.startswith("CAUSAL_")]


def _review_bundle_sha256(reviews: Iterable[dict[str, Any]]) -> str:
    ordered = sorted(reviews, key=lambda review: review.get("reviewer_role", ""))
    return sha256_json(ordered)


def _summary_sequence(summary: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(summary.get(field) for field in ("event_reminder_now", "event_cleared", "knownness", "alert_onset_frame", "clear_frame"))


def _primitive_at(summary: dict[str, Any], frame: int) -> dict[str, str]:
    for observation in summary["observations"]:
        if observation["frame_index"] == frame:
            return {field: observation[field] for field in PRIMITIVE_FIELDS} | {"actionability": observation["actionability"]}
    raise ValueError(f"frame {frame} absent from primitive summary")


def _unknown_summary(summary: dict[str, Any]) -> bool:
    return any(summary.get(field) == "UNKNOWN" for field in ("event_reminder_now", "event_cleared", "knownness"))


def _pairwise_review_metrics(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], tolerance: int) -> dict[str, Any]:
    event_ids = sorted(left)
    exact = sum(_summary_sequence(left[event_id]["summary"]) == _summary_sequence(right[event_id]["summary"]) for event_id in event_ids)
    boundary_values: list[bool] = []
    primitive_matches: Counter[str] = Counter()
    primitive_totals: Counter[str] = Counter()
    primitive_disagreements: Counter[str] = Counter()
    primitive_pairs: dict[str, list[tuple[str, str]]] = {field: [] for field in PRIMITIVE_FIELDS}
    derived_actionability_matches = 0
    derived_actionability_total = 0
    derived_actionability_disagreements = 0
    same_primitive_total = 0
    same_primitive_derived_matches = 0
    derived_clearance_matches = 0
    derived_knownness_matches = 0
    for event_id in event_ids:
        left_summary, right_summary = left[event_id]["summary"], right[event_id]["summary"]
        derived_clearance_matches += left_summary["event_cleared"] == right_summary["event_cleared"]
        derived_knownness_matches += left_summary["knownness"] == right_summary["knownness"]
        for field in ("alert_onset_frame", "clear_frame"):
            left_boundary, right_boundary = left_summary[field], right_summary[field]
            if left_boundary is None or right_boundary is None:
                boundary_values.append(left_boundary is None and right_boundary is None)
            else:
                boundary_values.append(abs(left_boundary - right_boundary) <= tolerance)
        for left_observation, right_observation in zip(left_summary["observations"], right_summary["observations"]):
            primitive_equal = True
            for field in PRIMITIVE_FIELDS:
                primitive_totals[field] += 1
                matches = left_observation[field] == right_observation[field]
                primitive_matches[field] += matches
                primitive_equal = primitive_equal and matches
                primitive_pairs[field].append((left_observation[field], right_observation[field]))
                if not matches:
                    primitive_disagreements[field] += 1
            actionability_matches = left_observation["actionability"] == right_observation["actionability"]
            derived_actionability_total += 1
            derived_actionability_matches += actionability_matches
            derived_actionability_disagreements += not actionability_matches
            if primitive_equal:
                same_primitive_total += 1
                same_primitive_derived_matches += actionability_matches
    primitive_agreement = {field: primitive_matches[field] / primitive_totals[field] if primitive_totals[field] else None for field in PRIMITIVE_FIELDS}
    primitive_classwise_agreement: dict[str, dict[str, Any]] = {}
    for field, pairs in primitive_pairs.items():
        primitive_classwise_agreement[field] = {}
        for value in sorted(PRIMITIVE_VALUES[field]):
            agreement_count = sum((left_value == value) == (right_value == value) for left_value, right_value in pairs)
            primitive_classwise_agreement[field][value] = {
                "agreement": agreement_count / len(pairs) if pairs else None,
                "left_count": sum(left_value == value for left_value, _ in pairs),
                "right_count": sum(right_value == value for _, right_value in pairs),
                "both_count": sum(left_value == value and right_value == value for left_value, right_value in pairs),
            }
    left_unknown_count = sum(_unknown_summary(left[event_id]["summary"]) for event_id in event_ids)
    right_unknown_count = sum(_unknown_summary(right[event_id]["summary"]) for event_id in event_ids)
    unknown_intersection = sum(_unknown_summary(left[event_id]["summary"]) and _unknown_summary(right[event_id]["summary"]) for event_id in event_ids)
    unknown_union = sum(_unknown_summary(left[event_id]["summary"]) or _unknown_summary(right[event_id]["summary"]) for event_id in event_ids)
    return {
        "event_count": len(event_ids),
        "event_exact_agreement": exact / len(event_ids),
        "boundary_consistency": sum(boundary_values) / len(boundary_values) if boundary_values else None,
        "boundary_comparison_count": len(boundary_values),
        "primitive_agreement": primitive_agreement,
        "primitive_classwise_agreement": primitive_classwise_agreement,
        "primitive_disagreement_counts": dict(sorted(primitive_disagreements.items())),
        "derived_actionability_agreement": derived_actionability_matches / derived_actionability_total if derived_actionability_total else None,
        "derived_actionability_disagreement_count": derived_actionability_disagreements,
        "derived_clearance_agreement": derived_clearance_matches / len(event_ids),
        "derived_knownness_agreement": derived_knownness_matches / len(event_ids),
        "primitive_disagreement_to_actionability_propagation_rate": derived_actionability_disagreements / sum(primitive_disagreements.values()) if sum(primitive_disagreements.values()) else 0.0,
        "same_primitive_to_derived_determinism_rate": same_primitive_derived_matches / same_primitive_total if same_primitive_total else None,
        "unknown_event_rate": unknown_union / len(event_ids),
        "unknown_event_union_count": unknown_union,
        "unknown_event_intersection_count": unknown_intersection,
        "left_unknown_event_rate": left_unknown_count / len(event_ids),
        "right_unknown_event_rate": right_unknown_count / len(event_ids),
        "unknown_event_intersection_rate": unknown_intersection / len(event_ids),
    }


def _test_yolo_free_truth(events: dict[str, dict[str, Any]], by_role: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    causal = _causal_roles(by_role)
    failures: list[str] = []
    if len(causal) < 2:
        return {"status": "NOT_EVALUABLE", "reason": "two independent primitive causal reviews are required"}
    insufficient_unknown = 0
    time_failures: list[str] = []
    for event_id, event in events.items():
        provenance = event["label_provenance"]
        if provenance["truth_constructible_without_yolo"] is not True or provenance["yolo_boxes_used"] is not False:
            failures.append(f"{event_id}: primitive truth provenance uses or depends on YOLO")
        if provenance["oracle_outputs_used"] or provenance["model_outputs_visible"]:
            failures.append(f"{event_id}: primitive review is output-contaminated")
        summaries = [by_role[role][event_id]["summary"] for role in causal]
        if event["evidence_sufficiency"] == "INSUFFICIENT":
            if all(summary["event_reminder_now"] == "UNKNOWN" for summary in summaries):
                insufficient_unknown += 1
            else:
                failures.append(f"{event_id}: insufficient evidence was not derived as UNKNOWN")
        for summary in summaries:
            if not summary["time_gate"]["passed"]:
                time_failures.append(event_id)
    failures.extend(f"{event_id}: positive event phase-duration contract failed" for event_id in sorted(set(time_failures)))
    return {
        "status": "FAIL" if failures else "PASS",
        "event_count": len(events),
        "insufficient_evidence_unknown_events": insufficient_unknown,
        "positive_time_coverage_failures": sorted(set(time_failures)),
        "failures": failures,
        "interpretation": "Primitive truth is output-blind and actionability is derived by the frozen rule." if not failures else "Primitive truth or its time/YOLO-free boundary is not established.",
    }


def _load_pairs(value: dict[str, Any], events: dict[str, dict[str, Any]], reviews: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    _require(value.get("schema_version") == PAIR_SCHEMA, "counterfactual pairs: schema mismatch")
    _require(value.get("protocol_id") == PROTOCOL_ID, "counterfactual pairs: protocol mismatch")
    _require(set(value) == {"schema_version", "protocol_id", "pair_builder", "items"}, "counterfactual pairs: pair-builder metadata missing or extra")
    builder = value.get("pair_builder")
    required_builder = {
        "stage",
        "review_bundle_sealed",
        "review_bundle_sha256",
        "yolo_role",
        "yolo_visible_to_reviewers",
        "yolo_used_for_truth",
        "primitive_labels_visible_to_pair_builder",
        "derived_labels_visible_to_pair_builder",
        "reviewed_event_phase_visible_to_pair_builder",
        "reviewed_motion_relation_visible_to_pair_builder",
        "pair_freeze_rule_version",
        "selection_fields",
        "selection_time_slot_source",
        "ordering_rule",
        "candidate_pair_universe_sha256",
        "candidate_pair_universe_count",
        "eligible_pair_count_before_label_access",
        "pair_count_frozen",
        "below_minimum_terminal",
    }
    _require(isinstance(builder, dict) and set(builder) == required_builder, "counterfactual pairs: incomplete pair-builder contract")
    pair_policy = contract["counterfactual_pair_policy"]
    _require(builder["stage"] == pair_policy["selection_stage"], "counterfactual pairs: pair selection occurred before review sealing")
    _require(builder["review_bundle_sealed"] is True, "counterfactual pairs: review bundle is not sealed")
    _require(builder["review_bundle_sha256"] == _review_bundle_sha256(reviews), "counterfactual pairs: sealed review bundle hash mismatch")
    _require(builder["yolo_role"] == pair_policy["yolo_role"], "counterfactual pairs: YOLO role is not selection-only")
    for field in ("yolo_visible_to_reviewers", "yolo_used_for_truth", "primitive_labels_visible_to_pair_builder", "derived_labels_visible_to_pair_builder", "reviewed_event_phase_visible_to_pair_builder", "reviewed_motion_relation_visible_to_pair_builder"):
        _require(builder[field] is False, f"counterfactual pairs: forbidden visibility/role flag {field}")
    _require(builder["pair_freeze_rule_version"] == pair_policy["version"], "counterfactual pairs: selection rule is not frozen")
    _require(builder["selection_fields"] == pair_policy["selection_fields"], "counterfactual pairs: selection fields are not frozen")
    _require(builder["selection_time_slot_source"] == pair_policy["selection_time_slot_source"], "counterfactual pairs: time-slot source is not frozen")
    _require(builder["ordering_rule"] == pair_policy["ordering_rule"], "counterfactual pairs: ordering rule is not frozen")
    universe_hash = builder["candidate_pair_universe_sha256"]
    _require(isinstance(universe_hash, str) and len(universe_hash) == 64 and all(char in "0123456789abcdef" for char in universe_hash.lower()), "counterfactual pairs: candidate universe hash is invalid")
    _require(isinstance(builder["candidate_pair_universe_count"], int) and not isinstance(builder["candidate_pair_universe_count"], bool) and builder["candidate_pair_universe_count"] >= 0, "counterfactual pairs: invalid candidate universe count")
    _require(isinstance(builder["eligible_pair_count_before_label_access"], int) and not isinstance(builder["eligible_pair_count_before_label_access"], bool) and builder["eligible_pair_count_before_label_access"] >= 0, "counterfactual pairs: invalid pre-label eligible pair count")
    _require(builder["candidate_pair_universe_count"] >= builder["eligible_pair_count_before_label_access"], "counterfactual pairs: eligible count exceeds candidate universe count")
    _require(builder["pair_count_frozen"] is True, "counterfactual pairs: pair count was not frozen before label access")
    _require(builder["below_minimum_terminal"] == pair_policy["below_minimum_terminal"], "counterfactual pairs: below-minimum terminal must be NOT_EVALUABLE")
    items = value.get("items")
    _require(isinstance(items, list), "counterfactual pairs: items missing")
    _require(builder["eligible_pair_count_before_label_access"] >= len(items), "counterfactual pairs: pre-label universe count is below selected pair count")
    seen: set[str] = set()
    required = {"pair_id", "pair_rank", "event_a_id", "event_b_id", "yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity", "selection_time_slot", "comparison_frame_index_a", "comparison_frame_index_b"}
    previous_rank = 0
    for index, item in enumerate(items):
        where = f"counterfactual pair {index}"
        _require(isinstance(item, dict) and set(item) == required, f"{where}: incomplete pair contract")
        pair_id = item["pair_id"]
        _require(isinstance(pair_id, str) and pair_id not in seen, f"{where}: invalid/duplicate pair_id")
        _require(isinstance(item["pair_rank"], int) and not isinstance(item["pair_rank"], bool) and item["pair_rank"] > previous_rank, f"{where}: pair ranks must be strictly increasing")
        previous_rank = item["pair_rank"]
        _require(item["event_a_id"] in events and item["event_b_id"] in events and item["event_a_id"] != item["event_b_id"], f"{where}: unknown/identical event")
        for field in ("yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity"):
            number = _number(item[field], f"{where}: {field}")
            _require(0 <= number <= 1, f"{where}: {field} outside [0,1]")
        _require(isinstance(item["selection_time_slot"], int) and not isinstance(item["selection_time_slot"], bool) and item["selection_time_slot"] >= 0, f"{where}: selection_time_slot must be a non-negative integer")
        for event_id in (item["event_a_id"], item["event_b_id"]):
            _require(item["selection_time_slot"] < len(events[event_id]["frame_indices"]), f"{where}: selection_time_slot outside event")
        for field, event_id in (("comparison_frame_index_a", item["event_a_id"]), ("comparison_frame_index_b", item["event_b_id"])):
            _require(item[field] in events[event_id]["frame_indices"], f"{where}: comparison frame outside event")
        _require(item["comparison_frame_index_a"] == events[item["event_a_id"]]["frame_indices"][item["selection_time_slot"]], f"{where}: event A comparison frame is not the frozen selection slot")
        _require(item["comparison_frame_index_b"] == events[item["event_b_id"]]["frame_indices"][item["selection_time_slot"]], f"{where}: event B comparison frame is not the frozen selection slot")
        seen.add(pair_id)
    return items


def _causal_frame_label(by_role: dict[str, dict[str, dict[str, Any]]], event_id: str, frame: int) -> str:
    values = []
    for role in _causal_roles(by_role):
        values.append(_primitive_at(by_role[role][event_id]["summary"], frame)["actionability"])
    if not values or any(value == "UNKNOWN" for value in values) or len(set(values)) != 1:
        return "UNKNOWN"
    return values[0]


def _causal_frame_primitive(by_role: dict[str, dict[str, dict[str, Any]]], event_id: str, frame: int, field: str) -> str:
    values = [_primitive_at(by_role[role][event_id]["summary"], frame)[field] for role in _causal_roles(by_role)]
    if not values or len(set(values)) != 1:
        return "UNKNOWN"
    return values[0]


def _causal_frame_path(by_role: dict[str, dict[str, dict[str, Any]]], event_id: str, frame: int) -> str:
    value = _causal_frame_primitive(by_role, event_id, frame, "path_relation")
    return value if value in {"BLOCKING_PATH", "NON_BLOCKING_PATH"} else "AMBIGUOUS"


def _test_physical_counterfactual(events: dict[str, dict[str, Any]], by_role: dict[str, dict[str, dict[str, Any]]], pairs: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    not_evaluable: list[dict[str, Any]] = []
    failures: list[str] = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        reasons: list[str] = []
        if pair["yolo_box_similarity"] < contract["yolo_box_similarity_threshold"]:
            reasons.append("box_similarity_below_threshold")
        if pair["distance_scale_similarity"] < contract["minimum_pair_distance_scale_similarity"]:
            reasons.append("distance_scale_mismatch")
        if pair["position_similarity"] < contract["minimum_pair_position_similarity"]:
            reasons.append("position_mismatch")
        if pair["visibility_similarity"] < contract["minimum_pair_visibility_similarity"]:
            reasons.append("visibility_mismatch")
        a_id, b_id = pair["event_a_id"], pair["event_b_id"]
        path_a = _causal_frame_path(by_role, a_id, pair["comparison_frame_index_a"])
        path_b = _causal_frame_path(by_role, b_id, pair["comparison_frame_index_b"])
        if {path_a, path_b} != {"BLOCKING_PATH", "NON_BLOCKING_PATH"}:
            reasons.append("primitive_path_relation_not_contrasted")
        phase_a = _causal_frame_primitive(by_role, a_id, pair["comparison_frame_index_a"], "phase")
        phase_b = _causal_frame_primitive(by_role, b_id, pair["comparison_frame_index_b"], "phase")
        if phase_a == "UNKNOWN" or phase_b == "UNKNOWN" or phase_a != phase_b:
            reasons.append("primitive_phase_mismatch")
        motion_a = _causal_frame_primitive(by_role, a_id, pair["comparison_frame_index_a"], "motion_relation")
        motion_b = _causal_frame_primitive(by_role, b_id, pair["comparison_frame_index_b"], "motion_relation")
        if motion_a == "UNKNOWN" or motion_b == "UNKNOWN" or motion_a != motion_b:
            reasons.append("primitive_motion_relation_mismatch")
        if reasons:
            not_evaluable.append({"pair_id": pair_id, "reasons": reasons})
            continue
        label_a = _causal_frame_label(by_role, a_id, pair["comparison_frame_index_a"])
        label_b = _causal_frame_label(by_role, b_id, pair["comparison_frame_index_b"])
        if "UNKNOWN" in {label_a, label_b}:
            not_evaluable.append({"pair_id": pair_id, "reasons": ["derived_actionability_unknown_or_disagreed"]})
            continue
        eligible.append({"pair_id": pair_id, "labels": [label_a, label_b], "similarity": pair["yolo_box_similarity"]})
        if label_a == label_b:
            failures.append(f"{pair_id}: predeclared physical-risk contrast received the same derived actionability")
    if failures:
        status = "FAIL"
    elif len(eligible) < contract["minimum_counterfactual_pairs"]:
        status = "NOT_EVALUABLE"
    elif len(eligible) > contract["maximum_counterfactual_pairs"]:
        status = "FAIL"
        failures.append("too_many_counterfactual_pairs_for_frozen_contract")
    else:
        status = "PASS"
    return {
        "status": status,
        "eligible_pair_count": len(eligible),
        "required_pair_range": [contract["minimum_counterfactual_pairs"], contract["maximum_counterfactual_pairs"]],
        "pairs": eligible,
        "not_evaluable_pairs": not_evaluable,
        "failures": failures,
        "interpretation": "Derived actionability distinguishes matched physical contrasts." if status == "PASS" else "The matched physical-risk contrast is not evaluable or was not distinguished.",
    }


def _oracle_metrics(value: Any, where: str) -> dict[str, int | float | None]:
    _require(isinstance(value, dict), f"{where}: metrics missing")
    result: dict[str, int | float | None] = {}
    for metric in ORACLE_METRICS:
        raw = value.get(metric)
        if metric == "response_delay_frames" and raw is None:
            result[metric] = None
        else:
            result[metric] = _number(raw, f"{where}: {metric}", integer=metric != "response_delay_frames")
    return result


def _compare_system_oracle(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    improvements: list[str] = []
    if candidate["positive_hits"] < baseline["positive_hits"]:
        problems.append("fewer_positive_hits")
    if candidate["critical_misses"] > baseline["critical_misses"]:
        problems.append("more_critical_misses")
    if candidate["false_alert_events"] > baseline["false_alert_events"]:
        problems.append("more_false_alert_events")
    if candidate["cleared_positives"] < baseline["cleared_positives"]:
        problems.append("fewer_cleared_positives")
    if candidate["response_delay_frames"] is not None and baseline["response_delay_frames"] is not None:
        if candidate["response_delay_frames"] > baseline["response_delay_frames"]:
            problems.append("later_response")
        elif candidate["response_delay_frames"] < baseline["response_delay_frames"]:
            improvements.append("earlier_response")
    for metric in ("positive_hits", "critical_misses", "false_alert_events", "cleared_positives"):
        if metric in {"positive_hits", "cleared_positives"} and candidate[metric] > baseline[metric]:
            improvements.append(f"improved_{metric}")
        if metric in {"critical_misses", "false_alert_events"} and candidate[metric] < baseline[metric]:
            improvements.append(f"improved_{metric}")
    return {"no_worse": not problems, "strict_improvement": bool(improvements), "improvements": improvements, "problems": problems}


def _load_oracle_opportunity(item: dict[str, Any], events: dict[str, dict[str, Any]], where: str, available: bool, required_flag: str) -> dict[str, Any]:
    required = set(ORACLE_OPPORTUNITY_FIELDS)
    _require(isinstance(item, dict) and set(item) == required, f"{where}: incomplete opportunity contract")
    event_ids = item["eligible_event_ids"]
    _require(isinstance(event_ids, list) and len(event_ids) == len(set(event_ids)) and all(event_id in events for event_id in event_ids), f"{where}: invalid eligible events")
    for field in ("eligible_for_native_task", "eligible_for_system_chain"):
        _require(isinstance(item[field], bool), f"{where}: {field} must be bool")
    _require(isinstance(item["required_inputs"], list) and all(isinstance(entry, str) and entry for entry in item["required_inputs"]), f"{where}: invalid required inputs")
    _require(isinstance(item["expected_improvement_dimension"], list) and all(isinstance(entry, str) and entry for entry in item["expected_improvement_dimension"]), f"{where}: invalid improvement dimensions")
    reason = item["not_evaluable_reason"]
    _require(reason is None or (isinstance(reason, str) and reason), f"{where}: invalid not_evaluable_reason")
    _require(item[required_flag] is available, f"{where}: {required_flag} does not match availability")
    if available:
        _require(event_ids and reason is None, f"{where}: available opportunity must have events and no not-evaluable reason")
        _require(item["required_inputs"] and item["expected_improvement_dimension"], f"{where}: available opportunity is incomplete")
    else:
        _require(not item[required_flag] and not event_ids and reason is not None, f"{where}: unavailable opportunity must be explicitly not evaluable")
    return item


def _load_oracle(value: dict[str, Any], contract: dict[str, Any], events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    _require(value.get("schema_version") == ORACLE_SCHEMA, "oracle manifest: schema mismatch")
    _require(value.get("protocol_id") == PROTOCOL_ID, "oracle manifest: protocol mismatch")
    _require(value.get("shared_execution") == contract["shared_execution"], "oracle manifest: shared execution mismatch")
    system_chain = value.get("system_chain")
    native = value.get("native_information_ceiling")
    _require(isinstance(system_chain, dict) and isinstance(native, dict), "oracle manifest: both evaluation paths are required")
    normalized: dict[str, Any] = {"system_chain": {}, "native_information_ceiling": {}}
    for arm in SYSTEM_ARMS:
        item = system_chain.get(arm)
        _require(isinstance(item, dict), f"oracle system chain: missing {arm}")
        _require(isinstance(item, dict) and isinstance(item.get("available", False), bool), f"oracle system chain: invalid {arm}")
        available = item["available"]
        normalized_item: dict[str, Any] = {"available": available, "reason": item.get("not_evaluable_reason")}
        if normalized_item["available"]:
            normalized_item["metrics"] = _oracle_metrics(item.get("metrics"), f"oracle system {arm}")
        if arm != "current_yolo" and normalized_item["available"]:
            opportunity = _load_oracle_opportunity(item.get("opportunity"), events, f"oracle system {arm}", available, "eligible_for_system_chain")
            normalized_item["opportunity"] = opportunity
        elif arm != "current_yolo":
            normalized_item["opportunity"] = _load_oracle_opportunity(item.get("opportunity"), events, f"oracle system {arm}", available, "eligible_for_system_chain")
        normalized["system_chain"][arm] = normalized_item
    for arm in ORACLE_ARMS:
        item = native.get(arm)
        _require(isinstance(item, dict) and isinstance(item.get("available", False), bool), f"oracle native path: invalid {arm}")
        available = item["available"]
        normalized_item = {"available": available, "reason": item.get("not_evaluable_reason")}
        opportunity = _load_oracle_opportunity(item.get("opportunity"), events, f"oracle native {arm}", available, "eligible_for_native_task")
        normalized_item["opportunity"] = opportunity
        if normalized_item["available"]:
            metrics = item.get("metrics")
            _require(isinstance(metrics, dict), f"oracle native {arm}: metrics missing")
            for metric in contract["native_information_ceiling_gates"][arm]:
                number = _number(metrics.get(metric), f"oracle native {arm}/{metric}")
                _require(0 <= number <= 1, f"oracle native {arm}/{metric}: outside [0,1]")
            normalized_item["metrics"] = metrics
        normalized["native_information_ceiling"][arm] = normalized_item
    return normalized


def _test_oracle_discrimination(oracle: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    baseline_item = oracle["system_chain"]["current_yolo"]
    if not baseline_item["available"]:
        return {"status": "NOT_EVALUABLE", "reason": "current YOLO metrics are missing"}
    baseline = baseline_item["metrics"]
    arms: dict[str, Any] = {}
    regressions: list[str] = []
    native_missing: list[str] = []
    native_failures: list[str] = []
    stack_ceiling: list[str] = []
    for arm in ORACLE_ARMS:
        system_item = oracle["system_chain"][arm]
        native_item = oracle["native_information_ceiling"][arm]
        if not system_item["available"] or not native_item["available"]:
            native_missing.append(arm)
            arms[arm] = {"status": "NOT_EVALUABLE", "system_available": system_item["available"], "native_available": native_item["available"], "reason": native_item.get("reason") or system_item.get("reason"), "system_opportunity": system_item.get("opportunity"), "native_opportunity": native_item.get("opportunity")}
            continue
        comparison = _compare_system_oracle(system_item["metrics"], baseline)
        opportunity = system_item["opportunity"]
        opportunity_count = len(opportunity["eligible_event_ids"])
        native_gates = contract["native_information_ceiling_gates"][arm]
        native_checks = {metric: native_item["metrics"][metric] >= threshold for metric, threshold in native_gates.items()}
        native_passed = all(native_checks.values())
        arm_status = "PASS"
        if not comparison["no_worse"]:
            regressions.append(arm)
            arm_status = "FAIL"
        elif not native_passed:
            native_failures.append(arm)
            arm_status = "NOT_EVALUABLE"
        elif opportunity_count > 0 and not comparison["strict_improvement"]:
            stack_ceiling.append(arm)
            arm_status = "FLAG_EVALUATION_STACK_CEILING_SUSPECTED"
        arms[arm] = {"status": arm_status, "system_chain": {**comparison, "opportunity": opportunity, "opportunity_event_count": opportunity_count}, "native_information_ceiling": {"passed": native_passed, "checks": native_checks, "metrics": native_item["metrics"], "opportunity": native_item["opportunity"]}}
    if regressions:
        status = "FAIL"
        interpretation = "A system-chain oracle regressed under unified alert metrics."
    elif native_missing or native_failures:
        status = "NOT_EVALUABLE"
        interpretation = "Native information ceiling or its declared opportunity is missing/failed; this is not negative evidence."
    elif stack_ceiling:
        status = "FLAG_EVALUATION_STACK_CEILING_SUSPECTED"
        interpretation = "Native information separates the physical task, but the system stack did not convert an eligible opportunity into improvement. Inspect adapter, kernel, event policy and metrics together."
    else:
        status = "PASS"
        interpretation = "The supplied system and native oracle paths are both evaluable and no worse/improved under their declared opportunity conditions."
    return {"status": status, "arms": arms, "regressions": regressions, "native_missing_or_failed": native_missing + native_failures, "stack_ceiling_suspected": stack_ceiling, "interpretation": interpretation}


def _test_blind_stability(by_role: dict[str, dict[str, dict[str, Any]]], contract: dict[str, Any]) -> dict[str, Any]:
    causal = _causal_roles(by_role)
    retrospective = [role for role in by_role if role.startswith("RETROSPECTIVE_")]
    if len(causal) < 2 or not retrospective:
        return {"status": "NOT_EVALUABLE", "reason": "two primitive causal and one retrospective isolated reviews are required"}
    causal_pairs = [{"left": left, "right": right, **_pairwise_review_metrics(by_role[left], by_role[right], contract["boundary_tolerance_frames"])} for left, right in combinations(causal, 2)]
    causal_retro_pairs = [{"left": left, "right": right, **_pairwise_review_metrics(by_role[left], by_role[right], contract["boundary_tolerance_frames"])} for left in causal for right in retrospective]
    event_consistency = sum(item["event_exact_agreement"] for item in causal_pairs) / len(causal_pairs)
    boundary_values = [item["boundary_consistency"] for item in causal_pairs if item["boundary_consistency"] is not None]
    boundary_consistency = sum(boundary_values) / len(boundary_values) if boundary_values else None
    primitive_field_consistency = {
        field: sum(item["primitive_agreement"][field] for item in causal_pairs if item["primitive_agreement"][field] is not None) / sum(item["primitive_agreement"][field] is not None for item in causal_pairs)
        for field in PRIMITIVE_FIELDS
    }
    primitive_values = [value for value in primitive_field_consistency.values() if value is not None]
    primitive_consistency = sum(primitive_values) / len(primitive_values) if primitive_values else None
    primitive_classwise_agreement = {
        field: {
            value: sum(item["primitive_classwise_agreement"][field][value]["agreement"] for item in causal_pairs) / len(causal_pairs)
            for value in sorted(PRIMITIVE_VALUES[field])
        }
        for field in PRIMITIVE_FIELDS
    }
    derived_actionability_consistency = sum(item["derived_actionability_agreement"] for item in causal_pairs) / len(causal_pairs)
    derived_clearance_consistency = sum(item["derived_clearance_agreement"] for item in causal_pairs) / len(causal_pairs)
    determinism_values = [item["same_primitive_to_derived_determinism_rate"] for item in causal_pairs if item["same_primitive_to_derived_determinism_rate"] is not None]
    primitive_to_derived_determinism = sum(determinism_values) / len(determinism_values) if determinism_values else None
    causal_retro_consistency = sum(item["event_exact_agreement"] for item in causal_retro_pairs) / len(causal_retro_pairs)
    causal_retro_actionability_consistency = sum(item["derived_actionability_agreement"] for item in causal_retro_pairs) / len(causal_retro_pairs)
    event_ids = sorted(next(iter(by_role.values())))
    unknown_event_rate = sum(any(_unknown_summary(by_role[role][event_id]["summary"]) for role in causal) for event_id in event_ids) / len(event_ids)
    retrospective_unknown_event_rate = sum(any(_unknown_summary(by_role[role][event_id]["summary"]) for role in retrospective) for event_id in event_ids) / len(event_ids)
    unknown_event_union_rate = sum(item["unknown_event_union_count"] for item in causal_pairs) / sum(item["event_count"] for item in causal_pairs)
    unknown_event_intersection_rate = sum(item["unknown_event_intersection_count"] for item in causal_pairs) / sum(item["event_count"] for item in causal_pairs)
    disagreement_source: Counter[str] = Counter()
    primitive_to_derived_disagreements = 0
    primitive_disagreement_total = 0
    for item in causal_pairs:
        disagreement_source.update(item["primitive_disagreement_counts"])
        primitive_to_derived_disagreements += item["derived_actionability_disagreement_count"]
        primitive_disagreement_total += sum(item["primitive_disagreement_counts"].values())
    causal_consensus_pairs: list[dict[str, Any]] = []
    unknown_to_known = 0
    reversals = 0
    resolvable_retro = 0
    for event_id in event_ids:
        causal_labels = [by_role[role][event_id]["summary"]["event_reminder_now"] for role in causal]
        retro_labels = [by_role[role][event_id]["summary"]["event_reminder_now"] for role in retrospective]
        if len(set(causal_labels)) == 1:
            consensus = causal_labels[0]
            retro_known = any(label != "UNKNOWN" for label in retro_labels)
            if consensus == "UNKNOWN" and retro_known:
                unknown_to_known += 1
            if consensus != "UNKNOWN" and retro_known:
                known_retro_labels = [label for label in retro_labels if label != "UNKNOWN"]
                resolvable_retro += len(known_retro_labels)
                reversals += sum(consensus != label for label in known_retro_labels)
            causal_consensus_pairs.append({"event_id": event_id, "causal_consensus": consensus, "retrospective_labels": retro_labels})
    failures: list[str] = []
    if event_consistency < contract["minimum_event_consistency"]:
        failures.append("event_consistency_below_gate")
    if boundary_consistency is None or boundary_consistency < contract["minimum_boundary_consistency"]:
        failures.append("boundary_consistency_below_gate")
    if primitive_consistency is None or primitive_consistency < contract["minimum_primitive_consistency"]:
        failures.append("primitive_consistency_below_gate")
    if any(value is None or value < contract["minimum_primitive_field_consistency"] for value in primitive_field_consistency.values()):
        failures.append("primitive_field_consistency_below_gate")
    if derived_actionability_consistency < contract["minimum_derived_actionability_consistency"]:
        failures.append("derived_actionability_consistency_below_gate")
    if derived_clearance_consistency < contract["minimum_derived_clearance_consistency"]:
        failures.append("derived_clearance_consistency_below_gate")
    if primitive_to_derived_determinism is None or primitive_to_derived_determinism < contract["minimum_primitive_to_derived_determinism"]:
        failures.append("primitive_to_derived_determinism_below_gate")
    if unknown_event_rate > contract["maximum_unknown_event_rate"]:
        failures.append("unknown_rate_above_gate")
    return {
        "status": "FAIL" if failures else "PASS",
        "causal_reviewer_count": len(causal),
        "retrospective_reviewer_count": len(retrospective),
        "event_consistency": event_consistency,
        "boundary_consistency": boundary_consistency,
        "primitive_consistency": primitive_consistency,
        "primitive_field_consistency": primitive_field_consistency,
        "primitive_classwise_agreement": primitive_classwise_agreement,
        "route_corridor_consistency": primitive_field_consistency.get("route_certainty"),
        "evidence_unknown_consistency": primitive_field_consistency.get("evidence_quality"),
        "derived_actionability_consistency": derived_actionability_consistency,
        "derived_clearance_consistency": derived_clearance_consistency,
        "primitive_to_derived_determinism": primitive_to_derived_determinism,
        "primitive_disagreement_source": dict(sorted(disagreement_source.items())),
        "primitive_disagreement_to_actionability_propagation_rate": primitive_to_derived_disagreements / primitive_disagreement_total if primitive_disagreement_total else 0.0,
        "causal_retrospective_consistency": causal_retro_consistency,
        "causal_retrospective_actionability_consistency": causal_retro_actionability_consistency,
        "retrospective_adjudicates_causal_truth": False,
        "causal_consensus_vs_retrospective": causal_consensus_pairs,
        "causal_unknown_to_retrospective_known_count": unknown_to_known,
        "causal_label_reversal_rate": reversals / resolvable_retro if resolvable_retro else None,
        "unknown_event_rate": unknown_event_rate,
        "unknown_event_union_rate": unknown_event_union_rate,
        "unknown_event_intersection_rate": unknown_event_intersection_rate,
        "retrospective_unknown_event_rate": retrospective_unknown_event_rate,
        "causal_pairwise": causal_pairs,
        "causal_vs_retrospective": causal_retro_pairs,
        "failures": failures,
    }


def _coverage(events: dict[str, dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(category for event in events.values() for category in event["coverage"])
    sources = {category: {events[event_id]["source_session_id"] for event_id, event in events.items() if category in event["coverage"]} for category in SCENARIO_CATEGORIES}
    missing_counts = [category for category in SCENARIO_CATEGORIES if counts[category] < contract["required_coverage_min_counts"][category]]
    missing_sources = [category for category in SCENARIO_CATEGORIES if len(sources[category]) < contract["required_coverage_min_source_sessions"][category]]
    session_counts = Counter(event["source_session_id"] for event in events.values())
    max_share = max(session_counts.values()) / len(events)
    discovery_counts = Counter(event["discovery_arm"] for event in events.values())
    distinct_discovery_arms = set(discovery_counts)
    discovery_policy = contract["discovery_arm_policy"]
    unclassified_event_count = sum(event.get("coverage_status") == "UNCLASSIFIED_PILOT_PENDING" for event in events.values())
    if contract["mode"] == "FORMAL_FROZEN":
        discovery_mix_passed = (
            len(distinct_discovery_arms) >= discovery_policy["minimum_distinct_arms_formal"]
            and len(distinct_discovery_arms & set(discovery_policy["independent_arms"])) >= discovery_policy["minimum_independent_arms_formal"]
            and discovery_policy["source_mask_arm"] in distinct_discovery_arms
        )
    else:
        discovery_mix_passed = len(distinct_discovery_arms) >= discovery_policy["minimum_distinct_arms_calibration"]
    coverage_not_formal = contract["mode"] == "CALIBRATION_BURNED" and unclassified_event_count > 0
    return {
        "counts": dict(sorted(counts.items())),
        "source_session_counts": {category: len(sources[category]) for category in SCENARIO_CATEGORIES},
        "unclassified_event_count": unclassified_event_count,
        "formal_category_coverage_established": not coverage_not_formal,
        "missing_count_categories": missing_counts,
        "missing_source_categories": missing_sources,
        "max_event_share_per_session": max_share,
        "discovery_arm_counts": dict(sorted(discovery_counts.items())),
        "distinct_discovery_arms": sorted(distinct_discovery_arms),
        "discovery_mix_passed": discovery_mix_passed,
        "passed": not missing_counts and not missing_sources and max_share <= contract["maximum_event_share_per_session"] and discovery_mix_passed,
    }


def run_judge_audit(
    contract: dict[str, Any],
    ledger: dict[str, Any],
    review_map: dict[str, Any],
    reviews: list[dict[str, Any]],
    pairs: dict[str, Any],
    oracle: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _contract(contract)
    events = _load_events(ledger, contract, _validate_registry(registry))
    indexed_map = _load_review_map(review_map, events, contract)
    by_role = _review_items(reviews, events, indexed_map, contract)
    loaded_pairs = _load_pairs(pairs, events, reviews, contract)
    loaded_oracle = _load_oracle(oracle, contract, events)
    coverage = _coverage(events, contract)
    test1 = _test_yolo_free_truth(events, by_role)
    test2 = _test_physical_counterfactual(events, by_role, loaded_pairs, contract)
    test3 = _test_oracle_discrimination(loaded_oracle, contract)
    test4 = _test_blind_stability(by_role, contract)
    count_ok = contract["minimum_events"] <= len(events) <= contract["maximum_events"]
    tests = {"test_1_primitive_yolo_free_truth": test1, "test_2_physical_counterfactual": test2, "test_3_oracle_two_path_discrimination": test3, "test_4_blind_review_stability": test4}
    if any(item["status"] == "FAIL" for item in tests.values()):
        status = "STOP_JUDGE_AUDIT_FAILED"
        next_action = "Preserve the primitive-level failing cases and repair the judge/evaluator contract; do not train or tune a model to erase the failure."
    elif not count_ok or not coverage["passed"]:
        status = "HOLD_JUDGE_AUDIT_COHORT"
        next_action = "Freeze the declared event/session count, time windows, per-category counts and per-category source-session counts before interpreting tests."
    elif any(item["status"] == "NOT_EVALUABLE" for item in tests.values()):
        status = "NOT_EVALUABLE_JUDGE_AUDIT_INPUTS"
        next_action = "Supply the missing primitive review, matched pair, or native oracle opportunity evidence; UNKNOWN and missing evidence remain unresolved."
    elif test3["status"] == "FLAG_EVALUATION_STACK_CEILING_SUSPECTED":
        status = "FLAG_EVALUATION_STACK_CEILING_SUSPECTED"
        next_action = "Inspect adapter, decision kernel, event policy and metrics together; do not call this a metric-only ceiling."
    elif contract["mode"] == "CALIBRATION_BURNED":
        status = "VALID_BURNED_CALIBRATION_PILOT"
        next_action = "Use only to repair the packet/primitive contract. Exclude every calibration event from the formal denominator."
    else:
        status = "VALID_JUDGE_AUDIT_CONSTRUCT"
        next_action = "Only now may the formal evaluator construct be used for later Development comparison; no product, safety, Android or default-App claim follows."
    return {
        "schema_version": REPORT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": contract["mode"],
        "cohort_role": contract["cohort_role"],
        "event_count": len(events),
        "source_session_count": len({item["source_session_id"] for item in events.values()}),
        "coverage": coverage,
        "tests": tests,
        "status": status,
        "next_action": next_action,
        "derived_actionability_rule_sha256": sha256_json(contract["derived_actionability_policy"]),
        "input_sha256": {
            "contract": sha256_json(contract),
            "event_ledger": sha256_json(ledger),
            "review_map": sha256_json(review_map),
            "pairs": sha256_json(pairs),
            "oracle": sha256_json(oracle),
            "reviews": [sha256_json(review) for review in reviews],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--event-ledger", type=Path, required=True)
    parser.add_argument("--review-map", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--counterfactual-pairs", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    contract = read_json(args.contract)
    ledger = read_json(args.event_ledger)
    review_map = read_json(args.review_map)
    reviews = [read_json(path) for path in args.reviews]
    pairs = read_json(args.counterfactual_pairs)
    oracle = read_json(args.oracle_manifest)
    registry = read_json(args.exclusion_registry) if args.exclusion_registry else None
    result = run_judge_audit(contract, ledger, review_map, reviews, pairs, oracle, registry)
    result["input_sha256"]["contract_file"] = sha256_file(args.contract)
    result["input_sha256"]["event_ledger_file"] = sha256_file(args.event_ledger)
    result["input_sha256"]["review_map_file"] = sha256_file(args.review_map)
    result["input_sha256"]["reviews_files"] = [sha256_file(path) for path in args.reviews]
    result["input_sha256"]["pairs_file"] = sha256_file(args.counterfactual_pairs)
    result["input_sha256"]["oracle_file"] = sha256_file(args.oracle_manifest)
    if args.exclusion_registry:
        result["input_sha256"]["exclusion_registry_file"] = sha256_file(args.exclusion_registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} events={result['event_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
