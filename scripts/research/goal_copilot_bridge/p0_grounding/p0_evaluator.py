"""Deterministic mechanics evaluator for P0 named-building entrance grounding.

This module evaluates mock or later protocol-conforming manifests. It does not
run a provider, model, tracker, Copilot Brain, or scientific cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "BA-P0-NAMED-BUILDING-ENTRANCE-GROUNDING-V1"
IOU_THRESHOLD = 0.5
RECALL_K = (1, 3, 5)
DECISION_STATUSES = {
    "GROUNDED",
    "AMBIGUOUS",
    "ABSTAIN_NO_RELIABLE_EVIDENCE",
    "INVALID_OBSERVATION",
}
EXPECTATIONS = {"MUST_GROUND", "MUST_BE_AMBIGUOUS", "MUST_ABSTAIN", "INVALID_OBSERVATION"}
PROVIDER_STATUSES = {"RUN_SUCCESS", "NOT_RUN", "RUN_FAILED", "INVALID_OUTPUT"}
EVIDENCE_VALIDITIES = {"VALID", "EXPIRED", "INVALID", "UNKNOWN"}
IDENTITY_SUPPORT = {"SUPPORTED", "INSUFFICIENT", "CONTRADICTED", "NOT_EVALUABLE"}
EVIDENCE_TYPES = {"ENTRANCE_STRUCTURE", "OPEN_VOCAB", "OCR_TEXT", "LOGO", "FACADE_RELATION", "POI_PRIOR"}
SOURCE_KINDS = {"MOCK", "RGB_PROVIDER", "OCR_PROVIDER", "VLM_PROVIDER", "MAP_PRIOR"}
ABSTENTION_REASONS = {
    "TARGET_NOT_VISIBLE",
    "NO_CANDIDATE",
    "AMBIGUOUS_CANDIDATES",
    "INSUFFICIENT_IDENTITY_EVIDENCE",
    "INSUFFICIENT_SPATIAL_EVIDENCE",
    "STALE_EVIDENCE",
    "INVALID_INPUT",
}

EPISODE_KEYS = {
    "schema_version",
    "episode_id",
    "goal_spec",
    "observation_window",
    "observation_valid",
    "target_visible",
    "target_instance_annotation",
    "acceptable_spatial_regions",
    "distractor_instances",
    "target_min_side_px",
    "visibility_fraction",
    "text_support",
    "scene_condition",
    "grounding_expectation",
}
OUTPUT_KEYS = {"schema_version", "episode_id", "provider_runs", "evidence", "candidates", "decision"}
REGION_KEYS = {"frame_id", "coordinate_space", "x_min", "y_min", "x_max", "y_max"}
PROVIDER_RUN_KEYS = {
    "provider_id", "status", "source_frame_ids", "evidence_ids", "candidate_ids", "failure_reason"
}
EVIDENCE_KEYS = {
    "provider_id",
    "evidence_id",
    "evidence_type",
    "source_frame_id",
    "source_timestamp_ms",
    "region_in_source_frame",
    "confidence",
    "validity",
    "expiry_timestamp_ms",
    "identity_claim",
    "provenance",
}
CANDIDATE_KEYS = {
    "candidate_id",
    "region",
    "category_label",
    "identity_hypothesis",
    "confidence",
    "provider_rank",
    "provider_ids",
    "evidence_ids",
}
DECISION_KEYS = {
    "status",
    "selected_candidate_id",
    "ranked_candidate_ids",
    "source_frame_id",
    "decision_timestamp_ms",
    "spatial_region",
    "goal_identity_support",
    "spatial_support",
    "confidence",
    "supporting_evidence_ids",
    "competing_candidate_ids",
    "abstention_reason",
    "persistence_handoff_token",
}
HANDOFF_KEYS = {"handoff_id", "candidate_id", "source_frame_id", "spatial_region", "evidence_ids"}
SCENE_KEYS = {
    "target_size", "visibility", "entrance_count", "same_class_distractor", "illumination", "view_angle"
}


class EpisodeContractError(ValueError):
    """Episode truth is malformed; evaluator authority cannot proceed."""


class OutputContractError(ValueError):
    """System output is malformed or breaks a fail-closed reference binding."""


def _require(condition: bool, message: str, error_type: type[ValueError]) -> None:
    if not condition:
        raise error_type(message)


def _exact_keys(value: Any, expected: set[str], path: str, error_type: type[ValueError]) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object", error_type)
    observed = set(value)
    _require(observed == expected, f"{path} keys differ: missing={sorted(expected-observed)} extra={sorted(observed-expected)}", error_type)
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _string(value: Any, path: str, error_type: type[ValueError]) -> str:
    _require(isinstance(value, str) and bool(value), f"{path} must be a non-empty string", error_type)
    return value


def _unique_strings(value: Any, path: str, error_type: type[ValueError], *, nonempty: bool = False) -> list[str]:
    _require(isinstance(value, list), f"{path} must be an array", error_type)
    _require(all(isinstance(item, str) and item for item in value), f"{path} must contain non-empty strings", error_type)
    _require(len(value) == len(set(value)), f"{path} must be unique", error_type)
    _require(not nonempty or bool(value), f"{path} must not be empty", error_type)
    return value


def _validate_region(
    value: Any,
    path: str,
    error_type: type[ValueError],
    allowed_frames: set[str] | None = None,
) -> Mapping[str, Any]:
    region = _exact_keys(value, REGION_KEYS, path, error_type)
    frame_id = _string(region["frame_id"], f"{path}.frame_id", error_type)
    _require(region["coordinate_space"] == "NORMALIZED_XYXY", f"{path} coordinate space must be NORMALIZED_XYXY", error_type)
    coordinates = [region[name] for name in ("x_min", "y_min", "x_max", "y_max")]
    _require(all(_is_number(item) and 0.0 <= float(item) <= 1.0 for item in coordinates), f"{path} coordinates must be finite normalized numbers", error_type)
    _require(float(region["x_min"]) < float(region["x_max"]), f"{path} x interval must have positive area", error_type)
    _require(float(region["y_min"]) < float(region["y_max"]), f"{path} y interval must have positive area", error_type)
    if allowed_frames is not None:
        _require(frame_id in allowed_frames, f"{path} references a frame outside the observation window", error_type)
    return region


def _same_region(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left[key] == right[key] for key in REGION_KEYS)


def region_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Return normalized XYXY IoU; regions on different frames never overlap."""
    if left["frame_id"] != right["frame_id"]:
        return 0.0
    x0 = max(float(left["x_min"]), float(right["x_min"]))
    y0 = max(float(left["y_min"]), float(right["y_min"]))
    x1 = min(float(left["x_max"]), float(right["x_max"]))
    y1 = min(float(left["y_max"]), float(right["y_max"]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = (float(left["x_max"]) - float(left["x_min"])) * (float(left["y_max"]) - float(left["y_min"]))
    right_area = (float(right["x_max"]) - float(right["x_min"])) * (float(right["y_max"]) - float(right["y_min"]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def validate_episode(value: Mapping[str, Any]) -> Mapping[str, Any]:
    episode = _exact_keys(value, EPISODE_KEYS, "episode", EpisodeContractError)
    _require(episode["schema_version"] == 1, "episode.schema_version must be 1", EpisodeContractError)
    _string(episode["episode_id"], "episode.episode_id", EpisodeContractError)

    goal = _exact_keys(episode["goal_spec"], {"goal_type", "target_name", "requested_relation"}, "episode.goal_spec", EpisodeContractError)
    _require(goal["goal_type"] == "NAMED_BUILDING_ENTRANCE", "goal_type drift", EpisodeContractError)
    _string(goal["target_name"], "episode.goal_spec.target_name", EpisodeContractError)
    _require(goal["requested_relation"] == "entrance_of", "requested_relation drift", EpisodeContractError)

    window = _exact_keys(episode["observation_window"], {"frame_ids", "start_timestamp_ms", "end_timestamp_ms"}, "episode.observation_window", EpisodeContractError)
    frame_ids = _unique_strings(window["frame_ids"], "episode.observation_window.frame_ids", EpisodeContractError, nonempty=True)
    _require(isinstance(window["start_timestamp_ms"], int) and window["start_timestamp_ms"] >= 0, "invalid start timestamp", EpisodeContractError)
    _require(isinstance(window["end_timestamp_ms"], int) and window["end_timestamp_ms"] >= window["start_timestamp_ms"], "invalid end timestamp", EpisodeContractError)
    frame_set = set(frame_ids)

    _require(type(episode["observation_valid"]) is bool, "observation_valid must be boolean", EpisodeContractError)
    _require(type(episode["target_visible"]) is bool, "target_visible must be boolean", EpisodeContractError)
    _require(episode["grounding_expectation"] in EXPECTATIONS, "unknown grounding expectation", EpisodeContractError)
    _require(episode["text_support"] in {"READABLE", "PARTIAL", "NONE", "NOT_APPLICABLE"}, "unknown text_support", EpisodeContractError)
    scene = _exact_keys(episode["scene_condition"], SCENE_KEYS, "episode.scene_condition", EpisodeContractError)
    _require(type(scene["same_class_distractor"]) is bool, "same_class_distractor must be boolean", EpisodeContractError)
    _require(scene["target_size"] in {"TINY", "SMALL", "MEDIUM", "LARGE", "ABSENT"}, "unknown target_size", EpisodeContractError)
    _require(scene["visibility"] in {"FULL", "PARTIAL", "HEAVY_OCCLUSION", "ABSENT"}, "unknown visibility", EpisodeContractError)
    _require(scene["entrance_count"] in {"SINGLE", "MULTIPLE"}, "unknown entrance_count", EpisodeContractError)
    _require(scene["illumination"] in {"DAYLIGHT", "LOW_LIGHT", "GLARE", "MIXED"}, "unknown illumination", EpisodeContractError)
    _require(scene["view_angle"] in {"FRONTAL", "OBLIQUE", "MIXED"}, "unknown view_angle", EpisodeContractError)

    _require(isinstance(episode["acceptable_spatial_regions"], list), "acceptable_spatial_regions must be an array", EpisodeContractError)
    for index, region in enumerate(episode["acceptable_spatial_regions"]):
        _validate_region(region, f"episode.acceptable_spatial_regions[{index}]", EpisodeContractError, frame_set)
    _require(isinstance(episode["distractor_instances"], list), "distractor_instances must be an array", EpisodeContractError)
    distractor_ids: set[str] = set()
    for index, item in enumerate(episode["distractor_instances"]):
        distractor = _exact_keys(item, {"distractor_instance_id", "semantic_role", "region"}, f"episode.distractor_instances[{index}]", EpisodeContractError)
        distractor_id = _string(distractor["distractor_instance_id"], f"episode.distractor_instances[{index}].distractor_instance_id", EpisodeContractError)
        _require(distractor_id not in distractor_ids, "duplicate distractor_instance_id", EpisodeContractError)
        distractor_ids.add(distractor_id)
        _require(distractor["semantic_role"] in {"OTHER_BUILDING_ENTRANCE", "NON_ENTRANCE_DOOR", "SAME_BUILDING_NON_TARGET_ENTRANCE"}, "unknown distractor semantic_role", EpisodeContractError)
        _validate_region(distractor["region"], f"episode.distractor_instances[{index}].region", EpisodeContractError, frame_set)

    if episode["target_visible"]:
        annotation = _exact_keys(episode["target_instance_annotation"], {"target_instance_id", "target_name", "relation", "regions"}, "episode.target_instance_annotation", EpisodeContractError)
        _string(annotation["target_instance_id"], "target_instance_id", EpisodeContractError)
        _require(annotation["target_name"] == goal["target_name"], "target annotation name differs from goal", EpisodeContractError)
        _require(annotation["relation"] == "entrance_of", "target annotation relation drift", EpisodeContractError)
        _require(isinstance(annotation["regions"], list) and bool(annotation["regions"]), "target annotation requires regions", EpisodeContractError)
        for index, region in enumerate(annotation["regions"]):
            _validate_region(region, f"episode.target_instance_annotation.regions[{index}]", EpisodeContractError, frame_set)
        _require(bool(episode["acceptable_spatial_regions"]), "visible target requires acceptable regions", EpisodeContractError)
        _require(_is_number(episode["target_min_side_px"]) and float(episode["target_min_side_px"]) > 0.0, "visible target requires target_min_side_px", EpisodeContractError)
        _require(_is_number(episode["visibility_fraction"]) and 0.0 <= float(episode["visibility_fraction"]) <= 1.0, "visible target requires visibility_fraction", EpisodeContractError)
    else:
        _require(episode["target_instance_annotation"] is None, "absent target annotation must be null", EpisodeContractError)
        _require(episode["acceptable_spatial_regions"] == [], "absent target cannot have acceptable regions", EpisodeContractError)
        _require(episode["target_min_side_px"] is None and episode["visibility_fraction"] is None, "absent target size/visibility must be null", EpisodeContractError)

    if not episode["observation_valid"]:
        _require(episode["grounding_expectation"] == "INVALID_OBSERVATION", "invalid observation must use INVALID_OBSERVATION expectation", EpisodeContractError)
    if episode["grounding_expectation"] == "MUST_GROUND":
        _require(episode["observation_valid"] and episode["target_visible"], "MUST_GROUND requires a valid visible target", EpisodeContractError)
    if episode["grounding_expectation"] == "MUST_BE_AMBIGUOUS":
        _require(episode["observation_valid"] and episode["target_visible"] and bool(episode["distractor_instances"]), "MUST_BE_AMBIGUOUS requires visible target and distractor", EpisodeContractError)
    return episode


def _evidence_is_current(item: Mapping[str, Any], decision_timestamp_ms: int) -> bool:
    if item["validity"] != "VALID":
        return False
    expiry = item["expiry_timestamp_ms"]
    return expiry is None or int(expiry) >= decision_timestamp_ms


def _identity_evidence_matches(item: Mapping[str, Any], target_name: str) -> bool:
    claim = item["identity_claim"]
    return claim["target_name"] == target_name and claim["relation"] == "entrance_of"


def validate_output(value: Mapping[str, Any], episode: Mapping[str, Any]) -> Mapping[str, Any]:
    output = _exact_keys(value, OUTPUT_KEYS, "output", OutputContractError)
    _require(output["schema_version"] == 1, "output.schema_version must be 1", OutputContractError)
    _require(output["episode_id"] == episode["episode_id"], "output episode_id mismatch", OutputContractError)
    frames = set(episode["observation_window"]["frame_ids"])

    _require(isinstance(output["provider_runs"], list), "provider_runs must be an array", OutputContractError)
    providers: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(output["provider_runs"]):
        run = _exact_keys(raw, PROVIDER_RUN_KEYS, f"output.provider_runs[{index}]", OutputContractError)
        provider_id = _string(run["provider_id"], f"provider_runs[{index}].provider_id", OutputContractError)
        _require(provider_id not in providers, "duplicate provider_id", OutputContractError)
        providers[provider_id] = run
        _require(run["status"] in PROVIDER_STATUSES, "unknown provider status", OutputContractError)
        source_frames = _unique_strings(run["source_frame_ids"], "provider source_frame_ids", OutputContractError)
        _require(set(source_frames) <= frames, "provider references frame outside window", OutputContractError)
        _unique_strings(run["evidence_ids"], "provider evidence_ids", OutputContractError)
        _unique_strings(run["candidate_ids"], "provider candidate_ids", OutputContractError)
        if run["status"] == "RUN_SUCCESS":
            _require(run["failure_reason"] is None, "successful provider cannot have failure_reason", OutputContractError)
        else:
            _string(run["failure_reason"], "failed/not-run provider failure_reason", OutputContractError)

    _require(isinstance(output["evidence"], list), "evidence must be an array", OutputContractError)
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(output["evidence"]):
        item = _exact_keys(raw, EVIDENCE_KEYS, f"output.evidence[{index}]", OutputContractError)
        evidence_id = _string(item["evidence_id"], f"evidence[{index}].evidence_id", OutputContractError)
        _require(evidence_id not in evidence_by_id, "duplicate evidence_id", OutputContractError)
        evidence_by_id[evidence_id] = item
        _require(item["evidence_type"] in EVIDENCE_TYPES, "unknown evidence_type", OutputContractError)
        provider_id = _string(item["provider_id"], f"evidence[{index}].provider_id", OutputContractError)
        _require(provider_id in providers, "evidence references unknown provider", OutputContractError)
        _require(providers[provider_id]["status"] == "RUN_SUCCESS", "evidence emitted by provider that did not succeed", OutputContractError)
        _require(evidence_id in providers[provider_id]["evidence_ids"], "provider run omits emitted evidence_id", OutputContractError)
        _require(item["source_frame_id"] in frames, "evidence references frame outside window", OutputContractError)
        _require(isinstance(item["source_timestamp_ms"], int) and item["source_timestamp_ms"] >= 0, "invalid evidence timestamp", OutputContractError)
        _require(
            episode["observation_window"]["start_timestamp_ms"]
            <= item["source_timestamp_ms"]
            <= episode["observation_window"]["end_timestamp_ms"],
            "evidence source timestamp is outside the observation window",
            OutputContractError,
        )
        if item["region_in_source_frame"] is not None:
            region = _validate_region(item["region_in_source_frame"], f"evidence[{index}].region", OutputContractError, frames)
            _require(region["frame_id"] == item["source_frame_id"], "evidence region/source frame drift", OutputContractError)
        _require(item["confidence"] is None or (_is_number(item["confidence"]) and 0.0 <= float(item["confidence"]) <= 1.0), "invalid evidence confidence", OutputContractError)
        _require(item["validity"] in EVIDENCE_VALIDITIES, "unknown evidence validity", OutputContractError)
        expiry = item["expiry_timestamp_ms"]
        _require(expiry is None or (isinstance(expiry, int) and expiry >= item["source_timestamp_ms"]), "invalid evidence expiry", OutputContractError)
        claim = _exact_keys(item["identity_claim"], {"target_name", "relation"}, "evidence.identity_claim", OutputContractError)
        _require(claim["target_name"] is None or isinstance(claim["target_name"], str), "identity target_name must be string or null", OutputContractError)
        _require(claim["relation"] in {"entrance_of", "none"}, "unknown identity relation", OutputContractError)
        provenance = _exact_keys(item["provenance"], {"implementation_id", "config_id", "source_kind"}, "evidence.provenance", OutputContractError)
        _string(provenance["implementation_id"], "provenance.implementation_id", OutputContractError)
        _string(provenance["config_id"], "provenance.config_id", OutputContractError)
        _require(provenance["source_kind"] in SOURCE_KINDS, "unknown provenance source_kind", OutputContractError)

    _require(isinstance(output["candidates"], list), "candidates must be an array", OutputContractError)
    candidates_by_id: dict[str, Mapping[str, Any]] = {}
    provider_ranks: set[int] = set()
    for index, raw in enumerate(output["candidates"]):
        candidate = _exact_keys(raw, CANDIDATE_KEYS, f"output.candidates[{index}]", OutputContractError)
        candidate_id = _string(candidate["candidate_id"], f"candidates[{index}].candidate_id", OutputContractError)
        _require(candidate_id not in candidates_by_id, "duplicate candidate_id", OutputContractError)
        candidates_by_id[candidate_id] = candidate
        _validate_region(candidate["region"], f"candidates[{index}].region", OutputContractError, frames)
        _string(candidate["category_label"], f"candidates[{index}].category_label", OutputContractError)
        _require(candidate["identity_hypothesis"] is None or isinstance(candidate["identity_hypothesis"], str), "candidate identity_hypothesis must be string or null", OutputContractError)
        _require(_is_number(candidate["confidence"]) and 0.0 <= float(candidate["confidence"]) <= 1.0, "invalid candidate confidence", OutputContractError)
        _require(isinstance(candidate["provider_rank"], int) and candidate["provider_rank"] >= 1, "invalid provider_rank", OutputContractError)
        _require(candidate["provider_rank"] not in provider_ranks, "provider_rank must be globally unique", OutputContractError)
        provider_ranks.add(candidate["provider_rank"])
        provider_ids = _unique_strings(candidate["provider_ids"], "candidate.provider_ids", OutputContractError, nonempty=True)
        evidence_ids = _unique_strings(candidate["evidence_ids"], "candidate.evidence_ids", OutputContractError)
        _require(set(provider_ids) <= set(providers), "candidate references unknown provider", OutputContractError)
        _require(all(providers[provider_id]["status"] == "RUN_SUCCESS" for provider_id in provider_ids), "candidate references unsuccessful provider", OutputContractError)
        _require(set(evidence_ids) <= set(evidence_by_id), "candidate references unknown evidence", OutputContractError)
        _require(all(evidence_by_id[evidence_id]["provider_id"] in provider_ids for evidence_id in evidence_ids), "candidate evidence/provider binding drift", OutputContractError)
        for provider_id in provider_ids:
            _require(candidate_id in providers[provider_id]["candidate_ids"], "provider run omits emitted candidate_id", OutputContractError)

    for provider_id, run in providers.items():
        _require(set(run["evidence_ids"]) <= set(evidence_by_id), f"provider {provider_id} references unknown evidence", OutputContractError)
        _require(set(run["candidate_ids"]) <= set(candidates_by_id), f"provider {provider_id} references unknown candidate", OutputContractError)

    decision = _exact_keys(output["decision"], DECISION_KEYS, "output.decision", OutputContractError)
    _require(decision["status"] in DECISION_STATUSES, "unknown decision status", OutputContractError)
    _require(isinstance(decision["decision_timestamp_ms"], int) and decision["decision_timestamp_ms"] >= 0, "invalid decision timestamp", OutputContractError)
    _require(decision["decision_timestamp_ms"] >= episode["observation_window"]["start_timestamp_ms"], "decision predates observation window", OutputContractError)
    ranked = _unique_strings(decision["ranked_candidate_ids"], "decision.ranked_candidate_ids", OutputContractError)
    _require(set(ranked) == set(candidates_by_id), "decision ranking must cover each candidate exactly once", OutputContractError)
    supporting = _unique_strings(decision["supporting_evidence_ids"], "decision.supporting_evidence_ids", OutputContractError)
    competing = _unique_strings(decision["competing_candidate_ids"], "decision.competing_candidate_ids", OutputContractError)
    _require(set(supporting) <= set(evidence_by_id), "decision references unknown supporting evidence", OutputContractError)
    _require(set(competing) <= set(candidates_by_id), "decision references unknown competing candidate", OutputContractError)
    _require(decision["goal_identity_support"] in IDENTITY_SUPPORT and decision["spatial_support"] in IDENTITY_SUPPORT, "unknown decision support state", OutputContractError)

    if decision["status"] == "GROUNDED":
        selected_id = _string(decision["selected_candidate_id"], "decision.selected_candidate_id", OutputContractError)
        _require(selected_id in candidates_by_id, "selected candidate does not exist", OutputContractError)
        selected = candidates_by_id[selected_id]
        source_frame = _string(decision["source_frame_id"], "decision.source_frame_id", OutputContractError)
        region = _validate_region(decision["spatial_region"], "decision.spatial_region", OutputContractError, frames)
        _require(source_frame == region["frame_id"] == selected["region"]["frame_id"], "selected source frame drift", OutputContractError)
        _require(_same_region(region, selected["region"]), "decision region must equal selected candidate region", OutputContractError)
        _require(decision["goal_identity_support"] == "SUPPORTED" and decision["spatial_support"] == "SUPPORTED", "GROUNDED requires identity and spatial support", OutputContractError)
        _require(_is_number(decision["confidence"]) and 0.0 <= float(decision["confidence"]) <= 1.0, "GROUNDED requires confidence", OutputContractError)
        _require(bool(supporting), "GROUNDED requires supporting evidence", OutputContractError)
        _require(set(supporting) <= set(selected["evidence_ids"]), "supporting evidence must bind selected candidate", OutputContractError)
        _require(selected_id not in competing, "selected candidate cannot compete with itself", OutputContractError)
        _require(decision["abstention_reason"] is None, "GROUNDED cannot have abstention reason", OutputContractError)
        handoff = _exact_keys(decision["persistence_handoff_token"], HANDOFF_KEYS, "decision.persistence_handoff_token", OutputContractError)
        _string(handoff["handoff_id"], "handoff.handoff_id", OutputContractError)
        _require(handoff["candidate_id"] == selected_id, "handoff candidate binding drift", OutputContractError)
        _require(handoff["source_frame_id"] == source_frame, "handoff source frame binding drift", OutputContractError)
        handoff_region = _validate_region(handoff["spatial_region"], "handoff.spatial_region", OutputContractError, frames)
        _require(_same_region(handoff_region, region), "handoff region binding drift", OutputContractError)
        handoff_evidence = _unique_strings(handoff["evidence_ids"], "handoff.evidence_ids", OutputContractError, nonempty=True)
        _require(handoff_evidence == supporting, "handoff evidence binding drift", OutputContractError)
    else:
        _require(decision["selected_candidate_id"] is None, "non-GROUNDED decision cannot select a candidate", OutputContractError)
        _require(decision["source_frame_id"] is None and decision["spatial_region"] is None, "non-GROUNDED decision cannot carry a region", OutputContractError)
        _require(decision["confidence"] is None, "non-GROUNDED decision confidence must be null", OutputContractError)
        _require(decision["persistence_handoff_token"] is None, "non-GROUNDED decision cannot carry handoff", OutputContractError)
        _require(not supporting, "non-GROUNDED decision cannot claim supporting evidence", OutputContractError)
        _require(decision["abstention_reason"] in ABSTENTION_REASONS, "non-GROUNDED decision requires a known abstention_reason", OutputContractError)
    return output


def _best_truth_iou(region: Mapping[str, Any], truth_regions: Sequence[Mapping[str, Any]]) -> float:
    return max((region_iou(region, truth) for truth in truth_regions), default=0.0)


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator if denominator else None}


def evaluate_episode(episode_value: Mapping[str, Any], output_value: Mapping[str, Any]) -> dict[str, Any]:
    episode = validate_episode(episode_value)
    try:
        output = validate_output(output_value, episode)
    except OutputContractError as error:
        return {
            "episode_id": episode["episode_id"],
            "protocol_id": PROTOCOL_ID,
            "observation_valid": episode["observation_valid"],
            "valid_system_output": False,
            "contract_error": str(error),
            "provider_availability": None,
            "brain_selection": None,
            "end_to_end": {"outcome": "INVALID_SYSTEM_OUTPUT", "success": False},
            "p1_handoff": {"required": False, "valid": False},
            "strata": _episode_strata(episode),
        }

    evidence_by_id = {item["evidence_id"]: item for item in output["evidence"]}
    candidates_by_id = {item["candidate_id"]: item for item in output["candidates"]}
    truth_regions = episode["acceptable_spatial_regions"]
    correct_candidates = [
        candidate for candidate in output["candidates"]
        if _best_truth_iou(candidate["region"], truth_regions) >= IOU_THRESHOLD
    ]
    correct_ids = {candidate["candidate_id"] for candidate in correct_candidates}
    correct_rank = min((int(candidate["provider_rank"]) for candidate in correct_candidates), default=None)
    decision_timestamp = output["decision"]["decision_timestamp_ms"]
    current_evidence = [item for item in output["evidence"] if _evidence_is_current(item, decision_timestamp)]
    identity_evidence_available = any(
        _identity_evidence_matches(item, episode["goal_spec"]["target_name"]) for item in current_evidence
    )
    provider_statuses = {run["provider_id"]: run["status"] for run in output["provider_runs"]}
    provider_failures = {run["status"] for run in output["provider_runs"] if run["status"] != "RUN_SUCCESS"}
    if any(run["status"] == "RUN_SUCCESS" and not run["candidate_ids"] for run in output["provider_runs"]):
        provider_failures.add("RUN_SUCCESS_NO_CANDIDATE")
    provider_layer = {
        "correct_candidate_available": bool(correct_candidates),
        "provider_recall_at_k": {str(k): correct_rank is not None and correct_rank <= k for k in RECALL_K},
        "correct_candidate_rank": correct_rank,
        "correct_candidate_provider_ids": sorted({provider_id for candidate in correct_candidates for provider_id in candidate["provider_ids"]}),
        "goal_identity_evidence_available": identity_evidence_available,
        "provider_run_statuses": provider_statuses,
        "provider_failure_classes": sorted(provider_failures),
        "target_min_side_px": episode["target_min_side_px"],
        "visibility_fraction": episode["visibility_fraction"],
    }

    decision = output["decision"]
    identifiable = bool(episode["observation_valid"] and episode["target_visible"] and correct_candidates)
    selected_id = decision["selected_candidate_id"]
    selected_correct = selected_id in correct_ids if selected_id is not None else None
    selected_is_distractor = None
    if selected_id is not None:
        distractor_regions = [item["region"] for item in episode["distractor_instances"]]
        selected_is_distractor = _best_truth_iou(candidates_by_id[selected_id]["region"], distractor_regions) >= IOU_THRESHOLD
    supporting_items = [evidence_by_id[evidence_id] for evidence_id in decision["supporting_evidence_ids"]]
    stale_used = decision["status"] == "GROUNDED" and any(
        not _evidence_is_current(item, decision_timestamp) for item in supporting_items
    )
    identity_match = None
    spatial_match = None
    if decision["status"] == "GROUNDED":
        identity_match = any(
            _evidence_is_current(item, decision_timestamp)
            and _identity_evidence_matches(item, episode["goal_spec"]["target_name"])
            for item in supporting_items
        )
        spatial_match = _best_truth_iou(decision["spatial_region"], truth_regions) >= IOU_THRESHOLD

    brain_rank = None
    if correct_ids:
        ranks = [index + 1 for index, candidate_id in enumerate(decision["ranked_candidate_ids"]) if candidate_id in correct_ids]
        brain_rank = min(ranks) if ranks else None
    rank_improvement = correct_rank - brain_rank if correct_rank is not None and brain_rank is not None else None
    brain_layer = {
        "identifiability": "IDENTIFIABLE" if identifiable else "NOT_IDENTIFIABLE",
        "top1_correct_given_available": (
            bool(decision["ranked_candidate_ids"] and decision["ranked_candidate_ids"][0] in correct_ids)
            if identifiable else None
        ),
        "wrong_instance_given_available": (
            decision["status"] == "GROUNDED" and selected_correct is False if identifiable else None
        ),
        "correct_abstention_under_ambiguity": (
            decision["status"] == "AMBIGUOUS"
            if episode["grounding_expectation"] == "MUST_BE_AMBIGUOUS" and identifiable else None
        ),
        "candidate_rank_improvement": rank_improvement if identifiable else None,
        "identity_match": identity_match if identifiable else None,
        "spatial_match": spatial_match if identifiable else None,
        "stale_evidence_used": stale_used if decision["status"] == "GROUNDED" else False,
    }

    outcome = _end_to_end_outcome(
        episode=episode,
        decision=decision,
        correct_available=bool(correct_candidates),
        selected_correct=selected_correct,
        selected_is_distractor=selected_is_distractor,
        identity_match=identity_match,
        spatial_match=spatial_match,
        stale_used=stale_used,
    )
    success_outcomes = {
        "CORRECT_GROUNDING", "CORRECT_AMBIGUITY", "CORRECT_ABSTENTION", "INVALID_OBSERVATION_HANDLED"
    }
    handoff_required = decision["status"] == "GROUNDED"
    return {
        "episode_id": episode["episode_id"],
        "protocol_id": PROTOCOL_ID,
        "observation_valid": episode["observation_valid"],
        "valid_system_output": True,
        "contract_error": None,
        "provider_availability": provider_layer,
        "brain_selection": brain_layer,
        "end_to_end": {"outcome": outcome, "success": outcome in success_outcomes},
        "p1_handoff": {
            "required": handoff_required,
            "valid": handoff_required,
            "source_frame_only": handoff_required,
            "claims_current_frame_tracking": False,
        },
        "strata": _episode_strata(episode),
    }


def _end_to_end_outcome(
    *,
    episode: Mapping[str, Any],
    decision: Mapping[str, Any],
    correct_available: bool,
    selected_correct: bool | None,
    selected_is_distractor: bool | None,
    identity_match: bool | None,
    spatial_match: bool | None,
    stale_used: bool,
) -> str:
    expectation = episode["grounding_expectation"]
    status = decision["status"]
    if expectation == "INVALID_OBSERVATION":
        return "INVALID_OBSERVATION_HANDLED" if status == "INVALID_OBSERVATION" else "INVALID_OBSERVATION_NOT_HANDLED"
    if not episode["target_visible"]:
        return "FALSE_GROUNDING_TARGET_ABSENT" if status == "GROUNDED" else "CORRECT_ABSTENTION"
    if status == "GROUNDED":
        if stale_used:
            return "STALE_EVIDENCE_USED"
        if spatial_match and not identity_match:
            return "GOAL_IDENTITY_ERROR"
        if selected_is_distractor:
            return "WRONG_INSTANCE"
        if identity_match and not spatial_match:
            return "SPATIAL_LOCALIZATION_ERROR"
        if selected_correct is False or not spatial_match:
            return "WRONG_INSTANCE"
        if identity_match and spatial_match:
            return "CORRECT_GROUNDING" if expectation == "MUST_GROUND" else "UNJUSTIFIED_GROUNDING"
        return "GOAL_IDENTITY_ERROR"
    if expectation == "MUST_BE_AMBIGUOUS":
        return "CORRECT_AMBIGUITY" if status == "AMBIGUOUS" else "MISSED_REQUIRED_AMBIGUITY"
    if expectation == "MUST_ABSTAIN":
        return "CORRECT_ABSTENTION" if status == "ABSTAIN_NO_RELIABLE_EVIDENCE" else "INCORRECT_ABSTENTION_STATUS"
    if not correct_available:
        return "PROVIDER_CORRECT_CANDIDATE_UNAVAILABLE"
    return "MISSED_REQUIRED_GROUNDING"


def _episode_strata(episode: Mapping[str, Any]) -> dict[str, str]:
    scene = episode["scene_condition"]
    return {
        "target_size": scene["target_size"],
        "visibility": scene["visibility"],
        "entrance_count": scene["entrance_count"],
        "text_support": episode["text_support"],
        "same_class_distractor": "YES" if scene["same_class_distractor"] else "NO",
        "target_presence": "PRESENT" if episode["target_visible"] else "ABSENT",
        "illumination": scene["illumination"],
        "view_angle": scene["view_angle"],
    }


def evaluate_batch(cases: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    results = [evaluate_episode(episode, output) for episode, output in cases]
    valid = [item for item in results if item["valid_system_output"]]
    provider_eligible = [
        item for item in valid
        if item["observation_valid"] and item["strata"]["target_presence"] == "PRESENT"
    ]
    identifiable = [item for item in valid if item["brain_selection"]["identifiability"] == "IDENTIFIABLE"]
    successes = [item for item in valid if item["end_to_end"]["success"]]
    wrong_instances = [item for item in valid if item["end_to_end"]["outcome"] == "WRONG_INSTANCE"]
    false_grounding = [item for item in valid if item["end_to_end"]["outcome"] == "FALSE_GROUNDING_TARGET_ABSENT"]
    aggregate = {
        "episode_count": len(results),
        "valid_system_output": _rate(len(valid), len(results)),
        "provider_correct_candidate_availability": _rate(
            sum(bool(item["provider_availability"]["correct_candidate_available"]) for item in provider_eligible),
            len(provider_eligible),
        ),
        "brain_top1_correct_given_available": _rate(
            sum(bool(item["brain_selection"]["top1_correct_given_available"]) for item in identifiable),
            len(identifiable),
        ),
        "wrong_instance_given_available": _rate(
            sum(bool(item["brain_selection"]["wrong_instance_given_available"]) for item in identifiable),
            len(identifiable),
        ),
        "end_to_end_success": _rate(len(successes), len(valid)),
        "wrong_instance_count": len(wrong_instances),
        "false_grounding_target_absent_count": len(false_grounding),
        "outcome_counts": _counts(item["end_to_end"]["outcome"] for item in results),
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "authority": "MOCK_MECHANICS_ONLY_NO_SCIENTIFIC_VERDICT",
        "aggregate": aggregate,
        "strata": _stratified_summaries(valid),
        "episodes": results,
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _stratified_summaries(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for field in (
        "target_size", "visibility", "entrance_count", "text_support",
        "same_class_distractor", "target_presence", "illumination", "view_angle",
    ):
        values = sorted({item["strata"][field] for item in results})
        summaries[field] = {}
        for value in values:
            subset = [item for item in results if item["strata"][field] == value]
            summaries[field][value] = {
                "episode_count": len(subset),
                "end_to_end_success": _rate(sum(item["end_to_end"]["success"] for item in subset), len(subset)),
                "outcome_counts": _counts(item["end_to_end"]["outcome"] for item in subset),
            }
    return summaries


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = evaluate_episode(_load_json(arguments.episode), _load_json(arguments.output))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid_system_output"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
