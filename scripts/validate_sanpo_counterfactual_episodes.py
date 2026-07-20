#!/usr/bin/env python3
"""Fail-closed validation for SANPO counterfactual episode manifests.

This tool validates human-reviewed event data only. It does not infer labels,
download data, select a model, or authorize production-model replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """The collection manifest cannot be used under the frozen contract."""


_ROUTE_VALIDATOR_PATH = Path(__file__).with_name("validate_explicit_route_intent_episode.py")
_ROUTE_SPEC = importlib.util.spec_from_file_location("explicit_route_intent_validator", _ROUTE_VALIDATOR_PATH)
if _ROUTE_SPEC is None or _ROUTE_SPEC.loader is None:  # pragma: no cover - repository corruption guard
    raise RuntimeError(f"cannot load route-intent validator: {_ROUTE_VALIDATOR_PATH}")
_ROUTE_VALIDATOR = importlib.util.module_from_spec(_ROUTE_SPEC)
_ROUTE_SPEC.loader.exec_module(_ROUTE_VALIDATOR)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _require_string(row: dict[str, Any], key: str, *, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where}.{key} must be a non-empty string")
    return value


def _resolve_local(root: Path, relative_path: str, *, where: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{where} escapes manifest root: {relative_path}") from error
    if not candidate.is_file():
        raise ContractError(f"{where} is not a local file: {relative_path}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_hash(root: Path, relative_path: str, expected: str, *, where: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ContractError(f"{where} must be a SHA256 hex string")
    actual = _sha256(_resolve_local(root, relative_path, where=where))
    if actual.lower() != expected.lower():
        raise ContractError(f"SHA256 mismatch for {where}")


def _positive_anchors(row: dict[str, Any], *, where: str) -> None:
    names = ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms")
    values = []
    for name in names:
        value = row.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractError(f"{where}.{name} must be a non-negative integer millisecond")
        values.append(value)
    if not values[0] <= values[1] < values[2]:
        raise ContractError(f"{where} anchors must satisfy first_visible <= alertable_start < passed_or_cleared")


def _negative_anchors(row: dict[str, Any], *, where: str) -> None:
    for name in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms"):
        if row.get(name) is not None:
            raise ContractError(f"{where}.{name} must be null for a matched negative")
    _require_string(row, "negative_reason", where=where)


def _duration_ms(row: dict[str, Any], policy: dict[str, Any], *, where: str) -> int:
    value = row.get("duration_ms")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{where}.duration_ms must be an integer")
    minimum = policy.get("minimum_duration_ms")
    maximum = policy.get("maximum_duration_ms")
    if not isinstance(minimum, int) or not isinstance(maximum, int) or minimum <= 0 or maximum < minimum:
        raise ContractError("config episode duration policy is invalid")
    if not minimum <= value <= maximum:
        raise ContractError(f"{where}.duration_ms must be within {minimum}..{maximum}")
    return value


def _interval(value: Any, *, expected: tuple[int, int], where: str) -> None:
    if not isinstance(value, list) or len(value) != 2 or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ContractError(f"{where} must be a two-integer interval")
    if tuple(value) != expected:
        raise ContractError(f"{where} must equal {list(expected)}")


def _risk_profile_and_lifecycle(
    row: dict[str, Any], *, scene_id: str, role: str, duration_ms: int, where: str,
) -> None:
    profile = row.get("risk_profile")
    intervals = row.get("lifecycle_intervals_ms")
    if not isinstance(profile, dict):
        raise ContractError(f"{where}.risk_profile must be an object")
    if not isinstance(intervals, dict):
        raise ContractError(f"{where}.lifecycle_intervals_ms must be an object")
    if profile.get("primary_hazard_type") != scene_id:
        raise ContractError(f"{where}.risk_profile.primary_hazard_type must equal scene_id")
    if role == "positive":
        if profile.get("corridor_relation") != "enters_or_blocks" or profile.get("lifecycle") != "approach_alertable_clear":
            raise ContractError(f"{where}.risk_profile is inconsistent with a positive event")
        first = row["first_visible_ms"]
        alertable = row["alertable_start_ms"]
        cleared = row["passed_or_cleared_ms"]
        if cleared > duration_ms:
            raise ContractError(f"{where}.passed_or_cleared_ms exceeds duration_ms")
        _interval(intervals.get("approach"), expected=(first, alertable), where=f"{where}.lifecycle_intervals_ms.approach")
        _interval(intervals.get("alertable"), expected=(alertable, cleared), where=f"{where}.lifecycle_intervals_ms.alertable")
        _interval(intervals.get("post_event"), expected=(cleared, duration_ms), where=f"{where}.lifecycle_intervals_ms.post_event")
        if set(intervals) != {"approach", "alertable", "post_event"}:
            raise ContractError(f"{where}.lifecycle_intervals_ms has unexpected positive keys")
    else:
        if profile.get("corridor_relation") != "outside_or_nonblocking" or profile.get("lifecycle") != "no_alert":
            raise ContractError(f"{where}.risk_profile is inconsistent with a matched negative")
        _interval(intervals.get("non_alert"), expected=(0, duration_ms), where=f"{where}.lifecycle_intervals_ms.non_alert")
        if set(intervals) != {"non_alert"}:
            raise ContractError(f"{where}.lifecycle_intervals_ms has unexpected negative keys")


def _annotation_evidence(
    row: dict[str, Any], *, config: dict[str, Any], root: Path, role: str, where: str,
) -> None:
    policy = config.get("annotation_evidence_schema")
    if not isinstance(policy, dict):
        raise ContractError("config.annotation_evidence_schema must be an object")
    evidence_path = _require_string(row, "annotation_evidence_path", where=where)
    _verify_file_hash(root, evidence_path, row.get("annotation_evidence_sha256"), where=f"{where}.annotation_evidence_path")
    evidence = _load_json(_resolve_local(root, evidence_path, where=f"{where}.annotation_evidence_path"))
    if evidence.get("schema") != policy.get("schema"):
        raise ContractError(f"{where}.annotation evidence has unexpected schema")
    if evidence.get("episode_id") != row["episode_id"]:
        raise ContractError(f"{where}.annotation evidence episode_id does not match")
    reviews = evidence.get("reviews")
    minimum = policy.get("minimum_independent_reviewers_per_episode")
    if not isinstance(reviews, list) or not isinstance(minimum, int) or minimum < 2:
        raise ContractError("annotation evidence policy is invalid")
    reviewer_ids: list[str] = []
    anchors_by_name: dict[str, list[int]] = defaultdict(list)
    for index, review in enumerate(reviews):
        review_where = f"{where}.annotation_evidence.reviews[{index}]"
        if not isinstance(review, dict):
            raise ContractError(f"{review_where} must be an object")
        reviewer_ids.append(_require_string(review, "reviewer_id", where=review_where))
        if review.get("reviewer_type") != "human":
            raise ContractError(f"{review_where}.reviewer_type must be human")
        if review.get("should_alert") is not row["expected_should_alert"]:
            raise ContractError(f"{review_where}.should_alert disagrees with episode")
        if config.get("route_conditioning_policy") is not None and review.get("critical") is not row.get("expected_critical"):
            raise ContractError(f"{review_where}.critical disagrees with episode")
        if role == "positive":
            for name in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms"):
                value = review.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ContractError(f"{review_where}.{name} must be a non-negative integer")
                anchors_by_name[name].append(value)
        elif any(review.get(name) is not None for name in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms")):
            raise ContractError(f"{review_where} must not contain positive anchors for a matched negative")
    if len(set(reviewer_ids)) < minimum:
        raise ContractError(f"{where}.annotation evidence needs {minimum} independent reviewers")
    if policy.get("reviewer_id_must_match_episode") and set(reviewer_ids) != set(row["annotation_reviewer_ids"]):
        raise ContractError(f"{where}.annotation reviewer IDs do not match episode")
    if role == "positive":
        tolerance = policy.get("positive_anchor_agreement_tolerance_ms")
        if not isinstance(tolerance, int) or tolerance < 0:
            raise ContractError("annotation evidence tolerance is invalid")
        for name, values in anchors_by_name.items():
            if max(values) - min(values) > tolerance:
                raise ContractError(f"{where}.annotation evidence {name} exceeds adjudication tolerance")

    route_policy = config.get("route_conditioning_policy")
    if route_policy is not None and route_policy.get("manifest_must_match_hashed_adjudication") is True:
        adjudication = evidence.get("adjudication")
        if not isinstance(adjudication, dict):
            raise ContractError(f"{where}.annotation evidence needs hashed adjudication")
        if adjudication.get("method") not in {"reviewer_consensus", "independent_human_adjudicator"}:
            raise ContractError(f"{where}.annotation adjudication method is invalid")
        if adjudication.get("should_alert") is not row["expected_should_alert"]:
            raise ContractError(f"{where}.manifest should_alert differs from adjudication")
        if adjudication.get("critical") is not row.get("expected_critical"):
            raise ContractError(f"{where}.manifest criticality differs from adjudication")
        if role == "positive":
            for name in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms"):
                if adjudication.get(name) != row.get(name):
                    raise ContractError(f"{where}.manifest {name} differs from adjudication")
        elif any(adjudication.get(name) is not None for name in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms")):
            raise ContractError(f"{where}.negative adjudication must not contain positive anchors")


def _route_intent_evidence(
    row: dict[str, Any], *, config: dict[str, Any], root: Path, duration_ms: int, where: str,
) -> dict[str, str] | None:
    """Validate an optional, history-only route trace bound to one event episode.

    Legacy SANPO collection configs have no route-conditioning policy and retain their existing
    behavior. A route-conditioned config makes the sidecar mandatory and rejects future-derived,
    stale, sparse, low-confidence, or risk-model-inferred routes.
    """
    policy = config.get("route_conditioning_policy")
    if policy is None:
        return None
    if not isinstance(policy, dict) or policy.get("required") is not True:
        raise ContractError("config.route_conditioning_policy must be a required policy object")

    route_path = _require_string(row, "route_intent_path", where=where)
    _verify_file_hash(root, route_path, row.get("route_intent_sha256"), where=f"{where}.route_intent_path")
    route = _load_json(_resolve_local(root, route_path, where=f"{where}.route_intent_path"))
    if route.get("episode_id") != row.get("episode_id"):
        raise ContractError(f"{where}.route intent episode_id does not match")
    try:
        route_report = _ROUTE_VALIDATOR.validate_episode(route, runtime=True)
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"{where}.route intent is invalid: {error}") from error

    route_intent_id = _require_string(route, "route_intent_id", where=f"{where}.route_intent")
    provider = route.get("provider")
    if not isinstance(provider, dict):
        raise ContractError(f"{where}.route_intent.provider must be an object")
    provider_id = _require_string(provider, "provider_id", where=f"{where}.route_intent.provider")
    samples = route.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError(f"{where}.route intent samples must be non-empty")

    minimum_confidence = policy.get("minimum_confidence")
    maximum_gap_ms = policy.get("maximum_valid_sample_gap_ms")
    endpoint_tolerance_ms = policy.get("endpoint_coverage_tolerance_ms")
    minimum_valid_fraction = policy.get("minimum_valid_sample_fraction")
    if (
        not isinstance(minimum_confidence, (int, float))
        or isinstance(minimum_confidence, bool)
        or not 0 <= float(minimum_confidence) <= 1
        or not isinstance(maximum_gap_ms, int)
        or maximum_gap_ms <= 0
        or not isinstance(endpoint_tolerance_ms, int)
        or endpoint_tolerance_ms < 0
        or not isinstance(minimum_valid_fraction, (int, float))
        or isinstance(minimum_valid_fraction, bool)
        or not 0 < float(minimum_valid_fraction) <= 1
    ):
        raise ContractError("config.route_conditioning_policy thresholds are invalid")

    valid_samples = [
        sample for sample in samples
        if sample.get("route_valid") is True and float(sample.get("confidence", -1)) >= float(minimum_confidence)
    ]
    if len(valid_samples) / len(samples) < float(minimum_valid_fraction):
        raise ContractError(f"{where}.route intent valid-sample fraction is below policy")
    timestamps = [int(sample["timestamp_ms"]) for sample in valid_samples]
    if timestamps[0] > endpoint_tolerance_ms or timestamps[-1] < duration_ms - endpoint_tolerance_ms:
        raise ContractError(f"{where}.route intent does not cover both episode endpoints")
    if any(current - previous > maximum_gap_ms for previous, current in zip(timestamps, timestamps[1:])):
        raise ContractError(f"{where}.route intent has a valid-sample gap above policy")

    if row.get("pair_role") == "positive":
        alertable = int(row["alertable_start_ms"])
        if not any(
            int(sample["timestamp_ms"]) <= alertable <= int(sample["valid_until_timestamp_ms"])
            for sample in valid_samples
        ):
            raise ContractError(f"{where}.route intent is not valid at alertable_start_ms")

    return {
        "route_intent_id": route_intent_id,
        "provider_id": provider_id,
        "parent_source_id": str(route_report["parent_source_id"]),
    }


def _capture_clock_evidence(
    row: dict[str, Any], *, config: dict[str, Any], root: Path, where: str,
) -> None:
    policy = config.get("route_conditioning_policy")
    if policy is None:
        return
    receipt_path = _require_string(row, "capture_clock_receipt_path", where=where)
    _verify_file_hash(
        root,
        receipt_path,
        row.get("capture_clock_receipt_sha256"),
        where=f"{where}.capture_clock_receipt_path",
    )
    receipt = _load_json(_resolve_local(root, receipt_path, where=f"{where}.capture_clock_receipt_path"))
    if receipt.get("schema") != policy.get("capture_clock_receipt_schema"):
        raise ContractError(f"{where}.capture clock receipt has unexpected schema")
    if receipt.get("episode_id") != row.get("episode_id"):
        raise ContractError(f"{where}.capture clock receipt episode_id does not match")
    if receipt.get("timestamps_strictly_monotonic") is not True:
        raise ContractError(f"{where}.capture timestamps are not strictly monotonic")
    if receipt.get("timestamp_unit") != "nanoseconds":
        raise ContractError(f"{where}.capture timestamp unit must be nanoseconds")
    frame_count = receipt.get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        raise ContractError(f"{where}.capture clock receipt frame_count must be positive")
    context = row.get("capture_context")
    if not isinstance(context, dict) or receipt.get("camera_frame") != context.get("camera_frame"):
        raise ContractError(f"{where}.capture clock camera_frame does not match capture_context")


def validate(config: dict[str, Any], manifest: dict[str, Any], *, root: Path, require_complete: bool) -> dict[str, Any]:
    if config.get("schema") != "blindassist_sanpo_counterfactual_episode_collection_v1":
        raise ContractError("unexpected collection config schema")
    design = config.get("design")
    if not isinstance(design, dict):
        raise ContractError("config.design must be an object")
    duration_policy = config.get("episode_duration_policy")
    if not isinstance(duration_policy, dict):
        raise ContractError("config.episode_duration_policy must be an object")
    sessions = config.get("sessions")
    scenes = config.get("scenes")
    if not isinstance(sessions, list) or not isinstance(scenes, list):
        raise ContractError("config sessions and scenes must be lists")
    session_ids = {_require_string(item, "session_id", where="config.sessions") for item in sessions if isinstance(item, dict)}
    scene_ids = {_require_string(item, "scene_id", where="config.scenes") for item in scenes if isinstance(item, dict)}
    if len(session_ids) != len(sessions) or len(scene_ids) != len(scenes):
        raise ContractError("configured session_id and scene_id values must be unique")
    if design.get("session_count") != len(session_ids) or design.get("scene_count") != len(scene_ids):
        raise ContractError("config design counts do not match session/scene inventory")

    if manifest.get("schema") != "blindassist_sanpo_counterfactual_episode_manifest_v1":
        raise ContractError("unexpected episode manifest schema")
    route_conditioned = config.get("route_conditioning_policy") is not None
    if route_conditioned:
        contract_id = _require_string(config, "contract_id", where="config")
        if manifest.get("contract_id") != contract_id:
            raise ContractError("manifest contract_id does not match config")
        if config.get("benchmark_only") is not True or manifest.get("benchmark_only") is not True:
            raise ContractError("route-conditioned truth must remain benchmark_only")
        if (
            config.get("production_model_replacement_authorized") is not False
            or manifest.get("production_model_replacement_authorized") is not False
        ):
            raise ContractError("route-conditioned truth cannot authorize production replacement")
        authority = config.get("authority")
        if not isinstance(authority, dict):
            raise ContractError("route-conditioned config.authority must be an object")
    receipts = manifest.get("source_receipts")
    episodes = manifest.get("episodes")
    if not isinstance(receipts, list) or not isinstance(episodes, list):
        raise ContractError("manifest source_receipts and episodes must be lists")

    receipt_by_id: dict[str, dict[str, Any]] = {}
    allowed_license = set(config["source_receipt_schema"]["allowed_license_status"])
    for index, receipt in enumerate(receipts):
        where = f"source_receipts[{index}]"
        if not isinstance(receipt, dict):
            raise ContractError(f"{where} must be an object")
        receipt_id = _require_string(receipt, "source_receipt_id", where=where)
        if receipt_id in receipt_by_id:
            raise ContractError(f"duplicate source_receipt_id: {receipt_id}")
        for field in ("source_owner_or_dataset", "collection_date", "license_evidence_path", "privacy_evidence_path", "reviewer_id", "raw_video_path", "episode_manifest_path"):
            _require_string(receipt, field, where=where)
        if receipt.get("license_status") not in allowed_license:
            raise ContractError(f"{where}.license_status is not allowed")
        if receipt.get("privacy_review_status") != config["source_receipt_schema"]["required_privacy_review_status"]:
            raise ContractError(f"{where}.privacy_review_status must be green")
        if config["source_receipt_schema"].get("hash_license_and_privacy_evidence") is True:
            _verify_file_hash(
                root,
                receipt["license_evidence_path"],
                receipt.get("license_evidence_sha256"),
                where=f"{where}.license_evidence_path",
            )
            _verify_file_hash(
                root,
                receipt["privacy_evidence_path"],
                receipt.get("privacy_evidence_sha256"),
                where=f"{where}.privacy_evidence_path",
            )
        _verify_file_hash(root, receipt["raw_video_path"], receipt.get("raw_video_sha256"), where=f"{where}.raw_video_path")
        _verify_file_hash(root, receipt["episode_manifest_path"], receipt.get("episode_manifest_sha256"), where=f"{where}.episode_manifest_path")
        receipt_by_id[receipt_id] = receipt

    required_episode_fields = config["episode_record_schema"]["required_fields"]
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[tuple[str, str, str]] = Counter()
    route_binding_by_episode_id: dict[str, dict[str, str]] = {}
    for index, episode in enumerate(episodes):
        where = f"episodes[{index}]"
        if not isinstance(episode, dict):
            raise ContractError(f"{where} must be an object")
        for field in required_episode_fields:
            if field not in episode:
                raise ContractError(f"{where} missing required field {field}")
        session_id = _require_string(episode, "session_id", where=where)
        scene_id = _require_string(episode, "scene_id", where=where)
        pair_id = _require_string(episode, "matched_pair_id", where=where)
        role = _require_string(episode, "pair_role", where=where)
        _require_string(episode, "episode_id", where=where)
        _require_string(episode, "risk_event_id", where=where)
        if session_id not in session_ids or scene_id not in scene_ids:
            raise ContractError(f"{where} uses a session or scene outside the config")
        if role not in set(config["episode_record_schema"]["pair_role_allowed"]):
            raise ContractError(f"{where}.pair_role is invalid")
        if not isinstance(episode.get("annotation_reviewer_ids"), list) or not episode["annotation_reviewer_ids"] or not all(isinstance(value, str) and value for value in episode["annotation_reviewer_ids"]):
            raise ContractError(f"{where}.annotation_reviewer_ids must be a non-empty string list")
        if len(set(episode["annotation_reviewer_ids"])) != len(episode["annotation_reviewer_ids"]):
            raise ContractError(f"{where}.annotation_reviewer_ids must be unique")
        receipt_id = _require_string(episode, "source_receipt_id", where=where)
        if receipt_id not in receipt_by_id:
            raise ContractError(f"{where} references unknown source_receipt_id")
        video_path = _require_string(episode, "video_path", where=where)
        _verify_file_hash(root, video_path, episode.get("video_sha256"), where=f"{where}.video_path")
        if config.get("route_conditioning_policy") is not None:
            if not isinstance(episode.get("expected_critical"), bool):
                raise ContractError(f"{where}.expected_critical must be boolean")
            if episode.get("expected_critical") and episode.get("expected_should_alert") is not True:
                raise ContractError(f"{where}.critical event must be alertable")
            _require_string(episode, "criticality_reason", where=where)
            _capture_clock_evidence(episode, config=config, root=root, where=where)
        if role == "positive":
            if episode.get("expected_should_alert") is not True:
                raise ContractError(f"{where}.expected_should_alert must be true for positive")
            _positive_anchors(episode, where=where)
        else:
            if episode.get("expected_should_alert") is not False:
                raise ContractError(f"{where}.expected_should_alert must be false for matched_negative")
            _negative_anchors(episode, where=where)
        duration_ms = _duration_ms(episode, duration_policy, where=where)
        _risk_profile_and_lifecycle(
            episode, scene_id=scene_id, role=role, duration_ms=duration_ms, where=where,
        )
        _annotation_evidence(episode, config=config, root=root, role=role, where=where)
        route_binding = _route_intent_evidence(
            episode, config=config, root=root, duration_ms=duration_ms, where=where,
        )
        if route_binding is not None:
            if route_binding["parent_source_id"] != receipt_id:
                raise ContractError(f"{where}.route intent parent_source_id does not match source_receipt_id")
            route_binding_by_episode_id[episode["episode_id"]] = route_binding
        context = episode.get("capture_context")
        if not isinstance(context, dict) or not all(isinstance(context.get(key), str) and context[key] for key in config["matrix_contract"]["matched_pair_members_must_share_capture_context"]):
            raise ContractError(f"{where}.capture_context must contain the configured capture context")
        pairs[pair_id].append(episode)
        counts[(session_id, scene_id, role)] += 1

    for pair_id, members in pairs.items():
        if len(members) != 2 or {member["pair_role"] for member in members} != {"positive", "matched_negative"}:
            raise ContractError(f"matched_pair {pair_id} must contain exactly one positive and one matched_negative")
        positive, negative = members
        for key in ("session_id", "scene_id", "source_receipt_id"):
            if positive[key] != negative[key]:
                raise ContractError(f"matched_pair {pair_id} crosses {key}")
        if positive["capture_context"] != negative["capture_context"]:
            raise ContractError(f"matched_pair {pair_id} has unmatched capture_context")
        if config.get("route_conditioning_policy") is not None:
            positive_route = route_binding_by_episode_id[positive["episode_id"]]
            negative_route = route_binding_by_episode_id[negative["episode_id"]]
            for key in ("route_intent_id", "provider_id"):
                if positive_route[key] != negative_route[key]:
                    raise ContractError(f"matched_pair {pair_id} crosses {key}")

    complete = manifest.get("collection_status") == "complete"
    if require_complete and not complete:
        raise ContractError("--require-complete needs collection_status=complete")
    if complete:
        expected_pairs = design.get("matched_pairs_per_session_scene")
        if not isinstance(expected_pairs, int) or expected_pairs <= 0:
            raise ContractError("design.matched_pairs_per_session_scene must be a positive integer")
        for session_id in session_ids:
            for scene_id in scene_ids:
                for role in ("positive", "matched_negative"):
                    if counts[(session_id, scene_id, role)] != expected_pairs:
                        raise ContractError(f"complete collection needs {expected_pairs} {role} episodes for {session_id}/{scene_id}")

    authority = config.get("authority")
    training_authorized = (
        authority.get("full_matrix_training") is True
        if isinstance(authority, dict)
        else True
    )
    return {
        "ok": True,
        "collection_status": manifest.get("collection_status", "unknown"),
        "episode_count": len(episodes),
        "matched_pair_count": len(pairs),
        "training_eligible": bool(complete and require_complete and training_authorized),
        "route_bound_episode_count": len(route_binding_by_episode_id),
        "route_conditioned_truth_eligible": bool(
            complete
            and require_complete
            and config.get("route_conditioning_policy") is not None
            and len(route_binding_by_episode_id) == len(episodes)
        ),
        "production_model_replacement_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a human-reviewed SANPO counterfactual episode manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(_load_json(args.config), _load_json(args.manifest), root=args.manifest.parent, require_complete=args.require_complete)
    except ContractError as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
