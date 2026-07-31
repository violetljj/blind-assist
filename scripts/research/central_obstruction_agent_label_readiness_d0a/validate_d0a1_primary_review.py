#!/usr/bin/env python3
"""Validate a complete non-isolated D0-A1 primary calibration review."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from .freeze_d0a1_pilot import DEFAULT_LOCK, load_json, load_lock, repo_path
from .freeze_input_universe import canonical_bytes, sha256_file


PRIMARY_REVIEW_NAME = "primary-review.json"
EVENTS_NAME = "primary-parent-events.jsonl"
VALIDATION_NAME = "primary-review-validation.json"


class PrimaryReviewValidationError(ValueError):
    """Fail-closed primary-review validation error."""


def derive_events(
    *,
    clip_id: str,
    source_id: str,
    session_id: str,
    frame_indices: list[int],
    labels: list[str],
    quality_states: list[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    start = 0
    for cursor in range(1, len(labels) + 1):
        closes = cursor == len(labels) or labels[cursor] != labels[start]
        if not closes:
            continue
        event_number = len(events)
        label = labels[start]
        qualities = sorted(set(quality_states[start:cursor]))
        events.append(
            {
                "parent_event_id": f"{clip_id}:P{event_number:02d}",
                "source_id": source_id,
                "session_id": session_id,
                "clip_id": clip_id,
                "start_observation_ordinal": start,
                "end_observation_ordinal": cursor - 1,
                "start_frame": frame_indices[start],
                "end_frame": frame_indices[cursor - 1],
                "observation_label": label,
                "camera_quality_states": qualities,
                "observation_count": cursor - start,
                "claim_critical": (
                    label
                    in {
                        "VISIBLE_CENTRAL_OBSTRUCTION_PRESENT",
                        "NOT_EVALUABLE",
                    }
                    or qualities != ["STABLE"]
                ),
                "claim_boundary": "VISIBLE_IMAGE_OBSERVATION_ONLY",
            }
        )
        start = cursor
    return events


def validate_primary_review(
    *,
    repo_root: Path,
    lock_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock = load_lock(repo_root, lock_path)
    manifest_path = output_root / "pilot-input-manifest.json"
    input_validation_path = output_root / "pilot-input-validation.json"
    review_path = output_root / PRIMARY_REVIEW_NAME
    if not manifest_path.is_file() or not input_validation_path.is_file() or not review_path.is_file():
        raise PrimaryReviewValidationError("pilot input, input validation, or primary review is missing")
    manifest = load_json(manifest_path, where="pilot manifest")
    input_validation = load_json(input_validation_path, where="pilot input validation")
    review = load_json(review_path, where="primary review")
    if input_validation.get("status") != "VALID":
        raise PrimaryReviewValidationError("pilot input validation is not VALID")
    prompt_path = repo_path(
        repo_root,
        lock["bindings"]["review_prompt"]["path"],
        where="review prompt",
    )
    if (
        review.get("protocol_id") != lock["protocol_id"]
        or review.get("phase") != "D0-A1"
        or review.get("evidence_instance") != lock["evidence_instance"]
        or review.get("pilot_input_manifest_sha256") != sha256_file(manifest_path)
        or review.get("prompt_sha256") != sha256_file(prompt_path)
    ):
        raise PrimaryReviewValidationError("review identity binding mismatch")
    context = review.get("review_context")
    base_context_valid = (
        context == "PRIMARY_CURRENT_TASK_NON_ISOLATED_SOURCE_ONLY"
        and review.get("prior_review_visible") is False
        and review.get("other_review_visible_before_submission") is False
    )
    transcription_context_valid = (
        context == "PRIMARY_CURRENT_TASK_NON_ISOLATED_TIMESTAMP_REPAIR_TRANSCRIPTION"
        and review.get("prior_review_visible") is True
        and review.get("other_review_visible_before_submission") is True
    )
    if (
        not (base_context_valid or transcription_context_valid)
        or review.get("isolated_context") is not False
        or review.get("source_only_view") is not True
        or review.get("candidate_output_visible") is not False
        or review.get("labels_generated_before_r1_lock") is not False
    ):
        raise PrimaryReviewValidationError("primary review context disclosure mismatch")
    if transcription_context_valid:
        predecessor = review.get("predecessor_review")
        if not isinstance(predecessor, dict):
            raise PrimaryReviewValidationError("timestamp-repair predecessor review is missing")
        predecessor_path = repo_path(
            repo_root,
            predecessor.get("path"),
            where="predecessor primary review",
        )
        if (
            not predecessor_path.is_file()
            or sha256_file(predecessor_path) != predecessor.get("sha256")
            or review.get("label_changes_from_predecessor") != 0
        ):
            raise PrimaryReviewValidationError("timestamp-repair predecessor binding mismatch")
    try:
        submitted_at = dt.datetime.fromisoformat(
            str(review.get("submitted_at_utc", "")).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise PrimaryReviewValidationError("primary review submission timestamp is invalid") from error
    if submitted_at.tzinfo is None or submitted_at > dt.datetime.now(dt.timezone.utc):
        raise PrimaryReviewValidationError("primary review submission timestamp is in the future")
    observations = manifest.get("observations")
    if not isinstance(observations, list):
        raise PrimaryReviewValidationError("pilot observations are missing")
    manifest_by_clip: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        manifest_by_clip.setdefault(observation["clip_id"], []).append(observation)
    for rows in manifest_by_clip.values():
        rows.sort(key=lambda row: row["clip_observation_ordinal"])
    clip_reviews = review.get("clip_reviews")
    if not isinstance(clip_reviews, list):
        raise PrimaryReviewValidationError("clip reviews are missing")
    review_by_clip = {
        row.get("clip_id"): row
        for row in clip_reviews
        if isinstance(row, dict)
    }
    if len(review_by_clip) != len(clip_reviews) or set(review_by_clip) != set(manifest_by_clip):
        raise PrimaryReviewValidationError("clip review identity set mismatch")
    labels_allowed = set(lock["observation_contract"]["labels"])
    qualities_allowed = set(lock["observation_contract"]["quality_states"])
    label_counts = {label: 0 for label in sorted(labels_allowed)}
    quality_counts = {quality: 0 for quality in sorted(qualities_allowed)}
    events: list[dict[str, Any]] = []
    observation_count = 0
    claim_critical_observation_count = 0
    sources_seen: set[str] = set()
    for clip_id, manifest_rows in manifest_by_clip.items():
        clip_review = review_by_clip[clip_id]
        indices = clip_review.get("source_frame_indices")
        labels = clip_review.get("labels")
        qualities = clip_review.get("quality_states")
        rationales = clip_review.get("rationale_codes")
        expected_indices = [row["source_frame_index"] for row in manifest_rows]
        if (
            indices != expected_indices
            or not isinstance(labels, list)
            or not isinstance(qualities, list)
            or not isinstance(rationales, list)
            or not (len(labels) == len(qualities) == len(rationales) == len(expected_indices))
        ):
            raise PrimaryReviewValidationError(f"clip observation alignment mismatch: {clip_id}")
        if any(label not in labels_allowed for label in labels):
            raise PrimaryReviewValidationError(f"unknown label: {clip_id}")
        if any(quality not in qualities_allowed for quality in qualities):
            raise PrimaryReviewValidationError(f"unknown quality state: {clip_id}")
        if any(not isinstance(reason, str) or not reason for reason in rationales):
            raise PrimaryReviewValidationError(f"missing rationale code: {clip_id}")
        source_id = manifest_rows[0]["source_id"]
        session_id = manifest_rows[0]["session_id"]
        sources_seen.add(source_id)
        for label, quality in zip(labels, qualities):
            label_counts[label] += 1
            quality_counts[quality] += 1
            observation_count += 1
            if label in {"VISIBLE_CENTRAL_OBSTRUCTION_PRESENT", "NOT_EVALUABLE"} or quality != "STABLE":
                claim_critical_observation_count += 1
        events.extend(
            derive_events(
                clip_id=clip_id,
                source_id=source_id,
                session_id=session_id,
                frame_indices=indices,
                labels=labels,
                quality_states=qualities,
            )
        )
    thresholds = lock["readiness_thresholds"]
    present_labels = sorted(label for label, count in label_counts.items() if count)
    present_qualities = sorted(quality for quality, count in quality_counts.items() if count)
    coverage_preconditions = {
        "minimum_calibration_sources": len(sources_seen) >= thresholds["minimum_calibration_sources"],
        "minimum_pilot_clips": len(clip_reviews) >= thresholds["minimum_pilot_clips"],
        "minimum_pilot_observations": observation_count >= thresholds["minimum_pilot_observations"],
        "required_observed_labels": set(thresholds["required_observed_labels"]).issubset(present_labels),
        "minimum_observed_quality_states": len(present_qualities) >= thresholds["minimum_observed_quality_states"],
        "maximum_not_evaluable_fraction": (
            label_counts["NOT_EVALUABLE"] / observation_count
            <= thresholds["maximum_not_evaluable_fraction"]
        ),
    }
    low_risk_observation_count = observation_count - claim_critical_observation_count
    frozen_audit_target = max(
        len(sources_seen),
        math.ceil(low_risk_observation_count * lock["audit_rule"]["low_risk_independent_audit_fraction"]),
    )
    return {
        "schema_version": "blindassist.central_obstruction_d0a1_primary_review_validation.v1",
        "protocol_id": lock["protocol_id"],
        "phase": "D0-A1",
        "evidence_instance": lock["evidence_instance"],
        "status": "VALID",
        "decision": "PRIMARY_CALIBRATION_PASS_COMPLETE_INDEPENDENT_PASS_REQUIRED",
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": sha256_file(lock_path),
        "prompt_sha256": sha256_file(prompt_path),
        "pilot_input_manifest_sha256": sha256_file(manifest_path),
        "pilot_input_validation_sha256": sha256_file(input_validation_path),
        "primary_review_sha256": sha256_file(review_path),
        "calibration_source_count": len(sources_seen),
        "pilot_clip_count": len(clip_reviews),
        "pilot_observation_count": observation_count,
        "label_counts": label_counts,
        "quality_counts": quality_counts,
        "primary_parent_event_count": len(events),
        "claim_critical_observation_count": claim_critical_observation_count,
        "low_risk_observation_count": low_risk_observation_count,
        "frozen_low_risk_independent_audit_target": frozen_audit_target,
        "coverage_preconditions": coverage_preconditions,
        "coverage_preconditions_pass": all(coverage_preconditions.values()),
        "isolated_review_pass_count": 0,
        "agreement_metrics_evaluated": False,
        "readiness_evaluated": False,
        "candidate_output_access": False,
        "production_source_overlap_count": 0,
        "d0a2_production_labeling_authorized": False,
        "d0b_authorized": False,
        "next_permitted_action": "D0-A1_FRESH_ISOLATED_SECOND_PASS_OVER_ALL_55_PILOT_OBSERVATIONS",
        "errors": [],
        "claim_ceiling": "NON_ISOLATED_PRIMARY_CALIBRATION_PASS_ONLY",
    }, events


def write_validation(
    *,
    repo_root: Path,
    lock_path: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    validation_path = output_root / VALIDATION_NAME
    events_path = output_root / EVENTS_NAME
    if validation_path.exists() or events_path.exists():
        raise PrimaryReviewValidationError("primary derived outputs already exist; refusing overwrite")
    result, events = validate_primary_review(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root,
    )
    token = uuid.uuid4().hex
    validation_tmp = output_root / f".{VALIDATION_NAME}.{token}.tmp"
    events_tmp = output_root / f".{EVENTS_NAME}.{token}.tmp"
    try:
        events_tmp.write_bytes(b"".join(canonical_bytes(event) for event in events))
        result["primary_parent_events_sha256"] = sha256_file(events_tmp)
        validation_tmp.write_bytes(canonical_bytes(result))
        os.replace(events_tmp, events_path)
        os.replace(validation_tmp, validation_path)
    except Exception:
        events_tmp.unlink(missing_ok=True)
        validation_tmp.unlink(missing_ok=True)
        raise
    return validation_path, events_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    lock = load_lock(repo_root, args.lock.resolve())
    output_root = args.output_root or repo_path(repo_root, lock["output_root"], where="output_root")
    validation_path, events_path = write_validation(
        repo_root=repo_root,
        lock_path=args.lock.resolve(),
        output_root=output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "decision": "PRIMARY_CALIBRATION_PASS_COMPLETE_INDEPENDENT_PASS_REQUIRED",
                "validation": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
                "events": str(events_path),
                "events_sha256": sha256_file(events_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
