#!/usr/bin/env python3
"""Validate the fresh D0-A1 isolated pass and compute initial agreement."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .freeze_d0a1_pilot import DEFAULT_LOCK, load_json, load_lock, repo_path
from .freeze_input_universe import canonical_bytes, sha256_file
from .validate_d0a1_primary_review import derive_events


SECOND_REVIEW_NAME = "isolated-second-review.json"
SECOND_EVENTS_NAME = "isolated-second-parent-events.jsonl"
AGREEMENT_NAME = "d0a1-initial-agreement.json"
ADJUDICATION_PACKET_NAME = "d0a1-adjudication-packet.json"


class IsolatedReviewValidationError(ValueError):
    """Fail-closed isolated-review validation error."""


def _parse_utc(value: Any, *, where: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise IsolatedReviewValidationError(f"{where} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise IsolatedReviewValidationError(f"{where} timestamp lacks timezone")
    return parsed.astimezone(dt.timezone.utc)


def _manifest_by_clip(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    observations = manifest.get("observations")
    if not isinstance(observations, list) or not observations:
        raise IsolatedReviewValidationError("pilot observations are missing")
    result: dict[str, list[dict[str, Any]]] = {}
    for row in observations:
        if not isinstance(row, dict) or not isinstance(row.get("clip_id"), str):
            raise IsolatedReviewValidationError("invalid pilot observation")
        result.setdefault(row["clip_id"], []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: row["clip_observation_ordinal"])
    return result


def _normalize_isolated_review(
    *,
    review: dict[str, Any],
    manifest_by_clip: dict[str, list[dict[str, Any]]],
    labels_allowed: set[str],
    qualities_allowed: set[str],
) -> tuple[dict[str, dict[str, list[Any]]], dict[str, int], dict[str, int]]:
    clip_reviews = review.get("clip_reviews")
    if not isinstance(clip_reviews, list):
        raise IsolatedReviewValidationError("isolated clip reviews are missing")
    expected_clip_order = list(manifest_by_clip)
    actual_clip_order = [row.get("clip_id") for row in clip_reviews if isinstance(row, dict)]
    if actual_clip_order != expected_clip_order or len(actual_clip_order) != len(clip_reviews):
        raise IsolatedReviewValidationError("isolated clip order or identity mismatch")
    label_counts = {label: 0 for label in sorted(labels_allowed)}
    quality_counts = {quality: 0 for quality in sorted(qualities_allowed)}
    normalized: dict[str, dict[str, list[Any]]] = {}
    for clip_review in clip_reviews:
        clip_id = clip_review["clip_id"]
        rows = clip_review.get("observations")
        expected = manifest_by_clip[clip_id]
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise IsolatedReviewValidationError(f"isolated observation count mismatch: {clip_id}")
        indices: list[int] = []
        labels: list[str] = []
        qualities: list[str] = []
        rationales: list[str] = []
        for actual, manifest_row in zip(rows, expected):
            if not isinstance(actual, dict):
                raise IsolatedReviewValidationError(f"invalid isolated observation: {clip_id}")
            if (
                actual.get("clip_observation_ordinal")
                != manifest_row["clip_observation_ordinal"]
                or actual.get("source_frame_index") != manifest_row["source_frame_index"]
            ):
                raise IsolatedReviewValidationError(f"isolated observation identity mismatch: {clip_id}")
            label = actual.get("label")
            quality = actual.get("quality_state")
            rationale = actual.get("rationale")
            if label not in labels_allowed:
                raise IsolatedReviewValidationError(f"unknown isolated label: {clip_id}")
            if quality not in qualities_allowed:
                raise IsolatedReviewValidationError(f"unknown isolated quality: {clip_id}")
            if not isinstance(rationale, str) or not rationale.strip():
                raise IsolatedReviewValidationError(f"missing isolated rationale: {clip_id}")
            indices.append(actual["source_frame_index"])
            labels.append(label)
            qualities.append(quality)
            rationales.append(rationale.strip())
            label_counts[label] += 1
            quality_counts[quality] += 1
        normalized[clip_id] = {
            "source_frame_indices": indices,
            "labels": labels,
            "quality_states": qualities,
            "rationales": rationales,
        }
    return normalized, label_counts, quality_counts


def _primary_by_clip(
    primary: dict[str, Any], manifest_by_clip: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, list[Any]]]:
    rows = primary.get("clip_reviews")
    if not isinstance(rows, list):
        raise IsolatedReviewValidationError("primary clip reviews are missing")
    by_clip = {row.get("clip_id"): row for row in rows if isinstance(row, dict)}
    if len(by_clip) != len(rows) or set(by_clip) != set(manifest_by_clip):
        raise IsolatedReviewValidationError("primary clip identity mismatch")
    for clip_id, manifest_rows in manifest_by_clip.items():
        expected_indices = [row["source_frame_index"] for row in manifest_rows]
        row = by_clip[clip_id]
        if (
            row.get("source_frame_indices") != expected_indices
            or len(row.get("labels", [])) != len(expected_indices)
            or len(row.get("quality_states", [])) != len(expected_indices)
        ):
            raise IsolatedReviewValidationError(f"primary observation alignment mismatch: {clip_id}")
    return by_clip


def _temporal_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap = max(
        0,
        min(left["end_observation_ordinal"], right["end_observation_ordinal"])
        - max(left["start_observation_ordinal"], right["start_observation_ordinal"])
        + 1,
    )
    union = (
        max(left["end_observation_ordinal"], right["end_observation_ordinal"])
        - min(left["start_observation_ordinal"], right["start_observation_ordinal"])
        + 1
    )
    return overlap / union


def _match_events(
    primary_events: list[dict[str, Any]],
    isolated_events: list[dict[str, Any]],
    *,
    minimum_iou: float,
    maximum_boundary_delta: int,
) -> list[tuple[int, int]]:
    candidates: dict[int, list[int]] = {}
    for isolated_index, isolated in enumerate(isolated_events):
        options: list[tuple[float, int, int]] = []
        for primary_index, primary in enumerate(primary_events):
            if (
                primary["clip_id"] != isolated["clip_id"]
                or primary["source_id"] != isolated["source_id"]
                or primary["observation_label"] != isolated["observation_label"]
            ):
                continue
            start_delta = abs(
                primary["start_observation_ordinal"] - isolated["start_observation_ordinal"]
            )
            end_delta = abs(
                primary["end_observation_ordinal"] - isolated["end_observation_ordinal"]
            )
            iou = _temporal_iou(primary, isolated)
            if iou >= minimum_iou and max(start_delta, end_delta) <= maximum_boundary_delta:
                options.append((-iou, max(start_delta, end_delta), primary_index))
        candidates[isolated_index] = [item[2] for item in sorted(options)]
    primary_to_isolated: dict[int, int] = {}

    def assign(isolated_index: int, seen: set[int]) -> bool:
        for primary_index in candidates[isolated_index]:
            if primary_index in seen:
                continue
            seen.add(primary_index)
            if primary_index not in primary_to_isolated or assign(
                primary_to_isolated[primary_index], seen
            ):
                primary_to_isolated[primary_index] = isolated_index
                return True
        return False

    for isolated_index in range(len(isolated_events)):
        assign(isolated_index, set())
    return sorted(
        ((primary_index, isolated_index) for primary_index, isolated_index in primary_to_isolated.items()),
        key=lambda pair: (primary_events[pair[0]]["clip_id"], pair[0], pair[1]),
    )


def _nearest_rank_p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def validate_isolated_review(
    *, repo_root: Path, lock_path: Path, output_root: Path, review_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    lock = load_lock(repo_root, lock_path)
    manifest_path = output_root / "pilot-input-manifest.json"
    input_validation_path = output_root / "pilot-input-validation.json"
    primary_path = output_root / "primary-review.json"
    primary_validation_path = output_root / "primary-review-validation.json"
    required = [manifest_path, input_validation_path, primary_path, primary_validation_path, review_path]
    if any(not path.is_file() for path in required):
        raise IsolatedReviewValidationError("required input, primary, or isolated review is missing")
    manifest = load_json(manifest_path, where="pilot manifest")
    input_validation = load_json(input_validation_path, where="pilot input validation")
    primary = load_json(primary_path, where="primary review")
    primary_validation = load_json(primary_validation_path, where="primary validation")
    review = load_json(review_path, where="isolated review")
    if input_validation.get("status") != "VALID" or primary_validation.get("status") != "VALID":
        raise IsolatedReviewValidationError("input or primary validation is not VALID")
    prompt_path = repo_path(repo_root, lock["bindings"]["review_prompt"]["path"], where="prompt")
    if (
        review.get("schema_version")
        != "blindassist.central_obstruction_d0a1_isolated_review.v1"
        or review.get("protocol_id") != lock["protocol_id"]
        or review.get("phase") != "D0-A1"
        or review.get("evidence_instance") != lock["evidence_instance"]
        or review.get("pilot_input_manifest_sha256") != sha256_file(manifest_path)
        or review.get("prompt_sha256", review.get("review_prompt_sha256"))
        != sha256_file(prompt_path)
    ):
        raise IsolatedReviewValidationError("isolated review identity binding mismatch")
    if (
        review.get("review_context") != "FRESH_ISOLATED_SECOND_PASS"
        or review.get("isolated_context") is not True
        or review.get("source_only_view") is not True
        or review.get("prior_review_visible") is not False
        or review.get("other_review_visible_before_submission") is not False
        or review.get("candidate_output_visible") is not False
        or not isinstance(review.get("attestation"), (str, dict))
    ):
        raise IsolatedReviewValidationError("isolated review context disclosure mismatch")
    if not isinstance(review.get("review_id"), str) or not isinstance(review.get("reviewer_id"), str):
        raise IsolatedReviewValidationError("isolated reviewer identity is missing")
    try:
        uuid.UUID(review["review_id"])
    except (ValueError, AttributeError) as error:
        raise IsolatedReviewValidationError("isolated review_id is not a UUID") from error
    if (
        not review["reviewer_id"].strip()
        or review["reviewer_id"] == primary.get("reviewer_id")
        or review["review_id"] == primary.get("review_id")
    ):
        raise IsolatedReviewValidationError("isolated reviewer identity is not fresh")
    if review.get("reviewer_type") != "CODEX_AGENT":
        raise IsolatedReviewValidationError("isolated reviewer type mismatch")
    submitted_at = _parse_utc(review.get("submitted_at_utc"), where="isolated review submission")
    frozen_at = _parse_utc(manifest.get("frozen_at_utc"), where="pilot freeze")
    if submitted_at < frozen_at or submitted_at > dt.datetime.now(dt.timezone.utc):
        raise IsolatedReviewValidationError("isolated review timestamp is outside the valid envelope")

    manifest_by_clip = _manifest_by_clip(manifest)
    labels_allowed = set(lock["observation_contract"]["labels"])
    qualities_allowed = set(lock["observation_contract"]["quality_states"])
    isolated_by_clip, isolated_label_counts, isolated_quality_counts = _normalize_isolated_review(
        review=review,
        manifest_by_clip=manifest_by_clip,
        labels_allowed=labels_allowed,
        qualities_allowed=qualities_allowed,
    )
    primary_by_clip = _primary_by_clip(primary, manifest_by_clip)
    isolated_events: list[dict[str, Any]] = []
    primary_events: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    agreements = 0
    claim_critical_total = 0
    claim_critical_agreements = 0
    quality_agreements = 0
    total = 0
    primary_not_evaluable = 0
    for clip_id, manifest_rows in manifest_by_clip.items():
        isolated_row = isolated_by_clip[clip_id]
        primary_row = primary_by_clip[clip_id]
        indices = isolated_row["source_frame_indices"]
        source_id = manifest_rows[0]["source_id"]
        session_id = manifest_rows[0]["session_id"]
        isolated_clip_events = derive_events(
            clip_id=clip_id,
            source_id=source_id,
            session_id=session_id,
            frame_indices=indices,
            labels=isolated_row["labels"],
            quality_states=isolated_row["quality_states"],
        )
        for event in isolated_clip_events:
            event["review_pass"] = "FRESH_ISOLATED_SECOND_PASS"
        isolated_events.extend(isolated_clip_events)
        primary_events.extend(
            derive_events(
                clip_id=clip_id,
                source_id=source_id,
                session_id=session_id,
                frame_indices=indices,
                labels=primary_row["labels"],
                quality_states=primary_row["quality_states"],
            )
        )
        for position, manifest_row in enumerate(manifest_rows):
            primary_label = primary_row["labels"][position]
            isolated_label = isolated_row["labels"][position]
            primary_quality = primary_row["quality_states"][position]
            isolated_quality = isolated_row["quality_states"][position]
            same_label = primary_label == isolated_label
            critical = (
                primary_label in {"VISIBLE_CENTRAL_OBSTRUCTION_PRESENT", "NOT_EVALUABLE"}
                or isolated_label in {"VISIBLE_CENTRAL_OBSTRUCTION_PRESENT", "NOT_EVALUABLE"}
                or primary_quality != "STABLE"
                or isolated_quality != "STABLE"
            )
            total += 1
            agreements += int(same_label)
            quality_agreements += int(primary_quality == isolated_quality)
            primary_not_evaluable += int(primary_label == "NOT_EVALUABLE")
            if critical:
                claim_critical_total += 1
                claim_critical_agreements += int(same_label)
            if not same_label:
                disagreements.append(
                    {
                        "item_id": f"{clip_id}:{manifest_row['clip_observation_ordinal']:02d}",
                        "source_id": manifest_row["source_id"],
                        "session_id": manifest_row["session_id"],
                        "clip_id": clip_id,
                        "clip_observation_ordinal": manifest_row["clip_observation_ordinal"],
                        "source_frame_index": manifest_row["source_frame_index"],
                        "review_image_path": manifest_row["review_image_path"],
                        "review_image_sha256": manifest_row["review_image_sha256"],
                        "primary_label": primary_label,
                        "primary_quality_state": primary_quality,
                        "primary_rationale_code": primary_row["rationale_codes"][position],
                        "isolated_label": isolated_label,
                        "isolated_quality_state": isolated_quality,
                        "isolated_rationale": isolated_row["rationales"][position],
                        "claim_critical_union": critical,
                    }
                )
    event_rule = lock["parent_event_rule"]["reviewer_event_match"]
    matches = _match_events(
        primary_events,
        isolated_events,
        minimum_iou=event_rule["minimum_temporal_iou"],
        maximum_boundary_delta=event_rule["maximum_start_or_end_delta_observations"],
    )
    boundary_deltas: list[int] = []
    for primary_index, isolated_index in matches:
        primary_event = primary_events[primary_index]
        isolated_event = isolated_events[isolated_index]
        boundary_deltas.extend(
            [
                abs(
                    primary_event["start_observation_ordinal"]
                    - isolated_event["start_observation_ordinal"]
                ),
                abs(
                    primary_event["end_observation_ordinal"]
                    - isolated_event["end_observation_ordinal"]
                ),
            ]
        )
    event_denominator = max(len(primary_events), len(isolated_events))
    overall_agreement = agreements / total if total else None
    critical_agreement = (
        claim_critical_agreements / claim_critical_total if claim_critical_total else None
    )
    event_match_rate = len(matches) / event_denominator if event_denominator else None
    boundary_p95 = _nearest_rank_p95(boundary_deltas)
    unresolved_fraction = len(disagreements) / total if total else None
    maximum_not_evaluable_fraction = max(
        primary_not_evaluable / total,
        isolated_label_counts["NOT_EVALUABLE"] / total,
    )
    thresholds = lock["readiness_thresholds"]
    threshold_checks = {
        "nonzero_denominators": total > 0 and claim_critical_total > 0 and event_denominator > 0,
        "overall_observation_label_agreement": (
            overall_agreement is not None
            and overall_agreement >= thresholds["overall_observation_label_agreement"]
        ),
        "claim_critical_observation_label_agreement": (
            critical_agreement is not None
            and critical_agreement
            >= thresholds["claim_critical_observation_label_agreement"]
        ),
        "parent_event_match_rate": (
            event_match_rate is not None
            and event_match_rate >= thresholds["parent_event_match_rate"]
        ),
        "boundary_delta_p95": (
            boundary_p95 is not None
            and boundary_p95 <= thresholds["boundary_delta_p95_max_observations"]
        ),
        "maximum_unresolved_fraction": (
            unresolved_fraction is not None
            and unresolved_fraction <= thresholds["maximum_unresolved_fraction"]
        ),
        "maximum_not_evaluable_fraction": (
            maximum_not_evaluable_fraction <= thresholds["maximum_not_evaluable_fraction"]
        ),
        "coverage_preconditions": primary_validation.get("coverage_preconditions_pass") is True,
        "isolated_review_complete": total == manifest.get("pilot_observation_count"),
    }
    adjudication_required = bool(disagreements)
    ready_without_adjudication = all(threshold_checks.values()) and not adjudication_required
    decision = (
        "MATERIAL_DISAGREEMENT_ADJUDICATION_REQUIRED"
        if adjudication_required
        else (
            "READY_FOR_D0_A2_PRIMARY_AGENT_LABELING"
            if ready_without_adjudication
            else "D0_A1_NOT_READY_THRESHOLDS_FAILED"
        )
    )
    result = {
        "schema_version": "blindassist.central_obstruction_d0a1_initial_agreement.v1",
        "protocol_id": lock["protocol_id"],
        "phase": "D0-A1",
        "evidence_instance": lock["evidence_instance"],
        "status": "VALID",
        "decision": decision,
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": sha256_file(lock_path),
        "prompt_sha256": sha256_file(prompt_path),
        "pilot_input_manifest_sha256": sha256_file(manifest_path),
        "primary_review_sha256": sha256_file(primary_path),
        "primary_review_validation_sha256": sha256_file(primary_validation_path),
        "isolated_review_source_sha256": sha256_file(review_path),
        "isolated_review_pass_count": 1,
        "pilot_observation_count": total,
        "primary_parent_event_count": len(primary_events),
        "isolated_parent_event_count": len(isolated_events),
        "primary_not_evaluable_count": primary_not_evaluable,
        "isolated_label_counts": isolated_label_counts,
        "isolated_quality_counts": isolated_quality_counts,
        "observation_label_agreement_count": agreements,
        "overall_observation_label_agreement": overall_agreement,
        "claim_critical_union_count": claim_critical_total,
        "claim_critical_label_agreement_count": claim_critical_agreements,
        "claim_critical_observation_label_agreement": critical_agreement,
        "quality_state_agreement_count": quality_agreements,
        "parent_event_match_count": len(matches),
        "parent_event_match_denominator": event_denominator,
        "parent_event_match_rate": event_match_rate,
        "boundary_delta_p95_observations": boundary_p95,
        "material_disagreement_count": len(disagreements),
        "unresolved_fraction": unresolved_fraction,
        "maximum_review_not_evaluable_fraction": maximum_not_evaluable_fraction,
        "threshold_checks": threshold_checks,
        "adjudication_required": adjudication_required,
        "readiness_evaluated": not adjudication_required,
        "d0a2_production_labeling_authorized": ready_without_adjudication,
        "d0b_authorized": False,
        "next_permitted_action": (
            "D0-A1_FRESH_ADJUDICATION_OF_FROZEN_MATERIAL_DISAGREEMENTS"
            if adjudication_required
            else (
                "D0-A2_DESIGN_PRIMARY_AGENT_LABELING"
                if ready_without_adjudication
                else "STOP_D0_A1_NOT_READY_REVIEW_FAILED_THRESHOLDS"
            )
        ),
        "claim_ceiling": "AGENT_LABELABILITY_CALIBRATION_ONLY_NOT_OBJECTIVE_TRUTH_OR_SAFETY",
        "errors": [],
    }
    packet = None
    if disagreements:
        packet = {
            "schema_version": "blindassist.central_obstruction_d0a1_adjudication_packet.v1",
            "protocol_id": lock["protocol_id"],
            "phase": "D0-A1",
            "evidence_instance": lock["evidence_instance"],
            "status": "FROZEN_MATERIAL_DISAGREEMENTS_PENDING_ADJUDICATION",
            "prompt": {"path": lock["bindings"]["review_prompt"]["path"], "sha256": sha256_file(prompt_path)},
            "pilot_input_manifest_sha256": sha256_file(manifest_path),
            "primary_review_sha256": sha256_file(primary_path),
            "isolated_review_source_sha256": sha256_file(review_path),
            "material_disagreement_count": len(disagreements),
            "items": disagreements,
            "candidate_output_access": False,
        }
    return result, isolated_events, packet


def write_validation(
    *, repo_root: Path, lock_path: Path, output_root: Path, review_path: Path
) -> tuple[Path, Path, Path, Path | None]:
    review_output = output_root / SECOND_REVIEW_NAME
    events_output = output_root / SECOND_EVENTS_NAME
    agreement_output = output_root / AGREEMENT_NAME
    packet_output = output_root / ADJUDICATION_PACKET_NAME
    destinations = [review_output, events_output, agreement_output, packet_output]
    if any(path.exists() for path in destinations):
        raise IsolatedReviewValidationError("isolated derived output already exists; refusing overwrite")
    result, events, packet = validate_isolated_review(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root,
        review_path=review_path,
    )
    token = uuid.uuid4().hex
    review_tmp = output_root / f".{SECOND_REVIEW_NAME}.{token}.tmp"
    events_tmp = output_root / f".{SECOND_EVENTS_NAME}.{token}.tmp"
    agreement_tmp = output_root / f".{AGREEMENT_NAME}.{token}.tmp"
    packet_tmp = output_root / f".{ADJUDICATION_PACKET_NAME}.{token}.tmp"
    try:
        shutil.copyfile(review_path, review_tmp)
        events_tmp.write_bytes(b"".join(canonical_bytes(event) for event in events))
        result["isolated_second_review_sha256"] = sha256_file(review_tmp)
        result["isolated_parent_events_sha256"] = sha256_file(events_tmp)
        if packet is not None:
            packet_tmp.write_bytes(canonical_bytes(packet))
            result["adjudication_packet_sha256"] = sha256_file(packet_tmp)
        agreement_tmp.write_bytes(canonical_bytes(result))
        os.replace(review_tmp, review_output)
        os.replace(events_tmp, events_output)
        if packet is not None:
            os.replace(packet_tmp, packet_output)
        os.replace(agreement_tmp, agreement_output)
    except Exception:
        for path in [review_tmp, events_tmp, agreement_tmp, packet_tmp]:
            path.unlink(missing_ok=True)
        raise
    return review_output, events_output, agreement_output, packet_output if packet is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--review-path", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = load_lock(repo_root, lock_path)
    output_root = args.output_root or repo_path(repo_root, lock["output_root"], where="output_root")
    review_output, events_output, agreement_output, packet_output = write_validation(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root.resolve(),
        review_path=args.review_path.resolve(),
    )
    agreement = load_json(agreement_output, where="initial agreement")
    print(
        json.dumps(
            {
                "status": "VALID",
                "decision": agreement["decision"],
                "review": str(review_output),
                "review_sha256": sha256_file(review_output),
                "events": str(events_output),
                "events_sha256": sha256_file(events_output),
                "agreement": str(agreement_output),
                "agreement_sha256": sha256_file(agreement_output),
                "adjudication_packet": str(packet_output) if packet_output else None,
                "adjudication_packet_sha256": sha256_file(packet_output) if packet_output else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
