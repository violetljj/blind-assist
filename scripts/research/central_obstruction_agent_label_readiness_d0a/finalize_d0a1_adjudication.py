#!/usr/bin/env python3
"""Validate D0-A1 adjudication and emit the immutable final readiness terminal."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .freeze_d0a1_pilot import DEFAULT_LOCK, load_json, load_lock, repo_path
from .freeze_input_universe import canonical_bytes, sha256_file
from .validate_d0a1_primary_review import derive_events


ADJUDICATION_REVIEW_NAME = "adjudication-review.json"
CANONICAL_LABELS_NAME = "canonical-calibration-labels.jsonl"
CANONICAL_EVENTS_NAME = "canonical-calibration-parent-events.jsonl"
FINAL_READINESS_NAME = "d0a1-final-readiness.json"


class AdjudicationValidationError(ValueError):
    """Fail-closed adjudication validation error."""


def _parse_utc(value: Any, *, where: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise AdjudicationValidationError(f"{where} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise AdjudicationValidationError(f"{where} timestamp lacks timezone")
    return parsed.astimezone(dt.timezone.utc)


def _review_labels_by_identity(review: dict[str, Any], *, isolated: bool) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    clip_reviews = review.get("clip_reviews")
    if not isinstance(clip_reviews, list):
        raise AdjudicationValidationError("raw review clips are missing")
    for clip in clip_reviews:
        clip_id = clip.get("clip_id")
        if not isinstance(clip_id, str):
            raise AdjudicationValidationError("raw review clip identity is invalid")
        if isolated:
            rows = clip.get("observations")
            if not isinstance(rows, list):
                raise AdjudicationValidationError("isolated observation rows are missing")
            for row in rows:
                ordinal = row.get("clip_observation_ordinal")
                item_id = f"{clip_id}:{ordinal:02d}"
                if item_id in result:
                    raise AdjudicationValidationError("duplicate isolated item identity")
                result[item_id] = {
                    "clip_id": clip_id,
                    "clip_observation_ordinal": ordinal,
                    "source_frame_index": row.get("source_frame_index"),
                    "label": row.get("label"),
                    "quality_state": row.get("quality_state"),
                }
        else:
            indices = clip.get("source_frame_indices")
            labels = clip.get("labels")
            qualities = clip.get("quality_states")
            if not (
                isinstance(indices, list)
                and isinstance(labels, list)
                and isinstance(qualities, list)
                and len(indices) == len(labels) == len(qualities)
            ):
                raise AdjudicationValidationError("primary observation rows are misaligned")
            for ordinal, (frame, label, quality) in enumerate(zip(indices, labels, qualities)):
                item_id = f"{clip_id}:{ordinal:02d}"
                if item_id in result:
                    raise AdjudicationValidationError("duplicate primary item identity")
                result[item_id] = {
                    "clip_id": clip_id,
                    "clip_observation_ordinal": ordinal,
                    "source_frame_index": frame,
                    "label": label,
                    "quality_state": quality,
                }
    return result


def finalize_adjudication(
    *, repo_root: Path, lock_path: Path, output_root: Path, review_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    lock = load_lock(repo_root, lock_path)
    manifest_path = output_root / "pilot-input-manifest.json"
    primary_path = output_root / "primary-review.json"
    isolated_path = output_root / "isolated-second-review.json"
    agreement_path = output_root / "d0a1-initial-agreement.json"
    packet_path = output_root / "d0a1-adjudication-packet.json"
    required = [manifest_path, primary_path, isolated_path, agreement_path, packet_path, review_path]
    if any(not path.is_file() for path in required):
        raise AdjudicationValidationError("required raw review, agreement, packet, or adjudication is missing")
    manifest = load_json(manifest_path, where="pilot manifest")
    primary = load_json(primary_path, where="primary review")
    isolated = load_json(isolated_path, where="isolated review")
    agreement = load_json(agreement_path, where="initial agreement")
    packet = load_json(packet_path, where="adjudication packet")
    review = load_json(review_path, where="adjudication review")
    prompt_path = repo_path(repo_root, lock["bindings"]["review_prompt"]["path"], where="prompt")
    if (
        agreement.get("status") != "VALID"
        or agreement.get("decision") != "MATERIAL_DISAGREEMENT_ADJUDICATION_REQUIRED"
        or packet.get("status") != "FROZEN_MATERIAL_DISAGREEMENTS_PENDING_ADJUDICATION"
    ):
        raise AdjudicationValidationError("initial agreement does not authorize adjudication")
    primary_sha = sha256_file(primary_path)
    isolated_sha = sha256_file(isolated_path)
    if (
        agreement.get("primary_review_sha256") != primary_sha
        or agreement.get(
            "isolated_second_review_sha256",
            agreement.get("isolated_review_source_sha256"),
        )
        != isolated_sha
        or packet.get("primary_review_sha256") != primary_sha
        or packet.get("isolated_review_source_sha256") != isolated_sha
    ):
        raise AdjudicationValidationError("raw review hash binding mismatch")
    if (
        review.get("schema_version")
        != "blindassist.central_obstruction_d0a1_adjudication_review.v1"
        or review.get("protocol_id") != lock["protocol_id"]
        or review.get("phase") != "D0-A1"
        or review.get("evidence_instance") != lock["evidence_instance"]
        or review.get("adjudication_packet_sha256", review.get("packet_sha256"))
        != sha256_file(packet_path)
        or review.get("prompt_sha256", review.get("review_prompt_sha256"))
        != sha256_file(prompt_path)
    ):
        raise AdjudicationValidationError("adjudication identity binding mismatch")
    if (
        review.get("review_context")
        != "FRESH_THIRD_AGENT_MATERIAL_DISAGREEMENT_ADJUDICATION"
        or review.get("source_only_view") is not True
        or review.get("candidate_output_visible") is not False
        or review.get("pair_labels_visible") is not True
        or review.get("aggregate_metrics_visible") is not False
        or review.get("reviewer_type") != "CODEX_AGENT"
        or not isinstance(review.get("attestation"), (str, dict))
    ):
        raise AdjudicationValidationError("adjudication context disclosure mismatch")
    review_id = review.get("review_id")
    reviewer_id = review.get("reviewer_id")
    if not isinstance(review_id, str) or not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise AdjudicationValidationError("adjudicator identity is missing")
    try:
        uuid.UUID(review_id)
    except (ValueError, AttributeError) as error:
        raise AdjudicationValidationError("adjudication review_id is not a UUID") from error
    if reviewer_id in {primary.get("reviewer_id"), isolated.get("reviewer_id")} or review_id in {
        primary.get("review_id"), isolated.get("review_id")
    }:
        raise AdjudicationValidationError("adjudicator identity is not fresh")
    submitted_at = _parse_utc(review.get("submitted_at_utc"), where="adjudication submission")
    agreement_at = _parse_utc(agreement.get("validated_at_utc"), where="initial agreement")
    if submitted_at < agreement_at or submitted_at > dt.datetime.now(dt.timezone.utc):
        raise AdjudicationValidationError("adjudication timestamp is outside the valid envelope")

    packet_items = packet.get("items")
    review_items = review.get("items")
    if not isinstance(packet_items, list) or not isinstance(review_items, list):
        raise AdjudicationValidationError("adjudication items are missing")
    expected_ids = [item.get("item_id") for item in packet_items]
    actual_ids = [item.get("item_id") for item in review_items if isinstance(item, dict)]
    if actual_ids != expected_ids or len(actual_ids) != len(review_items):
        raise AdjudicationValidationError("adjudication item order or identity mismatch")
    labels_allowed = set(lock["observation_contract"]["labels"])
    qualities_allowed = set(lock["observation_contract"]["quality_states"])
    dispositions = {"ADJUDICATED_LABEL", "QUARANTINE_NOT_EVALUABLE"}
    adjudication_by_id: dict[str, dict[str, Any]] = {}
    disposition_counts = {disposition: 0 for disposition in sorted(dispositions)}
    for expected, actual in zip(packet_items, review_items):
        if any(
            actual.get(field) != expected.get(field)
            for field in ["item_id", "clip_id", "clip_observation_ordinal", "source_frame_index"]
        ):
            raise AdjudicationValidationError("adjudication observation identity mismatch")
        final_label = actual.get("final_label")
        final_quality = actual.get("final_quality_state")
        disposition = actual.get("disposition")
        rationale = actual.get("rationale")
        if final_label not in labels_allowed or final_quality not in qualities_allowed:
            raise AdjudicationValidationError("adjudication label or quality is invalid")
        if disposition not in dispositions or not isinstance(rationale, str) or not rationale.strip():
            raise AdjudicationValidationError("adjudication disposition or rationale is invalid")
        if disposition == "QUARANTINE_NOT_EVALUABLE" and final_label != "NOT_EVALUABLE":
            raise AdjudicationValidationError("quarantined item must be NOT_EVALUABLE")
        disposition_counts[disposition] += 1
        adjudication_by_id[actual["item_id"]] = actual

    primary_by_id = _review_labels_by_identity(primary, isolated=False)
    isolated_by_id = _review_labels_by_identity(isolated, isolated=True)
    observations = manifest.get("observations")
    if (
        not isinstance(observations, list)
        or len(primary_by_id) != len(isolated_by_id)
        or len(primary_by_id) != len(observations)
    ):
        raise AdjudicationValidationError("raw review coverage mismatch")
    canonical_rows: list[dict[str, Any]] = []
    labels_by_clip: dict[str, list[str]] = {}
    qualities_by_clip: dict[str, list[str]] = {}
    frames_by_clip: dict[str, list[int]] = {}
    manifest_by_clip: dict[str, list[dict[str, Any]]] = {}
    label_counts = {label: 0 for label in sorted(labels_allowed)}
    for observation in observations:
        clip_id = observation["clip_id"]
        ordinal = observation["clip_observation_ordinal"]
        item_id = f"{clip_id}:{ordinal:02d}"
        primary_row = primary_by_id.get(item_id)
        isolated_row = isolated_by_id.get(item_id)
        if primary_row is None or isolated_row is None:
            raise AdjudicationValidationError("raw review item is missing")
        if (
            primary_row["source_frame_index"] != observation["source_frame_index"]
            or isolated_row["source_frame_index"] != observation["source_frame_index"]
        ):
            raise AdjudicationValidationError("raw review frame identity mismatch")
        if primary_row["label"] == isolated_row["label"]:
            if item_id in adjudication_by_id:
                raise AdjudicationValidationError("agreement item unexpectedly adjudicated")
            final_label = primary_row["label"]
            final_quality = primary_row["quality_state"]
            resolution = "TWO_REVIEW_LABEL_AGREEMENT"
        else:
            adjudication = adjudication_by_id.get(item_id)
            if adjudication is None:
                raise AdjudicationValidationError("material disagreement lacks adjudication")
            final_label = adjudication["final_label"]
            final_quality = adjudication["final_quality_state"]
            resolution = (
                "THIRD_AGENT_ADJUDICATED"
                if adjudication["disposition"] == "ADJUDICATED_LABEL"
                else "QUARANTINED_NOT_EVALUABLE"
            )
        label_counts[final_label] += 1
        canonical_rows.append(
            {
                "item_id": item_id,
                "source_id": observation["source_id"],
                "session_id": observation["session_id"],
                "clip_id": clip_id,
                "clip_observation_ordinal": ordinal,
                "source_frame_index": observation["source_frame_index"],
                "review_image_sha256": observation["review_image_sha256"],
                "primary_label": primary_row["label"],
                "isolated_label": isolated_row["label"],
                "primary_quality_state": primary_row["quality_state"],
                "isolated_quality_state": isolated_row["quality_state"],
                "canonical_label": final_label,
                "canonical_quality_state": final_quality,
                "resolution": resolution,
                "claim_boundary": "VISIBLE_IMAGE_OBSERVATION_ONLY",
            }
        )
        manifest_by_clip.setdefault(clip_id, []).append(observation)
        labels_by_clip.setdefault(clip_id, []).append(final_label)
        qualities_by_clip.setdefault(clip_id, []).append(final_quality)
        frames_by_clip.setdefault(clip_id, []).append(observation["source_frame_index"])
    canonical_events: list[dict[str, Any]] = []
    for clip_id, clip_rows in manifest_by_clip.items():
        events = derive_events(
            clip_id=clip_id,
            source_id=clip_rows[0]["source_id"],
            session_id=clip_rows[0]["session_id"],
            frame_indices=frames_by_clip[clip_id],
            labels=labels_by_clip[clip_id],
            quality_states=qualities_by_clip[clip_id],
        )
        for event in events:
            resolutions = sorted(
                {
                    canonical_rows[index]["resolution"]
                    for index in range(len(canonical_rows))
                    if canonical_rows[index]["clip_id"] == clip_id
                    and event["start_observation_ordinal"]
                    <= canonical_rows[index]["clip_observation_ordinal"]
                    <= event["end_observation_ordinal"]
                }
            )
            event["resolution_states"] = resolutions
            event["review_authority"] = "AGENT_CALIBRATION_CANARY_ONLY"
        canonical_events.extend(events)

    thresholds = lock["readiness_thresholds"]
    final_not_evaluable_fraction = label_counts["NOT_EVALUABLE"] / len(canonical_rows)
    threshold_checks = {
        "overall_observation_label_agreement": (
            agreement["overall_observation_label_agreement"]
            >= thresholds["overall_observation_label_agreement"]
        ),
        "claim_critical_observation_label_agreement": (
            agreement["claim_critical_observation_label_agreement"]
            >= thresholds["claim_critical_observation_label_agreement"]
        ),
        "parent_event_match_rate": (
            agreement["parent_event_match_rate"] >= thresholds["parent_event_match_rate"]
        ),
        "boundary_delta_p95": (
            agreement["boundary_delta_p95_observations"]
            <= thresholds["boundary_delta_p95_max_observations"]
        ),
        "maximum_unresolved_fraction": True,
        "maximum_not_evaluable_fraction": (
            final_not_evaluable_fraction <= thresholds["maximum_not_evaluable_fraction"]
        ),
        "coverage_preconditions": agreement["threshold_checks"]["coverage_preconditions"],
        "isolated_review_complete": agreement["threshold_checks"]["isolated_review_complete"],
        "adjudication_complete": len(review_items) == len(packet_items) > 0,
    }
    ready = all(threshold_checks.values())
    terminal = (
        "READY_FOR_D0_A2_PRIMARY_AGENT_LABELING"
        if ready
        else "AGENT_LABEL_PROTOCOL_NOT_RELIABLE"
    )
    result = {
        "schema_version": "blindassist.central_obstruction_d0a1_final_readiness.v1",
        "protocol_id": lock["protocol_id"],
        "phase": "D0-A1",
        "evidence_instance": lock["evidence_instance"],
        "status": "VALID",
        "terminal": terminal,
        "governance_outcome": "IMPLEMENTATION_DEBUGGED" if ready else "NOT_EVALUABLE",
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": sha256_file(lock_path),
        "prompt_sha256": sha256_file(prompt_path),
        "pilot_input_manifest_sha256": sha256_file(manifest_path),
        "primary_review_sha256": primary_sha,
        "isolated_review_sha256": isolated_sha,
        "initial_agreement_sha256": sha256_file(agreement_path),
        "adjudication_packet_sha256": sha256_file(packet_path),
        "adjudication_review_source_sha256": sha256_file(review_path),
        "raw_reviewer_metrics": {
            "overall_observation_label_agreement": agreement["overall_observation_label_agreement"],
            "claim_critical_observation_label_agreement": agreement[
                "claim_critical_observation_label_agreement"
            ],
            "parent_event_match_rate": agreement["parent_event_match_rate"],
            "boundary_delta_p95_observations": agreement["boundary_delta_p95_observations"],
            "material_disagreement_count": agreement["material_disagreement_count"],
        },
        "adjudication_item_count": len(review_items),
        "adjudication_disposition_counts": disposition_counts,
        "post_adjudication_unresolved_count": 0,
        "post_adjudication_unresolved_fraction": 0.0,
        "canonical_observation_count": len(canonical_rows),
        "canonical_parent_event_count": len(canonical_events),
        "canonical_label_counts": label_counts,
        "canonical_not_evaluable_fraction": final_not_evaluable_fraction,
        "threshold_checks": threshold_checks,
        "readiness_evaluated": True,
        "d0a2_production_labeling_authorized": ready,
        "d0b_authorized": False,
        "next_permitted_action": (
            "D0-A2_DESIGN_PRIMARY_AGENT_LABELING"
            if ready
            else "NEW_D0_A_VERSION_REVISE_OBSERVATION_OR_EVENT_WORKFLOW_ON_BURNED_CALIBRATION_ONLY"
        ),
        "failure_learning": {
            "observation": "Raw reviewer parent-event match remained below the frozen threshold after complete adjudication coverage.",
            "supported_inference": "The current observation/event workflow is not stable enough to scale into D0-A2.",
            "alternative_explanations": [
                "Foreground-versus-background depth ordering is ambiguous in single RGB observations.",
                "Close surfaces and edited or turning views create correlated label-boundary instability.",
                "The fixed five-observation clips may be too sparse for stable natural-event boundaries.",
            ],
            "reuse_candidates": [
                "Frozen disagreement observations as prompt-regression stress cases",
                "Raw reviewer boundaries as event-definition redesign diagnostics",
                "Canonical calibration ledger as Canary-only failure analysis",
            ],
            "information_gained": "Observation-level agreement was acceptable, but parent-event boundary reproducibility was not.",
        },
        "claim_ceiling": "AGENT_LABELABILITY_CALIBRATION_CANARY_ONLY_NOT_OBJECTIVE_TRUTH_MODEL_EFFECT_OR_SAFETY",
        "errors": [],
    }
    return result, canonical_rows, canonical_events


def write_finalization(
    *, repo_root: Path, lock_path: Path, output_root: Path, review_path: Path
) -> tuple[Path, Path, Path, Path]:
    review_output = output_root / ADJUDICATION_REVIEW_NAME
    labels_output = output_root / CANONICAL_LABELS_NAME
    events_output = output_root / CANONICAL_EVENTS_NAME
    readiness_output = output_root / FINAL_READINESS_NAME
    if any(path.exists() for path in [review_output, labels_output, events_output, readiness_output]):
        raise AdjudicationValidationError("final D0-A1 output already exists; refusing overwrite")
    result, labels, events = finalize_adjudication(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root,
        review_path=review_path,
    )
    token = uuid.uuid4().hex
    review_tmp = output_root / f".{ADJUDICATION_REVIEW_NAME}.{token}.tmp"
    labels_tmp = output_root / f".{CANONICAL_LABELS_NAME}.{token}.tmp"
    events_tmp = output_root / f".{CANONICAL_EVENTS_NAME}.{token}.tmp"
    readiness_tmp = output_root / f".{FINAL_READINESS_NAME}.{token}.tmp"
    try:
        shutil.copyfile(review_path, review_tmp)
        labels_tmp.write_bytes(b"".join(canonical_bytes(row) for row in labels))
        events_tmp.write_bytes(b"".join(canonical_bytes(row) for row in events))
        result["adjudication_review_sha256"] = sha256_file(review_tmp)
        result["canonical_labels_sha256"] = sha256_file(labels_tmp)
        result["canonical_events_sha256"] = sha256_file(events_tmp)
        readiness_tmp.write_bytes(canonical_bytes(result))
        os.replace(review_tmp, review_output)
        os.replace(labels_tmp, labels_output)
        os.replace(events_tmp, events_output)
        os.replace(readiness_tmp, readiness_output)
    except Exception:
        for path in [review_tmp, labels_tmp, events_tmp, readiness_tmp]:
            path.unlink(missing_ok=True)
        raise
    return review_output, labels_output, events_output, readiness_output


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
    review_output, labels_output, events_output, readiness_output = write_finalization(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root.resolve(),
        review_path=args.review_path.resolve(),
    )
    readiness = load_json(readiness_output, where="final readiness")
    print(
        json.dumps(
            {
                "status": "VALID",
                "terminal": readiness["terminal"],
                "review": str(review_output),
                "review_sha256": sha256_file(review_output),
                "canonical_labels": str(labels_output),
                "canonical_labels_sha256": sha256_file(labels_output),
                "canonical_events": str(events_output),
                "canonical_events_sha256": sha256_file(events_output),
                "readiness": str(readiness_output),
                "readiness_sha256": sha256_file(readiness_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
