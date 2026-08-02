"""Validate and seal the three burned primitive-review submissions.

The seal step is the boundary between RGB-blind review and pair selection.
It validates only the review contract and coverage, computes one canonical
bundle hash, and emits a receipt that a later pair-builder may read without
opening primitive labels.  It never derives actionability and never adjudicates
causal versus retrospective disagreement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_file, sha256_json
from .judge_audit import (
    CAUSAL_TEMPORAL_EVIDENCE_WINDOW,
    PRIMITIVE_OBSERVATION_POLICY,
    PRIMITIVE_POLICY_VERSION,
    PRIMITIVE_FIELDS,
    PRIMITIVE_VALUES,
    RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
    REVIEW_MAP_SCHEMA,
    REVIEW_SCHEMA,
    VISIBILITY_EVIDENCE_WINDOW,
    VISIBILITY_POLICY_VERSION,
)
from .prepare_judge_burned_pilot import FREEZE_SCHEMA


SEAL_SCHEMA = "blindassist.eval_validity_r0.judge_review_bundle_seal.v3"


class ReviewSealError(ValueError):
    """Raised when a review cannot cross the pair-selection boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewSealError(message)


def _validate_freeze(freeze: dict[str, Any]) -> set[str]:
    _require(freeze.get("schema_version") == FREEZE_SCHEMA, "pilot freeze schema mismatch")
    _require(freeze.get("protocol_id") == PROTOCOL_ID, "pilot freeze protocol mismatch")
    _require(freeze.get("status") == "CALIBRATION_BURNED_OUTPUT_BLIND_EVENTS_FROZEN", "pilot freeze is not output-blind and burned")
    items = freeze.get("items")
    _require(isinstance(items, list) and 8 <= len(items) <= 12, "pilot freeze event count is outside 8-12")
    event_ids = {item.get("pilot_event_id") for item in items if isinstance(item, dict)}
    _require(len(event_ids) == len(items) and all(isinstance(value, str) and value for value in event_ids), "pilot freeze event IDs are invalid")
    return event_ids


def _validate_map(review_map: dict[str, Any], event_ids: set[str]) -> dict[str, dict[str, str]]:
    _require(review_map.get("schema_version") == REVIEW_MAP_SCHEMA, "review map schema mismatch")
    _require(review_map.get("protocol_id") == PROTOCOL_ID, "review map protocol mismatch")
    result: dict[str, dict[str, str]] = {}
    for index, item in enumerate(review_map.get("items", [])):
        where = f"review map item {index}"
        _require(isinstance(item, dict), f"{where}: expected object")
        opaque, event_id, role = item.get("review_item_id"), item.get("parent_event_id"), item.get("reviewer_role")
        _require(isinstance(opaque, str) and opaque and isinstance(event_id, str) and event_id in event_ids, f"{where}: invalid mapping")
        _require(role in {"CAUSAL_A", "CAUSAL_B", "RETROSPECTIVE_C"}, f"{where}: invalid reviewer role")
        _require(opaque not in result, f"{where}: duplicate review item")
        result[opaque] = {"parent_event_id": event_id, "reviewer_role": role}
    _require(result, "review map is empty")
    return result


def _validate_review(review: dict[str, Any], expected_role: str, expected_view: str, event_ids: set[str], mapping: dict[str, dict[str, str]]) -> None:
    _require(review.get("schema_version") == REVIEW_SCHEMA, f"{expected_role}: review schema mismatch")
    _require(review.get("protocol_id") == PROTOCOL_ID, f"{expected_role}: review protocol mismatch")
    _require(review.get("reviewer_role") == expected_role and review.get("view") == expected_view, f"{expected_role}: role/view mismatch")
    _require(review.get("future_frame_access") is (expected_view == "RETROSPECTIVE"), f"{expected_role}: future-frame flag mismatch")
    _require(review.get("sealed_before_pair_selection") is True, f"{expected_role}: review was not sealed before pair selection")
    _require(review.get("isolated_context") is True and review.get("metadata_blind") is True, f"{expected_role}: isolation/metadata-blind flag mismatch")
    _require(review.get("primitive_policy_version") == PRIMITIVE_POLICY_VERSION, f"{expected_role}: primitive policy version mismatch")
    _require(review.get("visibility_evidence_window") == VISIBILITY_EVIDENCE_WINDOW, f"{expected_role}: visibility evidence window mismatch")
    _require(review.get("field_evidence_windows") == PRIMITIVE_OBSERVATION_POLICY["field_evidence_windows"], f"{expected_role}: field evidence windows mismatch")
    expected_temporal_window = CAUSAL_TEMPORAL_EVIDENCE_WINDOW if expected_view == "CAUSAL" else RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW
    _require(review.get("temporal_fields_evidence_window") == expected_temporal_window, f"{expected_role}: temporal evidence window mismatch")
    for field in ("other_review_visible_before_submission", "model_output_visible", "candidate_metadata_visible", "selection_reason_visible", "semantic_bucket_visible", "source_session_visible"):
        _require(review.get(field) is False, f"{expected_role}: forbidden visibility flag {field}")
    items = review.get("items")
    _require(isinstance(items, list) and len(items) == len(event_ids), f"{expected_role}: review item coverage mismatch")
    seen_events: set[str] = set()
    for index, item in enumerate(items):
        where = f"{expected_role} item {index}"
        _require(isinstance(item, dict) and set(item) == {"review_item_id", "primitive_observations"}, f"{where}: direct action or metadata field is present")
        opaque = item.get("review_item_id")
        _require(isinstance(opaque, str) and opaque in mapping, f"{where}: review item is not in private map")
        _require(mapping[opaque]["reviewer_role"] == expected_role, f"{where}: item belongs to a different reviewer")
        event_id = mapping[opaque]["parent_event_id"]
        _require(event_id not in seen_events, f"{where}: duplicate event")
        observations = item.get("primitive_observations")
        _require(isinstance(observations, list) and len(observations) == 60, f"{where}: primitive coverage must be 60 frames")
        frames: set[int] = set()
        for obs_index, observation in enumerate(observations):
            obs_where = f"{where} frame {obs_index}"
            _require(isinstance(observation, dict) and set(observation) == {"frame_index", *PRIMITIVE_FIELDS}, f"{obs_where}: direct action or metadata field is present")
            frame = observation.get("frame_index")
            _require(isinstance(frame, int) and not isinstance(frame, bool) and 0 <= frame < 60 and frame not in frames, f"{obs_where}: invalid frame index")
            frames.add(frame)
            for field in PRIMITIVE_FIELDS:
                _require(observation.get(field) in PRIMITIVE_VALUES[field], f"{obs_where}: invalid primitive {field}")
        _require(frames == set(range(60)), f"{where}: frame set is not 0..59")
        seen_events.add(event_id)
    _require(seen_events == event_ids, f"{expected_role}: event coverage mismatch")


def seal_reviews(*, freeze: dict[str, Any], review_map: dict[str, Any], reviews: list[dict[str, Any]], review_paths: list[Path], output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite review seal: {output}")
    event_ids = _validate_freeze(freeze)
    mapping = _validate_map(review_map, event_ids)
    expected = (("CAUSAL_A", "CAUSAL"), ("CAUSAL_B", "CAUSAL"), ("RETROSPECTIVE_C", "RETROSPECTIVE"))
    _require(len(reviews) == len(expected) == len(review_paths), "exactly three reviews are required")
    by_role = {review.get("reviewer_role"): review for review in reviews}
    _require(set(by_role) == {role for role, _ in expected}, "review role coverage mismatch")
    for role, view in expected:
        _validate_review(by_role[role], role, view, event_ids, mapping)
    ordered = [by_role[role] for role, _ in expected]
    bundle_hash = sha256_json(ordered)
    result = {
        "schema_version": SEAL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "PRIMITIVE_REVIEWS_SEALED_BEFORE_PAIR_SELECTION",
        "pilot_freeze_sha256": sha256_json(freeze),
        "review_bundle_sha256": bundle_hash,
        "review_count": len(ordered),
        "review_roles": [role for role, _ in expected],
        "review_files": [
            {"reviewer_role": review.get("reviewer_role"), "path": str(path), "sha256": sha256_file(path)}
            for path, review in zip(review_paths, reviews)
        ],
        "primitive_labels_opened_to_pair_builder": False,
        "derived_labels_opened_to_pair_builder": False,
        "reviewed_event_phase_opened_to_pair_builder": False,
        "reviewed_motion_relation_opened_to_pair_builder": False,
        "pair_selection_access": "HASH_ONLY",
        "retrospective_adjudicates_causal_truth": False,
        "next_gate": "Use only review_bundle_sha256 and this receipt for deterministic selection-only pair construction.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-freeze", type=Path, required=True)
    parser.add_argument("--review-map", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--review-retrospective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.review_a, args.review_b, args.review_retrospective]
    reviews = [read_json(path) for path in paths]
    result = seal_reviews(
        freeze=read_json(args.pilot_freeze),
        review_map=read_json(args.review_map),
        reviews=reviews,
        review_paths=paths,
        output=args.output,
    )
    print(f"status={result['status']} review_bundle_sha256={result['review_bundle_sha256']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
