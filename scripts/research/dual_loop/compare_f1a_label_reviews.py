#!/usr/bin/env python3
"""Validate and compare two candidate-blind F-1A visual label reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ReviewError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def require_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewError(f"{where} must be non-empty text")
    return value


def require_sha(value: Any, where: str) -> str:
    text = require_text(value, where)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ReviewError(f"{where} must be lowercase SHA-256")
    return text


def interval(value: Any, where: str, duration: float) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise ReviewError(f"{where} must be an object")
    start = value.get("start")
    end = value.get("end")
    if (
        not isinstance(start, (int, float))
        or isinstance(start, bool)
        or not isinstance(end, (int, float))
        or isinstance(end, bool)
    ):
        raise ReviewError(f"{where} bounds must be numeric")
    start = float(start)
    end = float(end)
    if not 0.0 <= start <= end <= duration + 0.1:
        raise ReviewError(f"{where} is outside the session duration")
    return start, end


def duration_by_input(manifest: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in manifest["inputs"]:
        if "duration_seconds" in item["input_evidence"]:
            duration = float(item["input_evidence"]["duration_seconds"])
        else:
            duration = (
                int(item["input_evidence"]["last_timestamp_ns"])
                - int(item["input_evidence"]["first_timestamp_ns"])
            ) / 1e9
        result[item["input_id"]] = duration
    return result


def validate_coverage(
    review: dict[str, Any], manifest: dict[str, Any], where: str
) -> None:
    coverage = review.get("timeline_coverage")
    if not isinstance(coverage, list):
        raise ReviewError(f"{where}.timeline_coverage must be an array")
    by_id = {
        require_text(item.get("input_id"), f"{where}.timeline_coverage.input_id"): item
        for item in coverage
        if isinstance(item, dict)
    }
    expected_ids = {item["input_id"] for item in manifest["inputs"]}
    if set(by_id) != expected_ids:
        raise ReviewError(f"{where}.timeline_coverage input identities mismatch")
    for item in manifest["inputs"]:
        input_id = item["input_id"]
        observed = by_id[input_id]
        if observed.get("full_timeline_coverage") is not True:
            raise ReviewError(f"{where}.{input_id} is not full-timeline")
        sheets = observed.get("contact_sheets_reviewed")
        if not isinstance(sheets, list):
            raise ReviewError(f"{where}.{input_id}.contact_sheets_reviewed must be an array")
        expected_names = {
            Path(sheet["path"]).name for sheet in item["contact_sheets"]
        }
        observed_names = {Path(str(path)).name for path in sheets}
        if observed_names != expected_names:
            raise ReviewError(f"{where}.{input_id} contact-sheet coverage mismatch")
        dense = observed.get("dense_frames_reviewed")
        if not isinstance(dense, list):
            raise ReviewError(f"{where}.{input_id}.dense_frames_reviewed must be an array")


def validate_review(
    review: dict[str, Any],
    *,
    manifest: dict[str, Any],
    spec: dict[str, Any],
    where: str,
    expected_role: str,
) -> None:
    if review.get("schema") != "blindassist_dual_loop_f1a_label_review_pass_v1":
        raise ReviewError(f"{where}.schema mismatch")
    if review.get("reviewer_type") != "ai_model":
        raise ReviewError(f"{where}.reviewer_type must be ai_model")
    if review.get("reviewer_role") != expected_role:
        raise ReviewError(f"{where}.reviewer_role mismatch")
    for key in (
        "reviewer_id",
        "provider",
        "model",
        "model_version",
        "review_run_id",
        "workflow_id",
    ):
        require_text(review.get(key), f"{where}.{key}")
    if review.get("workflow_id") != "dual_loop_f1a_existing_rgb_label_repair_v1":
        raise ReviewError(f"{where}.workflow_id mismatch")
    if require_sha(review.get("input_sha256"), f"{where}.input_sha256") != manifest[
        "bundle_subject_sha256"
    ]:
        raise ReviewError(f"{where}.input_sha256 mismatch")
    if require_sha(review.get("prompt_sha256"), f"{where}.prompt_sha256") != manifest[
        "prompt_sha256"
    ]:
        raise ReviewError(f"{where}.prompt_sha256 mismatch")
    if (
        review.get("isolated_context") is not True
        or review.get("other_review_visible_before_submission") is not False
        or review.get("candidate_output_visible") is not False
    ):
        raise ReviewError(f"{where} is not an isolated candidate-blind pass")
    confidence = review.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.65 <= float(confidence) <= 1.0
    ):
        raise ReviewError(f"{where}.confidence is below 0.65 or outside [0,1]")
    if review.get("abstained") is not False or review.get("abstain_reasons") not in (
        [],
        None,
    ):
        raise ReviewError(f"{where} abstained")
    validate_coverage(review, manifest, where)

    allowed_inputs = {item["input_id"] for item in manifest["inputs"]}
    durations = duration_by_input(manifest)
    allowed_positive = set(spec["label_contract"]["positive_event_types"])
    allowed_negative = set(spec["label_contract"]["negative_window_types"])
    allowed_regions = set(spec["label_contract"]["regions"])
    seen_ids: set[str] = set()

    for index, event in enumerate(review.get("events", [])):
        location = f"{where}.events[{index}]"
        if not isinstance(event, dict):
            raise ReviewError(f"{location} must be an object")
        item_id = require_text(event.get("item_id"), f"{location}.item_id")
        if item_id in seen_ids:
            raise ReviewError(f"{where} duplicate item_id: {item_id}")
        seen_ids.add(item_id)
        input_id = require_text(event.get("input_id"), f"{location}.input_id")
        if input_id not in allowed_inputs:
            raise ReviewError(f"{location}.input_id is not frozen")
        if event.get("event_type") not in allowed_positive:
            raise ReviewError(f"{location}.event_type is not allowed")
        if event.get("region") not in allowed_regions:
            raise ReviewError(f"{location}.region is not allowed")
        onset = interval(
            event.get("onset_interval_seconds"),
            f"{location}.onset_interval_seconds",
            durations[input_id],
        )
        alertable = interval(
            event.get("alertable_start_interval_seconds"),
            f"{location}.alertable_start_interval_seconds",
            durations[input_id],
        )
        clear = interval(
            event.get("end_or_clear_interval_seconds"),
            f"{location}.end_or_clear_interval_seconds",
            durations[input_id],
        )
        if onset[0] > alertable[1] or alertable[0] > clear[1]:
            raise ReviewError(f"{location} interval order is invalid")
        item_confidence = event.get("confidence")
        if not isinstance(item_confidence, (int, float)) or float(item_confidence) < 0.65:
            raise ReviewError(f"{location}.confidence is below 0.65")
        require_text(event.get("truth_provenance"), f"{location}.truth_provenance")

    minimum_negative = float(
        spec["label_contract"]["minimum_natural_negative_window_seconds"]
    )
    for index, window in enumerate(review.get("negative_windows", [])):
        location = f"{where}.negative_windows[{index}]"
        if not isinstance(window, dict):
            raise ReviewError(f"{location} must be an object")
        item_id = require_text(window.get("item_id"), f"{location}.item_id")
        if item_id in seen_ids:
            raise ReviewError(f"{where} duplicate item_id: {item_id}")
        seen_ids.add(item_id)
        input_id = require_text(window.get("input_id"), f"{location}.input_id")
        if input_id not in allowed_inputs:
            raise ReviewError(f"{location}.input_id is not frozen")
        if window.get("negative_type") not in allowed_negative:
            raise ReviewError(f"{location}.negative_type is not allowed")
        if window.get("should_alert") is not False:
            raise ReviewError(f"{location}.should_alert must be false")
        start, end = interval(
            window.get("window_interval_seconds"),
            f"{location}.window_interval_seconds",
            durations[input_id],
        )
        if end - start < minimum_negative:
            raise ReviewError(f"{location} is shorter than the frozen minimum")
        region = window.get("region")
        if region is not None and region not in allowed_regions:
            raise ReviewError(f"{location}.region is invalid")
        item_confidence = window.get("confidence")
        if not isinstance(item_confidence, (int, float)) or float(item_confidence) < 0.65:
            raise ReviewError(f"{location}.confidence is below 0.65")
        require_text(window.get("truth_provenance"), f"{location}.truth_provenance")
    if not isinstance(review.get("quarantines"), list):
        raise ReviewError(f"{where}.quarantines must be an array")


def midpoint(bounds: dict[str, Any]) -> float:
    return (float(bounds["start"]) + float(bounds["end"])) / 2.0


def event_distance(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left["input_id"] != right["input_id"] or left["event_type"] != right["event_type"]:
        return None
    alert_distance = abs(
        midpoint(left["alertable_start_interval_seconds"])
        - midpoint(right["alertable_start_interval_seconds"])
    )
    clear_distance = abs(
        midpoint(left["end_or_clear_interval_seconds"])
        - midpoint(right["end_or_clear_interval_seconds"])
    )
    if alert_distance > 3.0 or clear_distance > 4.0:
        return None
    return alert_distance + clear_distance


def window_overlap(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if (
        left["input_id"] != right["input_id"]
        or left["negative_type"] != right["negative_type"]
    ):
        return None
    left_bounds = left["window_interval_seconds"]
    right_bounds = right["window_interval_seconds"]
    overlap = min(float(left_bounds["end"]), float(right_bounds["end"])) - max(
        float(left_bounds["start"]), float(right_bounds["start"])
    )
    if overlap < 2.0:
        return None
    return overlap


def greedy_match(
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
    score,
    *,
    lower_is_better: bool,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(left_items):
        for right_index, right in enumerate(right_items):
            value = score(left, right)
            if value is not None:
                candidates.append((float(value), left_index, right_index))
    candidates.sort(reverse=not lower_is_better)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_items[left_index], right_items[right_index]))
    unmatched_left = [
        item for index, item in enumerate(left_items) if index not in used_left
    ]
    unmatched_right = [
        item for index, item in enumerate(right_items) if index not in used_right
    ]
    return matches, unmatched_left, unmatched_right


def compare_reviews(
    review_a: dict[str, Any], review_b: dict[str, Any]
) -> dict[str, Any]:
    event_matches, events_a_only, events_b_only = greedy_match(
        review_a["events"],
        review_b["events"],
        event_distance,
        lower_is_better=True,
    )
    window_matches, windows_a_only, windows_b_only = greedy_match(
        review_a["negative_windows"],
        review_b["negative_windows"],
        window_overlap,
        lower_is_better=False,
    )
    agreements: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for left, right in event_matches:
        if left["region"] != right["region"]:
            disagreements.append(
                {
                    "kind": "POSITIVE_REGION_DISAGREEMENT",
                    "review_a": left,
                    "review_b": right,
                }
            )
        else:
            agreements.append(
                {
                    "kind": "POSITIVE_EVENT",
                    "review_a": left,
                    "review_b": right,
                }
            )
    agreements.extend(
        {
            "kind": "NEGATIVE_WINDOW",
            "review_a": left,
            "review_b": right,
        }
        for left, right in window_matches
    )
    disagreements.extend(
        {"kind": "POSITIVE_A_ONLY", "review_a": item, "review_b": None}
        for item in events_a_only
    )
    disagreements.extend(
        {"kind": "POSITIVE_B_ONLY", "review_a": None, "review_b": item}
        for item in events_b_only
    )
    disagreements.extend(
        {"kind": "NEGATIVE_A_ONLY", "review_a": item, "review_b": None}
        for item in windows_a_only
    )
    disagreements.extend(
        {"kind": "NEGATIVE_B_ONLY", "review_a": None, "review_b": item}
        for item in windows_b_only
    )
    for index, item in enumerate(agreements, start=1):
        item["consensus_id"] = f"CONSENSUS-{index:03d}"
    for index, item in enumerate(disagreements, start=1):
        item["disagreement_id"] = f"DISAGREEMENT-{index:03d}"
    return {
        "agreements": agreements,
        "disagreements": disagreements,
        "status": "MODEL_CONSENSUS"
        if not disagreements
        else "INDEPENDENT_AI_ADJUDICATION_REQUIRED",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    review_a = json.loads(args.review_a.read_text(encoding="utf-8"))
    review_b = json.loads(args.review_b.read_text(encoding="utf-8"))
    validate_review(
        review_a,
        manifest=manifest,
        spec=spec,
        where="review_a",
        expected_role="gpt_task_reviewer",
    )
    validate_review(
        review_b,
        manifest=manifest,
        spec=spec,
        where="review_b",
        expected_role="codex_evidence_reviewer",
    )
    if review_a["reviewer_id"] == review_b["reviewer_id"]:
        raise ReviewError("reviewer identities must be distinct")
    comparison = compare_reviews(review_a, review_b)
    output = {
        "schema": "blindassist_dual_loop_f1a_review_comparison_v1",
        "protocol_id": spec["protocol_id"],
        "input_sha256": manifest["bundle_subject_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
        "review_a_path": str(args.review_a),
        "review_a_sha256": sha256_file(args.review_a),
        "review_b_path": str(args.review_b),
        "review_b_sha256": sha256_file(args.review_b),
        "reviewer_ids": [review_a["reviewer_id"], review_b["reviewer_id"]],
        **comparison,
    }
    write_json(args.output, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "agreement_count": len(output["agreements"]),
                "disagreement_count": len(output["disagreements"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
