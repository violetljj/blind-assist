from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_ID = "RISKSEG_R0_EVENT_EVAL_V1"
REVIEW_SCHEMA = "blindassist.riskseg_r0.event_truth_review.v1"
ADJUDICATION_SCHEMA = "blindassist.riskseg_r0.event_truth_adjudication.v1"
BUCKET_FLOORS = {
    "blocking_obstacle_positive": 8,
    "boundary_level_change_positive": 8,
    "parallel_curb_negative": 7,
    "normal_walkable_negative": 7,
}
POSITIVE_BUCKETS = {
    "blocking_obstacle_positive",
    "boundary_level_change_positive",
}
REVIEW_BUCKETS = set(BUCKET_FLOORS) | {"REJECT_AMBIGUOUS"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def excluded_sessions(ledger: dict[str, Any]) -> set[str]:
    roles = ledger["roles"]
    result: set[str] = set()
    for role in ("train", "dev"):
        result.update(key.split(":", 1)[1] for key in roles[role]["sessions"])
    result.update(item["source_session_id"] for item in roles["fixed_regression"]["events"])
    return result


def load_review(
    path: Path,
    expected_role: str,
    candidate_index_sha256: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError(f"{path.name}: review schema mismatch")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{path.name}: protocol mismatch")
    if value.get("reviewer_role") != expected_role:
        raise ValueError(f"{path.name}: reviewer role mismatch")
    if value.get("candidate_index_sha256") != candidate_index_sha256:
        raise ValueError(f"{path.name}: candidate index hash mismatch")
    if (
        value.get("isolated_context") is not True
        or value.get("rgb_only") is not True
        or value.get("candidate_output_visible") is not False
        or value.get("other_review_visible_before_submission") is not False
    ):
        raise ValueError(f"{path.name}: output-blind isolation contract failed")
    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path.name}: empty review items")
    return value


def validate_interval(value: Any, frame_count: int, where: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] < value[0]
        or value[1] >= frame_count
    ):
        raise ValueError(f"{where}: invalid frame interval")
    return value


def validate_review_coverage(
    review: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    where: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in review["items"]:
        candidate_id = item.get("event_candidate_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in candidates
            or candidate_id in result
        ):
            raise ValueError(f"{where}: invalid or duplicate event candidate id")
        candidate = candidates[candidate_id]
        if item.get("source_session_id") != candidate["source_session_id"]:
            raise ValueError(f"{where}: source session mismatch for {candidate_id}")
        if item.get("bucket") not in REVIEW_BUCKETS:
            raise ValueError(f"{where}: invalid bucket for {candidate_id}")
        confidence = item.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.65 <= float(confidence) <= 1
        ):
            raise ValueError(f"{where}: invalid confidence for {candidate_id}")
        result[candidate_id] = item
    if set(result) != set(candidates):
        missing = sorted(set(candidates) - set(result))
        extra = sorted(set(result) - set(candidates))
        raise ValueError(
            f"{where}: review coverage mismatch; missing={missing}, extra={extra}"
        )
    return result


def validate_final_items(
    items: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    blocked: set[str],
) -> tuple[Counter[str], Counter[str]]:
    if len(items) < 30:
        raise ValueError("HOLD_EVENT_EVAL_DATA: fewer than 30 parent events")
    ids: set[str] = set()
    candidate_ids: set[str] = set()
    bucket_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    bucket_sessions: dict[str, set[str]] = defaultdict(set)
    session_windows: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for item in items:
        event_id = item.get("parent_event_id")
        candidate_id = item.get("event_candidate_id")
        if not isinstance(event_id, str) or not event_id or event_id in ids:
            raise ValueError(f"invalid or duplicate parent_event_id: {event_id!r}")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in candidate_ids
            or candidate_id not in candidates
        ):
            raise ValueError(f"{event_id}: invalid or duplicate event_candidate_id")
        ids.add(event_id)
        candidate_ids.add(candidate_id)
        candidate = candidates[candidate_id]
        session_id = candidate["source_session_id"]
        if session_id in blocked:
            raise ValueError(f"{event_id}: excluded source session {session_id}")
        bucket = item.get("bucket")
        if bucket not in BUCKET_FLOORS:
            raise ValueError(f"{event_id}: invalid bucket {bucket!r}")
        confidence = item.get("truth_confidence")
        if (
            not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.65 <= float(confidence) <= 1
        ):
            raise ValueError(f"{event_id}: truth confidence must be 0.65..1")
        if bucket in POSITIVE_BUCKETS:
            alertable = validate_interval(
                item.get("alertable_interval_frames"),
                int(candidate["frame_count"]),
                f"{event_id} alertable",
            )
            passed = validate_interval(
                item.get("passed_interval_frames"),
                int(candidate["frame_count"]),
                f"{event_id} passed",
            )
            if passed[0] <= alertable[1]:
                raise ValueError(f"{event_id}: passed interval must follow alertable")
        elif (
            item.get("alertable_interval_frames") is not None
            or item.get("passed_interval_frames") is not None
        ):
            raise ValueError(f"{event_id}: negative event cannot have positive intervals")
        bucket_counts[bucket] += 1
        session_counts[session_id] += 1
        bucket_sessions[bucket].add(session_id)
        session_windows[session_id].append(
            (
                int(candidate["source_frame_start"]),
                int(candidate["source_frame_end"]),
                event_id,
            )
        )
    for bucket, floor in BUCKET_FLOORS.items():
        if bucket_counts[bucket] < floor:
            raise ValueError(
                f"HOLD_EVENT_EVAL_DATA: {bucket} has {bucket_counts[bucket]}, needs {floor}"
            )
        if len(bucket_sessions[bucket]) < 2:
            raise ValueError(f"HOLD_EVENT_EVAL_DATA: {bucket} has fewer than 2 sessions")
    if len(session_counts) < 8:
        raise ValueError("HOLD_EVENT_EVAL_DATA: fewer than 8 source sessions")
    total = len(items)
    for session_id, count in session_counts.items():
        if count / total > 0.25:
            raise ValueError(f"{session_id}: exceeds 25% of total events")
    for bucket in BUCKET_FLOORS:
        denominator = bucket_counts[bucket]
        per_session = Counter(
            candidates[item["event_candidate_id"]]["source_session_id"]
            for item in items
            if item["bucket"] == bucket
        )
        for session_id, count in per_session.items():
            if count / denominator > 0.5:
                raise ValueError(
                    f"{session_id}: exceeds 50% of bucket {bucket}"
                )
    for session_id, windows in session_windows.items():
        windows.sort()
        for previous, current in zip(windows, windows[1:]):
            if previous[1] >= current[0]:
                raise ValueError(
                    f"{session_id}: event windows overlap: {previous[2]} and {current[2]}"
                )
    return bucket_counts, session_counts


def freeze(
    ledger_path: Path,
    candidate_index_path: Path,
    review_a_path: Path,
    review_b_path: Path,
    adjudication_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    index = json.loads(candidate_index_path.read_text(encoding="utf-8"))
    if (
        index.get("schema_version")
        != "blindassist.riskseg_r0.event_candidate_index.v1"
        or index.get("protocol_id") != PROTOCOL_ID
        or index.get("candidate_output_visible") is not False
    ):
        raise ValueError("candidate index binding or output-blind state mismatch")
    candidate_index_sha = sha256_file(candidate_index_path)
    review_a = load_review(
        review_a_path,
        "primary_rgb_reviewer",
        candidate_index_sha,
    )
    review_b = load_review(
        review_b_path,
        "secondary_rgb_reviewer",
        candidate_index_sha,
    )
    if review_a.get("reviewer_id") == review_b.get("reviewer_id"):
        raise ValueError("independent reviewer identities must differ")
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if (
        adjudication.get("schema_version") != ADJUDICATION_SCHEMA
        or adjudication.get("protocol_id") != PROTOCOL_ID
        or adjudication.get("candidate_output_visible") is not False
        or adjudication.get("other_reviews_visible") is not True
        or adjudication.get("review_sha256s")
        != [sha256_file(review_a_path), sha256_file(review_b_path)]
        or adjudication.get("candidate_index_sha256") != candidate_index_sha
    ):
        raise ValueError("adjudication binding or visibility contract mismatch")
    candidates = {
        item["event_candidate_id"]: item for item in index.get("items", [])
    }
    if len(candidates) != len(index.get("items", [])):
        raise ValueError("duplicate event candidate ids")
    review_a_items = validate_review_coverage(review_a, candidates, review_a_path.name)
    review_b_items = validate_review_coverage(review_b, candidates, review_b_path.name)
    final_items = adjudication.get("final_items")
    if not isinstance(final_items, list):
        raise ValueError("adjudication final_items missing")
    for item in final_items:
        candidate_id = item.get("event_candidate_id")
        if candidate_id not in review_a_items or candidate_id not in review_b_items:
            raise ValueError(
                f"adjudication item was not covered by both reviews: {candidate_id!r}"
            )
        review_a_item = review_a_items[candidate_id]
        review_b_item = review_b_items[candidate_id]
        reviewed_buckets = {review_a_item["bucket"], review_b_item["bucket"]}
        if len(reviewed_buckets) == 1:
            reviewed_bucket = next(iter(reviewed_buckets))
            if reviewed_bucket == "REJECT_AMBIGUOUS":
                raise ValueError(
                    f"{candidate_id}: both reviewers rejected the candidate"
                )
            if item.get("bucket") != reviewed_bucket:
                raise ValueError(
                    f"{candidate_id}: adjudication contradicts reviewer bucket agreement"
                )
        else:
            reason = item.get("adjudication_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"{candidate_id}: material reviewer disagreement lacks adjudication_reason"
                )
    bucket_counts, session_counts = validate_final_items(
        final_items,
        candidates,
        excluded_sessions(ledger),
    )
    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    ledger_out = temporary / "truth_ledger.jsonl"
    with ledger_out.open("w", encoding="utf-8", newline="\n") as stream:
        for item in sorted(final_items, key=lambda value: value["parent_event_id"]):
            candidate = candidates[item["event_candidate_id"]]
            row = {
                "schema_version": "blindassist.riskseg_r0.event_truth_item.v1",
                **item,
                "source_session_id": candidate["source_session_id"],
                "sequence_id": candidate["sequence_id"],
                "source_frame_start": candidate["source_frame_start"],
                "source_frame_end": candidate["source_frame_end"],
                "rgb_sha256s": candidate["rgb_sha256s"],
                "source_mask_sha256s": candidate["source_mask_sha256s"],
                "truth_provenance": "two_isolated_rgb_reviews_plus_output_blind_adjudication",
                "outcome_access_state": "PIDNET_YOLO_AND_ORACLE_UNOPENED",
            }
            stream.write(canonical_json(row))
    receipt = {
        "schema_version": "blindassist.riskseg_r0.cohort_freeze_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "EVENT_EVAL_FROZEN_ADEQUATE",
        "parent_event_count": len(final_items),
        "source_session_count": len(session_counts),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "contract_ledger_sha256": sha256_file(ledger_path),
        "candidate_index_sha256": candidate_index_sha,
        "review_a_sha256": sha256_file(review_a_path),
        "review_b_sha256": sha256_file(review_b_path),
        "adjudication_sha256": sha256_file(adjudication_path),
        "truth_ledger_sha256": sha256_file(ledger_out),
        "implementation_sha256": sha256_file(Path(__file__)),
        "candidate_output_visible_before_freeze": False,
        "next_allowed_action": "PIDNet-S technical preflight",
    }
    (temporary / "cohort_freeze_receipt.json").write_text(
        canonical_json(receipt),
        encoding="utf-8",
    )
    temporary.replace(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-ledger", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.contract_ledger,
                args.candidate_index,
                args.review_a,
                args.review_b,
                args.adjudication,
                args.output,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
