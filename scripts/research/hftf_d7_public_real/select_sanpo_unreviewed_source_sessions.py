#!/usr/bin/env python3
"""Select SANPO-Real segmentation/pose windows from unseen source sessions.

This is an intake selector only.  It excludes candidate IDs with completed
reviews and excludes every raw source session represented by an existing
completed SANPO review.  It requires complete source media, segmentation, and
available pose objects, then takes at most one deterministic window per raw
session.  It never infers event truth, admission, or split roles.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


PRIMARY_REVIEW_FILES = (
    "reviews/review_a.jsonl",
    "reviews/review_b.jsonl",
    "reviews/review_c.jsonl",
    "reviews/geometry_review.jsonl",
    "reviews/counterexample_review.jsonl",
)


def _raw_session(row: dict[str, Any]) -> str:
    metadata = row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
    return str(metadata.get("raw_source_session_id") or row.get("source_id") or "")


def _frame_interval(row: dict[str, Any]) -> tuple[int, int]:
    try:
        start = int(row["start_frame_index"])
        end = int(row["end_frame_index"])
        count = int(row["frame_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"SANPO candidate has invalid frame interval: {row.get('candidate_id')}") from exc
    if start < 0 or end < start or end - start + 1 != count:
        raise ContractError(f"SANPO candidate has inconsistent frame interval: {row.get('candidate_id')}")
    return start, end


def _completed_review_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    bundle_root = root / "reviews" / "input_bundles"
    if bundle_root.is_dir():
        for path in sorted(bundle_root.glob("*/bundle_manifest.json")):
            manifest = load_json(path)
            if isinstance(manifest, dict) and manifest.get("dataset_id") == "SANPO-Real":
                ids.update(str(value) for value in manifest.get("candidate_ids", []))
    for relative in PRIMARY_REVIEW_FILES:
        path = root / relative
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            if row.get("dataset_id") == "SANPO-Real" and row.get("record_kind") == "COMPLETED_REVIEW" and row.get("review_completed") is True:
                candidate_id = str(row.get("candidate_id") or "")
                if candidate_id:
                    ids.add(candidate_id)
    return ids


def _inventory_sessions(inventory: dict[str, Any], *, require_pose: bool) -> tuple[set[str], Counter[str]]:
    records = inventory.get("records") if isinstance(inventory.get("records"), list) else None
    if records is None:
        raise ContractError("SANPO inventory has no records")
    complete: set[str] = set()
    reasons: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        session = str(record.get("source_session_id") or "")
        media = record.get("media") if isinstance(record.get("media"), dict) else {}
        try:
            frame_count = int(media.get("complete_frame_count", 0) or 0)
        except (TypeError, ValueError):
            frame_count = 0
        if not session or frame_count <= 0:
            reasons["INCOMPLETE_SOURCE_MEDIA"] += 1
            continue
        if require_pose:
            auxiliary = record.get("auxiliary") if isinstance(record.get("auxiliary"), dict) else {}
            pose = auxiliary.get("pose") if isinstance(auxiliary.get("pose"), list) else []
            if not pose:
                reasons["POSE_OBJECT_MISSING"] += 1
                continue
        complete.add(session)
    return complete, reasons


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    candidate_artifact = Path(args.candidate_artifact).resolve()
    inventory_path = Path(args.inventory).resolve()
    output_path = Path(args.output_path).resolve()
    if not candidate_artifact.is_file() or not inventory_path.is_file():
        raise ContractError("SANPO candidate artifact and inventory are required")
    if output_path.exists():
        raise ContractError(f"refusing to overwrite candidate selection: {output_path}")
    if args.max_count is not None and args.max_count <= 0:
        raise ContractError("--max-count must be positive")

    rows = load_jsonl(candidate_artifact)
    rows_by_id = {str(row.get("candidate_id") or ""): row for row in rows}
    if len(rows_by_id) != len(rows) or "" in rows_by_id:
        raise ContractError("SANPO candidate artifact has duplicate or missing candidate_id")
    inventory = load_json(inventory_path)
    if not isinstance(inventory, dict):
        raise ContractError("SANPO inventory must be an object")
    complete_sessions, inventory_reasons = _inventory_sessions(inventory, require_pose=args.require_pose)
    reviewed_ids = _completed_review_ids(output_root)
    reviewed_sessions = {
        _raw_session(rows_by_id[candidate_id])
        for candidate_id in reviewed_ids
        if candidate_id in rows_by_id and rows_by_id[candidate_id].get("dataset_id") == "SANPO-Real"
    }

    reasons: Counter[str] = Counter(inventory_reasons)
    eligible: list[dict[str, Any]] = []
    excluded_reviewed: list[str] = []
    excluded_reviewed_sessions: list[str] = []
    for row in rows:
        if row.get("dataset_id") != "SANPO-Real":
            reasons["DATASET_NOT_SANPO_REAL"] += 1
            continue
        candidate_id = str(row.get("candidate_id") or "")
        raw_session = _raw_session(row)
        if not raw_session:
            reasons["SOURCE_SESSION_MISSING"] += 1
            continue
        _frame_interval(row)
        metadata = row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
        if metadata.get("segmentation_complete") is not True:
            reasons["SEGMENTATION_INCOMPLETE"] += 1
            continue
        if raw_session not in complete_sessions:
            reasons["SOURCE_MEDIA_OR_POSE_INCOMPLETE"] += 1
            continue
        if candidate_id in reviewed_ids:
            excluded_reviewed.append(candidate_id)
            continue
        if raw_session in reviewed_sessions:
            excluded_reviewed_sessions.append(candidate_id)
            continue
        eligible.append(row)
        reasons["ELIGIBLE"] += 1

    eligible.sort(key=lambda row: (
        _raw_session(row),
        int(row.get("start_frame_index", 0)),
        str(row.get("candidate_id")),
    ))
    selected: list[dict[str, Any]] = []
    selected_sessions: set[str] = set()
    excluded_same_session: list[str] = []
    for row in eligible:
        session = _raw_session(row)
        if session in selected_sessions:
            excluded_same_session.append(str(row["candidate_id"]))
            continue
        selected.append(row)
        selected_sessions.add(session)
    if args.max_count is not None:
        selected = selected[: args.max_count]
    if not selected:
        raise ContractError("no unreviewed SANPO-Real candidate from an unseen complete source session remains")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, selected)
    selected_by_session = Counter(_raw_session(row) for row in selected)
    report = {
        "schema": "hftf_d7_public_real_sanpo_review_candidate_selection_v1",
        "record_kind": "SANPO_REVIEW_CANDIDATE_SELECTION",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_REVIEW_MATERIALIZATION",
        "candidate_artifact": {"path": str(candidate_artifact), "sha256": sha256_file(candidate_artifact)},
        "inventory_artifact": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "output_artifact": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "input_candidate_row_count": len(rows),
        "existing_completed_review_candidate_id_count": len(reviewed_ids),
        "existing_completed_review_raw_session_count": len(reviewed_sessions),
        "complete_source_session_count": len(complete_sessions),
        "eligible_unreviewed_unseen_session_count": len(eligible),
        "excluded_already_reviewed_count": len(excluded_reviewed),
        "excluded_existing_review_session_count": len(excluded_reviewed_sessions),
        "excluded_same_session_after_one_per_session_count": len(excluded_same_session),
        "selected_candidate_count": len(selected),
        "selected_raw_source_session_count": len(selected_by_session),
        "selected_candidates_by_raw_source_session": dict(sorted(selected_by_session.items())),
        "filter_reason_counts": dict(sorted(reasons.items())),
        "selection_contract": {
            "dataset_id": "SANPO-Real",
            "segmentation_complete": True,
            "complete_source_media": True,
            "pose_object_available": args.require_pose,
            "existing_completed_candidate_ids_excluded": True,
            "existing_reviewed_raw_sessions_excluded": True,
            "one_candidate_per_raw_source_session": True,
            "model_output_used_for_selection": False,
            "event_truth_inferred": False,
            "admission_assigned": False,
            "split_assigned": False,
        },
        "event_truth_inferred": False,
        "admission_assigned": False,
        "split_assigned": False,
        "notes": [
            "This artifact is a review-materialization input, not an event label.",
            "Unseen raw source session is a split/review identity safeguard, not proof of parent-event independence.",
            "Provider object hashes, pose binding, phase, ancestry, and event truth remain downstream gates.",
        ],
    }
    report_dir = output_root / "reports" / "sanpo_review_candidate_selection" / args.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "selection_report.json"
    write_json(report_path, report)
    receipt = {
        "schema": "hftf_d7_public_real_sanpo_review_candidate_selection_receipt_v1",
        "record_kind": "SANPO_REVIEW_CANDIDATE_SELECTION_RECEIPT",
        "run_id": args.run_id,
        "generated_at_utc": report["generated_at_utc"],
        "status": report["status"],
        "candidate_artifact": report["candidate_artifact"],
        "inventory_artifact": report["inventory_artifact"],
        "output_artifact": report["output_artifact"],
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "selected_candidate_count": len(selected),
        "selected_raw_source_session_count": len(selected_by_session),
        "excluded_existing_review_session_count": len(excluded_reviewed_sessions),
        "event_truth_inferred": False,
        "admission_assigned": False,
        "split_assigned": False,
    }
    receipt_path = output_root / "receipts" / f"sanpo_review_candidate_selection_receipt_{args.run_id}.json"
    if receipt_path.exists():
        raise ContractError(f"refusing to overwrite selection receipt: {receipt_path}")
    write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-count", type=int)
    parser.add_argument("--require-pose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import json

    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
