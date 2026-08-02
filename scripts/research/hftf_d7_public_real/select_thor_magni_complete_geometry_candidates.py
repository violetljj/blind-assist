#!/usr/bin/env python3
"""Select unreviewed THOR-MAGNI windows with complete synchronized geometry.

The selector only creates a review candidate artifact.  It never assigns an
event bucket, truth status, phase interval, admission, or split role.  Existing
review bundles are treated as consumed identity evidence and are excluded
before the new artifact is written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


def _eligible(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("dataset_id") != "THOR-MAGNI":
        return False, "DATASET_NOT_THOR_MAGNI"
    metadata = row.get("source_metadata")
    if not isinstance(metadata, dict):
        return False, "SOURCE_METADATA_MISSING"
    if metadata.get("camera_centroid_complete") is not True:
        return False, "CAMERA_CENTROID_INCOMPLETE"
    if metadata.get("scene_frame_complete") is not True:
        return False, "SCENE_FRAME_INCOMPLETE"
    try:
        scene_missing_rows = int(metadata.get("scene_frame_missing_row_count", -1))
        qtm_duplicate_rows = int(metadata.get("qtm_window_duplicate_row_count", -1))
    except (TypeError, ValueError):
        return False, "GEOMETRY_COMPLETENESS_FIELDS_INVALID"
    if scene_missing_rows != 0:
        return False, "SCENE_FRAME_MISSING_ROWS"
    if qtm_duplicate_rows != 0:
        return False, "QTM_WINDOW_DUPLICATE_ROWS"
    if row.get("model_output_visible_to_selector") is not False:
        return False, "MODEL_OUTPUT_VISIBLE_TO_SELECTOR"
    if row.get("native_geometry_used_for_selection") is not False:
        return False, "NATIVE_GEOMETRY_USED_FOR_SELECTION"
    if not row.get("frame_ids"):
        return False, "FRAME_IDS_MISSING"
    if not str(row.get("source_session_id") or ""):
        return False, "SOURCE_SESSION_MISSING"
    return True, "ELIGIBLE"


def _reviewed_ids(
    review_bundles_root: Path,
    *,
    ignored_batch_ids: set[str],
) -> tuple[set[str], int]:
    if not review_bundles_root.is_dir():
        raise ContractError(f"review bundle root missing: {review_bundles_root}")
    ids: set[str] = set()
    bundle_count = 0
    for path in sorted(review_bundles_root.glob("*/bundle_manifest.json")):
        if path.parent.name in ignored_batch_ids:
            continue
        manifest = load_json(path)
        if not isinstance(manifest, dict):
            raise ContractError(f"review bundle manifest is not an object: {path}")
        ids.update(str(value) for value in manifest.get("candidate_ids", []))
        bundle_count += 1
    return ids, bundle_count


def _has_reviewed_overlap_or_adjacency(
    row: dict[str, Any],
    *,
    reviewed_rows_by_session: dict[str, list[dict[str, Any]]],
    min_adjacency_gap_ns: int,
) -> bool:
    session = str(row.get("source_session_id") or "")
    start = int(row.get("start_timestamp_ns"))
    end = int(row.get("end_timestamp_ns"))
    for old in reviewed_rows_by_session.get(session, []):
        old_start = int(old.get("start_timestamp_ns"))
        old_end = int(old.get("end_timestamp_ns"))
        if max(start, old_start) < min(end, old_end):
            return True
        gap = min(abs(start - old_end), abs(old_start - end))
        if gap <= min_adjacency_gap_ns:
            return True
    return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    candidate_artifact = Path(args.candidate_artifact).resolve()
    review_bundles_root = Path(args.review_bundles_root).resolve()
    output_path = Path(args.output_path).resolve()
    ignored_batch_ids = {str(value) for value in args.ignore_batch_id}
    if not candidate_artifact.is_file():
        raise ContractError(f"candidate artifact missing: {candidate_artifact}")
    if output_path.exists():
        raise ContractError(f"refusing to overwrite candidate selection: {output_path}")

    rows = load_jsonl(candidate_artifact)
    rows_by_id = {str(row.get("candidate_id")): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ContractError("candidate artifact has duplicate or missing candidate_id")
    reviewed_ids, bundle_count = _reviewed_ids(
        review_bundles_root,
        ignored_batch_ids=ignored_batch_ids,
    )
    reviewed_rows_by_session: dict[str, list[dict[str, Any]]] = {}
    for candidate_id in reviewed_ids:
        old = rows_by_id.get(candidate_id)
        if old is None:
            continue
        reviewed_rows_by_session.setdefault(str(old.get("source_session_id") or ""), []).append(old)
    reason_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    excluded_reviewed: list[str] = []
    excluded_adjacency: list[str] = []
    for row in rows:
        ok, reason = _eligible(row)
        reason_counts[reason] += 1
        if not ok:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise ContractError("eligible candidate has no candidate_id")
        if candidate_id in reviewed_ids:
            excluded_reviewed.append(candidate_id)
            continue
        if _has_reviewed_overlap_or_adjacency(
            row,
            reviewed_rows_by_session=reviewed_rows_by_session,
            min_adjacency_gap_ns=args.min_adjacency_gap_ns,
        ):
            excluded_adjacency.append(candidate_id)
            continue
        eligible.append(row)

    eligible.sort(key=lambda row: (
        str(row.get("source_session_id") or ""),
        int(row.get("start_timestamp_ns", 0) or 0),
        str(row.get("candidate_id") or ""),
    ))
    selected_before_limit: list[dict[str, Any]] = []
    selected_rows_by_session: dict[str, list[dict[str, Any]]] = {}
    excluded_internal_adjacency: list[str] = []
    for row in eligible:
        if _has_reviewed_overlap_or_adjacency(
            row,
            reviewed_rows_by_session=selected_rows_by_session,
            min_adjacency_gap_ns=args.min_adjacency_gap_ns,
        ):
            excluded_internal_adjacency.append(str(row["candidate_id"]))
            continue
        selected_before_limit.append(row)
        selected_rows_by_session.setdefault(str(row.get("source_session_id") or ""), []).append(row)
    eligible = selected_before_limit
    if args.max_count is not None:
        if args.max_count <= 0:
            raise ContractError("--max-count must be positive")
        eligible = eligible[: args.max_count]
    if not eligible:
        raise ContractError("no unreviewed complete-geometry THOR-MAGNI candidates remain")
    candidate_ids = [str(row["candidate_id"]) for row in eligible]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ContractError("eligible candidate IDs are duplicated")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, eligible)
    by_session = Counter(str(row["source_session_id"]) for row in eligible)
    report = {
        "schema": "hftf_d7_public_real_thor_complete_geometry_candidate_selection_v1",
        "record_kind": "THOR_COMPLETE_GEOMETRY_CANDIDATE_SELECTION",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_REVIEW_MATERIALIZATION",
        "candidate_artifact": {"path": str(candidate_artifact), "sha256": sha256_file(candidate_artifact)},
        "output_artifact": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "review_bundles_root": str(review_bundles_root),
        "existing_review_bundle_count": bundle_count,
        "ignored_review_batch_ids": sorted(ignored_batch_ids),
        "existing_review_candidate_id_count": len(reviewed_ids),
        "input_candidate_row_count": len(rows),
        "eligible_before_review_exclusion_count": reason_counts.get("ELIGIBLE", 0),
        "excluded_already_reviewed_count": len(excluded_reviewed),
        "excluded_reviewed_overlap_or_adjacency_count": len(excluded_adjacency),
        "excluded_within_batch_overlap_or_adjacency_count": len(excluded_internal_adjacency),
        "selected_candidate_count": len(eligible),
        "selected_candidate_ids": candidate_ids,
        "selected_source_session_count": len(by_session),
        "selected_candidates_by_source_session": dict(sorted(by_session.items())),
        "input_filter_reason_counts": dict(sorted(reason_counts.items())),
        "selection_contract": {
            "dataset_id": "THOR-MAGNI",
            "camera_centroid_complete": True,
            "scene_frame_complete": True,
            "scene_frame_missing_row_count": 0,
            "qtm_window_duplicate_row_count": 0,
            "model_output_visible_to_selector": False,
            "native_geometry_used_for_selection": False,
            "existing_review_candidate_ids_excluded": True,
            "reviewed_overlap_or_adjacency_excluded": True,
            "within_batch_overlap_or_adjacency_excluded": True,
            "min_adjacency_gap_ns": args.min_adjacency_gap_ns,
        },
        "event_truth_inferred": False,
        "admission_assigned": False,
        "split_assigned": False,
        "notes": [
            "This artifact is a review-materialization input, not an event label.",
            "Complete synchronized geometry is necessary evidence but does not establish event truth or phase by itself.",
            "Near-duplicate, ancestry, and parent-event adjacency remain downstream review gates.",
        ],
    }
    report_dir = output_root / "reports" / "thor_complete_geometry_candidate_selection" / args.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "selection_report.json"
    write_json(report_path, report)
    receipt = {
        "schema": "hftf_d7_public_real_thor_complete_geometry_candidate_selection_receipt_v1",
        "record_kind": "THOR_COMPLETE_GEOMETRY_CANDIDATE_SELECTION_RECEIPT",
        "run_id": args.run_id,
        "generated_at_utc": report["generated_at_utc"],
        "status": report["status"],
        "candidate_artifact": report["candidate_artifact"],
        "output_artifact": report["output_artifact"],
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "selected_candidate_count": len(eligible),
        "selected_source_session_count": len(by_session),
        "excluded_already_reviewed_count": len(excluded_reviewed),
        "excluded_reviewed_overlap_or_adjacency_count": len(excluded_adjacency),
        "excluded_within_batch_overlap_or_adjacency_count": len(excluded_internal_adjacency),
        "event_truth_inferred": False,
        "admission_assigned": False,
    }
    receipt_path = output_root / "receipts" / f"thor_complete_geometry_candidate_selection_receipt_{args.run_id}.json"
    write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--review-bundles-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-count", type=int)
    parser.add_argument(
        "--min-adjacency-gap-ns",
        type=int,
        default=1_000_000_000,
        help="exclude a candidate whose same-session gap to a reviewed window is at most this value",
    )
    parser.add_argument(
        "--ignore-batch-id",
        action="append",
        default=[],
        help="ignore a pre-existing unreviewed bundle when recomputing a corrected selection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
