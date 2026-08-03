#!/usr/bin/env python3
"""Select model-blind THOR-MAGNI windows outside prior review neighborhoods.

This selector is an intake/review-assignment step only.  It excludes candidate
identities represented by existing THOR-MAGNI review bundles or completed
review rows, requires the source-synchronized QTM timestamp contract and native
geometry marker, and excludes windows that overlap or sit within a frozen time
gap of a prior review in the same source session.  At most one candidate per
source session is selected for each batch.  It never infers event truth,
admission, independence, or a split role.
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


def _interval(row: dict[str, Any]) -> tuple[int, int]:
    try:
        start = int(row["start_timestamp_ns"])
        end = int(row["end_timestamp_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"THOR-MAGNI candidate has invalid timestamp interval: {row.get('candidate_id')}") from exc
    if end <= start:
        raise ContractError(f"THOR-MAGNI candidate has non-positive timestamp interval: {row.get('candidate_id')}")
    return start, end


def _reviewed_ids(root: Path, *, ignored_batch_ids: set[str]) -> tuple[set[str], int]:
    bundle_root = root / "reviews" / "input_bundles"
    if not bundle_root.is_dir():
        raise ContractError(f"review bundle root missing: {bundle_root}")
    ids: set[str] = set()
    bundle_count = 0
    for path in sorted(bundle_root.glob("*/bundle_manifest.json")):
        if path.parent.name in ignored_batch_ids:
            continue
        manifest = load_json(path)
        if not isinstance(manifest, dict):
            raise ContractError(f"review bundle manifest is not an object: {path}")
        if manifest.get("dataset_id") == "THOR-MAGNI":
            ids.update(str(value) for value in manifest.get("candidate_ids", []))
            bundle_count += 1
    for relative in PRIMARY_REVIEW_FILES:
        path = root / relative
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            if (
                row.get("dataset_id") == "THOR-MAGNI"
                and row.get("record_kind") == "COMPLETED_REVIEW"
                and row.get("review_completed") is True
            ):
                candidate_id = str(row.get("candidate_id") or "")
                if candidate_id:
                    ids.add(candidate_id)
    return ids, bundle_count


def _has_overlap_or_adjacency(
    row: dict[str, Any],
    *,
    prior_rows: list[dict[str, Any]],
    min_adjacency_gap_ns: int,
) -> bool:
    start, end = _interval(row)
    for old in prior_rows:
        old_start, old_end = _interval(old)
        if max(start, old_start) < min(end, old_end):
            return True
        gap = min(abs(start - old_end), abs(old_start - end))
        if gap <= min_adjacency_gap_ns:
            return True
    return False


def _is_eligible_source_row(row: dict[str, Any]) -> tuple[bool, str | None]:
    if row.get("dataset_id") != "THOR-MAGNI":
        return False, "DATASET_NOT_THOR_MAGNI"
    if not str(row.get("candidate_id") or ""):
        raise ContractError("THOR-MAGNI candidate has no candidate_id")
    if not str(row.get("source_session_id") or ""):
        return False, "SOURCE_SESSION_MISSING"
    if row.get("native_geometry_available") is not True:
        return False, "NATIVE_GEOMETRY_UNAVAILABLE"
    if row.get("timestamp_semantics") != "SOURCE_SYNCHRONIZED_QTM_TIME":
        return False, "SOURCE_SYNC_TIMESTAMP_CONTRACT_MISSING"
    if row.get("model_output_visible_to_selector") is not False:
        return False, "MODEL_OUTPUT_VISIBILITY_NOT_FALSE"
    _interval(row)
    return True, None


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    candidate_artifact = Path(args.candidate_artifact).resolve()
    review_bundles_root = Path(args.review_bundles_root).resolve()
    output_path = Path(args.output_path).resolve()
    if not candidate_artifact.is_file():
        raise ContractError(f"candidate artifact missing: {candidate_artifact}")
    if output_path.exists():
        raise ContractError(f"refusing to overwrite candidate selection: {output_path}")
    if args.max_count is not None and args.max_count <= 0:
        raise ContractError("--max-count must be positive")
    if args.min_adjacency_gap_ns < 0:
        raise ContractError("--min-adjacency-gap-ns must be non-negative")

    rows = load_jsonl(candidate_artifact)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in rows_by_id:
            raise ContractError("THOR-MAGNI candidate artifact has duplicate or missing candidate_id")
        rows_by_id[candidate_id] = row

    reviewed_ids, bundle_count = _reviewed_ids(
        output_root,
        ignored_batch_ids={str(value) for value in args.ignore_batch_id},
    )
    reviewed_rows_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reviewed_ids_in_artifact = 0
    for candidate_id in reviewed_ids:
        old = rows_by_id.get(candidate_id)
        if old is None:
            continue
        reviewed_ids_in_artifact += 1
        if old.get("dataset_id") != "THOR-MAGNI":
            continue
        session = str(old.get("source_session_id") or "")
        if not session:
            raise ContractError(f"reviewed THOR-MAGNI candidate lacks source session: {candidate_id}")
        _interval(old)
        reviewed_rows_by_session[session].append(old)

    reason_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    excluded_reviewed: list[str] = []
    excluded_adjacency: list[str] = []
    for row in rows:
        ok, reason = _is_eligible_source_row(row)
        if not ok:
            reason_counts[str(reason)] += 1
            continue
        candidate_id = str(row["candidate_id"])
        session = str(row["source_session_id"])
        if candidate_id in reviewed_ids:
            excluded_reviewed.append(candidate_id)
            continue
        if _has_overlap_or_adjacency(
            row,
            prior_rows=reviewed_rows_by_session.get(session, []),
            min_adjacency_gap_ns=args.min_adjacency_gap_ns,
        ):
            excluded_adjacency.append(candidate_id)
            continue
        eligible.append(row)
        reason_counts["ELIGIBLE"] += 1

    eligible.sort(key=lambda row: (
        str(row.get("source_session_id") or ""),
        int(row.get("start_timestamp_ns", 0)),
        str(row.get("candidate_id") or ""),
    ))
    selected: list[dict[str, Any]] = []
    selected_sessions: set[str] = set()
    excluded_same_session: list[str] = []
    for row in eligible:
        session = str(row["source_session_id"])
        if session in selected_sessions:
            excluded_same_session.append(str(row["candidate_id"]))
            continue
        selected.append(row)
        selected_sessions.add(session)
        if args.max_count is not None and len(selected) >= args.max_count:
            break
    if not selected:
        raise ContractError("no unreviewed THOR-MAGNI candidate outside prior review neighborhoods remains")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, selected)
    selected_by_session = Counter(str(row["source_session_id"]) for row in selected)
    report = {
        "schema": "hftf_d7_public_real_thor_magni_review_candidate_selection_v1",
        "record_kind": "THOR_MAGNI_REVIEW_CANDIDATE_SELECTION",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_REVIEW_MATERIALIZATION",
        "candidate_artifact": {"path": str(candidate_artifact), "sha256": sha256_file(candidate_artifact)},
        "output_artifact": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "review_bundles_root": str(review_bundles_root),
        "existing_review_bundle_count": bundle_count,
        "existing_review_candidate_id_count": len(reviewed_ids),
        "existing_review_candidate_id_count_in_artifact": reviewed_ids_in_artifact,
        "existing_review_source_session_count": len(reviewed_rows_by_session),
        "input_candidate_row_count": len(rows),
        "eligible_outside_prior_review_neighborhood_count": len(eligible),
        "excluded_already_reviewed_count": len(excluded_reviewed),
        "excluded_reviewed_overlap_or_adjacency_count": len(excluded_adjacency),
        "excluded_within_session_after_one_per_session_count": len(excluded_same_session),
        "selected_candidate_count": len(selected),
        "selected_source_session_count": len(selected_by_session),
        "selected_candidates_by_source_session": dict(sorted(selected_by_session.items())),
        "input_filter_reason_counts": dict(sorted(reason_counts.items())),
        "selection_contract": {
            "dataset_id": "THOR-MAGNI",
            "source_synchronized_timestamp_semantics": "SOURCE_SYNCHRONIZED_QTM_TIME",
            "native_geometry_required": True,
            "existing_review_candidate_ids_excluded": True,
            "reviewed_overlap_or_adjacency_excluded": True,
            "one_candidate_per_source_session": True,
            "min_adjacency_gap_ns": args.min_adjacency_gap_ns,
            "model_output_used_for_selection": False,
            "event_truth_inferred": False,
            "admission_assigned": False,
            "independence_assigned": False,
            "split_assigned": False,
        },
        "event_truth_inferred": False,
        "admission_assigned": False,
        "independence_assigned": False,
        "split_assigned": False,
        "notes": [
            "This artifact is a review-materialization input, not an event label.",
            "Temporal exclusion is an identity/overlap safeguard and does not prove parent-event independence.",
            "QTM pose/tracks/intrinsics remain source-native Development evidence; missing depth or segmentation remains NOT_EVALUABLE.",
            "Member-specific license terms and any event-use authority remain review-gated.",
        ],
    }
    report_dir = output_root / "reports" / "thor_magni_review_candidate_selection" / args.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "selection_report.json"
    write_json(report_path, report)
    receipt = {
        "schema": "hftf_d7_public_real_thor_magni_review_candidate_selection_receipt_v1",
        "record_kind": "THOR_MAGNI_REVIEW_CANDIDATE_SELECTION_RECEIPT",
        "run_id": args.run_id,
        "generated_at_utc": report["generated_at_utc"],
        "status": report["status"],
        "candidate_artifact": report["candidate_artifact"],
        "output_artifact": report["output_artifact"],
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "selected_candidate_count": len(selected),
        "selected_source_session_count": len(selected_by_session),
        "excluded_already_reviewed_count": len(excluded_reviewed),
        "excluded_reviewed_overlap_or_adjacency_count": len(excluded_adjacency),
        "event_truth_inferred": False,
        "admission_assigned": False,
        "independence_assigned": False,
        "split_assigned": False,
    }
    receipt_path = output_root / "receipts" / f"thor_magni_review_candidate_selection_receipt_{args.run_id}.json"
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
    parser.add_argument("--review-bundles-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-count", type=int)
    parser.add_argument("--min-adjacency-gap-ns", type=int, default=1_000_000_000)
    parser.add_argument("--ignore-batch-id", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    import json

    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
