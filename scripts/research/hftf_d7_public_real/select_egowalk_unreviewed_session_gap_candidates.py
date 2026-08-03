#!/usr/bin/env python3
"""Select model-blind EgoWalk windows outside prior review neighborhoods.

The selector is an intake/review assignment step only.  It excludes candidate
identities already present in any review bundle and excludes same-session
windows that overlap or sit within a frozen temporal gap of an existing review
window.  At most one candidate per source session is selected so a fresh batch
does not concentrate on one recording.  It never assigns event truth,
admission, or a split role.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


def _interval(row: dict[str, Any]) -> tuple[int, int]:
    try:
        start = int(row["start_timestamp_ns"])
        end = int(row["end_timestamp_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"EgoWalk candidate has invalid timestamp interval: {row.get('candidate_id')}") from exc
    if end <= start:
        raise ContractError(f"EgoWalk candidate has non-positive timestamp interval: {row.get('candidate_id')}")
    return start, end


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


def _reviewed_ids(review_bundles_root: Path, *, ignored_batch_ids: set[str]) -> tuple[set[str], int]:
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
    rows_by_id = {str(row.get("candidate_id")): row for row in rows}
    if len(rows_by_id) != len(rows) or "None" in rows_by_id or "" in rows_by_id:
        raise ContractError("candidate artifact has duplicate or missing candidate_id")
    reviewed_ids, bundle_count = _reviewed_ids(
        review_bundles_root,
        ignored_batch_ids={str(value) for value in args.ignore_batch_id},
    )
    reviewed_rows_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_id in reviewed_ids:
        old = rows_by_id.get(candidate_id)
        if old is not None and old.get("dataset_id") == "EgoWalk":
            session = str(old.get("source_session_id") or "")
            if not session:
                raise ContractError(f"reviewed EgoWalk candidate lacks source session: {candidate_id}")
            _interval(old)
            reviewed_rows_by_session[session].append(old)

    reason_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    excluded_reviewed: list[str] = []
    excluded_adjacency: list[str] = []
    for row in rows:
        if row.get("dataset_id") != "EgoWalk":
            reason_counts["DATASET_NOT_EGOWALK"] += 1
            continue
        candidate_id = str(row.get("candidate_id") or "")
        session = str(row.get("source_session_id") or "")
        if not candidate_id:
            raise ContractError("EgoWalk candidate has no candidate_id")
        if not session:
            reason_counts["SOURCE_SESSION_MISSING"] += 1
            continue
        _interval(row)
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

    # Deterministic one-per-source-session selection.  The first window is
    # stable by source identity and source-time, not by any model score.
    eligible.sort(key=lambda row: (
        str(row.get("source_session_id") or ""),
        int(row.get("start_timestamp_ns", 0)),
        str(row.get("candidate_id") or ""),
    ))
    selected: list[dict[str, Any]] = []
    selected_sessions: set[str] = set()
    excluded_session_concentration: list[str] = []
    for row in eligible:
        session = str(row["source_session_id"])
        if session in selected_sessions:
            excluded_session_concentration.append(str(row["candidate_id"]))
            continue
        selected.append(row)
        selected_sessions.add(session)
    if args.max_count is not None:
        selected = selected[: args.max_count]
    if not selected:
        raise ContractError("no unreviewed EgoWalk candidates outside prior review neighborhoods remain")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, selected)
    by_session = Counter(str(row["source_session_id"]) for row in selected)
    report = {
        "schema": "hftf_d7_public_real_egowalk_review_candidate_selection_v1",
        "record_kind": "EGOWALK_REVIEW_CANDIDATE_SELECTION",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_REVIEW_MATERIALIZATION",
        "candidate_artifact": {"path": str(candidate_artifact), "sha256": sha256_file(candidate_artifact)},
        "output_artifact": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "review_bundles_root": str(review_bundles_root),
        "existing_review_bundle_count": bundle_count,
        "existing_review_candidate_id_count": len(reviewed_ids),
        "input_candidate_row_count": len(rows),
        "eligible_outside_prior_review_neighborhood_count": len(eligible),
        "excluded_already_reviewed_count": len(excluded_reviewed),
        "excluded_reviewed_overlap_or_adjacency_count": len(excluded_adjacency),
        "excluded_within_session_after_one_per_session_count": len(excluded_session_concentration),
        "selected_candidate_count": len(selected),
        "selected_source_session_count": len(by_session),
        "selected_candidates_by_source_session": dict(sorted(by_session.items())),
        "input_filter_reason_counts": dict(sorted(reason_counts.items())),
        "selection_contract": {
            "dataset_id": "EgoWalk",
            "existing_review_candidate_ids_excluded": True,
            "reviewed_overlap_or_adjacency_excluded": True,
            "one_candidate_per_source_session": True,
            "min_adjacency_gap_ns": args.min_adjacency_gap_ns,
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
            "Temporal exclusion is an identity/overlap safeguard and does not prove parent-event independence.",
            "Ancestry, near-duplicate, source terms, phase, and event truth remain downstream gates.",
        ],
    }
    report_dir = output_root / "reports" / "egowalk_review_candidate_selection" / args.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "selection_report.json"
    write_json(report_path, report)
    receipt = {
        "schema": "hftf_d7_public_real_egowalk_review_candidate_selection_receipt_v1",
        "record_kind": "EGOWALK_REVIEW_CANDIDATE_SELECTION_RECEIPT",
        "run_id": args.run_id,
        "generated_at_utc": report["generated_at_utc"],
        "status": report["status"],
        "candidate_artifact": report["candidate_artifact"],
        "output_artifact": report["output_artifact"],
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "selected_candidate_count": len(selected),
        "selected_source_session_count": len(by_session),
        "excluded_already_reviewed_count": len(excluded_reviewed),
        "excluded_reviewed_overlap_or_adjacency_count": len(excluded_adjacency),
        "event_truth_inferred": False,
        "admission_assigned": False,
        "split_assigned": False,
    }
    receipt_path = output_root / "receipts" / f"egowalk_review_candidate_selection_receipt_{args.run_id}.json"
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
