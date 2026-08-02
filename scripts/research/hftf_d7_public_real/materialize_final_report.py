#!/usr/bin/env python3
"""Materialize the twelve-item D7 status report from current receipts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json


TARGETS = {
    "BLOCKING_BODY_POSITIVE": 1500,
    "BOUNDARY_LEVEL_CHANGE_POSITIVE": 1000,
    "DYNAMIC_INTRUSION_POSITIVE": 1000,
    "HEAD_HAZARD_POSITIVE": 500,
    "PARALLEL_STRUCTURE_NEGATIVE": 1500,
    "SIDE_OBJECT_NONBLOCKING_NEGATIVE": 1000,
    "NORMAL_WALKABLE_NEGATIVE": 2000,
    "EGOMOTION_VISUAL_HARD_NEGATIVE": 1000,
    "HEAD_NONACTIONABLE_NEGATIVE": 500,
}


def _count_jsonl(path: Path) -> int:
    return len(load_jsonl(path))


def _load_intake_receipts(root: Path) -> list[dict[str, Any]]:
    """Load source-specific JSON receipts without treating them as event truth."""

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "receipts").glob("*.json")):
        try:
            value = load_json(path)
        except Exception:
            continue
        if isinstance(value, dict) and value.get("dataset_id"):
            rows.append(value)
    return rows


def _final_adjudication_candidate_ids(root: Path) -> set[str]:
    """Count final adjudication coverage across all immutable review batches."""

    candidate_ids: set[str] = set()
    for path in sorted((root / "reviews" / "adjudication_bundles").glob("*/FINAL_ADJUDICATOR/final_adjudication.jsonl")):
        for row in load_jsonl(path):
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id:
                candidate_ids.add(candidate_id)
    return candidate_ids


def _table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    width = len(rows[0])
    widths = [max(len(row[index]) for row in rows) for index in range(width)]
    rendered = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])) + " |"]
    rendered.append("| " + " | ".join("-" * max(3, width) for width in widths) + " |")
    rendered.extend(
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows[1:]
    )
    return rendered


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    required = {
        "registry": root / "manifests" / "dataset_registry.json",
        "sessions": root / "manifests" / "session_registry.jsonl",
        "candidates": root / "candidates" / "candidate_index.jsonl",
        "events": root / "adjudication" / "adjudicated_events.jsonl",
        "rejected": root / "adjudication" / "rejected_events.jsonl",
        "receipts": root / "receipts" / "source_receipts.jsonl",
        "validation": root / "reports" / "d7_validation_report.json",
        "role_isolation": root / "manifests" / "role_isolation_receipt.json",
        "learning_curve": root / "manifests" / "learning_curve_plan.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ContractError(f"missing final-report inputs: {', '.join(missing)}")

    registry = load_json(required["registry"])
    validation = load_json(required["validation"])
    role_isolation = load_json(required["role_isolation"])
    learning_curve = load_json(required["learning_curve"])
    if not all(isinstance(value, dict) for value in (registry, validation, role_isolation, learning_curve)):
        raise ContractError("final-report JSON inputs must be objects")
    candidates = load_jsonl(required["candidates"])
    sessions = load_jsonl(required["sessions"])
    events = load_jsonl(required["events"])
    rejected = load_jsonl(required["rejected"])
    receipts = load_jsonl(required["receipts"])
    intake_receipts = _load_intake_receipts(root)
    final_adjudication_count = len(_final_adjudication_candidate_ids(root))

    review_paths = {
        "RGB_REVIEWER_A": root / "reviews" / "review_a.jsonl",
        "RGB_REVIEWER_B": root / "reviews" / "review_b.jsonl",
        "RGB_REVIEWER_C": root / "reviews" / "review_c.jsonl",
        "GEOMETRY_EVIDENCE_REVIEWER": root / "reviews" / "geometry_review.jsonl",
        "COUNTEREXAMPLE_REVIEWER": root / "reviews" / "counterexample_review.jsonl",
    }
    review_rows = {role: load_jsonl(path) for role, path in review_paths.items() if path.is_file()}
    completed_review_counts = {
        role: sum(1 for row in rows if row.get("record_kind") == "COMPLETED_REVIEW" and row.get("review_completed") is True)
        for role, rows in review_rows.items()
    }
    completed_by_role = {
        role: {
            str(row.get("candidate_id")): str(row.get("event_bucket"))
            for row in rows
            if row.get("record_kind") == "COMPLETED_REVIEW" and row.get("review_completed") is True
        }
        for role, rows in review_rows.items()
    }
    rgb_roles = ["RGB_REVIEWER_A", "RGB_REVIEWER_B", "RGB_REVIEWER_C"]
    rgb_common_ids = set.intersection(*(set(completed_by_role.get(role, {})) for role in rgb_roles)) if all(role in completed_by_role for role in rgb_roles) else set()
    rgb_exact_agreement = sum(
        1 for candidate_id in rgb_common_ids
        if len({completed_by_role[role][candidate_id] for role in rgb_roles}) == 1
    )

    candidate_by_dataset = Counter(str(row.get("dataset_id", "UNKNOWN")) for row in candidates)
    admitted_by_bucket = Counter(str(row.get("event_bucket", "UNKNOWN")) for row in events)
    rejected_reasons = Counter(str(row.get("reason", "UNKNOWN")) for row in rejected)
    source_stats = registry.get("source_stats", {})
    intake_candidate_latest: dict[str, tuple[str, int]] = {}
    intake_rgb_counts = Counter()
    intake_rgb_bytes = Counter()
    for source_row in receipts:
        dataset_id = str(source_row.get("dataset_id", "UNKNOWN"))
        intake_rgb_counts[dataset_id] = max(
            intake_rgb_counts[dataset_id],
            int(source_row.get("rgb_media_count", 0) or 0),
        )
        intake_rgb_bytes[dataset_id] = max(
            intake_rgb_bytes[dataset_id],
            int(source_row.get("rgb_media_bytes", 0) or 0),
        )
    for receipt in intake_receipts:
        dataset_id = str(receipt.get("dataset_id", "UNKNOWN"))
        counts = receipt.get("counts", {})
        if isinstance(counts, dict):
            candidate_count = int(
                counts.get("candidate_windows", 0)
                or counts.get("candidate_count", 0)
                or receipt.get("candidate_count", 0)
                or 0
            )
            generated_at = str(receipt.get("generated_at_utc", ""))
            previous = intake_candidate_latest.get(dataset_id)
            if candidate_count and (previous is None or generated_at >= previous[0]):
                intake_candidate_latest[dataset_id] = (generated_at, candidate_count)
        elif receipt.get("candidate_count"):
            candidate_count = int(receipt.get("candidate_count", 0) or 0)
            generated_at = str(receipt.get("generated_at_utc", ""))
            previous = intake_candidate_latest.get(dataset_id)
            if previous is None or generated_at >= previous[0]:
                intake_candidate_latest[dataset_id] = (generated_at, candidate_count)
        intake_rgb_counts[dataset_id] = max(
            intake_rgb_counts[dataset_id],
            int(receipt.get("rgb_media_count", 0) or 0),
        )
        intake_rgb_bytes[dataset_id] = max(
            intake_rgb_bytes[dataset_id],
            int(receipt.get("rgb_media_bytes", 0) or 0),
        )
        members = receipt.get("members", [])
        if isinstance(members, list):
            video_members = [
                item for item in members
                if isinstance(item, dict) and item.get("member_kind") == "VIDEO"
            ]
            intake_rgb_counts[dataset_id] = max(intake_rgb_counts[dataset_id], len(video_members))
            intake_rgb_bytes[dataset_id] = max(
                intake_rgb_bytes[dataset_id],
                sum(int(item.get("file_size", 0) or 0) for item in video_members),
            )
    actual_sources = [
        row for row in receipts
        if row.get("retrieved_at_utc") or row.get("local_evidence_paths")
    ]
    inaccessible_sources = [
        row for row in receipts
        if str(row.get("access_status", "")).startswith("ACCESS_BLOCKED")
        or "REQUIRED" in str(row.get("access_status", ""))
        or "REVIEW_REQUIRED" in str(row.get("access_status", ""))
    ]
    validation_checks = validation.get("completion_checks", {})
    source_table = [["Dataset", "Access/status", "Receipt kind", "Top-level candidates", "Source-stat candidates", "RGB media"]]
    for row in sorted(receipts, key=lambda value: str(value.get("dataset_id"))):
        dataset_id = str(row.get("dataset_id"))
        source_table.append([
            dataset_id,
            str(row.get("access_status", "UNKNOWN")),
            str(row.get("source_hash_kind", "UNSPECIFIED")),
            str(candidate_by_dataset.get(dataset_id, 0)),
            str(int(source_stats.get(dataset_id, {}).get("candidate_windows", 0) or intake_candidate_latest.get(dataset_id, ("", 0))[1])),
            str(max(int(row.get("rgb_media_count", 0) or 0), intake_rgb_counts.get(dataset_id, 0))),
        ])

    bucket_table = [["Event bucket", "Admitted", "Target", "Status"]]
    for bucket, target in TARGETS.items():
        count = admitted_by_bucket.get(bucket, 0)
        bucket_table.append([bucket, str(count), str(target), "PASS" if count >= target else "NOT_EVALUABLE"])
    bucket_table.append(["NOT_EVALUABLE", str(len(rejected)), "—", "terminal; not a negative class"])

    report_status = str(validation.get("status", "UNKNOWN"))
    lines = [
        "# HFTF D7 public-real final status report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        f"Status: `{report_status}` according to the machine-readable validator snapshot below.",
        "This report is an evidence-bound status artifact, not event truth or product/safety authorization.",
        "",
        "## 1. Actual sources accessed",
        "",
        f"- Receipt rows with local evidence or retrieval timestamps: `{len(actual_sources)}` / `{len(receipts)}`.",
        f"- Sources explicitly blocked or requiring credentials/terms: `{len(inaccessible_sources)}`.",
        "- Public extracted EgoWalk trajectories/RGB, SANPO public media canary, open THOR tracks/LiDAR, and bounded selected-member THOR-MAGNI media were handled through source receipts; raw/gated sources were not bypassed.",
        "",
        "## 2. Download and parse statistics",
        "",
        f"- Registry session rows: `{len(sessions)}`; candidate frame rows: `{registry.get('candidate_discovery', {}).get('total_frame_count', 0)}`.",
        f"- Candidate dataset counts: `{dict(sorted(candidate_by_dataset.items()))}`.",
        f"- EgoWalk RGB media receipt: `{next((row.get('rgb_media_count', 0) for row in receipts if row.get('dataset_id') == 'EgoWalk'), 0)}` files, `{next((row.get('rgb_media_bytes', 0) for row in receipts if row.get('dataset_id') == 'EgoWalk'), 0)}` bytes.",
        f"- Source-intake media snapshot: `{dict(sorted(intake_rgb_counts.items()))}` RGB files; `{dict(sorted(intake_rgb_bytes.items()))}` bytes where receipts expose member sizes.",
        f"- Source stats snapshot: `{dict(sorted(source_stats.items()))}`.",
        "",
        "## 3. Candidate and admitted counts",
        "",
        f"- Candidate windows: `{len(candidates)}`; candidate target `[50000, 100000]`: `{len(candidates) >= 50000}`.",
        f"- Admitted parent events: `{len(events)}` / target `10000`; target reached: `{len(events) >= 10000}`.",
        "",
        "## 4. Event-bucket counts",
        "",
        "\n".join(_table(bucket_table)),
        "",
        "## 5. Source contribution",
        "",
        "\n".join(_table(source_table)),
        "",
        "## 6. Reviewer agreement",
        "",
        f"- Validation review rows: `{validation.get('counts', {}).get('review_rows', {})}`.",
        f"- Completed independent review rows: `{completed_review_counts}`; final adjudication outputs: `{final_adjudication_count}`; admitted events: `{len(events)}`.",
        f"- RGB A/B/C common completed candidates: `{len(rgb_common_ids)}`; exact bucket agreement: `{rgb_exact_agreement}`; this is a pilot agreement count, not event truth.",
        "- 10% re-review and adjudicator conflict-rate gates remain incomplete; missing agreement is not treated as agreement.",
        "",
        "## 7. NOT_EVALUABLE reasons",
        "",
    ]
    lines.append("The counts below are pending-package terminal rows; they are not final adjudicated labels.")
    lines.extend(f"- `{reason}`: `{count}`" for reason, count in sorted(rejected_reasons.items()))
    lines.extend([
        "",
        "## 8. Deduplication result",
        "",
        f"- Candidate duplicate IDs: `{validation.get('counts', {}).get('candidate_duplicate_ids', 'UNKNOWN')}`.",
        "- Temporal overlap, near-duplicate image graph, stereo/view collapse, and parent-event adjacency merge remain review-gated; no candidate was promoted as an event by this intake.",
        "",
        "## 9. Data roles and session isolation",
        "",
        f"- Role-isolation status: `{role_isolation.get('status', 'UNKNOWN')}`; ancestry role conflicts: `{role_isolation.get('counts', {}).get('ancestry_role_conflicts', 'UNKNOWN')}`.",
        f"- Validator session-disjoint check: `{validation_checks.get('splits_session_disjoint', False)}` (vacuous with zero admitted events).",
        "- Training, Confirmation, production, and event-truth authority remain false.",
        "",
        "## 10. Data gaps",
        "",
        f"- Independent review pilot completed rows: `{sum(completed_review_counts.values())}`; final adjudication outputs: `{final_adjudication_count}`; admitted events: `{len(events)}`.",
        "- EgoWalk raw recordings, Project Aria terms/download, JRDB full access, THOR synchronized video, archive-wide THOR-MAGNI media/review, Ego4D, and Ego-Exo4D remain blocked, partial, or not lawfully closed in the receipts.",
        "- Existing consumed/Development evidence and model outputs were not upgraded into D7 event truth.",
        "",
        "## 11. Fixed-model learning-curve plan",
        "",
        f"- Status: `{learning_curve.get('status', 'UNKNOWN')}`; sizes: `{learning_curve.get('training_event_counts', [])}`; seeds per size: `{learning_curve.get('seeds_per_count', 0)}`.",
        "- YOLO/HFTF/threshold/confirmation-length/backbone changes remain unauthorized before dataset completion.",
        "",
        "## 12. 10,000-event target",
        "",
        f"- Final answer for this run: `NO` — `{len(events)}` admitted parent events, `{len(candidates)}` candidates, validator status `{validation.get('status', 'UNKNOWN')}`.",
        "- Required action: obtain lawful independent review/adjudication and resolve role/near-duplicate gates; do not synthesize labels or treat missingness as negative evidence.",
        "",
        "## Machine-readable validation",
        "",
        f"- `reports/d7_validation_report.json`: status `{validation.get('status', 'UNKNOWN')}`.",
        f"- Completion checks: `{dict(sorted(validation_checks.items()))}`.",
    ])
    report_path = root / "reports" / "d7_final_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema": "hftf_d7_public_real_final_report_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "status": validation.get("status", "UNKNOWN"),
        "candidate_windows": len(candidates),
        "admitted_parent_events": len(events),
        "target_reached": len(events) >= 10000,
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
    }
    write_json(root / "manifests" / "final_report_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    import json

    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
