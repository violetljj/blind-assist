#!/usr/bin/env python3
"""Refresh D7 terminal reports without rewriting review or split surfaces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json


ROLES = {
    "RGB_REVIEWER_A": "review_a.jsonl",
    "RGB_REVIEWER_B": "review_b.jsonl",
    "RGB_REVIEWER_C": "review_c.jsonl",
    "GEOMETRY_EVIDENCE_REVIEWER": "geometry_review.jsonl",
    "COUNTEREXAMPLE_REVIEWER": "counterexample_review.jsonl",
}
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


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _completed_reviews(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for role, filename in ROLES.items():
        path = root / "reviews" / filename
        if not path.is_file():
            raise ContractError(f"review surface missing: {path}")
        result[role] = [row for row in load_jsonl(path) if row.get("review_completed") is True]
    return result


def _final_adjudication_count(root: Path) -> int:
    total = 0
    for path in sorted((root / "reviews" / "adjudication_bundles").glob("*/FINAL_ADJUDICATOR/final_adjudication.jsonl")):
        total += len(load_jsonl(path))
    return total


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    candidates = load_jsonl(root / "candidates" / "candidate_index.jsonl")
    events = load_jsonl(root / "manifests" / "event_manifest.jsonl")
    rejected = load_jsonl(root / "adjudication" / "rejected_events.jsonl")
    adjudicated = load_jsonl(root / "adjudication" / "adjudicated_events.jsonl")
    reviews = _completed_reviews(root)
    candidate_counts = Counter(str(row.get("dataset_id", "UNKNOWN")) for row in candidates)
    event_reasons = Counter(str(row.get("not_evaluable_reason") or "UNSPECIFIED") for row in events)
    admitted_buckets = Counter(str(row.get("event_bucket", "UNKNOWN")) for row in adjudicated)
    completed_counts = {role: len(rows) for role, rows in reviews.items()}
    rgb_ids = [
        {str(row.get("candidate_id")): str(row.get("event_bucket")) for row in reviews[role]}
        for role in ("RGB_REVIEWER_A", "RGB_REVIEWER_B", "RGB_REVIEWER_C")
    ]
    rgb_common = set.intersection(*(set(value) for value in rgb_ids)) if rgb_ids else set()
    rgb_exact = sum(1 for candidate_id in rgb_common if len({value[candidate_id] for value in rgb_ids}) == 1)
    rgb_positive_exact = sum(
        1
        for candidate_id in rgb_common
        if len({value[candidate_id] for value in rgb_ids}) == 1
        and str(rgb_ids[0][candidate_id]).endswith("_POSITIVE")
    )
    final_adjudication_count = _final_adjudication_count(root)
    validation = load_json(root / "reports" / "d7_validation_report.json")
    role_isolation = load_json(root / "manifests" / "role_isolation_receipt.json")

    quality_lines = [
        "# HFTF D7 dataset quality report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Terminal",
        "",
        "`NOT_COMPLETE`: candidate coverage is present, but independent role/adjudication gates and 10,000 admitted parent events are not complete.",
        "",
        "| Check | Result | Evidence |",
        "| --- | --- | --- |",
        f"| Candidate windows | `{len(candidates)}` | `candidates/candidate_index.jsonl` |",
        f"| Admitted parent events | `{len(adjudicated)}` | `adjudication/adjudicated_events.jsonl` |",
        "| Event truth | `NOT_AUTHORIZED` | no completed dataset-wide review/adjudication gate |",
        "| Training eligibility | `false` | all splits remain blocked |",
        "| Confirmation eligibility | `false` | no admitted event split |",
        "| Negative evidence from missing data | `false` | missingness remains NOT_EVALUABLE |",
        "",
        "## Candidate dataset counts",
        "",
    ]
    quality_lines.extend(f"- `{dataset}`: `{count}`" for dataset, count in sorted(candidate_counts.items()))
    quality_lines.extend([
        "",
        "## Review and adjudication coverage",
        "",
        f"- Completed independent review rows by role: `{completed_counts}`.",
        f"- Final adjudication rows materialized across batches: `{final_adjudication_count}`; admitted rows: `{len(adjudicated)}`.",
        f"- Current validator status: `{validation.get('status', 'UNKNOWN')}`; role isolation: `{role_isolation.get('status', 'UNKNOWN')}`.",
        "- A completed role row is evidence for its review batch only; it does not promote the remaining candidate shell rows.",
        "",
        "## Quality gates",
        "",
        "- Source-session isolation is the split unit; ancestry conflicts remain an audit hold until resolved.",
        "- Source-native geometry, RGB, and counterexample evidence remain separate from model output and event truth.",
        "- Missing capture timestamps, incomplete source-native event phases, and unresolved source terms remain NOT_EVALUABLE, never negative evidence.",
        "- YOLO/HFTF/threshold/confirmation-length/backbone changes remain unauthorized before dataset completion.",
        "",
        "## Terminal reasons",
        "",
    ])
    quality_lines.extend(f"- `{reason}`: `{count}`" for reason, count in sorted(event_reasons.items()))
    _write(root / "reports" / "dataset_quality_report.md", quality_lines)

    class_lines = [
        "# HFTF D7 class balance report",
        "",
        "No class balance claim is authorized because no parent event passed final adjudication.",
        "",
        "| Event bucket | Admitted count | Target | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    class_lines.extend(
        f"| {bucket} | {admitted_buckets.get(bucket, 0)} | {target} | `NOT_EVALUABLE` |"
        for bucket, target in TARGETS.items()
    )
    class_lines.append(f"| NOT_EVALUABLE | {len(rejected)} | — | `terminal; not a negative class` |")
    _write(root / "reports" / "class_balance_report.md", class_lines)

    duplicate_lines = [
        "# HFTF D7 duplicate audit report",
        "",
        "Status: `PARTIAL_INTAKE_ONLY`.",
        "",
        f"- Candidate rows audited for registry identity: `{len(candidates)}`.",
        f"- Candidate ID duplicate count: `{len(candidates) - len({str(row.get('candidate_id')) for row in candidates})}`.",
        f"- Event manifest rows: `{len(events)}`; admitted event rows: `{len(adjudicated)}`.",
        "- Temporal overlap, near-duplicate image graph, stereo/view collapse, and parent-event adjacency remain review-gated.",
        "- No candidate was admitted by this refresh.",
    ]
    _write(root / "reports" / "duplicate_audit_report.md", duplicate_lines)

    agreement_lines = [
        "# HFTF D7 label agreement report",
        "",
        "Status: `PILOT_AND_BATCH_REVIEW_ONLY`.",
        "",
        f"- Completed independent reviews by role: `{completed_counts}`.",
        f"- RGB A/B/C common completed candidates: `{len(rgb_common)}`; exact bucket agreement: `{rgb_exact}`; exact positive agreement: `{rgb_positive_exact}`.",
        f"- Final adjudications materialized: `{final_adjudication_count}`; admitted events: `{len(adjudicated)}`.",
        "- Agreement statistics, 10% re-review, conflict rates, and dataset-wide completion remain incomplete; missing agreement is not treated as agreement.",
        "- Counterexample SUPPORT on a negative bucket was contract-normalized only when its three intervals were contiguous; raw outputs are preserved in each batch's raw_unvalidated_outputs directory.",
    ]
    _write(root / "reports" / "label_agreement_report.md", agreement_lines)

    receipt = {
        "schema": "hftf_d7_public_real_terminal_report_refresh_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "candidate_windows": len(candidates),
        "admitted_parent_events": len(adjudicated),
        "completed_review_counts": completed_counts,
        "final_adjudication_count": final_adjudication_count,
        "rgb_common_completed": len(rgb_common),
        "rgb_exact_bucket_agreement": rgb_exact,
        "rgb_exact_positive_agreement": rgb_positive_exact,
        "report_hashes": {
            name: sha256_file(root / "reports" / name)
            for name in (
                "dataset_quality_report.md",
                "class_balance_report.md",
                "duplicate_audit_report.md",
                "label_agreement_report.md",
            )
        },
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
    }
    receipt_path = root / "manifests" / "terminal_report_refresh_receipt.json"
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
