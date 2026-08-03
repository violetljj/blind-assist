#!/usr/bin/env python3
"""Materialize a deterministic 10% admitted-event re-review audit.

The audit is intentionally separate from the primary review and adjudication
surfaces.  It records whether a fresh isolated review reproduces the original
admission gates, but it never rewrites an admitted event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_jsonl, sha256_file, utc_now, write_json


ROLES = (
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
)


def _sample_ids(candidate_ids: set[str], *, seed: str, count: int) -> list[str]:
    return sorted(
        candidate_ids,
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest(),
    )[:count]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"required re-review output is missing: {path}")
    return load_jsonl(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    admitted_rows = load_jsonl(root / "adjudication" / "adjudicated_events.jsonl")
    admitted_ids = {str(row.get("candidate_id") or "") for row in admitted_rows}
    admitted_ids.discard("")
    if not admitted_ids:
        raise ContractError("no admitted events are available for the 10% re-review audit")
    expected_count = max(1, math.ceil(len(admitted_ids) * 0.10))
    expected_ids = set(_sample_ids(admitted_ids, seed=args.seed, count=expected_count))

    # Use the current adjudicated-event surface as the authoritative original
    # admission record.  Historical final-adjudicator bundles can contain
    # duplicate rechecks with a non-admitted terminal for the same candidate.
    original_by_candidate = {
        str(row.get("candidate_id")): row
        for row in admitted_rows
        if str(row.get("candidate_id") or "") in expected_ids
    }

    all_rows: dict[str, dict[str, dict[str, Any]]] = {}
    batch_receipts: list[dict[str, Any]] = []
    for batch_id in args.batch_id:
        batch_root = root / "reviews" / "input_bundles" / batch_id
        bundle = json.loads((batch_root / "bundle_manifest.json").read_text(encoding="utf-8"))
        batch_ids = {str(value) for value in bundle.get("candidate_ids", [])}
        for candidate_id in batch_ids:
            if candidate_id not in expected_ids:
                raise ContractError(f"re-review batch contains an unselected candidate: {batch_id}:{candidate_id}")
        role_counts: dict[str, int] = {}
        for role in ROLES:
            role_rows = _rows(batch_root / role / "completed_review.jsonl")
            role_counts[role] = len(role_rows)
            by_candidate = {str(row.get("candidate_id") or ""): row for row in role_rows}
            if len(by_candidate) != len(role_rows) or set(by_candidate) != batch_ids:
                raise ContractError(f"re-review role candidate set mismatch: {batch_id}:{role}")
            for candidate_id, row in by_candidate.items():
                all_rows.setdefault(candidate_id, {})[role] = row
        batch_receipts.append({"batch_id": batch_id, "candidate_count": len(batch_ids), "role_counts": role_counts})

    if set(all_rows) != expected_ids:
        raise ContractError(f"re-review sample mismatch: expected {sorted(expected_ids)}, got {sorted(all_rows)}")
    if set(original_by_candidate) != expected_ids:
        raise ContractError("original adjudication output is missing a selected admitted candidate")

    candidate_reports: list[dict[str, Any]] = []
    for candidate_id in sorted(expected_ids):
        role_map = all_rows[candidate_id]
        rgb = [role_map[role] for role in ("RGB_REVIEWER_A", "RGB_REVIEWER_B", "RGB_REVIEWER_C")]
        rgb_buckets = [str(row.get("event_bucket") or "") for row in rgb]
        rgb_decisions = [str(row.get("decision") or "") for row in rgb]
        rgb_exact = len(set(rgb_buckets)) == 1
        rgb_all_support = rgb_exact and all(decision == "SUPPORT" for decision in rgb_decisions)
        geometry = role_map["GEOMETRY_EVIDENCE_REVIEWER"]
        counter = role_map["COUNTEREXAMPLE_REVIEWER"]
        original = original_by_candidate[candidate_id]
        reasons: list[str] = []
        if not rgb_exact or not rgb_all_support:
            reasons.append("RGB_REREVIEW_GATE_UNRESOLVED")
        if str(geometry.get("decision") or "") == "NOT_EVALUABLE":
            reasons.append("GEOMETRY_REREVIEW_NOT_EVALUABLE")
        if counter.get("counterexample_search_completed") is not True:
            reasons.append("COUNTEREXAMPLE_REREVIEW_INCOMPLETE")
        if rgb_all_support and rgb_buckets[0] != str(original.get("event_bucket") or ""):
            reasons.append("REREVIEW_BUCKET_CONFLICT_WITH_ORIGINAL")
        candidate_reports.append({
            "candidate_id": candidate_id,
            "original_event_bucket": original.get("event_bucket"),
            "original_admission_status": original.get("admission_status"),
            "rereview_rgb_decisions": rgb_decisions,
            "rereview_rgb_buckets": rgb_buckets,
            "rereview_rgb_exact_bucket": rgb_exact,
            "rereview_rgb_all_support": rgb_all_support,
            "rereview_geometry_decision": geometry.get("decision"),
            "rereview_counterexample_completed": counter.get("counterexample_search_completed") is True,
            "reproduces_original_admission": not reasons,
            "reasons": reasons,
        })

    unresolved_count = sum(not row["reproduces_original_admission"] for row in candidate_reports)
    rgb_disagreement_count = sum(not row["rereview_rgb_exact_bucket"] for row in candidate_reports)
    status = "COMPLETE_NO_CONFLICTS" if unresolved_count == 0 else "COMPLETE_WITH_CONFLICTS"
    report = {
        "schema": "hftf_d7_public_real_rereview_audit_v1",
        "generated_at_utc": utc_now(),
        "seed": args.seed,
        "admitted_event_count": len(admitted_ids),
        "required_sample_count": expected_count,
        "selected_sample_count": len(expected_ids),
        "selected_candidate_ids": sorted(expected_ids),
        "status": status,
        "unresolved_or_conflicted_count": unresolved_count,
        "rgb_disagreement_count": rgb_disagreement_count,
        "batch_receipts": batch_receipts,
        "candidate_reports": candidate_reports,
        "primary_adjudication_unchanged": True,
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
    }
    json_path = root / "reports" / "ten_percent_admitted_event_rereview_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# HFTF D7 10% admitted-event re-review audit",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{status}`.",
        "",
        f"- Original admitted parent events: `{len(admitted_ids)}`.",
        f"- Required deterministic sample (ceil 10%): `{expected_count}`.",
        f"- Fresh isolated re-review coverage: `{len(expected_ids)}`.",
        f"- RGB exact-bucket disagreement count: `{rgb_disagreement_count}`.",
        f"- Events reproducing every original admission gate: `{len(expected_ids) - unresolved_count}`.",
        "- This audit does not overwrite primary reviews or adjudicated events.",
        "- NOT_EVALUABLE and disagreement remain evidence of re-review instability or missing gates, not negative labels.",
        "",
        "| Candidate | Original bucket | RGB buckets | Geometry | Reproduces original admission | Reasons |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in candidate_reports:
        markdown.append(
            f"| {row['candidate_id']} | {row['original_event_bucket']} | "
            f"{','.join(row['rereview_rgb_buckets'])} | {row['rereview_geometry_decision']} | "
            f"{row['reproduces_original_admission']} | {','.join(row['reasons']) or 'none'} |"
        )
    md_path = root / "reports" / "ten_percent_admitted_event_rereview_report.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    receipt = {
        "schema": "hftf_d7_public_real_rereview_audit_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": report["generated_at_utc"],
        "status": status,
        "json_report": {"path": str(json_path), "sha256": sha256_file(json_path)},
        "markdown_report": {"path": str(md_path), "sha256": sha256_file(md_path)},
        "selected_sample_count": len(expected_ids),
        "required_sample_count": expected_count,
        "primary_adjudication_unchanged": True,
        "authority": report["authority"],
    }
    write_json(root / "receipts" / "ten_percent_admitted_event_rereview_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", default="D7-10PCT-REVIEW")
    parser.add_argument("--batch-id", action="append", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
