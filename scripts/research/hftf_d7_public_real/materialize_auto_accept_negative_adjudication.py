#!/usr/bin/env python3
"""Materialize the frozen auto-accept gate for independently supported negatives.

The gate is intentionally narrow: RGB A/B/C and the counterexample role must
independently SUPPORT the same negative bucket with the same valid continuous
negative interval; the source-native geometry role must be NOT_EVALUABLE and
non-opposing.  It cannot admit positive buckets or infer a negative from
missing geometry.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


RGB_ROLES = ("RGB_REVIEWER_A", "RGB_REVIEWER_B", "RGB_REVIEWER_C")
GEOMETRY_ROLE = "GEOMETRY_EVIDENCE_REVIEWER"
COUNTEREXAMPLE_ROLE = "COUNTEREXAMPLE_REVIEWER"
REVIEW_ROLES = RGB_ROLES + (GEOMETRY_ROLE, COUNTEREXAMPLE_ROLE)
NEGATIVE_BUCKETS = {
    "PARALLEL_STRUCTURE_NEGATIVE",
    "SIDE_OBJECT_NONBLOCKING_NEGATIVE",
    "NORMAL_WALKABLE_NEGATIVE",
    "EGOMOTION_VISUAL_HARD_NEGATIVE",
    "HEAD_NONACTIONABLE_NEGATIVE",
}


def _valid_interval(phases: object) -> bool:
    if not isinstance(phases, dict):
        return False
    interval = phases.get("continuous_negative_interval")
    if not isinstance(interval, dict):
        return False
    try:
        return int(interval["start_timestamp_ns"]) < int(interval["end_timestamp_ns"])
    except (KeyError, TypeError, ValueError):
        return False


def _interval_key(phases: object) -> tuple[int, int] | None:
    if not _valid_interval(phases):
        return None
    interval = phases["continuous_negative_interval"]  # type: ignore[index]
    return int(interval["start_timestamp_ns"]), int(interval["end_timestamp_ns"])


def _load_primary_reviews(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    paths = {
        "RGB_REVIEWER_A": root / "reviews" / "review_a.jsonl",
        "RGB_REVIEWER_B": root / "reviews" / "review_b.jsonl",
        "RGB_REVIEWER_C": root / "reviews" / "review_c.jsonl",
        GEOMETRY_ROLE: root / "reviews" / "geometry_review.jsonl",
        COUNTEREXAMPLE_ROLE: root / "reviews" / "counterexample_review.jsonl",
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for role, path in paths.items():
        result[role] = {str(row.get("candidate_id")): row for row in load_jsonl(path)}
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    source_batches = [item.strip() for item in args.source_batches.split(",") if item.strip()]
    if not source_batches:
        raise ContractError("at least one source review batch is required")
    output_root = root / "reviews" / "adjudication_bundles" / args.batch_id
    if output_root.exists():
        raise ContractError(f"adjudication output already exists: {output_root}")

    event_rows = load_jsonl(root / "manifests" / "event_manifest.jsonl")
    event_by_candidate = {str(row.get("candidate_id")): row for row in event_rows}
    existing_event_ids = {str(row.get("event_id")) for row in load_jsonl(root / "adjudication" / "adjudicated_events.jsonl")}
    primary = _load_primary_reviews(root)
    candidate_source: dict[str, str] = {}
    candidate_ids: list[str] = []
    for source_batch in source_batches:
        manifest = load_json(root / "reviews" / "input_bundles" / source_batch / "bundle_manifest.json")
        if not isinstance(manifest, dict) or manifest.get("status") != "READY_FOR_ISOLATED_REVIEW":
            raise ContractError(f"source review bundle is not ready: {source_batch}")
        for raw_id in manifest.get("candidate_ids", []):
            candidate_id = str(raw_id)
            if candidate_id in candidate_source:
                raise ContractError(f"candidate appears in multiple source batches: {candidate_id}")
            candidate_source[candidate_id] = source_batch
            candidate_ids.append(candidate_id)

    admitted_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    skip_reasons: defaultdict[str, int] = defaultdict(int)
    for candidate_id in candidate_ids:
        candidate = event_by_candidate.get(candidate_id)
        if candidate is None:
            skip_reasons["CANDIDATE_NOT_IN_EVENT_MANIFEST"] += 1
            continue
        if str(candidate.get("event_id")) in existing_event_ids:
            skip_reasons["ALREADY_ADJUDICATED"] += 1
            continue
        reviews = {role: primary[role].get(candidate_id) for role in REVIEW_ROLES}
        if any(row is None for row in reviews.values()):
            skip_reasons["ROLE_REVIEW_MISSING"] += 1
            continue
        if any(row.get("review_completed") is not True for row in reviews.values()):
            skip_reasons["ROLE_REVIEW_INCOMPLETE"] += 1
            continue
        if any(row.get("independent_observation_recorded", True) is False for row in reviews.values()):
            skip_reasons["NON_INDEPENDENT_TERMINAL_PRESENT"] += 1
            continue
        if any(row.get("model_output_visible") is not False for row in reviews.values()):
            skip_reasons["MODEL_OUTPUT_FIREWALL"] += 1
            continue

        rgb_buckets = [reviews[role].get("event_bucket") for role in RGB_ROLES]
        counter = reviews[COUNTEREXAMPLE_ROLE]
        geometry = reviews[GEOMETRY_ROLE]
        if any(reviews[role].get("decision") != "SUPPORT" for role in RGB_ROLES):
            skip_reasons["RGB_SUPPORT_GATE"] += 1
            continue
        if len(set(rgb_buckets)) != 1 or rgb_buckets[0] not in NEGATIVE_BUCKETS:
            skip_reasons["RGB_NEGATIVE_BUCKET_AGREEMENT"] += 1
            continue
        bucket = str(rgb_buckets[0])
        if counter.get("decision") != "SUPPORT" or counter.get("event_bucket") != bucket:
            skip_reasons["COUNTEREXAMPLE_GATE"] += 1
            continue
        if counter.get("counterexample_search_completed") is not True:
            skip_reasons["COUNTEREXAMPLE_SEARCH_INCOMPLETE"] += 1
            continue
        if geometry.get("decision") != "NOT_EVALUABLE" or geometry.get("event_bucket") != "NOT_EVALUABLE":
            skip_reasons["GEOMETRY_NON_OPPOSING_GATE"] += 1
            continue
        intervals = [_interval_key(reviews[role].get("phase_intervals")) for role in (*RGB_ROLES, COUNTEREXAMPLE_ROLE)]
        if any(interval is None for interval in intervals) or len(set(intervals)) != 1:
            skip_reasons["NEGATIVE_INTERVAL_GATE"] += 1
            continue
        start_ns, end_ns = intervals[0]  # type: ignore[misc]
        event_id = str(candidate.get("event_id"))
        phase_intervals = {
            "continuous_negative_interval": {
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": end_ns,
            }
        }
        admitted_rows.append(
            {
                "schema": "hftf_d7_public_real_completed_adjudication_v1",
                "record_kind": "COMPLETED_ADJUDICATION",
                "batch_id": args.batch_id,
                "candidate_id": candidate_id,
                "event_id": event_id,
                "dataset_id": candidate.get("dataset_id"),
                "adjudication_decision": "ADMIT",
                "admission_status": "ADMITTED",
                "event_bucket": bucket,
                "phase_intervals": phase_intervals,
                "model_output_visible": False,
                "training_authority": False,
                "confirmation_authority": False,
                "production_authority": False,
                "safety_authority": False,
                "default_app_authority": False,
                "model_authority": False,
                "threshold_authority": False,
                "reason_code": "AUTO_ACCEPT_NEGATIVE_GATE_PASS",
                "notes": (
                    "RGB Reviewer A/B/C independently SUPPORT the same negative bucket with the same valid "
                    "continuous negative interval; geometry is NOT_EVALUABLE and non-opposing; counterexample "
                    "review independently supports the same bucket."
                ),
            }
        )
        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "event_id": event_id,
                "source_review_batch": candidate_source[candidate_id],
                "event_bucket": bucket,
                "continuous_negative_interval": phase_intervals["continuous_negative_interval"],
                "gate": "AUTO_ACCEPT_NEGATIVE_GATE_PASS",
                "independent_review_roles": list(REVIEW_ROLES),
                "geometry_terminal": "NOT_EVALUABLE_NON_OPPOSING",
            }
        )

    if not admitted_rows:
        raise ContractError(f"no candidates passed the auto-accept negative gate: {dict(skip_reasons)}")
    output_path = output_root / "FINAL_ADJUDICATOR" / "final_adjudication.jsonl"
    write_jsonl(output_path, admitted_rows)
    write_jsonl(output_root / "FINAL_ADJUDICATOR_INPUT.jsonl", audit_rows)
    manifest = {
        "schema": "hftf_d7_public_real_adjudication_bundle_v1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "status": "READY_FOR_FINAL_ADJUDICATION",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "candidate_ids": [row["candidate_id"] for row in admitted_rows],
        "candidate_count": len(admitted_rows),
        "source_review_batches": source_batches,
        "source_review_gate": "AUTO_ACCEPT_NEGATIVE_GATE_PASS",
        "model_output_visible_in_any_input": False,
        "confirmation_authorized": False,
        "production_authorized": False,
        "training_authorized": False,
        "event_truth_inferred": False,
        "admission_assigned": True,
    }
    manifest_path = output_root / "bundle_manifest.json"
    write_json(manifest_path, manifest)
    receipt = {
        "schema": "hftf_d7_public_real_auto_accept_negative_gate_receipt_v1",
        "record_kind": "AUTO_ACCEPT_NEGATIVE_GATE_RECEIPT",
        "status": "READY_FOR_ADJUDICATION_INGEST",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "source_review_batches": source_batches,
        "candidate_count_examined": len(candidate_ids),
        "admitted_candidate_count": len(admitted_rows),
        "candidate_ids": [row["candidate_id"] for row in admitted_rows],
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "bundle_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "admission_assigned": True,
        "event_truth_inferred": False,
        "split_assigned": False,
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
    }
    receipt_path = root / "receipts" / f"auto_accept_negative_gate_receipt_{args.batch_id}.json"
    write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-batches", required=True, help="comma-separated source review bundle ids")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
    except ContractError as exc:
        raise SystemExit(str(exc))
