#!/usr/bin/env python3
"""Record an explicit NOT_EVALUABLE completion terminal for a review role.

This fallback is deliberately not an observation or a label.  It is used only
when a bounded reviewer execution produced no independent observable decision;
the receipt records that fact so downstream adjudication can fail closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


REVIEW_ROLES = {
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.role not in REVIEW_ROLES:
        raise ContractError(f"unsupported review role: {args.role}")
    output_root = Path(args.output_root).resolve()
    batch_root = output_root / "reviews" / "input_bundles" / args.batch_id
    bundle_manifest = load_json(batch_root / "bundle_manifest.json")
    if not isinstance(bundle_manifest, dict) or bundle_manifest.get("status") != "READY_FOR_ISOLATED_REVIEW":
        raise ContractError(f"review bundle is not ready: {batch_root}")
    if bundle_manifest.get("model_output_visible_in_any_input") is not False:
        raise ContractError("review bundle is not model-blind")
    manifest_path = batch_root / "manifests" / f"{args.role}.jsonl"
    inputs = load_jsonl(manifest_path)
    candidate_ids = [str(value) for value in bundle_manifest.get("candidate_ids", [])]
    by_id = {str(row.get("candidate_id")): row for row in inputs}
    if len(inputs) != len(candidate_ids) or set(by_id) != set(candidate_ids):
        raise ContractError(f"role manifest candidate mismatch: {manifest_path}")
    output_path = batch_root / args.role / "completed_review.jsonl"
    if output_path.exists() and not args.overwrite:
        raise ContractError(f"completed review already exists: {output_path}")

    note = (
        "FAIL_CLOSED_REVIEW_EXECUTION_TERMINAL: no independent observable decision recorded; "
        "NOT_EVALUABLE is not negative evidence. "
        + args.reason.strip()
    ).strip()
    rows = []
    for index, candidate_id in enumerate(candidate_ids):
        source = by_id[candidate_id]
        rows.append(
            {
                "schema": "hftf_d7_public_real_completed_review_v1",
                "record_kind": "COMPLETED_REVIEW",
                "batch_id": args.batch_id,
                "candidate_id": candidate_id,
                "dataset_id": source.get("dataset_id"),
                "decision": "NOT_EVALUABLE",
                "event_bucket": "NOT_EVALUABLE",
                "terminal": "NOT_EVALUABLE",
                "phase_intervals": None,
                "model_output_visible": False,
                "review_completed": True,
                "review_index": index,
                "review_input_id": stable_id("d7review-input", args.batch_id, args.role, candidate_id),
                "review_role": args.role,
                "source_native_geometry_only": args.role == "GEOMETRY_EVIDENCE_REVIEWER",
                "counterexample_search_completed": args.role == "COUNTEREXAMPLE_REVIEWER",
                "review_origin": "FAIL_CLOSED_EXECUTION_TERMINAL",
                "independent_observation_recorded": False,
                "review_note": note,
            }
        )
    write_jsonl(output_path, rows)
    receipt = {
        "schema": "hftf_d7_public_real_fail_closed_review_terminal_receipt_v1",
        "record_kind": "FAIL_CLOSED_REVIEW_TERMINAL",
        "status": "COMPLETED_NOT_EVALUABLE_WITHOUT_INDEPENDENT_OBSERVATION",
        "generated_at_utc": utc_now(),
        "batch_id": args.batch_id,
        "review_role": args.role,
        "candidate_count": len(rows),
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "reason": note,
        "independent_observation_recorded": False,
        "decision_inferred": False,
        "event_truth_assigned": False,
        "admission_assigned": False,
        "split_assigned": False,
    }
    receipt_path = output_root / "receipts" / f"fail_closed_review_terminal_{args.batch_id}_{args.role}.json"
    write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
    except ContractError as exc:
        raise SystemExit(str(exc))
