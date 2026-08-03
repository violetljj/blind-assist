#!/usr/bin/env python3
"""Materialize a fail-closed adjudication when the geometry gate is absent.

This is intentionally narrower than a human/adjudicator labeler. It can only
emit NOT_EVALUABLE terminals after all five review roles have been ingested and
the source-native geometry role has marked every candidate NOT_EVALUABLE. It
cannot emit an admission or a negative event label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


GEOMETRY_ROLE = "GEOMETRY_EVIDENCE_REVIEWER"


def _assert_model_blind(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"model_output_visible", "review_model_output_visible", "geometry_model_output_visible"}:
                if child is not False:
                    raise ContractError(f"model visibility is not false: {path}.{key}")
            _assert_model_blind(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_model_blind(child, path=f"{path}[{index}]")


def _build_rows(inputs: list[dict[str, Any]], *, batch_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(inputs):
        candidate_id = str(item.get("candidate_id") or "")
        event_id = str(item.get("event_id") or "")
        if not candidate_id or not event_id or item.get("batch_id") != batch_id:
            raise ContractError(f"adjudication input identity mismatch at index {index}")
        reviews = item.get("review_records")
        if not isinstance(reviews, dict) or set(reviews) != {
            "RGB_REVIEWER_A",
            "RGB_REVIEWER_B",
            "RGB_REVIEWER_C",
            "GEOMETRY_EVIDENCE_REVIEWER",
            "COUNTEREXAMPLE_REVIEWER",
        }:
            raise ContractError(f"five independent review records are required: {candidate_id}")
        geometry = reviews[GEOMETRY_ROLE]
        if geometry.get("event_bucket") != "NOT_EVALUABLE" or geometry.get("decision") != "NOT_EVALUABLE":
            raise ContractError(f"geometry gate is not uniformly NOT_EVALUABLE: {candidate_id}")
        _assert_model_blind(item, path=f"$[{index}]")
        rows.append({
            "schema": "hftf_d7_public_real_completed_adjudication_v1",
            "record_kind": "COMPLETED_ADJUDICATION",
            "batch_id": batch_id,
            "adjudication_index": index,
            "adjudication_input_id": item.get("adjudication_input_id"),
            "candidate_id": candidate_id,
            "event_id": event_id,
            "dataset_id": item.get("dataset_id"),
            "adjudication_decision": "NOT_EVALUABLE",
            "admission_status": "NOT_ADMITTED",
            "event_bucket": "NOT_EVALUABLE",
            "phase_intervals": None,
            "model_output_visible": False,
            "review_model_output_visible": False,
            "geometry_model_output_visible": False,
            "reason_code": "GEOMETRY_REQUIRED_GATE_NOT_EVALUABLE",
            "notes": (
                "All five independent review roles were ingested, but the "
                "source-native geometry role lacks semantic obstacle geometry, "
                "tracks, and segmentation; no event truth or negative label is admitted."
            ),
            "training_authorized": False,
            "confirmation_authorized": False,
            "production_authorized": False,
        })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    bundle_root = root / "reviews" / "adjudication_bundles" / args.batch_id
    bundle_manifest = load_json(bundle_root / "bundle_manifest.json")
    if bundle_manifest.get("status") != "READY_FOR_FINAL_ADJUDICATION":
        raise ContractError("adjudication bundle is not ready")
    input_path = bundle_root / "FINAL_ADJUDICATOR_INPUT.jsonl"
    inputs = load_jsonl(input_path)
    selected_ids = [str(value) for value in bundle_manifest.get("candidate_ids", [])]
    if len(inputs) != len(selected_ids) or [str(row.get("candidate_id")) for row in inputs] != selected_ids:
        raise ContractError("adjudication input candidate order/count mismatch")
    rows = _build_rows(inputs, batch_id=args.batch_id)
    output_path = bundle_root / "FINAL_ADJUDICATOR" / "final_adjudication.jsonl"
    if output_path.exists():
        raise ContractError(f"refusing to overwrite adjudication output: {output_path}")
    write_jsonl(output_path, rows)
    receipt = {
        "schema": "hftf_d7_public_real_conservative_adjudication_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "FINAL_ADJUDICATION_FAIL_CLOSED_NOT_EVALUABLE",
        "candidate_count": len(rows),
        "admitted_parent_events": 0,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "authority": {
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
        },
        "notes": [
            "This deterministic terminal is permitted only because all five roles were ingested and geometry was uniformly NOT_EVALUABLE.",
            "No negative or positive event label was synthesized from missing geometry.",
        ],
    }
    write_json(root / "receipts" / f"conservative_adjudication_receipt_{args.batch_id}.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
