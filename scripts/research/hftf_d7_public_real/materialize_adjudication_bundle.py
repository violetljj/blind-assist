#!/usr/bin/env python3
"""Materialize final-adjudicator inputs after independent reviews complete."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ingest_review_outputs import ROLE_TO_PRIMARY
from materialize_review_bundle import ALLOWED_EVENT_BUCKETS
from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


def _assert_no_model_output(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"model_hint", "candidate_selection", "model_output_visible_to_selector", "selection_role", "native_geometry_used_for_selection", "parent_independence_status", "required_confirmation_selection"}:
                raise ContractError(f"model discovery field leaked into adjudication input: {path}.{key}")
            if key in {"model_output_visible", "review_model_output_visible", "geometry_model_output_visible"} and child is not False:
                raise ContractError(f"model visibility is not false: {path}.{key}")
            _assert_no_model_output(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_model_output(child, path=f"{path}[{index}]")


def _find_review(rows: list[dict[str, Any]], *, role: str, candidate_id: str) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("candidate_id")) == candidate_id]
    if len(matches) != 1:
        raise ContractError(f"expected one {role} review for {candidate_id}, got {len(matches)}")
    row = matches[0]
    if row.get("record_kind") != "COMPLETED_REVIEW" or row.get("review_completed") is not True:
        raise ContractError(f"{role} review is not completed: {candidate_id}")
    if row.get("model_output_visible") is not False:
        raise ContractError(f"{role} review model visibility drift: {candidate_id}")
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    input_batch_root = root / "reviews" / "input_bundles" / args.batch_id
    input_manifest = load_json(input_batch_root / "bundle_manifest.json")
    if not isinstance(input_manifest, dict) or input_manifest.get("status") != "READY_FOR_ISOLATED_REVIEW":
        raise ContractError("the source review bundle is not ready")
    selected_ids = [str(value) for value in input_manifest.get("candidate_ids", [])]
    if not selected_ids:
        raise ContractError("the source review bundle has no candidate IDs")
    output_root = root / "reviews" / "adjudication_bundles" / args.batch_id
    if output_root.exists():
        raise ContractError(f"adjudication bundle already exists; refusing overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "evidence" / "contact_sheets").mkdir(parents=True, exist_ok=False)
    (output_root / "evidence" / "native_geometry").mkdir(parents=True, exist_ok=False)

    event_rows = load_jsonl(root / "manifests" / "event_manifest.jsonl")
    event_by_candidate = {str(row.get("candidate_id")): row for row in event_rows}
    review_rows = {
        role: load_jsonl(root / relative)
        for role, relative in ROLE_TO_PRIMARY.items()
    }
    rgb_manifest_rows = load_jsonl(input_batch_root / "manifests" / "RGB_REVIEWER_A.jsonl")
    geometry_manifest_rows = load_jsonl(input_batch_root / "manifests" / "GEOMETRY_EVIDENCE_REVIEWER.jsonl")
    rgb_input_by_candidate = {str(row.get("candidate_id")): row for row in rgb_manifest_rows}
    geometry_input_by_candidate = {str(row.get("candidate_id")): row for row in geometry_manifest_rows}
    inputs: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(selected_ids):
        event = event_by_candidate.get(candidate_id)
        if event is None:
            raise ContractError(f"candidate is absent from event manifest: {candidate_id}")
        raw_reviews = {
            role: _find_review(rows, role=role, candidate_id=candidate_id)
            for role, rows in review_rows.items()
        }
        rgb_input = rgb_input_by_candidate.get(candidate_id)
        geometry_input = geometry_input_by_candidate.get(candidate_id)
        if not isinstance(rgb_input, dict) or not isinstance(geometry_input, dict):
            raise ContractError(f"input manifest row missing for adjudication candidate: {candidate_id}")
        source_sheet = Path(str(rgb_input.get("contact_sheet_path"))).resolve()
        source_geometry = Path(str(geometry_input.get("native_geometry_path"))).resolve()
        if not source_sheet.is_file() or not source_geometry.is_file():
            raise ContractError(f"adjudication evidence file missing: {candidate_id}")
        sheet_path = output_root / "evidence" / "contact_sheets" / f"{candidate_id}.jpg"
        geometry_path = output_root / "evidence" / "native_geometry" / f"{candidate_id}.json"
        shutil.copy2(source_sheet, sheet_path)
        shutil.copy2(source_geometry, geometry_path)
        review_projection = {
            role: {
                "review_role": role,
                "decision": row.get("decision"),
                "event_bucket": row.get("event_bucket"),
                "phase_intervals": row.get("phase_intervals"),
                "evidence_basis": row.get("evidence_basis"),
                "source_native_geometry_only": row.get("source_native_geometry_only"),
                "counterexample_search_completed": row.get("counterexample_search_completed"),
                "review_notes": row.get("review_notes"),
                "reason_code": row.get("reason_code"),
                "model_output_visible": False,
            }
            for role, row in raw_reviews.items()
        }
        adjudication_input = {
            "schema": "hftf_d7_public_real_adjudication_input_v1",
            "record_kind": "ADJUDICATION_INPUT",
            "batch_id": args.batch_id,
            "adjudication_input_id": stable_id("d7adjudication-input", args.batch_id, candidate_id),
            "review_index": index,
            "candidate_id": candidate_id,
            "event_id": event.get("event_id"),
            "dataset_id": event.get("dataset_id"),
            "source_session_token": rgb_input.get("source_session_token"),
            "window_frame_ids": event.get("frame_ids") if isinstance(event.get("frame_ids"), list) else [],
            "window_start_timestamp_ns": event.get("start_timestamp_ns"),
            "window_end_timestamp_ns": event.get("end_timestamp_ns"),
            "contact_sheet_path": str(sheet_path.resolve()),
            "contact_sheet_sha256": sha256_file(sheet_path),
            "native_geometry_path": str(geometry_path.resolve()),
            "native_geometry_sha256": sha256_file(geometry_path),
            "review_records": review_projection,
            "allowed_event_buckets": ALLOWED_EVENT_BUCKETS,
            "model_output_visible": False,
            "instructions": (
                "Consume all five independent review records and the copied "
                "evidence. Resolve disagreement conservatively. Admit only "
                "when all required evidence, phase intervals, source-session "
                "identity, and counterexample gates are satisfied. If any "
                "required gate is missing or conflicted, output NOT_EVALUABLE "
                "or ESCALATE; missing geometry is not a negative."
            ),
        }
        _assert_no_model_output(adjudication_input)
        inputs.append(adjudication_input)

    manifest_path = output_root / "FINAL_ADJUDICATOR_INPUT.jsonl"
    write_jsonl(manifest_path, inputs)
    bundle_manifest = {
        "schema": "hftf_d7_public_real_adjudication_bundle_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_FINAL_ADJUDICATION",
        "candidate_count": len(inputs),
        "candidate_ids": selected_ids,
        "input_path": str(manifest_path.resolve()),
        "input_sha256": sha256_file(manifest_path),
        "all_independent_roles_present": True,
        "model_output_visible_in_any_input": False,
        "adjudication_writes_event_truth": False,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    }
    _assert_no_model_output(bundle_manifest)
    write_json(output_root / "bundle_manifest.json", bundle_manifest)
    receipt = {
        "schema": "hftf_d7_public_real_adjudication_bundle_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_FINAL_ADJUDICATION",
        "candidate_count": len(inputs),
        "input_path": str(manifest_path.resolve()),
        "input_sha256": sha256_file(manifest_path),
        "bundle_manifest_path": str((output_root / "bundle_manifest.json").resolve()),
        "bundle_manifest_sha256": sha256_file(output_root / "bundle_manifest.json"),
        "model_output_visible_in_any_input": False,
        "adjudicated_parent_events": 0,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    }
    write_json(root / "receipts" / f"adjudication_bundle_receipt_{args.batch_id}.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
