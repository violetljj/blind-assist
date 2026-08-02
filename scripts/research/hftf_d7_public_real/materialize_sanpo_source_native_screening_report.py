#!/usr/bin/env python3
"""Materialize a source-native-only review-priority queue for a SANPO batch.

This queue is for bounded review triage only.  It contains no event bucket,
decision, label, or admission field and must never be used as event truth.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


def _priority_row(manifest_row: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    frames = summary.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"geometry summary has no frames: {summary.get('candidate_id')}")
    deltas = [
        float(frame["median_abs_depth_delta_m_from_previous_source_frame"])
        for frame in frames
        if frame.get("median_abs_depth_delta_m_from_previous_source_frame") is not None
    ]
    medians = [
        float(frame["source_native_depth_stats"]["quantiles_m"]["p50"])
        for frame in frames
        if isinstance(frame.get("source_native_depth_stats"), dict)
        and isinstance(frame["source_native_depth_stats"].get("quantiles_m"), dict)
        and frame["source_native_depth_stats"]["quantiles_m"].get("p50") is not None
    ]
    valid_fractions = [
        float(frame["source_native_depth_stats"]["valid_positive_finite_fraction"])
        for frame in frames
        if isinstance(frame.get("source_native_depth_stats"), dict)
        and frame["source_native_depth_stats"].get("valid_positive_finite_fraction") is not None
    ]
    return {
        "schema": "hftf_d7_public_real_sanpo_source_native_screening_row_v1",
        "record_kind": "SOURCE_NATIVE_REVIEW_PRIORITY_ONLY",
        "dataset_id": "SANPO-Real",
        "candidate_id": manifest_row.get("candidate_id"),
        "review_input_id": manifest_row.get("review_input_id"),
        "source_session_token": summary.get("source_session_token"),
        "source_native_only": True,
        "rgb_included": False,
        "model_output_visible": False,
        "event_truth_inferred": False,
        "decision": None,
        "event_bucket": None,
        "admission_status": None,
        "screening_metrics": {
            "frame_count": len(frames),
            "median_depth_delta_m": statistics.median(deltas) if deltas else None,
            "max_depth_delta_m": max(deltas) if deltas else None,
            "depth_p50_range_m": max(medians) - min(medians) if medians else None,
            "minimum_valid_depth_fraction": min(valid_fractions) if valid_fractions else None,
        },
        "screening_semantics": "review-priority-only descriptive source-native depth metrics; not event truth",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    batch_root = root / "reviews" / "input_bundles" / args.batch_id
    manifest_path = batch_root / "manifests" / "GEOMETRY_EVIDENCE_REVIEWER.jsonl"
    if not manifest_path.is_file():
        raise ContractError(f"SANPO geometry manifest missing: {manifest_path}")
    rows: list[dict[str, Any]] = []
    for manifest_row in load_jsonl(manifest_path):
        geometry_path = Path(str(manifest_row.get("native_geometry_path") or ""))
        geometry = load_json(geometry_path)
        if not isinstance(geometry, dict):
            raise ContractError(f"SANPO native geometry is not an object: {geometry_path}")
        summary_path = Path(str(geometry.get("geometry_evidence_summary_path") or ""))
        if not summary_path.is_file():
            raise ContractError(f"SANPO geometry summary missing: {summary_path}")
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            raise ContractError(f"SANPO geometry summary is not an object: {summary_path}")
        rows.append(_priority_row(manifest_row, summary))
    rows.sort(
        key=lambda row: (
            -(row["screening_metrics"]["max_depth_delta_m"] or -1.0),
            -(row["screening_metrics"]["depth_p50_range_m"] or -1.0),
            str(row["candidate_id"]),
        )
    )
    for index, row in enumerate(rows, 1):
        row["review_priority_rank"] = index
    output_root = root / "development" / "sanpo_source_native_screening" / args.run_id
    if output_root.exists():
        raise ContractError(f"screening report output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    queue_path = output_root / "source_native_review_priority_queue.jsonl"
    write_jsonl(queue_path, rows)
    report = {
        "schema": "hftf_d7_public_real_sanpo_source_native_screening_report_v1",
        "record_kind": "DEVELOPMENT_SCREENING_REPORT",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "candidate_count": len(rows),
        "status": "DEVELOPMENT_REVIEW_PRIORITY_ONLY",
        "source_native_only": True,
        "rgb_included": False,
        "model_output_visible": False,
        "event_truth_inferred": False,
        "decision_fields_are_null": True,
        "ranking_rule": [
            "descending max median absolute depth delta across adjacent source frames",
            "then descending source-frame median-depth range",
            "then candidate_id ascending",
        ],
        "queue_path": str(queue_path.resolve()),
        "queue_sha256": sha256_file(queue_path),
    }
    report_path = output_root / "screening_report.json"
    write_json(report_path, report)
    return {**report, "report_path": str(report_path.resolve()), "report_sha256": sha256_file(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
