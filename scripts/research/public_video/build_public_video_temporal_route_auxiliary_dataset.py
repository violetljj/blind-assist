#!/usr/bin/env python3
"""Build the frozen r7.61 source-isolated temporal route auxiliary manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_dense_future_ego_trace_probe as dense
import run_public_video_ego_trace_distillation_probe as geometry
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_temporal_route_auxiliary_dataset_v1"


def evenly_spaced(values: list[int], maximum: int) -> list[int]:
    if len(values) <= maximum:
        return list(values)
    indices = np.linspace(0, len(values) - 1, int(maximum)).round().astype(int)
    return [values[index] for index in indices]


def build_source(source: dict[str, Any], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = {int(row["timestamp_ms"]): row for row in source["samples"]}
    horizons = list(map(int, policy["future_horizons_ms"]))
    history = int(policy["causal_history_ms"])
    maximum_time = max(samples)
    eligible = sorted(timestamp for timestamp, sample in samples.items()
                      if timestamp >= history and timestamp + max(horizons) <= maximum_time
                      and (not policy["requires_at_least_one_frozen_marker_detection"] or sample.get("detections")))
    selected = evenly_spaced(eligible, int(policy["maximum_samples_per_source"]))
    decode_times = sorted(set(selected + [timestamp + horizon for timestamp in selected for horizon in horizons]))
    frames = route_width.decode_at(Path(source["local_video_path"]), decode_times)
    frame_by_time = dict(zip(decode_times, frames))
    gray = {timestamp: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for timestamp, frame in frame_by_time.items()}
    bounds = list(map(float, policy["valid_trace_bounds_xy_norm"]))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rows = []
    hit_scores = []
    for timestamp in selected:
        anchors = []
        for horizon in horizons:
            flow = dis.calc(gray[timestamp + horizon], gray[timestamp], None)
            point = dense.map_anchor_with_flow(flow, list(map(float, policy["future_anchor_xy_norm"])))
            valid = point is not None and bounds[0] <= point[0] <= bounds[1] and bounds[0] <= point[1] <= bounds[1]
            if valid:
                anchors.append({"horizon_ms": horizon, "point_xy_norm": [float(point[0]), float(point[1])]})
        if policy["requires_all_future_anchors_valid"] and len(anchors) != len(horizons):
            continue
        detections = samples[timestamp].get("detections", [])
        hits = [geometry.point_box_distance(tuple(anchor["point_xy_norm"]), detections, 0.5) <= 1e-12 for anchor in anchors]
        hit_scores.append(sum(hits) / max(1, len(hits)))
        rows.append({
            "item_id": f"{source['source_id']}__t{timestamp:010d}",
            "source_id": source["source_id"],
            "source_video_sha256": source["video_sha256"],
            "local_video_path": source["local_video_path"],
            "timestamp_ms": timestamp,
            "causal_clip_window_ms": [timestamp - history, timestamp],
            "future_route_anchors": anchors,
            "marker_detection_count": len(detections),
            "teacher_marker_hit_fraction_diagnostic_only": float(hit_scores[-1]),
            "training_role": "route_auxiliary_only",
            "event_label": None,
            "human_event_truth_present": False,
        })
    return rows, {
        "source_id": source["source_id"], "eligible_count": len(eligible), "selected_count": len(selected),
        "accepted_count": len(rows), "teacher_marker_hit_fraction_mean": float(np.mean(hit_scores)) if hit_scores else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.source_contract, args.teacher_contract, args.teacher_validation, args.spatial_report,
                 args.output_root, args.output_report):
        mil.reject_independent_direction(path)
    manifest_path = args.output_root / "manifest.jsonl"
    if args.output_root.exists() or args.output_report.exists() or Path(str(args.output_report) + ".sha256").exists():
        raise ValueError("refusing to overwrite route auxiliary outputs")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source_contract = json.loads(args.source_contract.read_text(encoding="utf-8"))
    bindings = contract["bound_inputs"]
    for path, key in ((args.source_contract, "source_and_feature_contract_sha256"),
                      (args.teacher_contract, "dense_future_teacher_contract_sha256"),
                      (args.teacher_validation, "dense_future_teacher_validation_sha256"),
                      (args.spatial_report, "spatial_predictability_report_sha256")):
        if common.sha256_file(path) != bindings[key]:
            raise ValueError(f"bound input hash mismatch: {path}")
    reports = []
    for key, binding in source_contract["feature_reports"].items():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        reports.append(lifecycle.verify_json_sidecar(path))
    sources = []
    for report in reports:
        sources.extend(report["sources"])
    source_ids = [row["source_id"] for row in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source contract contains duplicate source ids")
    policy = {**contract["eligibility"], **contract["target"]}
    manifest = []
    summaries = []
    for source in sources:
        rows, summary = build_source(source, policy)
        manifest.extend(rows)
        summaries.append(summary)
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest), encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "source_contract_sha256": common.sha256_file(args.source_contract),
                   "teacher_validation_sha256": common.sha256_file(args.teacher_validation)},
        "manifest": {"path": str(manifest_path), "sha256": common.sha256_file(manifest_path), "row_count": len(manifest)},
        "source_count": len(sources), "sources": summaries,
        "source_isolation": contract["isolation"], "evidence_role": contract["evidence_role"],
        "authorization": contract["authorization"],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output_report) + ".sha256").write_text(common.sha256_file(args.output_report) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--teacher-validation", type=Path, required=True)
    parser.add_argument("--spatial-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "row_count": value["manifest"]["row_count"], "source_count": value["source_count"],
                      "output_sha256": common.sha256_file(args.output_report)}, ensure_ascii=False))
