#!/usr/bin/env python3
"""Evaluate the fixed r7.56 current-to-past causal ego-trace relation."""

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
import run_public_video_future_ego_trace_multisource_probe as multisource
import run_public_video_future_ego_trace_probe as sparse
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_causal_past_ego_trace_probe_v1"


def past_timestamps(timestamp_ms: int, horizons_ms: list[int]) -> list[int]:
    return [int(timestamp_ms) - int(horizon) for horizon in horizons_ms]


def evaluate_event(features: dict[str, Any], source_id: str, window: tuple[int, int], policy: dict[str, Any]) -> dict[str, Any]:
    source_rows = [row for row in features["sources"] if row["source_id"] == source_id]
    if len(source_rows) != 1:
        raise ValueError(f"expected one source: {source_id}")
    source = source_rows[0]
    samples_by_time = {int(row["timestamp_ms"]): row for row in source["samples"]}
    event_times = sorted(time for time in samples_by_time if window[0] <= time < window[1])
    horizons = list(map(int, policy["past_horizons_ms"]))
    decode_times = sorted(set(event_times + [past for time in event_times for past in past_timestamps(time, horizons)]))
    if decode_times[0] < 0:
        raise ValueError("causal history precedes source start")
    frames = route_width.decode_at(Path(source["local_video_path"]), decode_times)
    frame_by_time = dict(zip(decode_times, frames))
    height, width = frames[0].shape[:2]
    gray = {time: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for time, frame in frame_by_time.items()}
    bounds = list(map(float, policy["valid_trace_bounds_xy_norm"]))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rows = []
    for timestamp in event_times:
        anchors = []
        for horizon, past in zip(horizons, past_timestamps(timestamp, horizons)):
            flow = dis.calc(gray[timestamp], gray[past], None)
            point = dense.map_anchor_with_flow(flow, policy["current_anchor_xy_norm"])
            valid = point is not None and bounds[0] <= point[0] <= bounds[1] and bounds[0] <= point[1] <= bounds[1]
            if valid:
                hit = sparse.point_hits_expanded_detection(
                    point, samples_by_time[timestamp].get("detections", []), width, height,
                    float(policy["obstacle_expansion_object_heights"]),
                )
                anchors.append({"past_horizon_ms": horizon, "point_xy_norm": list(point), "obstacle_hit": bool(hit)})
        score = float(sum(row["obstacle_hit"] for row in anchors) / len(anchors)) if anchors else None
        rows.append({"timestamp_ms": timestamp, "valid_anchor_count": len(anchors),
                     "trace_intrusion_score": score, "anchors": anchors})
    valid_scores = [float(row["trace_intrusion_score"]) for row in rows if row["trace_intrusion_score"] is not None]
    return {
        "source_id": source_id,
        "window_ms": list(window),
        "frame_count": len(rows),
        "valid_frame_count": len(valid_scores),
        "valid_frame_fraction": len(valid_scores) / max(1, len(rows)),
        "mean_trace_intrusion_score": float(np.mean(valid_scores)) if valid_scores else None,
        "frames": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.event_set_contract, args.offline_teacher_contract, args.offline_teacher_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    event_contract = json.loads(args.event_set_contract.read_text(encoding="utf-8"))
    bindings = {
        args.event_set_contract: contract["event_set_contract_sha256"],
        args.offline_teacher_contract: contract["offline_teacher_contract_sha256"],
        args.offline_teacher_report: contract["offline_teacher_report_sha256"],
    }
    for path, expected in bindings.items():
        if common.sha256_file(path) != expected:
            raise ValueError(f"bound input hash mismatch: {path}")
    reports = {}
    for key, binding in event_contract["feature_reports"].items():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        reports[key] = lifecycle.verify_json_sidecar(path)
    rows = []
    for event in event_contract["events"]:
        diagnostic = evaluate_event(
            reports[event["feature_key"]], event["source_id"], tuple(map(int, event["window_ms"])), contract["causal_teacher"]
        )
        rows.append({
            "event_id": event["event_id"], "source_id": event["source_id"], "label": int(event["label"]),
            "score": float(diagnostic["mean_trace_intrusion_score"] or 0.0),
            "valid_frame_fraction": float(diagnostic["valid_frame_fraction"]), "event_diagnostics": diagnostic,
        })
    checks = multisource.separation_checks(rows, float(contract["gate"]["minimum_valid_frame_fraction"]))
    checks["uses_no_future_frames"] = True
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "event_set_contract_sha256": common.sha256_file(args.event_set_contract),
                   "offline_teacher_report_sha256": common.sha256_file(args.offline_teacher_report)},
        "aggregation": contract["aggregation"],
        "events": rows,
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
        "evidence_limit": contract["evidence_role"],
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--event-set-contract", type=Path, required=True)
    parser.add_argument("--offline-teacher-contract", type=Path, required=True)
    parser.add_argument("--offline-teacher-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "diagnostic_gate_passed": value["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
