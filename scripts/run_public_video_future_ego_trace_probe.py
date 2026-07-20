#!/usr/bin/env python3
"""Retrospective r7.53 probe using future frames as an offline ego-route teacher."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_future_ego_trace_probe_v1"


def map_future_anchor(h_current_to_future: np.ndarray, width: int, height: int, anchor_xy_norm: list[float]) -> tuple[float, float] | None:
    matrix = np.asarray(h_current_to_future, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    point = np.asarray([float(anchor_xy_norm[0]) * width, float(anchor_xy_norm[1]) * height, 1.0])
    mapped = inverse @ point
    if abs(float(mapped[2])) < 1e-9:
        return None
    return float(mapped[0] / mapped[2] / width), float(mapped[1] / mapped[2] / height)


def point_hits_expanded_detection(point: tuple[float, float], detections: list[dict[str, Any]], width: int, height: int, expansion_heights: float) -> bool:
    x = point[0] * width
    y = point[1] * height
    for detection in detections:
        x1, y1, x2, y2 = map(float, detection["xyxy"])
        object_height = max(1.0, y2 - y1)
        margin = expansion_heights * object_height
        if x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
            return True
    return False


def estimate_homography(current: np.ndarray, future: np.ndarray, policy: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    gray_a = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(future, cv2.COLOR_BGR2GRAY)
    height, width = gray_a.shape
    roi = policy["match_roi_y_norm"]
    mask = np.zeros_like(gray_a, dtype=np.uint8)
    mask[int(round(height * float(roi[0]))):int(round(height * float(roi[1]))), :] = 255
    orb = cv2.ORB_create(nfeatures=int(policy["maximum_features"]))
    key_a, desc_a = orb.detectAndCompute(gray_a, mask)
    key_b, desc_b = orb.detectAndCompute(gray_b, mask)
    if desc_a is None or desc_b is None:
        return None, {"match_count": 0, "inlier_count": 0, "reason": "missing_descriptors"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in pairs if first.distance < float(policy["ratio_test"]) * second.distance]
    if len(good) < int(policy["minimum_matches"]):
        return None, {"match_count": len(good), "inlier_count": 0, "reason": "insufficient_matches"}
    src = np.float32([key_a[row.queryIdx].pt for row in good]).reshape(-1, 1, 2)
    dst = np.float32([key_b[row.trainIdx].pt for row in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, float(policy["homography_ransac_reprojection_threshold_px"]))
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if matrix is None or inlier_count < int(policy["minimum_inliers"]):
        return None, {"match_count": len(good), "inlier_count": inlier_count, "reason": "insufficient_inliers"}
    return matrix, {"match_count": len(good), "inlier_count": inlier_count, "reason": "ok"}


def evaluate_event(features: dict[str, Any], source_id: str, window: tuple[int, int], policy: dict[str, Any]) -> dict[str, Any]:
    source_rows = [row for row in features["sources"] if row["source_id"] == source_id]
    if len(source_rows) != 1:
        raise ValueError(f"expected one source: {source_id}")
    source = source_rows[0]
    samples_by_time = {int(row["timestamp_ms"]): row for row in source["samples"]}
    event_times = sorted(time for time in samples_by_time if window[0] <= time < window[1])
    horizons = list(map(int, policy["future_horizons_ms"]))
    decode_times = sorted(set(event_times + [time + horizon for time in event_times for horizon in horizons]))
    frames = route_width.decode_at(Path(source["local_video_path"]), decode_times)
    frame_by_time = dict(zip(decode_times, frames))
    height, width = frames[0].shape[:2]
    bounds = list(map(float, policy["valid_trace_bounds_xy_norm"]))
    rows = []
    for timestamp in event_times:
        anchors = []
        diagnostics = []
        for horizon in horizons:
            matrix, diagnostic = estimate_homography(frame_by_time[timestamp], frame_by_time[timestamp + horizon], policy)
            point = None if matrix is None else map_future_anchor(matrix, width, height, policy["future_anchor_xy_norm"])
            valid = point is not None and bounds[0] <= point[0] <= bounds[1] and bounds[0] <= point[1] <= bounds[1]
            hit = valid and point_hits_expanded_detection(
                point, samples_by_time[timestamp].get("detections", []), width, height,
                float(policy["obstacle_expansion_object_heights"]),
            )
            if valid:
                anchors.append({"horizon_ms": horizon, "point_xy_norm": list(point), "obstacle_hit": bool(hit)})
            diagnostics.append({"horizon_ms": horizon, **diagnostic, "valid_anchor": bool(valid)})
        score = float(sum(row["obstacle_hit"] for row in anchors) / len(anchors)) if anchors else None
        rows.append({"timestamp_ms": timestamp, "valid_anchor_count": len(anchors), "trace_intrusion_score": score,
                     "anchors": anchors, "homography": diagnostics})
    valid_scores = [float(row["trace_intrusion_score"]) for row in rows if row["trace_intrusion_score"] is not None]
    return {
        "source_id": source_id,
        "window_ms": list(window),
        "frame_count": len(rows),
        "valid_frame_count": len(valid_scores),
        "valid_frame_fraction": len(valid_scores) / max(1, len(rows)),
        "median_trace_intrusion_score": float(median(valid_scores)) if valid_scores else None,
        "mean_trace_intrusion_score": float(np.mean(valid_scores)) if valid_scores else None,
        "frames": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.positive_features, args.negative_features, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    positive_features = lifecycle.verify_json_sidecar(args.positive_features)
    negative_features = lifecycle.verify_json_sidecar(args.negative_features)
    policy = contract["teacher"]
    positive = evaluate_event(positive_features, contract["evaluation"]["positive_source"], tuple(args.positive_window), policy)
    negative = evaluate_event(negative_features, contract["evaluation"]["negative_source"], tuple(args.negative_window), policy)
    minimum_valid = float(contract["evaluation"]["minimum_valid_frame_fraction"])
    checks = {
        "positive_valid_frame_fraction": positive["valid_frame_fraction"] >= minimum_valid,
        "negative_valid_frame_fraction": negative["valid_frame_fraction"] >= minimum_valid,
        "positive_exceeds_negative": positive["median_trace_intrusion_score"] is not None
        and negative["median_trace_intrusion_score"] is not None
        and positive["median_trace_intrusion_score"] > negative["median_trace_intrusion_score"],
    }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "contract_sha256": common.sha256_file(args.contract),
            "positive_feature_report_sha256": common.sha256_file(args.positive_features),
            "negative_feature_report_sha256": common.sha256_file(args.negative_features),
        },
        "positive": positive,
        "negative": negative,
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
    parser.add_argument("--positive-features", type=Path, required=True)
    parser.add_argument("--negative-features", type=Path, required=True)
    parser.add_argument("--positive-window", type=int, nargs=2, required=True)
    parser.add_argument("--negative-window", type=int, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "diagnostic_gate_passed": value["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
