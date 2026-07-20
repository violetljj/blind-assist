#!/usr/bin/env python3
"""Audit normalized device geometry against every frozen r797a route anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

SCHEMA = "blindassist_explicit_route_geometry_conformance_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_source(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for field in ("source_id", "local_video_path", "video_sha256"):
        if target.get(field) != incoming.get(field):
            raise ValueError(f"duplicate source metadata mismatch: {incoming.get('source_id')} {field}")
    by_time = {int(row["timestamp_ms"]): row for row in target["samples"]}
    for sample in incoming["samples"]:
        timestamp = int(sample["timestamp_ms"])
        if timestamp in by_time and by_time[timestamp] != sample:
            raise ValueError(f"duplicate source sample mismatch: {incoming['source_id']} {timestamp}")
        by_time[timestamp] = sample
    target["samples"] = [by_time[key] for key in sorted(by_time)]


def point_hits_normalized(
    point: tuple[float, float], detections: list[dict[str, Any]], width: int, height: int,
    expansion_heights: float,
) -> bool:
    """Mirror aspect-aware ExplicitRouteGeometryFusion in normalized coordinates."""
    for detection in detections:
        x1, y1, x2, y2 = map(float, detection["xyxy"])
        left, top, right, bottom = x1 / width, y1 / height, x2 / width, y2 / height
        object_height_px = max(1.0, y2 - y1)
        margin_px = expansion_heights * object_height_px
        margin_x, margin_y = margin_px / width, margin_px / height
        if (left - margin_x <= point[0] <= right + margin_x and
                top - margin_y <= point[1] <= bottom + margin_y):
            return True
    return False


def merge_sources(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for binding in report["inputs"]["verified_feature_reports"]:
        path = Path(binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {path}")
        for incoming in load_json(path)["sources"]:
            source_id = incoming["source_id"]
            if source_id in sources:
                merge_source(sources[source_id], incoming)
            else:
                sources[source_id] = {**incoming, "samples": list(incoming["samples"])}
    return sources


def video_dimensions(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"cannot read video dimensions: {path}")
    return width, height


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError("refusing to overwrite conformance output")
    report = load_json(args.oracle_report)
    contract = load_json(args.device_contract)
    expansion = float(contract["geometry"]["obstacle_expansion_object_heights"])
    sources = merge_sources(report)
    dimensions: dict[str, tuple[int, int]] = {}
    checked_frames = checked_anchors = 0
    hit_mismatches: list[dict[str, Any]] = []
    score_mismatches: list[dict[str, Any]] = []
    for event in report["events"]:
        source_id = event["parent_source_id"]
        source = sources[source_id]
        if source_id not in dimensions:
            dimensions[source_id] = video_dimensions(Path(source["local_video_path"]))
        width, height = dimensions[source_id]
        samples = {int(row["timestamp_ms"]): row for row in source["samples"]}
        for frame in event["frames"]:
            timestamp = int(frame["timestamp_ms"])
            detections = samples[timestamp].get("detections", [])
            hits = []
            for anchor in frame["anchors"]:
                point = tuple(map(float, anchor["point_xy_norm"]))
                actual = point_hits_normalized(point, detections, width, height, expansion)
                expected = bool(anchor["obstacle_hit"])
                hits.append(actual)
                checked_anchors += 1
                if actual != expected:
                    hit_mismatches.append({"item_id": event["item_id"], "timestamp_ms": timestamp,
                                           "horizon_ms": anchor["horizon_ms"], "expected": expected,
                                           "normalized_device_mirror": actual})
            actual_score = sum(hits) / len(hits) if hits else None
            expected_score = frame["trace_intrusion_score"]
            checked_frames += 1
            if actual_score != expected_score:
                score_mismatches.append({"item_id": event["item_id"], "timestamp_ms": timestamp,
                                         "expected": expected_score, "normalized_device_mirror": actual_score})
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "oracle_report_path": args.oracle_report.as_posix(),
            "oracle_report_sha256": sha256_file(args.oracle_report),
            "device_contract_path": args.device_contract.as_posix(),
            "device_contract_sha256": sha256_file(args.device_contract),
        },
        "policy": {"coordinate_space": "normalized_current_camera_frame_xy",
                   "obstacle_expansion_object_heights": expansion,
                   "inclusive_bounds": True},
        "summary": {"event_count": len(report["events"]), "source_count": len(dimensions),
                    "checked_frame_count": checked_frames, "checked_anchor_count": checked_anchors,
                    "hit_mismatch_count": len(hit_mismatches),
                    "score_mismatch_count": len(score_mismatches),
                    "exact_conformance": not hit_mismatches and not score_mismatches},
        "hit_mismatches": hit_mismatches,
        "score_mismatches": score_mismatches,
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--device-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args())["summary"], ensure_ascii=False))
