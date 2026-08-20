#!/usr/bin/env python3
"""Mine Goal Copilot target episodes from ADT evaluator-only ground truth."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RGB_STREAM = "214-1"


@dataclass(frozen=True)
class Thresholds:
    visible_ratio: float = 0.10
    min_track_frames: int = 12
    min_hidden_frames: int = 6
    min_lost_frames: int = 6
    min_approach_m: float = 0.25
    min_approach_fraction: float = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(zf: zipfile.ZipFile, member: str) -> Iterable[dict[str, str]]:
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        yield from csv.DictReader(text)


def nearest(rows: list[tuple[int, tuple[float, float, float]]], timestamp_ns: int):
    if not rows:
        return None
    times = [row[0] for row in rows]
    index = bisect.bisect_left(times, timestamp_ns)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    return min(candidates, key=lambda i: abs(times[i] - timestamp_ns))


def instance_name(instances: Any, uid: str) -> str:
    if isinstance(instances, dict):
        value = instances.get(uid) or instances.get(str(uid))
        if isinstance(value, dict):
            return str(value.get("instance_name") or value.get("prototype_name") or uid)
        for candidate in instances.values():
            if isinstance(candidate, list):
                for item in candidate:
                    item_uid = item.get("instance_id", item.get("object_uid")) if isinstance(item, dict) else None
                    if str(item_uid) == uid:
                        return str(item.get("instance_name") or item.get("prototype_name") or uid)
    return uid


def contiguous_segments(mask: list[bool]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index))
            start = None
    return segments


def median_endpoint(values: list[float | None], start: int, end: int, head: bool) -> float | None:
    materialized = [value for value in values[start:end] if value is not None and math.isfinite(value)]
    if len(materialized) < 4:
        return None
    width = max(2, len(materialized) // 4)
    return float(statistics.median(materialized[:width] if head else materialized[-width:]))


def load_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        required = {"instances.json", "aria_trajectory.csv", "scene_objects.csv"}
        missing = sorted(required - set(zf.namelist()))
        if missing:
            raise ValueError(f"ADT groundtruth archive missing {missing}")
        bbox_member = "2d_bounding_box_with_skeleton.csv" if "2d_bounding_box_with_skeleton.csv" in zf.namelist() else "2d_bounding_box.csv"
        instances = json.load(zf.open("instances.json"))
        boxes: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
        frame_times: set[int] = set()
        for row in csv_rows(zf, bbox_member):
            if row["stream_id"] != RGB_STREAM:
                continue
            uid, timestamp = str(row["object_uid"]), int(row["timestamp[ns]"])
            frame_times.add(timestamp)
            boxes[uid][timestamp] = {
                "visibility": float(row["visibility_ratio[%]"]),
                "x_min": float(row["x_min[pixel]"]), "x_max": float(row["x_max[pixel]"]),
                "y_min": float(row["y_min[pixel]"]), "y_max": float(row["y_max[pixel]"]),
            }

        device = []
        for row in csv_rows(zf, "aria_trajectory.csv"):
            device.append((int(row["tracking_timestamp_us"]) * 1000, (float(row["tx_world_device"]), float(row["ty_world_device"]), float(row["tz_world_device"]))))
        device.sort()

        static: dict[str, tuple[float, float, float]] = {}
        dynamic: dict[str, list[tuple[int, tuple[float, float, float]]]] = defaultdict(list)
        for row in csv_rows(zf, "scene_objects.csv"):
            uid, timestamp = str(row["object_uid"]), int(row["timestamp[ns]"])
            position = (float(row["t_wo_x[m]"]), float(row["t_wo_y[m]"]), float(row["t_wo_z[m]"]))
            if timestamp == -1:
                static[uid] = position
            else:
                dynamic[uid].append((timestamp, position))
        for rows in dynamic.values():
            rows.sort()

    return {"instances": instances, "boxes": boxes, "frame_times": sorted(frame_times), "device": device, "static": static, "dynamic": dynamic, "bbox_member": bbox_member}


def position_at(source: dict[str, Any], uid: str, timestamp_ns: int):
    if uid in source["static"]:
        return source["static"][uid]
    rows = source["dynamic"].get(uid, [])
    index = nearest(rows, timestamp_ns)
    return rows[index][1] if index is not None else None


def build_candidate(source: dict[str, Any], uid: str, thresholds: Thresholds) -> dict[str, Any] | None:
    times = source["frame_times"]
    target_boxes = source["boxes"][uid]
    visible = [target_boxes.get(t, {}).get("visibility", 0.0) >= thresholds.visible_ratio for t in times]
    segments = contiguous_segments(visible)
    qualifying = [(start, end) for start, end in segments if end - start >= thresholds.min_track_frames]
    if not qualifying:
        return None

    distances: list[float | None] = []
    centers: list[float | None] = []
    for timestamp in times:
        box = target_boxes.get(timestamp)
        centers.append(None if box is None else (box["x_min"] + box["x_max"]) / 2.0)
        object_position = position_at(source, uid, timestamp)
        device_index = nearest(source["device"], timestamp)
        distances.append(None if object_position is None or device_index is None else math.dist(source["device"][device_index][1], object_position))

    phases: list[str] = []
    first_start, _ = qualifying[0]
    phases.extend(["SEARCH", "ACQUIRE"] if first_start >= thresholds.min_hidden_frames else ["ACQUIRE"])
    phases.append("TRACK")
    reacquisition_gap = None
    for (_, left_end), (right_start, _) in zip(qualifying, qualifying[1:]):
        if right_start - left_end >= thresholds.min_lost_frames:
            reacquisition_gap = (left_end, right_start)
            phases.extend(["LOST", "REACQUIRE"])
            break

    best_drop = 0.0
    approach_interval = None
    for start, end in qualifying:
        first, last = median_endpoint(distances, start, end, True), median_endpoint(distances, start, end, False)
        if first is None or last is None:
            continue
        if first - last > best_drop:
            best_drop, approach_interval = first - last, (start, end, first, last)
    if approach_interval and best_drop >= max(thresholds.min_approach_m, thresholds.min_approach_fraction * approach_interval[2]):
        phases.append("APPROACH")

    def stamp(index: int) -> int:
        return times[min(max(index, 0), len(times) - 1)]

    center_values = [value for value in centers if value is not None]
    return {
        "target_uid": uid,
        "target_name": instance_name(source["instances"], uid),
        "frame_count": len(times), "visible_frame_count": sum(visible), "visible_fraction": sum(visible) / len(times),
        "phases": phases, "phase_count": len(set(phases)), "first_acquisition_timestamp_ns": stamp(first_start),
        "track_segments": [{"start_timestamp_ns": stamp(start), "end_timestamp_ns": stamp(end - 1), "frames": end - start} for start, end in qualifying],
        "reacquisition_gap": None if reacquisition_gap is None else {"lost_timestamp_ns": stamp(reacquisition_gap[0]), "reacquired_timestamp_ns": stamp(reacquisition_gap[1]), "missing_frames": reacquisition_gap[1] - reacquisition_gap[0]},
        "approach": None if approach_interval is None else {"start_timestamp_ns": stamp(approach_interval[0]), "end_timestamp_ns": stamp(approach_interval[1] - 1), "start_distance_m": approach_interval[2], "end_distance_m": approach_interval[3], "distance_drop_m": best_drop, "qualifies": "APPROACH" in phases},
        "bbox_center_x_range_px": [min(center_values), max(center_values)],
    }


def mine(path: Path, thresholds: Thresholds) -> dict[str, Any]:
    source = load_archive(path)
    candidates = [candidate for uid in sorted(source["boxes"]) if (candidate := build_candidate(source, uid, thresholds)) is not None]
    candidates.sort(key=lambda row: (-row["phase_count"], -row["visible_frame_count"], row["target_uid"]))
    coverage = {phase: sum(phase in row["phases"] for row in candidates) for phase in ("SEARCH", "ACQUIRE", "TRACK", "LOST", "REACQUIRE", "APPROACH")}
    return {
        "schema_version": "ba_adt_goal_episode_mining_v1", "route": "BA-ADT-REAL-EVIDENCE", "stage": "ADT-0",
        "input": {"groundtruth_archive": path.name, "sha256": sha256(path), "role": "GT_MINING_AND_EVALUATION_ONLY"},
        "rgb_estimator_access_count": 0, "bbox_member": source["bbox_member"], "frame_count": len(source["frame_times"]),
        "target_count_with_bbox": len(source["boxes"]), "candidate_count": len(candidates), "event_coverage": coverage,
        "thresholds": asdict(thresholds), "candidates": candidates,
        "claim_ceiling": "adt_sample_gt_only_episode_suitability_no_rgb_perception_or_navigation_claim",
        "terminal": "ADT0_SAMPLE_EPISODES_MINED" if candidates else "ADT0_SAMPLE_NO_ELIGIBLE_TARGET_EPISODE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = mine(args.groundtruth, Thresholds())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": result["terminal"], "candidate_count": result["candidate_count"], "event_coverage": result["event_coverage"]}, sort_keys=True))
    return 0 if result["candidate_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
