#!/usr/bin/env python3
"""Find future-demonstrated door approaches in public TartanAir sequences."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_remote_s5 import decode_depth_bytes, modality_member
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import door_targets, exact_door_label, file_hash
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou


MIN_PIXELS = 200
CURRENT_DEPTH_RANGE_M = (2.5, 8.0)
FUTURE_NEAR_DEPTH_M = 2.0
MIN_DEPTH_REDUCTION_M = 0.75
MAX_DEPTH_RATIO = 0.70
MIN_AREA_GROWTH = 1.30
HORIZON_FRAMES = 30
MIN_TRACKED_STEPS = 8
START_STRIDE = 5


def target_center(target: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = target["bbox_xyxy"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def match_target(previous: dict, candidates: list[dict], width: int, height: int) -> dict | None:
    px, py = target_center(previous)
    diagonal = math.hypot(width, height)
    ranked = []
    for candidate in candidates:
        cx, cy = target_center(candidate)
        center_distance = math.hypot(cx - px, cy - py) / diagonal
        overlap = iou(previous["bbox_xyxy"], candidate["bbox_xyxy"])
        depth_ratio = candidate["depth_median_m"] / max(previous["depth_median_m"], 1e-6)
        area_ratio = candidate["pixel_count"] / max(previous["pixel_count"], 1)
        if not (0.50 <= depth_ratio <= 2.0 and 0.35 <= area_ratio <= 3.0):
            continue
        if overlap < 0.05 and center_distance > 0.15:
            continue
        score = overlap - center_distance - 0.15 * abs(math.log(depth_ratio)) - 0.10 * abs(math.log(area_ratio))
        ranked.append((score, candidate))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def approach_summary(track: list[dict]) -> dict:
    start = track[0]
    closest = min(track, key=lambda item: item["depth_median_m"])
    min_depth = closest["depth_median_m"]
    max_area = max(item["pixel_count"] for item in track)
    depth_reduction = start["depth_median_m"] - min_depth
    depth_ratio = min_depth / start["depth_median_m"]
    area_growth = max_area / start["pixel_count"]
    approached = (
        len(track) >= MIN_TRACKED_STEPS
        and min_depth <= FUTURE_NEAR_DEPTH_M
        and depth_reduction >= MIN_DEPTH_REDUCTION_M
        and depth_ratio <= MAX_DEPTH_RATIO
        and area_growth >= MIN_AREA_GROWTH
    )
    center_x, _ = target_center(start)
    center_fraction = center_x / float(start["image_width"])
    action = "TURN_LEFT" if center_fraction < 0.42 else ("TURN_RIGHT" if center_fraction > 0.58 else "ADVANCE")
    return {
        "approached": approached,
        "tracked_steps": len(track),
        "start_depth_m": start["depth_median_m"],
        "future_min_depth_m": min_depth,
        "closest_frame_id": closest.get("frame_id"),
        "closest_target_bbox_xyxy": closest["bbox_xyxy"],
        "closest_target_pixel_count": closest["pixel_count"],
        "depth_reduction_m": depth_reduction,
        "depth_ratio": depth_ratio,
        "area_growth": area_growth,
        "target_center_x_fraction": center_fraction,
        "demonstrated_action": action,
    }


def frame_id(member: str) -> int:
    match = re.search(r"/(\d+)_lcam_front_seg\.png$", member)
    _require(match is not None, "unexpected TartanAir segmentation member")
    return int(match.group(1))


def trajectory_name(member: str) -> str:
    match = re.search(r"/Data_easy/(P\d+)/seg_lcam_front/", member)
    _require(match is not None, "unexpected TartanAir trajectory member")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--zip-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "future approach diagnostic already exists")
    data_root = args.zip_root / args.environment / "Data_easy"
    seg_path, depth_path = data_root / "seg_lcam_front.zip", data_root / "depth_lcam_front.zip"
    door_id = exact_door_label(args.label_root / args.environment / "seg_label_map.json")
    eligibility = {"minimum_connected_region_pixels": MIN_PIXELS, "minimum_valid_depth_fraction": 0.5, "valid_depth_range_m": [0.4, 10.0]}
    trajectories: dict[str, list[dict]] = {}
    with zipfile.ZipFile(seg_path) as seg_zip, zipfile.ZipFile(depth_path) as depth_zip:
        members = sorted((member for member in seg_zip.namelist() if member.endswith("_lcam_front_seg.png")), key=lambda member: (trajectory_name(member), frame_id(member)))
        for index, member in enumerate(members, start=1):
            segmentation = cv2.imdecode(np.frombuffer(seg_zip.read(member), np.uint8), cv2.IMREAD_UNCHANGED)
            _require(segmentation is not None and segmentation.ndim == 2, "invalid TartanAir segmentation")
            depth = decode_depth_bytes(depth_zip.read(modality_member(member, "depth")))
            targets = door_targets(segmentation, depth, door_id, eligibility)
            rows = [{key: value for key, value in target.items() if key != "mask"} | {"image_width": int(segmentation.shape[1]), "image_height": int(segmentation.shape[0])} for target in targets]
            trajectories.setdefault(trajectory_name(member), []).append({"frame_id": frame_id(member), "member": member, "targets": rows})
            if index % 1000 == 0:
                print(f"future-approach decoded {index}/{len(members)}", flush=True)

    episodes = []
    for trajectory, frames in sorted(trajectories.items()):
        by_id = {frame["frame_id"]: frame for frame in frames}
        for start in frames[::START_STRIDE]:
            for target_index, target in enumerate(start["targets"]):
                if not (CURRENT_DEPTH_RANGE_M[0] <= target["depth_median_m"] <= CURRENT_DEPTH_RANGE_M[1]):
                    continue
                current = target | {"frame_id": start["frame_id"]}
                track = [current]
                for future_id in range(start["frame_id"] + 1, start["frame_id"] + HORIZON_FRAMES + 1):
                    future = by_id.get(future_id)
                    if future is None:
                        break
                    matched = match_target(current, future["targets"], target["image_width"], target["image_height"])
                    if matched is None:
                        break
                    current = matched | {"frame_id": future_id}
                    track.append(current)
                summary = approach_summary(track)
                if summary["approached"]:
                    episodes.append({
                        "episode_id": f"{args.environment}/{trajectory}/{start['frame_id']:06d}/door-{target_index}",
                        "environment": args.environment,
                        "trajectory": trajectory,
                        "start_frame_id": start["frame_id"],
                        "target_bbox_xyxy": target["bbox_xyxy"],
                        "target_pixel_count": target["pixel_count"],
                        **summary,
                    })
    action_counts = {action: sum(row["demonstrated_action"] == action for row in episodes) for action in ("TURN_LEFT", "ADVANCE", "TURN_RIGHT")}
    payload = {
        "schema_version": "blindassist_tartanair_future_door_approach_development_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "environment": args.environment,
        "source": {"seg_zip_sha256": file_hash(seg_path), "depth_zip_sha256": file_hash(depth_path)},
        "contract": {"current_depth_range_m": list(CURRENT_DEPTH_RANGE_M), "future_near_depth_m": FUTURE_NEAR_DEPTH_M, "minimum_depth_reduction_m": MIN_DEPTH_REDUCTION_M, "maximum_depth_ratio": MAX_DEPTH_RATIO, "minimum_area_growth": MIN_AREA_GROWTH, "horizon_frames": HORIZON_FRAMES, "minimum_tracked_steps": MIN_TRACKED_STEPS, "start_stride": START_STRIDE},
        "trajectory_count": len(trajectories),
        "approach_episode_count": len(episodes),
        "action_counts": action_counts,
        "episodes": episodes,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("trajectory_count", "approach_episode_count", "action_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
