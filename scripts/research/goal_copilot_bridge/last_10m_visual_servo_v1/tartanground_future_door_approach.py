#!/usr/bin/env python3
"""Find future-demonstrated door approaches in one TartanGround trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.future_door_approach_dev import CURRENT_DEPTH_RANGE_M, HORIZON_FRAMES, MIN_PIXELS, START_STRIDE, approach_summary, match_target
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_remote_s5 import decode_depth_bytes
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import door_targets, file_hash


def frame_id(member: str) -> int:
    match = re.search(r"(?:^|/)(\d+)_lcam_front_seg\.png$", member)
    _require(match is not None, "unexpected TartanGround segmentation member")
    return int(match.group(1))


def depth_member(seg_member: str) -> str:
    return seg_member.replace("seg_lcam_front/", "depth_lcam_front/").replace("_seg.png", "_depth.png")


def semantic_id(label_zip: Path, class_name: str) -> int:
    with zipfile.ZipFile(label_zip) as archive:
        payload = json.loads(archive.read("seg_label_map.json"))
    name_map = payload.get("name_map", payload)
    matches = [int(value) for name, value in name_map.items() if name.lower() == class_name.lower()]
    _require(len(matches) == 1, f"TartanGround semantic class unavailable: {class_name}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--label-zip", type=Path, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "TartanGround approach diagnostic already exists")
    seg_path, depth_path = args.trajectory_root / "seg_lcam_front.zip", args.trajectory_root / "depth_lcam_front.zip"
    door_id = semantic_id(args.label_zip, args.class_name)
    eligibility = {"minimum_connected_region_pixels": MIN_PIXELS, "minimum_valid_depth_fraction": 0.5, "valid_depth_range_m": [0.4, 10.0]}
    frames = []
    with zipfile.ZipFile(seg_path) as seg_zip, zipfile.ZipFile(depth_path) as depth_zip:
        members = sorted((member for member in seg_zip.namelist() if member.endswith("_lcam_front_seg.png")), key=frame_id)
        for index, member in enumerate(members, start=1):
            segmentation = cv2.imdecode(np.frombuffer(seg_zip.read(member), np.uint8), cv2.IMREAD_UNCHANGED)
            _require(segmentation is not None and segmentation.ndim == 2, "invalid TartanGround segmentation")
            depth = decode_depth_bytes(depth_zip.read(depth_member(member)))
            targets = door_targets(segmentation, depth, door_id, eligibility)
            rows = [{key: value for key, value in target.items() if key != "mask"} | {"image_width": int(segmentation.shape[1]), "image_height": int(segmentation.shape[0])} for target in targets]
            frames.append({"frame_id": frame_id(member), "targets": rows})
            if index % 1000 == 0:
                print(f"tartanground-approach decoded {index}/{len(members)}", flush=True)
    by_id = {frame["frame_id"]: frame for frame in frames}
    episodes = []
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
                    "episode_id": f"{args.environment}/{args.trajectory}/{start['frame_id']:06d}/{args.class_name}-{target_index}",
                    "environment": args.environment,
                    "trajectory": args.trajectory,
                    "start_frame_id": start["frame_id"],
                    "target_bbox_xyxy": target["bbox_xyxy"],
                    "target_pixel_count": target["pixel_count"],
                    **summary,
                })
    payload = {
        "schema_version": "blindassist_tartanground_future_door_approach_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "environment": args.environment,
        "trajectory": args.trajectory,
        "semantic_class": args.class_name,
        "source": {"seg_zip_sha256": file_hash(seg_path), "depth_zip_sha256": file_hash(depth_path), "label_zip_sha256": file_hash(args.label_zip)},
        "trajectory_frame_count": len(frames),
        "approach_episode_count": len(episodes),
        "episodes": episodes,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"trajectory_frame_count": len(frames), "approach_episode_count": len(episodes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
