#!/usr/bin/env python3
"""Materialize far-guidance and near-stop frames from future-demonstrated routes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_future_approach_cohort_dev import cluster_events, component_mask_for_bbox, member
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_remote_s5 import decode_depth_bytes
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import exact_door_label, file_hash


STOP_DEPTH_M = 1.50
ROUTE_LOOKAHEAD_FRAMES = 30
IMU_SAMPLES_PER_FRAME = 10


def route_plan(imu_zip: zipfile.ZipFile, environment: str, trajectory: str, frame: int) -> dict:
    base = f"{environment}/Data_easy/{trajectory}/imu/"
    with imu_zip.open(base + "pos_global.npy") as stream:
        positions = np.load(stream)
    with imu_zip.open(base + "ori_global.npy") as stream:
        orientations = np.load(stream)
    current = min(frame * IMU_SAMPLES_PER_FRAME, len(positions) - 1)
    waypoint = min((frame + ROUTE_LOOKAHEAD_FRAMES) * IMU_SAMPLES_PER_FRAME, len(positions) - 1)
    displacement_body = Rotation.from_euler("xyz", orientations[current]).inv().apply(positions[waypoint] - positions[current])
    bearing_fraction = 0.5 + float(displacement_body[1]) / max(abs(float(displacement_body[0])), 0.1) / 2.0
    return {
        "source": "PREDECLARED_REPLAY_WAYPOINT_PROXY",
        "lookahead_frames": ROUTE_LOOKAHEAD_FRAMES,
        "lookahead_seconds": 3.0,
        "bearing_fraction": bearing_fraction,
        "body_displacement_xyz_m": [float(value) for value in displacement_body],
        "derived_without_semantic_or_target_truth": True,
    }


def servo_phases(cluster: list[dict], stop_depth_m: float = STOP_DEPTH_M) -> list[dict]:
    start_frame = min(row["start_frame_id"] for row in cluster)
    starts = [row for row in cluster if row["start_frame_id"] == start_frame]
    phases = [{"phase": "FAR_GUIDANCE", "frame_id": start_frame, "targets": [
        {"bbox_xyxy": row["target_bbox_xyxy"], "depth_m": row["start_depth_m"], "desired_action": row["demonstrated_action"]}
        for row in starts
    ]}]
    near = min(starts, key=lambda row: row["future_min_depth_m"])
    if near["future_min_depth_m"] <= stop_depth_m:
        phases.append({"phase": "NEAR_STOP", "frame_id": near["closest_frame_id"], "targets": [{
            "bbox_xyxy": near["closest_target_bbox_xyxy"], "depth_m": near["future_min_depth_m"], "desired_action": "STOP"
        }]})
    return phases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", type=Path, action="append", required=True)
    parser.add_argument("--zip-root", type=Path, action="append", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"), required=True)
    parser.add_argument("--case-prefix", required=True)
    args = parser.parse_args()
    _require(len(args.diagnostic) == len(args.zip_root), "diagnostic/ZIP source count mismatch")
    _require(not any(path.exists() for path in (args.payload_root, args.public, args.private)), "future servo cohort already exists")
    args.payload_root.mkdir(parents=True)
    public_cases, private_cases, receipts = [], [], []
    case_index = 0
    for diagnostic_path, zip_root in zip(args.diagnostic, args.zip_root, strict=True):
        diagnostic = _read(diagnostic_path)
        environment = diagnostic["environment"]
        data_root = zip_root / environment / "Data_easy"
        image_path, depth_path, seg_path = (data_root / f"{name}_lcam_front.zip" for name in ("image", "depth", "seg"))
        imu_path = data_root / "imu.zip"
        _require(imu_path.is_file(), "future servo route-plan IMU unavailable")
        door_id = exact_door_label(args.label_root / environment / "seg_label_map.json")
        with zipfile.ZipFile(image_path) as image_zip, zipfile.ZipFile(depth_path) as depth_zip, zipfile.ZipFile(seg_path) as seg_zip, zipfile.ZipFile(imu_path) as imu_zip:
            for cluster in cluster_events(diagnostic["episodes"]):
                trajectory = cluster[0]["trajectory"]
                for phase in servo_phases(cluster):
                    case_index += 1
                    case_id = f"{args.case_prefix}-{case_index:03d}"
                    case_root = args.payload_root / case_id
                    case_root.mkdir()
                    frame = phase["frame_id"]
                    image_bytes = image_zip.read(member(environment, trajectory, frame, "image"))
                    depth_bytes = depth_zip.read(member(environment, trajectory, frame, "depth"))
                    seg_bytes = seg_zip.read(member(environment, trajectory, frame, "seg"))
                    current_path, current_depth_path = case_root / "current.png", case_root / "depth.png"
                    current_path.write_bytes(image_bytes)
                    current_depth_path.write_bytes(depth_bytes)
                    segmentation = cv2.imdecode(np.frombuffer(seg_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    _require(segmentation is not None and segmentation.ndim == 2, "invalid servo segmentation")
                    decode_depth_bytes(depth_bytes)
                    legal_targets = []
                    for target_index, target in enumerate(phase["targets"], start=1):
                        mask = component_mask_for_bbox(segmentation, door_id, target["bbox_xyxy"])
                        mask_path = case_root / f"target-{target_index:02d}.png"
                        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                        legal_targets.append({
                            "target_bbox_xyxy": target["bbox_xyxy"],
                            "target_mask_path": str(mask_path.resolve()),
                            "target_mask_sha256": file_hash(mask_path),
                            "target_depth_m": target["depth_m"],
                            "desired_action": target["desired_action"],
                        })
                    public_cases.append({
                        "case_id": case_id,
                        "episode_id": f"{environment}/{trajectory}/{frame:06d}",
                        "goal_text_original": "帮我找一扇沿当前路线可以接近的门",
                        "goal_contract": {"goal_type": "ROUTE_APPROACHABLE_DOOR", "reference_mode": "SET_VALUED", "canonical_prompt": "door"},
                        "query": {"image_path": str(current_path.resolve()), "image_sha256": file_hash(current_path)},
                        "range_sensor": {"depth_path": str(current_depth_path.resolve()), "depth_sha256": file_hash(current_depth_path), "metric_unit": "meter"},
                        "route_plan": route_plan(imu_zip, environment, trajectory, frame),
                    })
                    private_cases.append({"case_id": case_id, "phase": phase["phase"], "future_demonstrated_positive_only": True, "legal_targets": legal_targets})
        receipts.append({"environment": environment, "diagnostic_sha256": file_hash(diagnostic_path), "imu_zip_sha256": file_hash(imu_path), "independent_event_count": len(cluster_events(diagnostic["episodes"]))})
    public = {"schema_version": "blindassist_future_servo_public_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": args.role, "provider_truth_access": False, "cases": public_cases}
    private = {"schema_version": "blindassist_future_servo_private_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "PRIVATE_EVALUATOR_ONLY", "positive_only_truth": True, "stop_depth_m": STOP_DEPTH_M, "source_receipts": receipts, "cases": private_cases}
    _atomic_json(args.public, public)
    _atomic_json(args.private, private)
    print(json.dumps({"case_count": len(public_cases), "far_count": sum(row["phase"] == "FAR_GUIDANCE" for row in private_cases), "stop_count": sum(row["phase"] == "NEAR_STOP" for row in private_cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
