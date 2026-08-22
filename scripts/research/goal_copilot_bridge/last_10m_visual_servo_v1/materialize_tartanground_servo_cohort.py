#!/usr/bin/env python3
"""Materialize a public/private RGB-D servo cohort from TartanGround."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_future_approach_cohort_dev import cluster_events, component_mask_for_bbox
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_future_servo_cohort import ROUTE_LOOKAHEAD_FRAMES, STOP_DEPTH_M, servo_phases
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_remote_s5 import decode_depth_bytes
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import file_hash
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.tartanground_future_door_approach import semantic_id


GOAL_TEXT_ORIGINAL = "帮我找一扇沿当前路线可以接近的门"
GOAL_CONTRACT = {
    "goal_type": "ROUTE_APPROACHABLE_DOOR",
    "reference_mode": "SET_VALUED",
    "canonical_prompt": "door",
    "mapping_scope": "GLOBAL_FROZEN",
}
# This predecessor committed the same goal text and canonical prompt before any
# TartanGround target truth was inspected.
PRETRUTH_CONTRACT_COMMIT = "58643839"


def tg_member(frame: int, modality: str) -> str:
    suffix = {
        "image": f"{frame:06d}_lcam_front.png",
        "depth": f"{frame:06d}_lcam_front_depth.png",
        "seg": f"{frame:06d}_lcam_front_seg.png",
    }[modality]
    return f"{modality}_lcam_front/{suffix}"


def read_poses(metadata_zip: zipfile.ZipFile) -> np.ndarray:
    values = np.loadtxt(io.BytesIO(metadata_zip.read("pose_lcam_front.txt")))
    _require(values.ndim == 2 and values.shape[1] == 7, "invalid TartanGround camera pose")
    return values


def route_plan(poses: np.ndarray, frame: int) -> dict:
    current = min(frame, len(poses) - 1)
    waypoint = min(frame + ROUTE_LOOKAHEAD_FRAMES, len(poses) - 1)
    displacement_world = poses[waypoint, :3] - poses[current, :3]
    displacement_body = Rotation.from_quat(poses[current, 3:7]).inv().apply(displacement_world)
    yaw = math.atan2(float(displacement_body[1]), max(float(displacement_body[0]), 1e-6))
    bearing_fraction = min(1.0, max(0.0, 0.5 + yaw / (math.pi / 2.0)))
    return {
        "source": "PREDECLARED_DIFF_DRIVE_REPLAY_WAYPOINT",
        "lookahead_frames": ROUTE_LOOKAHEAD_FRAMES,
        "lookahead_seconds": 3.0,
        "bearing_fraction": bearing_fraction,
        "body_displacement_xyz_m": [float(value) for value in displacement_body],
        "derived_without_semantic_or_target_truth": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-root", type=Path, action="append", required=True)
    parser.add_argument("--label-zip", type=Path, action="append", required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"), required=True)
    parser.add_argument("--case-prefix", required=True)
    args = parser.parse_args()
    _require(len(args.diagnostic) == len(args.trajectory_root) == len(args.label_zip), "diagnostic/trajectory/label source count mismatch")
    _require(not any(path.exists() for path in (args.payload_root, args.public, args.private)), "TartanGround servo cohort already exists")
    args.payload_root.mkdir(parents=True)
    public_cases, private_cases, receipts = [], [], []
    case_index = 0
    for diagnostic_path, trajectory_root, label_zip in zip(args.diagnostic, args.trajectory_root, args.label_zip, strict=True):
        diagnostic = _read(diagnostic_path)
        environment, trajectory = diagnostic["environment"], diagnostic["trajectory"]
        _require(trajectory_root.name == trajectory, "TartanGround trajectory root mismatch")
        door_id = semantic_id(label_zip, "door")
        paths = {name: trajectory_root / f"{name}_lcam_front.zip" for name in ("image", "depth", "seg")}
        metadata_path = trajectory_root / "metadata.zip"
        with zipfile.ZipFile(paths["image"]) as image_zip, zipfile.ZipFile(paths["depth"]) as depth_zip, zipfile.ZipFile(paths["seg"]) as seg_zip, zipfile.ZipFile(metadata_path) as metadata_zip:
            poses = read_poses(metadata_zip)
            clusters = cluster_events(diagnostic["episodes"])
            for cluster in clusters:
                for phase in servo_phases(cluster):
                    case_index += 1
                    case_id = f"{args.case_prefix}-{case_index:03d}"
                    case_root = args.payload_root / case_id
                    case_root.mkdir()
                    frame = phase["frame_id"]
                    image_bytes = image_zip.read(tg_member(frame, "image"))
                    depth_bytes = depth_zip.read(tg_member(frame, "depth"))
                    seg_bytes = seg_zip.read(tg_member(frame, "seg"))
                    current_path, depth_path = case_root / "current.png", case_root / "depth.png"
                    current_path.write_bytes(image_bytes)
                    depth_path.write_bytes(depth_bytes)
                    segmentation = cv2.imdecode(np.frombuffer(seg_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    _require(segmentation is not None and segmentation.ndim == 2, "invalid TartanGround segmentation")
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
                        "goal_text_original": GOAL_TEXT_ORIGINAL,
                        "goal_contract": GOAL_CONTRACT,
                        "goal_contract_provenance": {"created_before_target_truth": True, "pretruth_commit": PRETRUTH_CONTRACT_COMMIT},
                        "query": {"image_path": str(current_path.resolve()), "image_sha256": file_hash(current_path)},
                        "range_sensor": {"depth_path": str(depth_path.resolve()), "depth_sha256": file_hash(depth_path), "metric_unit": "meter"},
                        "route_plan": route_plan(poses, frame),
                    })
                    private_cases.append({"case_id": case_id, "phase": phase["phase"], "future_demonstrated_positive_only": True, "legal_targets": legal_targets})
        receipts.append({
            "environment": environment,
            "trajectory": trajectory,
            "diagnostic_sha256": file_hash(diagnostic_path),
            "metadata_zip_sha256": file_hash(metadata_path),
            "label_zip_sha256": file_hash(label_zip),
            "independent_event_count": len(cluster_events(diagnostic["episodes"])),
        })
    created = datetime.now(timezone.utc).isoformat()
    public = {"schema_version": "blindassist_tartanground_servo_public_v1", "created_at_utc": created, "role": args.role, "provider_truth_access": False, "cases": public_cases}
    private = {"schema_version": "blindassist_tartanground_servo_private_v1", "created_at_utc": created, "role": "PRIVATE_EVALUATOR_ONLY", "positive_only_truth": True, "stop_depth_m": STOP_DEPTH_M, "source_receipts": receipts, "cases": private_cases}
    _atomic_json(args.public, public)
    _atomic_json(args.private, private)
    print(json.dumps({"case_count": len(public_cases), "far_count": sum(row["phase"] == "FAR_GUIDANCE" for row in private_cases), "stop_count": sum(row["phase"] == "NEAR_STOP" for row in private_cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
