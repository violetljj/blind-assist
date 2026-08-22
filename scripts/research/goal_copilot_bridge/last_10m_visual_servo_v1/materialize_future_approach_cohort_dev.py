#!/usr/bin/env python3
"""Materialize independent future-demonstrated approach events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.future_door_approach_dev import HORIZON_FRAMES
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_remote_s5 import decode_depth_bytes
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import exact_door_label, file_hash
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou


def cluster_events(episodes: list[dict]) -> list[list[dict]]:
    clusters = []
    for trajectory in sorted({episode["trajectory"] for episode in episodes}):
        rows = sorted((episode for episode in episodes if episode["trajectory"] == trajectory), key=lambda row: row["start_frame_id"])
        current = []
        for row in rows:
            if current and row["start_frame_id"] - current[-1]["start_frame_id"] > HORIZON_FRAMES:
                clusters.append(current)
                current = []
            current.append(row)
        if current:
            clusters.append(current)
    return clusters


def member(environment: str, trajectory: str, frame: int, modality: str) -> str:
    suffix = {"image": f"{frame:06d}_lcam_front.png", "depth": f"{frame:06d}_lcam_front_depth.png", "seg": f"{frame:06d}_lcam_front_seg.png"}[modality]
    return f"{environment}/Data_easy/{trajectory}/{modality}_lcam_front/{suffix}"


def component_mask_for_bbox(segmentation: np.ndarray, door_id: int, bbox: list[float]) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((segmentation == door_id).astype(np.uint8), 8)
    ranked = []
    for component in range(1, count):
        x, y, width, height = [int(value) for value in stats[component, :4]]
        candidate = [x, y, x + width, y + height]
        ranked.append((iou(candidate, bbox), component))
    _require(bool(ranked) and max(ranked)[0] >= 0.99, "future target component mismatch")
    return labels == max(ranked)[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", type=Path, action="append", required=True)
    parser.add_argument("--zip-root", type=Path, action="append", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"), default="DEVELOPMENT_ONLY")
    parser.add_argument("--case-prefix", default="future-approach-dev-case")
    args = parser.parse_args()
    _require(len(args.diagnostic) == len(args.zip_root), "diagnostic/ZIP source count mismatch")
    _require(not any(path.exists() for path in (args.payload_root, args.public, args.private)), "future approach cohort already exists")
    args.payload_root.mkdir(parents=True)
    public_cases, private_cases, source_receipts = [], [], []
    event_index = 0
    for diagnostic_path, zip_root in zip(args.diagnostic, args.zip_root, strict=True):
        diagnostic = _read(diagnostic_path)
        environment = diagnostic["environment"]
        clusters = cluster_events(diagnostic["episodes"])
        data_root = zip_root / environment / "Data_easy"
        image_zip_path, depth_zip_path, seg_zip_path = (data_root / f"{name}_lcam_front.zip" for name in ("image", "depth", "seg"))
        door_id = exact_door_label(args.label_root / environment / "seg_label_map.json")
        with zipfile.ZipFile(image_zip_path) as image_zip, zipfile.ZipFile(depth_zip_path) as depth_zip, zipfile.ZipFile(seg_zip_path) as seg_zip:
            for cluster in clusters:
                start_frame = min(row["start_frame_id"] for row in cluster)
                starts = [row for row in cluster if row["start_frame_id"] == start_frame]
                trajectory = starts[0]["trajectory"]
                event_index += 1
                case_id = f"{args.case_prefix}-{event_index:03d}"
                case_root = args.payload_root / case_id
                case_root.mkdir()
                image_bytes = image_zip.read(member(environment, trajectory, start_frame, "image"))
                depth_bytes = depth_zip.read(member(environment, trajectory, start_frame, "depth"))
                seg_bytes = seg_zip.read(member(environment, trajectory, start_frame, "seg"))
                image_path, depth_path = case_root / "current.png", case_root / "depth.png"
                image_path.write_bytes(image_bytes)
                depth_path.write_bytes(depth_bytes)
                segmentation = cv2.imdecode(np.frombuffer(seg_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                _require(segmentation is not None and segmentation.ndim == 2, "invalid future approach segmentation")
                depth = decode_depth_bytes(depth_bytes)
                legal_targets = []
                for target_index, start in enumerate(starts, start=1):
                    mask = component_mask_for_bbox(segmentation, door_id, start["target_bbox_xyxy"])
                    mask_path = case_root / f"target-{target_index:02d}.png"
                    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                    legal_targets.append({
                        "target_bbox_xyxy": start["target_bbox_xyxy"],
                        "target_mask_path": str(mask_path.resolve()),
                        "target_mask_sha256": file_hash(mask_path),
                        "start_depth_m": start["start_depth_m"],
                        "future_min_depth_m": start["future_min_depth_m"],
                        "tracked_steps": start["tracked_steps"],
                        "demonstrated_action": start["demonstrated_action"],
                    })
                public_cases.append({
                    "case_id": case_id,
                    "episode_id": f"{environment}/{trajectory}/{start_frame:06d}",
                    "goal_text_original": "帮我找一扇沿当前路线可以接近的门",
                    "goal_contract": {"goal_type": "ROUTE_APPROACHABLE_DOOR", "reference_mode": "SET_VALUED", "canonical_prompt": "door"},
                    "query": {"image_path": str(image_path.resolve()), "image_sha256": file_hash(image_path)},
                    "range_sensor": {"depth_path": str(depth_path.resolve()), "depth_sha256": file_hash(depth_path), "metric_unit": "meter"},
                })
                private_cases.append({
                    "case_id": case_id,
                    "future_demonstrated_positive_only": True,
                    "unobserved_alternatives_are_not_negatives": True,
                    "legal_targets": legal_targets,
                })
        source_receipts.append({"environment": environment, "diagnostic_sha256": file_hash(diagnostic_path), "image_zip_sha256": file_hash(image_zip_path), "depth_zip_sha256": file_hash(depth_zip_path), "seg_zip_sha256": file_hash(seg_zip_path), "independent_event_count": len(clusters)})
    public = {"schema_version": "blindassist_future_approach_public_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": args.role, "provider_truth_access": False, "cases": public_cases}
    private = {"schema_version": "blindassist_future_approach_private_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "PRIVATE_EVALUATOR_ONLY", "positive_only_truth": True, "source_receipts": source_receipts, "public_body_sha256": hashlib.sha256((json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(), "cases": private_cases}
    _atomic_json(args.public, public)
    _atomic_json(args.private, private)
    print(json.dumps({"case_count": len(public_cases), "sources": source_receipts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
