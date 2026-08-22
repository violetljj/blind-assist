#!/usr/bin/env python3
"""Qualify a TartanAir environment's frozen denominator before RGB access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_remote_s5 import decode_depth_bytes, modality_member
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import door_targets, exact_door_label, file_hash


def qualify(manifest_path: Path, label_root: Path, zip_root: Path, output_path: Path) -> dict:
    _require(not output_path.exists(), "functional environment qualification already exists")
    manifest = _read(manifest_path)
    _require(manifest.get("created_before_selected_rgb_depth_segmentation_access") is True, "functional qualification manifest drift")
    environment = manifest["source"]["environments"][0]
    door_id = exact_door_label(label_root / environment / "seg_label_map.json")
    data_root = zip_root / environment / "Data_easy"
    seg_path, depth_path = data_root / "seg_lcam_front.zip", data_root / "depth_lcam_front.zip"
    _require(seg_path.is_file() and depth_path.is_file(), "functional qualification ZIP missing")
    counts = {"eligible": 0, "near": 0, "far": 0}
    with zipfile.ZipFile(seg_path) as seg_zip, zipfile.ZipFile(depth_path) as depth_zip:
        members = sorted(member for member in seg_zip.namelist() if member.endswith("_lcam_front_seg.png"))
        for seg_member in members:
            segmentation = cv2.imdecode(np.frombuffer(seg_zip.read(seg_member), np.uint8), cv2.IMREAD_UNCHANGED)
            _require(segmentation is not None and segmentation.ndim == 2, "invalid qualification segmentation")
            if int((segmentation == door_id).sum()) < int(manifest["eligibility"]["minimum_connected_region_pixels"]):
                continue
            depth = decode_depth_bytes(depth_zip.read(modality_member(seg_member, "depth")))
            targets = door_targets(segmentation, depth, door_id, manifest["eligibility"])
            if not targets:
                continue
            near = any(target["bbox_xyxy"][0] <= segmentation.shape[1] / 2.0 <= target["bbox_xyxy"][2] and target["depth_median_m"] <= float(manifest["eligibility"]["near_threshold_m"]) for target in targets)
            counts["eligible"] += 1
            counts["near" if near else "far"] += 1
    required_near, required_far = int(manifest["eligibility"]["near_take"]), int(manifest["eligibility"]["far_take"])
    qualified = counts["near"] >= required_near and counts["far"] >= required_far
    payload = {"schema_version": "blindassist_tartanair_functional_environment_qualification_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "PRE_PROVIDER_DENOMINATOR_QUALIFICATION", "manifest_sha256": file_hash(manifest_path), "environment": environment, "seg_zip_sha256": file_hash(seg_path), "depth_zip_sha256": file_hash(depth_path), "rgb_payload_access": False, "provider_calls": 0, "counts": counts, "required": {"near": required_near, "far": required_far}, "qualified": qualified, "terminal": "FUNCTIONAL_ENVIRONMENT_DENOMINATOR_QUALIFIED" if qualified else "FUNCTIONAL_ENVIRONMENT_DENOMINATOR_NOT_QUALIFIED"}
    _atomic_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--zip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.manifest, args.label_root, args.zip_root, args.output)
    print(json.dumps({"environment": result["environment"], **result["counts"], "qualified": result["qualified"], "terminal": result["terminal"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
