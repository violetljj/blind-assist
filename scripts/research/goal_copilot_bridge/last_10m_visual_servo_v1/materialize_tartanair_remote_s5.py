#!/usr/bin/env python3
"""Materialize a frozen S5 cohort directly from official remote TartanAir ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_nyuv2_door_depth import PRIVATE_SCHEMA, PUBLIC_SCHEMA
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth, door_targets, exact_door_label, file_hash


PROTOCOL_ID = "BLINDASSIST_TARTANAIR_CURRENT_FRAME_COMPLETION_S5_V1"
ROSTER_SCHEMA = "blindassist_tartanair_remote_s5_roster_v1"


def modality_member(seg_member: str, modality: str) -> str:
    if modality == "image":
        return seg_member.replace("/seg_lcam_front/", "/image_lcam_front/").replace("_lcam_front_seg.png", "_lcam_front.png")
    if modality == "depth":
        return seg_member.replace("/seg_lcam_front/", "/depth_lcam_front/").replace("_lcam_front_seg.png", "_lcam_front_depth.png")
    raise ValueError(f"unsupported modality: {modality}")


def decode_depth_bytes(raw: bytes) -> np.ndarray:
    encoded = np.frombuffer(raw, np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    _require(decoded is not None and decoded.dtype == np.uint8 and decoded.ndim == 3 and decoded.shape[2] == 4, "invalid remote TartanAir depth")
    return np.squeeze(np.ascontiguousarray(decoded).view("<f4"), axis=-1)


def json_body_hash(value: dict) -> str:
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--private-mask-root", type=Path, required=True)
    parser.add_argument("--zip-root", type=Path)
    args = parser.parse_args()
    _require(not any(path.exists() for path in (args.payload_root, args.roster, args.public, args.private, args.private_mask_root)), "S5 materialization output already exists")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _require(manifest.get("protocol_id") == PROTOCOL_ID and manifest.get("created_before_selected_rgb_depth_segmentation_access") is True, "S5 manifest drift")
    environment = manifest["source"]["environments"][0]
    revision = manifest["source"]["revision"]
    door_id = exact_door_label(args.label_root / environment / "seg_label_map.json")
    base = f"datasets/{manifest['source']['repository']}@{revision}/{environment}/Data_easy"

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    def open_source(filename: str):
        if args.zip_root is not None:
            path = args.zip_root / environment / "Data_easy" / filename
            _require(path.is_file(), f"local S5 ZIP missing: {path}")
            return path.open("rb")
        return fs.open(f"{base}/{filename}", "rb")

    eligible = []
    with open_source("seg_lcam_front.zip") as seg_file, open_source("depth_lcam_front.zip") as depth_file:
        with zipfile.ZipFile(seg_file) as seg_zip, zipfile.ZipFile(depth_file) as depth_zip:
            members = sorted(member for member in seg_zip.namelist() if member.endswith("_lcam_front_seg.png"))
            for index, seg_member in enumerate(members, start=1):
                seg = cv2.imdecode(np.frombuffer(seg_zip.read(seg_member), np.uint8), cv2.IMREAD_UNCHANGED)
                _require(seg is not None and seg.ndim == 2, "invalid remote S5 segmentation")
                if int((seg == door_id).sum()) < int(manifest["eligibility"]["minimum_connected_region_pixels"]):
                    continue
                depth_member = modality_member(seg_member, "depth")
                depth = decode_depth_bytes(depth_zip.read(depth_member))
                targets = door_targets(seg, depth, door_id, manifest["eligibility"])
                if not targets:
                    continue
                parts = Path(seg_member).parts
                trajectory = next(part for part in parts if part.startswith("P") and part[1:].isdigit())
                frame_id = Path(seg_member).name.split("_")[0]
                source_key = f"{environment}/{trajectory}/{frame_id}"
                near = any(target["bbox_xyxy"][0] <= seg.shape[1] / 2.0 <= target["bbox_xyxy"][2] and target["depth_median_m"] <= float(manifest["eligibility"]["near_threshold_m"]) for target in targets)
                eligible.append({"source_key": source_key, "source_key_sha256": hashlib.sha256(source_key.encode()).hexdigest(), "seg_member": seg_member, "image_member": modality_member(seg_member, "image"), "depth_member": depth_member, "stratum": "NEAR" if near else "FAR", "targets": [{key: value for key, value in target.items() if key != "mask"} for target in targets]})
                if index % 2000 == 0:
                    print(f"scanned {index}/{len(members)} eligible={len(eligible)}", flush=True)
    near = sorted((row for row in eligible if row["stratum"] == "NEAR"), key=lambda row: row["source_key_sha256"])
    far = sorted((row for row in eligible if row["stratum"] == "FAR"), key=lambda row: row["source_key_sha256"])
    near_take, far_take = int(manifest["eligibility"]["near_take"]), int(manifest["eligibility"]["far_take"])
    _require(len(near) >= near_take and len(far) >= far_take, f"S5 denominator insufficient: near={len(near)} far={len(far)}")
    selected = near[:near_take] + far[:far_take]
    roster = {"schema_version": ROSTER_SCHEMA, "protocol_id": PROTOCOL_ID, "manifest_sha256": file_hash(args.manifest), "repository_revision": revision, "eligible_case_count": len(eligible), "eligible_near_count": len(near), "eligible_far_count": len(far), "selection_observed_before_provider_calls": True, "provider_truth_access": False, "cases": [{"case_id": f"tartanair-s5-case-{index:03d}", **row} for index, row in enumerate(selected, start=1)]}
    roster["roster_body_sha256"] = json_body_hash(roster)
    _atomic_json(args.roster, roster)

    args.payload_root.mkdir(parents=True)
    args.private_mask_root.mkdir(parents=True)
    public_cases, private_cases = [], []
    with open_source("seg_lcam_front.zip") as seg_file, open_source("depth_lcam_front.zip") as depth_file, open_source("image_lcam_front.zip") as image_file:
        with zipfile.ZipFile(seg_file) as seg_zip, zipfile.ZipFile(depth_file) as depth_zip, zipfile.ZipFile(image_file) as image_zip:
            for row in roster["cases"]:
                case_root = args.payload_root / row["case_id"]
                case_root.mkdir()
                image_path, depth_path, seg_path = case_root / "rgb.png", case_root / "depth.png", case_root / "seg.png"
                image_path.write_bytes(image_zip.read(row["image_member"]))
                depth_path.write_bytes(depth_zip.read(row["depth_member"]))
                seg_path.write_bytes(seg_zip.read(row["seg_member"]))
                seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
                targets = door_targets(seg, decode_depth(depth_path), door_id, manifest["eligibility"])
                legal = []
                for target_index, target in enumerate(targets, start=1):
                    mask_path = args.private_mask_root / f"{row['case_id']}-target-{target_index:02d}.png"
                    Image.fromarray((target["mask"] * 255).astype(np.uint8)).save(mask_path)
                    legal.append({"target_bbox_xyxy": target["bbox_xyxy"], "target_mask_path": str(mask_path.resolve()), "target_mask_sha256": file_hash(mask_path), "target_pixel_count": target["pixel_count"], "target_depth_median_m": target["depth_median_m"], "target_depth_p10_m": target["depth_p10_m"], "target_depth_p90_m": target["depth_p90_m"], "depth_valid_fraction": target["depth_valid_fraction"]})
                public_cases.append({"case_id": row["case_id"], "episode_id": row["source_key"], "goal_contract": manifest["public_goal_contract"], "query": {"image_path": str(image_path.resolve()), "image_sha256": file_hash(image_path)}, "range_sensor": {"depth_path": str(depth_path.resolve()), "depth_sha256": file_hash(depth_path)}})
                private_cases.append({"case_id": row["case_id"], "source_key": row["source_key"], "stratum": row["stratum"], "legal_targets": legal, "true_interaction_range": any(target["target_depth_median_m"] <= 2.0 for target in legal)})
    public = {"schema_version": PUBLIC_SCHEMA, "protocol_id": PROTOCOL_ID, "manifest_sha256": file_hash(args.manifest), "roster_body_sha256": roster["roster_body_sha256"], "private_truth_access": False, "cases": public_cases}
    _atomic_json(args.public, public)
    private = {"schema_version": PRIVATE_SCHEMA, "protocol_id": PROTOCOL_ID, "public_input_sha256": file_hash(args.public), "interaction_range_m": 2.0, "cases": private_cases}
    _atomic_json(args.private, private)
    print(json.dumps({"eligible": len(eligible), "near": len(near), "far": len(far), "selected": len(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
