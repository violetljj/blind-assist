#!/usr/bin/env python3
"""Materialize the frozen fresh TartanAir S2 current-frame door cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_nyuv2_door_depth import PRIVATE_SCHEMA, PUBLIC_SCHEMA


PROTOCOL_ID = "BLINDASSIST_TARTANAIR_CURRENT_FRAME_COMPLETION_S2_V1"
SUPPORTED_PROTOCOL_IDS = {
    PROTOCOL_ID,
    "BLINDASSIST_TARTANAIR_CURRENT_FRAME_COMPLETION_S3_V1",
    "BLINDASSIST_TARTANAIR_CURRENT_FRAME_COMPLETION_S4_V1",
}
ROSTER_SCHEMA = "blindassist_tartanair_current_frame_completion_roster_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def decode_depth(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    _require(raw is not None and raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 4, f"invalid TartanAir depth: {path}")
    return np.squeeze(np.ascontiguousarray(raw).view("<f4"), axis=-1)


def exact_door_label(label_map_path: Path) -> int:
    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    name_map = payload.get("name_map", {})
    _require(isinstance(name_map, dict) and "door" in name_map, "exact door class absent")
    _require(not any(str(name).lower() == "door" for name in name_map if name != "door"), "door taxonomy ambiguity")
    return int(name_map["door"])


def door_targets(seg: np.ndarray, depth: np.ndarray, door_label: int, eligibility: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(seg.ndim == 2 and seg.shape == depth.shape, "TartanAir segmentation/depth shape mismatch")
    count, labels, stats, _ = cv2.connectedComponentsWithStats((seg == door_label).astype(np.uint8), 8)
    targets = []
    for component in range(1, count):
        pixels = int(stats[component, cv2.CC_STAT_AREA])
        if pixels < int(eligibility["minimum_connected_region_pixels"]):
            continue
        mask = labels == component
        lower, upper = map(float, eligibility["valid_depth_range_m"])
        valid = mask & np.isfinite(depth) & (depth >= lower) & (depth <= upper)
        fraction = float(valid.sum() / pixels)
        if fraction < float(eligibility["minimum_valid_depth_fraction"]):
            continue
        values = depth[valid]
        x, y, width, height = [int(value) for value in stats[component, :4]]
        targets.append({
            "component_id": component,
            "bbox_xyxy": [x, y, x + width, y + height],
            "pixel_count": pixels,
            "depth_valid_fraction": fraction,
            "depth_median_m": float(np.median(values)),
            "depth_p10_m": float(np.quantile(values, 0.10)),
            "depth_p90_m": float(np.quantile(values, 0.90)),
            "mask": mask,
        })
    return targets


def freeze_roster(manifest_path: Path, dataset_root: Path, label_root: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "TartanAir roster already exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_id = manifest.get("protocol_id")
    _require(protocol_id in SUPPORTED_PROTOCOL_IDS and manifest.get("created_before_selected_rgb_depth_segmentation_access") is True, "TartanAir manifest drift")
    eligibility = manifest["eligibility"]
    eligible = []
    for environment in manifest["source"]["environments"]:
        door_label = exact_door_label(label_root / environment / "seg_label_map.json")
        environment_root = dataset_root / environment / "Data_easy"
        for trajectory in sorted(environment_root.glob("P*")):
            for seg_path in sorted((trajectory / "seg_lcam_front").glob("*_lcam_front_seg.png")):
                frame_id = seg_path.name.split("_")[0]
                image_path = trajectory / "image_lcam_front" / f"{frame_id}_lcam_front.png"
                depth_path = trajectory / "depth_lcam_front" / f"{frame_id}_lcam_front_depth.png"
                _require(image_path.is_file() and depth_path.is_file(), "TartanAir synchronized frame missing")
                seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
                _require(seg is not None and seg.ndim == 2, "TartanAir segmentation format drift")
                targets = door_targets(seg, decode_depth(depth_path), door_label, eligibility)
                if not targets:
                    continue
                source_key = f"{environment}/{trajectory.name}/{frame_id}"
                if eligibility.get("near_requires_center_ray", False):
                    image_width = int(seg.shape[1])
                    near = any(
                        target["bbox_xyxy"][0] <= image_width / 2.0 <= target["bbox_xyxy"][2]
                        and target["depth_median_m"] <= float(eligibility["near_threshold_m"])
                        for target in targets
                    )
                else:
                    near = any(target["depth_median_m"] <= float(eligibility["near_threshold_m"]) for target in targets)
                eligible.append({
                    "source_key": source_key,
                    "source_key_sha256": hashlib.sha256(source_key.encode("utf-8")).hexdigest(),
                    "environment": environment,
                    "trajectory": trajectory.name,
                    "frame_id": frame_id,
                    "image_path": str(image_path.resolve()),
                    "depth_path": str(depth_path.resolve()),
                    "seg_path": str(seg_path.resolve()),
                    "door_label": door_label,
                    "stratum": "NEAR" if near else "FAR",
                    "targets": [{key: value for key, value in target.items() if key != "mask"} for target in targets],
                })
    near = sorted((row for row in eligible if row["stratum"] == "NEAR"), key=lambda row: row["source_key_sha256"])
    far = sorted((row for row in eligible if row["stratum"] == "FAR"), key=lambda row: row["source_key_sha256"])
    near_take, far_take = int(eligibility["near_take"]), int(eligibility["far_take"])
    _require(len(near) >= near_take and len(far) >= far_take, f"TartanAir denominator insufficient: near={len(near)} far={len(far)}")
    selected = near[:near_take] + far[:far_take]
    cohort_name = "s4" if protocol_id.endswith("_S4_V1") else ("s3" if protocol_id.endswith("_S3_V1") else "s2")
    cases = [{"case_id": f"tartanair-{cohort_name}-case-{index:03d}", **row} for index, row in enumerate(selected, start=1)]
    roster = {"schema_version": ROSTER_SCHEMA, "protocol_id": protocol_id, "manifest_sha256": file_hash(manifest_path), "eligible_case_count": len(eligible), "eligible_near_count": len(near), "eligible_far_count": len(far), "selection_observed_before_provider_calls": True, "provider_truth_access": False, "cases": cases}
    roster["roster_body_sha256"] = _json_hash(roster)
    _atomic_json(output_path, roster)
    return roster


def materialize_inputs(roster_path: Path, manifest_path: Path, public_output: Path, private_output: Path, private_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(not public_output.exists() and not private_output.exists(), "TartanAir inputs already exist")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_id = manifest.get("protocol_id")
    _require(protocol_id in SUPPORTED_PROTOCOL_IDS and roster.get("protocol_id") == protocol_id, "TartanAir roster/manifest protocol drift")
    declared = roster.pop("roster_body_sha256")
    _require(_json_hash(roster) == declared, "TartanAir roster body drift")
    roster["roster_body_sha256"] = declared
    public_cases, private_cases = [], []
    for row in roster["cases"]:
        image_path, depth_path, seg_path = map(Path, (row["image_path"], row["depth_path"], row["seg_path"]))
        seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
        targets = door_targets(seg, decode_depth(depth_path), int(row["door_label"]), manifest["eligibility"])
        _require([target["component_id"] for target in targets] == [target["component_id"] for target in row["targets"]], "TartanAir target drift")
        legal_targets = []
        for index, target in enumerate(targets, start=1):
            mask_path = (private_root / f"{row['case_id']}-target-{index:02d}.png").resolve()
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            _require(not mask_path.exists(), "refusing to overwrite TartanAir private mask")
            Image.fromarray((target["mask"] * 255).astype(np.uint8)).save(mask_path)
            legal_targets.append({"target_bbox_xyxy": target["bbox_xyxy"], "target_mask_path": str(mask_path), "target_mask_sha256": file_hash(mask_path), "target_pixel_count": target["pixel_count"], "target_depth_median_m": target["depth_median_m"], "target_depth_p10_m": target["depth_p10_m"], "target_depth_p90_m": target["depth_p90_m"], "depth_valid_fraction": target["depth_valid_fraction"]})
        public_case = {"case_id": row["case_id"], "episode_id": row["source_key"], "goal_contract": manifest["public_goal_contract"], "query": {"image_path": str(image_path), "image_sha256": file_hash(image_path)}}
        if "public_range_sensor" in manifest.get("provider", {}):
            public_case["range_sensor"] = {"depth_path": str(depth_path.resolve()), "depth_sha256": file_hash(depth_path)}
        public_cases.append(public_case)
        private_cases.append({"case_id": row["case_id"], "source_key": row["source_key"], "stratum": row["stratum"], "legal_targets": legal_targets, "true_interaction_range": any(target["target_depth_median_m"] <= 2.0 for target in legal_targets)})
    public = {"schema_version": PUBLIC_SCHEMA, "protocol_id": protocol_id, "manifest_sha256": file_hash(manifest_path), "roster_body_sha256": roster["roster_body_sha256"], "private_truth_access": False, "cases": public_cases}
    _atomic_json(public_output, public)
    private = {"schema_version": PRIVATE_SCHEMA, "protocol_id": protocol_id, "public_input_sha256": file_hash(public_output), "interaction_range_m": 2.0, "cases": private_cases}
    _atomic_json(private_output, private)
    return public, private


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    roster = sub.add_parser("freeze-roster")
    for name in ("manifest", "dataset-root", "label-root", "output"):
        roster.add_argument(f"--{name}", required=True, type=Path)
    inputs = sub.add_parser("materialize-inputs")
    for name in ("roster", "manifest", "public-output", "private-output", "private-root"):
        inputs.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze-roster":
        freeze_roster(args.manifest, args.dataset_root, args.label_root, args.output)
    else:
        materialize_inputs(args.roster, args.manifest, args.public_output, args.private_output, args.private_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
