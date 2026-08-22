#!/usr/bin/env python3
"""Development-only floor-connected door-region completion verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ground_plane import ground_mask_from_depth
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.mask_depth_completion_dev import binary_mask_iou, mask_observation
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


CONTACT_DILATION_PX = 15
CONTACT_PIXEL_MIN = 50
CONTACT_DEPTH_MAX_M = 2.0
MASK_HEIGHT_FRACTION_MIN = 0.40
DINO_IOU_MIN = 0.85
TARGET_MASK_IOU_MIN = 0.30


def observer_connected_ground(ground_mask: np.ndarray) -> np.ndarray:
    """Keep ground components connected to the observer's bottom-center area."""
    if ground_mask.ndim != 2:
        raise ValueError("ground mask must be HxW")
    height, width = ground_mask.shape
    closed = cv2.morphologyEx(ground_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
    seed = np.zeros_like(closed)
    seed[int(height * 0.90) :, int(width * 0.25) : int(width * 0.75)] = True
    seed_labels = set(int(value) for value in np.unique(labels[seed & closed]) if value != 0)
    if not seed_labels and count > 1:
        bottom_labels = set(int(value) for value in np.unique(labels[int(height * 0.85) :, :]) if value != 0)
        if bottom_labels:
            seed_labels = {max(bottom_labels, key=lambda value: int(stats[value, cv2.CC_STAT_AREA]))}
    return np.isin(labels, list(seed_labels)) if seed_labels else np.zeros_like(closed)


def functional_observation(door_mask: np.ndarray, connected_ground: np.ndarray, depth: np.ndarray) -> dict[str, Any]:
    if door_mask.shape != connected_ground.shape or door_mask.shape != depth.shape:
        raise ValueError("functional-region geometry mismatch")
    dilated = cv2.dilate(door_mask.astype(np.uint8), np.ones((CONTACT_DILATION_PX, CONTACT_DILATION_PX), np.uint8)) > 0
    contact = dilated & connected_ground & ~door_mask
    valid_contact = contact & np.isfinite(depth) & (depth >= 0.4) & (depth <= 8.0)
    values = depth[valid_contact]
    return {
        "ground_contact_pixel_count": int(valid_contact.sum()),
        "ground_contact_depth_median_m": float(np.median(values)) if values.size else None,
        "ground_contact_depth_p20_m": float(np.percentile(values, 20.0)) if values.size else None,
        "connected_ground_pixel_count": int(connected_ground.sum()),
    }


def select_functional_candidate(candidates: list[dict[str, Any]], dino_candidates: list[dict[str, Any]], width: int) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        box = validated_box(candidate["bbox_xyxy"], "functional candidate")
        overlaps = [iou(box, validated_box(row["bbox_xyxy"], "DINO candidate")) for row in dino_candidates]
        best_index = int(np.argmax(overlaps)) if overlaps else None
        best_iou = overlaps[best_index] if best_index is not None else 0.0
        if (
            box[0] <= width / 2.0 <= box[2]
            and candidate["mask_height_fraction"] >= MASK_HEIGHT_FRACTION_MIN
            and candidate["ground_contact_pixel_count"] >= CONTACT_PIXEL_MIN
            and candidate["ground_contact_depth_median_m"] is not None
            and candidate["ground_contact_depth_median_m"] <= CONTACT_DEPTH_MAX_M
            and best_iou >= DINO_IOU_MIN
        ):
            eligible.append(candidate | {"dino_consensus_iou": best_iou, "dino_candidate": dino_candidates[best_index]})
    return max(eligible, key=lambda row: (row["dino_consensus_iou"], row["proposal_score"], -row["provider_rank"])) if eligible else None


def run_provider(public_path: Path, parent_run_path: Path, model_path: Path, text_encoder_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    public, parent = _read(public_path), _read(parent_run_path)
    _require(parent.get("private_truth_access") is False and parent.get("public_sha256") == sha256(public_path), "functional parent boundary mismatch")
    public_cases = {case["case_id"]: case for case in public["cases"]}
    parent_cases = {case["case_id"]: case for case in parent["cases"]}
    masks_dir = output_path.parent / "selected_masks"
    ground_dir = output_path.parent / "ground_masks"
    masks_dir.mkdir(parents=True, exist_ok=False)
    ground_dir.mkdir(parents=True, exist_ok=False)

    from ultralytics import YOLOE

    model = YOLOE(str(model_path.resolve()))
    previous = Path.cwd()
    os.chdir(text_encoder_path.resolve().parent)
    try:
        model.set_classes(["door"])
    finally:
        os.chdir(previous)
    rows = []
    for case_id in sorted(public_cases):
        case, parent_case = public_cases[case_id], parent_cases[case_id]
        image_path, depth_path = Path(case["query"]["image_path"]), Path(case["range_sensor"]["depth_path"])
        _require(sha256(image_path) == case["query"]["image_sha256"] and sha256(depth_path) == case["range_sensor"]["depth_sha256"], "functional public input drift")
        depth = decode_depth(depth_path)
        ground, plane = ground_mask_from_depth(depth)
        connected = observer_connected_ground(ground)
        ground_path = ground_dir / f"{case_id}.png"
        Image.fromarray(connected.astype(np.uint8) * 255).save(ground_path)
        result = model.predict(source=str(image_path), verbose=False, device=device, imgsz=640, conf=0.001, max_det=100)[0]
        _require(result.masks is None or len(result.boxes) == len(result.masks.data), "functional YOLOE box/mask mismatch")
        indices = [] if result.masks is None else sorted(range(len(result.boxes)), key=lambda index: float(result.boxes.conf[index]), reverse=True)[:10]
        candidates = []
        for rank, index in enumerate(indices, start=1):
            door_mask = result.masks.data[index].detach().cpu().numpy() >= 0.5
            try:
                mask_row = mask_observation(door_mask, depth, result.boxes.xyxy[index].detach().cpu().tolist(), depth.shape[1], depth.shape[0])
            except ValueError:
                continue
            candidates.append(mask_row | functional_observation(door_mask, connected, depth) | {"provider_rank": rank, "proposal_score": float(result.boxes.conf[index]), "result_index": index})
        selected = select_functional_candidate(candidates, parent_case["dino_candidates"], depth.shape[1])
        if selected is not None:
            selected_mask = (result.masks.data[selected["result_index"]].detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
            selected_path = masks_dir / f"{case_id}.png"
            Image.fromarray(selected_mask).save(selected_path)
            selected = selected | {"selected_mask_path": str(selected_path.resolve()), "selected_mask_sha256": sha256(selected_path)}
        rows.append({"case_id": case_id, "image_width": int(depth.shape[1]), "image_height": int(depth.shape[0]), "plane": plane, "connected_ground_mask_path": str(ground_path.resolve()), "connected_ground_mask_sha256": sha256(ground_path), "candidates": candidates, "selected_candidate": selected, "completion": selected is not None})
        print(f"functional-region {len(rows)}/{len(public_cases)} case={case_id} completion={selected is not None}", flush=True)
    payload = {"schema_version": "blindassist_functional_region_completion_development_prediction_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "public_sha256": sha256(public_path), "parent_run_sha256": sha256(parent_run_path), "private_truth_access": False, "provider": {"model_sha256": sha256(model_path), "text_encoder_sha256": sha256(text_encoder_path), "contact_dilation_px": CONTACT_DILATION_PX, "contact_pixel_min": CONTACT_PIXEL_MIN, "contact_depth_max_m": CONTACT_DEPTH_MAX_M, "mask_height_fraction_min": MASK_HEIGHT_FRACTION_MIN, "dino_iou_min": DINO_IOU_MIN, "device": device}, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def evaluate(public_path: Path, private_path: Path, prediction_path: Path, output_path: Path) -> dict[str, Any]:
    private, prediction = _read(private_path), _read(prediction_path)
    _require(prediction.get("private_truth_access") is False and prediction.get("public_sha256") == sha256(public_path), "functional prediction boundary mismatch")
    truths = {case["case_id"]: case for case in private["cases"]}
    predictions = {case["case_id"]: case for case in prediction["cases"]}
    rows = []
    for case_id in sorted(truths):
        truth, observed = truths[case_id], predictions[case_id]
        targets = truth["legal_targets"]
        opportunity = any(target["target_bbox_xyxy"][0] <= observed["image_width"] / 2.0 <= target["target_bbox_xyxy"][2] and float(target["target_depth_median_m"]) <= 2.0 for target in targets)
        selected = observed["selected_candidate"]
        matched_index, best_mask_iou, best_bbox_iou, bbox_matched_index = None, None, None, None
        if selected is not None:
            selected_path = Path(selected["selected_mask_path"])
            _require(sha256(selected_path) == selected["selected_mask_sha256"], "functional selected mask drift")
            with Image.open(selected_path) as opened:
                selected_mask = np.asarray(opened.convert("L")) > 0
            target_masks = []
            for target in targets:
                target_path = Path(target["target_mask_path"])
                _require(sha256(target_path) == target["target_mask_sha256"], "functional private mask drift")
                with Image.open(target_path) as opened:
                    target_masks.append(np.asarray(opened.convert("L")) > 0)
            overlaps = [binary_mask_iou(selected_mask, target_mask) for target_mask in target_masks]
            best_mask_iou = max(overlaps) if overlaps else None
            matched_index = int(np.argmax(overlaps)) if overlaps and best_mask_iou >= TARGET_MASK_IOU_MIN else None
            bbox_overlaps = [iou(validated_box(selected["bbox_xyxy"], f"{case_id} selected bbox"), validated_box(target["target_bbox_xyxy"], f"{case_id} target bbox")) for target in targets]
            best_bbox_iou = max(bbox_overlaps) if bbox_overlaps else None
            bbox_matched_index = int(np.argmax(bbox_overlaps)) if bbox_overlaps and best_bbox_iou >= 0.30 else None
        matched_depth = float(targets[matched_index]["target_depth_median_m"]) if matched_index is not None else None
        bbox_matched_depth = float(targets[bbox_matched_index]["target_depth_median_m"]) if bbox_matched_index is not None else None
        correct = selected is not None and matched_depth is not None and matched_depth <= 2.0
        bbox_correct = selected is not None and bbox_matched_depth is not None and bbox_matched_depth <= 2.0
        rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "target_selected": matched_index is not None, "selected_target_mask_iou": best_mask_iou, "matched_target_depth_m": matched_depth, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct, "bbox_diagnostic_iou": best_bbox_iou, "bbox_diagnostic_target_selected": bbox_matched_index is not None, "bbox_diagnostic_correct_completion": bbox_correct, "bbox_diagnostic_false_completion": selected is not None and not bbox_correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    payload = {"schema_version": "blindassist_functional_region_completion_development_evaluation_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "prediction_sha256": sha256(prediction_path), "private_sha256": sha256(private_path), "opportunity_count": opportunities, "decision_count": sum(row["completion_decision"] for row in rows), "correct_count": correct, "false_count": false, "coverage": correct / opportunities if opportunities else None, "bbox_diagnostic": {"correct_count": sum(row["bbox_diagnostic_correct_completion"] for row in rows), "false_count": sum(row["bbox_diagnostic_false_completion"] for row in rows), "coverage": sum(row["bbox_diagnostic_correct_completion"] for row in rows) / opportunities if opportunities else None}, "rows": rows, "terminal": "DEV_FUNCTIONAL_REGION_COMPLETION_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "DEV_FUNCTIONAL_REGION_COMPLETION_NOT_PROMISING"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_path, evaluation_path = args.output / "prediction.json", args.output / "evaluation.json"
    run_provider(args.public, args.parent_run, args.model, args.text_encoder, prediction_path, args.device)
    result = evaluate(args.public, args.private, prediction_path, evaluation_path)
    print(json.dumps({key: result[key] for key in ("opportunity_count", "decision_count", "correct_count", "false_count", "coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
