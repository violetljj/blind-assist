#!/usr/bin/env python3
"""Development-only YOLOE mask-depth completion on a consumed cohort.

The provider uses only the parent public input, its aligned public depth, and
the parent's public DINO proposals. Private target masks are never loaded by
the provider. Evaluation happens only after the prediction receipt is sealed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


MASK_DEPTH_MAX_M = 2.0
MASK_HEIGHT_FRACTION_MIN = 0.40
DINO_IOU_MIN = 0.85
TARGET_IOU_MIN = 0.30
TARGET_MASK_IOU_MIN = 0.30
CENTER_BAND_HALF_WIDTH_PX = 10


def mask_observation(mask: np.ndarray, depth: np.ndarray, box: Sequence[float], width: int, height: int) -> dict[str, Any]:
    _require(mask.shape == depth.shape == (height, width), "mask/depth geometry mismatch")
    binary = mask >= 0.5
    ys, xs = np.nonzero(binary)
    _require(xs.size > 0, "empty YOLOE mask")
    valid = depth[binary & np.isfinite(depth) & (depth >= 0.4) & (depth <= 8.0)]
    _require(valid.size > 0, "YOLOE mask has no valid public depth")
    center = width // 2
    left, right = max(0, center - CENTER_BAND_HALF_WIDTH_PX), min(width, center + CENTER_BAND_HALF_WIDTH_PX + 1)
    return {
        "bbox_xyxy": [float(value) for value in validated_box(box, "YOLOE mask box")],
        "mask_pixel_count": int(binary.sum()),
        "mask_height_fraction": float((ys.max() - ys.min() + 1) / height),
        "mask_center_band_pixel_count": int(binary[:, left:right].sum()),
        "mask_depth_median_m": float(np.median(valid)),
        "mask_depth_p20_m": float(np.percentile(valid, 20.0)),
        "mask_valid_depth_fraction": float(valid.size / binary.sum()),
    }


def select_candidate(candidates: list[dict[str, Any]], dino_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        box = validated_box(candidate["bbox_xyxy"], "YOLOE mask candidate")
        overlaps = [iou(box, validated_box(row["bbox_xyxy"], "DINO candidate")) for row in dino_candidates]
        best_index = int(np.argmax(overlaps)) if overlaps else None
        best_iou = overlaps[best_index] if best_index is not None else 0.0
        if (
            candidate["mask_height_fraction"] >= MASK_HEIGHT_FRACTION_MIN
            and candidate["mask_center_band_pixel_count"] > 0
            and candidate["mask_depth_median_m"] <= MASK_DEPTH_MAX_M
            and best_iou >= DINO_IOU_MIN
        ):
            eligible.append(candidate | {"dino_consensus_iou": best_iou, "dino_candidate": dino_candidates[best_index]})
    return max(eligible, key=lambda row: (row["dino_consensus_iou"], row["proposal_score"], -row["provider_rank"])) if eligible else None


def binary_mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    _require(left.shape == right.shape, "mask IoU geometry mismatch")
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def run_provider(public_path: Path, parent_run_path: Path, model_path: Path, text_encoder_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not output_path.exists(), "mask-depth development prediction already exists")
    public, parent = _read(public_path), _read(parent_run_path)
    _require(parent.get("private_truth_access") is False and parent.get("public_sha256") == sha256(public_path), "parent provider boundary mismatch")
    public_cases = {case["case_id"]: case for case in public["cases"]}
    parent_cases = {case["case_id"]: case for case in parent["cases"]}
    _require(set(public_cases) == set(parent_cases), "mask-depth parent roster mismatch")

    import torch
    from ultralytics import YOLOE

    model = YOLOE(str(model_path.resolve()))
    masks_dir = output_path.parent / "selected_masks"
    masks_dir.mkdir(parents=True, exist_ok=False)
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
        _require(sha256(image_path) == case["query"]["image_sha256"] and sha256(depth_path) == case["range_sensor"]["depth_sha256"], "mask-depth public input drift")
        depth = decode_depth(depth_path)
        result = model.predict(source=str(image_path), verbose=False, device=device, imgsz=640, conf=0.001, max_det=100)[0]
        _require(result.masks is None or len(result.boxes) == len(result.masks.data), "YOLOE box/mask count mismatch")
        ranked_indices = [] if result.masks is None else sorted(range(len(result.boxes)), key=lambda index: float(result.boxes.conf[index]), reverse=True)[:10]
        candidates = []
        for rank, index in enumerate(ranked_indices, start=1):
            try:
                observed = mask_observation(result.masks.data[index].detach().cpu().numpy(), depth, result.boxes.xyxy[index].detach().cpu().tolist(), depth.shape[1], depth.shape[0])
            except ValueError:
                continue
            candidates.append(observed | {"provider_rank": rank, "proposal_score": float(result.boxes.conf[index]), "result_index": index})
        selected = select_candidate(candidates, parent_case["dino_candidates"])
        if selected is not None:
            mask_path = masks_dir / f"{case_id}.png"
            selected_mask = (result.masks.data[selected["result_index"]].detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
            Image.fromarray(selected_mask).save(mask_path)
            selected = selected | {"selected_mask_path": str(mask_path.resolve()), "selected_mask_sha256": sha256(mask_path)}
        rows.append({"case_id": case_id, "image_width": int(depth.shape[1]), "image_height": int(depth.shape[0]), "candidates": candidates, "selected_candidate": selected, "completion": selected is not None})
        print(f"mask-depth {len(rows)}/{len(public_cases)} case={case_id} completion={selected is not None}", flush=True)
    payload = {
        "schema_version": "blindassist_mask_depth_completion_development_prediction_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "public_sha256": sha256(public_path),
        "parent_run_sha256": sha256(parent_run_path),
        "private_truth_access": False,
        "provider": {"model_sha256": sha256(model_path), "text_encoder_sha256": sha256(text_encoder_path), "mask_depth_max_m": MASK_DEPTH_MAX_M, "mask_height_fraction_min": MASK_HEIGHT_FRACTION_MIN, "dino_iou_min": DINO_IOU_MIN, "center_band_half_width_px": CENTER_BAND_HALF_WIDTH_PX, "device": device},
        "cases": rows,
    }
    _atomic_json(output_path, payload)
    return payload


def evaluate(public_path: Path, private_path: Path, prediction_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "mask-depth development evaluation already exists")
    private, prediction = _read(private_path), _read(prediction_path)
    _require(prediction.get("private_truth_access") is False and prediction.get("public_sha256") == sha256(public_path), "mask-depth prediction boundary mismatch")
    truths = {case["case_id"]: case for case in private["cases"]}
    predictions = {case["case_id"]: case for case in prediction["cases"]}
    _require(set(truths) == set(predictions), "mask-depth evaluation roster mismatch")
    rows = []
    for case_id in sorted(truths):
        truth, observed = truths[case_id], predictions[case_id]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in truth["legal_targets"]]
        opportunity = any(box[0] <= observed["image_width"] / 2.0 <= box[2] and float(target["target_depth_median_m"]) <= 2.0 for box, target in zip(targets, truth["legal_targets"], strict=True))
        selected = observed["selected_candidate"]
        matched_index = None
        best_mask_iou = None
        if selected is not None:
            selected_mask_path = Path(selected["selected_mask_path"])
            _require(selected_mask_path.is_file() and sha256(selected_mask_path) == selected["selected_mask_sha256"], "selected provider mask drift")
            with Image.open(selected_mask_path) as opened:
                selected_mask = np.asarray(opened.convert("L")) > 0
            target_masks = []
            for target in truth["legal_targets"]:
                target_path = Path(target["target_mask_path"])
                _require(target_path.is_file() and sha256(target_path) == target["target_mask_sha256"], "private target mask drift")
                with Image.open(target_path) as opened:
                    target_masks.append(np.asarray(opened.convert("L")) > 0)
            overlaps = [binary_mask_iou(selected_mask, target_mask) for target_mask in target_masks]
            best_mask_iou = max(overlaps) if overlaps else None
            matched_index = int(np.argmax(overlaps)) if overlaps and best_mask_iou >= TARGET_MASK_IOU_MIN else None
        matched_depth = float(truth["legal_targets"][matched_index]["target_depth_median_m"]) if matched_index is not None else None
        correct = selected is not None and matched_depth is not None and matched_depth <= 2.0
        rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "target_selected": matched_index is not None, "selected_target_mask_iou": best_mask_iou, "matched_target_depth_m": matched_depth, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    payload = {
        "schema_version": "blindassist_mask_depth_completion_development_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "prediction_sha256": sha256(prediction_path),
        "private_sha256": sha256(private_path),
        "opportunity_count": opportunities,
        "decision_count": sum(row["completion_decision"] for row in rows),
        "correct_count": correct,
        "false_count": false,
        "coverage": correct / opportunities if opportunities else None,
        "rows": rows,
        "terminal": "DEV_MASK_DEPTH_COMPLETION_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "DEV_MASK_DEPTH_COMPLETION_NOT_PROMISING",
    }
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
