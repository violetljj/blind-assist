#!/usr/bin/env python3
"""Development-only fusion of a public-data door mask with sealed S4 proposals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ade20k_functional_aperture_dev import (
    DEPTH_PERCENTILE,
    MAX_DEPTH_M,
    MIN_APPARENT_HEIGHT_M,
    MIN_DINO_IOU,
    MIN_HEIGHT_FRACTION,
    MIN_POSITIVE_FRACTION,
    region_depth_percentile,
)
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_PUBLIC_DOOR_SEMANTIC_FUSION_DEV_V1"
RUN_SCHEMA = "blindassist_public_door_semantic_fusion_dev_run_v1"
MODEL_SHA256 = "c01293ae986a14c15aadd3c1f44054fe9a5af97b0aa3c57e418480d2bcbe0f9b"


def door_fraction(class_map: np.ndarray, box: Sequence[float], width: int, height: int, door_id: int) -> float:
    x1, y1, x2, y2 = validated_box(box, "door semantic candidate")
    sx, sy = class_map.shape[1] / width, class_map.shape[0] / height
    crop = class_map[max(0, int(np.floor(y1 * sy))):min(class_map.shape[0], int(np.ceil(y2 * sy))), max(0, int(np.floor(x1 * sx))):min(class_map.shape[1], int(np.ceil(x2 * sx)))]
    return float((crop == door_id).sum() / crop.size) if crop.size else 0.0


def select_candidate(candidates: Sequence[Mapping[str, Any]], dino_candidates: Sequence[Mapping[str, Any]], class_map: np.ndarray, depth: np.ndarray, width: int, height: int, door_id: int) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        box = validated_box(candidate["bbox_xyxy"], "YOLOE candidate")
        if not box[0] <= width / 2.0 <= box[2]:
            continue
        height_fraction = (box[3] - box[1]) / height
        depth_p20 = region_depth_percentile(depth, box, width, height)
        if depth_p20 is None or height_fraction < MIN_HEIGHT_FRACTION or depth_p20 > MAX_DEPTH_M or depth_p20 * height_fraction < MIN_APPARENT_HEIGHT_M:
            continue
        overlaps = [iou(box, validated_box(row["bbox_xyxy"], "DINO candidate")) for row in dino_candidates]
        best_iou = max(overlaps, default=0.0)
        fraction = door_fraction(class_map, box, width, height, door_id)
        if best_iou >= MIN_DINO_IOU and fraction >= MIN_POSITIVE_FRACTION:
            eligible.append(dict(candidate) | {"sensor_region_depth_p20_m": depth_p20, "height_fraction": height_fraction, "apparent_height_proxy_m": depth_p20 * height_fraction, "dino_consensus_iou": best_iou, "door_pixel_fraction": fraction})
    return max(eligible, key=lambda row: (float(row["door_pixel_fraction"]), float(row["dino_consensus_iou"]), float(row["proposal_score"]))) if eligible else None


def run(public_path: Path, sealed_run_path: Path, model_path: Path, training_receipt_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not output_path.exists(), "public door semantic development output already exists")
    public, sealed, training = _read(public_path), _read(sealed_run_path), _read(training_receipt_path)
    _require(sealed.get("private_truth_access") is False and sealed.get("public_sha256") == sha256(public_path), "sealed S4 boundary mismatch")
    _require(training.get("private_truth_access") is False and model_path.is_file() and sha256(model_path) == MODEL_SHA256, "public door model boundary drift")
    public_cases = {case["case_id"]: case for case in public["cases"]}
    sealed_cases = {case["case_id"]: case for case in sealed["cases"]}
    _require(set(public_cases) == set(sealed_cases), "public door semantic roster mismatch")

    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(model_path.resolve()))
    door_ids = [int(key) for key, value in model.names.items() if value == "door"]
    _require(len(door_ids) == 1, "trained door label contract drift")
    rows = []
    for index, case_id in enumerate(sorted(public_cases), start=1):
        public_case, sealed_case = public_cases[case_id], sealed_cases[case_id]
        image_path, depth_path = Path(public_case["query"]["image_path"]), Path(public_case["range_sensor"]["depth_path"])
        _require(sha256(image_path) == public_case["query"]["image_sha256"] and sha256(depth_path) == public_case["range_sensor"]["depth_sha256"], "public RGB-D drift")
        with Image.open(image_path) as image:
            width, height = image.size
        started = time.perf_counter()
        result = model.predict(source=str(image_path), imgsz=640, device=device, verbose=False)[0]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        selected = select_candidate(sealed_case["yoloe_candidates"], sealed_case["dino_candidates"], result.semantic_mask.data.detach().cpu().numpy(), decode_depth(depth_path), width, height, door_ids[0])
        rows.append({"case_id": case_id, "selected_candidate": selected, "completion": selected is not None, "semantic_latency_ms": (time.perf_counter() - started) * 1000.0})
        print(f"public-door-semantic {index}/{len(public_cases)} case={case_id} completion={selected is not None}", flush=True)
    payload = {"schema_version": RUN_SCHEMA, "protocol_id": PROTOCOL_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "public_sha256": sha256(public_path), "sealed_s4_run_sha256": sha256(sealed_run_path), "training_receipt_sha256": sha256(training_receipt_path), "private_truth_access": False, "stateless_current_frame_only": True, "development_only": True, "threshold_prompt_model_or_rule_sweep": False, "provider": {"model_sha256": MODEL_SHA256, "ultralytics": ultralytics.__version__, "minimum_door_pixel_fraction": MIN_POSITIVE_FRACTION, "sensor_depth_percentile": DEPTH_PERCENTILE, "sensor_depth_max_m": MAX_DEPTH_M, "minimum_height_fraction": MIN_HEIGHT_FRACTION, "minimum_apparent_height_m": MIN_APPARENT_HEIGHT_M, "minimum_dino_iou": MIN_DINO_IOU, "device": device}, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def evaluate(private_path: Path, run_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "public door semantic evaluation already exists")
    private, run_payload = _read(private_path), _read(run_path)
    _require(run_payload.get("schema_version") == RUN_SCHEMA and run_payload.get("private_truth_access") is False, "public door semantic evaluation boundary mismatch")
    truths, runs = ({case["case_id"]: case for case in payload["cases"]} for payload in (private, run_payload))
    _require(set(truths) == set(runs), "public door semantic evaluation roster mismatch")
    rows = []
    for case_id in sorted(truths):
        legal, selected = truths[case_id]["legal_targets"], runs[case_id]["selected_candidate"]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in legal]
        opportunity = any(box[0] <= 320.0 <= box[2] and float(target["target_depth_median_m"]) <= 2.0 for box, target in zip(targets, legal, strict=True))
        matched = None
        if selected is not None:
            overlaps = [iou(validated_box(selected["bbox_xyxy"], f"{case_id} selected"), target) for target in targets]
            matched = int(np.argmax(overlaps)) if overlaps and max(overlaps) >= 0.30 else None
        correct = selected is not None and matched is not None and float(legal[matched]["target_depth_median_m"]) <= 2.0
        rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "target_selected": matched is not None, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    payload = {"schema_version": "blindassist_public_door_semantic_fusion_dev_evaluation_v1", "protocol_id": PROTOCOL_ID, "case_count": len(rows), "completion_opportunity_count": opportunities, "completion_decision_count": correct + false, "correct_completion_count": correct, "false_completion_count": false, "missed_completion_opportunity_count": sum(row["missed_completion_opportunity"] for row in rows), "correct_completion_coverage": correct / opportunities if opportunities else None, "rows": rows, "development_only": True, "confirmation_claim_authorized": False, "terminal": "DEV_PUBLIC_DOOR_SEMANTIC_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "DEV_PUBLIC_DOOR_SEMANTIC_NOT_PROMISING"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("public", "sealed-run", "model", "training-receipt", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    execute.add_argument("--device", default="cuda:0")
    evaluation = sub.add_parser("evaluate")
    for name in ("private", "run", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.public, args.sealed_run, args.model, args.training_receipt, args.output, args.device)
    else:
        result = evaluate(args.private, args.run, args.output)
        print(json.dumps({key: result[key] for key in ("completion_opportunity_count", "correct_completion_count", "false_completion_count", "correct_completion_coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
