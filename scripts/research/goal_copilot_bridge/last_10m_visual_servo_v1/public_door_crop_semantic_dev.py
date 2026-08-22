#!/usr/bin/env python3
"""Development-only candidate-crop semantic re-observation on sealed S4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ade20k_functional_aperture_dev import DEPTH_PERCENTILE, MAX_DEPTH_M, MIN_APPARENT_HEIGHT_M, MIN_DINO_IOU, MIN_HEIGHT_FRACTION, MIN_POSITIVE_FRACTION, region_depth_percentile
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.public_door_semantic_fusion_dev import MODEL_SHA256, evaluate
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_PUBLIC_DOOR_CROP_SEMANTIC_DEV_V1"
RUN_SCHEMA = "blindassist_public_door_semantic_fusion_dev_run_v1"


def crop_box(image: Image.Image, box: Sequence[float]) -> Image.Image:
    x1, y1, x2, y2 = validated_box(box, "candidate crop")
    return image.crop((max(0, int(np.floor(x1))), max(0, int(np.floor(y1))), min(image.width, int(np.ceil(x2))), min(image.height, int(np.ceil(y2)))))


def run(public_path: Path, sealed_run_path: Path, model_path: Path, training_receipt_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not output_path.exists(), "crop semantic development output already exists")
    public, sealed, training = _read(public_path), _read(sealed_run_path), _read(training_receipt_path)
    _require(sealed.get("private_truth_access") is False and sealed.get("public_sha256") == sha256(public_path), "sealed S4 boundary mismatch")
    _require(training.get("private_truth_access") is False and sha256(model_path) == MODEL_SHA256, "public door model boundary drift")
    public_cases = {case["case_id"]: case for case in public["cases"]}
    sealed_cases = {case["case_id"]: case for case in sealed["cases"]}
    _require(set(public_cases) == set(sealed_cases), "crop semantic roster mismatch")

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
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        depth = decode_depth(depth_path)
        eligible = []
        latency_ms = 0.0
        for candidate in sealed_case["yoloe_candidates"]:
            box = validated_box(candidate["bbox_xyxy"], "YOLOE candidate")
            if not box[0] <= width / 2.0 <= box[2]:
                continue
            height_fraction = (box[3] - box[1]) / height
            depth_p20 = region_depth_percentile(depth, box, width, height)
            overlaps = [iou(box, validated_box(row["bbox_xyxy"], "DINO candidate")) for row in sealed_case["dino_candidates"]]
            best_iou = max(overlaps, default=0.0)
            if depth_p20 is None or height_fraction < MIN_HEIGHT_FRACTION or depth_p20 > MAX_DEPTH_M or depth_p20 * height_fraction < MIN_APPARENT_HEIGHT_M or best_iou < MIN_DINO_IOU:
                continue
            started = time.perf_counter()
            result = model.predict(source=crop_box(image, box), imgsz=640, device=device, verbose=False)[0]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latency_ms += (time.perf_counter() - started) * 1000.0
            class_map = result.semantic_mask.data.detach().cpu().numpy()
            fraction = float((class_map == door_ids[0]).sum() / class_map.size)
            if fraction >= MIN_POSITIVE_FRACTION:
                eligible.append(dict(candidate) | {"sensor_region_depth_p20_m": depth_p20, "height_fraction": height_fraction, "apparent_height_proxy_m": depth_p20 * height_fraction, "dino_consensus_iou": best_iou, "crop_door_pixel_fraction": fraction})
        selected = max(eligible, key=lambda row: (float(row["crop_door_pixel_fraction"]), float(row["dino_consensus_iou"]), float(row["proposal_score"]))) if eligible else None
        rows.append({"case_id": case_id, "selected_candidate": selected, "completion": selected is not None, "semantic_latency_ms": latency_ms})
        print(f"public-door-crop {index}/{len(public_cases)} case={case_id} eligible={len(eligible)} completion={selected is not None}", flush=True)
    payload = {"schema_version": RUN_SCHEMA, "protocol_id": PROTOCOL_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "public_sha256": sha256(public_path), "sealed_s4_run_sha256": sha256(sealed_run_path), "training_receipt_sha256": sha256(training_receipt_path), "private_truth_access": False, "stateless_current_frame_only": True, "development_only": True, "threshold_prompt_model_or_rule_sweep": False, "provider": {"model_sha256": MODEL_SHA256, "ultralytics": ultralytics.__version__, "observation": "exact YOLOE candidate crop resized to 640", "minimum_crop_door_pixel_fraction": MIN_POSITIVE_FRACTION, "sensor_depth_percentile": DEPTH_PERCENTILE, "sensor_depth_max_m": MAX_DEPTH_M, "minimum_height_fraction": MIN_HEIGHT_FRACTION, "minimum_apparent_height_m": MIN_APPARENT_HEIGHT_M, "minimum_dino_iou": MIN_DINO_IOU, "device": device}, "cases": rows}
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
