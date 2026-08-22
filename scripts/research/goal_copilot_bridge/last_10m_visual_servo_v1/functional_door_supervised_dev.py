#!/usr/bin/env python3
"""Development-only supervised functional-door verifier for sealed S2 outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.sensor_depth_fusion import evaluate as evaluate_completion
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_SUPERVISED_FUNCTIONAL_DOOR_VERIFIER_DEV_V1"
CONFIDENCE_THRESHOLD = 0.25
MATCH_IOU_THRESHOLD = 0.30
ROOM_DOOR_CLASS = 0
FURNITURE_DOOR_CLASSES = {2, 3}


def verify_candidate(candidate: Mapping[str, Any] | None, detections: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if candidate is None:
        return None, {"reason": "NO_PARENT_CANDIDATE", "room_match": None, "furniture_match": None}
    box = validated_box(candidate["bbox_xyxy"], "parent candidate")
    matched = []
    for detection in detections:
        overlap = iou(box, validated_box(detection["bbox_xyxy"], "functional detection"))
        if overlap >= MATCH_IOU_THRESHOLD:
            matched.append(dict(detection) | {"candidate_iou": overlap})
    room = max((row for row in matched if int(row["class_id"]) == ROOM_DOOR_CLASS), key=lambda row: float(row["confidence"]), default=None)
    furniture = max((row for row in matched if int(row["class_id"]) in FURNITURE_DOOR_CLASSES), key=lambda row: float(row["confidence"]), default=None)
    accepted = room is not None and (furniture is None or float(room["confidence"]) > float(furniture["confidence"]))
    evidence = {
        "reason": "ROOM_DOOR_WINS" if accepted else ("FURNITURE_DOOR_WINS" if furniture is not None else "NO_ROOM_DOOR_MATCH"),
        "room_match": room,
        "furniture_match": furniture,
    }
    return (dict(candidate) | {"functional_door_evidence": evidence}) if accepted else None, evidence


def run(public_path: Path, sensor_run_path: Path, weights_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "supervised verifier development output already exists")
    public, parent = _read(public_path), _read(sensor_run_path)
    _require(parent.get("private_truth_access") is False and parent.get("public_sha256") == sha256(public_path), "supervised verifier parent boundary mismatch")
    _require(weights_path.is_file(), "functional-door weights missing")

    import torch
    import ultralytics
    from ultralytics import YOLO

    public_cases = {case["case_id"]: case for case in public["cases"]}
    parent_cases = {case["case_id"]: case for case in parent["cases"]}
    _require(set(public_cases) == set(parent_cases), "supervised verifier roster mismatch")
    model = YOLO(str(weights_path.resolve()))
    rows = []
    for index, case_id in enumerate(sorted(public_cases), start=1):
        image_path = Path(public_cases[case_id]["query"]["image_path"])
        result = model.predict(source=str(image_path), imgsz=640, conf=CONFIDENCE_THRESHOLD, device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist(), strict=True):
                class_index = int(class_id)
                detections.append({"bbox_xyxy": [float(value) for value in box], "confidence": float(confidence), "class_id": class_index, "class_name": str(result.names[class_index])})
        selected, evidence = verify_candidate(parent_cases[case_id]["selected_candidate"], detections)
        rows.append({"case_id": case_id, "functional_detections": detections, "verification": evidence, "selected_candidate": selected, "completion": selected is not None})
        print(f"functional-door {index}/{len(public_cases)} case={case_id} detections={len(detections)} completion={selected is not None}", flush=True)
    payload = {
        "schema_version": "blindassist_supervised_functional_door_verifier_dev_run_v1",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_sha256": sha256(public_path),
        "sensor_depth_run_sha256": sha256(sensor_run_path),
        "private_truth_access": False,
        "provider_public_aligned_sensor_depth": True,
        "development_only": True,
        "threshold_prompt_model_or_rule_sweep": False,
        "provider": {"weights_path": str(weights_path.resolve()), "weights_sha256": sha256(weights_path), "ultralytics": ultralytics.__version__, "torch": torch.__version__, "python": platform.python_version(), "confidence_threshold": CONFIDENCE_THRESHOLD, "match_iou_threshold": MATCH_IOU_THRESHOLD},
        "selection_rule": "Accept sealed YOLOE+DINO+aligned-depth candidate only when a matched room-door detection exists and outranks matched cabinet/refrigerator-door evidence.",
        "cases": rows,
    }
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("public", "sensor-run", "weights", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    evaluation = sub.add_parser("evaluate")
    for name in ("private", "run", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.public, args.sensor_run, args.weights, args.output)
    else:
        result = evaluate_completion(args.private, args.run, args.output)
        print(json.dumps({key: result[key] for key in ("completion_opportunity_count", "correct_completion_count", "false_completion_count", "correct_completion_coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
