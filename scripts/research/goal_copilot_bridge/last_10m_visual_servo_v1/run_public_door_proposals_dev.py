#!/usr/bin/env python3
"""Run public-data door models as positive-only proposal providers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


def semantic_components(class_map: np.ndarray, door_id: int, width: int, height: int, minimum_pixels: int = 16) -> list[dict]:
    count, _, stats, _ = cv2.connectedComponentsWithStats((class_map == door_id).astype(np.uint8), 8)
    sx, sy = width / class_map.shape[1], height / class_map.shape[0]
    candidates = []
    for component in range(1, count):
        x, y, box_width, box_height, pixels = [int(value) for value in stats[component]]
        if pixels < minimum_pixels:
            continue
        candidates.append({"bbox_xyxy": [x * sx, y * sy, (x + box_width) * sx, (y + box_height) * sy], "proposal_score": pixels / class_map.size, "semantic_pixel_count": pixels})
    return sorted(candidates, key=lambda row: row["proposal_score"], reverse=True)[:10]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"), default="DEVELOPMENT_ONLY")
    args = parser.parse_args()
    _require(not args.output.exists(), "public door proposal output already exists")
    public = _read(args.public)
    _require(public.get("provider_truth_access") is False and args.model.is_file(), "public door proposal boundary mismatch")
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    door_ids = [int(key) for key, name in model.names.items() if name == "door"]
    _require(len(door_ids) == 1, "public door model taxonomy drift")
    rows = []
    for index, case in enumerate(public["cases"], start=1):
        image_path = Path(case["query"]["image_path"])
        _require(sha256(image_path) == case["query"]["image_sha256"], "public approach image drift")
        with Image.open(image_path) as opened:
            width, height = opened.size
        result = model.predict(source=str(image_path), imgsz=640, conf=args.confidence, device=args.device, verbose=False)[0]
        if model.task == "semantic":
            candidates = semantic_components(result.semantic_mask.data.detach().cpu().numpy(), door_ids[0], width, height)
        else:
            candidates = []
            boxes = result.boxes
            for box, score, class_id in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist(), strict=True):
                if int(class_id) == door_ids[0]:
                    candidates.append({"bbox_xyxy": box, "proposal_score": float(score)})
            candidates = sorted(candidates, key=lambda row: row["proposal_score"], reverse=True)[:10]
        candidates = [candidate | {"provider_rank": rank} for rank, candidate in enumerate(candidates, start=1)]
        rows.append({"case_id": case["case_id"], "image_width": width, "image_height": height, "candidates": candidates})
        print(f"public-door-proposals {index}/{len(public['cases'])} case={case['case_id']} candidates={len(candidates)}", flush=True)
    payload = {
        "schema_version": "blindassist_public_door_proposal_development_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "private_truth_access": False,
        "public_sha256": sha256(args.public),
        "provider": {"model_path": str(args.model.resolve()), "model_sha256": sha256(args.model), "model_task": model.task, "door_class_id": door_ids[0], "confidence": args.confidence, "device": args.device},
        "cases": rows,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"case_count": len(rows), "cases_with_candidates": sum(bool(row["candidates"]) for row in rows), "provider": payload["provider"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
