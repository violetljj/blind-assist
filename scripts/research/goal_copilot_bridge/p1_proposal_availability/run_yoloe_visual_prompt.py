#!/usr/bin/env python3
"""Run a GT-blind YOLOE visual-prompt bounded proposal arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


PREDICTION_SCHEMA = "blindassist_p1_pa0_prediction_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_frame(video_path: str, frame_index: int):
    import cv2

    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"could not decode frame {frame_index} from {video_path}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-limit", type=int, help="Provider mechanics smoke only; omit for the formal seven-case arm")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

    public = json.loads(args.public.read_text(encoding="utf-8"))
    if public["provider_contract"]["maximum_candidates"] != 10:
        raise ValueError("unexpected candidate cap")
    video_hashes: dict[str, str] = {}
    selected_cases = public["cases"] if args.case_limit is None else public["cases"][: args.case_limit]
    for case in selected_cases:
        video_path = case["query"]["rgb_video_path"]
        video_hashes.setdefault(video_path, sha256(Path(video_path)))
        if video_hashes[video_path] != case["query"]["rgb_video_sha256"]:
            raise ValueError(f"query video hash mismatch: {video_path}")
    outputs = []
    torch.cuda.reset_peak_memory_stats()
    for case in selected_cases:
        query = read_frame(case["query"]["rgb_video_path"], int(case["query"]["video_frame_index"]))
        target = case["target_specification"]
        exemplar = read_frame(target["exemplar_rgb_video_path"], int(target["exemplar_video_frame_index"]))
        model = YOLOE(str(args.model))
        started = time.perf_counter()
        result = model.predict(
            source=query,
            refer_image=exemplar,
            visual_prompts={"bboxes": [target["exemplar_bbox_xyxy"]], "cls": [0]},
            predictor=YOLOEVPSegPredictor,
            verbose=False,
            device=args.device,
            imgsz=640,
            conf=0.001,
            max_det=100,
        )[0]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ranked = sorted(
            zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )[:10]
        outputs.append({
            "case_id": case["case_id"],
            "latency_ms": elapsed_ms,
            "candidates": [
                {"rank": rank, "bbox_xyxy": [float(value) for value in box], "proposal_score": float(score), "source": "yoloe_visual_prompt"}
                for rank, (score, box) in enumerate(ranked, start=1)
            ],
        })
    atomic_json(args.output, {
        "schema_version": PREDICTION_SCHEMA,
        "protocol_id": public["protocol_id"],
        "public_input_sha256": sha256(args.public),
        "private_truth_access": False,
        "formal_run": args.case_limit is None,
        "case_limit": args.case_limit,
        "provider": {
            "name": "YOLOE-26n-seg visual prompt",
            "ultralytics_version": ultralytics.__version__,
            "model_path": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "device": args.device,
            "imgsz": 640,
            "confidence_floor": 0.001,
            "provider_max_det": 100,
            "bounded_pool_size": 10,
            "raw_pre_nms_proposals": "NOT_EXPOSED_BY_PROVIDER_INTERFACE",
            "threshold_or_configuration_sweep": False,
            "ranking": "PROVIDER_PROPOSAL_SCORE_DESCENDING",
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None,
        "cases": outputs,
        "forbidden_components_used": [],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
