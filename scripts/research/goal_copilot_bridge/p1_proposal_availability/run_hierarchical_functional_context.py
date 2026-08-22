#!/usr/bin/env python3
"""Run frozen semantic-support plus functional-context hierarchical proposals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from PIL import Image

from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_TEXT_ENCODER_SHA256,
    EXPECTED_ULTRALYTICS_VERSION,
    content_sha256,
    sha256,
    validate_public,
    validated_box,
)


PROTOCOL_ID = "P1-HRG0-SEMANTIC-SUPPORTED-FUNCTIONAL-CONTEXT-V1"
PREDICTION_SCHEMA = "blindassist_p1_hrg0_prediction_v1"
CONTEXT_SCALE = 1.5
IMAGE_SIZE = 640
CONFIDENCE_FLOOR = 0.001
YOLOE_MAX_DET = 100
SEMANTIC_SUPPORT_POOL_SIZE = 10
BOUNDED_POOL_SIZE = 10


def expand_box(box: Sequence[float], *, scale: float, width: int, height: int) -> list[float]:
    source = validated_box(list(box), "functional source box")
    center_x = (source[0] + source[2]) / 2.0
    center_y = (source[1] + source[3]) / 2.0
    half_width = (source[2] - source[0]) * scale / 2.0
    half_height = (source[3] - source[1]) * scale / 2.0
    return [
        max(0.0, center_x - half_width), max(0.0, center_y - half_height),
        min(float(width), center_x + half_width), min(float(height), center_y + half_height),
    ]


def _contains(box: Sequence[float], point: tuple[float, float]) -> bool:
    return float(box[0]) <= point[0] <= float(box[2]) and float(box[1]) <= point[1] <= float(box[3])


def hierarchical_candidates(
    semantic: Sequence[Mapping[str, Any]],
    functional: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    semantic_pool = list(semantic[:SEMANTIC_SUPPORT_POOL_SIZE])
    rows = []
    for functional_rank, item in enumerate(functional, start=1):
        source_box = validated_box(list(item["bbox_xyxy"]), "functional proposal")
        center = ((source_box[0] + source_box[2]) / 2.0, (source_box[1] + source_box[3]) / 2.0)
        supporting = [
            rank for rank, candidate in enumerate(semantic_pool, start=1)
            if _contains(candidate["bbox_xyxy"], center)
        ]
        rows.append({
            "bbox_xyxy": expand_box(source_box, scale=CONTEXT_SCALE, width=width, height=height),
            "functional_source_bbox_xyxy": source_box,
            "functional_provider_rank": functional_rank,
            "functional_score": float(item["score"]),
            "functional_label": str(item["label"]),
            "semantic_supported": bool(supporting),
            "best_semantic_support_rank": min(supporting) if supporting else None,
            "context_scale": CONTEXT_SCALE,
            "source": "hrg0_semantic_supported_functional_context",
        })
    rows.sort(key=lambda row: (
        not row["semantic_supported"],
        row["functional_provider_rank"],
    ))
    return [{"rank": rank, **row} for rank, row in enumerate(rows[:BOUNDED_POOL_SIZE], start=1)]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--prompt-map", required=True, type=Path)
    parser.add_argument("--yoloe-model", required=True, type=Path)
    parser.add_argument("--text-encoder", required=True, type=Path)
    parser.add_argument("--grounding-dino-model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("HRG0 prediction already exists; refusing replay")
    public = json.loads(args.public.read_text(encoding="utf-8"))
    prompt_map = json.loads(args.prompt_map.read_text(encoding="utf-8"))
    cases = validate_public(public, prompt_map, args.public.resolve().parent)
    yoloe_path = args.yoloe_model.resolve()
    encoder_path = args.text_encoder.resolve()
    if not yoloe_path.is_file() or sha256(yoloe_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("HRG0 YOLOE checkpoint drift")
    if encoder_path.name != "mobileclip2_b.ts" or not encoder_path.is_file() or sha256(encoder_path) != EXPECTED_TEXT_ENCODER_SHA256:
        raise ValueError("HRG0 MobileCLIP2 text encoder drift")

    import torch
    import ultralytics
    from ultralytics import YOLOE

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise ValueError("HRG0 Ultralytics version drift")
    uses_cuda = args.device.startswith("cuda")
    if uses_cuda and not torch.cuda.is_available():
        raise ValueError("requested HRG0 CUDA device is unavailable")
    if uses_cuda:
        torch.cuda.reset_peak_memory_stats()
    yoloe = YOLOE(str(yoloe_path))
    semantic_by_case: dict[str, list[dict[str, Any]]] = {}
    yoloe_latency = {}
    for case in cases:
        previous_directory = Path.cwd()
        os.chdir(encoder_path.parent)
        try:
            yoloe.set_classes([case["canonical_prompt"]])
        finally:
            os.chdir(previous_directory)
        started = time.perf_counter()
        result = yoloe.predict(
            source=str(case["image_path"]), verbose=False, device=args.device,
            imgsz=IMAGE_SIZE, conf=CONFIDENCE_FLOOR, max_det=YOLOE_MAX_DET,
        )[0]
        if uses_cuda:
            torch.cuda.synchronize()
        yoloe_latency[case["case_id"]] = (time.perf_counter() - started) * 1000.0
        ranked = sorted(zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True), key=lambda pair: pair[0], reverse=True)
        semantic_by_case[case["case_id"]] = [
            {"bbox_xyxy": [float(value) for value in box], "score": float(score)} for score, box in ranked
        ]

    metadata = [{"id": case["case_id"], "path": str(case["image_path"]), "image_sha256": sha256(case["image_path"])} for case in cases]
    functional, dino_runtime = dino.run_inference(args.grounding_dino_model_dir.resolve(), metadata)
    functional_by_case = {row["image_id"]: row for row in functional}
    outputs = []
    for case in cases:
        with Image.open(case["image_path"]) as image:
            width, height = image.size
        functional_rows = functional_by_case[case["case_id"]]["proposals"]
        candidates = hierarchical_candidates(semantic_by_case[case["case_id"]], functional_rows, width=width, height=height)
        outputs.append({
            "case_id": case["case_id"], "candidates": candidates,
            "semantic_postprocessed_count": len(semantic_by_case[case["case_id"]]),
            "functional_postprocessed_count": len(functional_rows),
            "semantic_latency_ms": yoloe_latency[case["case_id"]],
        })
    _atomic_json(output, {
        "schema_version": PREDICTION_SCHEMA, "protocol_id": PROTOCOL_ID,
        "public_input_sha256": sha256(args.public), "prompt_map_sha256": content_sha256(prompt_map),
        "private_truth_access": False,
        "provider": {
            "yoloe_model_sha256": EXPECTED_MODEL_SHA256,
            "text_encoder_sha256": EXPECTED_TEXT_ENCODER_SHA256,
            "ultralytics_version": EXPECTED_ULTRALYTICS_VERSION,
            "grounding_dino_repository": dino.MODEL_REPOSITORY,
            "grounding_dino_revision": dino.MODEL_REVISION,
            "grounding_dino_weights_sha256": dino.WEIGHTS_SHA256,
            "functional_prompt": dino.PROMPT,
            "functional_box_threshold": dino.BOX_THRESHOLD,
            "functional_text_threshold": dino.TEXT_THRESHOLD,
            "functional_nms_iou": dino.NMS_IOU_THRESHOLD,
            "semantic_support_pool_size": SEMANTIC_SUPPORT_POOL_SIZE,
            "functional_context_scale": CONTEXT_SCALE,
            "bounded_pool_size": BOUNDED_POOL_SIZE,
            "ranking": "SEMANTIC_CENTER_SUPPORTED_FIRST_THEN_FUNCTIONAL_PROVIDER_RANK",
            "identity_selection": "FORBIDDEN",
            "threshold_prompt_model_or_scale_sweep": False,
        },
        "grounding_dino_runtime": dino_runtime,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if uses_cuda else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if uses_cuda else None,
        "cases": outputs,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
