#!/usr/bin/env python3
"""Run GT-blind YOLOE text-prompt proposals for a legal PA3 cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_ULTRALYTICS_VERSION,
    PREDICTION_SCHEMA,
    PROTOCOL_ID,
    content_sha256,
    sha256,
    validate_public,
)


IMAGE_SIZE = 640
CONFIDENCE_FLOOR = 0.001
PROVIDER_MAX_DET = 100
BOUNDED_POOL_SIZE = 10


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--prompt-map", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("PA3 output already exists; refusing replay")
    public = json.loads(args.public.read_text(encoding="utf-8"))
    prompt_map = json.loads(args.prompt_map.read_text(encoding="utf-8"))
    cases = validate_public(public, prompt_map, args.public.resolve().parent)
    if not args.model.is_file() or sha256(args.model) != EXPECTED_MODEL_SHA256:
        raise ValueError("PA3 requires the frozen PA0-PA2 YOLOE-26n-seg checkpoint")

    import torch
    import ultralytics
    from ultralytics import YOLOE

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise ValueError("PA3 requires the frozen PA0-PA2 Ultralytics provider version")
    uses_cuda = args.device.startswith("cuda")
    if uses_cuda and not torch.cuda.is_available():
        raise ValueError("requested PA3 CUDA device is unavailable")
    model = YOLOE(str(args.model))
    if uses_cuda:
        torch.cuda.reset_peak_memory_stats()
    outputs = []
    for case in cases:
        model.set_classes([case["canonical_prompt"]])
        started = time.perf_counter()
        result = model.predict(
            source=str(case["image_path"]),
            verbose=False,
            device=args.device,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_FLOOR,
            max_det=PROVIDER_MAX_DET,
        )[0]
        if uses_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ranked = sorted(
            zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )
        outputs.append({
            "case_id": case["case_id"],
            "latency_ms": elapsed_ms,
            "candidates": [
                {
                    "rank": rank,
                    "bbox_xyxy": [float(value) for value in box],
                    "proposal_score": float(score),
                    "source": "yoloe_goal_semantic_text_prompt",
                }
                for rank, (score, box) in enumerate(ranked[:BOUNDED_POOL_SIZE], start=1)
            ],
            "provider_postprocessed_candidate_count": len(ranked),
        })
    atomic_json(args.output, {
        "schema_version": PREDICTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "public_input_sha256": sha256(args.public),
        "prompt_map_sha256": content_sha256(prompt_map),
        "private_truth_access": False,
        "provider": {
            "name": "YOLOE-26n-seg goal-semantic text prompt",
            "ultralytics_version": ultralytics.__version__,
            "model_path": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "device": args.device,
            "imgsz": IMAGE_SIZE,
            "confidence_floor": CONFIDENCE_FLOOR,
            "provider_max_det": PROVIDER_MAX_DET,
            "bounded_pool_size": BOUNDED_POOL_SIZE,
            "identity_selection": "FORBIDDEN",
            "threshold_or_configuration_sweep": False,
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if uses_cuda else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if uses_cuda else None,
        "cases": outputs,
        "forbidden_components_used": [],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
