#!/usr/bin/env python3
"""Run GT-blind YOLOE text-prompt proposals for a legal PA3 cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

from scripts.research.goal_copilot_bridge.p1_proposal_availability.authorize_pa3 import validate_execution_authorization

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_TEXT_ENCODER_SHA256,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--prompt-map", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--text-encoder", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--dispatch-journal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    output_path = args.output.resolve()
    journal_path = args.dispatch_journal.resolve()
    model_path = args.model.resolve()
    text_encoder_path = args.text_encoder.resolve()
    validate_execution_authorization(
        args.authorization,
        args.public,
        output_path,
        journal_path,
    )
    public = json.loads(args.public.read_text(encoding="utf-8"))
    prompt_map = json.loads(args.prompt_map.read_text(encoding="utf-8"))
    cases = validate_public(public, prompt_map, args.public.resolve().parent)
    if not model_path.is_file() or sha256(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("PA3 requires the frozen PA0-PA2 YOLOE-26n-seg checkpoint")
    if text_encoder_path.name != "mobileclip2_b.ts" or not text_encoder_path.is_file() or sha256(text_encoder_path) != EXPECTED_TEXT_ENCODER_SHA256:
        raise ValueError("PA3 requires the frozen MobileCLIP2 text encoder")

    import torch
    import ultralytics
    from ultralytics import YOLOE

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise ValueError("PA3 requires the frozen PA0-PA2 Ultralytics provider version")
    uses_cuda = args.device.startswith("cuda")
    if uses_cuda and not torch.cuda.is_available():
        raise ValueError("requested PA3 CUDA device is unavailable")
    model = YOLOE(str(model_path))
    if uses_cuda:
        torch.cuda.reset_peak_memory_stats()
    journal: dict[str, Any] = {
        "schema_version": "blindassist_p1_pa3_dispatch_journal_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "STARTED",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_input_sha256": sha256(args.public),
        "authorization_receipt_sha256": sha256(args.authorization),
        "prediction_output_path": str(output_path),
        "provider_model_calls_dispatched": 0,
        "provider_model_calls_completed": 0,
        "retry_or_replay_authorized": False,
    }
    atomic_json(journal_path, journal)
    outputs = []
    try:
        for case in cases:
            journal.update({
                "status": "DISPATCHING",
                "active_case_id": case["case_id"],
                "provider_model_calls_dispatched": journal["provider_model_calls_dispatched"] + 1,
            })
            atomic_json(journal_path, journal)
            previous_directory = Path.cwd()
            os.chdir(text_encoder_path.parent)
            try:
                model.set_classes([case["canonical_prompt"]])
            finally:
                os.chdir(previous_directory)
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
            journal.update({
                "status": "ACTIVE",
                "active_case_id": None,
                "provider_model_calls_completed": journal["provider_model_calls_completed"] + 1,
            })
            atomic_json(journal_path, journal)
        prediction = {
            "schema_version": PREDICTION_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "public_input_sha256": sha256(args.public),
            "prompt_map_sha256": content_sha256(prompt_map),
            "execution_authorization_sha256": sha256(args.authorization),
            "dispatch_journal_path": str(journal_path),
            "private_truth_access": False,
            "provider": {
                "name": "YOLOE-26n-seg goal-semantic text prompt",
                "ultralytics_version": ultralytics.__version__,
                "model_path": str(model_path),
                "model_sha256": sha256(model_path),
                "text_encoder_path": str(text_encoder_path),
                "text_encoder_sha256": sha256(text_encoder_path),
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
        }
        atomic_json(output_path, prediction)
        journal.update({
            "status": "COMPLETED",
            "active_case_id": None,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "prediction_sha256": sha256(output_path),
        })
        atomic_json(journal_path, journal)
    except BaseException as error:
        journal.update({
            "status": "FAILED_SEALED",
            "active_case_id": journal.get("active_case_id"),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failure_class": error.__class__.__name__,
            "failure_message": str(error),
            "provider_model_calls_in_doubt": journal["provider_model_calls_dispatched"] - journal["provider_model_calls_completed"],
        })
        atomic_json(journal_path, journal)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
