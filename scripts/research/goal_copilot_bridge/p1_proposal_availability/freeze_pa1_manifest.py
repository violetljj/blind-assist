#!/usr/bin/env python3
"""Freeze the single-arm P1-PA1 tiled-rescue manifest before prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import run_yoloe_tiled_rescue as provider


SCHEMA = "blindassist_p1_pa1_tiled_manifest_v1"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--pa0-prediction", type=Path, required=True)
    parser.add_argument("--pa0-evaluation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider-source", type=Path, required=True)
    parser.add_argument("--evaluator-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("PA1 manifest already exists")
    public = json.loads(args.public.read_text(encoding="utf-8"))
    private = json.loads(args.private.read_text(encoding="utf-8"))
    if private["public_input_sha256"] != sha256(args.public):
        raise ValueError("PA1 source public/private binding mismatch")
    parent_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    atomic_json(args.output, {
        "schema_version": SCHEMA,
        "protocol_id": provider.PROTOCOL_ID,
        "claim_role": "POST_OUTCOME_SELECTED_CONSUMED_DEVELOPMENT_MECHANISM_DIAGNOSTIC_ONLY",
        "parent_commit": parent_commit,
        "inputs": {
            "public_input_sha256": sha256(args.public),
            "private_eval_input_sha256": sha256(args.private),
            "sealed_pa0_prediction_sha256": sha256(args.pa0_prediction),
            "sealed_pa0_evaluation_sha256": sha256(args.pa0_evaluation),
            "cases": len(public["cases"]),
        },
        "single_variable": "REPLACE_FULL_FRAME_640_WITH_FIXED_2X2_OVERLAP20_TILE_TO_640_SEARCH",
        "provider": {
            "name": "YOLOE-26n-seg visual prompt fixed 2x2 tiled rescue",
            "model_sha256": sha256(args.model),
            "configuration": {
                "tile_layout": provider.TILE_LAYOUT,
                "tile_overlap_fraction": provider.TILE_OVERLAP,
                "imgsz": provider.IMAGE_SIZE,
                "confidence_floor": provider.CONFIDENCE_FLOOR,
                "provider_max_det_per_tile": provider.PROVIDER_MAX_DET_PER_TILE,
                "global_dedup_iou": provider.GLOBAL_DEDUP_IOU,
                "bounded_pool_size": provider.BOUNDED_POOL_SIZE,
            },
            "ranking": "PROVIDER_PROPOSAL_SCORE_DESCENDING_THEN_CANDIDATE_ID",
            "sweep": False,
        },
        "primary_endpoint": {
            "metric": "TARGET_VISIBLE_RECALL_AT_10",
            "correct_iou_threshold": 0.30,
            "signal": "STRICTLY_ABOVE_0_OF_7",
        },
        "diagnostics": [
            "RECALL_AT_1_3_5_10_FOR_IOU_0.10_0.30_0.50",
            "FIRST_CORRECT_RANK",
            "SEALED_PA0_ABSENT_CASES_RESCUED",
            "FULL_RANK_PRESENT_BUT_K10_ABSENT",
            "PRE_AND_POST_GLOBAL_DEDUP_COUNTS",
            "INFERENCE_IMAGES_LATENCY_AND_CUDA_MEMORY",
        ],
        "adjudication": {
            "primary_nonzero": "P1_PA1_FIXED_TILED_SCALE_RESCUE_SIGNAL_ON_FAILURE_COHORT",
            "primary_zero": "P1_PA1_FIXED_TILED_SCALE_RESCUE_NOT_SUPPORTED_ON_FAILURE_COHORT",
            "pre_nms_attribution": "NOT_EVALUABLE_PROVIDER_INTERFACE",
        },
        "implementation": {
            "provider_source_sha256": sha256(args.provider_source),
            "evaluator_source_sha256": sha256(args.evaluator_source),
        },
        "forbidden": [
            "PARENT_SEMANTICS_OR_PARENT_PROVIDER",
            "VERIFIER_OR_IDENTITY_SELECTION",
            "MEMORY_OR_REACQUISITION",
            "VLM_VIO_SLAM_GEOMETRY",
            "THRESHOLD_TILE_OVERLAP_RESOLUTION_OR_K_SWEEP",
            "ANDROID_OR_DEFAULT_APP",
        ],
        "claim_ceiling": "FAILURE_COHORT_SCALE_RESCUE_MECHANISM_ONLY_NO_MODEL_SELECTION_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
