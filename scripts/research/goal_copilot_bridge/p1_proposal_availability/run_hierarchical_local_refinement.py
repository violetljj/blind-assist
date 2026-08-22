#!/usr/bin/env python3
"""Refine frozen HRG0 coarse regions with crop-local functional grounding."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from PIL import Image

from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import content_sha256, sha256, validate_public, validated_box
from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_hierarchical_functional_context import (
    PREDICTION_SCHEMA as HRG0_PREDICTION_SCHEMA,
    PROTOCOL_ID as HRG0_PROTOCOL_ID,
)


PROTOCOL_ID = "P1-HRG1-COARSE-TO-LOCAL-FUNCTIONAL-REFINEMENT-V1"
PREDICTION_SCHEMA = "blindassist_p1_hrg1_prediction_v1"
PARENT_REGION_POOL_SIZE = 5
LOCAL_CANDIDATES_PER_PARENT = 2
BOUNDED_POOL_SIZE = PARENT_REGION_POOL_SIZE * LOCAL_CANDIDATES_PER_PARENT


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def map_local_box(local_box: Sequence[float], crop_xyxy: Sequence[int]) -> list[float]:
    box = validated_box(list(local_box), "HRG1 local box")
    return [box[0] + crop_xyxy[0], box[1] + crop_xyxy[1], box[2] + crop_xyxy[0], box[3] + crop_xyxy[1]]


def _crop_box(parent_box: Sequence[float], width: int, height: int) -> list[int]:
    box = validated_box(list(parent_box), "HRG1 parent region")
    crop = [
        max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1]))),
        min(width, int(math.ceil(box[2]))), min(height, int(math.ceil(box[3]))),
    ]
    if crop[2] - crop[0] < 2 or crop[3] - crop[1] < 2:
        raise ValueError("HRG1 parent crop is degenerate")
    return crop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--prompt-map", required=True, type=Path)
    parser.add_argument("--parent-prediction", required=True, type=Path)
    parser.add_argument("--grounding-dino-model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("HRG1 prediction already exists; refusing replay")
    public = json.loads(args.public.read_text(encoding="utf-8"))
    prompt_map = json.loads(args.prompt_map.read_text(encoding="utf-8"))
    cases = validate_public(public, prompt_map, args.public.resolve().parent)
    parent = json.loads(args.parent_prediction.read_text(encoding="utf-8"))
    if parent.get("schema_version") != HRG0_PREDICTION_SCHEMA or parent.get("protocol_id") != HRG0_PROTOCOL_ID:
        raise ValueError("HRG1 parent prediction contract mismatch")
    if parent.get("public_input_sha256") != sha256(args.public) or parent.get("private_truth_access") is not False:
        raise ValueError("HRG1 parent prediction public/private binding mismatch")
    parent_by_case = {row["case_id"]: row for row in parent["cases"]}
    if set(parent_by_case) != {case["case_id"] for case in cases}:
        raise ValueError("HRG1 parent/public case mismatch")

    crop_records: list[dict[str, Any]] = []
    with ExitStack() as stack:
        temporary_root = Path(stack.enter_context(TemporaryDirectory(prefix="blindassist-hrg1-")))
        for case in cases:
            with Image.open(case["image_path"]) as source:
                rgb = source.convert("RGB")
                width, height = rgb.size
                for parent_item in parent_by_case[case["case_id"]]["candidates"][:PARENT_REGION_POOL_SIZE]:
                    crop = _crop_box(parent_item["bbox_xyxy"], width, height)
                    crop_id = f"{case['case_id']}::parent-{parent_item['rank']:02d}"
                    crop_path = temporary_root / f"{len(crop_records):04d}.jpg"
                    rgb.crop(tuple(crop)).save(crop_path, format="JPEG", quality=95)
                    crop_records.append({
                        "id": crop_id, "path": str(crop_path), "image_sha256": sha256(crop_path),
                        "case_id": case["case_id"],
                        "crop_xyxy": crop, "parent_rank": parent_item["rank"],
                        "parent_semantic_supported": parent_item["semantic_supported"],
                    })
        inference, runtime = dino.run_inference(args.grounding_dino_model_dir.resolve(), crop_records)
        inferred = {row["image_id"]: row["proposals"] for row in inference}
        outputs = []
        for case in cases:
            rows = []
            records = sorted((row for row in crop_records if row["case_id"] == case["case_id"]), key=lambda row: row["parent_rank"])
            for record in records:
                for local_rank, item in enumerate(inferred[record["id"]][:LOCAL_CANDIDATES_PER_PARENT], start=1):
                    rows.append({
                        "bbox_xyxy": map_local_box(item["bbox_xyxy"], record["crop_xyxy"]),
                        "score": float(item["score"]), "label": str(item["label"]),
                        "parent_region_rank": record["parent_rank"], "local_provider_rank": local_rank,
                        "parent_semantic_supported": record["parent_semantic_supported"],
                        "source": "hrg1_crop_local_functional_refinement",
                    })
            outputs.append({
                "case_id": case["case_id"],
                "candidates": [{"rank": rank, **row} for rank, row in enumerate(rows[:BOUNDED_POOL_SIZE], start=1)],
                "parent_regions_processed": len(records),
            })
    _atomic_json(output, {
        "schema_version": PREDICTION_SCHEMA, "protocol_id": PROTOCOL_ID,
        "public_input_sha256": sha256(args.public), "prompt_map_sha256": content_sha256(prompt_map),
        "parent_prediction_sha256": sha256(args.parent_prediction), "private_truth_access": False,
        "provider": {
            "grounding_dino_repository": dino.MODEL_REPOSITORY, "grounding_dino_revision": dino.MODEL_REVISION,
            "grounding_dino_weights_sha256": dino.WEIGHTS_SHA256, "functional_prompt": dino.PROMPT,
            "functional_box_threshold": dino.BOX_THRESHOLD, "functional_text_threshold": dino.TEXT_THRESHOLD,
            "functional_nms_iou": dino.NMS_IOU_THRESHOLD,
            "parent_region_pool_size": PARENT_REGION_POOL_SIZE,
            "local_candidates_per_parent": LOCAL_CANDIDATES_PER_PARENT,
            "bounded_pool_size": BOUNDED_POOL_SIZE,
            "ranking": "PARENT_REGION_RANK_THEN_LOCAL_FUNCTIONAL_PROVIDER_RANK",
            "identity_selection": "FORBIDDEN", "threshold_prompt_model_or_pool_sweep": False,
        },
        "grounding_dino_runtime": runtime, "cases": outputs,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
