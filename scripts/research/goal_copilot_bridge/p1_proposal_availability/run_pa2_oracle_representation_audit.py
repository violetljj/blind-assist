#!/usr/bin/env python3
"""Run the consumed-Development P1-PA2 oracle representation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any


PROTOCOL_ID = "P1-PA2-TARGET-REPRESENTATION-OBSERVABILITY-AUDIT-V1"
SCHEMA_VERSION = "blindassist_p1_pa2_oracle_representation_audit_v1"
IMAGE_SIZE = 640
CONFIDENCE_FLOOR = 0.001
PROVIDER_MAX_DET = 100
BOUNDED_POOL_SIZE = 10
ORACLE_ROI_SCALE = 3.0
CONTEXT_PROMPT_SCALE = 2.0
IOU_THRESHOLDS = (0.10, 0.30, 0.50)


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


def validated_box(box: list[float], label: str) -> list[float]:
    values = [float(value) for value in box]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must be four finite coordinates")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"{label} must have positive area")
    return values


def expand_box(box: list[float], scale: float, image_width: int, image_height: int) -> list[float]:
    x1, y1, x2, y2 = validated_box(box, "box")
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_width, half_height = (x2 - x1) * scale / 2.0, (y2 - y1) * scale / 2.0
    return [
        max(0.0, center_x - half_width),
        max(0.0, center_y - half_height),
        min(float(image_width), center_x + half_width),
        min(float(image_height), center_y + half_height),
    ]


def integer_crop_box(box: list[float], image_width: int, image_height: int) -> list[int]:
    x1, y1, x2, y2 = validated_box(box, "crop box")
    crop = [
        max(0, int(math.floor(x1))),
        max(0, int(math.floor(y1))),
        min(image_width, int(math.ceil(x2))),
        min(image_height, int(math.ceil(y2))),
    ]
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError("integer crop has no area")
    return crop


def remap_box(box: list[float], crop_box: list[int]) -> list[float]:
    return [box[0] - crop_box[0], box[1] - crop_box[1], box[2] - crop_box[0], box[3] - crop_box[1]]


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def prepare_arm(query, target_box: list[float], arm: str):
    height, width = query.shape[:2]
    if arm == "exact_target_target_only":
        crop_box = integer_crop_box(target_box, width, height)
    elif arm in {"oracle_roi_target_only", "oracle_roi_target_plus_context"}:
        crop_box = integer_crop_box(expand_box(target_box, ORACLE_ROI_SCALE, width, height), width, height)
    else:
        raise ValueError(f"unknown arm: {arm}")
    x1, y1, x2, y2 = crop_box
    return query[y1:y2, x1:x2], crop_box, remap_box(target_box, crop_box)


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted(cases[0]["arms"]) if cases else []
    summary: dict[str, Any] = {}
    for arm in arms:
        rows = [case["arms"][arm] for case in cases]
        summary[arm] = {
            "case_count": len(rows),
            "mean_provider_postprocessed_candidates": sum(row["candidate_count"] for row in rows) / len(rows),
            "median_latency_ms": sorted(row["latency_ms"] for row in rows)[len(rows) // 2],
            "recall_full_rank": {
                str(threshold): sum(row["first_correct_rank"][str(threshold)] is not None for row in rows) / len(rows)
                for threshold in IOU_THRESHOLDS
            },
            "recall_at_10": {
                str(threshold): sum(
                    row["first_correct_rank"][str(threshold)] is not None
                    and row["first_correct_rank"][str(threshold)] <= BOUNDED_POOL_SIZE
                    for row in rows
                ) / len(rows)
                for threshold in IOU_THRESHOLDS
            },
        }
    return summary


def classify(summary: dict[str, Any]) -> str:
    exact = summary["exact_target_target_only"]["recall_full_rank"]["0.3"]
    roi = summary["oracle_roi_target_only"]["recall_full_rank"]["0.3"]
    context = summary["oracle_roi_target_plus_context"]["recall_full_rank"]["0.3"]
    if exact == 0.0 and roi == 0.0 and context == 0.0:
        return "A_CURRENT_VISUAL_PROMPT_REPRESENTATION_SIGNAL_NOT_OBSERVED"
    if roi > 0.0 and context <= roi:
        return "B_TARGET_SIGNAL_EXISTS_SEARCH_OR_LOCALIZATION_REMAINS_PLAUSIBLE"
    if context > roi:
        return "C_CONTEXT_CONDITIONED_PROPOSAL_SIGNAL_OBSERVED"
    return "MIXED_OR_INSUFFICIENT_REPRESENTATION_OBSERVABILITY_SIGNAL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

    public = json.loads(args.public.read_text(encoding="utf-8"))
    private = json.loads(args.private.read_text(encoding="utf-8"))
    if private.get("public_input_sha256") != sha256(args.public):
        raise ValueError("PA0 public/private identity mismatch")
    if len(public.get("cases", [])) != 7 or len(private.get("cases", [])) != 7:
        raise ValueError("PA2 requires the complete seven-case consumed cohort")
    if args.output.exists():
        raise ValueError("PA2 output already exists")

    truth = {case["case_id"]: case for case in private["cases"]}
    model = YOLOE(str(args.model))
    outputs = []
    torch.cuda.reset_peak_memory_stats()
    arm_names = ("exact_target_target_only", "oracle_roi_target_only", "oracle_roi_target_plus_context")
    for case in public["cases"]:
        case_id = case["case_id"]
        target_box = validated_box(truth[case_id]["target_bbox_xyxy"], f"{case_id} target")
        query = read_frame(case["query"]["rgb_video_path"], int(case["query"]["video_frame_index"]))
        target = case["target_specification"]
        exemplar = read_frame(target["exemplar_rgb_video_path"], int(target["exemplar_video_frame_index"]))
        exemplar_height, exemplar_width = exemplar.shape[:2]
        target_prompt = validated_box(target["exemplar_bbox_xyxy"], f"{case_id} exemplar")
        context_prompt = expand_box(target_prompt, CONTEXT_PROMPT_SCALE, exemplar_width, exemplar_height)
        arm_outputs = {}
        for arm in arm_names:
            source, crop_box, local_target = prepare_arm(query, target_box, arm)
            prompt_box = context_prompt if arm == "oracle_roi_target_plus_context" else target_prompt
            started = time.perf_counter()
            result = model.predict(
                source=source,
                refer_image=exemplar,
                visual_prompts={"bboxes": [prompt_box], "cls": [0]},
                predictor=YOLOEVPSegPredictor,
                verbose=False,
                device=args.device,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_FLOOR,
                max_det=PROVIDER_MAX_DET,
            )[0]
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            ranked = sorted(
                zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True),
                key=lambda pair: pair[0],
                reverse=True,
            )
            candidates = []
            for rank, (score, box) in enumerate(ranked, start=1):
                candidate_box = [float(value) for value in box]
                candidates.append({
                    "rank": rank,
                    "bbox_xyxy_in_crop": candidate_box,
                    "proposal_score": float(score),
                    "target_iou": iou(candidate_box, local_target),
                })
            arm_outputs[arm] = {
                "crop_box_xyxy_in_query": crop_box,
                "target_bbox_xyxy_in_crop": local_target,
                "prompt_bbox_xyxy_in_exemplar": prompt_box,
                "candidate_count": len(candidates),
                "latency_ms": elapsed_ms,
                "best_iou": max((candidate["target_iou"] for candidate in candidates), default=0.0),
                "first_correct_rank": {
                    str(threshold): next(
                        (candidate["rank"] for candidate in candidates if candidate["target_iou"] >= threshold), None
                    )
                    for threshold in IOU_THRESHOLDS
                },
                "provider_postprocessed_candidates": candidates,
            }
        outputs.append({"case_id": case_id, "arms": arm_outputs})

    summary = summarize(outputs)
    atomic_json(args.output, {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evidence_role": "POST_OUTCOME_SELECTED_CONSUMED_DEVELOPMENT_ORACLE_MECHANISM_DIAGNOSTIC_ONLY",
        "claim_ceiling": "FAILURE_COHORT_ORACLE_REPRESENTATION_OBSERVABILITY_ONLY_NO_MODEL_SELECTION_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM",
        "oracle_truth_access": True,
        "oracle_truth_use": "QUERY_CROP_CONSTRUCTION_AND_PRIVATE_EVALUATION_ONLY",
        "public_input_sha256": sha256(args.public),
        "private_eval_input_sha256": sha256(args.private),
        "implementation_sha256": sha256(Path(__file__)),
        "provider": {
            "name": "YOLOE-26n-seg visual prompt",
            "ultralytics_version": ultralytics.__version__,
            "model_path": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "device": args.device,
            "imgsz": IMAGE_SIZE,
            "confidence_floor": CONFIDENCE_FLOOR,
            "provider_max_det": PROVIDER_MAX_DET,
            "bounded_pool_size": BOUNDED_POOL_SIZE,
            "oracle_roi_scale": ORACLE_ROI_SCALE,
            "context_prompt_scale": CONTEXT_PROMPT_SCALE,
            "prompt_embedding_similarity": "NOT_EXPOSED_BY_PROVIDER_INTERFACE",
            "pre_nms_proposals": "NOT_EXPOSED_BY_PROVIDER_INTERFACE",
            "threshold_or_configuration_sweep": False,
        },
        "summary": summary,
        "diagnostic_branch": classify(summary),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None,
        "cases": outputs,
        "forbidden_components_used": [],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
