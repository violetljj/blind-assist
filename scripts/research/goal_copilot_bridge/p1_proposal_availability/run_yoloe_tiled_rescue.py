#!/usr/bin/env python3
"""Run the frozen P1-PA1 2x2 tiled YOLOE visual-prompt rescue arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any


PROTOCOL_ID = "P1-PA1-TARGET-PROPOSAL-RESCUE-V1"
PREDICTION_SCHEMA = "blindassist_p1_pa1_tiled_prediction_v1"
TILE_LAYOUT = "2x2"
TILE_OVERLAP = 0.20
IMAGE_SIZE = 640
CONFIDENCE_FLOOR = 0.001
PROVIDER_MAX_DET_PER_TILE = 100
GLOBAL_DEDUP_IOU = 0.50
BOUNDED_POOL_SIZE = 10


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


def two_by_two_tiles(image, overlap_fraction: float = TILE_OVERLAP):
    height, width = image.shape[:2]
    tile_width = min(width, math.ceil(width / (2.0 - overlap_fraction)))
    tile_height = min(height, math.ceil(height / (2.0 - overlap_fraction)))
    origins = [(0, 0), (width - tile_width, 0), (0, height - tile_height), (width - tile_width, height - tile_height)]
    return [(image[y:y + tile_height, x:x + tile_width], x, y) for x, y in origins]


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def deduplicate(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    decisions = []
    ordered = sorted(candidates, key=lambda row: (-float(row["proposal_score"]), str(row["candidate_id"])))
    for candidate in ordered:
        suppressor = next((earlier for earlier in kept if iou(candidate["bbox_xyxy"], earlier["bbox_xyxy"]) >= GLOBAL_DEDUP_IOU), None)
        decisions.append({
            "candidate_id": candidate["candidate_id"],
            "retained": suppressor is None,
            "suppressed_by_candidate_id": None if suppressor is None else suppressor["candidate_id"],
        })
        if suppressor is None:
            kept.append(candidate)
    return kept, decisions


def validate_manifest(manifest_path: Path, public_path: Path, model_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("PA1 manifest protocol mismatch")
    if manifest["inputs"]["public_input_sha256"] != sha256(public_path):
        raise ValueError("PA1 public input hash mismatch")
    if manifest["provider"]["model_sha256"] != sha256(model_path):
        raise ValueError("PA1 model hash mismatch")
    expected = {
        "tile_layout": TILE_LAYOUT,
        "tile_overlap_fraction": TILE_OVERLAP,
        "imgsz": IMAGE_SIZE,
        "confidence_floor": CONFIDENCE_FLOOR,
        "provider_max_det_per_tile": PROVIDER_MAX_DET_PER_TILE,
        "global_dedup_iou": GLOBAL_DEDUP_IOU,
        "bounded_pool_size": BOUNDED_POOL_SIZE,
    }
    if manifest["provider"]["configuration"] != expected:
        raise ValueError("PA1 provider configuration drift")
    if manifest["implementation"]["provider_source_sha256"] != sha256(Path(__file__)):
        raise ValueError("PA1 provider source drift")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-limit", type=int, help="GT-blind mechanics smoke only; omit for formal execution")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

    manifest = validate_manifest(args.manifest, args.public, args.model)
    public = json.loads(args.public.read_text(encoding="utf-8"))
    selected_cases = public["cases"] if args.case_limit is None else public["cases"][: args.case_limit]
    if args.case_limit is None and args.output.exists():
        raise ValueError("formal PA1 prediction output already exists")
    video_hashes: dict[str, str] = {}
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
        tiles = two_by_two_tiles(query)
        model = YOLOE(str(args.model))
        started = time.perf_counter()
        results = model.predict(
            source=[tile for tile, _, _ in tiles],
            refer_image=exemplar,
            visual_prompts={"bboxes": [target["exemplar_bbox_xyxy"]], "cls": [0]},
            predictor=YOLOEVPSegPredictor,
            verbose=False,
            device=args.device,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_FLOOR,
            max_det=PROVIDER_MAX_DET_PER_TILE,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        mapped = []
        tile_trace = []
        frame_height, frame_width = query.shape[:2]
        for tile_index, (result, (_, origin_x, origin_y)) in enumerate(zip(results, tiles, strict=True)):
            tile_candidates = []
            ranked = sorted(
                zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True),
                key=lambda pair: pair[0],
                reverse=True,
            )
            for tile_rank, (score, box) in enumerate(ranked, start=1):
                mapped_box = [
                    max(0.0, min(float(frame_width), float(box[0] + origin_x))),
                    max(0.0, min(float(frame_height), float(box[1] + origin_y))),
                    max(0.0, min(float(frame_width), float(box[2] + origin_x))),
                    max(0.0, min(float(frame_height), float(box[3] + origin_y))),
                ]
                if mapped_box[2] <= mapped_box[0] or mapped_box[3] <= mapped_box[1]:
                    continue
                candidate = {
                    "candidate_id": f"{case['case_id']}-tile-{tile_index}-rank-{tile_rank:03d}",
                    "bbox_xyxy": mapped_box,
                    "proposal_score": float(score),
                    "source": "yoloe_visual_prompt_tiled",
                    "tile_index": tile_index,
                    "tile_rank": tile_rank,
                }
                mapped.append(candidate)
                tile_candidates.append(candidate)
            tile_trace.append({
                "tile_index": tile_index,
                "origin_xy": [origin_x, origin_y],
                "tile_width": int(tiles[tile_index][0].shape[1]),
                "tile_height": int(tiles[tile_index][0].shape[0]),
                "provider_postprocessed_candidates": tile_candidates,
            })
        full_rank, dedup_decisions = deduplicate(mapped)
        raw_candidates = [{**candidate, "full_rank": rank} for rank, candidate in enumerate(full_rank, start=1)]
        final = raw_candidates[:BOUNDED_POOL_SIZE]
        outputs.append({
            "case_id": case["case_id"],
            "latency_ms": elapsed_ms,
            "inference_images": len(tiles),
            "tile_candidate_trace": tile_trace,
            "pre_dedup_candidate_count": len(mapped),
            "dedup_decisions": dedup_decisions,
            "raw_candidates": raw_candidates,
            "bounded_pool_decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "full_rank": candidate["full_rank"],
                    "retained": candidate["full_rank"] <= BOUNDED_POOL_SIZE,
                    "removal_reason": None if candidate["full_rank"] <= BOUNDED_POOL_SIZE else "BOUNDED_K10_CAP",
                }
                for candidate in raw_candidates
            ],
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "rank": rank,
                    "bbox_xyxy": candidate["bbox_xyxy"],
                    "proposal_score": candidate["proposal_score"],
                    "source": candidate["source"],
                    "tile_index": candidate["tile_index"],
                    "tile_rank": candidate["tile_rank"],
                }
                for rank, candidate in enumerate(final, start=1)
            ],
        })
    atomic_json(args.output, {
        "schema_version": PREDICTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": sha256(args.manifest),
        "public_input_sha256": sha256(args.public),
        "private_truth_access": False,
        "formal_run": args.case_limit is None,
        "case_limit": args.case_limit,
        "provider": {
            "name": "YOLOE-26n-seg visual prompt fixed 2x2 tiled rescue",
            "ultralytics_version": ultralytics.__version__,
            "model_path": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "device": args.device,
            **manifest["provider"]["configuration"],
            "ranking": "PROVIDER_PROPOSAL_SCORE_DESCENDING_THEN_CANDIDATE_ID",
            "raw_candidate_stage": "PER_TILE_PROVIDER_POSTPROCESSED_THEN_GLOBAL_DEDUP_BEFORE_BOUNDED_K10",
            "pre_nms_candidates": "NOT_EXPOSED_BY_PROVIDER_INTERFACE",
            "threshold_or_configuration_sweep": False,
        },
        "inference_images": sum(case["inference_images"] for case in outputs),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None,
        "cases": outputs,
        "forbidden_components_used": [],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
