#!/usr/bin/env python3
"""Run the frozen DINOv-SwinL R5 teacher on the 97-frame R4 cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from mine_goal_episodes import sha256
from run_rgb_observer import deduplicate_candidates


DINOV_SOURCE_REVISION = "53bf20d5cfdbb86fa35141a1cff432d4923599f2"
CHECKPOINT_SHA256 = "167fec1f006af8d2d53c662290dd2dff8e667aa66c8c0836af1181533d334a9a"
CHECKPOINT_BYTES = 902_781_487
IMAGE_SIZE = 1408
EXEMPLAR_CONFIDENCE_FLOOR = 0.60
EXEMPLAR_COUNT = 5
FORMAL_FRAME_RANGES = ((2039, 2065), (3222, 3271), (3292, 3311))
INTERNAL_SCORE_FILTER = "official_demo_0.12_with_0.04_adaptive_fallback"
FRAME_TOP_K = 100
GLOBAL_NMS_IOU = 0.70
CHECKPOINT_INTERVAL = 5


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def formal_frame_indices() -> list[int]:
    result = [index for start, end in FORMAL_FRAME_RANGES for index in range(start, end + 1)]
    if len(result) != 97 or len(result) != len(set(result)):
        raise AssertionError("frozen R5 cohort must contain exactly 97 unique frames")
    return result


def select_trusted_exemplars(observations: dict) -> list[dict]:
    """Choose five RGB-only detector boxes before the first LOST transition."""
    events = observations["events"]
    acquired = next(row["frame_index"] for row in events if row["event"] == "ACQUIRED")
    lost = next(row["frame_index"] for row in events if row["event"] == "LOST" and row["frame_index"] > acquired)
    candidates = [
        row
        for row in observations["frames"][acquired:lost]
        if row["observation_source"] == "detector"
        and row["bbox_xyxy"] is not None
        and float(row["target_confidence"]) >= EXEMPLAR_CONFIDENCE_FLOOR
    ]
    ranked = sorted(candidates, key=lambda row: (-float(row["target_confidence"]), int(row["frame_index"])))
    if len(ranked) < EXEMPLAR_COUNT:
        raise ValueError("first acquired segment has fewer than five frozen trusted exemplars")
    return [
        {
            "frame_index": int(row["frame_index"]),
            "bbox_xyxy": [float(value) for value in row["bbox_xyxy"]],
            "confidence": float(row["target_confidence"]),
            "selection_rule": "top5_detector_confidence_in_first_acquired_segment_with_0.60_floor",
        }
        for row in ranked[:EXEMPLAR_COUNT]
    ]


def load_frames(capture: Any, frame_indices: list[int]) -> dict[int, Any]:
    """Decode ordered contiguous ranges once so GPU inference never waits on repeated MP4 seeks."""
    import cv2

    result = {}
    previous = None
    for frame_index in sorted(set(frame_indices)):
        if previous is None or frame_index != previous + 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"could not read source frame {frame_index}")
        result[frame_index] = image
        previous = frame_index
    return result


def resize_rgb_tensor(image_bgr, image_size: int, torch):
    import cv2
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    resized = transforms.Resize(image_size, interpolation=Image.BICUBIC)(image)
    array = np.asarray(resized)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).cuda(non_blocking=True)
    return tensor, resized.height, resized.width


def scaled_box_mask(box: list[float], source_width: int, source_height: int, width: int, height: int, torch):
    import numpy as np

    scale_x, scale_y = width / source_width, height / source_height
    x1 = max(0, min(width - 1, int(np.floor(box[0] * scale_x))))
    y1 = max(0, min(height - 1, int(np.floor(box[1] * scale_y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(box[2] * scale_x))))
    y2 = max(y1 + 1, min(height, int(np.ceil(box[3] * scale_y))))
    mask = torch.zeros((1, height, width), dtype=torch.uint8)
    mask[:, y1:y2, x1:x2] = 255
    return mask


def inverse_sigmoid(value, torch, eps: float = 1e-5):
    value = value.clamp(min=0, max=1)
    return torch.log(value.clamp(min=eps) / (1 - value).clamp(min=eps))


def encode_exemplars(model, images: dict[int, Any], exemplars: list[dict], source_size: dict, torch):
    content = []
    last_attention = None
    last_padded = None
    for exemplar in sorted(exemplars, key=lambda row: row["frame_index"]):
        image = images[exemplar["frame_index"]]
        tensor, height, width = resize_rgb_tensor(image, IMAGE_SIZE, torch)
        target = scaled_box_mask(
            exemplar["bbox_xyxy"], source_size["width"], source_size["height"], width, height, torch
        )
        frame = {
            "image": tensor,
            "height": height,
            "width": width,
            "targets": [{"rand_shape": target, "pb": torch.tensor([1.0])}],
        }
        features, _, padded_h, padded_w = model.model.get_encoder_feature([frame])
        query, _, attention = model.model.get_visual_prompt_content_feature(
            features, target, padded_h, padded_w
        )
        content.append(query)
        last_attention = attention
        last_padded = (padded_h, padded_w)
    averaged = torch.stack(content).mean(0)
    point_coords = torch.ones(1, 4, device="cuda", dtype=torch.float32)
    point_coords[:, :2] = 0.0
    initial_box = inverse_sigmoid(point_coords[None], torch)
    return averaged, initial_box, last_attention, last_padded


def masks_to_candidates(raw_masks, raw_scores, source_width: int, source_height: int) -> list[dict]:
    masks = raw_masks.detach().float().cpu()
    while masks.ndim > 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim == 2:
        masks = masks[None]
    scores = raw_scores.detach().float().flatten().cpu().tolist()
    if masks.ndim != 3 or len(scores) != len(masks):
        raise ValueError(f"unexpected DINOv output shapes masks={tuple(masks.shape)} scores={len(scores)}")
    mask_height, mask_width = masks.shape[-2:]
    scale_x, scale_y = source_width / mask_width, source_height / mask_height
    candidates = []
    for score, mask in zip(scores, masks, strict=True):
        coordinates = (mask > 0).nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            continue
        y1, x1 = coordinates.min(dim=0).values.tolist()
        y2, x2 = coordinates.max(dim=0).values.tolist()
        candidates.append({
            "bbox_xyxy": [x1 * scale_x, y1 * scale_y, (x2 + 1) * scale_x, (y2 + 1) * scale_y],
            "confidence": float(score),
        })
    return sorted(
        deduplicate_candidates(candidates, overlap_threshold=GLOBAL_NMS_IOU),
        key=lambda candidate: candidate["confidence"],
        reverse=True,
    )[:FRAME_TOP_K]


def infer_frame(model, image, prompt, source_size: dict, torch) -> list[dict]:
    tensor, height, width = resize_rgb_tensor(image, IMAGE_SIZE, torch)
    target = {"image": tensor, "height": height, "width": width}
    features, mask_features, padded_h, padded_w = model.model.get_encoder_feature([target])
    content, initial_box, attention, _ = prompt
    masks, _, original_masks, scores = model.model.evaluate_demo_content_openset_multi_with_content_features(
        [target], mask_features, features, content, initial_box, attention, padded_h, padded_w
    )
    usable_masks = original_masks if original_masks is not None else masks
    return masks_to_candidates(usable_masks, scores, source_size["width"], source_size["height"])


def build_output(args, observations, exemplars, frames, processed, elapsed, peak, checkpoint_hash, checkpoint_bytes):
    return {
        "schema_version": "ba_adt_visual_upper_bound_r5_teacher_v3",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT1_SMALL_TARGET_VISUAL_UPPER_BOUND_R5",
        "attempt": "ATTEMPT_03_DINOV_SWINL",
        "teacher": {
            "name": "DINOv Swin-L",
            "implementation": "official-UX-Decoder-DINOv-visual-in-context-demo-path",
            "source_revision": DINOV_SOURCE_REVISION,
            "checkpoint_path_name": args.checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": checkpoint_bytes,
            "dtype": "float16_autocast",
            "image_size": IMAGE_SIZE,
            "visual_exemplars": exemplars,
            "visual_exemplar_count": len(exemplars),
            "visual_exemplar_geometry": "full_image_binary_box_mask",
            "prompt_aggregation": "mean_of_five_content_embeddings",
            "candidate_source": "positive_logit_predicted_masks_converted_to_boxes",
            "internal_score_filter": INTERNAL_SCORE_FILTER,
            "global_nms_iou": GLOBAL_NMS_IOU,
            "frame_top_k": FRAME_TOP_K,
            "threshold_or_geometry_sweep": False,
        },
        "inputs": {
            "video_path_name": args.video.name,
            "video_sha256": sha256(args.video),
            "observations_path_name": args.observations.name,
            "observations_sha256": sha256(args.observations),
        },
        "groundtruth_argument_supported": False,
        "future_location_or_visibility_input_supported": False,
        "formal_run": args.mechanics_frame is None and set(processed) == set(formal_frame_indices()),
        "mechanics_frame": args.mechanics_frame,
        "frozen_cohort_source": "consumed_R4_W2_W3_W4_eligibility_ranges",
        "frozen_frame_ranges_inclusive": [list(pair) for pair in FORMAL_FRAME_RANGES],
        "source_frame_size": observations["frame_size"],
        "source_frame_count_expected": len(observations["frames"]),
        "frame_count": len(frames),
        "processed_frame_indices": processed,
        "processed_frame_count": len(processed),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak["allocated"],
        "peak_cuda_reserved_bytes": peak["reserved"],
        "frames": frames,
        "claim_ceiling": "consumed_development_teacher_capability_only_no_edge_product_or_safety_claim",
    }


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--dinov-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanics-frame", type=int, help="Outcome-blind smoke frame outside the frozen cohort")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    checkpoint_hash = sha256(args.checkpoint)
    checkpoint_bytes = args.checkpoint.stat().st_size
    if CHECKPOINT_SHA256 != "PENDING_MECHANICAL_PREFLIGHT" and checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("DINOv checkpoint hash mismatch")
    if CHECKPOINT_BYTES and checkpoint_bytes != CHECKPOINT_BYTES:
        raise ValueError("DINOv checkpoint length mismatch")
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    if observations.get("groundtruth_argument_supported") is not False:
        raise ValueError("R5 teacher requires an RGB-only observation lineage")
    exemplars = select_trusted_exemplars(observations)
    formal_indices = formal_frame_indices()
    if args.mechanics_frame is not None and args.mechanics_frame in formal_indices:
        raise ValueError("mechanics frame must be outside the frozen 97-frame cohort")
    requested = formal_indices if args.mechanics_frame is None else [args.mechanics_frame]
    frames = [{"frame_index": index, "candidates": []} for index in range(len(observations["frames"]))]
    processed: list[int] = []
    elapsed_before = 0.0
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("attempt") != "ATTEMPT_03_DINOV_SWINL" or prior["teacher"]["source_revision"] != DINOV_SOURCE_REVISION:
            raise ValueError("resume identity mismatch")
        if prior["inputs"]["video_sha256"] != sha256(args.video) or prior["inputs"]["observations_sha256"] != sha256(args.observations):
            raise ValueError("resume input hash mismatch")
        if prior["teacher"]["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("resume checkpoint hash mismatch")
        frames = prior["frames"]
        processed = [int(value) for value in prior["processed_frame_indices"]]
        elapsed_before = float(prior["elapsed_seconds"])

    source = args.dinov_source.resolve()
    if not source.is_dir():
        raise ValueError("DINOv source directory does not exist")
    sys.path.insert(0, str(source))
    os.chdir(source)
    import torch
    from dinov.BaseModel import BaseModel
    from dinov import build_model
    from utils.arguments import load_opt_from_config_file

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    config = source / "configs" / "dinov_sam_coco_swinl_train.yaml"
    options = load_opt_from_config_file(str(config))
    model = BaseModel(options, build_model(options)).from_pretrained(str(args.checkpoint)).eval().cuda()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("could not open source video")
    try:
        images = load_frames(capture, [row["frame_index"] for row in exemplars] + requested)
    finally:
        capture.release()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        prompt = encode_exemplars(model, images, exemplars, observations["frame_size"], torch)
        for ordinal, frame_index in enumerate(requested):
            if frame_index in processed:
                continue
            frames[frame_index]["candidates"] = infer_frame(
                model, images[frame_index], prompt, observations["frame_size"], torch
            )
            processed.append(frame_index)
            elapsed = elapsed_before + time.perf_counter() - started
            peak = {
                "allocated": int(torch.cuda.max_memory_allocated()),
                "reserved": int(torch.cuda.max_memory_reserved()),
            }
            if len(processed) % CHECKPOINT_INTERVAL == 0 or ordinal == len(requested) - 1:
                payload = build_output(
                    args, observations, exemplars, frames, processed, elapsed, peak, checkpoint_hash, checkpoint_bytes
                )
                atomic_json(args.output, payload)
                print(json.dumps({
                    "processed": len(processed), "expected": len(requested), "frame_index": frame_index,
                    "elapsed_seconds": elapsed, "peak_reserved_bytes": peak["reserved"],
                }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
