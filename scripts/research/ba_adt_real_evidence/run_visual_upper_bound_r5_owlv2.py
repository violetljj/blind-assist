#!/usr/bin/env python3
"""Run the frozen OWLv2-large R5 visual-query teacher without GT access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
from PIL import Image

from mine_goal_episodes import sha256
from run_rgb_observer import deduplicate_candidates, two_by_two_tiles
from run_visual_upper_bound_r5 import select_trusted_exemplar


MODEL_ID = "google/owlv2-large-patch14-ensemble"
MODEL_REVISION = "95e26936e865f87db1742128404b3c035d47d89d"
MODEL_SAFETENSORS_SHA256 = "d1c2261503c55aaf400667a843a54a5167e3c696334674c4093d6d10f7f40075"
MODEL_SAFETENSORS_BYTES = 1_750_520_144
TILE_OVERLAP = 0.20
TILE_TOP_K = 20
FRAME_TOP_K = 50
TILE_NMS_IOU = 0.30
GLOBAL_NMS_IOU = 0.50
CHECKPOINT_INTERVAL = 25
SEARCH_FRAMES_PER_BATCH = 2
TORCH_DTYPE = "bfloat16"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def crop_exemplar(video: Path, exemplar: dict) -> Image.Image:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, exemplar["frame_index"])
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise ValueError("could not read frozen exemplar frame")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = exemplar["bbox_xyxy"]
    left = max(0, min(width - 1, int(x1)))
    top = max(0, min(height - 1, int(y1)))
    right = max(left + 1, min(width, int(x2 + 0.999999)))
    bottom = max(top + 1, min(height, int(y2 + 0.999999)))
    crop = image[top:bottom, left:right]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def infer_search_frames(model, processor, query_pixels, pending: list[dict], device: str) -> None:
    """Fill candidates for up to two search frames in one eight-tile GPU batch."""
    import torch

    search = [entry for entry in pending if entry["search_active"]]
    if not search:
        return
    flat_tiles = []
    ownership = []
    for entry in search:
        frame_image = entry.pop("image")
        entry["_frame_height"], entry["_frame_width"] = frame_image.shape[:2]
        tiles = two_by_two_tiles(frame_image, TILE_OVERLAP)
        for tile_index, (tile, origin_x, origin_y) in enumerate(tiles):
            flat_tiles.append(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)))
            ownership.append((entry, tile_index, origin_x, origin_y, tile.shape[0], tile.shape[1]))
    inputs = processor(images=flat_tiles, return_tensors="pt")
    inputs["query_pixel_values"] = query_pixels.expand(len(flat_tiles), -1, -1, -1)
    inputs = {
        key: value.to(device, dtype=getattr(torch, TORCH_DTYPE)) if value.is_floating_point() else value.to(device)
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model.image_guided_detection(**inputs)
    results = processor.post_process_image_guided_detection(
        outputs,
        threshold=0.0,
        nms_threshold=TILE_NMS_IOU,
        target_sizes=torch.tensor(
            [(height, width) for _, _, _, _, height, width in ownership],
            device=outputs.logits.device,
        ),
    )
    collected = {entry["frame_index"]: [] for entry in search}
    for result, (entry, _, origin_x, origin_y, _, _) in zip(results, ownership, strict=True):
        pairs = zip(result["scores"].tolist(), result["boxes"].tolist(), strict=True)
        ranked = sorted(pairs, key=lambda pair: pair[0], reverse=True)[:TILE_TOP_K]
        for confidence, box in ranked:
            width, height = entry["_frame_width"], entry["_frame_height"]
            clipped = [max(0.0, min(float(width), float(box[0] + origin_x))),
                       max(0.0, min(float(height), float(box[1] + origin_y))),
                       max(0.0, min(float(width), float(box[2] + origin_x))),
                       max(0.0, min(float(height), float(box[3] + origin_y)))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            collected[entry["frame_index"]].append({
                "bbox_xyxy": clipped,
                "confidence": float(confidence),
            })
    for entry in search:
        entry["candidates"] = sorted(
            deduplicate_candidates(collected[entry["frame_index"]], overlap_threshold=GLOBAL_NMS_IOU),
            key=lambda candidate: candidate["confidence"], reverse=True,
        )[:FRAME_TOP_K]
        entry.pop("_frame_height")
        entry.pop("_frame_width")


def build_output(args, observations: dict, exemplar: dict, frames: list[dict], elapsed: float, peak: dict) -> dict:
    return {
        "schema_version": "ba_adt_visual_upper_bound_r5_teacher_v2",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT1_SMALL_TARGET_VISUAL_UPPER_BOUND_R5",
        "attempt": "ATTEMPT_02_OWLV2_LARGE",
        "teacher": {
            "name": "OWLv2 large patch14 ensemble",
            "implementation": "huggingface-transformers-slow-image-processor-cached-query",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
            "model_safetensors_bytes": MODEL_SAFETENSORS_BYTES,
            "dtype": TORCH_DTYPE,
            "native_image_size": 1008,
            "patch_size": 14,
            "visual_exemplar": exemplar,
            "visual_exemplar_count": 1,
            "search_schedule": "r1_rgb_only_target_visible_false_after_frozen_exemplar",
            "tile_layout": "2x2",
            "tile_overlap_fraction": TILE_OVERLAP,
            "tile_postprocess_threshold": 0.0,
            "tile_nms_iou": TILE_NMS_IOU,
            "tile_top_k": TILE_TOP_K,
            "global_nms_iou": GLOBAL_NMS_IOU,
            "frame_top_k": FRAME_TOP_K,
            "search_frames_per_gpu_batch": SEARCH_FRAMES_PER_BATCH,
            "maximum_tile_batch": SEARCH_FRAMES_PER_BATCH * 4,
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
        "formal_run": args.stop_after_frame is None,
        "stop_after_frame": args.stop_after_frame,
        "source_frame_size": observations["frame_size"],
        "source_frame_count_expected": len(observations["frames"]),
        "frame_count": len(frames),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak.get("allocated"),
        "peak_cuda_reserved_bytes": peak.get("reserved"),
        "frames": frames,
        "claim_ceiling": "consumed_development_teacher_capability_only_no_edge_product_or_safety_claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-after-frame", type=int, help="Mechanics smoke only; formal R5 must omit")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    if observations.get("groundtruth_argument_supported") is not False:
        raise ValueError("R5 teacher requires an RGB-only observation lineage")
    exemplar = select_trusted_exemplar(observations)
    video_hash = sha256(args.video)
    observations_hash = sha256(args.observations)
    frames: list[dict] = []
    elapsed_before = 0.0
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("attempt") != "ATTEMPT_02_OWLV2_LARGE":
            raise ValueError("resume attempt identity mismatch")
        if prior["inputs"]["video_sha256"] != video_hash or prior["inputs"]["observations_sha256"] != observations_hash:
            raise ValueError("resume input hash mismatch")
        if prior["teacher"]["model_revision"] != MODEL_REVISION:
            raise ValueError("resume model revision mismatch")
        frames = prior["frames"]
        elapsed_before = float(prior.get("elapsed_seconds", 0.0))

    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=False)
    model = Owlv2ForObjectDetection.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=getattr(torch, TORCH_DTYPE)
    ).eval().to(args.device)
    query = crop_exemplar(args.video, exemplar)
    query_pixels = processor(query_images=query, return_tensors="pt")["query_pixel_values"].to(
        args.device, dtype=getattr(torch, TORCH_DTYPE)
    )
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, len(frames))
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    inference_frames = 0
    expected = len(observations["frames"])
    pending: list[dict] = []
    stop_requested = False
    try:
        while len(frames) < expected:
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"video ended at frame {len(frames)} before expected {expected}")
            frame_index = len(frames) + len(pending)
            row = observations["frames"][frame_index]
            search_active = frame_index > exemplar["frame_index"] and not bool(row["target_visible"])
            entry = {"frame_index": frame_index, "search_active": search_active, "candidates": []}
            if search_active:
                entry["image"] = image
            pending.append(entry)
            search_pending = sum(row["search_active"] for row in pending)
            if args.stop_after_frame is not None and frame_index >= args.stop_after_frame:
                stop_requested = True
            if search_pending < SEARCH_FRAMES_PER_BATCH and len(frames) + len(pending) < expected and not stop_requested:
                continue
            infer_search_frames(model, processor, query_pixels, pending, args.device)
            inference_frames += search_pending
            for completed in pending:
                completed.pop("image", None)
                frames.append(completed)
            pending.clear()
            elapsed = elapsed_before + time.perf_counter() - started
            peak = {"allocated": int(torch.cuda.max_memory_allocated()), "reserved": int(torch.cuda.max_memory_reserved())}
            if len(frames) % CHECKPOINT_INTERVAL == 0:
                atomic_json(args.output, build_output(args, observations, exemplar, frames, elapsed, peak))
                rate = len(frames) / max(elapsed, 1e-9)
                print(json.dumps({"frames": len(frames), "expected": expected,
                                  "inference_frames_this_process": inference_frames,
                                  "elapsed_seconds": elapsed,
                                  "eta_seconds": (expected - len(frames)) / rate}), flush=True)
            if stop_requested:
                break
    finally:
        capture.release()

    elapsed = elapsed_before + time.perf_counter() - started
    peak = {"allocated": int(torch.cuda.max_memory_allocated()), "reserved": int(torch.cuda.max_memory_reserved())}
    output = build_output(args, observations, exemplar, frames, elapsed, peak)
    atomic_json(args.output, output)
    print(json.dumps({"status": "VALID", "frame_count": len(frames), "formal_run": output["formal_run"],
                      "inference_frames_this_process": inference_frames}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
