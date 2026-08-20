#!/usr/bin/env python3
"""Produce causal target observations from RGB only."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]) + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def choose_target(candidates: list[dict[str, Any]], previous: list[float] | None):
    if not candidates:
        return None, 0.0
    if previous is None:
        selected = max(candidates, key=lambda row: row["confidence"])
        return selected, 0.0
    selected = max(candidates, key=lambda row: 0.7 * iou(row["bbox_xyxy"], previous) + 0.3 * row["confidence"])
    return selected, iou(selected["bbox_xyxy"], previous)


def appearance_embedding(image, bbox: list[float]):
    """Return a compact RGB-only crop descriptor suitable for instance ranking."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).reshape(-1)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    gradient = np.zeros(8, dtype=np.float32)
    for bin_index in range(8):
        mask = ((angle >= bin_index * 45.0) & (angle < (bin_index + 1) * 45.0))
        gradient[bin_index] = float(magnitude[mask].sum())
    color = color.astype(np.float32)
    color /= max(float(color.sum()), 1e-6)
    gradient /= max(float(gradient.sum()), 1e-6)
    embedding = np.concatenate((0.85 * color, 0.15 * gradient))
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 0 else None


def cosine_similarity(left, right) -> float:
    import numpy as np

    if left is None or right is None:
        return 0.0
    return float(np.clip(np.dot(left, right), 0.0, 1.0))


def bbox_shape(bbox: list[float], frame_width: int, frame_height: int) -> tuple[float, float]:
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    return width / height, math.sqrt(width * height / (frame_width * frame_height))


@dataclass
class TargetMemory:
    max_templates: int = 5
    templates: list[Any] = field(default_factory=list)
    aspect_ratios: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    last_reliable_box: list[float] | None = None
    last_seen_frame: int | None = None
    last_template_frame: int | None = None

    def remember(self, image, bbox: list[float], frame_index: int, force: bool = False) -> bool:
        embedding = appearance_embedding(image, bbox)
        if embedding is None:
            return False
        if self.templates and not force:
            anchor_similarity = cosine_similarity(embedding, self.templates[0])
            if anchor_similarity < 0.72:
                return False
            if self.last_template_frame is not None and frame_index - self.last_template_frame < 12:
                return False
            if max(cosine_similarity(embedding, template) for template in self.templates) > 0.995:
                return False
        aspect_ratio, scale = bbox_shape(bbox, image.shape[1], image.shape[0])
        if len(self.templates) >= self.max_templates:
            self.templates.pop(1)
            self.aspect_ratios.pop(1)
            self.scales.pop(1)
        self.templates.append(embedding)
        self.aspect_ratios.append(aspect_ratio)
        self.scales.append(scale)
        self.last_template_frame = frame_index
        self.last_reliable_box = list(bbox)
        self.last_seen_frame = frame_index
        return True

    def score(self, image, bbox: list[float], frame_index: int) -> dict[str, float]:
        import numpy as np

        embedding = appearance_embedding(image, bbox)
        appearance = max((cosine_similarity(embedding, template) for template in self.templates), default=0.0)
        aspect_ratio, scale = bbox_shape(bbox, image.shape[1], image.shape[0])
        median_aspect = float(np.median(self.aspect_ratios)) if self.aspect_ratios else aspect_ratio
        median_scale = float(np.median(self.scales)) if self.scales else scale
        shape = math.exp(-abs(math.log(max(aspect_ratio, 1e-6) / max(median_aspect, 1e-6))))
        scale_score = math.exp(-abs(math.log(max(scale, 1e-6) / max(median_scale, 1e-6))))
        spatial = 0.0
        if self.last_reliable_box is not None:
            old_x = (self.last_reliable_box[0] + self.last_reliable_box[2]) / 2.0
            old_y = (self.last_reliable_box[1] + self.last_reliable_box[3]) / 2.0
            new_x = (bbox[0] + bbox[2]) / 2.0
            new_y = (bbox[1] + bbox[3]) / 2.0
            distance = math.hypot((new_x - old_x) / image.shape[1], (new_y - old_y) / image.shape[0])
            age = frame_index - (self.last_seen_frame if self.last_seen_frame is not None else frame_index)
            spatial = math.exp(-4.0 * distance) * math.exp(-age / 30.0)
        total = 0.80 * appearance + 0.10 * shape + 0.05 * scale_score + 0.05 * spatial
        return {"appearance": appearance, "shape": shape, "scale": scale_score, "spatial": spatial, "score": total, "embedding": embedding}


def rank_redetection_candidates(memory: TargetMemory, image, candidates: list[dict[str, Any]], frame_index: int):
    ranked = []
    for candidate in candidates:
        scored = memory.score(image, candidate["bbox_xyxy"], frame_index)
        ranked.append({**candidate, **scored})
    return sorted(ranked, key=lambda row: row["score"], reverse=True)


def candidate_compatible_hit_count(history, current: dict[str, Any]) -> int:
    compatible = 1
    for earlier in history:
        if earlier is None:
            continue
        appearance = cosine_similarity(current["embedding"], earlier["embedding"])
        if appearance >= 0.90 and iou(current["bbox_xyxy"], earlier["bbox_xyxy"]) >= 0.03:
            compatible += 1
    return compatible


def candidate_confirmed(history, current: dict[str, Any], required_hits: int) -> bool:
    return candidate_compatible_hit_count(history, current) >= required_hits


def initial_anchor_candidate(candidates: list[dict[str, Any]], memory: TargetMemory):
    """Bootstrap identity once; later target-free frames must use redetection."""
    if memory.templates:
        return None
    return max(candidates, key=lambda row: row["confidence"], default=None)


def flow_bbox(previous_gray, current_gray, points, bbox, width: int, height: int):
    """Propagate one bbox by robust sparse optical flow; fail closed on weak flow."""
    import cv2
    import numpy as np

    if previous_gray is None or points is None or len(points) < 3 or bbox is None:
        return None, None
    moved, status, errors = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
    if moved is None or status is None:
        return None, None
    valid = status.reshape(-1).astype(bool)
    if errors is not None:
        valid &= errors.reshape(-1) < 30.0
    old_points = points.reshape(-1, 2)[valid]
    new_points = moved.reshape(-1, 2)[valid]
    if len(new_points) < 3:
        return None, None
    displacement = np.median(new_points - old_points, axis=0)
    if abs(float(displacement[0])) > width * 0.08 or abs(float(displacement[1])) > height * 0.08:
        return None, None
    propagated = [
        max(0.0, min(width - 1.0, bbox[0] + float(displacement[0]))),
        max(0.0, min(height - 1.0, bbox[1] + float(displacement[1]))),
        max(0.0, min(width - 1.0, bbox[2] + float(displacement[0]))),
        max(0.0, min(height - 1.0, bbox[3] + float(displacement[1]))),
    ]
    if propagated[2] - propagated[0] < 3 or propagated[3] - propagated[1] < 3:
        return None, None
    return propagated, new_points.reshape(-1, 1, 2).astype("float32")


def seed_flow_points(gray, bbox):
    import cv2
    import numpy as np

    mask = np.zeros_like(gray)
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    mask[max(0, y1):min(gray.shape[0], y2), max(0, x1):min(gray.shape[1], x2)] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=40, qualityLevel=0.01, minDistance=3, blockSize=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="RGB-only ADT target observation adapter")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, for example cpu or 0")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--flow-max-gap", type=int, default=0, help="Maximum detector-missing frames filled by RGB optical flow")
    parser.add_argument("--instance-redetection", action="store_true")
    parser.add_argument("--redetection-generator", choices=("class-yolo", "yoloe-visual-prompt"), default="class-yolo")
    parser.add_argument("--redetection-model", type=Path, help="YOLOE checkpoint used only for LOST-frame proposal generation")
    parser.add_argument("--redetect-confidence", type=float, default=0.02)
    parser.add_argument("--redetect-appearance-threshold", type=float, default=0.82)
    parser.add_argument("--redetect-score-threshold", type=float, default=0.75)
    parser.add_argument("--redetect-margin", type=float, default=0.08)
    parser.add_argument("--redetect-confirm-hits", type=int, default=2)
    parser.add_argument("--redetect-confirm-window", type=int, default=3)
    parser.add_argument("--short-reconnect-max-gap", type=int, default=15)
    parser.add_argument("--memory-max-templates", type=int, default=5)
    parser.add_argument("--memory-update-quarantine", type=int, default=8)
    parser.add_argument("--candidate-diagnostics", action="store_true", help="Record RGB-only proposal and verifier traces for isolated GT failure accounting")
    args = parser.parse_args()

    from ultralytics import YOLO, YOLOE
    from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor
    import ultralytics

    model = YOLO(str(args.model))
    if args.redetection_generator == "yoloe-visual-prompt" and not args.instance_redetection:
        raise ValueError("YOLOE redetection requires --instance-redetection")
    if args.redetection_generator == "yoloe-visual-prompt" and args.redetection_model is None:
        raise ValueError("YOLOE redetection requires --redetection-model")
    redetection_model = YOLOE(str(args.redetection_model)) if args.redetection_generator == "yoloe-visual-prompt" else None
    redetection_prompt_ready = False
    inference_confidence = min(args.confidence, args.redetect_confidence) if args.instance_redetection and redetection_model is None else args.confidence
    results = model.predict(source=str(args.video), stream=True, verbose=False, device=args.device, imgsz=args.imgsz, conf=inference_confidence)
    frames = []
    events = []
    previous_bbox = None
    previous_nearness = None
    approach_ema = 0.0
    previous_visible = False
    ever_visible = False
    previous_gray = None
    flow_points = None
    flow_gap = 0
    last_detector_confidence = 0.0
    target_memory = TargetMemory(max_templates=args.memory_max_templates)
    redetection_history = deque(maxlen=args.redetect_confirm_window - 1)
    lost_frames = 0
    stable_tracked_frames = 0
    memory_quarantine = 0
    fps = None
    frame_size = None
    for frame_index, result in enumerate(results):
        fps = float(result.speed.get("fps", 0.0)) or fps
        height, width = result.orig_shape
        import cv2
        current_gray = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2GRAY)
        frame_size = {"width": int(width), "height": int(height)}
        candidates = []
        for box, confidence, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist(), strict=True):
            if str(result.names[int(class_id)]).casefold() == args.target_class.casefold():
                candidates.append({"bbox_xyxy": [float(value) for value in box], "confidence": float(confidence)})
        primary_candidates = [candidate for candidate in candidates if candidate["confidence"] >= args.confidence]
        selected, association_iou = choose_target(primary_candidates, previous_bbox)
        observation_source = "none"
        redetection_details = None
        redetection_probe = None
        redetection_trace = None
        if args.instance_redetection and previous_bbox is None:
            anchor = initial_anchor_candidate(primary_candidates, target_memory)
            selected = anchor
            association_iou = 0.0
            if anchor is not None:
                observation_source = "detector"
            elif target_memory.templates:
                redetection_candidates = candidates
                if redetection_model is not None:
                    prompted = redetection_model.predict(source=result.orig_img, verbose=False, device=args.device, imgsz=args.imgsz, conf=args.redetect_confidence)[0]
                    redetection_candidates = [
                        {"bbox_xyxy": [float(value) for value in box], "confidence": float(confidence)}
                        for box, confidence in zip(prompted.boxes.xyxy.tolist(), prompted.boxes.conf.tolist(), strict=True)
                    ]
                ranked = rank_redetection_candidates(target_memory, result.orig_img, redetection_candidates, frame_index)
                if ranked:
                    top = ranked[0]
                    margin = top["score"] - ranked[1]["score"] if len(ranked) > 1 else 1.0
                    eligible = top["appearance"] >= args.redetect_appearance_threshold and top["score"] >= args.redetect_score_threshold and margin >= args.redetect_margin
                    short_reconnect = lost_frames <= args.short_reconnect_max_gap and top["confidence"] >= args.confidence and top["spatial"] >= 0.35
                    compatible_hits = candidate_compatible_hit_count(redetection_history, top)
                    confirmed = eligible and compatible_hits >= args.redetect_confirm_hits
                    if args.candidate_diagnostics:
                        redetection_trace = {
                            "search_active": True,
                            "candidate_count": len(ranked),
                            "top_margin": float(margin),
                            "top_eligible": bool(eligible),
                            "top_short_reconnect": bool(short_reconnect),
                            "top_confirmation_compatible_hits": compatible_hits,
                            "top_confirmed": bool(confirmed),
                            "candidates": [
                                {
                                    "rank": rank,
                                    "bbox_xyxy": [float(value) for value in candidate["bbox_xyxy"]],
                                    "confidence": float(candidate["confidence"]),
                                    "appearance": float(candidate["appearance"]),
                                    "shape": float(candidate["shape"]),
                                    "scale": float(candidate["scale"]),
                                    "spatial": float(candidate["spatial"]),
                                    "score": float(candidate["score"]),
                                    "considered_by_verifier": rank == 0,
                                    "verifier_eligible": bool(eligible) if rank == 0 else False,
                                }
                                for rank, candidate in enumerate(ranked)
                            ],
                        }
                    redetection_probe = {key: float(top[key]) for key in ("appearance", "shape", "scale", "spatial", "score")}
                    redetection_probe.update({"margin": float(margin), "eligible": bool(eligible), "short_reconnect": bool(short_reconnect)})
                    if eligible and short_reconnect:
                        selected = top
                        observation_source = "detector_reconnect"
                        redetection_history.clear()
                    elif confirmed:
                        selected = top
                        observation_source = "instance_redetection"
                        redetection_details = {key: float(top[key]) for key in ("appearance", "shape", "scale", "spatial", "score")}
                        redetection_details["margin"] = float(margin)
                        redetection_history.clear()
                        memory_quarantine = args.memory_update_quarantine
                    else:
                        redetection_history.append(top if eligible else None)
                else:
                    if args.candidate_diagnostics:
                        redetection_trace = {"search_active": True, "candidate_count": 0, "candidates": []}
                    redetection_history.append(None)
        if selected is not None:
            if observation_source == "none":
                observation_source = "detector"
            flow_gap = 0
            last_detector_confidence = selected["confidence"]
        elif args.flow_max_gap > 0 and flow_gap < args.flow_max_gap:
            propagated, moved_points = flow_bbox(previous_gray, current_gray, flow_points, previous_bbox, width, height)
            if propagated is not None:
                flow_gap += 1
                selected = {"bbox_xyxy": propagated, "confidence": last_detector_confidence * (0.94 ** flow_gap)}
                flow_points = moved_points
                association_iou = 1.0
                observation_source = "optical_flow"
        if selected is None:
            visible = False
            bbox = None
            confidence = 0.0
            bearing = None
            nearness = None
            approach_rate = None
            tracking_quality = 0.0
            observation_quality = 0.0
            if flow_gap >= args.flow_max_gap:
                previous_bbox = None
                previous_nearness = None
                flow_points = None
            lost_frames += 1
            stable_tracked_frames = 0
        else:
            visible = True
            bbox = selected["bbox_xyxy"]
            confidence = selected["confidence"]
            center_x = (bbox[0] + bbox[2]) / 2.0
            bearing = (center_x - width / 2.0) / (width / 2.0)
            nearness = math.sqrt(max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)))
            raw_rate = 0.0 if previous_nearness is None else (nearness - previous_nearness) * 30.0
            approach_ema = 0.8 * approach_ema + 0.2 * raw_rate
            approach_rate = approach_ema
            tracking_quality = confidence * (0.5 + 0.5 * association_iou) if previous_bbox is not None else confidence * 0.5
            observation_quality = min(confidence, tracking_quality + 0.15)
            if observation_source == "optical_flow":
                tracking_quality *= 0.6
                observation_quality = min(observation_quality, tracking_quality)
            previous_bbox = bbox
            previous_nearness = nearness
            lost_frames = 0
            stable_tracked_frames += 1
            if observation_source == "detector" or flow_points is None or len(flow_points) < 6:
                flow_points = seed_flow_points(current_gray, bbox)
            if args.instance_redetection:
                if not target_memory.templates and observation_source == "detector" and confidence >= args.confidence:
                    anchored = target_memory.remember(result.orig_img, bbox, frame_index, force=True)
                    if anchored and redetection_model is not None and not redetection_prompt_ready:
                        redetection_model.predict(
                            source=result.orig_img,
                            refer_image=result.orig_img,
                            visual_prompts={"bboxes": [bbox], "cls": [0]},
                            verbose=False,
                            device=args.device,
                            imgsz=args.imgsz,
                            conf=args.redetect_confidence,
                            predictor=YOLOEVPSegPredictor,
                        )
                        redetection_prompt_ready = True
                elif observation_source == "detector" and stable_tracked_frames >= 3 and memory_quarantine <= 0 and association_iou >= 0.15:
                    target_memory.remember(result.orig_img, bbox, frame_index)
                if target_memory.templates:
                    target_memory.last_reliable_box = list(bbox)
                    target_memory.last_seen_frame = frame_index
            memory_quarantine = max(0, memory_quarantine - 1)

        if visible and not previous_visible:
            events.append({"frame_index": frame_index, "event": "REACQUIRED" if ever_visible else "ACQUIRED"})
        elif not visible and previous_visible:
            events.append({"frame_index": frame_index, "event": "LOST"})
        ever_visible = ever_visible or visible
        previous_visible = visible
        frame_record = {
                "frame_index": frame_index,
                "timestamp_s": frame_index / 30.0,
                "target_visible": visible,
                "target_class": args.target_class,
                "bbox_xyxy": bbox,
                "target_confidence": confidence,
                "target_bearing_normalized": bearing,
                "bearing_unit": "normalized_image_x",
                "relative_nearness": nearness,
                "approach_rate_per_s": approach_rate,
                "tracking_quality": tracking_quality,
                "observation_quality": observation_quality,
                "observation_source": observation_source,
                "redetection_details": redetection_details,
                "redetection_probe": redetection_probe,
            }
        if args.candidate_diagnostics:
            frame_record["redetection_trace"] = redetection_trace
        frames.append(frame_record)
        previous_gray = current_gray

    output = {
        "schema_version": "ba_adt_rgb_observation_v3" if args.instance_redetection else "ba_adt_rgb_observation_v2",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-1-CANARY",
        "input": {"video": args.video.name, "sha256": sha256(args.video), "role": "RGB_SYSTEM_INPUT"},
        "groundtruth_argument_supported": False,
        "model": {"path_name": args.model.name, "sha256": sha256(args.model), "runtime": f"ultralytics-{ultralytics.__version__}", "device": str(args.device), "target_class": args.target_class, "imgsz": args.imgsz, "confidence_floor": args.confidence, "flow_max_gap": args.flow_max_gap},
        "instance_redetection": {"enabled": args.instance_redetection, "candidate_generator": args.redetection_generator, "candidate_model": None if args.redetection_model is None else {"path_name": args.redetection_model.name, "sha256": sha256(args.redetection_model), "visual_prompt_source": "first_confirmed_rgb_anchor_bbox"}, "candidate_confidence_floor": args.redetect_confidence if redetection_model is not None else inference_confidence, "appearance_threshold": args.redetect_appearance_threshold, "score_threshold": args.redetect_score_threshold, "top1_top2_margin": args.redetect_margin, "confirmation": f"{args.redetect_confirm_hits}-of-{args.redetect_confirm_window}", "short_reconnect_max_gap": args.short_reconnect_max_gap, "appearance_weight": 0.80, "memory_templates": len(target_memory.templates), "memory_update_quarantine_frames": args.memory_update_quarantine, "candidate_diagnostics": args.candidate_diagnostics},
        "frame_count": len(frames),
        "frame_size": frame_size,
        "visible_frame_count": sum(row["target_visible"] for row in frames),
        "events": events,
        "frames": frames,
        "limitations": ["preview_mp4_frame_order_time_proxy", "normalized_image_bearing_not_calibrated_degrees", "bbox_scale_nearness_not_metric_distance", "single_visual_prompt_or_single_class_candidate_generator", "handcrafted_color_gradient_instance_embedding", "optional_sparse_optical_flow_translation_only"],
        "claim_ceiling": "rgb_only_observation_mechanics_no_accuracy_or_navigation_claim",
        "terminal": "ADT1_RGB_OBSERVATIONS_PRODUCED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": output["terminal"], "frames": len(frames), "visible_frames": output["visible_frame_count"], "events": len(events)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
