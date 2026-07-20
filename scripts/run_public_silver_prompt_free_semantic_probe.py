#!/usr/bin/env python3
"""Probe static-obstacle semantics with a frozen prompt-free YOLOE model.

The prompt-free model supplies detections from its fixed built-in vocabulary;
it receives no public-silver labels, text prompts, source masks, or synthetic
images. A preregistered subset of surface-material and barrier/furniture class
names is converted into deterministic corridor-relative temporal features.
The probe reports semantic-only and COCO-trajectory+semantic LOSO results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_object_trajectory_probe as trajectory
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_prompt_free_semantic_probe_v1"
SEMANTIC_GROUPS = {
    "surface_material": {
        "sand",
        "sand bar",
        "sand box",
        "dirt",
        "dirt field",
        "dirt road",
        "dirt track",
        "earth",
    },
    "barrier_structure": {
        "barrier",
        "construction site",
        "furniture",
        "obstacle course",
    },
}


def semantic_group(class_name: str) -> str | None:
    normalized = class_name.strip().lower()
    for group, names in SEMANTIC_GROUPS.items():
        if normalized in names:
            return group
    return None


def semantic_frame_vector(detections: Sequence[dict[str, Any]]) -> np.ndarray:
    rows = list(detections)
    values: list[float] = []
    for group in SEMANTIC_GROUPS:
        selected = [row for row in rows if row["semantic_group"] == group]
        values.extend([
            min(len(selected), 5) / 5.0,
            max((row["confidence"] for row in selected), default=0.0),
            max((row["area"] for row in selected), default=0.0),
            max((row["bottom"] for row in selected), default=0.0),
            max((row["corridor_overlap"] for row in selected), default=0.0),
            max((row["threat"] for row in selected), default=0.0),
            sum(row["area"] * row["corridor_overlap"] for row in selected),
        ])
    return np.asarray(values, dtype=np.float64)


def semantic_episode_vector(frames: Sequence[Sequence[dict[str, Any]]]) -> np.ndarray:
    if len(frames) < 2:
        raise ValueError("semantic episode needs at least two frames")
    values = np.stack([semantic_frame_vector(frame) for frame in frames])
    time = np.linspace(-1.0, 1.0, len(values))
    slope = (time[:, None] * values).sum(axis=0) / float(np.sum(time * time))
    return np.concatenate([
        values.mean(axis=0),
        values.max(axis=0),
        values[-1],
        values[-1] - values[0],
        slope,
    ])


def extract_semantic_detections(
    model: Any,
    episodes: Sequence[dict[str, Any]],
    *,
    image_size: int,
    confidence: float,
) -> tuple[list[list[list[dict[str, Any]]]], list[dict[str, Any]]]:
    paths = [frame["path"] for episode in episodes for frame in episode["frames"]]
    results = []
    for path in paths:
        prediction = model.predict(
            path,
            imgsz=image_size,
            conf=confidence,
            max_det=100,
            verbose=False,
        )
        if len(prediction) != 1:
            raise RuntimeError("prompt-free semantic detector must return exactly one result per frame")
        results.append(prediction[0])
    if len(results) != len(paths):
        raise RuntimeError("prompt-free semantic detector returned an unexpected result count")
    all_frames: list[list[list[dict[str, Any]]]] = []
    summaries: list[dict[str, Any]] = []
    cursor = 0
    for episode in episodes:
        episode_frames: list[list[dict[str, Any]]] = []
        class_counts: dict[str, int] = {}
        for _frame in episode["frames"]:
            result = results[cursor]
            cursor += 1
            height, width = result.orig_shape
            detections: list[dict[str, Any]] = []
            if result.boxes is not None:
                for box, score, class_id in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                ):
                    class_name = model.names[int(class_id)]
                    group = semantic_group(class_name)
                    if group is None:
                        continue
                    normalized = trajectory.normalize_detection(
                        class_name,
                        float(score),
                        box.tolist(),
                        width=width,
                        height=height,
                    )
                    normalized["semantic_group"] = group
                    normalized["group"] = "other" if group == "surface_material" else "furniture"
                    detections.append(normalized)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            episode_frames.append(detections)
        all_frames.append(episode_frames)
        summaries.append({
            "episode_id": episode["episode_id"],
            "semantic_detection_count": sum(len(frame) for frame in episode_frames),
            "semantic_class_counts": dict(sorted(class_counts.items())),
        })
    return all_frames, summaries


def evaluation(
    features: np.ndarray,
    labels: np.ndarray,
    episodes: Sequence[dict[str, Any]],
    *,
    ridge: float,
) -> dict[str, Any]:
    episode_ids = [episode["episode_id"] for episode in episodes]
    source_ids = [episode["source_id"] for episode in episodes]
    return common.leave_one_source_group_out(
        features,
        labels,
        episode_ids,
        source_ids,
        ridge=ridge,
        class_balanced=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.semantic_weights, args.coco_weights, args.output):
        mil.reject_independent_direction(path)
    if (
        not args.package_root.is_dir()
        or not args.semantic_weights.is_file()
        or not args.coco_weights.is_file()
    ):
        raise FileNotFoundError("package root or detector weights are missing")
    episodes, excluded = common.load_episode_specs(args.package_root)
    labels = np.asarray([episode["label"] for episode in episodes], dtype=np.int64)
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOE

    semantic_model = YOLOE(str(args.semantic_weights))
    semantic_frames, semantic_summaries = extract_semantic_detections(
        semantic_model,
        episodes,
        image_size=args.image_size,
        confidence=args.semantic_confidence,
    )
    semantic_features = np.stack([
        semantic_episode_vector(frames) for frames in semantic_frames
    ])

    coco_model = YOLO(str(args.coco_weights))
    coco_frames = trajectory.extract_frame_detections(
        coco_model,
        episodes,
        image_size=args.image_size,
        confidence=args.coco_confidence,
    )
    trajectory_features = np.stack([
        trajectory.episode_vector(frames)[0] for frames in coco_frames
    ])
    fused_features = np.concatenate([trajectory_features, semantic_features], axis=1)

    semantic_first = evaluation(semantic_features, labels, episodes, ridge=args.ridge)
    semantic_second = evaluation(semantic_features, labels, episodes, ridge=args.ridge)
    fused_first = evaluation(fused_features, labels, episodes, ridge=args.ridge)
    fused_second = evaluation(fused_features, labels, episodes, ridge=args.ridge)
    semantic_repeat = semantic_first == semantic_second
    fused_repeat = fused_first == fused_second
    fused_metrics = fused_first["metrics"]
    gate = bool(
        fused_repeat
        and fused_metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and fused_metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
        and fused_metrics["candidate_alert_recall"] >= args.minimum_class_recall
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(args.package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "feature_source": {
            "semantic_model": "YOLOE prompt-free segmentation model with fixed built-in vocabulary",
            "semantic_weights": str(args.semantic_weights),
            "semantic_weights_sha256": common.sha256_file(args.semantic_weights),
            "semantic_confidence": args.semantic_confidence,
            "image_size": args.image_size,
            "runtime": {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "semantic_class_groups": {
                group: sorted(names) for group, names in SEMANTIC_GROUPS.items()
            },
            "semantic_feature_dimension": int(semantic_features.shape[1]),
            "coco_weights": str(args.coco_weights),
            "coco_weights_sha256": common.sha256_file(args.coco_weights),
            "trajectory_feature_dimension": int(trajectory_features.shape[1]),
            "fusion_feature_dimension": int(fused_features.shape[1]),
            "semantic_feature_sha256": hashlib.sha256(
                np.asarray(semantic_features, dtype="<f8").tobytes(order="C")
            ).hexdigest(),
            "trainable_backbone_parameters": 0,
            "text_prompt_used": False,
            "source_masks_used": False,
            "synthetic_images_used": False,
        },
        "semantic_only_evaluation": {
            **semantic_first,
            "repeat_exact": semantic_repeat,
        },
        "trajectory_plus_semantic_evaluation": {
            **fused_first,
            "repeat_exact": fused_repeat,
        },
        "semantic_pair_alignment": common.counterfactual_delta_alignment(
            semantic_features,
            episodes,
        ),
        "fusion_pair_alignment": common.counterfactual_delta_alignment(
            fused_features,
            episodes,
        ),
        "episode_semantic_summaries": semantic_summaries,
        "linear_separability_gate": {
            "passed": gate,
            "thresholds": {
                "balanced_accuracy_gte": args.minimum_balanced_accuracy,
                "each_class_recall_gte": args.minimum_class_recall,
            },
        },
        "evidence_limit": "Frozen prompt-free vocabulary predictions are proposals, not event truth or production evidence.",
        "training_execution_authorized": True,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--semantic-weights", type=Path, required=True)
    parser.add_argument("--coco-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../artifacts.local/cache/ultralytics-trajectory"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--semantic-confidence", type=float, default=0.01)
    parser.add_argument("--coco-confidence", type=float, default=0.15)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if (
        args.image_size <= 0
        or not 0 < args.semantic_confidence < 1
        or not 0 < args.coco_confidence < 1
        or args.ridge <= 0
    ):
        parser.error("image size, confidence values, and ridge must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    semantic = report["semantic_only_evaluation"]["metrics"]
    fused = report["trajectory_plus_semantic_evaluation"]["metrics"]
    print(json.dumps({
        "ok": True,
        "semantic_balanced_accuracy": semantic["balanced_accuracy"],
        "fusion_balanced_accuracy": fused["balanced_accuracy"],
        "fusion_candidate_no_alert_recall": fused["candidate_no_alert_recall"],
        "fusion_candidate_alert_recall": fused["candidate_alert_recall"],
        "linear_separable": report["linear_separability_gate"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
