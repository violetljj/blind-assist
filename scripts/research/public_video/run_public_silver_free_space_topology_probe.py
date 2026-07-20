#!/usr/bin/env python3
"""Probe public-video episodes with frozen free-space topology plus object trajectories.

This is an isolated public-silver mainline diagnostic. It builds a compact,
adaptive path descriptor only from the frozen SANPO segmentation logits, then
tests that descriptor alone and concatenated with the existing frozen COCO
object-trajectory vector. It does not read code, data, weights, or outputs from
the independent secondary corridor-causal direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

import run_public_silver_frozen_feature_probe as common
import run_public_silver_object_trajectory_probe as trajectory


SCHEMA = "blindassist_public_silver_free_space_topology_probe_v1"
THRESHOLDS = (0.35, 0.50, 0.65)


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this probe's scope: {path}")


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < 4 or min(values.shape[:2]) < 8:
        raise ValueError("semantic logits must be finite HxWxK with at least four classes")
    if not np.isfinite(values).all():
        raise ValueError("semantic logits must be finite")
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def contiguous_width(row: np.ndarray, center: int, threshold: float) -> float:
    mask = np.asarray(row >= threshold, dtype=bool)
    if not mask[center]:
        return 0.0
    left = center
    right = center
    while left > 0 and mask[left - 1]:
        left -= 1
    while right + 1 < len(mask) and mask[right + 1]:
        right += 1
    return float((right - left + 1) / len(mask))


def trace_adaptive_path(walkable: np.ndarray, *, horizon_ratio: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(walkable, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 8 or not np.isfinite(values).all():
        raise ValueError("walkable map must be a finite 2D array")
    height, width = values.shape
    horizon = int(round(height * horizon_ratio))
    smoothed = cv2.GaussianBlur(values.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=0.6)
    bottom_profile = smoothed[int(height * 0.82):].mean(axis=0)
    allowed = slice(max(0, int(width * 0.08)), min(width, int(width * 0.92)))
    allowed_indices = np.arange(allowed.start, allowed.stop)
    center_bias = 1e-4 * np.abs(allowed_indices - (width - 1) / 2.0)
    center = int(np.argmax(bottom_profile[allowed] - center_bias) + allowed.start)
    centers: list[int] = []
    probabilities: list[float] = []
    search_radius = max(2, int(round(width * 0.08)))
    for row_index in range(height - 1, horizon - 1, -1):
        left = max(0, center - search_radius)
        right = min(width, center + search_radius + 1)
        candidate_indices = np.arange(left, right)
        continuity_bias = 1e-4 * np.abs(candidate_indices - center)
        candidate = int(np.argmax(smoothed[row_index, left:right] - continuity_bias) + left)
        center = candidate
        centers.append(center)
        probabilities.append(float(values[row_index, center]))
    return np.asarray(centers[::-1], dtype=np.int64), np.asarray(probabilities[::-1], dtype=np.float64)


def frame_feature_names() -> list[str]:
    names: list[str] = []
    for threshold in THRESHOLDS:
        prefix = f"path_width_t{str(threshold).replace('.', '')}"
        names.extend([
            f"{prefix}_mean",
            f"{prefix}_minimum",
            f"{prefix}_q10",
            f"{prefix}_lower_mean",
            f"{prefix}_narrow_fraction",
        ])
    names.extend([
        "path_nonwalkable_mean",
        "path_nonwalkable_q90",
        "path_nonwalkable_maximum",
        "path_boundary_mean",
        "path_boundary_maximum",
        "path_obstacle_mean",
        "path_obstacle_maximum",
        "path_unknown_mean",
        "path_unknown_maximum",
        "path_center_offset_mean",
        "path_center_offset_maximum",
        "path_center_range",
        "path_center_slope",
    ])
    for vertical in range(3):
        for lateral in range(3):
            names.extend([
                f"grid_v{vertical}_x{lateral}_nonwalkable_mean",
                f"grid_v{vertical}_x{lateral}_nonwalkable_q90",
            ])
    return names


FRAME_FEATURE_NAMES = frame_feature_names()


def free_space_topology_frame(logits: np.ndarray, *, size: int = 64) -> tuple[np.ndarray, dict[str, Any]]:
    probabilities = softmax_probabilities(logits)
    resized = cv2.resize(probabilities.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    resized = resized / np.maximum(resized.sum(axis=-1, keepdims=True), 1e-8)
    walkable = resized[..., 0].astype(np.float64)
    centers, path_walkable = trace_adaptive_path(walkable)
    horizon = size - len(centers)
    rows = np.arange(horizon, size)
    lower_start = int(len(rows) * 0.55)

    values: list[float] = []
    width_profiles: dict[str, list[float]] = {}
    for threshold in THRESHOLDS:
        widths = np.asarray([
            contiguous_width(walkable[row], int(center), threshold)
            for row, center in zip(rows, centers)
        ], dtype=np.float64)
        width_profiles[str(threshold)] = widths.tolist()
        values.extend([
            float(widths.mean()),
            float(widths.min()),
            float(np.quantile(widths, 0.10)),
            float(widths[lower_start:].mean()),
            float(np.mean(widths < 0.18)),
        ])

    nonwalkable = 1.0 - path_walkable
    path_classes = resized[rows, centers]
    values.extend([
        float(nonwalkable.mean()),
        float(np.quantile(nonwalkable, 0.90)),
        float(nonwalkable.max()),
        float(path_classes[:, 1].mean()),
        float(path_classes[:, 1].max()),
        float(path_classes[:, 2].mean()),
        float(path_classes[:, 2].max()),
        float(path_classes[:, 3].mean()),
        float(path_classes[:, 3].max()),
    ])

    normalized_centers = centers / max(size - 1, 1)
    offsets = np.abs(normalized_centers - 0.5)
    time = np.linspace(-1.0, 1.0, len(normalized_centers))
    slope = float(np.dot(time, normalized_centers - normalized_centers.mean()) / max(np.dot(time, time), 1e-12))
    values.extend([
        float(offsets.mean()),
        float(offsets.max()),
        float(np.ptp(normalized_centers)),
        slope,
    ])

    lower = resized[int(size * 0.30):]
    vertical_edges = np.linspace(0, lower.shape[0], 4, dtype=int)
    lateral_edges = np.linspace(0, size, 4, dtype=int)
    for vertical in range(3):
        for lateral in range(3):
            region = 1.0 - lower[
                vertical_edges[vertical]:vertical_edges[vertical + 1],
                lateral_edges[lateral]:lateral_edges[lateral + 1],
                0,
            ]
            values.extend([float(region.mean()), float(np.quantile(region, 0.90))])

    vector = np.asarray(values, dtype=np.float64)
    if len(vector) != len(FRAME_FEATURE_NAMES) or not np.isfinite(vector).all():
        raise RuntimeError("free-space topology feature contract is inconsistent")
    return vector, {
        "path_center_mean": float(normalized_centers.mean()),
        "path_center_range": float(np.ptp(normalized_centers)),
        "path_nonwalkable_mean": float(nonwalkable.mean()),
        "minimum_width_t050": float(min(width_profiles["0.5"])),
    }


def topology_episode_vector(frame_vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(frame_vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("topology episode needs at least two frame vectors")
    time = np.linspace(-1.0, 1.0, len(values))
    slope = (time[:, None] * (values - values.mean(axis=0))).sum(axis=0) / max(float(np.dot(time, time)), 1e-12)
    return np.concatenate([
        values.mean(axis=0),
        values.max(axis=0),
        values[-1] - values[0],
        slope,
    ])


def extract_topology_features(
    model: Any,
    episodes: Sequence[dict[str, Any]],
    *,
    input_size: int,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for episode in episodes:
        frame_vectors: list[np.ndarray] = []
        frame_summaries: list[dict[str, Any]] = []
        paths = [Path(frame["path"]) for frame in episode["frames"]]
        for start in range(0, len(paths), batch_size):
            images: list[np.ndarray] = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(np.asarray(
                        image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR),
                        dtype=np.float32,
                    ))
            logits_batch = model.predict(np.stack(images), batch_size=batch_size, verbose=0)
            for logits in logits_batch:
                vector, summary = free_space_topology_frame(logits)
                frame_vectors.append(vector)
                frame_summaries.append(summary)
        vectors.append(topology_episode_vector(np.stack(frame_vectors)))
        summaries.append({
            "episode_id": episode["episode_id"],
            "frame_count": len(frame_vectors),
            "mean_path_nonwalkable": float(np.mean([row["path_nonwalkable_mean"] for row in frame_summaries])),
            "minimum_width_t050": float(np.min([row["minimum_width_t050"] for row in frame_summaries])),
            "maximum_path_center_range": float(np.max([row["path_center_range"] for row in frame_summaries])),
        })
    return np.stack(vectors), summaries


def evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    episodes: Sequence[dict[str, Any]],
    *,
    ridge: float,
) -> dict[str, Any]:
    episode_ids = [row["episode_id"] for row in episodes]
    source_ids = [row["source_id"] for row in episodes]
    first = common.leave_one_source_group_out(
        features, labels, episode_ids, source_ids, ridge=ridge, class_balanced=True,
    )
    second = common.leave_one_source_group_out(
        features, labels, episode_ids, source_ids, ridge=ridge, class_balanced=True,
    )
    return {
        "split": "leave_one_source_group_out",
        "group_key": "source_id",
        "frame_or_session_leakage": False,
        "ridge": ridge,
        **first,
        "repeat_exact": first == second,
        "counterfactual_delta_alignment": common.counterfactual_delta_alignment(features, episodes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.package_root, args.segmentation_weights, args.detector_weights,
        args.cache_dir, args.output,
    ):
        reject_independent_direction(path)
    package_root = args.package_root.resolve()
    segmentation_weights = args.segmentation_weights.resolve()
    detector_weights = args.detector_weights.resolve()
    if not package_root.is_dir() or not segmentation_weights.is_file() or not detector_weights.is_file():
        raise FileNotFoundError("package root, segmentation weights, or detector weights are missing")

    episodes, excluded = common.load_episode_specs(package_root)
    labels = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("free-space topology probe requires both classes")

    os.environ["KERAS_BACKEND"] = args.backend
    random.seed(args.seed)
    np.random.seed(args.seed)
    import keras
    keras.utils.set_random_seed(args.seed)
    if args.backend == "torch":
        import torch
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    segmentation_model = common.shared.sanpo_segmentation_model.build_mobilenetv3_lraspp(
        keras, args.input_size, backbone_alpha=1.0, decoder_channels=96,
        detail_output_stride=8, semantic_output_stride=32,
    )
    segmentation_model.load_weights(segmentation_weights)
    topology_features, topology_summaries = extract_topology_features(
        segmentation_model, episodes, input_size=args.input_size, batch_size=args.batch_size,
    )

    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    from ultralytics import YOLO
    detector = YOLO(str(detector_weights))
    trajectory_features, detection_summaries = trajectory.extract(
        detector, episodes, image_size=args.image_size, confidence=args.confidence,
    )

    topology_evaluation = evaluate(topology_features, labels, episodes, ridge=args.ridge)
    fusion_features = np.concatenate([trajectory_features, topology_features], axis=1)
    fusion_evaluation = evaluate(fusion_features, labels, episodes, ridge=args.ridge)
    metrics = fusion_evaluation["metrics"]
    gate = bool(
        fusion_evaluation["repeat_exact"]
        and metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and metrics["candidate_alert_recall"] >= args.minimum_class_recall
        and metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
    )

    feature_digest = hashlib.sha256()
    feature_digest.update(np.asarray(topology_features, dtype="<f8").tobytes(order="C"))
    feature_digest.update(np.asarray(trajectory_features, dtype="<f8").tobytes(order="C"))
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "class_counts": {
            "candidate_no_alert": int(np.sum(labels == 0)),
            "candidate_alert": int(np.sum(labels == 1)),
        },
        "feature_source": {
            "segmentation_model": "frozen MobileNetV3Small+LR-ASPP logits",
            "segmentation_weights": str(segmentation_weights),
            "segmentation_weights_sha256": common.sha256_file(segmentation_weights),
            "detector_model": detector_weights.name,
            "detector_weights": str(detector_weights),
            "detector_weights_sha256": common.sha256_file(detector_weights),
            "topology_frame_dimension": len(FRAME_FEATURE_NAMES),
            "topology_episode_dimension": int(topology_features.shape[1]),
            "trajectory_dimension": int(trajectory_features.shape[1]),
            "fusion_dimension": int(fusion_features.shape[1]),
            "feature_sha256": feature_digest.hexdigest(),
            "trainable_parameters": 0,
            "isolation": "independent-direction code, data, weights, metrics, and outputs are not read",
        },
        "topology_only_evaluation": topology_evaluation,
        "trajectory_plus_topology_evaluation": fusion_evaluation,
        "linear_separability_gate": {
            "passed": gate,
            "thresholds": {
                "balanced_accuracy_gte": args.minimum_balanced_accuracy,
                "each_class_recall_gte": args.minimum_class_recall,
            },
        },
        "episode_topology_summaries": topology_summaries,
        "episode_detection_summaries": detection_summaries,
        "episodes": [
            {key: value for key, value in row.items() if key != "frames"} | {"frame_count": len(row["frames"])}
            for row in episodes
        ],
        "excluded_abstentions": excluded,
        "evidence_limit": "Tiny GPT/VLM-labelled provisional set; frozen segmentation and detector outputs are features, never event truth.",
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
    parser.add_argument("--segmentation-weights", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("torch",), default="torch")
    parser.add_argument("--input-size", type=int, choices=(384,), default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if args.batch_size <= 0 or not 0 < args.confidence < 1 or args.ridge <= 0:
        parser.error("batch size and ridge must be positive; confidence must be in (0,1)")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    topology = report["topology_only_evaluation"]["metrics"]["balanced_accuracy"]
    fusion = report["trajectory_plus_topology_evaluation"]["metrics"]["balanced_accuracy"]
    print(json.dumps({
        "ok": True,
        "episode_count": report["episode_count"],
        "topology_balanced_accuracy": topology,
        "fusion_balanced_accuracy": fusion,
        "linear_separable": report["linear_separability_gate"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
