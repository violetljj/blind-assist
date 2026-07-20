#!/usr/bin/env python3
"""Probe provisional episodes with camera-motion-compensated near-field change.

ORB matches and a RANSAC homography estimate global camera motion between
sampled frames. Residual change inside a fixed near-field corridor, together
with match failure and inlier support, forms a compact frozen descriptor.
The descriptor is evaluated alone and concatenated with the existing frozen
object-trajectory representation under leave-one-source-group-out folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_object_trajectory_probe as trajectory


SCHEMA = "blindassist_public_silver_motion_compensated_occupancy_probe_v1"
DIFF_THRESHOLDS = (0.12, 0.18, 0.25)


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this probe's scope: {path}")


def corridor_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    y = yy / max(size - 1, 1)
    x = xx / max(size - 1, 1)
    half_width = 0.16 + np.clip((y - 0.35) / 0.65, 0.0, 1.0) * 0.34
    corridor = (y >= 0.35) & (np.abs(x - 0.5) <= half_width)
    lower = corridor & (y >= 0.58)
    return corridor, lower


def region_stats(values: np.ndarray, mask: np.ndarray) -> list[float]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    if not len(selected):
        raise ValueError("motion region is empty")
    result = [
        float(selected.mean()),
        float(np.quantile(selected, 0.75)),
        float(np.quantile(selected, 0.90)),
    ]
    result.extend(float(np.mean(selected >= threshold)) for threshold in DIFF_THRESHOLDS)
    return result


def frame_pair_descriptor(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    size: int = 320,
) -> tuple[np.ndarray, dict[str, Any]]:
    if previous is None or current is None:
        raise ValueError("motion descriptor needs two decoded images")
    previous = cv2.resize(previous, (size, size), interpolation=cv2.INTER_AREA)
    current = cv2.resize(current, (size, size), interpolation=cv2.INTER_AREA)
    gray_previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=8)
    key_previous, desc_previous = orb.detectAndCompute(gray_previous, None)
    key_current, desc_current = orb.detectAndCompute(gray_current, None)

    good: list[Any] = []
    if desc_previous is not None and desc_current is not None:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_previous, desc_current, k=2)
        good = [first for first, second in matches if first.distance < 0.72 * second.distance]

    homography = None
    inlier_ratio = 0.0
    if len(good) >= 8:
        source = np.float32([key_previous[match.queryIdx].pt for match in good])
        target = np.float32([key_current[match.trainIdx].pt for match in good])
        homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if homography is not None and inliers is not None:
            inlier_ratio = float(np.mean(inliers))

    success = homography is not None
    transform = homography if success else np.eye(3, dtype=np.float64)
    warped = cv2.warpPerspective(gray_previous, transform, (size, size))
    valid = cv2.warpPerspective(np.ones_like(gray_previous), transform, (size, size)) > 0
    compensated = cv2.absdiff(warped, gray_current).astype(np.float64) / 255.0
    raw = cv2.absdiff(gray_previous, gray_current).astype(np.float64) / 255.0
    corridor, lower = corridor_masks(size)
    compensated_corridor = corridor & valid
    compensated_lower = lower & valid
    if not compensated_corridor.any() or not compensated_lower.any():
        compensated_corridor = corridor
        compensated_lower = lower

    key_minimum = max(1, min(len(key_previous), len(key_current)))
    match_support = len(good) / key_minimum
    values = [
        float(success),
        float(len(key_previous) / 2000.0),
        float(len(key_current) / 2000.0),
        float(len(good) / 2000.0),
        float(match_support),
        inlier_ratio,
        float(valid[corridor].mean()),
    ]
    values.extend(region_stats(compensated, compensated_corridor))
    values.extend(region_stats(compensated, compensated_lower))
    values.extend(region_stats(raw, corridor))
    values.extend(region_stats(raw, lower))
    vector = np.asarray(values, dtype=np.float64)
    return vector, {
        "homography_success": success,
        "previous_keypoints": len(key_previous),
        "current_keypoints": len(key_current),
        "good_matches": len(good),
        "match_support": float(match_support),
        "inlier_ratio": inlier_ratio,
        "compensated_lower_mean": float(region_stats(compensated, compensated_lower)[0]),
        "compensated_lower_fraction_ge_018": float(region_stats(compensated, compensated_lower)[4]),
    }


def episode_vector(pair_vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(pair_vectors, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("motion episode needs at least one frame transition")
    return np.concatenate([
        values.mean(axis=0),
        values.max(axis=0),
        values.min(axis=0),
        values[-1] - values[0],
    ])


def compact_episode_vector(pair_vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(pair_vectors, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] != 31:
        raise ValueError("compact motion episode needs Nx31 frame-transition features")
    success = values[:, 0] > 0.5
    reliable_lower_mean = values[success, 13] if np.any(success) else np.zeros(1, dtype=np.float64)
    reliable_lower_q90 = values[success, 15] if np.any(success) else np.zeros(1, dtype=np.float64)
    reliable_lower_fraction = values[success, 17] if np.any(success) else np.zeros(1, dtype=np.float64)
    return np.asarray([
        float(success.mean()),
        float(1.0 - success.mean()),
        float(values[:, 4].min()),
        float(values[:, 5].mean()),
        float(reliable_lower_mean.max()),
        float(reliable_lower_q90.max()),
        float(reliable_lower_fraction.max()),
    ], dtype=np.float64)


def extract_motion_features(
    episodes: Sequence[dict[str, Any]],
    *,
    size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    compact_vectors: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for episode in episodes:
        images = [cv2.imread(frame["path"], cv2.IMREAD_COLOR) for frame in episode["frames"]]
        if len(images) < 2 or any(image is None for image in images):
            raise ValueError(f"motion episode has missing or insufficient images: {episode['episode_id']}")
        pair_vectors: list[np.ndarray] = []
        pair_summaries: list[dict[str, Any]] = []
        for previous, current in zip(images, images[1:]):
            vector, summary = frame_pair_descriptor(previous, current, size=size)
            pair_vectors.append(vector)
            pair_summaries.append(summary)
        stacked = np.stack(pair_vectors)
        vectors.append(episode_vector(stacked))
        compact_vectors.append(compact_episode_vector(stacked))
        summaries.append({
            "episode_id": episode["episode_id"],
            "transition_count": len(pair_vectors),
            "homography_success_count": sum(row["homography_success"] for row in pair_summaries),
            "minimum_good_matches": min(row["good_matches"] for row in pair_summaries),
            "mean_good_matches": float(np.mean([row["good_matches"] for row in pair_summaries])),
            "mean_match_support": float(np.mean([row["match_support"] for row in pair_summaries])),
            "mean_inlier_ratio": float(np.mean([row["inlier_ratio"] for row in pair_summaries])),
            "maximum_compensated_lower_mean": max(row["compensated_lower_mean"] for row in pair_summaries),
            "maximum_compensated_lower_fraction_ge_018": max(
                row["compensated_lower_fraction_ge_018"] for row in pair_summaries
            ),
        })
    return np.stack(vectors), np.stack(compact_vectors), summaries


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
    for path in (args.package_root, args.detector_weights, args.cache_dir, args.output):
        reject_independent_direction(path)
    package_root = args.package_root.resolve()
    detector_weights = args.detector_weights.resolve()
    if not package_root.is_dir() or not detector_weights.is_file():
        raise FileNotFoundError("package root or detector weights are missing")
    episodes, excluded = common.load_episode_specs(package_root)
    labels = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("motion occupancy probe requires both classes")

    motion_features, compact_motion_features, motion_summaries = extract_motion_features(
        episodes, size=args.motion_size,
    )
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    from ultralytics import YOLO
    detector = YOLO(str(detector_weights))
    trajectory_features, detection_summaries = trajectory.extract(
        detector, episodes, image_size=args.image_size, confidence=args.confidence,
    )

    motion_evaluation = evaluate(motion_features, labels, episodes, ridge=args.ridge)
    compact_motion_evaluation = evaluate(compact_motion_features, labels, episodes, ridge=args.ridge)
    fusion_features = np.concatenate([trajectory_features, motion_features], axis=1)
    fusion_evaluation = evaluate(fusion_features, labels, episodes, ridge=args.ridge)
    compact_fusion_features = np.concatenate([trajectory_features, compact_motion_features], axis=1)
    compact_fusion_evaluation = evaluate(compact_fusion_features, labels, episodes, ridge=args.ridge)
    metrics = compact_fusion_evaluation["metrics"]
    gate = bool(
        compact_fusion_evaluation["repeat_exact"]
        and metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and metrics["candidate_alert_recall"] >= args.minimum_class_recall
        and metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(motion_features, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(compact_motion_features, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(trajectory_features, dtype="<f8").tobytes(order="C"))
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
            "motion": "ORB ratio matches + RANSAC homography + compensated/raw grayscale change in fixed near-field corridor",
            "motion_size": args.motion_size,
            "motion_dimension": int(motion_features.shape[1]),
            "compact_motion_dimension": int(compact_motion_features.shape[1]),
            "compact_motion_contract": "registration success/failure, minimum match support, mean inlier ratio, and maximum reliable compensated lower-corridor residual",
            "detector_model": detector_weights.name,
            "detector_weights_sha256": common.sha256_file(detector_weights),
            "trajectory_dimension": int(trajectory_features.shape[1]),
            "fusion_dimension": int(fusion_features.shape[1]),
            "compact_fusion_dimension": int(compact_fusion_features.shape[1]),
            "feature_sha256": digest.hexdigest(),
            "trainable_parameters": 0,
            "isolation": "independent-direction code, data, weights, metrics, and outputs are not read",
        },
        "motion_only_evaluation": motion_evaluation,
        "compact_motion_only_evaluation": compact_motion_evaluation,
        "trajectory_plus_motion_evaluation": fusion_evaluation,
        "trajectory_plus_compact_motion_evaluation": compact_fusion_evaluation,
        "linear_separability_gate": {
            "passed": gate,
            "thresholds": {
                "balanced_accuracy_gte": args.minimum_balanced_accuracy,
                "each_class_recall_gte": args.minimum_class_recall,
            },
        },
        "episode_motion_summaries": motion_summaries,
        "episode_detection_summaries": detection_summaries,
        "episodes": [
            {key: value for key, value in row.items() if key != "frames"} | {"frame_count": len(row["frames"])}
            for row in episodes
        ],
        "excluded_abstentions": excluded,
        "evidence_limit": "Sparse provisional episodes; image registration failure is a feature, not event truth or promotion evidence.",
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
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-size", type=int, choices=(320,), default=320)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or args.ridge <= 0:
        parser.error("confidence must be in (0,1) and ridge must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "episode_count": report["episode_count"],
        "motion_balanced_accuracy": report["motion_only_evaluation"]["metrics"]["balanced_accuracy"],
        "fusion_balanced_accuracy": report["trajectory_plus_motion_evaluation"]["metrics"]["balanced_accuracy"],
        "compact_motion_balanced_accuracy": report["compact_motion_only_evaluation"]["metrics"]["balanced_accuracy"],
        "compact_fusion_balanced_accuracy": report["trajectory_plus_compact_motion_evaluation"]["metrics"]["balanced_accuracy"],
        "linear_separable": report["linear_separability_gate"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
