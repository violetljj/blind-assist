#!/usr/bin/env python3
"""Probe a fixed multi-channel episode risk profile.

The profile joins frozen COCO object trajectories with sixteen preregistered,
interpretable static channels: registered local change, absolute clearance,
adaptive-path occupancy, and lateral detour.  No feature, mask, class, or
threshold is selected from the labels.  Rice Street is external pressure only
and is never included in a fitted fold.
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

import run_public_silver_background_normalized_static_probe as background
import run_public_silver_frozen_feature_probe as common
import run_public_silver_object_trajectory_probe as trajectory
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_segformer_free_space_probe as clearance


SCHEMA = "blindassist_public_silver_multichannel_risk_profile_probe_v1"
BACKGROUND_FIELDS = (
    "median_mean_excess",
    "q75_mean_excess",
    "median_q90_excess",
    "q75_q90_excess",
)
CLEARANCE_FIELDS = (
    "median_lower_nonwalkable_mean",
    "q75_lower_nonwalkable_mean",
    "median_core_nonwalkable_mean",
    "q75_core_nonwalkable_mean",
)
PATH_FIELDS = (
    "median_path_nonwalkable_mean",
    "q75_path_nonwalkable_mean",
    "median_path_lower_nonwalkable_mean",
    "q75_path_lower_nonwalkable_mean",
)
DETOUR_FIELDS = (
    "median_path_offset_mean",
    "q75_path_offset_mean",
    "median_path_offset_maximum",
    "q75_path_offset_maximum",
)
PROFILE_FIELDS = BACKGROUND_FIELDS + CLEARANCE_FIELDS + PATH_FIELDS + DETOUR_FIELDS


def compact_profile(
    background_scores: dict[str, float | int | None],
    clearance_scores: dict[str, float | int],
) -> np.ndarray:
    values: list[float] = []
    for key in BACKGROUND_FIELDS:
        value = background_scores.get(key)
        if value is None:
            # Registration failure is encoded explicitly as zero local excess;
            # reliability remains governed by the frozen descriptor contract.
            value = 0.0
        values.append(float(value))
    values.extend(float(clearance_scores[key]) for key in CLEARANCE_FIELDS + PATH_FIELDS + DETOUR_FIELDS)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(PROFILE_FIELDS),) or not np.isfinite(result).all():
        raise ValueError("compact profile must be a finite fixed-length vector")
    return result


def extract_profile(
    episodes: Sequence[dict[str, Any]],
    teacher: clearance.FrozenTeacher,
    *,
    motion_size: int,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for episode in episodes:
        images = background._episode_images(episode)
        motion = background.score_descriptors(background.image_descriptors(images, size=motion_size))
        free_space = clearance.score(teacher.describe(images, batch_size=batch_size))
        vector = compact_profile(motion, free_space)
        vectors.append(vector)
        summaries.append({
            "episode_id": episode["episode_id"],
            "background_reliable_transition_count": motion["reliable_transition_count"],
            "background_transition_count": motion["transition_count"],
            "profile": {name: float(value) for name, value in zip(PROFILE_FIELDS, vector)},
        })
    return np.stack(vectors), summaries


def matrix_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def pair_directional_projection(
    features: np.ndarray,
    episodes: Sequence[dict[str, Any]],
    labels: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    fitted = common.fit_episode_ridge(features, labels, ridge=ridge, class_balanced=True)
    direction = fitted["kernel"][:, 1] - fitted["kernel"][:, 0]
    grouped: dict[str, list[int]] = {}
    for index, episode in enumerate(episodes):
        pair_id = episode.get("counterfactual_pair_id")
        if isinstance(pair_id, str) and pair_id:
            grouped.setdefault(pair_id, []).append(index)
    rows: list[dict[str, Any]] = []
    for pair_id, indices in sorted(grouped.items()):
        negative = [index for index in indices if labels[index] == 0]
        positive = [index for index in indices if labels[index] == 1]
        if not negative or not positive:
            continue
        delta = features[positive].mean(axis=0) - features[negative].mean(axis=0)
        projection = float(delta @ direction)
        rows.append({"counterfactual_pair_id": pair_id, "alert_minus_no_alert_projection": projection, "ordered": projection > 0.0})
    return {
        "full_set_direction_coefficient_sha256": fitted["coefficient_sha256"],
        "pairs": rows,
        "ordered_pair_count": sum(row["ordered"] for row in rows),
        "pair_count": len(rows),
        "all_pairs_ordered": bool(rows) and all(row["ordered"] for row in rows),
    }


def evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: Sequence[str],
    source_ids: Sequence[str],
    *,
    ridge: float,
) -> dict[str, Any]:
    first = common.leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=ridge, class_balanced=True)
    second = common.leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=ridge, class_balanced=True)
    return {**first, "repeat_exact": first == second}


def rice_profile(
    video: Path,
    review_path: Path,
    teacher: clearance.FrozenTeacher,
    *,
    motion_size: int,
    batch_size: int,
) -> dict[str, Any]:
    review = common.load_json(review_path).get("review") or {}
    fields = {
        "pre_clear": "pre_risk_clear_window_ms",
        "risk": "risk_present_window_ms",
        "post_clear": "stable_post_clear_window_ms",
    }
    result: dict[str, Any] = {}
    for name, field in fields.items():
        window = review.get(field)
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"Rice review window is invalid: {name}")
        images = background.decode_video_window(video, int(window[0]), int(window[1]), interval_ms=1000)
        motion = background.score_descriptors(background.image_descriptors(images, size=motion_size))
        free_space = clearance.score(teacher.describe(images, batch_size=batch_size))
        vector = compact_profile(motion, free_space)
        result[name] = {"window_ms": window, "profile": vector, "named_profile": dict(zip(PROFILE_FIELDS, vector.tolist()))}
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.detector_weights, args.model_dir, args.rice_video, args.rice_review, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    episodes, excluded = common.load_episode_specs(args.package_root.resolve())
    labels = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    episode_ids = [row["episode_id"] for row in episodes]
    source_ids = [row["source_id"] for row in episodes]

    os.environ["YOLO_CONFIG_DIR"] = str(args.cache_dir.resolve())
    args.cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    detector = YOLO(str(args.detector_weights.resolve()))
    trajectory_features, detection_summaries = trajectory.extract(
        detector, episodes, image_size=args.image_size, confidence=args.confidence
    )
    teacher = clearance.FrozenTeacher(args.model_dir.resolve())
    profile_features, profile_summaries = extract_profile(
        episodes, teacher, motion_size=args.motion_size, batch_size=args.batch_size
    )
    fused_features = np.concatenate([trajectory_features, profile_features], axis=1)

    trajectory_eval = evaluate(trajectory_features, labels, episode_ids, source_ids, ridge=args.ridge)
    profile_eval = evaluate(profile_features, labels, episode_ids, source_ids, ridge=args.ridge)
    fusion_eval = evaluate(fused_features, labels, episode_ids, source_ids, ridge=args.ridge)
    fusion_metrics = fusion_eval["metrics"]
    baseline_balanced = trajectory_eval["metrics"]["balanced_accuracy"]
    gate = bool(
        fusion_eval["repeat_exact"]
        and fusion_metrics["balanced_accuracy"] > baseline_balanced
        and fusion_metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and fusion_metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
        and fusion_metrics["candidate_alert_recall"] >= args.minimum_class_recall
    )

    rice = rice_profile(
        args.rice_video.resolve(), args.rice_review.resolve(), teacher,
        motion_size=args.motion_size, batch_size=args.batch_size,
    )
    fitted = common.fit_episode_ridge(fused_features, labels, ridge=args.ridge, class_balanced=True)
    profile_direction = fitted["kernel"][-len(PROFILE_FIELDS):, 1] - fitted["kernel"][-len(PROFILE_FIELDS):, 0]
    rice_projection = {
        name: float(row["profile"] @ profile_direction)
        for name, row in rice.items()
    }
    rice_pressure = {
        "profile_channel_projection": rice_projection,
        "open_ordered": rice_projection["risk"] > rice_projection["pre_clear"],
        "close_ordered": rice_projection["risk"] > rice_projection["post_clear"],
        "note": "Trajectory channels are unavailable for the long Rice windows; this pressure check projects only the sixteen profile channels and does not affect the linear gate.",
    }

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_feature_root_cause_diagnosis_after_r717_failure",
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "class_counts": {"candidate_no_alert": int(np.sum(labels == 0)), "candidate_alert": int(np.sum(labels == 1))},
        "feature_contract": {
            "trajectory_dimension": int(trajectory_features.shape[1]),
            "risk_profile_dimension": len(PROFILE_FIELDS),
            "fusion_dimension": int(fused_features.shape[1]),
            "risk_profile_fields": list(PROFILE_FIELDS),
            "threshold_fitted": False,
            "mask_or_class_search": False,
            "trainable_feature_parameters": 0,
            "detector_weights_sha256": common.sha256_file(args.detector_weights),
            "segformer_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
            "trajectory_matrix_sha256": matrix_sha256(trajectory_features),
            "profile_matrix_sha256": matrix_sha256(profile_features),
            "fusion_matrix_sha256": matrix_sha256(fused_features),
        },
        "evaluation": {
            "split": "leave_one_source_group_out",
            "group_key": "source_id",
            "ridge": args.ridge,
            "class_balanced": True,
            "trajectory_only": trajectory_eval,
            "risk_profile_only": profile_eval,
            "trajectory_plus_risk_profile": fusion_eval,
        },
        "matched_pair_audit": {
            "profile_delta_alignment": common.counterfactual_delta_alignment(profile_features, episodes),
            "fusion_delta_alignment": common.counterfactual_delta_alignment(fused_features, episodes),
            "profile_directional_projection": pair_directional_projection(profile_features, episodes, labels, ridge=args.ridge),
            "fusion_directional_projection": pair_directional_projection(fused_features, episodes, labels, ridge=args.ridge),
        },
        "rice_external_pressure": rice_pressure,
        "linear_feature_gate": {
            "passed": gate,
            "thresholds": {
                "fusion_balanced_accuracy_gte": args.minimum_balanced_accuracy,
                "fusion_each_class_recall_gte": args.minimum_class_recall,
                "fusion_strictly_improves_trajectory_baseline": True,
                "repeat_exact": True,
            },
            "prototype_bootstrap_authorized": gate,
        },
        "episode_profile_summaries": profile_summaries,
        "episode_detection_summaries": detection_summaries,
        "evidence_limit": "Retrospective 19-episode GPT/VLM-silver diagnosis. It cannot rescue r7.17 and is not calibration, blind evaluation, human accuracy, Android-change, or production-promotion evidence.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rice-video", type=Path, required=True)
    parser.add_argument("--rice-review", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../artifacts.local/cache/ultralytics-risk-profile"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--motion-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if args.ridge <= 0 or not 0 < args.confidence < 1 or args.motion_size < 32 or args.batch_size < 1:
        parser.error("invalid probe settings")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    metrics = report["evaluation"]["trajectory_plus_risk_profile"]["metrics"]
    print(json.dumps({
        "ok": True,
        "linear_feature_gate_passed": report["linear_feature_gate"]["passed"],
        "fusion_balanced_accuracy": metrics["balanced_accuracy"],
        "output_sha256": common.sha256_file(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
