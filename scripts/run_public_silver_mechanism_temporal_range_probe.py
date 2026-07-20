#!/usr/bin/env python3
"""Audit two source-normalized temporal risk channels on qualified pairs.

Dynamic pairs use the within-episode range of the maximum frozen-object
area-times-corridor-overlap. Static pairs use the within-episode range of
camera-motion-compensated lower-corridor residual. Each held pair is evaluated
with a log-midpoint threshold derived only from the other source pairs of the
same mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_motion_compensated_occupancy_probe as motion
import run_public_silver_object_trajectory_probe as trajectory


SCHEMA = "blindassist_public_silver_mechanism_temporal_range_probe_v1"
DYNAMIC = "dynamic_agent_approach"
STATIC = "static_corridor_narrowing"


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this probe's scope: {path}")


def load_qualified_pair_contract(report_path: Path) -> dict[str, list[str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "blindassist_public_silver_mechanism_coverage_v1":
        raise ValueError("mechanism report schema mismatch")
    if (report.get("mechanism_coverage_gate") or {}).get("passed") is not True:
        raise ValueError("mechanism coverage gate must pass before temporal-range audit")
    threshold = float((report.get("thresholds") or {}).get("minimum_episode_confidence_per_pair", 1.0))
    qualified: dict[str, list[str]] = {DYNAMIC: [], STATIC: []}
    for pair in report.get("matched_pairs", []):
        mechanism = pair.get("mechanism")
        if mechanism in qualified and float(pair.get("minimum_episode_confidence", 0.0)) >= threshold:
            qualified[mechanism].append(pair["counterfactual_pair_id"])
    if any(len(pair_ids) < 3 for pair_ids in qualified.values()):
        raise ValueError("each mechanism needs at least three qualified pairs")
    return qualified


def dynamic_temporal_range(frames: Sequence[Sequence[dict[str, Any]]]) -> float:
    if len(frames) < 2:
        raise ValueError("dynamic temporal range needs at least two frames")
    occupancy = np.asarray([
        max((row["area"] * row["corridor_overlap"] for row in detections), default=0.0)
        for detections in frames
    ], dtype=np.float64)
    return float(np.ptp(occupancy))


def static_temporal_range(images: Sequence[np.ndarray], *, size: int) -> tuple[float, int]:
    if len(images) < 3:
        raise ValueError("static temporal range needs at least three frames")
    reliable: list[float] = []
    for previous, current in zip(images, images[1:]):
        vector, summary = motion.frame_pair_descriptor(previous, current, size=size)
        if summary["homography_success"]:
            reliable.append(float(vector[13]))
    if len(reliable) < 2:
        return 0.0, len(reliable)
    return float(np.ptp(np.asarray(reliable, dtype=np.float64))), len(reliable)


def log_midpoint_threshold(maximum_negative: float, minimum_positive: float) -> float:
    if maximum_negative < 0 or minimum_positive <= maximum_negative:
        raise ValueError("training pair scores are not separable")
    epsilon = 1e-9
    return float(math.exp((math.log(maximum_negative + epsilon) + math.log(minimum_positive + epsilon)) / 2.0))


def leave_one_pair_out(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 3:
        raise ValueError("leave-one-pair-out needs at least three pairs")
    folds: list[dict[str, Any]] = []
    endpoint_correct = 0
    pair_correct = 0
    for held_index, held in enumerate(rows):
        training = [row for index, row in enumerate(rows) if index != held_index]
        maximum_negative = max(float(row["no_alert_score"]) for row in training)
        minimum_positive = min(float(row["alert_score"]) for row in training)
        training_separable = maximum_negative < minimum_positive
        threshold = (
            log_midpoint_threshold(maximum_negative, minimum_positive)
            if training_separable
            else None
        )
        no_prediction = int(threshold is not None and held["no_alert_score"] >= threshold)
        alert_prediction = int(threshold is not None and held["alert_score"] >= threshold)
        no_correct = no_prediction == 0
        alert_correct = alert_prediction == 1
        ordering_correct = held["alert_score"] > held["no_alert_score"]
        endpoint_correct += int(no_correct) + int(alert_correct)
        pair_correct += int(ordering_correct)
        folds.append({
            "held_out_pair_id": held["counterfactual_pair_id"],
            "held_out_source_id": held["source_id"],
            "training_pair_ids": [row["counterfactual_pair_id"] for row in training],
            "training_maximum_no_alert_score": maximum_negative,
            "training_minimum_alert_score": minimum_positive,
            "training_scores_separable": training_separable,
            "threshold": threshold,
            "held_out_no_alert_score": held["no_alert_score"],
            "held_out_alert_score": held["alert_score"],
            "no_alert_prediction": no_prediction,
            "alert_prediction": alert_prediction,
            "endpoint_predictions_correct": no_correct and alert_correct,
            "pair_ordering_correct": ordering_correct,
        })
    return {
        "folds": folds,
        "held_out_pair_count": len(rows),
        "pair_ordering_correct_count": pair_correct,
        "pair_ordering_rate": pair_correct / len(rows),
        "held_out_endpoint_count": 2 * len(rows),
        "held_out_endpoint_correct_count": endpoint_correct,
        "held_out_endpoint_accuracy": endpoint_correct / (2 * len(rows)),
        "all_training_folds_separable": all(row["training_scores_separable"] for row in folds),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.package_root, args.mechanism_report, args.detector_weights,
        args.cache_dir, args.output,
    ):
        reject_independent_direction(path)
    package_root = args.package_root.resolve()
    mechanism_report = args.mechanism_report.resolve()
    detector_weights = args.detector_weights.resolve()
    if not package_root.is_dir() or not mechanism_report.is_file() or not detector_weights.is_file():
        raise FileNotFoundError("package root, mechanism report, or detector weights are missing")

    qualified = load_qualified_pair_contract(mechanism_report)
    episodes, excluded = common.load_episode_specs(package_root)
    by_pair: dict[str, list[int]] = {}
    for index, episode in enumerate(episodes):
        pair_id = episode.get("counterfactual_pair_id")
        if pair_id:
            by_pair.setdefault(pair_id, []).append(index)

    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    from ultralytics import YOLO
    detector = YOLO(str(detector_weights))
    frame_detections = trajectory.extract_frame_detections(
        detector, episodes, image_size=args.image_size, confidence=args.confidence,
    )

    mechanism_rows: dict[str, list[dict[str, Any]]] = {DYNAMIC: [], STATIC: []}
    for mechanism, pair_ids in qualified.items():
        for pair_id in pair_ids:
            indices = by_pair.get(pair_id, [])
            if len(indices) != 2:
                raise ValueError(f"qualified pair must contain exactly two episodes: {pair_id}")
            negative = next((index for index in indices if episodes[index]["label"] == 0), None)
            positive = next((index for index in indices if episodes[index]["label"] == 1), None)
            if negative is None or positive is None:
                raise ValueError(f"qualified pair is missing alert/no-alert endpoints: {pair_id}")
            if episodes[negative]["source_id"] != episodes[positive]["source_id"]:
                raise ValueError(f"qualified pair crosses sources: {pair_id}")

            reliable_transitions: dict[str, int] | None = None
            if mechanism == DYNAMIC:
                negative_score = dynamic_temporal_range(frame_detections[negative])
                positive_score = dynamic_temporal_range(frame_detections[positive])
                score_contract = "range(max_object_area_times_corridor_overlap_per_frame)"
            else:
                negative_images = [
                    cv2.imread(frame["path"], cv2.IMREAD_COLOR) for frame in episodes[negative]["frames"]
                ]
                positive_images = [
                    cv2.imread(frame["path"], cv2.IMREAD_COLOR) for frame in episodes[positive]["frames"]
                ]
                if any(image is None for image in negative_images + positive_images):
                    raise ValueError(f"cannot decode static pair images: {pair_id}")
                negative_score, negative_reliable = static_temporal_range(
                    negative_images, size=args.motion_size,
                )
                positive_score, positive_reliable = static_temporal_range(
                    positive_images, size=args.motion_size,
                )
                reliable_transitions = {
                    "no_alert": negative_reliable,
                    "alert": positive_reliable,
                }
                if min(reliable_transitions.values()) < 2:
                    raise ValueError(f"static pair lacks two reliable registered transitions: {pair_id}")
                score_contract = "range(reliable_registered_lower_corridor_mean_residual)"

            mechanism_rows[mechanism].append({
                "counterfactual_pair_id": pair_id,
                "source_id": episodes[negative]["source_id"],
                "no_alert_episode_id": episodes[negative]["episode_id"],
                "alert_episode_id": episodes[positive]["episode_id"],
                "no_alert_score": negative_score,
                "alert_score": positive_score,
                "score_delta": positive_score - negative_score,
                "score_contract": score_contract,
                "reliable_registered_transitions": reliable_transitions,
            })

    evaluations = {
        mechanism: leave_one_pair_out(rows)
        for mechanism, rows in mechanism_rows.items()
    }
    gate = all(
        evaluation["pair_ordering_rate"] == 1.0
        and evaluation["held_out_endpoint_accuracy"] == 1.0
        and evaluation["all_training_folds_separable"]
        for evaluation in evaluations.values()
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "mechanism_report": str(mechanism_report),
        "mechanism_report_sha256": common.sha256_file(mechanism_report),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "qualified_pair_contract": qualified,
        "channels": {
            DYNAMIC: {
                "feature": "frozen object proposal occupancy temporal range",
                "score": "peak-to-peak of per-frame max(area*corridor_overlap)",
                "trainable_parameters": 0,
            },
            STATIC: {
                "feature": "camera-motion-compensated near-field residual temporal range",
                "score": "peak-to-peak of reliable registered lower-corridor mean residual",
                "trainable_parameters": 0,
            },
        },
        "pair_scores": mechanism_rows,
        "leave_one_pair_out": evaluations,
        "temporal_range_gate": {
            "passed": gate,
            "requirements": {
                "pair_ordering_rate": 1.0,
                "held_out_endpoint_accuracy": 1.0,
                "all_training_folds_separable": True,
            },
        },
        "runtime_interpretation": {
            "absolute_scene_classifier_recommended": False,
            "mechanism_specific_channels_recommended": gate,
            "rolling_or_recent_safe_baseline_required": True,
            "pixel_segmentation_role": "auxiliary_only",
            "next_head_contract": "two temporal risk channels with counterfactual ranking plus lifecycle change supervision",
        },
        "isolation_contract": {
            "public_video_mainline_only": True,
            "independent_model_direction_data_used": False,
            "independent_model_direction_code_used": False,
            "independent_model_direction_metrics_used_as_gate": False,
        },
        "evidence_limit": "Six qualified GPT/VLM provisional pairs only; mechanism-channel feasibility, not human truth, calibration, blind evaluation, or production promotion.",
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
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320)
    parser.add_argument("--motion-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error("confidence must be in (0,1)")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "temporal_range_gate": report["temporal_range_gate"]["passed"],
        "dynamic_pair_ordering_rate": report["leave_one_pair_out"][DYNAMIC]["pair_ordering_rate"],
        "static_pair_ordering_rate": report["leave_one_pair_out"][STATIC]["pair_ordering_rate"],
        "dynamic_endpoint_accuracy": report["leave_one_pair_out"][DYNAMIC]["held_out_endpoint_accuracy"],
        "static_endpoint_accuracy": report["leave_one_pair_out"][STATIC]["held_out_endpoint_accuracy"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
