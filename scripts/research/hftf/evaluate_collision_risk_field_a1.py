#!/usr/bin/env python3
"""Compare frozen detector, depth, 3D, and probability collision-risk arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_metric3d_clearance_field_a0 import BANDS, HORIZONS_M
from evaluate_motion_conditioned_occupancy_a0 import (
    EXPECTED_RAFT_SHA256,
    FEATURE_NAMES,
    extract_motion,
    sha256,
)
from produce_external_rgb_metric_depth_observations import UniDepthSource


SCHEMA = "blindassist_hftf_collision_risk_field_a1"
EXPECTED_YOLO_SHA256 = (
    "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
)
ARM_NAMES = (
    "yolo_center_box",
    "bbox_unidepth",
    "unidepth_2d_corridor",
    "unidepth_3d_envelope",
    "motion_probability_field",
    "sensor_depth_oracle",
)


def image_third_band(center_x: float, width: int) -> str:
    fraction = center_x / width
    if fraction < 1.0 / 3.0:
        return "left"
    if fraction < 2.0 / 3.0:
        return "center"
    return "right"


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    decisions = scores >= threshold
    truth = labels.astype(bool)
    tp = int(np.sum(decisions & truth))
    tn = int(np.sum(~decisions & ~truth))
    fp = int(np.sum(decisions & ~truth))
    fn = int(np.sum(~decisions & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": 1.0 - specificity,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def frozen_probability(features: np.ndarray, model: dict[str, Any]) -> float:
    if model["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("frozen feature order mismatch")
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights_intercept_then_features"], dtype=np.float64)
    design = np.concatenate(([1.0], (features - mean) / scale))
    return float(1.0 / (1.0 + np.exp(-np.clip(design @ weights, -40, 40))))


def two_d_clearances(
    depth: np.ndarray, fx: float, cx: float
) -> dict[str, float | None]:
    height, width = depth.shape
    rows, columns = np.mgrid[0:height, 0:width]
    lateral = (columns - cx) * depth / fx
    valid = (
        np.isfinite(depth)
        & (depth > 0.1)
        & (depth < 10.0)
        & (rows >= int(0.10 * height))
        & (rows < int(0.90 * height))
    )
    output: dict[str, float | None] = {}
    for band, (minimum, maximum) in BANDS.items():
        values = depth[valid & (lateral >= minimum) & (lateral < maximum)]
        output[band] = float(np.quantile(values, 0.02)) if len(values) else None
    return output


def detection_depths(
    boxes: np.ndarray, confidences: np.ndarray, depth: np.ndarray, fx: float, cx: float
) -> list[dict[str, float | str]]:
    height, width = depth.shape
    output: list[dict[str, float | str]] = []
    for box, confidence in zip(boxes, confidences, strict=True):
        x1, y1, x2, y2 = [float(value) for value in box]
        roi_x1 = max(0, min(width - 1, int(x1 + 0.25 * (x2 - x1))))
        roi_x2 = max(roi_x1 + 1, min(width, int(x2 - 0.25 * (x2 - x1))))
        roi_y1 = max(0, min(height - 1, int(y1 + 0.25 * (y2 - y1))))
        roi_y2 = max(roi_y1 + 1, min(height, int(y2 - 0.25 * (y2 - y1))))
        values = depth[roi_y1:roi_y2, roi_x1:roi_x2]
        values = values[np.isfinite(values) & (values > 0.1) & (values < 10.0)]
        if not len(values):
            continue
        distance = float(np.median(values))
        center_x = 0.5 * (x1 + x2)
        lateral = (center_x - cx) * distance / fx
        band = next(
            (name for name, (minimum, maximum) in BANDS.items() if minimum <= lateral < maximum),
            None,
        )
        if band is not None:
            output.append(
                {"band": band, "depth_m": distance, "confidence": float(confidence)}
            )
    return output


def evaluate(
    report: dict[str, Any],
    model: dict[str, Any],
    raft_weights: Path,
    yolo_weights: Path,
    unidepth: UniDepthSource,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict[str, Any]:
    if sha256(raft_weights) != EXPECTED_RAFT_SHA256:
        raise ValueError("unexpected RAFT-small checkpoint")
    if sha256(yolo_weights) != EXPECTED_YOLO_SHA256:
        raise ValueError("unexpected YOLO11n checkpoint")
    if model.get("raft_sha256") != EXPECTED_RAFT_SHA256:
        raise ValueError("frozen model RAFT identity mismatch")

    frames = report["frames"]
    motion = extract_motion(frames, raft_weights)
    from ultralytics import YOLO

    detector = YOLO(str(yolo_weights), task="detect")
    paths = [frame["frame_path"] for frame in frames]
    detector_results = detector.predict(
        source=paths, imgsz=320, conf=0.25, device=0, verbose=False, stream=True
    )
    score_rows = {name: [] for name in ARM_NAMES}
    labels: list[float] = []
    details = []

    for frame, detection in zip(frames, detector_results, strict=True):
        bgr = cv2.imread(frame["frame_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame["frame_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        row = {"intrinsics_fx_fy_cx_cy": [fx, fy, cx, cy]}
        depth, _ = unidepth.infer(rgb, row)
        boxes = detection.boxes.xyxy.detach().cpu().numpy()
        confidences = detection.boxes.conf.detach().cpu().numpy()
        thirds = {name: 0.0 for name in BANDS}
        for box, confidence in zip(boxes, confidences, strict=True):
            name = image_third_band(0.5 * float(box[0] + box[2]), depth.shape[1])
            thirds[name] = max(thirds[name], float(confidence))
        box_depths = detection_depths(boxes, confidences, depth, fx, cx)
        corridor = two_d_clearances(depth, fx, cx)

        if frame["sensor"]["status"] != "VALID" or frame["metric3d"]["status"] != "VALID":
            continue
        predicted = frame["metric3d"]
        for band in BANDS:
            truth_clearance = frame["sensor"]["bands"][band]["clearance_m"]
            predicted_clearance = predicted["bands"][band]["clearance_m"]
            confidence = predicted["bands"][band].get("clearance_log1p_confidence")
            if truth_clearance is None or predicted_clearance is None or confidence is None:
                continue
            for horizon in HORIZONS_M:
                label = float(float(truth_clearance) <= horizon)
                static = np.asarray(
                    [
                        float(predicted_clearance) - horizon,
                        float(predicted_clearance),
                        horizon,
                        float(confidence),
                        float(predicted["ground_plane_median_residual_m"]),
                        math.log1p(int(predicted["bands"][band]["obstacle_points"])),
                        float(band == "left"),
                        float(band == "center"),
                    ],
                    dtype=np.float64,
                )
                probability = frozen_probability(
                    np.concatenate((static, motion[frame["frame_path"]])), model
                )
                box_score = max(
                    (
                        float(value["confidence"])
                        for value in box_depths
                        if value["band"] == band and float(value["depth_m"]) <= horizon
                    ),
                    default=0.0,
                )
                corridor_occupied = corridor[band] is not None and float(corridor[band]) <= horizon
                envelope_occupied = float(predicted_clearance) <= horizon
                scores = {
                    "yolo_center_box": thirds[band],
                    "bbox_unidepth": box_score,
                    "unidepth_2d_corridor": 0.999 if corridor_occupied else 0.001,
                    "unidepth_3d_envelope": 0.999 if envelope_occupied else 0.001,
                    "motion_probability_field": probability,
                    "sensor_depth_oracle": label,
                }
                labels.append(label)
                for name, value in scores.items():
                    score_rows[name].append(value)
                details.append(
                    {
                        "frame_path": frame["frame_path"],
                        "sequence_id": frame["sequence_id"],
                        "timestamp": frame["timestamp"],
                        "band": band,
                        "horizon_m": horizon,
                        "label_occupied": bool(label),
                        "scores": scores,
                    }
                )

    label_array = np.asarray(labels, dtype=np.float64)
    thresholds = {
        "yolo_center_box": 0.25,
        "bbox_unidepth": 0.25,
        "unidepth_2d_corridor": 0.50,
        "unidepth_3d_envelope": 0.50,
        "motion_probability_field": 0.50,
        "sensor_depth_oracle": 0.50,
    }
    arms = {
        name: binary_metrics(label_array, np.asarray(score_rows[name]), thresholds[name])
        for name in ARM_NAMES
    }
    candidate = arms["motion_probability_field"]
    deterministic = arms["unidepth_3d_envelope"]
    comparators = [
        arms[name]["mcc"]
        for name in ARM_NAMES
        if name not in ("motion_probability_field", "sensor_depth_oracle")
    ]
    brier_reduction = (
        deterministic["brier_score"] - candidate["brier_score"]
    ) / deterministic["brier_score"]
    gates = {
        "known_opportunities_at_least_1500": len(labels) >= 1500,
        "brier_reduction_vs_3d_at_least_0_15": brier_reduction >= 0.15,
        "occupied_recall_at_least_0_85": candidate["recall"] >= 0.85,
        "false_positive_rate_at_most_0_15": candidate["false_positive_rate"] <= 0.15,
        "mcc_strictly_best_non_oracle": candidate["mcc"] > max(comparators),
    }
    return {
        "schema": SCHEMA,
        "opportunities": len(labels),
        "arms": arms,
        "probability_brier_reduction_vs_3d": brier_reduction,
        "gates": gates,
        "details": details,
        "status": (
            "COLLISION_RISK_FIELD_A1_DEVELOPMENT_PASS"
            if all(gates.values())
            else "COLLISION_RISK_FIELD_A1_DEVELOPMENT_FAIL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--unidepth-repo", type=Path, required=True)
    parser.add_argument("--unidepth-model-name", default="lpiccinelli/unidepth-v2-vits14")
    parser.add_argument("--unidepth-resolution-level", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--intrinsics-fx-fy-cx-cy", nargs=4, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    source = UniDepthSource(
        args.unidepth_repo,
        args.unidepth_model_name,
        args.unidepth_resolution_level,
        args.device,
    )
    result = evaluate(
        json.loads(args.report.read_text(encoding="utf-8")),
        model,
        args.raft_weights,
        args.yolo_weights,
        source,
        *args.intrinsics_fx_fy_cx_cy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "details"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
