#!/usr/bin/env python3
"""Window-LOSO low-capacity occupancy probability with RAFT motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.optimize import minimize
from torchvision.models.optical_flow import raft_small

from evaluate_metric3d_clearance_field_a0 import BANDS, HORIZONS_M
from evaluate_metric3d_probabilistic_occupancy_a0 import (
    _log_loss,
    expected_calibration_error,
    load_report,
)


SCHEMA = "blindassist_hftf_motion_conditioned_occupancy_a0"
EXPECTED_RAFT_SHA256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
FEATURE_NAMES = (
    "clearance_margin_m",
    "clearance_m",
    "horizon_m",
    "clearance_log1p_confidence",
    "ground_plane_residual_m",
    "log1p_obstacle_points",
    "band_left",
    "band_center",
    "flow_median",
    "flow_p90",
    "affine_tx_norm",
    "affine_ty_norm",
    "abs_affine_rotation_rad",
    "abs_affine_log_scale",
    "affine_inlier_fraction",
    "residual_flow_median",
    "residual_flow_p90",
    "motion_missing",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_tensor(path: str) -> torch.Tensor:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 128), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div(127.5).sub(1.0)


def motion_summary(flow: np.ndarray) -> np.ndarray:
    _, height, width = flow.shape
    magnitude = np.sqrt((flow[0] / width) ** 2 + (flow[1] / height) ** 2)
    y, x = np.mgrid[4:height:8, 4:width:8]
    source = np.stack((x.ravel(), y.ravel()), axis=1).astype(np.float32)
    sampled = flow[:, y, x].transpose(1, 2, 0).reshape(-1, 2)
    target = source + sampled.astype(np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        translation = np.median(sampled, axis=0)
        matrix = np.asarray([[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]])
        inlier_fraction = 0.0
    else:
        inlier_fraction = float(np.mean(inliers))
    full_y, full_x = np.mgrid[0:height, 0:width]
    predicted = np.stack(
        (
            matrix[0, 0] * full_x + matrix[0, 1] * full_y + matrix[0, 2] - full_x,
            matrix[1, 0] * full_x + matrix[1, 1] * full_y + matrix[1, 2] - full_y,
        )
    )
    residual = flow - predicted
    residual_magnitude = np.sqrt((residual[0] / width) ** 2 + (residual[1] / height) ** 2)
    scale = max(1e-8, math.sqrt(float(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)))
    rotation = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return np.asarray(
        [
            float(np.median(magnitude)),
            float(np.quantile(magnitude, 0.90)),
            float(matrix[0, 2] / width),
            float(matrix[1, 2] / height),
            abs(rotation),
            abs(math.log(scale)),
            inlier_fraction,
            float(np.median(residual_magnitude)),
            float(np.quantile(residual_magnitude, 0.90)),
            0.0,
        ],
        dtype=np.float64,
    )


def extract_motion(
    frames: list[dict[str, Any]], weights: Path, batch_size: int = 16
) -> dict[str, np.ndarray]:
    if sha256(weights) != EXPECTED_RAFT_SHA256:
        raise ValueError("unexpected RAFT-small checkpoint")
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_sequence[str(frame["sequence_id"])].append(frame)
    pairs = []
    output = {}
    for sequence, values in sorted(by_sequence.items()):
        values.sort(key=lambda value: float(value["timestamp"]))
        output[values[0]["frame_path"]] = np.asarray([0.0] * 9 + [1.0])
        for previous, current in zip(values, values[1:]):
            pairs.append((previous["frame_path"], current["frame_path"]))
    model = raft_small(weights=None, progress=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            previous = torch.stack([_image_tensor(pair[0]) for pair in batch]).to(device)
            current = torch.stack([_image_tensor(pair[1]) for pair in batch]).to(device)
            flows = model(previous, current)[-1].cpu().numpy()
            for pair, flow in zip(batch, flows, strict=True):
                output[pair[1]] = motion_summary(flow)
    if len(output) != len(frames):
        raise RuntimeError("motion feature coverage mismatch")
    return output


def build_rows(reports: list[dict[str, Any]], motion: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    groups = []
    for report in reports:
        for frame in report["frames"]:
            if frame["sensor"]["status"] != "VALID" or frame["metric3d"]["status"] != "VALID":
                continue
            model = frame["metric3d"]
            for band in BANDS:
                truth = frame["sensor"]["bands"][band]["clearance_m"]
                predicted = model["bands"][band]["clearance_m"]
                confidence = model["bands"][band].get("clearance_log1p_confidence")
                if truth is None or predicted is None or confidence is None:
                    continue
                for horizon in HORIZONS_M:
                    static = [
                        float(predicted) - horizon,
                        float(predicted),
                        horizon,
                        float(confidence),
                        float(model["ground_plane_median_residual_m"]),
                        math.log1p(int(model["bands"][band]["obstacle_points"])),
                        float(band == "left"),
                        float(band == "center"),
                    ]
                    features.append(np.concatenate((np.asarray(static), motion[frame["frame_path"]])))
                    labels.append(float(truth) <= horizon)
                    groups.append(str(frame["sequence_id"]))
    return np.stack(features), np.asarray(labels, dtype=np.float64), np.asarray(groups)


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 0.01,
    positive_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    sample_weight = np.where(y == 1.0, positive_weight, 1.0)
    weight_sum = float(np.sum(sample_weight))
    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        loss = float(
            np.sum(sample_weight * (np.logaddexp(0.0, logits) - y * logits))
            / weight_sum
            + 0.5 * l2 * np.sum(weights[1:] ** 2)
        )
        gradient = design.T @ (sample_weight * (probabilities - y)) / weight_sum
        gradient[1:] += l2 * weights[1:]
        return loss, gradient
    result = minimize(lambda value: objective(value), np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return result.x, mean, scale


def predict_logistic(x: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, mean, scale = fitted
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    return 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -40, 40)))


def evaluate(
    reports: list[dict[str, Any]],
    weights_path: Path,
    positive_weight: float = 1.0,
) -> dict[str, Any]:
    frames_by_path = {}
    for report in reports:
        for frame in report["frames"]:
            frames_by_path.setdefault(frame["frame_path"], frame)
    motion = extract_motion(list(frames_by_path.values()), weights_path)
    x, y, groups = build_rows(reports, motion)
    probabilities = np.zeros(len(y), dtype=np.float64)
    fold_rows = {}
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        fitted = fit_logistic(
            x[train], y[train], positive_weight=positive_weight
        )
        probabilities[test] = predict_logistic(x[test], fitted)
        fold_rows[group] = {"train_rows": int(np.sum(train)), "test_rows": int(np.sum(test))}
    baseline = np.where(x[:, 0] <= 0, 0.999, 0.001)
    brier = float(np.mean((probabilities - y) ** 2))
    baseline_brier = float(np.mean((baseline - y) ** 2))
    log_loss = _log_loss(probabilities.tolist(), y.astype(bool).tolist())
    baseline_log_loss = _log_loss(baseline.tolist(), y.astype(bool).tolist())
    high_clear = probabilities <= 0.05
    positive = y == 1
    recall = float(np.mean(probabilities[positive] >= 0.50))
    false_clear = float(np.mean(y[high_clear])) if int(np.sum(high_clear)) else None
    effects = {
        "brier_reduction_vs_deterministic": (baseline_brier - brier) / baseline_brier,
        "log_loss_reduction_vs_deterministic": (baseline_log_loss - log_loss) / baseline_log_loss,
    }
    ece = expected_calibration_error(probabilities.tolist(), y.astype(bool).tolist())
    gates = {
        "brier_reduction_at_least_0_15": effects["brier_reduction_vs_deterministic"] >= 0.15,
        "log_loss_reduction_at_least_0_20": effects["log_loss_reduction_vs_deterministic"] >= 0.20,
        "ece_at_most_0_10": ece <= 0.10,
        "high_clear_false_clear_at_most_0_05": false_clear is not None and false_clear <= 0.05,
        "high_clear_coverage_at_least_0_10": float(np.mean(high_clear)) >= 0.10,
        "occupied_recall_at_least_0_85": recall >= 0.85,
    }
    final_fit = fit_logistic(x, y, positive_weight=positive_weight)
    return {
        "schema": SCHEMA,
        "features": list(FEATURE_NAMES),
        "groups": len(set(groups.tolist())),
        "opportunities": len(y),
        "folds": fold_rows,
        "brier_score": brier,
        "deterministic_brier_score": baseline_brier,
        "log_loss": log_loss,
        "deterministic_log_loss": baseline_log_loss,
        "expected_calibration_error": ece,
        "high_clear_coverage": float(np.mean(high_clear)),
        "high_clear_false_clear_rate": false_clear,
        "occupied_recall_at_probability_0_50": recall,
        "effects": effects,
        "gates": gates,
        "final_model": {
            "weights_intercept_then_features": final_fit[0].tolist(),
            "feature_mean": final_fit[1].tolist(),
            "feature_scale": final_fit[2].tolist(),
            "l2": 0.01,
            "positive_weight": positive_weight,
        },
        "status": "MOTION_CONDITIONED_OCCUPANCY_A0_WINDOW_LOSO_PASS" if all(gates.values()) else "MOTION_CONDITIONED_OCCUPANCY_A0_WINDOW_LOSO_FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, nargs="+", required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    args = parser.parse_args()
    result = evaluate(
        [load_report(path) for path in args.report],
        args.raft_weights,
        args.positive_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"folds", "final_model"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
