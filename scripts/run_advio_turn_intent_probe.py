#!/usr/bin/env python3
"""Probe advance route-turn signal from causal IMU on one isolated ADVIO sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "blindassist_public_visual_inertial_turn_intent_probe_v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def signed_angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = np.sum(first * second, axis=1)
    return np.degrees(np.arctan2(cross, dot))


def interpolate_xz(pose: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(timestamps, pose[:, 0], pose[:, 1]),
        np.interp(timestamps, pose[:, 0], pose[:, 3]),
    ])


def window_statistics(sensor: np.ndarray, end: float, duration: float) -> np.ndarray:
    rows = sensor[(sensor[:, 0] > end - duration) & (sensor[:, 0] <= end)]
    if len(rows) < 3:
        raise ValueError(f"insufficient sensor history at {end}")
    time = rows[:, 0] - rows[:, 0].mean()
    denominator = float(np.sum(time * time))
    values = rows[:, 1:4]
    slope = np.sum(time[:, None] * (values - values.mean(axis=0)), axis=0) / denominator
    return np.concatenate([
        values.mean(axis=0), values.std(axis=0), values[-1],
        values.min(axis=0), values.max(axis=0), slope,
    ])


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("AUROC requires both classes")
    return float((np.sum(positives[:, None] > negatives[None, :]) +
                  0.5 * np.sum(positives[:, None] == negatives[None, :])) /
                 (len(positives) * len(negatives)))


def fit_weighted_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = np.asarray([len(y) / (2.0 * counts[label]) for label in y])
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ (weights[:, None] * design) + regularizer,
                                   design.T @ (weights * y.astype(np.float64)))
    return coefficients, mean, scale


def predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coefficients, mean, scale = model
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ coefficients


def contiguous_oof(x: np.ndarray, y: np.ndarray, timestamps: np.ndarray, folds: int,
                   guard_seconds: float, alpha: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    boundaries = np.linspace(timestamps.min(), timestamps.max() + 1e-9, folds + 1)
    scores = np.full(len(x), np.nan, dtype=np.float64)
    reports = []
    for fold in range(folds):
        low, high = boundaries[fold], boundaries[fold + 1]
        test = (timestamps >= low) & ((timestamps < high) if fold < folds - 1 else (timestamps <= high))
        train = ((timestamps < low - guard_seconds) | (timestamps >= high + guard_seconds))
        classes = sorted(set(y[train].tolist()))
        if classes != [0, 1]:
            raise ValueError(f"training fold {fold} lacks a class: {classes}")
        model = fit_weighted_ridge(x[train], y[train], alpha)
        scores[test] = predict(model, x[test])
        reports.append({
            "fold": fold, "test_start_seconds": float(low), "test_end_seconds": float(high),
            "train_count": int(train.sum()), "test_count": int(test.sum()),
            "train_turn_count": int(y[train].sum()), "test_turn_count": int(y[test].sum()),
        })
    return scores, reports


def build_examples(root: Path, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gyro = np.loadtxt(root / "iphone/gyro.csv", delimiter=",")
    accelerometer = np.loadtxt(root / "iphone/accelerometer.csv", delimiter=",")
    pose = np.loadtxt(root / "ground-truth/pose.csv", delimiter=",")
    past = float(spec["past_imu_window_seconds"])
    recent = float(spec["recent_course_window_seconds"])
    future_start = float(spec["future_course_start_seconds"])
    future_end = float(spec["future_course_end_seconds"])
    start = max(gyro[0, 0], accelerometer[0, 0], pose[0, 0]) + max(past, recent)
    end = min(gyro[-1, 0], accelerometer[-1, 0], pose[-1, 0]) - future_end
    timestamps = np.arange(start, end, float(spec["step_seconds"]))
    recent_vector = interpolate_xz(pose, timestamps) - interpolate_xz(pose, timestamps - recent)
    future_vector = interpolate_xz(pose, timestamps + future_end) - interpolate_xz(pose, timestamps + future_start)
    angles = signed_angle_degrees(recent_vector, future_vector)
    valid = ((np.linalg.norm(recent_vector, axis=1) >= float(spec["minimum_recent_displacement_meters"])) &
             (np.linalg.norm(future_vector, axis=1) >= float(spec["minimum_future_displacement_meters"])))
    timestamps = timestamps[valid]
    angles = angles[valid]
    features = np.stack([
        np.concatenate([window_statistics(gyro, timestamp, past),
                        window_statistics(accelerometer, timestamp, past)])
        for timestamp in timestamps
    ])
    labels = (np.abs(angles) >= float(spec["turn_threshold_degrees"])).astype(np.int64)
    return features, labels, timestamps, angles


def run(contract_path: Path, root: Path, output_cache: Path, output_report: Path) -> dict[str, Any]:
    for path in (output_cache, output_report, Path(str(output_report) + ".sha256")):
        if path.exists():
            raise ValueError("refusing to overwrite turn-intent output")
    contract = load_json(contract_path)
    bound = contract["bound_inputs"]
    audit_path = Path(bound["acquisition_audit_path"])
    if file_hash(audit_path) != bound["acquisition_audit_sha256"]:
        raise ValueError("acquisition audit hash mismatch")
    audit = load_json(audit_path)
    if audit.get("audit_passed") is not True:
        raise ValueError("acquisition audit did not pass")
    for relative, key in [("iphone/gyro.csv", "gyro_sha256"),
                          ("iphone/accelerometer.csv", "accelerometer_sha256"),
                          ("ground-truth/pose.csv", "ground_truth_pose_sha256")]:
        if file_hash(root / relative) != bound[key]:
            raise ValueError(f"input hash mismatch: {relative}")
    x, y, timestamps, angles = build_examples(root, contract["sampling"])
    probe = contract["probe"]
    scores, folds = contiguous_oof(x, y, timestamps, int(probe["contiguous_fold_count"]),
                                   float(probe["guard_seconds"]), float(probe["alpha"]))
    if not np.isfinite(scores).all():
        raise ValueError("non-finite OOF prediction")
    predicted = (scores >= float(probe["decision_threshold"])).astype(np.int64)
    turn_recall = float((predicted[y == 1] == 1).mean())
    straight_recall = float((predicted[y == 0] == 0).mean())
    auroc = roc_auc(y, scores)
    balanced = (turn_recall + straight_recall) / 2.0
    gate = contract["feasibility_gate"]
    checks = {
        "minimum_retained_samples": len(y) >= int(gate["minimum_retained_samples"]),
        "minimum_turn_samples": int(y.sum()) >= int(gate["minimum_turn_samples"]),
        "minimum_straight_samples": int((y == 0).sum()) >= int(gate["minimum_straight_samples"]),
        "oof_auroc": auroc >= float(gate["oof_auroc_at_least"]),
        "oof_balanced_accuracy": balanced >= float(gate["oof_balanced_accuracy_at_least"]),
        "all_training_folds_have_both_classes": all(row["train_turn_count"] not in (0, row["train_count"]) for row in folds),
        "all_predictions_finite": bool(np.isfinite(scores).all()),
    }
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, features=x.astype(np.float32), labels=y, timestamps=timestamps,
                        future_course_angle_degrees=angles.astype(np.float32), oof_scores=scores.astype(np.float32))
    report = {
        "schema": SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": contract_path.as_posix(), "contract_sha256": file_hash(contract_path),
        "source_id": audit["source_id"], "sample_count": int(len(y)), "feature_dimension": int(x.shape[1]),
        "turn_sample_count": int(y.sum()), "straight_sample_count": int((y == 0).sum()),
        "metrics": {"oof_auroc": auroc, "oof_balanced_accuracy": balanced,
                    "turn_recall": turn_recall, "straight_recall": straight_recall},
        "folds": folds, "checks": checks, "single_sequence_feasibility_passed": bool(all(checks.values())),
        "cache": {"path": output_cache.as_posix(), "sha256": file_hash(output_cache)},
        "authorization": contract["authorization"],
        "limitations": [
            "One contiguous sequence can establish pipeline feasibility only; it cannot establish cross-source generalization.",
            "Future ground-truth course is an auxiliary label only and is never an input feature.",
            "This target represents route turning, not obstacle risk, warning need, or event truth.",
            "CC BY-NC 4.0 prohibits production or commercial use of this source."
        ],
    }
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = file_hash(output_report)
    Path(str(output_report) + ".sha256").write_text(f"{digest}  {output_report.name}\n", encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.contract, args.root, args.output_cache, args.output_report)
    print(json.dumps({"sample_count": report["sample_count"], "metrics": report["metrics"],
                      "single_sequence_feasibility_passed": report["single_sequence_feasibility_passed"]}))


if __name__ == "__main__":
    main()
