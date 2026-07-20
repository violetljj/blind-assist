#!/usr/bin/env python3
"""Rotation-invariant OFAT for the ADVIO causal turn-intent diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_advio_turn_intent_probe as base


SCHEMA = "blindassist_public_visual_inertial_turn_intent_invariant_probe_v1"


def scalar_statistics(timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    centered_time = timestamps - timestamps.mean()
    denominator = float(np.sum(centered_time * centered_time))
    slope = float(np.sum(centered_time * (values - values.mean())) / denominator)
    return np.asarray([values.mean(), values.std(), values[-1], values.min(), values.max(), slope])


def invariant_window_statistics(sensor: np.ndarray, end: float, duration: float) -> np.ndarray:
    rows = sensor[(sensor[:, 0] > end - duration) & (sensor[:, 0] <= end)]
    if len(rows) < 3:
        raise ValueError(f"insufficient sensor history at {end}")
    magnitude = np.linalg.norm(rows[:, 1:4], axis=1)
    delta_magnitude = np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1) / np.diff(rows[:, 0])
    return np.concatenate([
        scalar_statistics(rows[:, 0], magnitude),
        np.asarray([delta_magnitude.mean(), delta_magnitude.std(), delta_magnitude[-1],
                    delta_magnitude.min(), delta_magnitude.max()]),
    ])


def build_examples(root: Path, sampling: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_x, labels, timestamps, angles = base.build_examples(root, sampling)
    del raw_x
    gyro = np.loadtxt(root / "iphone/gyro.csv", delimiter=",")
    accelerometer = np.loadtxt(root / "iphone/accelerometer.csv", delimiter=",")
    duration = float(sampling["past_imu_window_seconds"])
    features = np.stack([
        np.concatenate([invariant_window_statistics(gyro, timestamp, duration),
                        invariant_window_statistics(accelerometer, timestamp, duration)])
        for timestamp in timestamps
    ])
    return features, labels, timestamps, angles


def run(contract_path: Path, r794_contract_path: Path, root: Path,
        output_cache: Path, output_report: Path) -> dict[str, Any]:
    for path in (output_cache, output_report, Path(str(output_report) + ".sha256")):
        if path.exists():
            raise ValueError("refusing to overwrite invariant probe output")
    contract = base.load_json(contract_path)
    bound = contract["bound_inputs"]
    if base.file_hash(r794_contract_path) != bound["r794_contract_sha256"]:
        raise ValueError("r794 contract hash mismatch")
    failure_path = Path(bound["r794_failure_report_path"])
    if base.file_hash(failure_path) != bound["r794_failure_report_sha256"]:
        raise ValueError("r794 failure report hash mismatch")
    failure = base.load_json(failure_path)
    if failure.get("single_sequence_feasibility_passed") is not False:
        raise ValueError("r794 is not a frozen failure")
    parent = base.load_json(r794_contract_path)
    x, y, timestamps, angles = build_examples(root, parent["sampling"])
    probe = parent["probe"]
    scores, folds = base.contiguous_oof(x, y, timestamps, int(probe["contiguous_fold_count"]),
                                        float(probe["guard_seconds"]), float(probe["alpha"]))
    predicted = (scores >= float(probe["decision_threshold"])).astype(np.int64)
    turn_recall = float((predicted[y == 1] == 1).mean())
    straight_recall = float((predicted[y == 0] == 0).mean())
    auroc = base.roc_auc(y, scores)
    balanced = (turn_recall + straight_recall) / 2.0
    previous_cache = np.load(failure["cache"]["path"])
    identity_matches = (np.array_equal(timestamps, previous_cache["timestamps"]) and
                        np.array_equal(y, previous_cache["labels"]) and
                        np.allclose(angles, previous_cache["future_course_angle_degrees"], atol=1e-5))
    gate = contract["feasibility_gate"]
    checks = {
        "oof_auroc": auroc >= float(gate["oof_auroc_at_least"]),
        "oof_balanced_accuracy": balanced >= float(gate["oof_balanced_accuracy_at_least"]),
        "turn_recall": turn_recall >= float(gate["turn_recall_at_least"]),
        "straight_recall": straight_recall >= float(gate["straight_recall_at_least"]),
        "sample_identity_matches_r794": bool(identity_matches),
        "all_predictions_finite": bool(np.isfinite(scores).all()),
    }
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, features=x.astype(np.float32), labels=y, timestamps=timestamps,
                        future_course_angle_degrees=angles.astype(np.float32), oof_scores=scores.astype(np.float32))
    report = {
        "schema": SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": contract_path.as_posix(), "contract_sha256": base.file_hash(contract_path),
        "source_id": failure["source_id"], "sample_count": int(len(y)), "feature_dimension": int(x.shape[1]),
        "turn_sample_count": int(y.sum()), "straight_sample_count": int((y == 0).sum()),
        "metrics": {"oof_auroc": auroc, "oof_balanced_accuracy": balanced,
                    "turn_recall": turn_recall, "straight_recall": straight_recall},
        "delta_vs_r794": {"oof_auroc": auroc - float(failure["metrics"]["oof_auroc"]),
                           "oof_balanced_accuracy": balanced - float(failure["metrics"]["oof_balanced_accuracy"])},
        "folds": folds, "checks": checks, "single_sequence_feasibility_passed": bool(all(checks.values())),
        "cache": {"path": output_cache.as_posix(), "sha256": base.file_hash(output_cache)},
        "authorization": contract["authorization"],
        "limitations": failure["limitations"] + [
            "A failed result closes further linear-head variants on this single sequence; a passed result still requires independent sequences."
        ],
    }
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = base.file_hash(output_report)
    Path(str(output_report) + ".sha256").write_text(f"{digest}  {output_report.name}\n", encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r794-contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.contract, args.r794_contract, args.root, args.output_cache, args.output_report)
    print(json.dumps({"metrics": report["metrics"], "delta_vs_r794": report["delta_vs_r794"],
                      "single_sequence_feasibility_passed": report["single_sequence_feasibility_passed"]}))


if __name__ == "__main__":
    main()
