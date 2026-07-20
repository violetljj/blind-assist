#!/usr/bin/env python3
"""Test causal IMU confirmation of a route turn already underway."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_advio_turn_intent_invariant_probe as invariant
import run_advio_turn_intent_probe as base


SCHEMA = "blindassist_public_visual_inertial_turn_confirmation_probe_v1"


def build_examples(root: Path, target: dict[str, Any], past_window: float,
                   step_seconds: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gyro = np.loadtxt(root / "iphone/gyro.csv", delimiter=",")
    accelerometer = np.loadtxt(root / "iphone/accelerometer.csv", delimiter=",")
    pose = np.loadtxt(root / "ground-truth/pose.csv", delimiter=",")
    oldest = float(target["prior_course_window_seconds"][0])
    start = max(gyro[0, 0], accelerometer[0, 0], pose[0, 0]) + max(oldest, past_window)
    end = min(gyro[-1, 0], accelerometer[-1, 0], pose[-1, 0])
    timestamps = np.arange(start, end, step_seconds)
    prior = base.interpolate_xz(pose, timestamps - 0.5) - base.interpolate_xz(pose, timestamps - 1.0)
    recent = base.interpolate_xz(pose, timestamps) - base.interpolate_xz(pose, timestamps - 0.5)
    angles = base.signed_angle_degrees(prior, recent)
    minimum = float(target["minimum_segment_displacement_meters"])
    valid = (np.linalg.norm(prior, axis=1) >= minimum) & (np.linalg.norm(recent, axis=1) >= minimum)
    timestamps = timestamps[valid]
    angles = angles[valid]
    features = np.stack([
        np.concatenate([invariant.invariant_window_statistics(gyro, timestamp, past_window),
                        invariant.invariant_window_statistics(accelerometer, timestamp, past_window)])
        for timestamp in timestamps
    ])
    labels = (np.abs(angles) >= float(target["turn_threshold_degrees"])).astype(np.int64)
    return features, labels, timestamps, angles


def run(contract_path: Path, r794_contract_path: Path, root: Path,
        output_cache: Path, output_report: Path) -> dict[str, Any]:
    for path in (output_cache, output_report, Path(str(output_report) + ".sha256")):
        if path.exists():
            raise ValueError("refusing to overwrite turn-confirmation output")
    contract = base.load_json(contract_path)
    bound = contract["bound_inputs"]
    if base.file_hash(r794_contract_path) != bound["r794_contract_sha256"]:
        raise ValueError("r794 contract hash mismatch")
    failure_path = Path(bound["r795_failure_report_path"])
    if base.file_hash(failure_path) != bound["r795_failure_report_sha256"]:
        raise ValueError("r795 failure report hash mismatch")
    failure = base.load_json(failure_path)
    if failure.get("single_sequence_feasibility_passed") is not False:
        raise ValueError("r795 is not a frozen failure")
    parent = base.load_json(r794_contract_path)
    probe = parent["probe"]
    x, y, timestamps, angles = build_examples(
        root, contract["target"], float(contract["input_and_probe"]["past_imu_window_seconds"])
    )
    scores, folds = base.contiguous_oof(x, y, timestamps, int(probe["contiguous_fold_count"]),
                                        float(probe["guard_seconds"]), float(probe["alpha"]))
    predicted = (scores >= float(contract["input_and_probe"]["decision_threshold"])).astype(np.int64)
    turn_recall = float((predicted[y == 1] == 1).mean())
    straight_recall = float((predicted[y == 0] == 0).mean())
    auroc = base.roc_auc(y, scores)
    balanced = (turn_recall + straight_recall) / 2.0
    gate = contract["feasibility_gate"]
    checks = {
        "minimum_retained_samples": len(y) >= int(gate["minimum_retained_samples"]),
        "minimum_turn_samples": int(y.sum()) >= int(gate["minimum_turn_samples"]),
        "minimum_straight_samples": int((y == 0).sum()) >= int(gate["minimum_straight_samples"]),
        "oof_auroc": auroc >= float(gate["oof_auroc_at_least"]),
        "oof_balanced_accuracy": balanced >= float(gate["oof_balanced_accuracy_at_least"]),
        "turn_recall": turn_recall >= float(gate["turn_recall_at_least"]),
        "straight_recall": straight_recall >= float(gate["straight_recall_at_least"]),
        "all_predictions_finite": bool(np.isfinite(scores).all()),
    }
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, features=x.astype(np.float32), labels=y, timestamps=timestamps,
                        causal_course_angle_degrees=angles.astype(np.float32), oof_scores=scores.astype(np.float32))
    report = {
        "schema": SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": contract_path.as_posix(), "contract_sha256": base.file_hash(contract_path),
        "source_id": failure["source_id"], "sample_count": int(len(y)), "feature_dimension": int(x.shape[1]),
        "turn_sample_count": int(y.sum()), "straight_sample_count": int((y == 0).sum()),
        "metrics": {"oof_auroc": auroc, "oof_balanced_accuracy": balanced,
                    "turn_recall": turn_recall, "straight_recall": straight_recall},
        "folds": folds, "checks": checks, "single_sequence_turn_confirmation_passed": bool(all(checks.values())),
        "cache": {"path": output_cache.as_posix(), "sha256": base.file_hash(output_cache)},
        "authorization": contract["authorization"],
        "limitations": [
            "The label uses only trajectory up to the current timestamp; no future pose enters label or input.",
            "A pass authorizes only a benchmark-only Android sensor interface, not route prediction or risk output.",
            "One noncommercial ADVIO sequence cannot establish cross-device or production generalization."
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
    print(json.dumps({"metrics": report["metrics"],
                      "single_sequence_turn_confirmation_passed": report["single_sequence_turn_confirmation_passed"]}))


if __name__ == "__main__":
    main()
