#!/usr/bin/env python3
"""Evaluate a fixed prefix-only camera scale calibration on consumed clearance."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

BANDS = ("left", "center", "right")
HORIZONS_M = (1.0, 1.5, 2.0)
CALIBRATION_FRAMES = 10


def candidate_field(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("candidate", row.get("metric3d", {}))


def calibration_ratios(rows: list[dict[str, Any]]) -> list[float]:
    ratios = []
    for row in rows[:CALIBRATION_FRAMES]:
        candidate = candidate_field(row)
        sensor = row["sensor"]
        if candidate.get("status") != "VALID" or sensor.get("status") != "VALID":
            continue
        for band in BANDS:
            predicted = candidate["bands"][band]["clearance_m"]
            truth = sensor["bands"][band]["clearance_m"]
            if predicted is not None and truth is not None and predicted > 0:
                ratios.append(float(truth) / float(predicted))
    return ratios


def evaluate(
    report: dict[str, Any], calibration_scope: str = "per_sequence"
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in report["frames"]:
        groups.setdefault(str(row["sequence_id"]), []).append(row)
    if not groups:
        raise ValueError("report contains no frames")
    if calibration_scope not in {"per_sequence", "global_first_sequence"}:
        raise ValueError(f"unsupported calibration scope: {calibration_scope}")
    for rows in groups.values():
        rows.sort(key=lambda row: float(row["timestamp"]))

    ordered_sequences = sorted(groups)
    global_scale = None
    global_calibration_sequence = None
    if calibration_scope == "global_first_sequence":
        global_calibration_sequence = ordered_sequences[0]
        ratios = calibration_ratios(groups[global_calibration_sequence])
        if not ratios:
            raise ValueError("global calibration sequence has no valid pairs")
        global_scale = float(statistics.median(ratios))

    clear_errors: list[float] = []
    decisions: list[bool] = []
    false_clear: list[bool] = []
    delta_errors: list[float] = []
    scales: dict[str, float] = {}
    calibration_pairs = 0
    evaluation_frames = 0
    paired_valid_frames = 0
    previous: dict[tuple[str, str], tuple[float, float]] = {}

    for sequence in ordered_sequences:
        rows = groups[sequence]
        if len(rows) <= CALIBRATION_FRAMES:
            raise ValueError(f"{sequence} has insufficient frames")
        if calibration_scope == "per_sequence":
            ratios = calibration_ratios(rows)
            if not ratios:
                raise ValueError(f"{sequence} has no valid calibration pairs")
            scales[sequence] = float(statistics.median(ratios))
            calibration_pairs += len(ratios)
            evaluation_rows = rows[CALIBRATION_FRAMES:]
        else:
            scales[sequence] = float(global_scale)
            evaluation_rows = (
                rows[CALIBRATION_FRAMES:]
                if sequence == global_calibration_sequence
                else rows
            )
            if sequence == global_calibration_sequence:
                calibration_pairs = len(calibration_ratios(rows))

        for row in evaluation_rows:
            evaluation_frames += 1
            candidate = candidate_field(row)
            sensor = row["sensor"]
            if candidate.get("status") != "VALID" or sensor.get("status") != "VALID":
                continue
            paired_valid_frames += 1
            for band in BANDS:
                predicted = candidate["bands"][band]["clearance_m"]
                truth = sensor["bands"][band]["clearance_m"]
                if predicted is None or truth is None:
                    continue
                calibrated = float(predicted) * scales[sequence]
                truth_value = float(truth)
                clear_errors.append(abs(calibrated - truth_value))
                key = (sequence, band)
                if key in previous:
                    previous_truth, previous_candidate = previous[key]
                    delta_errors.append(
                        abs(
                            (calibrated - previous_candidate)
                            - (truth_value - previous_truth)
                        )
                    )
                previous[key] = (truth_value, calibrated)
                for horizon in HORIZONS_M:
                    truth_occupied = truth_value <= horizon
                    predicted_occupied = calibrated <= horizon
                    decisions.append(truth_occupied == predicted_occupied)
                    false_clear.append(truth_occupied and not predicted_occupied)

    if not clear_errors or not decisions or not delta_errors:
        raise ValueError("evaluation contains insufficient paired observations")
    summary = {
        "schema": "hftf_camera_scale_calibrated_clearance_r0",
        "source_report_status": report.get("status"),
        "protocol": {
            "calibration_frames_per_sequence": CALIBRATION_FRAMES,
            "calibration_statistic": "median sensor_clearance / candidate_clearance",
            "calibration_scope": calibration_scope,
            "global_calibration_sequence": global_calibration_sequence,
            "shared_scale_across_bands": True,
            "threshold_or_model_search": False,
            "data_role": "already consumed development diagnostic",
        },
        "sequences": len(groups),
        "calibration_pairs": calibration_pairs,
        "evaluation_frames": evaluation_frames,
        "paired_valid_frames": paired_valid_frames,
        "paired_valid_fraction": paired_valid_frames / evaluation_frames,
        "scales": scales,
        "clearance_mae_m": statistics.fmean(clear_errors),
        "collision_agreement": statistics.fmean(decisions),
        "false_clear_rate": statistics.fmean(false_clear),
        "temporal_clearance_delta_mae_m": statistics.fmean(delta_errors),
    }
    summary["gates"] = {
        "paired_valid_fraction_at_least_0_90": summary["paired_valid_fraction"]
        >= 0.90,
        "clearance_mae_at_most_0_25m": summary["clearance_mae_m"] <= 0.25,
        "collision_agreement_at_least_0_90": summary["collision_agreement"]
        >= 0.90,
        "false_clear_rate_at_most_0_05": summary["false_clear_rate"] <= 0.05,
        "temporal_delta_mae_at_most_0_15m": summary[
            "temporal_clearance_delta_mae_m"
        ]
        <= 0.15,
    }
    summary["status"] = (
        "CAMERA_SCALE_CALIBRATED_CLEARANCE_DEVELOPMENT_PASS"
        if all(summary["gates"].values())
        else "CAMERA_SCALE_CALIBRATED_CLEARANCE_DEVELOPMENT_FAIL"
    )
    summary["claim_ceiling"] = (
        "same-source prefix calibration on already-consumed development frames only; "
        "no unseen camera, final-camera, alert, safety, or production authority"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-scope",
        choices=("per_sequence", "global_first_sequence"),
        default="per_sequence",
    )
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.report.read_text(encoding="utf-8")),
        args.calibration_scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
