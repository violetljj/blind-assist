#!/usr/bin/env python3
"""Evaluate RGB-only observations against ADT GT in a separate process."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

from mine_goal_episodes import RGB_STREAM, csv_rows, sha256
from run_rgb_observer import iou
import zipfile


def longest_false_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else None


def nearest_index(times: list[int], timestamp: int) -> int | None:
    position = bisect.bisect_left(times, timestamp)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
    if not candidates:
        return None
    index = min(candidates, key=lambda item: abs(times[item] - timestamp))
    return index if abs(times[index] - timestamp) <= 20_000_000 else None


def aligned_row(frame: dict, gt: dict | None, width: float, height: float) -> dict:
    gt_visible = gt is not None and float(gt["visibility_ratio[%]"]) >= 0.10
    gt_box = None if gt is None else [
        height - float(gt["y_max[pixel]"]),
        float(gt["x_min[pixel]"]),
        height - float(gt["y_min[pixel]"]),
        float(gt["x_max[pixel]"]),
    ]
    predicted_box = frame["bbox_xyxy"]
    overlap = iou(predicted_box, gt_box) if predicted_box is not None and gt_box is not None else 0.0
    localized = predicted_box is not None and overlap >= 0.10
    if gt_visible and predicted_box is not None and gt_box is not None:
        predicted_center = ((predicted_box[0] + predicted_box[2]) / 2.0 - width / 2.0) / (width / 2.0)
        truth_center = ((gt_box[0] + gt_box[2]) / 2.0 - width / 2.0) / (width / 2.0)
        bearing_error = abs(predicted_center - truth_center)
        predicted_scale = float(frame["relative_nearness"])
        truth_scale = math.sqrt(max(0.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) / (width * height)))
    else:
        bearing_error = predicted_scale = truth_scale = None
    return {"gt_visible": gt_visible, "predicted_visible": predicted_box is not None, "localized": localized, "iou": overlap, "bearing_error_normalized": bearing_error, "predicted_scale": predicted_scale, "truth_scale": truth_scale, "observation_quality": float(frame["observation_quality"])}


def metrics(rows: list[dict]) -> dict:
    visible_rows = [row for row in rows if row["gt_visible"]]
    invisible_rows = [row for row in rows if not row["gt_visible"]]
    localized_mask = [row["localized"] for row in visible_rows]
    bearing_errors = [row["bearing_error_normalized"] for row in rows if row["bearing_error_normalized"] is not None]
    scale_rows = [row for row in rows if row["predicted_scale"] is not None]
    visible_segments = []
    start = None
    for index, row in enumerate([*rows, {"gt_visible": False}]):
        if row["gt_visible"] and start is None:
            start = index
        elif not row["gt_visible"] and start is not None:
            visible_segments.append((start, index))
            start = None
    reacquisition_delays = []
    for previous, current in zip(visible_segments, visible_segments[1:]):
        if current[0] - previous[1] < 6:
            continue
        localized = next((index for index in range(current[0], current[1]) if rows[index]["localized"]), None)
        reacquisition_delays.append(None if localized is None else localized - current[0])
    lag = 15
    approach_matches = []
    for index in range(lag, len(rows)):
        earlier, later = rows[index - lag], rows[index]
        if any(value is None for value in (earlier["predicted_scale"], later["predicted_scale"], earlier["truth_scale"], later["truth_scale"])):
            continue
        truth_delta = later["truth_scale"] - earlier["truth_scale"]
        if abs(truth_delta) < 0.002:
            continue
        predicted_delta = later["predicted_scale"] - earlier["predicted_scale"]
        approach_matches.append((predicted_delta > 0) == (truth_delta > 0))
    localized_quality = [row["observation_quality"] for row in visible_rows if row["localized"]]
    missed_quality = [row["observation_quality"] for row in visible_rows if not row["localized"]]
    return {
        "evaluated_frames": len(rows),
        "gt_visible_frames": len(visible_rows),
        "localized_recall_iou_0_10": sum(localized_mask) / len(localized_mask) if localized_mask else None,
        "false_visible_rate_when_gt_invisible": sum(row["predicted_visible"] for row in invisible_rows) / len(invisible_rows) if invisible_rows else None,
        "longest_localization_dropout_frames_while_gt_visible": longest_false_run(localized_mask),
        "mean_iou_when_gt_visible": sum(row["iou"] for row in visible_rows) / len(visible_rows) if visible_rows else None,
        "mean_abs_bearing_error_normalized": sum(bearing_errors) / len(bearing_errors) if bearing_errors else None,
        "bbox_scale_correlation": pearson([row["predicted_scale"] for row in scale_rows], [row["truth_scale"] for row in scale_rows]),
        "eligible_reacquisition_count": len(reacquisition_delays),
        "reacquisition_success_within_30_frames": sum(delay is not None and delay <= 30 for delay in reacquisition_delays) / len(reacquisition_delays) if reacquisition_delays else None,
        "reacquisition_delay_frames": reacquisition_delays,
        "approach_direction_accuracy_lag15": sum(approach_matches) / len(approach_matches) if approach_matches else None,
        "approach_direction_comparison_count": len(approach_matches),
        "observation_quality_mean_when_localized": sum(localized_quality) / len(localized_quality) if localized_quality else None,
        "observation_quality_mean_when_missed": sum(missed_quality) / len(missed_quality) if missed_quality else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--target-uid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prediction = json.loads(args.observations.read_text(encoding="utf-8"))
    if prediction.get("groundtruth_argument_supported") is not False:
        raise ValueError("observer firewall declaration missing")
    with zipfile.ZipFile(args.groundtruth) as zf:
        member = "2d_bounding_box_with_skeleton.csv" if "2d_bounding_box_with_skeleton.csv" in zf.namelist() else "2d_bounding_box.csv"
        rgb_rows = [row for row in csv_rows(zf, member) if row["stream_id"] == RGB_STREAM]
        trajectory_times = [int(row["tracking_timestamp_us"]) * 1000 for row in csv_rows(zf, "aria_trajectory.csv")]
    target_by_index = {}
    for row in rgb_rows:
        if str(row["object_uid"]) != args.target_uid:
            continue
        index = nearest_index(trajectory_times, int(row["timestamp[ns]"]))
        if index is not None:
            target_by_index[index] = row
    frames = prediction["frames"]
    width = float(prediction.get("frame_size", {}).get("width", 1408))
    height = float(prediction.get("frame_size", {}).get("height", 1408))
    calibration_count = max(1, len(trajectory_times) // 4)
    maximum_offset = max(0, len(frames) - len(trajectory_times))
    scores = []
    for offset in range(maximum_offset + 1):
        rows = [aligned_row(frames[offset + index], target_by_index.get(index), width, height) for index in range(calibration_count)]
        scores.append(sum(row["iou"] for row in rows if row["gt_visible"]))
    offset = max(range(len(scores)), key=lambda item: (scores[item], -item))
    aligned = min(len(trajectory_times), len(frames) - offset)
    calibration_rows = [aligned_row(frames[offset + index], target_by_index.get(index), width, height) for index in range(min(calibration_count, aligned))]
    evaluation_rows = [aligned_row(frames[offset + index], target_by_index.get(index), width, height) for index in range(calibration_count, aligned)]
    report_metrics = metrics(evaluation_rows)
    report_metrics.update({"aligned_frames": aligned, "prediction_frames": len(frames), "gt_trajectory_frames": len(trajectory_times), "gt_target_rows": len(target_by_index)})
    output = {
        "schema_version": "ba_adt_rgb_observation_evaluation_v2",
        "route": "BA-ADT-REAL-EVIDENCE", "stage": "ADT-1-CANARY",
        "inputs": {"observations_sha256": sha256(args.observations), "groundtruth_sha256": sha256(args.groundtruth), "target_uid": args.target_uid},
        "isolation": {"observer_received_gt": False, "evaluator_received_rgb_pixels": False},
        "alignment": "fixed_frame_offset_selected_on_first_gt_quarter_then_evaluated_on_remaining_frames",
        "alignment_offset_frames": offset,
        "alignment_calibration_frames": len(calibration_rows),
        "alignment_calibration_metrics": metrics(calibration_rows),
        "evaluation_start_gt_frame": calibration_count,
        "gt_to_preview_coordinate_transform": "rotate_90_degrees_clockwise_x_prime_equals_height_minus_y_y_prime_equals_x",
        "metrics": report_metrics,
        "claim_ceiling": "development_rgb_observation_diagnostic_no_navigation_or_metric_bearing_claim",
        "terminal": "ADT1_RGB_OBSERVATIONS_EVALUATED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": output["terminal"], "alignment_offset_frames": offset, "metrics": report_metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
