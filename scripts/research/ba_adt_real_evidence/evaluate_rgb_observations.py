#!/usr/bin/env python3
"""Evaluate RGB-only observations against ADT GT in a separate process."""

from __future__ import annotations

import argparse
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
    frame_times = sorted({int(row["timestamp[ns]"]) for row in rgb_rows})
    target_by_time = {int(row["timestamp[ns]"]): row for row in rgb_rows if str(row["object_uid"]) == args.target_uid}
    frames = prediction["frames"]
    width = float(prediction.get("frame_size", {}).get("width", 1408))
    height = float(prediction.get("frame_size", {}).get("height", 1408))
    aligned = min(len(frames), len(frame_times))
    evaluated = []
    predicted_scales, truth_scales = [], []
    for index in range(aligned):
        gt = target_by_time.get(frame_times[index])
        gt_visible = gt is not None and float(gt["visibility_ratio[%]"]) >= 0.10
        gt_box = None if gt is None else [float(gt["x_min[pixel]"]), height - float(gt["y_max[pixel]"]), float(gt["x_max[pixel]"]), height - float(gt["y_min[pixel]"])]
        predicted_box = frames[index]["bbox_xyxy"]
        overlap = iou(predicted_box, gt_box) if predicted_box is not None and gt_box is not None else 0.0
        localized = predicted_box is not None and overlap >= 0.10
        if gt_visible and predicted_box is not None and gt_box is not None:
            predicted_center = ((predicted_box[0] + predicted_box[2]) / 2.0 - width / 2.0) / (width / 2.0)
            truth_center = ((gt_box[0] + gt_box[2]) / 2.0 - width / 2.0) / (width / 2.0)
            bearing_error = abs(predicted_center - truth_center)
            predicted_scales.append(float(frames[index]["relative_nearness"]))
            truth_scales.append(math.sqrt(max(0.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) / (width * height))))
        else:
            bearing_error = None
        evaluated.append({"gt_visible": gt_visible, "predicted_visible": predicted_box is not None, "localized": localized, "iou": overlap, "bearing_error_normalized": bearing_error})

    visible_rows = [row for row in evaluated if row["gt_visible"]]
    invisible_rows = [row for row in evaluated if not row["gt_visible"]]
    localized_mask = [row["localized"] for row in visible_rows]
    bearing_errors = [row["bearing_error_normalized"] for row in evaluated if row["bearing_error_normalized"] is not None]
    metrics = {
        "aligned_frames": aligned,
        "prediction_frames": len(frames),
        "gt_target_rows": len(target_by_time),
        "gt_visible_frames": len(visible_rows),
        "localized_recall_iou_0_10": sum(localized_mask) / len(localized_mask) if localized_mask else None,
        "false_visible_rate_when_gt_invisible": sum(row["predicted_visible"] for row in invisible_rows) / len(invisible_rows) if invisible_rows else None,
        "longest_localization_dropout_frames_while_gt_visible": longest_false_run(localized_mask),
        "mean_iou_when_gt_visible": sum(row["iou"] for row in visible_rows) / len(visible_rows) if visible_rows else None,
        "mean_abs_bearing_error_normalized": sum(bearing_errors) / len(bearing_errors) if bearing_errors else None,
        "bbox_scale_correlation": pearson(predicted_scales, truth_scales),
    }
    output = {
        "schema_version": "ba_adt_rgb_observation_evaluation_v1",
        "route": "BA-ADT-REAL-EVIDENCE", "stage": "ADT-1-CANARY",
        "inputs": {"observations_sha256": sha256(args.observations), "groundtruth_sha256": sha256(args.groundtruth), "target_uid": args.target_uid},
        "isolation": {"observer_received_gt": False, "evaluator_received_rgb_pixels": False},
        "alignment": "preview_mp4_frame_index_to_ordered_gt_rgb_timestamp_proxy",
        "gt_to_preview_coordinate_transform": "x_same_y_flipped_about_frame_height",
        "metrics": metrics,
        "claim_ceiling": "sample_rgb_detector_localization_diagnostic_no_navigation_or_metric_bearing_claim",
        "terminal": "ADT1_RGB_OBSERVATIONS_EVALUATED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": output["terminal"], "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
