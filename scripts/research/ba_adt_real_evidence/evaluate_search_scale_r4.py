#!/usr/bin/env python3
"""Evaluate LOST search scale arms on the fixed ADT R1 failure windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import zipfile

from evaluate_rgb_observations import gt_box, nearest_index
from mine_goal_episodes import RGB_STREAM, csv_rows, sha256
from run_rgb_observer import iou


VISIBILITY_ELIGIBLE = 0.50
SOURCE_MIN_DIMENSION_ELIGIBLE_PX = 4.0
MIN_ELIGIBLE_FRAMES_PER_WINDOW = 3


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_truth(groundtruth: Path, target_uid: str):
    with zipfile.ZipFile(groundtruth) as zf:
        member = "2d_bounding_box_with_skeleton.csv" if "2d_bounding_box_with_skeleton.csv" in zf.namelist() else "2d_bounding_box.csv"
        rgb_rows = [row for row in csv_rows(zf, member) if row["stream_id"] == RGB_STREAM]
        trajectory_times = [int(row["tracking_timestamp_us"]) * 1000 for row in csv_rows(zf, "aria_trajectory.csv")]
    target_by_index = {}
    distractors_by_index = {}
    for row in rgb_rows:
        index = nearest_index(trajectory_times, int(row["timestamp[ns]"]))
        if index is None or float(row["visibility_ratio[%]"]) < 0.10:
            continue
        if str(row["object_uid"]) == target_uid:
            target_by_index[index] = row
        else:
            distractors_by_index.setdefault(index, []).append(row)
    return target_by_index, distractors_by_index


def candidate_state(frame: dict, target_box: list[float], distractors: list[dict], height: float) -> dict:
    trace = frame.get("redetection_trace") or {}
    candidates = trace.get("candidates", [])
    overlaps = [iou(candidate["bbox_xyxy"], target_box) for candidate in candidates]
    predicted_box = frame.get("bbox_xyxy")
    localized = predicted_box is not None and iou(predicted_box, target_box) >= 0.10
    distractor_overlap = max((iou(predicted_box, gt_box(row, height)) for row in distractors), default=0.0) if predicted_box is not None else 0.0
    return {
        "search_active": bool(trace.get("search_active")),
        "correct_candidate": any(value >= 0.10 for value in overlaps),
        "best_candidate_iou": max(overlaps, default=0.0),
        "localized": localized,
        "wrong_instance_output": predicted_box is not None and not localized and distractor_overlap >= 0.10,
        "observation_source": frame.get("observation_source"),
    }


def median_or_none(values: list[float | int]) -> float | None:
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--r1-failure-accounting", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--target-uid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arm_paths = dict(args.arm)
    if len(arm_paths) != len(args.arm):
        raise ValueError("duplicate arm name")
    arms = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in arm_paths.items()}
    for name, arm in arms.items():
        if arm.get("groundtruth_argument_supported") is not False:
            raise ValueError(f"arm {name} is not RGB-only")
        if not arm.get("instance_redetection", {}).get("candidate_diagnostics"):
            raise ValueError(f"arm {name} lacks candidate diagnostics")

    accounting = json.loads(args.r1_failure_accounting.read_text(encoding="utf-8"))
    fixed = [row for row in accounting["metrics"]["opportunities"] if row["outcome"] == "NO_CANDIDATE"]
    if len(fixed) != 5:
        raise ValueError(f"expected five fixed NO_CANDIDATE windows, got {len(fixed)}")
    target_by_index, distractors_by_index = load_truth(args.groundtruth, args.target_uid)
    offset = int(accounting["alignment_offset_frames"])
    evaluation_start = int(accounting["evaluation_start_gt_frame"])
    first_arm = next(iter(arms.values()))
    width = float(first_arm["frame_size"]["width"])
    height = float(first_arm["frame_size"]["height"])

    window_reports = []
    for fixed_window_index, opportunity in enumerate(fixed):
        start = int(opportunity["gt_revisible_evaluation_frame"])
        end = start + int(opportunity["gt_visible_segment_frames"])
        truth_rows = []
        for evaluation_index in range(start, end):
            gt_index = evaluation_start + evaluation_index
            target = target_by_index.get(gt_index)
            if target is None:
                continue
            target_box = gt_box(target, height)
            min_dimension = min(target_box[2] - target_box[0], target_box[3] - target_box[1])
            truth_rows.append({
                "evaluation_index": evaluation_index,
                "preview_index": offset + gt_index,
                "target_box": target_box,
                "distractors": distractors_by_index.get(gt_index, []),
                "visibility": float(target["visibility_ratio[%]"]),
                "source_min_dimension_px": min_dimension,
                "eligible": float(target["visibility_ratio[%]"]) >= VISIBILITY_ELIGIBLE and min_dimension >= SOURCE_MIN_DIMENSION_ELIGIBLE_PX,
            })
        eligible_rows = [row for row in truth_rows if row["eligible"]]
        window_eligible = len(eligible_rows) >= MIN_ELIGIBLE_FRAMES_PER_WINDOW
        first_eligible = eligible_rows[0]["evaluation_index"] if window_eligible else None
        arm_reports = {}
        for arm_name, arm in arms.items():
            rows = []
            for truth in truth_rows:
                preview_index = truth["preview_index"]
                if preview_index >= len(arm["frames"]):
                    raise ValueError(f"arm {arm_name} ends before fixed window {fixed_window_index}")
                rows.append({**truth, **candidate_state(arm["frames"][preview_index], truth["target_box"], truth["distractors"], height)})
            metric_rows = [] if first_eligible is None else [
                row for row in rows if row["evaluation_index"] >= first_eligible
            ]
            first_correct = next((row for row in metric_rows if row["correct_candidate"]), None)
            first_reacquired = next((row for row in metric_rows if row["localized"]), None)
            eligible_search = [row for row in rows if row["eligible"] and row["search_active"]]
            proposal_delta = None if first_eligible is None or first_correct is None else first_correct["evaluation_index"] - first_eligible
            reacquisition_delta = None if first_eligible is None or first_reacquired is None else first_reacquired["evaluation_index"] - first_eligible
            system_miss_censored = first_eligible is not None and first_correct is None
            confirmation_censored = first_correct is not None and first_reacquired is None
            arm_reports[arm_name] = {
                "candidate_recall_on_eligible_lost_search_frames": (sum(row["correct_candidate"] for row in eligible_search) / len(eligible_search)) if eligible_search else None,
                "eligible_lost_search_frames": len(eligible_search),
                "eligible_lost_search_frames_with_correct_candidate": sum(row["correct_candidate"] for row in eligible_search),
                "first_correct_proposal_evaluation_frame": None if first_correct is None else first_correct["evaluation_index"],
                "first_correct_proposal_relative_to_first_eligible_frames": proposal_delta,
                "first_reacquired_evaluation_frame": None if first_reacquired is None else first_reacquired["evaluation_index"],
                "reacquisition_relative_to_first_eligible_frames": reacquisition_delta,
                "confirmation_overhead_frames": None if first_correct is None or first_reacquired is None else first_reacquired["evaluation_index"] - first_correct["evaluation_index"],
                "t_system_miss_frames": proposal_delta,
                "t_system_miss_censored": system_miss_censored,
                "t_system_miss_censor_lower_bound_frames": end - first_eligible if system_miss_censored else None,
                "t_confirmation_frames": None if first_correct is None or first_reacquired is None else first_reacquired["evaluation_index"] - first_correct["evaluation_index"],
                "t_confirmation_censored": confirmation_censored,
                "t_confirmation_censor_lower_bound_frames": end - first_correct["evaluation_index"] if confirmation_censored else None,
                "best_candidate_iou": max((row["best_candidate_iou"] for row in rows), default=0.0),
                "wrong_instance_output_frames": sum(row["wrong_instance_output"] for row in rows),
                "search_frames_total": arm["instance_redetection"].get("search_frames"),
                "inference_images_total": arm["instance_redetection"].get("inference_images"),
            }
        window_reports.append({
            "fixed_window_index": fixed_window_index,
            "original_opportunity_index": opportunity["opportunity_index"],
            "window_eligible": window_eligible,
            "eligible_frame_count": len(eligible_rows),
            "first_eligible_evaluation_frame": first_eligible,
            "gt_invisible_duration_frames": opportunity["preceding_gt_invisible_gap_frames"],
            "t_subdetectable_frames": None if first_eligible is None else first_eligible - start,
            "arms": arm_reports,
        })

    aggregate = {}
    eligible_windows = [row for row in window_reports if row["window_eligible"]]
    for arm_name in arms:
        reports = [row["arms"][arm_name] for row in eligible_windows]
        proposal_deltas = [row["first_correct_proposal_relative_to_first_eligible_frames"] for row in reports if row["first_correct_proposal_relative_to_first_eligible_frames"] is not None]
        reacquisition_deltas = [row["reacquisition_relative_to_first_eligible_frames"] for row in reports if row["reacquisition_relative_to_first_eligible_frames"] is not None]
        confirmation = [row["confirmation_overhead_frames"] for row in reports if row["confirmation_overhead_frames"] is not None]
        lost_search_denominator = sum(row["eligible_lost_search_frames"] for row in reports)
        lost_search_correct = sum(row["eligible_lost_search_frames_with_correct_candidate"] for row in reports)
        aggregate[arm_name] = {
            "eligible_window_count": len(eligible_windows),
            "windows_with_correct_proposal": sum(row["first_correct_proposal_evaluation_frame"] is not None for row in reports),
            "windows_reacquired": sum(row["first_reacquired_evaluation_frame"] is not None for row in reports),
            "candidate_recall_on_eligible_lost_search_frames": lost_search_correct / lost_search_denominator if lost_search_denominator else None,
            "eligible_lost_search_frame_count": lost_search_denominator,
            "eligible_lost_search_frames_with_correct_candidate": lost_search_correct,
            "proposal_within_30_frames_rate": sum(delta <= 30 for delta in proposal_deltas) / len(eligible_windows) if eligible_windows else None,
            "reacquired_within_30_frames_rate": sum(delta <= 30 for delta in reacquisition_deltas) / len(eligible_windows) if eligible_windows else None,
            "median_frames_to_first_correct_proposal": median_or_none(proposal_deltas),
            "median_frames_to_reacquired": median_or_none(reacquisition_deltas),
            "median_confirmation_overhead_frames": median_or_none(confirmation),
            "wrong_instance_output_frames": sum(row["wrong_instance_output_frames"] for row in reports),
            "search_frames_total": arms[arm_name]["instance_redetection"].get("search_frames"),
            "inference_images_total": arms[arm_name]["instance_redetection"].get("inference_images"),
        }

    fixed_window_supported = any(
        aggregate[name]["candidate_recall_on_eligible_lost_search_frames"]
        > aggregate["S0"]["candidate_recall_on_eligible_lost_search_frames"]
        or aggregate[name]["windows_reacquired"] > aggregate["S0"]["windows_reacquired"]
        for name in ("S1", "S2")
    ) if {"S0", "S1", "S2"}.issubset(aggregate) else None
    terminal = (
        "ADT1_SMALL_TARGET_SEARCH_SCALE_R4_NOT_SUPPORTED_ON_FIXED_WINDOWS"
        if fixed_window_supported is False
        else "ADT1_SMALL_TARGET_SEARCH_SCALE_R4_EVALUATED"
    )
    output = {
        "schema_version": "ba_adt_small_target_search_scale_r4_evaluation_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-1-SMALL-TARGET-SEARCH-SCALE-R4",
        "inputs": {"arms": {name: {"path_name": path.name, "sha256": sha256(path), "candidate_generator": arms[name]["instance_redetection"].get("candidate_generator", "class-yolo")} for name, path in arm_paths.items()}, "groundtruth_sha256": sha256(args.groundtruth), "r1_failure_accounting_sha256": sha256(args.r1_failure_accounting), "target_uid": args.target_uid},
        "eligibility": {"visibility_min": VISIBILITY_ELIGIBLE, "source_bbox_min_dimension_px": SOURCE_MIN_DIMENSION_ELIGIBLE_PX, "minimum_eligible_frames_per_window": MIN_ELIGIBLE_FRAMES_PER_WINDOW, "role": "consumed_development_small_target_search_comparison_not_product_detectability_truth"},
        "fixed_window_count": len(window_reports),
        "eligible_window_count": len(eligible_windows),
        "excluded_windows": [row["fixed_window_index"] for row in window_reports if not row["window_eligible"]],
        "aggregate": aggregate,
        "windows": window_reports,
        "duration_definition": {"T_invisible": "preceding GT-invisible gap", "T_subdetectable": "GT first visible to first R4-eligible frame", "T_system_miss": "first R4-eligible frame to first correct proposal; right-censored at fixed-window end when no proposal appears", "T_confirmation": "first correct proposal to localized REACQUIRED; right-censored at fixed-window end when confirmation does not complete"},
        "fixed_window_scale_support": fixed_window_supported,
        "claim_ceiling": "consumed_development_search_scale_capability_no_navigation_product_safety_or_deployment_claim",
        "terminal": terminal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": output["terminal"], "eligible_window_count": len(eligible_windows), "aggregate": aggregate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
