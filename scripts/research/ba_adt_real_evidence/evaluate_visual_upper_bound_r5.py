#!/usr/bin/env python3
"""Evaluate one GT-blind visual-query teacher on the frozen R4 denominator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from evaluate_rgb_observations import gt_box
from evaluate_search_scale_r4 import (
    MIN_ELIGIBLE_FRAMES_PER_WINDOW,
    SOURCE_MIN_DIMENSION_ELIGIBLE_PX,
    VISIBILITY_ELIGIBLE,
    load_truth,
)
from mine_goal_episodes import sha256
from run_rgb_observer import iou


CORRECT_IOU = 0.10


def median_or_none(values):
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--r1-failure-accounting", type=Path, required=True)
    parser.add_argument("--r4-evaluation", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--target-uid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    teacher = json.loads(args.teacher_output.read_text(encoding="utf-8"))
    if teacher.get("groundtruth_argument_supported") is not False:
        raise ValueError("teacher output is not GT-blind")
    if not teacher.get("formal_run"):
        raise ValueError("mechanical preflight output cannot be evaluated as formal R5")
    accounting = json.loads(args.r1_failure_accounting.read_text(encoding="utf-8"))
    r4 = json.loads(args.r4_evaluation.read_text(encoding="utf-8"))
    if r4["terminal"] != "ADT1_SMALL_TARGET_SEARCH_SCALE_R4_NOT_SUPPORTED_ON_FIXED_WINDOWS":
        raise ValueError("R4 parent terminal is not frozen")
    fixed = [row for row in accounting["metrics"]["opportunities"] if row["outcome"] == "NO_CANDIDATE"]
    if len(fixed) != 5:
        raise ValueError("R5 requires exactly five frozen R1 NO_CANDIDATE windows")

    target_by_index, distractors_by_index = load_truth(args.groundtruth, args.target_uid)
    offset = int(accounting["alignment_offset_frames"])
    evaluation_start = int(accounting["evaluation_start_gt_frame"])
    height = float(teacher["source_frame_size"]["height"])
    if len(teacher["frames"]) <= max(offset + evaluation_start + int(row["gt_revisible_evaluation_frame"]) for row in fixed):
        raise ValueError("teacher output ends before the frozen windows")

    window_reports = []
    all_success_margins = []
    all_success_sizes = []
    for fixed_window_index, opportunity in enumerate(fixed):
        start = int(opportunity["gt_revisible_evaluation_frame"])
        end = start + int(opportunity["gt_visible_segment_frames"])
        rows = []
        for evaluation_index in range(start, end):
            gt_index = evaluation_start + evaluation_index
            target = target_by_index.get(gt_index)
            if target is None:
                continue
            target_box = gt_box(target, height)
            min_dimension = min(target_box[2] - target_box[0], target_box[3] - target_box[1])
            eligible = (
                float(target["visibility_ratio[%]"]) >= VISIBILITY_ELIGIBLE
                and min_dimension >= SOURCE_MIN_DIMENSION_ELIGIBLE_PX
            )
            preview_index = offset + gt_index
            candidates = teacher["frames"][preview_index]["candidates"]
            correct = [row for row in candidates if iou(row["bbox_xyxy"], target_box) >= CORRECT_IOU]
            incorrect = [row for row in candidates if iou(row["bbox_xyxy"], target_box) < CORRECT_IOU]
            distractors = [gt_box(row, height) for row in distractors_by_index.get(gt_index, [])]
            wrong_instance = [
                candidate
                for candidate in incorrect
                if max((iou(candidate["bbox_xyxy"], box) for box in distractors), default=0.0) >= CORRECT_IOU
            ]
            correct_score = max((row["confidence"] for row in correct), default=None)
            strongest_wrong_score = max((row["confidence"] for row in incorrect), default=None)
            margin = None if correct_score is None else correct_score - (strongest_wrong_score or 0.0)
            rows.append({
                "evaluation_index": evaluation_index,
                "preview_frame_index": preview_index,
                "eligible": eligible,
                "source_target_min_dimension_px": min_dimension,
                "candidate_count": len(candidates),
                "correct_candidate_count": len(correct),
                "wrong_instance_proposal_count": len(wrong_instance),
                "best_correct_confidence": correct_score,
                "strongest_wrong_confidence": strongest_wrong_score,
                "teacher_confidence_margin": margin,
            })
        eligible_rows = [row for row in rows if row["eligible"]]
        window_eligible = len(eligible_rows) >= MIN_ELIGIBLE_FRAMES_PER_WINDOW
        successful = [row for row in eligible_rows if row["correct_candidate_count"] > 0]
        first_success = successful[0] if successful else None
        margins = [row["teacher_confidence_margin"] for row in successful]
        all_success_margins.extend(margins)
        if first_success is not None:
            all_success_sizes.append(first_success["source_target_min_dimension_px"])
        window_reports.append({
            "fixed_window_index": fixed_window_index,
            "original_opportunity_index": opportunity["opportunity_index"],
            "window_eligible": window_eligible,
            "eligible_frame_count": len(eligible_rows),
            "eligible_frames_with_correct_proposal": len(successful),
            "candidate_recall": len(successful) / len(eligible_rows) if window_eligible else None,
            "first_eligible_evaluation_frame": eligible_rows[0]["evaluation_index"] if window_eligible else None,
            "first_correct_proposal_evaluation_frame": None if first_success is None else first_success["evaluation_index"],
            "first_correct_proposal_latency_frames": None if first_success is None or not window_eligible else first_success["evaluation_index"] - eligible_rows[0]["evaluation_index"],
            "minimum_target_size_at_first_success_px": None if first_success is None else first_success["source_target_min_dimension_px"],
            "wrong_instance_proposal_count": sum(row["wrong_instance_proposal_count"] for row in eligible_rows),
            "median_teacher_confidence_margin": median_or_none(margins),
            "frames": eligible_rows,
        })

    eligible_windows = [row for row in window_reports if row["window_eligible"]]
    if len(eligible_windows) != r4["eligible_window_count"] or sum(row["eligible_frame_count"] for row in eligible_windows) != 97:
        raise ValueError("R5 denominator drifted from frozen R4 3-window/97-frame cohort")
    correct_windows = sum(row["eligible_frames_with_correct_proposal"] > 0 for row in eligible_windows)
    eligible_frames = sum(row["eligible_frame_count"] for row in eligible_windows)
    correct_frames = sum(row["eligible_frames_with_correct_proposal"] for row in eligible_windows)
    if correct_windows == 0:
        terminal = "R5_TEACHER_0_OF_3_CLOSE_APPEARANCE_ONLY_TINY_TARGET_REDETECTION"
        next_authority = "LAST10M_DESTINATION_GROUNDING_AND_SPATIAL_MEMORY"
    elif correct_windows == 1:
        terminal = "R5_TEACHER_1_OF_3_AMBIGUOUS_ONE_MECHANISM_DIFFERENT_TEACHER_B_ALLOWED"
        next_authority = "ONE_MECHANISM_DIFFERENT_TEACHER_B_ONLY"
    else:
        terminal = "R5_TEACHER_AT_LEAST_2_OF_3_RGB_INFORMATION_EXPLOITABLE"
        next_authority = "TEACHER_TO_EDGE_PROTOCOL_DESIGN_ALLOWED"

    output = {
        "schema_version": "ba_adt_visual_upper_bound_r5_evaluation_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT1_SMALL_TARGET_VISUAL_UPPER_BOUND_R5",
        "inputs": {
            "teacher_output_sha256": sha256(args.teacher_output),
            "r1_failure_accounting_sha256": sha256(args.r1_failure_accounting),
            "r4_evaluation_sha256": sha256(args.r4_evaluation),
            "groundtruth_sha256": sha256(args.groundtruth),
            "target_uid": args.target_uid,
        },
        "eligibility": r4["eligibility"],
        "correct_proposal_iou_threshold": CORRECT_IOU,
        "eligible_window_count": len(eligible_windows),
        "eligible_frame_count": eligible_frames,
        "correct_proposal_windows": correct_windows,
        "candidate_recall": correct_frames / eligible_frames,
        "eligible_frames_with_correct_proposal": correct_frames,
        "wrong_instance_proposal_count": sum(row["wrong_instance_proposal_count"] for row in eligible_windows),
        "first_correct_proposal_latency_frames_by_window": [row["first_correct_proposal_latency_frames"] for row in eligible_windows],
        "minimum_target_size_at_first_success_px": min(all_success_sizes) if all_success_sizes else None,
        "median_teacher_confidence_margin": median_or_none(all_success_margins),
        "minimum_teacher_confidence_margin": min(all_success_margins) if all_success_margins else None,
        "windows": window_reports,
        "terminal": terminal,
        "next_authority": next_authority,
        "no_r6_r7_same_window_rescue": True,
        "claim_ceiling": "consumed_development_teacher_capability_only_no_edge_product_or_safety_claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("terminal", "correct_proposal_windows", "candidate_recall", "wrong_instance_proposal_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
