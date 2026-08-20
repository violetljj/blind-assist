#!/usr/bin/env python3
"""Attribute ADT reacquisition failures without exposing GT to the RGB observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from evaluate_rgb_observations import aligned_row, gt_box, nearest_index
from mine_goal_episodes import RGB_STREAM, csv_rows, sha256
from run_rgb_observer import iou


def account_opportunities(rows: list[dict], minimum_gap: int = 6) -> dict:
    visible_segments = []
    start = None
    for index, row in enumerate([*rows, {"gt_visible": False}]):
        if row["gt_visible"] and start is None:
            start = index
        elif not row["gt_visible"] and start is not None:
            visible_segments.append((start, index))
            start = None

    opportunities = []
    for visible_segment_pair_index, (previous, current) in enumerate(zip(visible_segments, visible_segments[1:])):
        if current[0] - previous[1] < minimum_gap:
            continue
        start, end = current
        localized = next((index for index in range(start, end) if rows[index]["localized"]), None)
        search_end = localized + 1 if localized is not None else end
        search_rows = rows[start:search_end]
        first_candidate = next((offset for offset, row in enumerate(search_rows) if row["correct_candidate_present"]), None)
        correct_candidate_rows = [row for row in search_rows if row["correct_candidate_present"]]
        eligible_rows = [row for row in search_rows if row["correct_top_eligible"]]
        if localized is not None:
            outcome = "SUCCESS"
        elif not correct_candidate_rows:
            outcome = "NO_CANDIDATE"
        elif not eligible_rows:
            outcome = "CANDIDATE_REJECTED"
        else:
            outcome = "CONFIRMATION_FAILED"
        opportunities.append({
            "opportunity_index": len(opportunities),
            "visible_segment_pair_index": visible_segment_pair_index,
            "gt_revisible_evaluation_frame": start,
            "gt_visible_segment_frames": end - start,
            "preceding_gt_invisible_gap_frames": start - previous[1],
            "outcome": outcome,
            "reacquisition_delay_frames": None if localized is None else localized - start,
            "first_valid_candidate_latency_frames": first_candidate,
            "gt_revisible_to_first_correct_proposal_frames": first_candidate,
            "best_candidate_iou": max((row["best_candidate_iou"] for row in search_rows), default=0.0),
            "lost_search_frames": sum(row["search_active"] for row in search_rows),
            "correct_candidate_frames": len(correct_candidate_rows),
            "correct_top_eligible_frames": len(eligible_rows),
        })

    lost_rows = [row for row in rows if row["gt_visible"] and row["search_active"]]
    failures = [row for row in opportunities if row["outcome"] != "SUCCESS"]
    failure_counts = {name: sum(row["outcome"] == name for row in failures) for name in ("NO_CANDIDATE", "CANDIDATE_REJECTED", "CONFIRMATION_FAILED")}
    return {
        "eligible_reacquisition_count": len(opportunities),
        "successful_reacquisition_count": sum(row["outcome"] == "SUCCESS" for row in opportunities),
        "failed_reacquisition_count": len(failures),
        "failure_counts": failure_counts,
        "candidate_recall_during_lost": (sum(row["correct_candidate_present"] for row in lost_rows) / len(lost_rows)) if lost_rows else None,
        "gt_visible_lost_search_frame_count": len(lost_rows),
        "gt_visible_lost_frames_with_correct_candidate": sum(row["correct_candidate_present"] for row in lost_rows),
        "opportunities": opportunities,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--target-uid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposal-iou-threshold", type=float, default=0.10)
    args = parser.parse_args()

    prediction = json.loads(args.observations.read_text(encoding="utf-8"))
    if prediction.get("groundtruth_argument_supported") is not False:
        raise ValueError("observer firewall declaration missing")
    if not prediction.get("instance_redetection", {}).get("candidate_diagnostics"):
        raise ValueError("observations do not contain candidate diagnostics")

    with zipfile.ZipFile(args.groundtruth) as zf:
        member = "2d_bounding_box_with_skeleton.csv" if "2d_bounding_box_with_skeleton.csv" in zf.namelist() else "2d_bounding_box.csv"
        rgb_rows = [row for row in csv_rows(zf, member) if row["stream_id"] == RGB_STREAM]
        trajectory_times = [int(row["tracking_timestamp_us"]) * 1000 for row in csv_rows(zf, "aria_trajectory.csv")]
    target_by_index = {}
    distractors_by_index = {}
    for row in rgb_rows:
        index = nearest_index(trajectory_times, int(row["timestamp[ns]"]))
        if index is None or float(row["visibility_ratio[%]"]) < 0.10:
            continue
        if str(row["object_uid"]) == args.target_uid:
            target_by_index[index] = row
        else:
            distractors_by_index.setdefault(index, []).append(row)

    frames = prediction["frames"]
    width = float(prediction.get("frame_size", {}).get("width", 1408))
    height = float(prediction.get("frame_size", {}).get("height", 1408))
    calibration_count = max(1, len(trajectory_times) // 4)
    maximum_offset = max(0, len(frames) - len(trajectory_times))
    scores = []
    for offset in range(maximum_offset + 1):
        aligned = [aligned_row(frames[offset + index], target_by_index.get(index), distractors_by_index.get(index, []), width, height) for index in range(calibration_count)]
        scores.append(sum(row["iou"] for row in aligned if row["gt_visible"]))
    offset = max(range(len(scores)), key=lambda item: (scores[item], -item))
    aligned_count = min(len(trajectory_times), len(frames) - offset)
    rows = []
    for index in range(calibration_count, aligned_count):
        frame = frames[offset + index]
        truth = target_by_index.get(index)
        base = aligned_row(frame, truth, distractors_by_index.get(index, []), width, height)
        trace = frame.get("redetection_trace") or {}
        truth_box = gt_box(truth, height)
        proposal_ious = [iou(candidate["bbox_xyxy"], truth_box) if truth_box is not None else 0.0 for candidate in trace.get("candidates", [])]
        correct_indices = [candidate_index for candidate_index, overlap in enumerate(proposal_ious) if overlap >= args.proposal_iou_threshold]
        base.update({
            "search_active": bool(trace.get("search_active")),
            "correct_candidate_present": bool(correct_indices) and base["gt_visible"],
            "correct_top_eligible": bool(correct_indices) and correct_indices[0] == 0 and bool(trace.get("top_eligible")) and base["gt_visible"],
            "best_candidate_iou": max(proposal_ious, default=0.0),
        })
        rows.append(base)

    accounting = account_opportunities(rows)
    output = {
        "schema_version": "ba_adt_redetection_failure_accounting_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-1-REDETECTION-FAILURE-ACCOUNTING",
        "inputs": {"observations_sha256": sha256(args.observations), "groundtruth_sha256": sha256(args.groundtruth), "target_uid": args.target_uid},
        "isolation": {"observer_received_gt": False, "failure_accountant_received_rgb_pixels": False},
        "alignment_offset_frames": offset,
        "evaluation_start_gt_frame": calibration_count,
        "proposal_iou_threshold": args.proposal_iou_threshold,
        "metrics": accounting,
        "claim_ceiling": "consumed_development_failure_attribution_not_model_utility_or_navigation_evidence",
        "terminal": "ADT1_REDETECTION_FAILURE_ACCOUNTING_COMPLETE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": output["terminal"], **{key: accounting[key] for key in ("eligible_reacquisition_count", "successful_reacquisition_count", "failed_reacquisition_count", "failure_counts", "candidate_recall_during_lost")}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
