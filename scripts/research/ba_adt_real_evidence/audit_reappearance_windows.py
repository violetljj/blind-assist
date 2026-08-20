#!/usr/bin/env python3
"""Audit the fixed ADT NO_CANDIDATE windows and prepare diagnostic oracle proposals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import zipfile

from evaluate_rgb_observations import gt_box, nearest_index
from mine_goal_episodes import RGB_STREAM, csv_rows, sha256
from run_rgb_observer import appearance_embedding, cosine_similarity, iou


VISIBILITY_USABLE = 0.50
MIN_DIMENSION_USABLE_PX = 24.0
MIN_USABLE_FRAMES = 3


def summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def vector_magnitude(row: dict, keys: tuple[str, str, str]) -> float:
    return math.sqrt(sum(float(row[key]) ** 2 for key in keys))


def load_truth(groundtruth: Path, target_uid: str):
    with zipfile.ZipFile(groundtruth) as zf:
        member = "2d_bounding_box_with_skeleton.csv" if "2d_bounding_box_with_skeleton.csv" in zf.namelist() else "2d_bounding_box.csv"
        rgb_rows = [row for row in csv_rows(zf, member) if row["stream_id"] == RGB_STREAM]
        trajectory = list(csv_rows(zf, "aria_trajectory.csv"))
        instances = json.loads(zf.read("instances.json"))
    trajectory_times = [int(row["tracking_timestamp_us"]) * 1000 for row in trajectory]
    target_by_index = {}
    visible_by_index = {}
    for row in rgb_rows:
        index = nearest_index(trajectory_times, int(row["timestamp[ns]"]))
        if index is None:
            continue
        if str(row["object_uid"]) == target_uid:
            target_by_index[index] = row
        if float(row["visibility_ratio[%]"]) >= 0.10:
            visible_by_index.setdefault(index, []).append(row)
    return trajectory, target_by_index, visible_by_index, instances


def proposal_metrics(frame: dict, target_box: list[float] | None) -> dict:
    trace = frame.get("redetection_trace") or {}
    candidates = trace.get("candidates", [])
    overlaps = [iou(candidate["bbox_xyxy"], target_box) if target_box is not None else 0.0 for candidate in candidates]
    return {
        "search_active": bool(trace.get("search_active")),
        "candidate_count": len(candidates),
        "best_candidate_iou": max(overlaps, default=0.0),
        "top_candidate_iou": overlaps[0] if overlaps else 0.0,
        "correct_candidate_present": any(value >= 0.10 for value in overlaps),
        "top_candidate_bbox": candidates[0]["bbox_xyxy"] if candidates else None,
    }


def diagnostic_class(frame_rows: list[dict]) -> tuple[str, str]:
    usable = [row for row in frame_rows if row["size_visibility_usable"]]
    mostly_visible = [row for row in frame_rows if row["visibility_ratio"] >= VISIBILITY_USABLE]
    if len(mostly_visible) < MIN_USABLE_FRAMES:
        return "UNOBSERVABLE_OR_HEAVILY_OCCLUDED", f"fewer than {MIN_USABLE_FRAMES} frames have visibility >= {VISIBILITY_USABLE}"
    if len(usable) < MIN_USABLE_FRAMES:
        return "TOO_SMALL_WHEN_VISIBLE", f"fewer than {MIN_USABLE_FRAMES} frames also reach min bbox dimension >= {MIN_DIMENSION_USABLE_PX}px"
    if max((row["same_prototype_distractors"] for row in frame_rows), default=0) > 0:
        return "RGB_INSTANCE_IDENTITY_AMBIGUITY_RISK", "another visible instance shares the target prototype"
    return "VISIBLE_SCALE_SUFFICIENT_MODEL_MISS", "GT visibility and size proxy are sufficient but neither RGB proposal generator covers the target"


def oracle_status(frame_rows: list[dict], oracle_success: bool | None) -> tuple[str, str]:
    if oracle_success is None:
        return "NOT_RUN", "no isolated oracle observation was supplied"
    if oracle_success:
        return "PASS_WITH_LATENCY", "the unchanged downstream chain eventually reacquired from injected GT proposals"
    traced = [row for row in frame_rows if row["oracle_candidate_traced"]]
    eligible = [row for row in traced if row["oracle_candidate_eligible"]]
    if len(frame_rows) < 2:
        return "FAIL_2_OF_3_INSUFFICIENT_VISIBLE_FRAMES", "the target is GT-visible for only one frame, so 2-of-3 confirmation is impossible"
    if not traced:
        return "ORACLE_NOT_CONSUMED", "the fixed window did not enter the expected LOST proposal path"
    if not eligible:
        return "VERIFIER_REJECTED_ALL", "all injected GT crops stayed below the unchanged verifier gates"
    return "CONFIRMATION_FAILED", "eligible GT proposals appeared but the unchanged temporal confirmation did not complete"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1-observations", type=Path, required=True)
    parser.add_argument("--yoloe-observations", type=Path, required=True)
    parser.add_argument("--r1-failure-accounting", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--target-uid", required=True)
    parser.add_argument("--oracle-proposals-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--oracle-observations", type=Path, action="append")
    args = parser.parse_args()

    import cv2
    import numpy as np

    r1 = json.loads(args.r1_observations.read_text(encoding="utf-8"))
    yoloe = json.loads(args.yoloe_observations.read_text(encoding="utf-8"))
    accounting = json.loads(args.r1_failure_accounting.read_text(encoding="utf-8"))
    oracle_by_window = {}
    oracle_hashes = []
    for oracle_path in args.oracle_observations or []:
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        diagnostic_input = oracle.get("gt_derived_diagnostic_input", {})
        if diagnostic_input.get("formal_evaluator_must_reject") is not True:
            raise ValueError("oracle observations are not marked fail-closed")
        for fixed_window_index in diagnostic_input.get("fixed_failure_window_indices", []):
            if fixed_window_index in oracle_by_window:
                raise ValueError(f"duplicate oracle observation for fixed window {fixed_window_index}")
            oracle_by_window[int(fixed_window_index)] = oracle
        oracle_hashes.append({"name": oracle_path.name, "sha256": sha256(oracle_path)})
    if r1.get("groundtruth_argument_supported") is not False or yoloe.get("groundtruth_argument_supported") is not False:
        raise ValueError("RGB observation firewall declaration missing")

    trajectory, target_by_index, visible_by_index, instances = load_truth(args.groundtruth, args.target_uid)
    target_instance = instances[args.target_uid]
    offset = int(accounting["alignment_offset_frames"])
    evaluation_start = int(accounting["evaluation_start_gt_frame"])
    failed = [row for row in accounting["metrics"]["opportunities"] if row["outcome"] == "NO_CANDIDATE"]
    if len(failed) != 5:
        raise ValueError(f"expected exactly five fixed NO_CANDIDATE windows, got {len(failed)}")

    proposals_by_preview_frame = {}
    relevant_preview_frames = set()
    window_bounds = []
    for fixed_index, opportunity in enumerate(failed):
        start = int(opportunity["gt_revisible_evaluation_frame"])
        end = start + int(opportunity["gt_visible_segment_frames"])
        window_bounds.append((fixed_index, start, end, opportunity))
        for evaluation_index in range(start, end):
            gt_index = evaluation_start + evaluation_index
            target = target_by_index.get(gt_index)
            preview_index = offset + gt_index
            relevant_preview_frames.add(preview_index)
            if target is not None and float(target["visibility_ratio[%]"]) >= 0.10:
                proposals_by_preview_frame[str(preview_index)] = {
                    "bbox_xyxy": gt_box(target, float(r1["frame_size"]["height"])),
                    "target_uid": args.target_uid,
                    "fixed_window_index": fixed_index,
                }

    anchor_preview_index = next(index for index, frame in enumerate(r1["frames"]) if frame.get("observation_source") == "detector" and frame.get("bbox_xyxy") is not None)
    relevant_preview_frames.add(anchor_preview_index)
    last_reliable_by_window = {}
    for fixed_index, start, _, _ in window_bounds:
        last_reliable = None
        for evaluation_index in range(start - 1, -1, -1):
            gt_index = evaluation_start + evaluation_index
            target = target_by_index.get(gt_index)
            frame = r1["frames"][offset + gt_index]
            box = gt_box(target, float(r1["frame_size"]["height"])) if target is not None else None
            if target is not None and float(target["visibility_ratio[%]"]) >= 0.10 and frame.get("bbox_xyxy") is not None and iou(frame["bbox_xyxy"], box) >= 0.10:
                last_reliable = evaluation_index
                relevant_preview_frames.add(offset + gt_index)
                break
        last_reliable_by_window[fixed_index] = last_reliable

    capture = cv2.VideoCapture(str(args.video))
    images = {}
    frame_index = 0
    while capture.isOpened() and relevant_preview_frames:
        ok, image = capture.read()
        if not ok:
            break
        if frame_index in relevant_preview_frames:
            images[frame_index] = image.copy()
            relevant_preview_frames.remove(frame_index)
        frame_index += 1
    capture.release()
    if relevant_preview_frames:
        raise RuntimeError(f"video frames unavailable: {sorted(relevant_preview_frames)[:5]}")

    anchor_frame = r1["frames"][anchor_preview_index]
    anchor_embedding = appearance_embedding(images[anchor_preview_index], anchor_frame["bbox_xyxy"])
    height = float(r1["frame_size"]["height"])
    width = float(r1["frame_size"]["width"])
    per_window_rows = {}
    for fixed_index, start, end, _ in window_bounds:
        rows = []
        for evaluation_index in range(start, end):
            gt_index = evaluation_start + evaluation_index
            preview_index = offset + gt_index
            target = target_by_index.get(gt_index)
            if target is None:
                continue
            target_box = gt_box(target, height)
            box_width = target_box[2] - target_box[0]
            box_height = target_box[3] - target_box[1]
            x1, y1, x2, y2 = [int(round(value)) for value in target_box]
            image = images[preview_index]
            crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
            blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size else 0.0
            embedding = appearance_embedding(image, target_box)
            visible_rows = visible_by_index.get(gt_index, [])
            same_prototype = 0
            same_category = 0
            for distractor in visible_rows:
                uid = str(distractor["object_uid"])
                if uid == args.target_uid or uid not in instances:
                    continue
                same_prototype += instances[uid].get("prototype_name") == target_instance.get("prototype_name")
                same_category += instances[uid].get("category_uid") == target_instance.get("category_uid")
            trajectory_row = trajectory[gt_index]
            visibility = float(target["visibility_ratio[%]"])
            r1_frame = r1["frames"][preview_index]
            yoloe_frame = yoloe["frames"][preview_index]
            oracle = oracle_by_window.get(fixed_index)
            oracle_frame = oracle["frames"][preview_index] if oracle is not None and preview_index < len(oracle["frames"]) else None
            oracle_localized = oracle_frame is not None and oracle_frame.get("bbox_xyxy") is not None and iou(oracle_frame["bbox_xyxy"], target_box) >= 0.10
            oracle_trace = oracle_frame.get("redetection_trace") if oracle_frame is not None else None
            oracle_candidates = (oracle_trace or {}).get("candidates", [])
            oracle_candidate = next((candidate for candidate in oracle_candidates if candidate.get("candidate_source") == "gt_diagnostic_oracle"), None)
            rows.append({
                "evaluation_frame": evaluation_index,
                "preview_frame": preview_index,
                "visibility_ratio": visibility,
                "bbox_width_px": box_width,
                "bbox_height_px": box_height,
                "bbox_area_fraction": box_width * box_height / (width * height),
                "crop_laplacian_variance": blur,
                "appearance_similarity_to_initial_anchor": cosine_similarity(embedding, anchor_embedding),
                "camera_angular_speed_rad_s": vector_magnitude(trajectory_row, ("angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device")),
                "camera_linear_speed_m_s": vector_magnitude(trajectory_row, ("device_linear_velocity_x_device", "device_linear_velocity_y_device", "device_linear_velocity_z_device")),
                "visible_distractor_count": max(0, len(visible_rows) - 1),
                "same_prototype_distractors": same_prototype,
                "same_category_distractors": same_category,
                "size_visibility_usable": visibility >= VISIBILITY_USABLE and min(box_width, box_height) >= MIN_DIMENSION_USABLE_PX,
                "r1": proposal_metrics(r1_frame, target_box),
                "yoloe": proposal_metrics(yoloe_frame, target_box),
                "oracle_localized": oracle_localized,
                "oracle_observation_source": oracle_frame.get("observation_source") if oracle_frame is not None else None,
                "oracle_candidate_traced": oracle_candidate is not None,
                "oracle_candidate_appearance": None if oracle_candidate is None else oracle_candidate["appearance"],
                "oracle_candidate_score": None if oracle_candidate is None else oracle_candidate["score"],
                "oracle_candidate_eligible": bool(oracle_candidate and oracle_candidate["verifier_eligible"]),
                "oracle_candidate_confirmed": bool(oracle_trace and oracle_trace.get("top_confirmed") and oracle_candidate and oracle_candidate["rank"] == 0),
                "gt_bbox_xyxy": target_box,
            })
        per_window_rows[fixed_index] = rows

    window_reports = []
    selected_frames = []
    for fixed_index, start, end, opportunity in window_bounds:
        rows = per_window_rows[fixed_index]
        usable = [row for row in rows if row["size_visibility_usable"]]
        first_usable = usable[0] if usable else None
        oracle_available = fixed_index in oracle_by_window
        oracle_first = next((row for row in rows if row["oracle_localized"]), None) if oracle_available else None
        oracle_success = (oracle_first is not None) if oracle_available else None
        classification, reason = diagnostic_class(rows)
        downstream_status, downstream_reason = oracle_status(rows, oracle_success)
        last_reliable = last_reliable_by_window[fixed_index]
        best_evidence = max(rows, key=lambda row: (row["size_visibility_usable"], row["visibility_ratio"], min(row["bbox_width_px"], row["bbox_height_px"]), row["crop_laplacian_variance"]))
        frame_choices = [
            ("last reliable", None if last_reliable is None else offset + evaluation_start + last_reliable),
            ("GT first visible", rows[0]["preview_frame"]),
            ("first size-visible", None if first_usable is None else first_usable["preview_frame"]),
            ("best visible RGB", best_evidence["preview_frame"]),
        ]
        selected_frames.append((fixed_index, frame_choices))
        window_reports.append({
            "fixed_window_index": fixed_index,
            "original_opportunity_index": opportunity["opportunity_index"],
            "gt_first_visible_evaluation_frame": start,
            "gt_first_visible_preview_frame": rows[0]["preview_frame"],
            "last_reliable_evaluation_frame": last_reliable,
            "preceding_gt_invisible_frames": opportunity["preceding_gt_invisible_gap_frames"],
            "gt_visible_but_missed_frames": len(rows),
            "size_visibility_eligible_but_missed_frames": len(usable),
            "first_size_visibility_eligible_latency_frames": None if first_usable is None else first_usable["evaluation_frame"] - start,
            "visibility_ratio": summary([row["visibility_ratio"] for row in rows]),
            "bbox_width_px": summary([row["bbox_width_px"] for row in rows]),
            "bbox_height_px": summary([row["bbox_height_px"] for row in rows]),
            "bbox_min_dimension_px": summary([min(row["bbox_width_px"], row["bbox_height_px"]) for row in rows]),
            "bbox_min_dimension_at_640_input_px": summary([min(row["bbox_width_px"], row["bbox_height_px"]) * 640.0 / max(width, height) for row in rows]),
            "bbox_area_fraction": summary([row["bbox_area_fraction"] for row in rows]),
            "crop_laplacian_variance": summary([row["crop_laplacian_variance"] for row in rows]),
            "camera_angular_speed_rad_s": summary([row["camera_angular_speed_rad_s"] for row in rows]),
            "camera_linear_speed_m_s": summary([row["camera_linear_speed_m_s"] for row in rows]),
            "appearance_similarity_to_initial_anchor": summary([row["appearance_similarity_to_initial_anchor"] for row in rows]),
            "max_visible_distractors": max(row["visible_distractor_count"] for row in rows),
            "max_same_prototype_distractors": max(row["same_prototype_distractors"] for row in rows),
            "max_same_category_distractors": max(row["same_category_distractors"] for row in rows),
            "r1_proposal_frames": sum(row["r1"]["candidate_count"] > 0 for row in rows),
            "r1_total_proposals": sum(row["r1"]["candidate_count"] for row in rows),
            "r1_best_candidate_iou": max(row["r1"]["best_candidate_iou"] for row in rows),
            "yoloe_proposal_frames": sum(row["yoloe"]["candidate_count"] > 0 for row in rows),
            "yoloe_total_proposals": sum(row["yoloe"]["candidate_count"] for row in rows),
            "yoloe_best_candidate_iou": max(row["yoloe"]["best_candidate_iou"] for row in rows),
            "oracle_reacquired": oracle_success,
            "oracle_reacquisition_latency_frames": None if oracle_first is None else oracle_first["evaluation_frame"] - start,
            "oracle_reacquisition_source": None if oracle_first is None else oracle_first["oracle_observation_source"],
            "oracle_downstream_status": downstream_status,
            "oracle_downstream_reason": downstream_reason,
            "oracle_injected_trace_frames": sum(row["oracle_candidate_traced"] for row in rows),
            "oracle_eligible_frames": sum(row["oracle_candidate_eligible"] for row in rows),
            "oracle_confirmed_frames": sum(row["oracle_candidate_confirmed"] for row in rows),
            "oracle_first_eligible_latency_frames": next((row["evaluation_frame"] - start for row in rows if row["oracle_candidate_eligible"]), None),
            "oracle_max_appearance": max((row["oracle_candidate_appearance"] for row in rows if row["oracle_candidate_appearance"] is not None), default=None),
            "duration_decomposition": {
                "gt_invisible_duration_frames": opportunity["preceding_gt_invisible_gap_frames"],
                "gt_visible_but_below_size_visibility_proxy_frames": len(rows) - len(usable),
                "detectable_but_missed_duration_frames": len(usable),
                "note": "per-opportunity decomposition; not a decomposition of the global longest-dropout statistic",
            },
            "primary_diagnostic_class": classification,
            "classification_reason": reason,
            "timeline": {
                "last_reliable_preview_frame": None if last_reliable is None else offset + evaluation_start + last_reliable,
                "gt_first_visible_preview_frame": rows[0]["preview_frame"],
                "first_size_visibility_eligible_preview_frame": None if first_usable is None else first_usable["preview_frame"],
                "first_r1_correct_proposal_preview_frame": next((row["preview_frame"] for row in rows if row["r1"]["correct_candidate_present"]), None),
                "first_yoloe_correct_proposal_preview_frame": next((row["preview_frame"] for row in rows if row["yoloe"]["correct_candidate_present"]), None),
                "oracle_reacquired_preview_frame": None if oracle_first is None else oracle_first["preview_frame"],
            },
        })

    cell_width, cell_height = 360, 240
    sheet = np.full((cell_height * len(selected_frames), cell_width * 4, 3), 245, dtype=np.uint8)
    row_by_preview = {row["preview_frame"]: row for rows in per_window_rows.values() for row in rows}
    for row_index, (fixed_index, choices) in enumerate(selected_frames):
        for column_index, (label, preview_index) in enumerate(choices):
            if preview_index is None:
                continue
            image = images[preview_index].copy()
            metric = row_by_preview.get(preview_index)
            if metric is not None:
                gt = [int(round(value)) for value in metric["gt_bbox_xyxy"]]
                cv2.rectangle(image, (gt[0], gt[1]), (gt[2], gt[3]), (0, 220, 0), 3)
                r1_top = metric["r1"]["top_candidate_bbox"]
                yoloe_top = metric["yoloe"]["top_candidate_bbox"]
                if r1_top is not None:
                    box = [int(round(value)) for value in r1_top]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                if yoloe_top is not None:
                    box = [int(round(value)) for value in yoloe_top]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            resized = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(resized, (0, 0), (cell_width, 42), (0, 0, 0), -1)
            cv2.putText(resized, f"W{fixed_index} {label} f={preview_index}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if metric is not None:
                cv2.putText(resized, f"vis={metric['visibility_ratio']:.2f} box={metric['bbox_width_px']:.0f}x{metric['bbox_height_px']:.0f} blur={metric['crop_laplacian_variance']:.0f}", (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1, cv2.LINE_AA)
            y0, x0 = row_index * cell_height, column_index * cell_width
            sheet[y0:y0 + cell_height, x0:x0 + cell_width] = resized

    args.oracle_proposals_output.parent.mkdir(parents=True, exist_ok=True)
    oracle_proposal_output = {
        "schema_version": "ba_adt_gt_oracle_proposals_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-1-REAPPEARANCE-OBSERVABILITY-DIAGNOSTIC",
        "source_role": "GT_DERIVED_DIAGNOSTIC_ORACLE_PROPOSALS",
        "inputs": {"groundtruth_sha256": sha256(args.groundtruth), "r1_observations_sha256": sha256(args.r1_observations), "target_uid": args.target_uid},
        "fixed_failure_window_count": len(failed),
        "fixed_failure_window_indices": [row[0] for row in window_bounds],
        "proposals_by_preview_frame": proposals_by_preview_frame,
        "formal_rgb_evaluator_use_forbidden": True,
        "claim_ceiling": "diagnostic_oracle_only",
    }
    args.oracle_proposals_output.write_text(json.dumps(oracle_proposal_output, indent=2) + "\n", encoding="utf-8")
    for fixed_index, _, _, _ in window_bounds:
        window_path = args.oracle_proposals_output.with_name(f"{args.oracle_proposals_output.stem}_window_{fixed_index}{args.oracle_proposals_output.suffix}")
        window_payload = {**oracle_proposal_output, "fixed_failure_window_count": 1, "fixed_failure_window_indices": [fixed_index], "proposals_by_preview_frame": {key: value for key, value in proposals_by_preview_frame.items() if value["fixed_window_index"] == fixed_index}}
        window_path.write_text(json.dumps(window_payload, indent=2) + "\n", encoding="utf-8")
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.contact_sheet), sheet)
    output = {
        "schema_version": "ba_adt_reappearance_observability_audit_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-1-REAPPEARANCE-OBSERVABILITY-DIAGNOSTIC",
        "inputs": {"r1_observations_sha256": sha256(args.r1_observations), "yoloe_observations_sha256": sha256(args.yoloe_observations), "r1_failure_accounting_sha256": sha256(args.r1_failure_accounting), "groundtruth_sha256": sha256(args.groundtruth), "video_sha256": sha256(args.video), "oracle_observations": oracle_hashes},
        "fixed_window_count": len(window_reports),
        "diagnostic_thresholds": {"gt_visibility_usable_min": VISIBILITY_USABLE, "bbox_min_dimension_usable_px": MIN_DIMENSION_USABLE_PX, "minimum_usable_frames": MIN_USABLE_FRAMES, "note": "transparent size-visibility proxy, not a learned detectability truth"},
        "oracle_success_count": None if not oracle_by_window else sum(row["oracle_reacquired"] for row in window_reports if row["oracle_reacquired"] is not None),
        "classification_counts": {name: sum(row["primary_diagnostic_class"] == name for row in window_reports) for name in sorted({row["primary_diagnostic_class"] for row in window_reports})},
        "duration_totals_across_fixed_windows": {
            "gt_invisible_duration_frames": sum(row["duration_decomposition"]["gt_invisible_duration_frames"] for row in window_reports),
            "gt_visible_but_below_size_visibility_proxy_frames": sum(row["duration_decomposition"]["gt_visible_but_below_size_visibility_proxy_frames"] for row in window_reports),
            "detectable_but_missed_duration_frames": sum(row["duration_decomposition"]["detectable_but_missed_duration_frames"] for row in window_reports),
        },
        "windows": window_reports,
        "contact_sheet": args.contact_sheet.name,
        "claim_ceiling": "consumed_development_observability_diagnostic_no_rgb_only_model_utility_navigation_or_identity_truth_claim",
        "terminal": "ADT1_REAPPEARANCE_OBSERVABILITY_AUDIT_COMPLETE" if len(oracle_by_window) == len(window_reports) else "ADT1_REAPPEARANCE_OBSERVABILITY_AUDIT_PREPARED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": output["terminal"], "fixed_window_count": len(window_reports), "oracle_success_count": output["oracle_success_count"], "classification_counts": output["classification_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
