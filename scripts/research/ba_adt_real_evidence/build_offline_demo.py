#!/usr/bin/env python3
"""Render an ADT offline Goal Copilot demo with an evaluator-only GT overlay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

from evaluate_rgb_observations import aligned_row, nearest_index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_gc1_winner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load policy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class PolicyObservation:
    target_visible: bool
    target_bearing: float | None
    relative_nearness: float | None
    left_free: bool = False
    right_free: bool = False
    forward_free: bool = False
    interaction_ready: bool = False


def copilot_state(visible: bool, ever_visible: bool, missing_frames: int, reacquired_age: int, approach_delta: float | None) -> str:
    if visible:
        if not ever_visible:
            return "ACQUIRED"
        if reacquired_age < 15:
            return "REACQUIRED"
        if approach_delta is not None and approach_delta > 0.002:
            return "APPROACHING"
        return "TRACKING"
    if not ever_visible:
        return "SEARCHING"
    if missing_frames <= 30:
        return "UNCERTAIN"
    return "LOST"


def load_gt(archive: Path, target_uid: str):
    with zipfile.ZipFile(archive) as zf:
        trajectory_times = [
            int(row["tracking_timestamp_us"]) * 1000
            for row in csv.DictReader(io.TextIOWrapper(zf.open("aria_trajectory.csv"), encoding="utf-8"))
        ]
        member = "2d_bounding_box_with_skeleton.csv" if "2d_bounding_box_with_skeleton.csv" in zf.namelist() else "2d_bounding_box.csv"
        target_by_index = {}
        for row in csv.DictReader(io.TextIOWrapper(zf.open(member), encoding="utf-8")):
            if row["stream_id"] != "214-1" or str(row["object_uid"]) != target_uid:
                continue
            index = nearest_index(trajectory_times, int(row["timestamp[ns]"]))
            if index is not None:
                target_by_index[index] = row
    return trajectory_times, target_by_index


def text_line(image, text: str, x: int, y: int, color=(230, 230, 230), scale=0.55):
    import cv2

    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the BA ADT prerecorded offline copilot demo")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--target-uid", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-timeline", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--start-gt-frame", type=int)
    parser.add_argument("--end-gt-frame", type=int)
    args = parser.parse_args()

    import cv2
    import numpy as np

    prediction = json.loads(args.observations.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    policy = load_policy(args.policy)
    trajectory_times, target_by_index = load_gt(args.groundtruth, args.target_uid)
    offset = int(evaluation["alignment_offset_frames"])
    start = args.start_gt_frame if args.start_gt_frame is not None else int(evaluation["evaluation_start_gt_frame"])
    end = args.end_gt_frame if args.end_gt_frame is not None else min(len(trajectory_times), len(prediction["frames"]) - offset)
    if not 0 <= start < end <= len(trajectory_times):
        raise ValueError(f"invalid GT frame interval [{start}, {end})")

    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, offset + start)
    video_size = 704
    panel_width = 480
    output_size = (video_size + panel_width, video_size)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")

    width = float(prediction["frame_size"]["width"])
    height = float(prediction["frame_size"]["height"])
    scale_x, scale_y = video_size / width, video_size / height
    timeline = []
    snapshots = []
    previous_key = None
    previous_visible = False
    ever_visible = False
    missing_frames = 10**9
    reacquired_age = 10**9
    last_bearing_deg = None
    nearness_history = []
    task_belief = 0.0
    rendered = 0
    localized = 0
    gt_visible_count = 0

    try:
        for gt_index in range(start, end):
            ok, source_frame = capture.read()
            if not ok:
                break
            observation = prediction["frames"][offset + gt_index]
            row = aligned_row(observation, target_by_index.get(gt_index), width, height)
            visible = bool(observation["target_visible"])
            if visible:
                missing_frames = 0
                if not previous_visible:
                    reacquired_age = 0
                else:
                    reacquired_age += 1
                bearing_normalized = float(observation["target_bearing_normalized"])
                last_bearing_deg = bearing_normalized * 45.0
                nearness = float(observation["relative_nearness"])
                nearness_history.append(nearness)
                approach_delta = nearness - nearness_history[-16] if len(nearness_history) >= 16 else None
            else:
                missing_frames += 1
                reacquired_age += 1
                nearness = None
                approach_delta = None
            state = copilot_state(visible, ever_visible, missing_frames, reacquired_age, approach_delta)
            policy_observation = PolicyObservation(visible, last_bearing_deg if not visible else bearing_normalized * 45.0, nearness)
            task_belief = policy.update_task_belief(task_belief, policy_observation)
            proposals = policy.propose_actions("FIND_APPROACH", policy_observation, task_belief)
            action = policy.select_action(proposals, policy_observation, completion_claim=False)
            display_guidance = "HOLD (clearance unknown)" if action == "STOP" and visible else action
            key = (state, action)
            if key != previous_key:
                timeline.append({
                    "gt_frame": gt_index,
                    "video_frame": offset + gt_index,
                    "timestamp_s": (offset + gt_index) / fps,
                    "state": state,
                    "policy_action": action,
                    "display_guidance": display_guidance,
                    "target_visible_rgb": visible,
                    "last_seen_bearing_proxy_deg": last_bearing_deg,
                })
                previous_key = key

            frame = cv2.resize(source_frame, (video_size, video_size))
            predicted_box = observation["bbox_xyxy"]
            gt_box = None
            if target_by_index.get(gt_index) is not None:
                gt_box = aligned_row({**observation, "bbox_xyxy": None}, target_by_index[gt_index], width, height)
                gt_raw = target_by_index[gt_index]
                gt_xyxy = [height - float(gt_raw["y_max[pixel]"]), float(gt_raw["x_min[pixel]"]), height - float(gt_raw["y_min[pixel]"]), float(gt_raw["x_max[pixel]"])]
                p1 = (round(gt_xyxy[0] * scale_x), round(gt_xyxy[1] * scale_y))
                p2 = (round(gt_xyxy[2] * scale_x), round(gt_xyxy[3] * scale_y))
                cv2.rectangle(frame, p1, p2, (255, 0, 255), 2)
            if predicted_box is not None:
                p1 = (round(predicted_box[0] * scale_x), round(predicted_box[1] * scale_y))
                p2 = (round(predicted_box[2] * scale_x), round(predicted_box[3] * scale_y))
                cv2.rectangle(frame, p1, p2, (0, 220, 0) if row["localized"] else (0, 0, 255), 3)
            canvas = np.zeros((video_size, video_size + panel_width, 3), dtype=np.uint8)
            canvas[:, :video_size] = frame
            canvas[:, video_size:] = (24, 24, 24)
            x = video_size + 24
            text_line(canvas, "BlindAssist ADT Offline Copilot", x, 36, (255, 255, 255), 0.65)
            text_line(canvas, f"Goal: find {args.target_name}", x, 72)
            text_line(canvas, f"State: {state}", x, 116, (80, 220, 255), 0.7)
            text_line(canvas, f"Guidance: {display_guidance}", x, 154, (80, 220, 255), 0.62)
            text_line(canvas, f"RGB visible: {visible}", x, 204)
            text_line(canvas, f"bearing proxy: {last_bearing_deg:+.1f} deg" if last_bearing_deg is not None else "bearing proxy: unknown", x, 236)
            text_line(canvas, f"relative nearness: {nearness:.4f}" if nearness is not None else "relative nearness: unknown", x, 268)
            text_line(canvas, f"observation quality: {observation['observation_quality']:.3f}", x, 300)
            text_line(canvas, "Evaluator-only GT", x, 360, (255, 0, 255), 0.65)
            text_line(canvas, f"GT visible: {row['gt_visible']}", x, 396)
            text_line(canvas, f"IoU: {row['iou']:.3f}", x, 428)
            error = row["bearing_error_normalized"]
            text_line(canvas, f"bearing error norm: {error:.4f}" if error is not None else "bearing error norm: n/a", x, 460)
            text_line(canvas, "green/red = RGB prediction", x, 520, (0, 220, 0))
            text_line(canvas, "magenta = GT evaluator only", x, 552, (255, 0, 255))
            text_line(canvas, "Prerecorded ADT replay", x, 626, (170, 170, 170))
            text_line(canvas, "NOT closed-loop navigation", x, 658, (170, 170, 170))
            writer.write(canvas)
            if rendered in {0, max(0, (end - start) // 5), max(0, 2 * (end - start) // 5), max(0, 3 * (end - start) // 5), max(0, 4 * (end - start) // 5), max(0, end - start - 1)}:
                snapshots.append(canvas.copy())
            rendered += 1
            gt_visible_count += int(row["gt_visible"])
            localized += int(row["localized"])
            ever_visible = ever_visible or visible
            previous_visible = visible
    finally:
        writer.release()
        capture.release()

    if rendered == 0:
        raise RuntimeError("no frames rendered")
    if snapshots:
        thumb_width = 592
        thumbs = [cv2.resize(item, (thumb_width, 352)) for item in snapshots]
        sheet = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:6])]) if len(thumbs) >= 6 else np.hstack(thumbs)
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.contact_sheet), sheet)

    receipt = {
        "schema_version": "ba_adt_offline_copilot_demo_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-2-DEVELOPMENT-DEMO",
        "goal": f"find {args.target_name}",
        "inputs": {
            "video_sha256": sha256(args.video),
            "observations_sha256": sha256(args.observations),
            "evaluation_sha256": sha256(args.evaluation),
            "groundtruth_sha256": sha256(args.groundtruth),
            "frozen_policy_sha256": sha256(args.policy),
        },
        "isolation": {"rgb_observer_received_gt": False, "gt_used_only_by_renderer_evaluator": True},
        "alignment_offset_frames": offset,
        "gt_frame_interval": [start, start + rendered],
        "rendered_frames": rendered,
        "fps": fps,
        "timeline": timeline,
        "demo_metrics": {
            "gt_visible_frames": gt_visible_count,
            "localized_frames_iou_0_10": localized,
            "localized_recall_iou_0_10": localized / gt_visible_count if gt_visible_count else None,
        },
        "adapter_contract": {
            "bearing": "normalized_image_x_times_45_degree_policy_proxy_not_camera_calibration",
            "nearness": "sqrt_predicted_bbox_area_fraction_not_metric_distance",
            "clearance": "unknown_and_fail_closed_false",
            "completion_claim": False,
        },
        "artifacts": {"video": args.output_video.name, "contact_sheet": args.contact_sheet.name},
        "claim_ceiling": "prerecorded_development_replay_not_closed_loop_navigation_or_safety_evidence",
        "terminal": "ADT2_OFFLINE_DEMO_RENDERED",
    }
    args.output_timeline.parent.mkdir(parents=True, exist_ok=True)
    args.output_timeline.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": receipt["terminal"], "frames": rendered, "timeline_events": len(timeline), "localized_recall": receipt["demo_metrics"]["localized_recall_iou_0_10"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
