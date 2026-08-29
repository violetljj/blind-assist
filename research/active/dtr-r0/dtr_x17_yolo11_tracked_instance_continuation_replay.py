"""Full replay of YOLO11 tracked-instance-authorized X7 continuation.

The frozen five overlapping stitched-image tiles are evaluated by
YOLO11n-seg on CUDA for six movable COCO classes.  A current X7 cell receives
birth authority only when one CausalPersonTracker ID of the same class exists
in consecutive frames and at least one projected X7 trajectory height anchor
lies inside that instance polygon in both frames.
Missing images, masks, poses, projections, CUDA, or association fail closed.
Authorized births alone use the unchanged X14 0.50-second transport, followed
by the unchanged X15/X3 prediction lifecycle and scorer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x3_full_lag_floxel_replay as x3  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
import dtr_x7_full_static_world_anchor_replay as x7  # noqa: E402
import dtr_x8_rgb_static_veto_falsifier as x8  # noqa: E402
import dtr_x9_full_rgb_static_veto_replay as x9  # noqa: E402
import dtr_x14_rgb_authorized_motion_continuation_falsifier as x14  # noqa: E402
import jrdb_mask_lidar_bridge as mask_bridge  # noqa: E402
import jrdb_rgb_bridge as rgb_bridge  # noqa: E402
from real_observation_adapter import BBox, CausalPersonTracker, Detection  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402


SCHEMA = "blindassist-dtr-x17-yolo11-tracked-instance-continuation-replay-v1"
LEDGER_SCHEMA = "blindassist-dtr-x17-yolo11-tracked-instance-continuation-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x17-yolo11-tracked-instance-continuation-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-x17-yolo11-tracked-instance-continuation-freeze-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x17-yolo11-tracked-instance-continuation-materialization-v1"
SEQUENCE_SCHEMA = "blindassist-dtr-x17-sequence-materialization-v1"
PROGRESS_SCHEMA = "blindassist-dtr-x17-progress-v1"
COCO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MODEL_SHA256 = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
TIMELINE_FRAMES = x9.TIMELINE_FRAMES


@dataclass(frozen=True)
class InstanceMask:
    class_id: int
    track_id: str
    mask: np.ndarray


def _paths(root: Path, sequence: str | None = None) -> dict[str, Path]:
    base = root if sequence is None else root / "sequences" / sequence
    return {
        "freeze": root / "freeze.json",
        "lock": base / "materialize.lock.json",
        "progress": base / "progress.json",
        "ledger": base / "lag-floxel.npz",
        "manifest": base / "lag-floxel.json",
        "materialization": root / "materialization.json",
        "predictions": root / "predictions.json",
        "result": root / "result.json",
    }


def _source_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "lag-floxel.npz", base / "lag-floxel.json"


def _baseline_rows(args: argparse.Namespace) -> dict[str, Any]:
    _baseline, rows = x3._load_baseline(args)
    require(len(rows) == 6, "x17_sequence_count")
    return rows


def _fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    require(sha256_file(args.model.resolve(strict=True)) == MODEL_SHA256, "x17_model_hash")
    inputs = {}
    for sequence in sorted(_baseline_rows(args)):
        ledger, manifest = _source_paths(args.x7_root.resolve(strict=True), sequence)
        value = json.loads(manifest.resolve(strict=True).read_text(encoding="utf-8"))
        require(value.get("schema") == x7.LEDGER_SCHEMA and value.get("truth_blind") is True, f"x17_x7_manifest:{sequence}")
        require(value.get("ledger_sha256") == sha256_file(ledger.resolve(strict=True)), f"x17_x7_hash:{sequence}")
        bag = Path(value["source"]["bag"]).resolve(strict=True)
        require(value["source"]["bag_sha256"] == sha256_file(bag), f"x17_bag_hash:{sequence}")
        inputs[sequence] = {
            "x7_ledger_sha256": sha256_file(ledger),
            "x7_manifest_sha256": sha256_file(manifest),
            "bag": str(bag),
            "bag_sha256": sha256_file(bag),
        }
    return {
        "schema": FREEZE_SCHEMA,
        "truth_blind_materialization": True,
        "oracle": False,
        "algorithm_files": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())}
            for path in (__file__, mask_bridge.__file__, rgb_bridge.__file__, x14.__file__, x7.__file__, x3.__file__)
        ],
        "source_config": {
            "base": "SEALED_X7_GEOMETRY_AND_VELOCITY_CANDIDATES",
            "birth_authority": "YOLO11_CONSECUTIVE_TRACKED_MOVABLE_INSTANCE_MASK_MEMBERSHIP",
            "stitched_topic": x8.STITCHED_TOPIC,
            "tile_width": rgb_bridge.TILE_WIDTH,
            "tile_starts": list(rgb_bridge.TILE_STARTS),
            "coco_classes": {str(class_id): name for class_id, name in COCO_CLASSES.items()},
            "model": {"path": str(args.model.resolve()), "sha256": MODEL_SHA256},
            "backend": "CUDA_REQUIRED",
            "provider_runtime": {"image_size": rgb_bridge.INFERENCE_SIZE, "confidence": rgb_bridge.DETECTOR_CONFIDENCE, "nms_iou": rgb_bridge.DETECTOR_NMS_IOU, "max_detections": rgb_bridge.DETECTOR_MAX_DET, "augment": False},
            "tracker": "real_observation_adapter.CausalPersonTracker defaults, one tracker per COCO class",
            "mask_association": "jrdb_mask_lidar_bridge.mask_candidates bound to same tracked detection",
            "anchor_membership": "SAME_X7_TRAJECTORY_HEIGHT_ANCHOR_INSIDE_PREVIOUS_AND_CURRENT_MASK",
            "height_anchors_m": list(x8.c22.HEIGHT_ANCHORS_M),
            "continuation": "UNCHANGED_X14_TRANSPORT",
            "continuation_s": x14.CONTINUATION_S,
            "missing_evidence_policy": "FAIL_CLOSED_NO_NEW_BIRTH",
            "timeline_frames": TIMELINE_FRAMES,
        },
        "frozen_downstream": {
            "cell_velocity": "UNCHANGED_X7",
            "motion_bounds": "UNCHANGED_X7_X3",
            "route_entry_geometry": "UNCHANGED_R7",
            "event_lifecycle": "UNCHANGED_X15_X3",
            "scorer": "UNCHANGED_X15_X3",
        },
        "inputs": {
            "x14_result_sha256": sha256_file(args.x14_result.resolve(strict=True)),
            "x15_result_sha256": sha256_file(args.x15_result.resolve(strict=True)),
            "x7_result_sha256": sha256_file(args.x7_result.resolve(strict=True)),
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve(strict=True)),
            "baseline_result_sha256": sha256_file(args.baseline_result.resolve(strict=True)),
            "roster_sha256": sha256_file(args.roster.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
            "calibration_cameras_sha256": sha256_file(args.calibration_dir.resolve(strict=True) / "cameras.yaml"),
            "sequences": inputs,
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    freeze = _paths(root)["freeze"]
    fingerprint = _fingerprint(args)
    if freeze.exists():
        require(json.loads(freeze.read_text(encoding="utf-8")) == fingerprint, "x17_freeze_drift")
    else:
        write_json(freeze, fingerprint)
    return {"schema": FREEZE_SCHEMA, "status": "READY", "sequences": sorted(_baseline_rows(args)), "freeze": str(freeze), "freeze_sha256": sha256_file(freeze)}


def _make_model(model_path: Path) -> Any:
    import torch
    from ultralytics import YOLO

    require(torch.cuda.is_available(), "x17_cuda_required")
    require(torch.cuda.device_count() > 0, "x17_cuda_device_missing")
    return YOLO(str(model_path), task="segment")


def _candidate_classes(predictions: Sequence[Any]) -> list[int]:
    output = []
    for prediction in predictions:
        boxes, masks = prediction.boxes, prediction.masks
        if boxes is None or not len(boxes):
            continue
        require(masks is not None and len(masks.xy) == len(boxes), "x17_mask_class_alignment")
        classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
        coordinates = boxes.xyxy.detach().cpu().numpy()
        for index, polygon in enumerate(masks.xy):
            values = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            x1, y1, x2, y2 = coordinates[index]
            if len(values) >= 3 and x2 > x1 and y2 > y1:
                output.append(int(classes[index]))
    return output


def _full_mask(candidate: Any) -> np.ndarray:
    mask = np.zeros((rgb_bridge.IMAGE_HEIGHT, rgb_bridge.IMAGE_WIDTH), dtype=np.uint8)
    polygon = np.rint(np.asarray(candidate.polygon_xy)).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0] + candidate.tile_start, 0, rgb_bridge.IMAGE_WIDTH - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, rgb_bridge.IMAGE_HEIGHT - 1)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def _infer(model: Any, image: np.ndarray, trackers: Mapping[int, CausalPersonTracker], time_s: float) -> list[InstanceMask]:
    crops = [image[:, start : start + rgb_bridge.TILE_WIDTH] for start in rgb_bridge.TILE_STARTS]
    predictions = model.predict(
        source=crops,
        device=0,
        imgsz=rgb_bridge.INFERENCE_SIZE,
        conf=rgb_bridge.DETECTOR_CONFIDENCE,
        iou=rgb_bridge.DETECTOR_NMS_IOU,
        classes=list(COCO_CLASSES),
        max_det=rgb_bridge.DETECTOR_MAX_DET,
        augment=False,
        batch=len(crops),
        verbose=False,
    )
    candidates = mask_bridge.mask_candidates(predictions)
    classes = _candidate_classes(predictions)
    require(len(candidates) == len(classes), "x17_mask_candidate_class_alignment")
    output = []
    for class_id in COCO_CLASSES:
        members = [(candidate, cls) for candidate, cls in zip(candidates, classes) if cls == class_id]
        if not members:
            trackers[class_id].update([], time_s=time_s)
            continue
        rows = np.asarray([candidate.bbox_xyxy + [candidate.confidence] for candidate, _cls in members], dtype=np.float64)
        kept = rgb_bridge.nms(rows)
        selected = []
        for row in kept:
            index = next(index for index, (candidate, _cls) in enumerate(members) if np.allclose(row[:4], candidate.bbox_xyxy) and abs(float(row[4]) - candidate.confidence) < 1e-6)
            selected.append(members[index][0])
        detections = [Detection(BBox(*candidate.bbox_xyxy), candidate.confidence) for candidate in selected]
        tracked = trackers[class_id].update(detections, time_s=time_s)
        output.extend(InstanceMask(class_id, item.track_id, _full_mask(candidate)) for item, candidate in zip(tracked, selected))
    return output


def _mask_member(mask: np.ndarray, point: np.ndarray) -> bool:
    if not np.all(np.isfinite(point)):
        return False
    x_value, y_value = int(round(float(point[0]))), int(round(float(point[1])))
    return 0 <= y_value < mask.shape[0] and 0 <= x_value < mask.shape[1] and bool(mask[y_value, x_value])


def _authorize(
    row: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    previous_pose: dict[str, Any],
    current_pose: dict[str, Any],
    delta_s: float,
    previous_masks: Sequence[InstanceMask],
    current_masks: Sequence[InstanceMask],
    calibration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    positions, velocities, _counts, _support = row
    count = len(positions)
    authorized = np.zeros(count, dtype=bool)
    if not count:
        return authorized, {"associated_mask_pairs": 0, "projected_anchors": 0, "authorized_cells": 0}
    current_xy, velocity_world = x5._ego_to_world(positions.astype(np.float64), velocities.astype(np.float64), current_pose)
    previous_xy = current_xy - velocity_world * float(delta_s)
    anchors = np.asarray(x8.c22.HEIGHT_ANCHORS_M, dtype=np.float64)

    def expand(values: np.ndarray) -> np.ndarray:
        return np.column_stack((np.repeat(values[:, 0], len(anchors)), np.repeat(values[:, 1], len(anchors)), np.tile(anchors, count)))

    previous_world = expand(previous_xy)
    current_world = expand(current_xy)
    previous_by_key = {(item.class_id, item.track_id): item for item in previous_masks}
    pairs = [(previous_by_key[(item.class_id, item.track_id)], item) for item in current_masks if (item.class_id, item.track_id) in previous_by_key]
    before_points = x8.c22._project(previous_world, previous_pose, calibration)
    after_points = x8.c22._project(current_world, current_pose, calibration)
    projected = int(np.count_nonzero(np.all(np.isfinite(before_points), axis=1) & np.all(np.isfinite(after_points), axis=1)))
    for sample_index, (before_point, after_point) in enumerate(zip(before_points, after_points)):
        cell_index = sample_index // len(anchors)
        if not authorized[cell_index]:
            authorized[cell_index] = any(_mask_member(before.mask, before_point) and _mask_member(after.mask, after_point) for before, after in pairs)
    return authorized, {"associated_mask_pairs": len(pairs), "projected_anchors": projected, "authorized_cells": int(np.count_nonzero(authorized))}


def _empty_row() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.empty((0, 2), np.float64), np.empty((0, 2), np.float64), np.empty(0, np.int32), np.empty(0, np.float32))


def _stream_births(
    *,
    model: Any,
    bag_path: Path,
    source: Mapping[str, np.ndarray],
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    poses: Mapping[int, dict[str, Any]],
    calibration: dict[str, Any],
    progress_path: Path,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as error:
        raise RuntimeError("dtr_x17 requires rosbags") from error
    typestore = get_typestore(Stores.ROS1_NOETIC)
    target_ns = {int(frame): round(float(timestamps[int(frame)]) * 1e9) for frame in frames}
    by_stamp = {stamp: frame for frame, stamp in target_ns.items()}
    require(len(by_stamp) == len(target_ns), "x17_duplicate_frame_timestamp")
    ordered = sorted(by_stamp)
    source_rows = [x5._frame_arrays(source, frame) for frame in frames]
    index_by_frame = {frame: index for index, frame in enumerate(frames)}
    authorized = [_empty_row() for _ in frames]
    diagnostics = [{"frame": frame, "input_cells": int(len(row[0])), "authorized_cells": 0, "associated_track_masks": 0, "projected_anchors": 0, "birth_seconds": 0.0} for frame, row in zip(frames, source_rows)]
    previous: tuple[int, list[InstanceMask]] | None = None
    trackers = {class_id: CausalPersonTracker() for class_id in COCO_CLASSES}
    seen: set[int] = set()
    matched = inference_frames = invalid_decode = 0
    started_all = time.perf_counter()
    with Reader(bag_path) as reader:
        connections = [item for item in reader.connections if item.topic.lstrip("/") == x8.STITCHED_TOPIC]
        require(len(connections) == 1, "x17_stitched_topic_missing")
        for connection in connections:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, _bag_time, raw in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            match = x9._nearest_frame(x9.stamp_ns(message.header.stamp), ordered, by_stamp)
            if match is None:
                continue
            frame, _delta_ns = match
            if frame in seen:
                continue
            seen.add(frame)
            matched += 1
            image = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (rgb_bridge.IMAGE_HEIGHT, rgb_bridge.IMAGE_WIDTH):
                invalid_decode += 1
                previous = None
                continue
            started = time.perf_counter()
            masks = _infer(model, image, trackers, timestamps[frame])
            inference_frames += 1
            if previous is not None and previous[0] == frame - 1 and frame - 1 in poses and frame in poses:
                index = index_by_frame[frame]
                admitted, diag = _authorize(
                    source_rows[index],
                    previous_pose=poses[frame - 1],
                    current_pose=poses[frame],
                    delta_s=timestamps[frame] - timestamps[frame - 1],
                    previous_masks=previous[1],
                    current_masks=masks,
                    calibration=calibration,
                )
                current = source_rows[index]
                keep = np.flatnonzero(admitted)
                authorized[index] = (current[0][keep], current[1][keep], current[2][keep], current[3][keep])
                diagnostics[index]["authorized_cells"] = int(len(authorized[index][0]))
                diagnostics[index]["associated_track_masks"] = int(diag["associated_mask_pairs"])
                diagnostics[index]["projected_anchors"] = int(diag["projected_anchors"])
                diagnostics[index]["birth_seconds"] = time.perf_counter() - started
            previous = (frame, masks)
            if inference_frames % 100 == 0:
                write_json(progress_path, {"schema": PROGRESS_SCHEMA, "stage": "YOLO11_TRACKED_MASK_AUTHORIZATION", "completed_frames": inference_frames, "total_frames": len(frames), "active_frame": frame, "last_activity_unix_s": time.time()})
    return authorized, diagnostics, {"requested_frames": len(frames), "matched_frames": matched, "inference_frames": inference_frames, "invalid_decodes": invalid_decode, "elapsed_s": time.perf_counter() - started_all}


def materialize_sequence(args: argparse.Namespace) -> dict[str, Any]:
    require(args.sequence is not None, "x17_sequence_required")
    root = args.root.resolve()
    freeze = _paths(root)["freeze"].resolve(strict=True)
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x17_freeze_drift")
    require(args.sequence in _baseline_rows(args), f"x17_unknown_sequence:{args.sequence}")
    sequence = args.sequence
    paths = _paths(root, sequence)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    if paths["ledger"].exists() and paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("ledger_sha256") == sha256_file(paths["ledger"]):
            return {"schema": SEQUENCE_SCHEMA, "status": "SEQUENCE_COMPLETE", "sequence": sequence, "resumed_from_sealed_ledger": True}
    x3._acquire_lock(paths["lock"])
    try:
        source_path, source_manifest_path = _source_paths(args.x7_root.resolve(strict=True), sequence)
        source = x1._load_sealed(source_path.resolve(strict=True), source_manifest_path.resolve(strict=True), x7.LEDGER_SCHEMA)
        frames = [int(frame) for frame in source["frames"]]
        timestamps = {int(frame): float(stamp) for frame, stamp in zip(source["frames"], source["frame_time_s"])}
        require(frames == list(range(frames[0], frames[-1] + 1)), f"x17_noncontiguous:{sequence}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        bag_path = Path(source_manifest["source"]["bag"]).resolve(strict=True)
        pose_samples, pose_audit = x9._read_poses(bag_path)
        poses = {}
        for frame in frames:
            try:
                poses[frame] = x8.c22.interpolate_pose(pose_samples, round(timestamps[frame] * 1e9))
            except (AssertionError, RuntimeError, ValueError):
                pass
        calibration = x8.c22.load_calibration(args.calibration_dir.resolve(strict=True))
        model = _make_model(args.model.resolve(strict=True))
        births, diagnostics, camera_audit = _stream_births(model=model, bag_path=bag_path, source=source, frames=frames, timestamps=timestamps, poses=poses, calibration=calibration, progress_path=paths["progress"])
        rows = []
        for index, frame in enumerate(frames):
            started = time.perf_counter()
            pieces = []
            source_index = index
            while source_index >= 0 and timestamps[frame] - timestamps[frames[source_index]] <= x14.CONTINUATION_S + 1e-9:
                if len(births[source_index][0]) and frames[source_index] in poses and frame in poses:
                    pieces.append(x14._transport(births[source_index], source_pose=poses[frames[source_index]], target_pose=poses[frame], delta_s=timestamps[frame] - timestamps[frames[source_index]]))
                source_index -= 1
            output = tuple(np.concatenate([piece[column] for piece in pieces], axis=0) for column in range(4)) if pieces else _empty_row()
            rows.append(output)
            diagnostics[index]["continued_cells"] = int(len(output[0]))
            diagnostics[index]["continuation_seconds"] = time.perf_counter() - started
        atomic_npz(paths["ledger"], **x5._pack_rows(frames, timestamps, rows))
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "oracle": False,
            "sequence": sequence,
            "frames": len(frames),
            "online_information_boundary": "frame t uses sealed X7 cells, stitched RGB tiles through t, YOLO11 masks with causal tracker IDs at t-1/t, ego poses through t, and authorized births no older than 0.50 seconds",
            "birth_rule": "FROZEN_YOLO11_SAME_CLASS_CAUSAL_TRACK_ID_AND_TWO_FRAME_X7_ANCHOR_MASK_MEMBERSHIP",
            "association_rule": "REAL_OBSERVATION_ADAPTER_CAUSAL_PERSON_TRACKER_DEFAULTS_PER_CLASS",
            "continuation_rule": "UNCHANGED_X14_X7_VELOCITY_TRANSPORT_FOR_R1_CLEAR_GRACE",
            "continuation_s": x14.CONTINUATION_S,
            "missing_evidence_policy": "FAIL_CLOSED_NO_NEW_BIRTH",
            "source": {"freeze_sha256": sha256_file(freeze), "x7_ledger_sha256": sha256_file(source_path), "x7_manifest_sha256": sha256_file(source_manifest_path), "bag": str(bag_path), "bag_sha256": sha256_file(bag_path), "pose_audit": pose_audit, "camera_audit": camera_audit, "model_sha256": MODEL_SHA256},
            "diagnostics": {"input_cells": int(sum(row["input_cells"] for row in diagnostics)), "authorized_cells": int(sum(row["authorized_cells"] for row in diagnostics)), "continued_cells": int(sum(row["continued_cells"] for row in diagnostics)), "frames": diagnostics},
            "ledger": str(paths["ledger"]),
            "ledger_sha256": sha256_file(paths["ledger"]),
        }
        write_json(paths["manifest"], manifest)
        write_json(paths["progress"], {"schema": PROGRESS_SCHEMA, "status": "COMPLETE", "sequence": sequence, "last_activity_unix_s": time.time()})
        return {"schema": SEQUENCE_SCHEMA, "status": "SEQUENCE_COMPLETE", "sequence": sequence, "frames": len(frames), "manifest_sha256": sha256_file(paths["manifest"])}
    finally:
        if paths["lock"].exists():
            paths["lock"].unlink()


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    freeze = _paths(root)["freeze"]
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x17_freeze_drift")
    receipts = []
    frames = input_cells = authorized = continued = 0
    for sequence in sorted(_baseline_rows(args)):
        paths = _paths(root, sequence)
        manifest = json.loads(paths["manifest"].resolve(strict=True).read_text(encoding="utf-8"))
        require(manifest.get("schema") == LEDGER_SCHEMA and manifest.get("truth_blind") is True, f"x17_manifest:{sequence}")
        require(manifest.get("ledger_sha256") == sha256_file(paths["ledger"].resolve(strict=True)), f"x17_hash:{sequence}")
        frames += int(manifest["frames"])
        input_cells += int(manifest["diagnostics"]["input_cells"])
        authorized += int(manifest["diagnostics"]["authorized_cells"])
        continued += int(manifest["diagnostics"]["continued_cells"])
        receipts.append({"sequence": sequence, "manifest_sha256": sha256_file(paths["manifest"])})
    require(frames == TIMELINE_FRAMES, f"x17_timeline_frames:{frames}")
    receipt = {"schema": MATERIALIZATION_SCHEMA, "status": "COMPLETE", "truth_blind": True, "sequences": len(receipts), "frames": frames, "input_cells": input_cells, "authorized_cells": authorized, "continued_cells": continued, "continuation_s": x14.CONTINUATION_S, "backend": {"python": platform.python_version(), "opencv": cv2.__version__, "model": "yolo11n-seg", "device": "CUDA"}, "freeze_sha256": sha256_file(freeze), "sequence_manifests": receipts}
    write_json(_paths(root)["materialization"], receipt)
    return receipt


def predict(args: argparse.Namespace) -> dict[str, Any]:
    previous = x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = LEDGER_SCHEMA, PREDICTION_SCHEMA
        result = x3.predict(args)
    finally:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    result["prediction_boundary"] = "sealed X17 YOLO11 tracked-instance-authorized bounded-continuation ledgers and unchanged X15/X3 route lifecycle only; no labels or evaluator identity"
    result["scorer_compatibility_arm_key"] = {"X3_LAG_FLOXEL": "X17_YOLO11_TRACKED_INSTANCE_CONTINUATION"}
    write_json(_paths(args.root.resolve(strict=True))["predictions"], result)
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    previous = x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = SCHEMA, LEDGER_SCHEMA, PREDICTION_SCHEMA
        result = x3.score(args)
    finally:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    met = bool(result["gate"]["passed"])
    result["schema"] = SCHEMA
    result["status"] = "DTR_X17_YOLO11_TRACKED_INSTANCE_CONTINUATION_GATE_MET" if met else "DTR_X17_YOLO11_TRACKED_INSTANCE_CONTINUATION_GATE_NOT_MET"
    result["metrics"]["X17_YOLO11_TRACKED_INSTANCE_CONTINUATION"] = result["metrics"].pop("X3_LAG_FLOXEL")
    result["decision"]["next"] = "FREEZE_X17_AND_CONFIRM_ON_NEW_SOURCE_DISJOINT_COHORT" if met else "ATTRIBUTE_X17_WITHOUT_PARAMETER_SWEEP"
    result["evidence_boundary"] = ["Full six-sequence replay on the consumed Development cohort; not source-disjoint confirmation.", "All X17 ledgers and predictions are sealed before native OBB truth is opened by the unchanged X15/X3 scorer.", "Only consecutive same-class CausalPersonTracker IDs whose YOLO11 polygons contain an X7 trajectory anchor in both frames can originate motion.", "Missing image, mask, pose, projection, tracker match, or CUDA evidence fails closed; only authorized birth can use frozen 0.50-second X14 continuation.", "X7 geometry/velocity, route geometry, lifecycle, and scorer remain unchanged."]
    write_json(_paths(args.root.resolve(strict=True))["result"], result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "materialize", "assemble", "predict", "score"))
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x17" / "yolo11-tracked-instance-continuation-replay-20260829-v2")
    parser.add_argument("--sequence")
    parser.add_argument("--x7-root", type=Path, default=x7_root)
    parser.add_argument("--x7-result", type=Path, default=x7_root / "result.json")
    parser.add_argument("--x14-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x14" / "rgb-authorized-motion-continuation-falsifier-20260829" / "result.json")
    parser.add_argument("--x15-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x15" / "full-rgb-authorized-continuation-replay-20260829" / "result.json")
    parser.add_argument("--baseline-predictions", type=Path, default=c31 / "baseline-predictions.json")
    parser.add_argument("--baseline-result", type=Path, default=c31 / "result.json")
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c31_fresh_confirmation_roster.json")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--calibration-dir", type=Path, default=REPO / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration")
    parser.add_argument("--model", type=Path, default=REPO / "artifacts.local" / "models" / "yolo11n-seg.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {"prepare": prepare, "materialize": materialize_sequence, "assemble": assemble, "predict": predict, "score": score}[args.mode](args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
