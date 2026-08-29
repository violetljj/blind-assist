"""Replay frozen X8 RGB-static veto over all six X7 Development sequences.

X9 changes only source authority: sealed X7 static-world-anchor cells are vetoed
when the frozen X8 stitched-RGB rule independently confirms ego-rigid static
motion.  Missing RGB, pose, projection, or LK support retains the X7 candidate.
X3 velocity, motion bounds, route geometry, lifecycle, scorer, and the full X7
gate are unchanged.

Run ``prepare`` first, one ``materialize --sequence`` worker per sequence, then
``assemble``, ``predict``, and ``score``.  Native OBB truth is opened only by
``score`` after all candidate ledgers and predictions are sealed.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import os
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
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402
from jrdb_rgb_bridge import stamp_ns  # noqa: E402


SCHEMA = "blindassist-dtr-x9-full-rgb-static-veto-replay-v1"
LEDGER_SCHEMA = "blindassist-dtr-x9-full-rgb-static-veto-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x9-full-rgb-static-veto-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-x9-full-rgb-static-veto-freeze-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x9-full-rgb-static-veto-materialization-v1"
SEQUENCE_SCHEMA = "blindassist-dtr-x9-sequence-materialization-v1"
TIMELINE_FRAMES = 4811
SOURCE_SUPPORTED_FRAMES = 4787
WARMUP_FRAMES_PER_SEQUENCE = 4


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
    require(len(rows) == 6, "x9_sequence_count")
    return rows


def _fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    rows = _baseline_rows(args)
    source_inputs = {}
    for sequence in sorted(rows):
        ledger, manifest = _source_paths(args.x7_root.resolve(strict=True), sequence)
        value = json.loads(manifest.resolve(strict=True).read_text(encoding="utf-8"))
        require(value.get("schema") == x7.LEDGER_SCHEMA, f"x9_x7_schema:{sequence}")
        require(value.get("truth_blind") is True, f"x9_x7_truth:{sequence}")
        require(value.get("ledger_sha256") == sha256_file(ledger.resolve(strict=True)), f"x9_x7_hash:{sequence}")
        source_inputs[sequence] = {
            "ledger_sha256": sha256_file(ledger),
            "manifest_sha256": sha256_file(manifest),
            "bag_sha256": str(value["source"]["bag_sha256"]),
        }
    return {
        "schema": FREEZE_SCHEMA,
        "truth_blind_materialization": True,
        "algorithm_files": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())}
            for path in (__file__, x8.__file__, x7.__file__, x3.__file__)
        ],
        "source_config": {
            "base": "SEALED_X7_STATIC_WORLD_ANCHOR",
            "veto": "FROZEN_X8_RGB_STATIC_CONFIDENCE",
            "missing_evidence_policy": "RETAIN_X7_CANDIDATE",
            "stitched_topic": x8.STITCHED_TOPIC,
            "maximum_image_delta_s": x8.MAXIMUM_IMAGE_DELTA_S,
            "height_anchors_m": list(x8.c22.HEIGHT_ANCHORS_M),
            "forward_backward_sigma_px": x8.c22.FB_SIGMA_PX,
            "static_agreement_sigma_px": x8.c22.AGREEMENT_SIGMA_PX,
            "motion_strength_sigma_px": x8.c22.MOTION_SIGMA_PX,
            "cell_quantile": x8.c22.COMPONENT_QUANTILE,
            "decision_confidence": x8.c22.DECISION_CONFIDENCE,
            "lk_window": list(x8.c22.LK_WINDOW),
            "lk_levels": x8.c22.LK_LEVELS,
            "timeline_frames": TIMELINE_FRAMES,
            "source_supported_frames": SOURCE_SUPPORTED_FRAMES,
            "fail_closed_warmup_frames_per_sequence": WARMUP_FRAMES_PER_SEQUENCE,
        },
        "frozen_downstream": {
            "cell_velocity": "UNCHANGED_X7_X3",
            "motion_bounds": "UNCHANGED_X7_X3",
            "route_entry_geometry": "UNCHANGED_R7",
            "event_lifecycle": "UNCHANGED_X3",
            "scorer": "UNCHANGED_X7_X3",
        },
        "full_gate": {
            "minimum_contact_recall": x3.MINIMUM_CONTACT_RECALL,
            "maximum_false_segments": x3.MAXIMUM_FALSE_SEGMENTS,
            "minimum_event_f1": x3.MINIMUM_EVENT_F1,
            "minimum_median_lead_s": x3.MINIMUM_MEDIAN_LEAD_S,
            "minimum_dropout_recovery": x3.MINIMUM_DROPOUT_RECOVERY,
            "false_segments_must_be_below_pdc": True,
        },
        "inputs": {
            "x8_result_sha256": sha256_file(args.x8_result.resolve(strict=True)),
            "x7_result_sha256": sha256_file(args.x7_result.resolve(strict=True)),
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve(strict=True)),
            "baseline_result_sha256": sha256_file(args.baseline_result.resolve(strict=True)),
            "roster_sha256": sha256_file(args.roster.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
            "calibration_cameras_sha256": sha256_file(args.calibration_dir.resolve(strict=True) / "cameras.yaml"),
            "x7_sequences": source_inputs,
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    freeze = _paths(root)["freeze"]
    fingerprint = _fingerprint(args)
    if freeze.exists():
        require(json.loads(freeze.read_text(encoding="utf-8")) == fingerprint, "x9_freeze_drift")
    else:
        write_json(freeze, fingerprint)
    return {
        "schema": FREEZE_SCHEMA,
        "status": "READY",
        "sequences": sorted(_baseline_rows(args)),
        "freeze": str(freeze),
        "freeze_sha256": sha256_file(freeze),
    }


def _read_poses(bag_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as error:
        raise RuntimeError("dtr_x9 requires rosbags") from error
    typestore = get_typestore(Stores.ROS1_NOETIC)
    poses: list[dict[str, Any]] = []
    with Reader(bag_path) as reader:
        topic_names = {item.topic.lstrip("/") for item in reader.connections}
        selected = [item for item in reader.connections if item.topic.lstrip("/") == "tf"]
        require(len(selected) == 1, "x9_tf_topic_missing")
        connection = selected[0]
        if connection.msgtype not in typestore.fielddefs:
            typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, _bag_time, raw in reader.messages(connections=selected):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            for item in message.transforms:
                if (
                    item.header.frame_id.lstrip("/") == "odom"
                    and item.child_frame_id.lstrip("/") == "base_link"
                ):
                    transform = item.transform
                    poses.append(
                        {
                            "timestamp_ns": stamp_ns(item.header.stamp),
                            "translation": [
                                float(transform.translation.x),
                                float(transform.translation.y),
                                float(transform.translation.z),
                            ],
                            "quaternion_xyzw": [
                                float(transform.rotation.x),
                                float(transform.rotation.y),
                                float(transform.rotation.z),
                                float(transform.rotation.w),
                            ],
                        }
                    )
    poses.sort(key=lambda row: int(row["timestamp_ns"]))
    require(len(poses) >= 2, "x9_pose_samples_missing")
    return poses, {
        "stitched_topic_present": x8.STITCHED_TOPIC in topic_names,
        "raw_camera_topics_present": [topic for topic in x8.RAW_TOPICS if topic in topic_names],
        "dynamic_pose_topic": "tf",
        "dynamic_pose_edge": "odom->base_link",
    }


def _nearest_frame(stamp: int, ordered_stamps: Sequence[int], by_stamp: Mapping[int, int]) -> tuple[int, int] | None:
    index = bisect_left(ordered_stamps, stamp)
    candidates = [ordered_stamps[item] for item in (index - 1, index) if 0 <= item < len(ordered_stamps)]
    if not candidates:
        return None
    target = min(candidates, key=lambda value: (abs(value - stamp), value))
    delta = abs(target - stamp)
    if delta > round(x8.MAXIMUM_IMAGE_DELTA_S * 1e9):
        return None
    return int(by_stamp[target]), int(delta)


def _stream_veto(
    *,
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
        raise RuntimeError("dtr_x9 requires rosbags") from error
    typestore = get_typestore(Stores.ROS1_NOETIC)
    target_ns = {int(frame): round(float(timestamps[int(frame)]) * 1e9) for frame in frames}
    by_stamp = {stamp: frame for frame, stamp in target_ns.items()}
    require(len(by_stamp) == len(target_ns), "x9_duplicate_frame_timestamp")
    ordered_stamps = sorted(by_stamp)
    rows = [x5._frame_arrays(source, frame) for frame in frames]
    diagnostics = [
        {
            "frame": int(frame),
            "input_cells": int(len(row[0])),
            "vetoed_static_cells": 0,
            "retained_cells": int(len(row[0])),
            "visual_status": "RETAIN_MISSING_VISUAL_EVIDENCE",
            "seconds": 0.0,
            "tracks": 0,
            "valid_tracks": 0,
            "static_closer_tracks": 0,
            "static_cells": 0,
        }
        for frame, row in zip(frames, rows)
    ]
    index_by_frame = {int(frame): index for index, frame in enumerate(frames)}
    matched: dict[int, float] = {}
    decoded = invalid_decode = 0
    last_frame: int | None = None
    last_image: np.ndarray | None = None
    started_all = time.perf_counter()
    with Reader(bag_path) as reader:
        selected = [
            item for item in reader.connections if item.topic.lstrip("/") == x8.STITCHED_TOPIC
        ]
        require(len(selected) == 1, "x9_stitched_topic_missing")
        connection = selected[0]
        if connection.msgtype not in typestore.fielddefs:
            typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, _bag_time, raw in reader.messages(connections=selected):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            match = _nearest_frame(stamp_ns(message.header.stamp), ordered_stamps, by_stamp)
            if match is None:
                continue
            frame, delta_ns = match
            if frame in matched:
                continue
            image = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (x8.c22.IMAGE_HEIGHT, x8.c22.IMAGE_WIDTH):
                invalid_decode += 1
                last_frame, last_image = None, None
                continue
            decoded += 1
            matched[frame] = delta_ns / 1e9
            index = index_by_frame[frame]
            if (
                last_frame == frame - 1
                and last_image is not None
                and frame - 1 in poses
                and frame in poses
            ):
                started = time.perf_counter()
                confidence, diag = x8._static_confidence(
                    previous_gray=last_image,
                    current_gray=image,
                    previous_pose=poses[frame - 1],
                    current_pose=poses[frame],
                    dt_s=timestamps[frame] - timestamps[frame - 1],
                    row=rows[index],
                    calibration=calibration,
                )
                static = confidence >= x8.c22.DECISION_CONFIDENCE
                keep = np.flatnonzero(~static)
                current = rows[index]
                rows[index] = (
                    current[0][keep], current[1][keep], current[2][keep], current[3][keep]
                )
                seconds = time.perf_counter() - started
                diagnostics[index] = {
                    "frame": frame,
                    "input_cells": int(len(current[0])),
                    "vetoed_static_cells": int(np.count_nonzero(static)),
                    "retained_cells": int(len(keep)),
                    "visual_status": "EVALUATED",
                    "seconds": seconds,
                    **diag,
                }
            last_frame, last_image = frame, image
            if decoded % 100 == 0:
                write_json(
                    progress_path,
                    {
                        "schema": "blindassist-dtr-x9-progress-v1",
                        "completed_rgb_frames": decoded,
                        "total_frames": len(frames),
                        "percent": 100.0 * decoded / len(frames),
                        "active_frame": frame,
                        "last_activity_unix_s": time.time(),
                    },
                )
    return rows, diagnostics, {
        "requested_frames": len(frames),
        "matched_frames": len(matched),
        "evaluated_frames": sum(row["visual_status"] == "EVALUATED" for row in diagnostics),
        "retained_missing_visual_frames": sum(
            row["visual_status"] != "EVALUATED" for row in diagnostics
        ),
        "decoded_images": decoded,
        "invalid_decodes": invalid_decode,
        "maximum_match_delta_s": max(matched.values()) if matched else None,
        "elapsed_s": time.perf_counter() - started_all,
    }


def materialize_sequence(args: argparse.Namespace) -> dict[str, Any]:
    require(args.sequence is not None, "x9_sequence_required")
    root = args.root.resolve()
    freeze_path = _paths(root)["freeze"].resolve(strict=True)
    require(json.loads(freeze_path.read_text(encoding="utf-8")) == _fingerprint(args), "x9_freeze_drift")
    rows = _baseline_rows(args)
    require(args.sequence in rows, f"x9_unknown_sequence:{args.sequence}")
    sequence = args.sequence
    paths = _paths(root, sequence)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    if paths["ledger"].exists() and paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("ledger_sha256") == sha256_file(paths["ledger"]):
            return {
                "schema": SEQUENCE_SCHEMA,
                "status": "SEQUENCE_COMPLETE",
                "sequence": sequence,
                "frames": int(manifest["frames"]),
                "resumed_from_sealed_ledger": True,
                "manifest_sha256": sha256_file(paths["manifest"]),
            }
    x3._acquire_lock(paths["lock"])
    try:
        source_path, source_manifest_path = _source_paths(
            args.x7_root.resolve(strict=True), sequence
        )
        source = x1._load_sealed(
            source_path.resolve(strict=True),
            source_manifest_path.resolve(strict=True),
            x7.LEDGER_SCHEMA,
        )
        frames = [int(frame) for frame in source["frames"]]
        timestamps = {
            int(frame): float(stamp)
            for frame, stamp in zip(source["frames"], source["frame_time_s"])
        }
        require(frames == list(range(frames[0], frames[-1] + 1)), f"x9_noncontiguous:{sequence}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        bag_path = Path(source_manifest["source"]["bag"]).resolve(strict=True)
        pose_samples, camera_audit = _read_poses(bag_path)
        poses = {}
        for frame in frames:
            try:
                poses[frame] = x8.c22.interpolate_pose(
                    pose_samples, round(timestamps[frame] * 1e9)
                )
            except (AssertionError, RuntimeError, ValueError):
                pass
        calibration = x8.c22.load_calibration(args.calibration_dir.resolve(strict=True))
        output_rows, diagnostics, visual = _stream_veto(
            bag_path=bag_path,
            source=source,
            frames=frames,
            timestamps=timestamps,
            poses=poses,
            calibration=calibration,
            progress_path=paths["progress"],
        )
        arrays = x5._pack_rows(frames, timestamps, output_rows)
        atomic_npz(paths["ledger"], **arrays)
        evaluated_seconds = [
            float(row["seconds"])
            for row in diagnostics
            if row["visual_status"] == "EVALUATED"
        ]
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "oracle": False,
            "sequence": sequence,
            "frames": len(frames),
            "online_information_boundary": "frame t uses sealed X7 cells, current/past stitched RGB through t, and bag ego poses through t",
            "rule": "FROZEN_X8_RGB_STATIC_VETO_AFTER_X7_STATIC_WORLD_ANCHOR",
            "missing_evidence_policy": "RETAIN_X7_CANDIDATE",
            "frozen_downstream": {
                "cell_velocity": "UNCHANGED_X7_X3",
                "motion_bounds": "UNCHANGED_X7_X3",
                "route_entry_geometry": "UNCHANGED_R7",
                "event_lifecycle": "UNCHANGED_X3",
            },
            "source": {
                "freeze_sha256": sha256_file(freeze_path),
                "x7_ledger": str(source_path),
                "x7_ledger_sha256": sha256_file(source_path),
                "x7_manifest": str(source_manifest_path),
                "x7_manifest_sha256": sha256_file(source_manifest_path),
                "bag": str(bag_path),
                "bag_sha256": sha256_file(bag_path),
                "calibration": calibration,
                "camera_audit": camera_audit,
            },
            "diagnostics": {
                "input_cells": int(sum(row["input_cells"] for row in diagnostics)),
                "vetoed_static_cells": int(sum(row["vetoed_static_cells"] for row in diagnostics)),
                "retained_cells": int(sum(row["retained_cells"] for row in diagnostics)),
                "source_compute_p95_s": (
                    float(np.quantile(np.asarray(evaluated_seconds), 0.95, method="higher"))
                    if evaluated_seconds
                    else None
                ),
                "runtime_boundary": "projection, LK, confidence, and veto only; bag scan, image matching, and image decode excluded",
                "visual": visual,
            },
            "ledger": str(paths["ledger"]),
            "ledger_sha256": sha256_file(paths["ledger"]),
        }
        write_json(paths["manifest"], manifest)
        write_json(
            paths["progress"],
            {
                "schema": "blindassist-dtr-x9-progress-v1",
                "status": "COMPLETE",
                "completed_rgb_frames": visual["matched_frames"],
                "total_frames": len(frames),
                "percent": 100.0,
                "last_activity_unix_s": time.time(),
            },
        )
        return {
            "schema": SEQUENCE_SCHEMA,
            "status": "SEQUENCE_COMPLETE",
            "sequence": sequence,
            "frames": len(frames),
            "input_cells": manifest["diagnostics"]["input_cells"],
            "vetoed_static_cells": manifest["diagnostics"]["vetoed_static_cells"],
            "retained_cells": manifest["diagnostics"]["retained_cells"],
            "visual": visual,
            "manifest": str(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
        }
    finally:
        if paths["lock"].exists():
            paths["lock"].unlink()


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    freeze = _paths(root)["freeze"]
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x9_freeze_drift")
    receipts = []
    total_frames = total_input = total_vetoed = total_retained = 0
    matched = evaluated = missing = 0
    for sequence in sorted(_baseline_rows(args)):
        paths = _paths(root, sequence)
        manifest = json.loads(paths["manifest"].resolve(strict=True).read_text(encoding="utf-8"))
        require(manifest.get("schema") == LEDGER_SCHEMA, f"x9_schema:{sequence}")
        require(manifest.get("truth_blind") is True, f"x9_truth:{sequence}")
        require(manifest.get("ledger_sha256") == sha256_file(paths["ledger"].resolve(strict=True)), f"x9_hash:{sequence}")
        diagnostics = manifest["diagnostics"]
        visual = diagnostics["visual"]
        total_frames += int(manifest["frames"])
        total_input += int(diagnostics["input_cells"])
        total_vetoed += int(diagnostics["vetoed_static_cells"])
        total_retained += int(diagnostics["retained_cells"])
        matched += int(visual["matched_frames"])
        evaluated += int(visual["evaluated_frames"])
        missing += int(visual["retained_missing_visual_frames"])
        receipts.append(
            {
                "sequence": sequence,
                "frames": int(manifest["frames"]),
                "manifest": str(paths["manifest"]),
                "manifest_sha256": sha256_file(paths["manifest"]),
            }
        )
    require(total_frames == TIMELINE_FRAMES, f"x9_timeline_frame_count:{total_frames}")
    require(
        TIMELINE_FRAMES - SOURCE_SUPPORTED_FRAMES
        == WARMUP_FRAMES_PER_SEQUENCE * len(receipts),
        "x9_warmup_frame_count",
    )
    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "COMPLETE",
        "truth_blind": True,
        "sequences": len(receipts),
        "frames": total_frames,
        "source_supported_frames": SOURCE_SUPPORTED_FRAMES,
        "fail_closed_warmup_frames": TIMELINE_FRAMES - SOURCE_SUPPORTED_FRAMES,
        "input_cells": total_input,
        "vetoed_static_cells": total_vetoed,
        "retained_cells": total_retained,
        "visual": {
            "matched_frames": matched,
            "evaluated_frames": evaluated,
            "retained_missing_visual_frames": missing,
        },
        "backend": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "source": "CPU_SPARSE_PYRLK",
        },
        "freeze": str(freeze),
        "freeze_sha256": sha256_file(freeze),
        "sequence_manifests": receipts,
    }
    write_json(_paths(root)["materialization"], receipt)
    return receipt


def predict(args: argparse.Namespace) -> dict[str, Any]:
    previous_ledger, previous_prediction = x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.LEDGER_SCHEMA = LEDGER_SCHEMA
        x3.PREDICTION_SCHEMA = PREDICTION_SCHEMA
        result = x3.predict(args)
    finally:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous_ledger, previous_prediction
    result["prediction_boundary"] = (
        "sealed full X9 X7-plus-RGB-static-veto ledgers and frozen global route lifecycle only; "
        "no labels, roster event details, evaluator identity, or outcomes"
    )
    result["scorer_compatibility_arm_key"] = {"X3_LAG_FLOXEL": "X9_RGB_STATIC_VETO"}
    write_json(_paths(args.root.resolve(strict=True))["predictions"], result)
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    previous = (x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA)
    try:
        x3.SCHEMA = SCHEMA
        x3.LEDGER_SCHEMA = LEDGER_SCHEMA
        x3.PREDICTION_SCHEMA = PREDICTION_SCHEMA
        result = x3.score(args)
    finally:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    met = bool(result["gate"]["passed"])
    result["schema"] = SCHEMA
    result["status"] = (
        "DTR_X9_FULL_RGB_STATIC_VETO_GATE_MET"
        if met
        else "DTR_X9_FULL_RGB_STATIC_VETO_GATE_NOT_MET"
    )
    result["metrics"]["X9_RGB_STATIC_VETO"] = result["metrics"].pop("X3_LAG_FLOXEL")
    result["decision"]["next"] = (
        "FREEZE_X9_AND_CONFIRM_ON_NEW_SOURCE_DISJOINT_COHORT"
        if met
        else "CLOSE_OR_ATTRIBUTE_X9_FULL_REPLAY_WITHOUT_PARAMETER_SWEEP"
    )
    result["evidence_boundary"] = [
        "Full six-sequence replay on the already opened C31/X0 Development cohort; not new source-disjoint confirmation.",
        "All X9 ledgers and predictions were sealed before native OBB truth was opened by this scorer.",
        "X9 changes only source authority by applying frozen X8 independent RGB static veto after sealed X7 cells.",
        "Missing visual evidence retains X7 candidates; X3 velocity, route, lifecycle, and scorer are unchanged.",
        "Real-device latency, product benefit, and safety are not established.",
    ]
    write_json(_paths(args.root.resolve(strict=True))["result"], result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    x8_root = REPO / "artifacts.local" / "evidence" / "dtr-x8" / "rgb-static-veto-falsifier-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "materialize", "assemble", "predict", "score"))
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x9" / "full-rgb-static-veto-replay-20260829-v2")
    parser.add_argument("--sequence")
    parser.add_argument("--x7-root", type=Path, default=x7_root)
    parser.add_argument("--x7-result", type=Path, default=x7_root / "result.json")
    parser.add_argument("--x8-result", type=Path, default=x8_root / "result.json")
    parser.add_argument("--baseline-predictions", type=Path, default=c31 / "baseline-predictions.json")
    parser.add_argument("--baseline-result", type=Path, default=c31 / "result.json")
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c31_fresh_confirmation_roster.json")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--calibration-dir", type=Path, default=REPO / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        payload = prepare(args)
    elif args.mode == "materialize":
        payload = materialize_sequence(args)
    elif args.mode == "assemble":
        payload = assemble(args)
    elif args.mode == "predict":
        payload = predict(args)
    else:
        payload = score(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
