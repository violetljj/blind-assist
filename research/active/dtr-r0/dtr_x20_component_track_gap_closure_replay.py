"""Replay X7 through X13-birth-gated component/track gap closure.

Raw frozen-X13 birth cells may create or refresh ancestry only when their
current anchor is inside a live YOLO11 instance mask.  After that birth, a
frame may continue only current X7 cells with both the same source component
ID and membership in the same live track mask.  No other component in the
mask can be absorbed.  Image, pose, track-ID, or component-support loss drops
the ancestry immediately.  The resulting births use unchanged X14 0.50-second
transport and unchanged X3 route lifecycle/scoring.
"""

from __future__ import annotations

import argparse
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
import dtr_x13_stitched_dynamic_birth_authority_falsifier as x13  # noqa: E402
import dtr_x14_rgb_authorized_motion_continuation_falsifier as x14  # noqa: E402
import dtr_x17_yolo11_tracked_instance_continuation_replay as x17  # noqa: E402
import dtr_x18_x15_seeded_yolo11_track_bridge_replay as x18  # noqa: E402
import dtr_x19_rgb_birth_seeded_yolo11_track_bridge_replay as x19  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402
from real_observation_adapter import CausalPersonTracker  # noqa: E402


SCHEMA = "blindassist-dtr-x20-component-track-gap-closure-replay-v1"
LEDGER_SCHEMA = "blindassist-dtr-x20-component-track-gap-closure-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x20-component-track-gap-closure-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-x20-component-track-gap-closure-freeze-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x20-component-track-gap-closure-materialization-v1"
SEQUENCE_SCHEMA = "blindassist-dtr-x20-sequence-materialization-v1"
PROGRESS_SCHEMA = "blindassist-dtr-x20-progress-v1"
TIMELINE_FRAMES = x17.TIMELINE_FRAMES
ARM = "X20_COMPONENT_TRACK_GAP_CLOSURE"


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
    require(len(rows) == 6, "x20_sequence_count")
    return rows


def _sealed_x7(args: argparse.Namespace, sequence: str) -> tuple[Path, Path, dict[str, Any]]:
    ledger, manifest = _source_paths(args.x7_root.resolve(strict=True), sequence)
    value = json.loads(manifest.resolve(strict=True).read_text(encoding="utf-8"))
    require(value.get("schema") == x7.LEDGER_SCHEMA and value.get("truth_blind") is True, f"x20_x7_manifest:{sequence}")
    require(value.get("ledger_sha256") == sha256_file(ledger.resolve(strict=True)), f"x20_x7_hash:{sequence}")
    return ledger, manifest, value


def _fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    require(sha256_file(args.model.resolve(strict=True)) == x17.MODEL_SHA256, "x20_model_hash")
    sources = {}
    for sequence in sorted(_baseline_rows(args)):
        ledger, manifest, value = _sealed_x7(args, sequence)
        bag = Path(value["source"]["bag"]).resolve(strict=True)
        require(value["source"]["bag_sha256"] == sha256_file(bag), f"x20_bag_hash:{sequence}")
        sources[sequence] = {
            "x7_ledger_sha256": sha256_file(ledger),
            "x7_manifest_sha256": sha256_file(manifest),
            "bag_sha256": sha256_file(bag),
        }
    return {
        "schema": FREEZE_SCHEMA,
        "truth_blind_materialization": True,
        "oracle": False,
        "algorithm_files": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())}
            for path in (__file__, x19.__file__, x13.__file__, x17.__file__, x18.__file__, x14.__file__, x3.__file__)
        ],
        "source_config": {
            "candidate_base": "SEALED_X7",
            "ancestry_origin": "RECOMPUTED_RAW_X13_BIRTH_CELL_INSIDE_CURRENT_TRACK_MASK",
            "ancestry_key": "CLASS_ID_TRACK_ID_X7_COMPONENT_ID",
            "gap_support": "CURRENT_X7_SAME_COMPONENT_ID_AND_CURRENT_SAME_TRACK_MASK_MEMBERSHIP",
            "forbidden": "NO_MASK_WIDE_X7_COMPONENT_ABSORPTION",
            "drop_policy": "DROP_IMMEDIATELY_ON_IMAGE_POSE_TRACK_ID_OR_COMPONENT_SUPPORT_BREAK",
            "provider": "UNCHANGED_X19_X17_YOLO11_FIVE_TILES_PER_CLASS_TRACKERS",
            "model_sha256": x17.MODEL_SHA256,
            "x13_decision_confidence": x8.c22.DECISION_CONFIDENCE,
            "continuation_s": x14.CONTINUATION_S,
            "new_duration_or_threshold": False,
        },
        "frozen_downstream": {
            "transport": "UNCHANGED_X14_0_50_SECOND",
            "route_lifecycle": "UNCHANGED_X3",
            "scorer": "UNCHANGED_X3",
        },
        "inputs": {
            "x13_result_sha256": sha256_file(args.x13_result.resolve(strict=True)),
            "x15_result_sha256": sha256_file(args.x15_result.resolve(strict=True)),
            "x17_result_sha256": sha256_file(args.x17_result.resolve(strict=True)),
            "x19_result_sha256": sha256_file(args.x19_result.resolve(strict=True)),
            "x7_result_sha256": sha256_file(args.x7_result.resolve(strict=True)),
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve(strict=True)),
            "baseline_result_sha256": sha256_file(args.baseline_result.resolve(strict=True)),
            "roster_sha256": sha256_file(args.roster.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
            "calibration_cameras_sha256": sha256_file(args.calibration_dir.resolve(strict=True) / "cameras.yaml"),
            "sequences": sources,
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    freeze = _paths(root)["freeze"]
    value = _fingerprint(args)
    if freeze.exists():
        require(json.loads(freeze.read_text(encoding="utf-8")) == value, "x20_freeze_drift")
    else:
        write_json(freeze, value)
    return {"schema": FREEZE_SCHEMA, "status": "READY", "sequences": sorted(_baseline_rows(args)), "freeze_sha256": sha256_file(freeze)}


def _empty_row() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), np.float64),
        np.empty((0, 2), np.float64),
        np.empty(0, np.int32),
        np.empty(0, np.float32),
    )


def _component_ids(source: Mapping[str, np.ndarray], frame: int) -> np.ndarray:
    matches = np.flatnonzero(source["frames"] == int(frame))
    require(len(matches) == 1, f"x20_component_frame:{frame}")
    index = int(matches[0])
    start = int(source["offsets"][index])
    stop = int(source["offsets"][index + 1])
    values = source["component_id"][start:stop].astype(np.int32)
    require(len(values) == stop - start, f"x20_component_count:{frame}")
    return values


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
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as error:
        raise RuntimeError("dtr_x20 requires rosbags") from error

    typestore = get_typestore(Stores.ROS1_NOETIC)
    target_ns = {frame: round(timestamps[frame] * 1e9) for frame in frames}
    by_stamp = {stamp: frame for frame, stamp in target_ns.items()}
    require(len(by_stamp) == len(target_ns), "x20_duplicate_frame_timestamp")
    ordered = sorted(by_stamp)
    rows = [x5._frame_arrays(source, frame) for frame in frames]
    component_rows = [_component_ids(source, frame) for frame in frames]
    index_by_frame = {frame: index for index, frame in enumerate(frames)}
    for frame, row, component_ids in zip(frames, rows, component_rows):
        require(len(row[0]) == len(component_ids), f"x20_component_alignment:{frame}")

    births = [_empty_row() for _ in frames]
    diagnostics = [
        {
            "frame": frame,
            "x7_cells": int(len(row[0])),
            "x13_birth_cells": 0,
            "visible_tracks": 0,
            "ancestry_states": 0,
            "created_or_refreshed_states": 0,
            "gap_supported_states": 0,
            "authorized_cells": 0,
        }
        for frame, row in zip(frames, rows)
    ]
    trackers = {class_id: CausalPersonTracker() for class_id in x17.COCO_CLASSES}
    ancestry: set[tuple[int, str, int]] = set()
    previous: tuple[int, np.ndarray] | None = None
    seen: set[int] = set()
    matched = inferred = invalid = gaps = pose_breaks = support_drops = 0

    with Reader(bag_path) as reader:
        connections = [item for item in reader.connections if item.topic.lstrip("/") == x8.STITCHED_TOPIC]
        require(len(connections) == 1, "x20_stitched_topic_missing")
        connection = connections[0]
        if connection.msgtype not in typestore.fielddefs:
            typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, _bag_time, raw in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            match = x9._nearest_frame(x9.stamp_ns(message.header.stamp), ordered, by_stamp)
            if match is None:
                continue
            frame, _delta = match
            if frame in seen:
                continue
            seen.add(frame)
            matched += 1
            image = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (x17.rgb_bridge.IMAGE_HEIGHT, x17.rgb_bridge.IMAGE_WIDTH):
                invalid += 1
                previous = None
                ancestry.clear()
                trackers = {class_id: CausalPersonTracker() for class_id in x17.COCO_CLASSES}
                continue
            if previous is not None and previous[0] != frame - 1:
                gaps += 1
                previous = None
                ancestry.clear()
                trackers = {class_id: CausalPersonTracker() for class_id in x17.COCO_CLASSES}

            masks = x17._infer(model, image, trackers, timestamps[frame])
            inferred += 1
            index = index_by_frame[frame]
            current = rows[index]
            component_ids = component_rows[index]
            valid_pose_chain = previous is not None and previous[0] == frame - 1 and frame - 1 in poses and frame in poses
            if not valid_pose_chain:
                if ancestry:
                    pose_breaks += 1
                ancestry.clear()
                diagnostics[index]["visible_tracks"] = len(masks)
                previous = (frame, image)
                continue

            mask_by_track = {(mask.class_id, mask.track_id): mask for mask in masks}
            live = {(class_id, track_id) for class_id, tracker in trackers.items() for track_id in tracker._tracks}
            prior = {key for key in ancestry if key[:2] in live}
            support_by_track: dict[tuple[int, str], np.ndarray] = {}
            for track_key, mask in mask_by_track.items():
                membership, _hits = x18._cell_membership(current, pose=poses[frame], masks=[mask], calibration=calibration)
                support_by_track[track_key] = membership

            confidence, _diag = x13._dynamic_confidence(
                previous_gray=cv2.cvtColor(previous[1], cv2.COLOR_BGR2GRAY),
                current_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                previous_pose=poses[frame - 1],
                current_pose=poses[frame],
                dt_s=timestamps[frame] - timestamps[frame - 1],
                row=current,
                calibration=calibration,
            )
            raw_birth = confidence >= x8.c22.DECISION_CONFIDENCE
            authorized = np.zeros(len(current[0]), dtype=bool)
            next_ancestry: set[tuple[int, str, int]] = set()
            gap_supported = 0

            for class_id, track_id, component_id in prior:
                membership = support_by_track.get((class_id, track_id))
                if membership is None:
                    support_drops += 1
                    continue
                supported = membership & (component_ids == component_id)
                if not np.any(supported):
                    support_drops += 1
                    continue
                authorized |= supported
                next_ancestry.add((class_id, track_id, component_id))
                gap_supported += 1

            refreshed: set[tuple[int, str, int]] = set()
            for track_key, membership in support_by_track.items():
                birth_indices = np.flatnonzero(raw_birth & membership)
                if not len(birth_indices):
                    continue
                authorized[birth_indices] = True
                for component_id in np.unique(component_ids[birth_indices]):
                    key = (track_key[0], track_key[1], int(component_id))
                    next_ancestry.add(key)
                    refreshed.add(key)

            ancestry = next_ancestry
            authorized_indices = np.flatnonzero(authorized)
            births[index] = tuple(column[authorized_indices] for column in current)
            diagnostics[index].update(
                {
                    "x13_birth_cells": int(np.count_nonzero(raw_birth)),
                    "visible_tracks": len(masks),
                    "ancestry_states": len(ancestry),
                    "created_or_refreshed_states": len(refreshed),
                    "gap_supported_states": gap_supported,
                    "authorized_cells": len(authorized_indices),
                }
            )
            previous = (frame, image)
            if inferred % 100 == 0:
                write_json(
                    progress_path,
                    {
                        "schema": PROGRESS_SCHEMA,
                        "stage": "X13_BIRTH_GATED_COMPONENT_TRACK_GAP_CLOSURE",
                        "completed_frames": inferred,
                        "total_frames": len(frames),
                        "active_frame": frame,
                        "last_activity_unix_s": time.time(),
                    },
                )
    return births, diagnostics, {
        "requested_frames": len(frames),
        "matched_frames": matched,
        "inferred_frames": inferred,
        "invalid_decodes": invalid,
        "image_gaps": gaps,
        "pose_chain_breaks_with_state": pose_breaks,
        "component_or_mask_support_drops": support_drops,
    }


def materialize_sequence(args: argparse.Namespace) -> dict[str, Any]:
    require(args.sequence is not None, "x20_sequence_required")
    root = args.root.resolve()
    freeze = _paths(root)["freeze"].resolve(strict=True)
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x20_freeze_drift")
    require(args.sequence in _baseline_rows(args), f"x20_unknown_sequence:{args.sequence}")
    sequence = args.sequence
    paths = _paths(root, sequence)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    if paths["ledger"].exists() and paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("ledger_sha256") == sha256_file(paths["ledger"]):
            return {"schema": SEQUENCE_SCHEMA, "status": "SEQUENCE_COMPLETE", "sequence": sequence, "resumed_from_sealed_ledger": True}

    x3._acquire_lock(paths["lock"])
    try:
        source_path, source_manifest_path, source_manifest = _sealed_x7(args, sequence)
        source = x1._load_sealed(source_path, source_manifest_path, x7.LEDGER_SCHEMA)
        frames = [int(frame) for frame in source["frames"]]
        timestamps = {int(frame): float(stamp) for frame, stamp in zip(source["frames"], source["frame_time_s"])}
        bag_path = Path(source_manifest["source"]["bag"]).resolve(strict=True)
        pose_samples, pose_audit = x9._read_poses(bag_path)
        poses = {}
        for frame in frames:
            try:
                poses[frame] = x8.c22.interpolate_pose(pose_samples, round(timestamps[frame] * 1e9))
            except (AssertionError, RuntimeError, ValueError):
                pass
        calibration = x8.c22.load_calibration(args.calibration_dir.resolve(strict=True))
        model = x17._make_model(args.model.resolve(strict=True))
        births, diagnostics, visual = _stream_births(
            model=model,
            bag_path=bag_path,
            source=source,
            frames=frames,
            timestamps=timestamps,
            poses=poses,
            calibration=calibration,
            progress_path=paths["progress"],
        )

        output_rows = []
        for index, frame in enumerate(frames):
            pieces = []
            source_index = index
            while source_index >= 0 and timestamps[frame] - timestamps[frames[source_index]] <= x14.CONTINUATION_S + 1e-9:
                if len(births[source_index][0]) and frames[source_index] in poses and frame in poses:
                    pieces.append(
                        x14._transport(
                            births[source_index],
                            source_pose=poses[frames[source_index]],
                            target_pose=poses[frame],
                            delta_s=timestamps[frame] - timestamps[frames[source_index]],
                        )
                    )
                source_index -= 1
            output = tuple(np.concatenate([piece[column] for piece in pieces], axis=0) for column in range(4)) if pieces else _empty_row()
            output_rows.append(output)
            diagnostics[index]["continued_cells"] = int(len(output[0]))

        atomic_npz(paths["ledger"], **x5._pack_rows(frames, timestamps, output_rows))
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "oracle": False,
            "sequence": sequence,
            "frames": len(frames),
            "birth_rule": "RAW_X13_BIRTH_CELL_INSIDE_CURRENT_TRACK_MASK",
            "gap_rule": "SAME_TRACK_ID_AND_SAME_X7_COMPONENT_ID_WITH_CURRENT_MASK_MEMBERSHIP",
            "forbidden": "NO_OTHER_X7_COMPONENT_ABSORPTION",
            "drop_rule": "IMMEDIATE_ON_IMAGE_POSE_TRACK_ID_OR_COMPONENT_SUPPORT_BREAK",
            "continuation_rule": "UNCHANGED_X14_0_50_SECOND_TRANSPORT",
            "source": {
                "freeze_sha256": sha256_file(freeze),
                "x7_ledger_sha256": sha256_file(source_path),
                "bag_sha256": sha256_file(bag_path),
                "model_sha256": x17.MODEL_SHA256,
                "pose_audit": pose_audit,
                "visual": visual,
            },
            "diagnostics": {
                "input_cells": int(sum(row["x7_cells"] for row in diagnostics)),
                "x13_birth_cells": int(sum(row["x13_birth_cells"] for row in diagnostics)),
                "authorized_cells": int(sum(row["authorized_cells"] for row in diagnostics)),
                "continued_cells": int(sum(row["continued_cells"] for row in diagnostics)),
                "created_or_refreshed_states": int(sum(row["created_or_refreshed_states"] for row in diagnostics)),
                "gap_supported_states": int(sum(row["gap_supported_states"] for row in diagnostics)),
                "frames": diagnostics,
            },
            "ledger": str(paths["ledger"]),
            "ledger_sha256": sha256_file(paths["ledger"]),
        }
        write_json(paths["manifest"], manifest)
        write_json(paths["progress"], {"schema": PROGRESS_SCHEMA, "status": "COMPLETE", "sequence": sequence})
        return {"schema": SEQUENCE_SCHEMA, "status": "SEQUENCE_COMPLETE", "sequence": sequence, "frames": len(frames), "manifest_sha256": sha256_file(paths["manifest"])}
    finally:
        if paths["lock"].exists():
            paths["lock"].unlink()


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    freeze = _paths(root)["freeze"]
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x20_freeze_drift")
    receipts = []
    frames = inputs = x13_births = authorized = continued = 0
    for sequence in sorted(_baseline_rows(args)):
        paths = _paths(root, sequence)
        manifest = json.loads(paths["manifest"].resolve(strict=True).read_text(encoding="utf-8"))
        require(manifest.get("schema") == LEDGER_SCHEMA and manifest.get("ledger_sha256") == sha256_file(paths["ledger"].resolve(strict=True)), f"x20_manifest:{sequence}")
        diag = manifest["diagnostics"]
        frames += int(manifest["frames"])
        inputs += int(diag["input_cells"])
        x13_births += int(diag["x13_birth_cells"])
        authorized += int(diag["authorized_cells"])
        continued += int(diag["continued_cells"])
        receipts.append({"sequence": sequence, "manifest_sha256": sha256_file(paths["manifest"])})
    require(frames == TIMELINE_FRAMES, f"x20_timeline_frames:{frames}")
    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "COMPLETE",
        "truth_blind": True,
        "sequences": len(receipts),
        "frames": frames,
        "input_cells": inputs,
        "x13_birth_cells": x13_births,
        "authorized_cells": authorized,
        "continued_cells": continued,
        "continuation_s": x14.CONTINUATION_S,
        "backend": {"python": platform.python_version(), "opencv": cv2.__version__, "model": "yolo11n-seg", "device": "CUDA"},
        "freeze_sha256": sha256_file(freeze),
        "sequence_manifests": receipts,
    }
    write_json(_paths(root)["materialization"], receipt)
    return receipt


def predict(args: argparse.Namespace) -> dict[str, Any]:
    previous = x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = LEDGER_SCHEMA, PREDICTION_SCHEMA
        result = x3.predict(args)
    finally:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    result["prediction_boundary"] = "sealed X20 X13-birth-gated same-track/same-component gap closure plus unchanged X14/X3 lifecycle; no labels"
    result["scorer_compatibility_arm_key"] = {"X3_LAG_FLOXEL": ARM}
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
    result["status"] = "DTR_X20_COMPONENT_TRACK_GAP_CLOSURE_GATE_MET" if met else "DTR_X20_COMPONENT_TRACK_GAP_CLOSURE_GATE_NOT_MET"
    result["metrics"][ARM] = result["metrics"].pop("X3_LAG_FLOXEL")
    result["decision"]["next"] = "FREEZE_X20_AND_CONFIRM_SOURCE_DISJOINT" if met else "CLOSE_COMPONENT_TRACK_GAP_CLOSURE_WITHOUT_SWEEP"
    result["evidence_boundary"] = [
        "X20 uses recomputed raw X13 births as the only ancestry origin.",
        "A gap cell must retain both the same live tracker ID and the same sealed-X7 component ID; no other component in the mask is admitted.",
        "Native truth opens only in the unchanged X3 scorer on the consumed six-sequence Development cohort.",
    ]
    write_json(_paths(args.root.resolve(strict=True))["result"], result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "materialize", "assemble", "predict", "score"))
    parser.add_argument("--sequence")
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x20" / "component-track-gap-closure-replay-20260829")
    parser.add_argument("--x7-root", type=Path, default=x7_root)
    parser.add_argument("--x7-result", type=Path, default=x7_root / "result.json")
    parser.add_argument("--x13-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x13" / "stitched-dynamic-birth-authority-falsifier-20260829" / "result.json")
    parser.add_argument("--x15-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x15" / "full-rgb-authorized-continuation-replay-20260829" / "result.json")
    parser.add_argument("--x17-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x17" / "yolo11-tracked-instance-continuation-replay-20260829-v2" / "result.json")
    parser.add_argument("--x19-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x19" / "rgb-birth-seeded-yolo11-track-bridge-replay-20260829" / "result.json")
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
    payload = {
        "prepare": prepare,
        "materialize": materialize_sequence,
        "assemble": assemble,
        "predict": predict,
        "score": score,
    }[args.mode](args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
