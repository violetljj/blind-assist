"""Replay X7 births authorized by X15-seeded YOLO11 track lifecycles.

X18 reuses X17's frozen YOLO11 five-tile instance source and per-class
CausalPersonTracker defaults.  A live tracker ID becomes permanently seeded
when its current mask contains any projected sealed-X15 cell anchor.  Only X7
cells whose current projected anchors lie inside a currently visible seeded
track mask can originate a birth.  Seeds are removed when the underlying
tracker ID ages out; missing visual evidence cannot create or extend a birth,
while an already authorized cell may finish the unchanged X14 0.50-second
continuation.  X15 and X7 ledgers are sealed algorithm inputs; labels remain
scorer-only.
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
import dtr_x14_rgb_authorized_motion_continuation_falsifier as x14  # noqa: E402
import dtr_x15_full_rgb_authorized_continuation_replay as x15  # noqa: E402
import dtr_x17_yolo11_tracked_instance_continuation_replay as x17  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402
from real_observation_adapter import CausalPersonTracker  # noqa: E402


SCHEMA = "blindassist-dtr-x18-x15-seeded-yolo11-track-bridge-replay-v1"
LEDGER_SCHEMA = "blindassist-dtr-x18-x15-seeded-yolo11-track-bridge-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x18-x15-seeded-yolo11-track-bridge-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-x18-x15-seeded-yolo11-track-bridge-freeze-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x18-x15-seeded-yolo11-track-bridge-materialization-v1"
SEQUENCE_SCHEMA = "blindassist-dtr-x18-sequence-materialization-v1"
PROGRESS_SCHEMA = "blindassist-dtr-x18-progress-v1"
TIMELINE_FRAMES = x17.TIMELINE_FRAMES


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
    require(len(rows) == 6, "x18_sequence_count")
    return rows


def _sealed_source(root: Path, sequence: str, schema: str) -> tuple[Path, Path, dict[str, Any]]:
    ledger, manifest = _source_paths(root.resolve(strict=True), sequence)
    value = json.loads(manifest.resolve(strict=True).read_text(encoding="utf-8"))
    require(value.get("schema") == schema and value.get("truth_blind") is True, f"x18_source_manifest:{sequence}:{schema}")
    require(value.get("ledger_sha256") == sha256_file(ledger.resolve(strict=True)), f"x18_source_hash:{sequence}:{schema}")
    return ledger, manifest, value


def _fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    require(sha256_file(args.model.resolve(strict=True)) == x17.MODEL_SHA256, "x18_model_hash")
    sources = {}
    for sequence in sorted(_baseline_rows(args)):
        x7_ledger, x7_manifest, x7_value = _sealed_source(args.x7_root, sequence, x7.LEDGER_SCHEMA)
        x15_ledger, x15_manifest, _x15_value = _sealed_source(args.x15_root, sequence, x15.LEDGER_SCHEMA)
        bag = Path(x7_value["source"]["bag"]).resolve(strict=True)
        require(x7_value["source"]["bag_sha256"] == sha256_file(bag), f"x18_bag_hash:{sequence}")
        sources[sequence] = {
            "x7_ledger_sha256": sha256_file(x7_ledger),
            "x7_manifest_sha256": sha256_file(x7_manifest),
            "x15_ledger_sha256": sha256_file(x15_ledger),
            "x15_manifest_sha256": sha256_file(x15_manifest),
            "bag": str(bag),
            "bag_sha256": sha256_file(bag),
        }
    return {
        "schema": FREEZE_SCHEMA,
        "truth_blind_materialization": True,
        "oracle": False,
        "algorithm_files": [{"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())} for path in (__file__, x17.__file__, x15.__file__, x14.__file__, x3.__file__)],
        "source_config": {
            "candidate_base": "SEALED_X7",
            "seed_source": "SEALED_X15_CURRENT_CELL_ANCHOR_IN_TRACK_MASK",
            "provider": "UNCHANGED_X17_YOLO11_FIVE_TILES_AND_PER_CLASS_CAUSAL_TRACKERS",
            "model_sha256": x17.MODEL_SHA256,
            "tile_width": x17.rgb_bridge.TILE_WIDTH,
            "tile_starts": list(x17.rgb_bridge.TILE_STARTS),
            "classes": {str(class_id): name for class_id, name in x17.COCO_CLASSES.items()},
            "tracker": "CausalPersonTracker defaults per class",
            "seed_lifetime": "UNTIL_TRACKER_ID_AGES_OUT",
            "birth_rule": "CURRENT_X7_ANCHOR_INSIDE_CURRENT_MASK_OF_LIVE_SEEDED_TRACK",
            "continuation_s": x14.CONTINUATION_S,
            "missing_visual_policy": "NO_BIRTH_EXISTING_CONTINUATION_EXPIRES_NORMALLY",
        },
        "frozen_downstream": {"transport": "UNCHANGED_X14", "route_lifecycle": "UNCHANGED_X15_X3", "scorer": "UNCHANGED_X15_X3"},
        "inputs": {
            "x15_result_sha256": sha256_file(args.x15_result.resolve(strict=True)),
            "x17_result_sha256": sha256_file(args.x17_result.resolve(strict=True)),
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
        require(json.loads(freeze.read_text(encoding="utf-8")) == value, "x18_freeze_drift")
    else:
        write_json(freeze, value)
    return {"schema": FREEZE_SCHEMA, "status": "READY", "sequences": sorted(_baseline_rows(args)), "freeze_sha256": sha256_file(freeze)}


def _empty_row() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.empty((0, 2), np.float64), np.empty((0, 2), np.float64), np.empty(0, np.int32), np.empty(0, np.float32))


def _cell_membership(
    row: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    pose: dict[str, Any],
    masks: Sequence[x17.InstanceMask],
    calibration: dict[str, Any],
) -> tuple[np.ndarray, dict[tuple[int, str], bool]]:
    positions, velocities, _counts, _support = row
    count = len(positions)
    cells = np.zeros(count, dtype=bool)
    hits = {(mask.class_id, mask.track_id): False for mask in masks}
    if not count or not masks:
        return cells, hits
    world_xy, _velocity = x5._ego_to_world(positions.astype(np.float64), velocities.astype(np.float64), pose)
    anchors = np.asarray(x8.c22.HEIGHT_ANCHORS_M, dtype=np.float64)
    world = np.column_stack((np.repeat(world_xy[:, 0], len(anchors)), np.repeat(world_xy[:, 1], len(anchors)), np.tile(anchors, count)))
    points = x8.c22._project(world, pose, calibration)
    for sample_index, point in enumerate(points):
        cell_index = sample_index // len(anchors)
        for mask in masks:
            if x17._mask_member(mask.mask, point):
                cells[cell_index] = True
                hits[(mask.class_id, mask.track_id)] = True
    return cells, hits


def _stream_births(
    *,
    model: Any,
    bag_path: Path,
    x7_source: Mapping[str, np.ndarray],
    x15_source: Mapping[str, np.ndarray],
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
        raise RuntimeError("dtr_x18 requires rosbags") from error
    typestore = get_typestore(Stores.ROS1_NOETIC)
    target_ns = {frame: round(timestamps[frame] * 1e9) for frame in frames}
    by_stamp = {stamp: frame for frame, stamp in target_ns.items()}
    require(len(by_stamp) == len(target_ns), "x18_duplicate_frame_timestamp")
    ordered = sorted(by_stamp)
    x7_rows = [x5._frame_arrays(x7_source, frame) for frame in frames]
    x15_rows = [x5._frame_arrays(x15_source, frame) for frame in frames]
    index_by_frame = {frame: index for index, frame in enumerate(frames)}
    births = [_empty_row() for _ in frames]
    diagnostics = [{"frame": frame, "x7_cells": int(len(row[0])), "x15_seed_cells": int(len(x15_rows[index][0])), "visible_tracks": 0, "seeded_live_tracks": 0, "authorized_cells": 0} for index, (frame, row) in enumerate(zip(frames, x7_rows))]
    trackers = {class_id: CausalPersonTracker() for class_id in x17.COCO_CLASSES}
    seeded: set[tuple[int, str]] = set()
    seen: set[int] = set()
    matched = inferred = invalid = 0
    with Reader(bag_path) as reader:
        connections = [item for item in reader.connections if item.topic.lstrip("/") == x8.STITCHED_TOPIC]
        require(len(connections) == 1, "x18_stitched_topic_missing")
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
            seen.add(frame); matched += 1
            image = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (x17.rgb_bridge.IMAGE_HEIGHT, x17.rgb_bridge.IMAGE_WIDTH):
                invalid += 1
                continue
            masks = x17._infer(model, image, trackers, timestamps[frame]); inferred += 1
            live = {(class_id, track_id) for class_id, tracker in trackers.items() for track_id in tracker._tracks}
            seeded.intersection_update(live)
            index = index_by_frame[frame]
            if frame in poses:
                _seed_cells, seed_hits = _cell_membership(x15_rows[index], pose=poses[frame], masks=masks, calibration=calibration)
                seeded.update(key for key, hit in seed_hits.items() if hit)
                seeded_masks = [mask for mask in masks if (mask.class_id, mask.track_id) in seeded]
                authorized, _hits = _cell_membership(x7_rows[index], pose=poses[frame], masks=seeded_masks, calibration=calibration)
                keep = np.flatnonzero(authorized)
                current = x7_rows[index]
                births[index] = (current[0][keep], current[1][keep], current[2][keep], current[3][keep])
                diagnostics[index].update({"visible_tracks": len(masks), "seeded_live_tracks": len(seeded), "authorized_cells": len(keep)})
            if inferred % 100 == 0:
                write_json(progress_path, {"schema": PROGRESS_SCHEMA, "stage": "X15_SEEDED_YOLO11_TRACK_AUTHORIZATION", "completed_frames": inferred, "total_frames": len(frames), "active_frame": frame, "last_activity_unix_s": time.time()})
    return births, diagnostics, {"requested_frames": len(frames), "matched_frames": matched, "inferred_frames": inferred, "invalid_decodes": invalid}


def materialize_sequence(args: argparse.Namespace) -> dict[str, Any]:
    require(args.sequence is not None, "x18_sequence_required")
    root = args.root.resolve(); freeze = _paths(root)["freeze"].resolve(strict=True)
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x18_freeze_drift")
    require(args.sequence in _baseline_rows(args), f"x18_unknown_sequence:{args.sequence}")
    sequence = args.sequence; paths = _paths(root, sequence); paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    if paths["ledger"].exists() and paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("ledger_sha256") == sha256_file(paths["ledger"]):
            return {"schema": SEQUENCE_SCHEMA, "status": "SEQUENCE_COMPLETE", "sequence": sequence, "resumed_from_sealed_ledger": True}
    x3._acquire_lock(paths["lock"])
    try:
        x7_path, x7_manifest_path, x7_manifest = _sealed_source(args.x7_root, sequence, x7.LEDGER_SCHEMA)
        x15_path, x15_manifest_path, _x15_manifest = _sealed_source(args.x15_root, sequence, x15.LEDGER_SCHEMA)
        x7_source = x1._load_sealed(x7_path, x7_manifest_path, x7.LEDGER_SCHEMA)
        x15_source = x1._load_sealed(x15_path, x15_manifest_path, x15.LEDGER_SCHEMA)
        require(np.array_equal(x7_source["frames"], x15_source["frames"]), f"x18_frame_alignment:{sequence}")
        frames = [int(frame) for frame in x7_source["frames"]]
        timestamps = {int(frame): float(stamp) for frame, stamp in zip(x7_source["frames"], x7_source["frame_time_s"])}
        bag_path = Path(x7_manifest["source"]["bag"]).resolve(strict=True)
        pose_samples, pose_audit = x9._read_poses(bag_path)
        poses = {}
        for frame in frames:
            try:
                poses[frame] = x8.c22.interpolate_pose(pose_samples, round(timestamps[frame] * 1e9))
            except (AssertionError, RuntimeError, ValueError):
                pass
        calibration = x8.c22.load_calibration(args.calibration_dir.resolve(strict=True))
        model = x17._make_model(args.model.resolve(strict=True))
        births, diagnostics, visual = _stream_births(model=model, bag_path=bag_path, x7_source=x7_source, x15_source=x15_source, frames=frames, timestamps=timestamps, poses=poses, calibration=calibration, progress_path=paths["progress"])
        rows = []
        for index, frame in enumerate(frames):
            pieces = []; source_index = index
            while source_index >= 0 and timestamps[frame] - timestamps[frames[source_index]] <= x14.CONTINUATION_S + 1e-9:
                if len(births[source_index][0]) and frames[source_index] in poses and frame in poses:
                    pieces.append(x14._transport(births[source_index], source_pose=poses[frames[source_index]], target_pose=poses[frame], delta_s=timestamps[frame] - timestamps[frames[source_index]]))
                source_index -= 1
            output = tuple(np.concatenate([piece[column] for piece in pieces], axis=0) for column in range(4)) if pieces else _empty_row()
            rows.append(output); diagnostics[index]["continued_cells"] = int(len(output[0]))
        atomic_npz(paths["ledger"], **x5._pack_rows(frames, timestamps, rows))
        manifest = {
            "schema": LEDGER_SCHEMA, "truth_blind": True, "oracle": False, "sequence": sequence, "frames": len(frames),
            "seed_rule": "LIVE_TRACK_ID_PERMANENTLY_SEEDED_BY_CURRENT_X15_ANCHOR_MEMBERSHIP_UNTIL_TRACK_DEATH",
            "birth_rule": "CURRENT_X7_ANCHOR_INSIDE_CURRENT_MASK_OF_LIVE_SEEDED_TRACK",
            "continuation_rule": "UNCHANGED_X14_0_50_SECOND_TRANSPORT",
            "missing_evidence_policy": "NO_BIRTH_EXISTING_CONTINUATION_EXPIRES",
            "source": {"freeze_sha256": sha256_file(freeze), "x7_ledger_sha256": sha256_file(x7_path), "x15_ledger_sha256": sha256_file(x15_path), "bag_sha256": sha256_file(bag_path), "model_sha256": x17.MODEL_SHA256, "pose_audit": pose_audit, "visual": visual},
            "diagnostics": {"x7_input_cells": int(sum(row["x7_cells"] for row in diagnostics)), "x15_seed_cells": int(sum(row["x15_seed_cells"] for row in diagnostics)), "authorized_cells": int(sum(row["authorized_cells"] for row in diagnostics)), "continued_cells": int(sum(row["continued_cells"] for row in diagnostics)), "frames": diagnostics},
            "ledger": str(paths["ledger"]), "ledger_sha256": sha256_file(paths["ledger"]),
        }
        write_json(paths["manifest"], manifest); write_json(paths["progress"], {"schema": PROGRESS_SCHEMA, "status": "COMPLETE", "sequence": sequence})
        return {"schema": SEQUENCE_SCHEMA, "status": "SEQUENCE_COMPLETE", "sequence": sequence, "frames": len(frames), "manifest_sha256": sha256_file(paths["manifest"])}
    finally:
        if paths["lock"].exists():
            paths["lock"].unlink()


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True); freeze = _paths(root)["freeze"]
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x18_freeze_drift")
    receipts = []; frames = x7_cells = seed_cells = authorized = continued = 0
    for sequence in sorted(_baseline_rows(args)):
        paths = _paths(root, sequence); manifest = json.loads(paths["manifest"].resolve(strict=True).read_text(encoding="utf-8"))
        require(manifest.get("schema") == LEDGER_SCHEMA and manifest.get("ledger_sha256") == sha256_file(paths["ledger"].resolve(strict=True)), f"x18_manifest:{sequence}")
        diag = manifest["diagnostics"]; frames += int(manifest["frames"]); x7_cells += int(diag["x7_input_cells"]); seed_cells += int(diag["x15_seed_cells"]); authorized += int(diag["authorized_cells"]); continued += int(diag["continued_cells"])
        receipts.append({"sequence": sequence, "manifest_sha256": sha256_file(paths["manifest"])})
    require(frames == TIMELINE_FRAMES, f"x18_timeline_frames:{frames}")
    receipt = {"schema": MATERIALIZATION_SCHEMA, "status": "COMPLETE", "truth_blind": True, "sequences": len(receipts), "frames": frames, "x7_input_cells": x7_cells, "x15_seed_cells": seed_cells, "authorized_cells": authorized, "continued_cells": continued, "continuation_s": x14.CONTINUATION_S, "backend": {"python": platform.python_version(), "opencv": cv2.__version__, "model": "yolo11n-seg", "device": "CUDA"}, "freeze_sha256": sha256_file(freeze), "sequence_manifests": receipts}
    write_json(_paths(root)["materialization"], receipt); return receipt


def predict(args: argparse.Namespace) -> dict[str, Any]:
    previous = x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = LEDGER_SCHEMA, PREDICTION_SCHEMA; result = x3.predict(args)
    finally:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    result["prediction_boundary"] = "sealed X18 X15-seeded YOLO11 track-bridge ledgers plus unchanged X15/X3 lifecycle; no labels"
    result["scorer_compatibility_arm_key"] = {"X3_LAG_FLOXEL": "X18_X15_SEEDED_YOLO11_TRACK_BRIDGE"}
    write_json(_paths(args.root.resolve(strict=True))["predictions"], result); return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    previous = x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = SCHEMA, LEDGER_SCHEMA, PREDICTION_SCHEMA; result = x3.score(args)
    finally:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    met = bool(result["gate"]["passed"]); result["schema"] = SCHEMA
    result["status"] = "DTR_X18_X15_SEEDED_YOLO11_TRACK_BRIDGE_GATE_MET" if met else "DTR_X18_X15_SEEDED_YOLO11_TRACK_BRIDGE_GATE_NOT_MET"
    result["metrics"]["X18_X15_SEEDED_YOLO11_TRACK_BRIDGE"] = result["metrics"].pop("X3_LAG_FLOXEL")
    result["decision"]["next"] = "FREEZE_X18_AND_CONFIRM_SOURCE_DISJOINT" if met else "ATTRIBUTE_X18_WITHOUT_PARAMETER_SWEEP"
    result["evidence_boundary"] = ["Consumed six-sequence Development replay, not confirmation.", "X15 and X7 ledgers plus current/past RGB and ego pose are algorithm inputs; native OBB truth opens only in the unchanged scorer.", "A tracker seed persists only through that tracker ID lifecycle; missing visual cannot extend the separate 0.50-second X14 continuation."]
    write_json(_paths(args.root.resolve(strict=True))["result"], result); return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    x15_root = REPO / "artifacts.local" / "evidence" / "dtr-x15" / "full-rgb-authorized-continuation-replay-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "materialize", "assemble", "predict", "score")); parser.add_argument("--sequence")
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x18" / "x15-seeded-yolo11-track-bridge-replay-20260829")
    parser.add_argument("--x7-root", type=Path, default=x7_root); parser.add_argument("--x7-result", type=Path, default=x7_root / "result.json")
    parser.add_argument("--x15-root", type=Path, default=x15_root); parser.add_argument("--x15-result", type=Path, default=x15_root / "result.json")
    parser.add_argument("--x17-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x17" / "yolo11-tracked-instance-continuation-replay-20260829-v2" / "result.json")
    parser.add_argument("--baseline-predictions", type=Path, default=c31 / "baseline-predictions.json"); parser.add_argument("--baseline-result", type=Path, default=c31 / "result.json")
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c31_fresh_confirmation_roster.json")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip"); parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--calibration-dir", type=Path, default=REPO / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration")
    parser.add_argument("--model", type=Path, default=REPO / "artifacts.local" / "models" / "yolo11n-seg.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); payload = {"prepare": prepare, "materialize": materialize_sequence, "assemble": assemble, "predict": predict, "score": score}[args.mode](args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
