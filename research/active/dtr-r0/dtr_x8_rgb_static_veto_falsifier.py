"""Veto X7 cells only when independent RGB confirms an ego-rigid static point.

This post-outcome 60-frame falsifier reuses the frozen X6 roster and its gate.
The X7 cell velocity, route test, scorer, and lifecycle are unchanged.  For each
X7 cell, sparse stitched-RGB tracks test the static-world reprojection at five
C22 height anchors.  A cell is removed only when valid forward/backward tracks
agree with the static reprojection, disagree with the velocity-conditioned
moving alternative, and meet C22's unchanged confidence rule.  Missing RGB,
invalid projection, or failed tracking always retains the X7 candidate.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
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

import dtr_c22_ego_rigid_visual_motion as c22  # noqa: E402
import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x2_floxel_error_slice_canary as x2  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
import dtr_x6_static_world_persistence_falsifier as x6  # noqa: E402
import dtr_x7_full_static_world_anchor_replay as x7  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402
from jrdb_rgb_bridge import stamp_ns  # noqa: E402


SCHEMA = "blindassist-dtr-x8-rgb-static-veto-falsifier-v1"
LEDGER_SCHEMA = "blindassist-dtr-x8-rgb-static-veto-ledger-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x8-rgb-static-veto-materialization-v1"
STITCHED_TOPIC = "ros_indigosdk_node/stitched_image0/compressed"
RAW_TOPICS = tuple(f"ros_indigosdk_node/image{index}/compressed" for index in range(10))
MAXIMUM_IMAGE_DELTA_S = 0.001


def _sequence_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "rgb-static-veto.npz", base / "rgb-static-veto.json"


def _source_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "lag-floxel.npz", base / "lag-floxel.json"


def _nearest_targets(stamp: int, ordered: Sequence[int]) -> tuple[int, ...]:
    index = bisect_left(ordered, stamp)
    return tuple(ordered[item] for item in (index - 1, index) if 0 <= item < len(ordered))


def _read_visual_context(
    bag_path: Path, target_ns: Mapping[int, int]
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as error:
        raise RuntimeError("dtr_x8 requires rosbags") from error

    typestore = get_typestore(Stores.ROS1_NOETIC)
    by_stamp = {value: frame for frame, value in target_ns.items()}
    ordered = sorted(by_stamp)
    best: dict[int, tuple[int, bytes]] = {}
    poses: list[dict[str, Any]] = []
    with Reader(bag_path) as reader:
        topic_names = {item.topic.lstrip("/") for item in reader.connections}
        selected = [
            item
            for item in reader.connections
            if item.topic.lstrip("/") in {"tf", STITCHED_TOPIC}
        ]
        for connection in selected:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(
                    get_types_from_msg(connection.msgdef.data, connection.msgtype)
                )
        for connection, _bag_time, raw in reader.messages(connections=selected):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            topic = connection.topic.lstrip("/")
            if topic == "tf":
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
                continue
            image_stamp = stamp_ns(message.header.stamp)
            for target in _nearest_targets(image_stamp, ordered):
                delta = abs(image_stamp - target)
                if delta <= round(MAXIMUM_IMAGE_DELTA_S * 1e9) and (
                    target not in best or delta < best[target][0]
                ):
                    best[target] = (delta, bytes(message.data))
    poses.sort(key=lambda row: int(row["timestamp_ns"]))
    images: dict[int, np.ndarray] = {}
    deltas = {}
    for target, (delta, payload) in best.items():
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        frame = by_stamp[target]
        if image is not None and image.shape == (c22.IMAGE_HEIGHT, c22.IMAGE_WIDTH):
            images[frame] = image
            deltas[str(frame)] = delta / 1e9
    return images, poses, {
        "stitched_topic_present": STITCHED_TOPIC in topic_names,
        "raw_camera_topics_present": [topic for topic in RAW_TOPICS if topic in topic_names],
        "requested_images": len(target_ns),
        "matched_images": len(images),
        "maximum_match_delta_s": max(deltas.values()) if deltas else None,
        "match_delta_s_by_frame": deltas,
    }


def _static_confidence(
    *,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_pose: dict[str, Any],
    current_pose: dict[str, Any],
    dt_s: float,
    row: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    calibration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    positions, velocities, _counts, _support = row
    count = len(positions)
    if not count:
        return np.empty(0, dtype=np.float32), {"tracks": 0, "valid_tracks": 0, "static_cells": 0}
    current_world_xy, velocity_world = x5._ego_to_world(
        positions.astype(np.float64), velocities.astype(np.float64), current_pose
    )
    previous_world_xy = current_world_xy - velocity_world * float(dt_s)
    anchors = np.asarray(c22.HEIGHT_ANCHORS_M, dtype=np.float64)

    def expand(xy: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                np.repeat(xy[:, 0], len(anchors)),
                np.repeat(xy[:, 1], len(anchors)),
                np.tile(anchors, count),
            )
        )

    previous_world = expand(previous_world_xy)
    current_world = expand(current_world_xy)
    source = c22._project(previous_world, previous_pose, calibration)
    rigid = c22._project(previous_world, current_pose, calibration)
    moving = c22._project(current_world, current_pose, calibration)
    finite = np.all(np.isfinite(source), axis=1)
    finite &= np.all(np.isfinite(rigid), axis=1) & np.all(np.isfinite(moving), axis=1)
    finite &= (source[:, 1] >= 2.0) & (source[:, 1] < c22.IMAGE_HEIGHT - 2.0)
    indices = np.flatnonzero(finite)
    confidence = np.zeros(len(source), dtype=np.float64)
    static_closer = np.zeros(len(source), dtype=bool)
    if len(indices):
        points = source[indices]
        actual, valid, fb_error, _lk_error = c22._lk_tracks(
            previous_gray, current_gray, points
        )
        rigid_u = c22._unwrap(rigid[indices, 0], points[:, 0])
        moving_u = c22._unwrap(moving[indices, 0], points[:, 0])
        actual_u = c22._unwrap(actual[:, 0], points[:, 0])
        static_error = np.linalg.norm(
            np.column_stack((actual_u - rigid_u, actual[:, 1] - rigid[indices, 1])),
            axis=1,
        )
        moving_error = np.linalg.norm(
            np.column_stack((actual_u - moving_u, actual[:, 1] - moving[indices, 1])),
            axis=1,
        )
        proposed_motion = np.linalg.norm(
            np.column_stack((moving_u - rigid_u, moving[indices, 1] - rigid[indices, 1])),
            axis=1,
        )
        q = np.exp(-0.5 * (fb_error / c22.FB_SIGMA_PX) ** 2)
        q *= np.exp(-0.5 * (static_error / c22.AGREEMENT_SIGMA_PX) ** 2)
        q *= 1.0 - np.exp(-0.5 * (proposed_motion / c22.MOTION_SIGMA_PX) ** 2)
        closer = static_error < moving_error
        q[~valid | ~closer] = 0.0
        confidence[indices] = q
        static_closer[indices] = closer & valid

    samples = confidence.reshape(count, len(anchors))
    output = np.zeros(count, dtype=np.float32)
    for index, values in enumerate(samples):
        positive = values[values > 0.0]
        if len(positive):
            coverage = len(positive) / len(values)
            output[index] = float(
                np.quantile(positive, c22.COMPONENT_QUANTILE) * coverage
            )
    return output, {
        "tracks": int(len(indices)),
        "valid_tracks": int(np.count_nonzero(confidence > 0.0)),
        "static_closer_tracks": int(np.count_nonzero(static_closer)),
        "static_cells": int(np.count_nonzero(output >= c22.DECISION_CONFIDENCE)),
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    x0_path = args.x0_result.resolve(strict=True)
    x0 = json.loads(x0_path.read_text(encoding="utf-8"))
    selected = x5._selected_frames(x0)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    calibration = c22.load_calibration(args.calibration_dir.resolve(strict=True))
    manifests = []
    frame_seconds = []
    for sequence in sorted(selected):
        source_path, source_manifest_path = _source_paths(
            args.x7_root.resolve(strict=True), sequence
        )
        source = x1._load_sealed(source_path, source_manifest_path, x7.LEDGER_SCHEMA)
        source_frames = [int(frame) for frame in source["frames"]]
        timestamp_by_frame = {
            int(frame): float(stamp)
            for frame, stamp in zip(source["frames"], source["frame_time_s"])
        }
        output_frames = sorted(selected[sequence])
        needed = sorted(set(output_frames) | {frame - 1 for frame in output_frames})
        require(set(needed) <= set(source_frames), f"x8_source_frames:{sequence}")
        target_ns = {frame: round(timestamp_by_frame[frame] * 1e9) for frame in needed}
        bag_path = args.bag_root.resolve(strict=True) / f"{sequence}.bag"
        images, pose_samples, camera = _read_visual_context(bag_path, target_ns)
        poses = {}
        for frame in needed:
            try:
                poses[frame] = c22.interpolate_pose(pose_samples, target_ns[frame])
            except (AssertionError, RuntimeError, ValueError):
                pass
        rows = []
        diagnostics = []
        for frame in output_frames:
            current = x5._frame_arrays(source, frame)
            previous = frame - 1
            started = time.perf_counter()
            missing = []
            if previous not in images:
                missing.append("PREVIOUS_RGB")
            if frame not in images:
                missing.append("CURRENT_RGB")
            if previous not in poses:
                missing.append("PREVIOUS_POSE")
            if frame not in poses:
                missing.append("CURRENT_POSE")
            if missing:
                confidence = np.zeros(len(current[0]), dtype=np.float32)
                diag = {"tracks": 0, "valid_tracks": 0, "static_cells": 0}
            else:
                confidence, diag = _static_confidence(
                    previous_gray=images[previous],
                    current_gray=images[frame],
                    previous_pose=poses[previous],
                    current_pose=poses[frame],
                    dt_s=timestamp_by_frame[frame] - timestamp_by_frame[previous],
                    row=current,
                    calibration=calibration,
                )
            static = confidence >= c22.DECISION_CONFIDENCE
            keep = np.flatnonzero(~static)
            seconds = time.perf_counter() - started
            frame_seconds.append(seconds)
            rows.append((current[0][keep], current[1][keep], current[2][keep], current[3][keep]))
            diagnostics.append(
                {
                    "frame": frame,
                    "previous_frame": previous,
                    "input_cells": int(len(current[0])),
                    "vetoed_static_cells": int(np.count_nonzero(static)),
                    "retained_cells": int(len(keep)),
                    "missing_visual_evidence": missing,
                    "seconds": seconds,
                    **diag,
                }
            )
        arrays = x5._pack_rows(output_frames, timestamp_by_frame, rows)
        ledger_path, manifest_path = _sequence_paths(root, sequence)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(ledger_path, **arrays)
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "selection_post_outcome": True,
            "sequence": sequence,
            "rule": "VETO_ONLY_IF_RGB_STATIC_CONFIDENCE_AT_LEAST_C22_DECISION_AND_STATIC_CLOSER_THAN_MOVING",
            "missing_evidence_policy": "RETAIN_X7_CANDIDATE",
            "fixed_visual_constants": {
                "height_anchors_m": list(c22.HEIGHT_ANCHORS_M),
                "forward_backward_sigma_px": c22.FB_SIGMA_PX,
                "static_agreement_sigma_px": c22.AGREEMENT_SIGMA_PX,
                "motion_strength_sigma_px": c22.MOTION_SIGMA_PX,
                "cell_quantile": c22.COMPONENT_QUANTILE,
                "decision_confidence": c22.DECISION_CONFIDENCE,
                "lk_window": list(c22.LK_WINDOW),
                "lk_levels": c22.LK_LEVELS,
            },
            "frozen_downstream": {
                "cell_velocity": "UNCHANGED_X7",
                "motion_bounds": "UNCHANGED_X7",
                "route_entry_geometry": "UNCHANGED_R7",
                "event_scorer": "UNCHANGED_X6_FRAME_LOCAL_FALSIFIER",
                "alert_lifecycle": "NOT_USED",
            },
            "camera_audit": camera,
            "source": {
                "x7_ledger": str(source_path),
                "x7_ledger_sha256": sha256_file(source_path),
                "x7_manifest": str(source_manifest_path),
                "x7_manifest_sha256": sha256_file(source_manifest_path),
                "bag": str(bag_path),
                "bag_sha256": sha256_file(bag_path),
                "calibration": calibration,
            },
            "diagnostics": {
                "frames": diagnostics,
                "input_cells": sum(row["input_cells"] for row in diagnostics),
                "vetoed_static_cells": sum(row["vetoed_static_cells"] for row in diagnostics),
                "retained_cells": sum(row["retained_cells"] for row in diagnostics),
            },
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
        }
        write_json(manifest_path, manifest)
        manifests.append(manifest)
    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "truth_blind": True,
        "selection_post_outcome": True,
        "sequences": len(manifests),
        "frames": sum(len(row["diagnostics"]["frames"]) for row in manifests),
        "input_cells": sum(row["diagnostics"]["input_cells"] for row in manifests),
        "vetoed_static_cells": sum(row["diagnostics"]["vetoed_static_cells"] for row in manifests),
        "retained_cells": sum(row["diagnostics"]["retained_cells"] for row in manifests),
        "source_compute_p95_s": float(np.quantile(np.asarray(frame_seconds), 0.95, method="higher")),
        "runtime_boundary": "projection, LK, confidence, and veto included; bag scan, image matching, and image decode excluded",
        "backend": {"kind": "cpu", "opencv": cv2.__version__, "processor": platform.processor()},
        "sequence_manifests": {
            row["sequence"]: sha256_file(_sequence_paths(root, row["sequence"])[1])
            for row in manifests
        },
    }
    write_json(root / "materialization.json", receipt)
    return receipt


def _load_ledgers(args: argparse.Namespace, sequences: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for sequence in sorted(set(sequences)):
        path, manifest = _sequence_paths(args.root.resolve(strict=True), sequence)
        output[sequence] = x1._load_sealed(path, manifest, LEDGER_SCHEMA)
    return output


def score(args: argparse.Namespace) -> dict[str, Any]:
    x0_path = args.x0_result.resolve(strict=True)
    x0 = json.loads(x0_path.read_text(encoding="utf-8"))
    units = x2._selected_units(x0)
    ledgers = _load_ledgers(
        args, [str(row["sequence"]) for row in units] + [x1.SEQUENCE]
    )
    positive = x5._score_positive(args, ledgers[x1.SEQUENCE])
    rows = []
    for unit in units:
        risk_cells = x2._risk_cells(
            ledgers[str(unit["sequence"])], int(unit["frame"])
        )
        rows.append({**unit, "rgb_static_veto_route_risk_cells": risk_cells, "suppressed": risk_cells == 0})
    source_rows = [row for row in rows if str(row["primary_cause"]) in x2.SOURCE_FAILURES]
    require(len(source_rows) == 34, "x8_source_error_count")
    suppressed = sum(bool(row["suppressed"]) for row in source_rows)
    required = math.ceil(x2.MINIMUM_SUPPRESSION_RATE * len(source_rows))
    materialization = json.loads((args.root.resolve(strict=True) / "materialization.json").read_text(encoding="utf-8"))
    p95_s = float(materialization["source_compute_p95_s"])
    gate = {
        "positive_correct_frames_at_least_two": positive["correct_frames"] >= 2,
        "positive_correct_route_frames_at_least_two": positive["correct_route_entry_frames"] >= 2,
        "source_error_suppression_at_least_24_of_34": suppressed >= required,
        "source_compute_p95_within_one_scan_period": p95_s <= x6.SOURCE_COMPUTE_BUDGET_S,
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": "DTR_X8_RGB_STATIC_VETO_FALSIFIER_GATE_MET" if met else "DTR_X8_RGB_STATIC_VETO_FALSIFIER_GATE_NOT_MET",
        "question": "Can independent ego-rigid RGB static evidence veto X7 pseudo-motion without deleting candidates when visual evidence is absent?",
        "positive": positive,
        "error_slice": {
            "all_units": len(rows),
            "source_error_units": len(source_rows),
            "suppressed_source_error_units": suppressed,
            "retained_source_error_units": len(source_rows) - suppressed,
            "suppression_rate": suppressed / len(source_rows),
            "required_suppression_units": required,
        },
        "units": rows,
        "gate": gate,
        "runtime": {
            "source_compute_p95_s": p95_s,
            "median_observed_scan_period_s": x6.SOURCE_COMPUTE_BUDGET_S,
            "boundary": materialization["runtime_boundary"],
        },
        "decision": {
            "mechanism_headroom": met,
            "next": "IMPLEMENT_FULL_X7_RGB_STATIC_VETO_REPLAY" if met else "CLOSE_X8_STITCHED_RGB_STATIC_VETO_WITHOUT_PARAMETER_SWEEP",
        },
        "claim_limits": [
            "Post-outcome diagnostic on the opened X6 60-frame roster; not confirmation.",
            "Only stitched RGB static evidence is tested; raw camera availability is audited but not consumed.",
            "Missing images, invalid projections, and failed tracks retain X7 candidates.",
            "The frame-local result does not establish full lifecycle performance, runtime deployment, user benefit, or safety.",
        ],
        "sources": {
            "x7_result_sha256": sha256_file(args.x7_result.resolve(strict=True)),
            "x0_result_sha256": sha256_file(x0_path),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
        },
    }
    write_json(args.root.resolve(strict=True) / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x8" / "rgb-static-veto-falsifier-20260829")
    parser.add_argument("--x7-root", type=Path, default=x7_root)
    parser.add_argument("--x7-result", type=Path, default=x7_root / "result.json")
    parser.add_argument("--x0-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x0" / "motion-source-attribution" / "result.json")
    parser.add_argument("--bag-root", type=Path, default=REPO / "artifacts.local" / "datasets" / "dtr-c31-jrdb-fresh-confirmation")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--calibration-dir", type=Path, default=REPO / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] | None = None
    if args.mode in {"materialize", "run"}:
        payload = materialize(args)
    if args.mode in {"score", "run"}:
        payload = score(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
