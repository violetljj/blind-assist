"""Test CIWT RGB track motion as fail-closed dynamic authority for X7 cells.

This post-outcome exploratory falsifier reuses the frozen X0/X6 60-frame
roster, X7 cells, C22 stitched projection, height anchors, confidence sigmas,
quantile, and decision threshold.  A cell is retained only when one current
CIWT box uniquely covers a projected anchor, the same track ID exists exactly
once in the preceding frame, and both seam-safe box-center and bottom-center
motion agree more closely with the moving than the ego-rigid static
projection.  Missing, ambiguous, malformed, or insufficient evidence rejects
the cell.  Native boxes, known-height tracks, and evaluator truth are never
used during materialization.

The bundled CIWT files have incomplete local generation/training provenance.
Consequently this is an exploratory source falsifier only, not authority
promotion or source-disjoint confirmation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_c22_ego_rigid_visual_motion as c22  # noqa: E402
import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x2_floxel_error_slice_canary as x2  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
import dtr_x7_full_static_world_anchor_replay as x7  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402
from jrdb_rgb_bridge import stamp_ns  # noqa: E402


SCHEMA = "blindassist-dtr-x12-ciwt-track-motion-agreement-falsifier-v1"
LEDGER_SCHEMA = "blindassist-dtr-x12-ciwt-track-motion-authority-ledger-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x12-ciwt-track-motion-authority-materialization-v1"
X8_SUPPRESSED_SOURCE_ERRORS = 27
MINIMUM_SOURCE_ERROR_SUPPRESSION = 24
CIWT_INDEX_BY_SEQUENCE = {
    "huang-2-2019-01-25_0": 11,
    "huang-basement-2019-01-25_0": 12,
    "huang-lane-2019-02-12_0": 13,
    "memorial-court-2019-03-16_0": 15,
    "meyer-green-2019-03-16_0": 16,
    "tressider-2019-03-16_1": 25,
}


@dataclass(frozen=True)
class TrackBox:
    frame: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def _sequence_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "track-motion-authority.npz", base / "track-motion-authority.json"


def _source_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "lag-floxel.npz", base / "lag-floxel.json"


def _ciwt_path(root: Path, sequence: str) -> Path:
    require(sequence in CIWT_INDEX_BY_SEQUENCE, f"x12_ciwt_sequence:{sequence}")
    return root / f"{CIWT_INDEX_BY_SEQUENCE[sequence]:04d}.txt"


def _read_pose_samples(bag_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as error:
        raise RuntimeError("dtr_x12 requires rosbags") from error

    typestore = get_typestore(Stores.ROS1_NOETIC)
    poses: list[dict[str, Any]] = []
    with Reader(bag_path) as reader:
        connections = [item for item in reader.connections if item.topic.lstrip("/") == "tf"]
        for connection in connections:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, _bag_time, raw in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            for item in message.transforms:
                if item.header.frame_id.lstrip("/") != "odom" or item.child_frame_id.lstrip("/") != "base_link":
                    continue
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
    require(bool(poses), f"x12_pose_samples:{bag_path.name}")
    return poses, {"topic": "/tf", "authority": "odom_to_base_link"}


def _load_ciwt(path: Path) -> tuple[dict[int, list[TrackBox]], dict[str, Any]]:
    by_frame: dict[int, list[TrackBox]] = {}
    malformed = 0
    non_pedestrian = 0
    geometry_placeholders_only = True
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 18:
            malformed += 1
            continue
        if values[2] != "Pedestrian":
            non_pedestrian += 1
            continue
        try:
            row = TrackBox(
                frame=int(float(values[0])),
                track_id=int(float(values[1])),
                x1=float(values[6]),
                y1=float(values[7]),
                x2=float(values[8]),
                y2=float(values[9]),
                score=float(values[17]),
            )
        except ValueError as error:
            raise RuntimeError(f"x12_ciwt_parse:{path.name}:{line_number}") from error
        geometry_placeholders_only &= all(float(item) == -1.0 for item in values[10:17])
        by_frame.setdefault(row.frame, []).append(row)
    require(malformed == 0, f"x12_ciwt_malformed:{path.name}:{malformed}")
    require(non_pedestrian == 0, f"x12_ciwt_class:{path.name}:{non_pedestrian}")
    rows = sum(len(items) for items in by_frame.values())
    return by_frame, {
        "rows": rows,
        "frames_with_tracks": len(by_frame),
        "track_ids": len({item.track_id for items in by_frame.values() for item in items}),
        "score_minimum": min(item.score for items in by_frame.values() for item in items),
        "score_maximum": max(item.score for items in by_frame.values() for item in items),
        "three_d_fields_all_minus_one": geometry_placeholders_only,
    }


def _circular_width(x1: float, x2: float) -> float:
    return (float(x2) - float(x1)) % float(c22.IMAGE_WIDTH)


def _contains(box: TrackBox, point: np.ndarray) -> bool:
    x = float(point[0]) % float(c22.IMAGE_WIDTH)
    x1 = float(box.x1) % float(c22.IMAGE_WIDTH)
    width = _circular_width(box.x1, box.x2)
    dx = (x - x1) % float(c22.IMAGE_WIDTH)
    return dx <= width + 1e-9 and box.y1 - 1e-9 <= float(point[1]) <= box.y2 + 1e-9


def _box_references(box: TrackBox) -> tuple[np.ndarray, np.ndarray]:
    center_x = (float(box.x1) + 0.5 * _circular_width(box.x1, box.x2)) % float(c22.IMAGE_WIDTH)
    return (
        np.asarray([center_x, 0.5 * (box.y1 + box.y2)], dtype=np.float64),
        np.asarray([center_x, box.y2], dtype=np.float64),
    )


def _track_displacements(previous: TrackBox, current: TrackBox) -> tuple[np.ndarray, np.ndarray]:
    output = []
    for before, after in zip(_box_references(previous), _box_references(current)):
        dx = float(after[0] - before[0])
        dx -= round(dx / float(c22.IMAGE_WIDTH)) * float(c22.IMAGE_WIDTH)
        output.append(np.asarray([dx, float(after[1] - before[1])], dtype=np.float64))
    return output[0], output[1]


def _dynamic_confidence(
    *,
    previous_pose: dict[str, Any],
    current_pose: dict[str, Any],
    dt_s: float,
    row: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    previous_boxes: Sequence[TrackBox],
    current_boxes: Sequence[TrackBox],
    calibration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    positions, velocities, _counts, _support = row
    count = len(positions)
    if not count:
        return np.empty(0, dtype=np.float32), {
            "projected_anchors": 0,
            "unique_current_box_anchors": 0,
            "same_id_previous_anchors": 0,
            "moving_closer_anchors": 0,
            "authorized_cells": 0,
        }

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
    finite &= (source[:, 1] >= 0.0) & (source[:, 1] < c22.IMAGE_HEIGHT)
    finite &= (moving[:, 1] >= 0.0) & (moving[:, 1] < c22.IMAGE_HEIGHT)

    previous_by_id: dict[int, list[TrackBox]] = {}
    for box in previous_boxes:
        previous_by_id.setdefault(box.track_id, []).append(box)

    confidence = np.zeros(len(source), dtype=np.float64)
    unique_current = 0
    same_id_previous = 0
    moving_closer = 0
    for index in np.flatnonzero(finite):
        matches = [box for box in current_boxes if _contains(box, moving[index])]
        if len(matches) != 1:
            continue
        unique_current += 1
        current_box = matches[0]
        previous_matches = previous_by_id.get(current_box.track_id, [])
        if len(previous_matches) != 1:
            continue
        same_id_previous += 1
        displacements = _track_displacements(previous_matches[0], current_box)
        source_x = float(source[index, 0])
        rigid_u = float(c22._unwrap(np.asarray([rigid[index, 0]]), np.asarray([source_x]))[0])
        moving_u = float(c22._unwrap(np.asarray([moving[index, 0]]), np.asarray([source_x]))[0])
        proposed_motion = math.hypot(moving_u - rigid_u, float(moving[index, 1] - rigid[index, 1]))
        reference_confidence = []
        reference_moving_closer = []
        for displacement in displacements:
            actual_x = source_x + float(displacement[0])
            actual_y = float(source[index, 1] + displacement[1])
            static_error = math.hypot(actual_x - rigid_u, actual_y - float(rigid[index, 1]))
            moving_error = math.hypot(actual_x - moving_u, actual_y - float(moving[index, 1]))
            closer = moving_error < static_error
            q = math.exp(-0.5 * (moving_error / c22.AGREEMENT_SIGMA_PX) ** 2)
            q *= 1.0 - math.exp(-0.5 * (proposed_motion / c22.MOTION_SIGMA_PX) ** 2)
            reference_confidence.append(q if closer else 0.0)
            reference_moving_closer.append(closer)
        if all(reference_moving_closer):
            moving_closer += 1
            confidence[index] = min(reference_confidence)

    samples = confidence.reshape(count, len(anchors))
    output = np.zeros(count, dtype=np.float32)
    for index, values in enumerate(samples):
        positive = values[values > 0.0]
        if len(positive):
            coverage = len(positive) / len(values)
            output[index] = float(np.quantile(positive, c22.COMPONENT_QUANTILE) * coverage)
    return output, {
        "projected_anchors": int(np.count_nonzero(finite)),
        "unique_current_box_anchors": unique_current,
        "same_id_previous_anchors": same_id_previous,
        "moving_closer_anchors": moving_closer,
        "authorized_cells": int(np.count_nonzero(output >= c22.DECISION_CONFIDENCE)),
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    x0_path = args.x0_result.resolve(strict=True)
    selected = x5._selected_frames(json.loads(x0_path.read_text(encoding="utf-8")))
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    calibration = c22.load_calibration(args.calibration_dir.resolve(strict=True))
    manifests = []
    frame_seconds = []
    for sequence in sorted(selected):
        source_path, source_manifest_path = _source_paths(args.x7_root.resolve(strict=True), sequence)
        source = x1._load_sealed(source_path, source_manifest_path, x7.LEDGER_SCHEMA)
        source_frames = [int(frame) for frame in source["frames"]]
        timestamp_by_frame = {
            int(frame): float(stamp)
            for frame, stamp in zip(source["frames"], source["frame_time_s"])
        }
        output_frames = sorted(selected[sequence])
        needed = sorted(set(output_frames) | {frame - 1 for frame in output_frames})
        require(set(needed) <= set(source_frames), f"x12_source_frames:{sequence}")

        tracker_path = _ciwt_path(args.ciwt_root.resolve(strict=True), sequence).resolve(strict=True)
        boxes, tracker_audit = _load_ciwt(tracker_path)
        bag_path = (args.bag_root.resolve(strict=True) / f"{sequence}.bag").resolve(strict=True)
        pose_samples, pose_authority = _read_pose_samples(bag_path)
        poses = {}
        for frame in needed:
            try:
                poses[frame] = c22.interpolate_pose(
                    pose_samples, round(timestamp_by_frame[frame] * 1e9)
                )
            except (AssertionError, RuntimeError, ValueError):
                pass

        rows = []
        diagnostics = []
        for frame in output_frames:
            current = x5._frame_arrays(source, frame)
            previous = frame - 1
            started = time.perf_counter()
            missing = []
            if previous not in poses:
                missing.append("PREVIOUS_POSE")
            if frame not in poses:
                missing.append("CURRENT_POSE")
            if previous not in boxes:
                missing.append("PREVIOUS_TRACK_FRAME")
            if frame not in boxes:
                missing.append("CURRENT_TRACK_FRAME")
            if missing:
                confidence = np.zeros(len(current[0]), dtype=np.float32)
                diag = {
                    "projected_anchors": 0,
                    "unique_current_box_anchors": 0,
                    "same_id_previous_anchors": 0,
                    "moving_closer_anchors": 0,
                    "authorized_cells": 0,
                }
            else:
                confidence, diag = _dynamic_confidence(
                    previous_pose=poses[previous],
                    current_pose=poses[frame],
                    dt_s=timestamp_by_frame[frame] - timestamp_by_frame[previous],
                    row=current,
                    previous_boxes=boxes[previous],
                    current_boxes=boxes[frame],
                    calibration=calibration,
                )
            authorized = confidence >= c22.DECISION_CONFIDENCE
            keep = np.flatnonzero(authorized)
            seconds = time.perf_counter() - started
            frame_seconds.append(seconds)
            rows.append((current[0][keep], current[1][keep], current[2][keep], current[3][keep]))
            diagnostics.append(
                {
                    "frame": frame,
                    "previous_frame": previous,
                    "input_cells": int(len(current[0])),
                    "authorized_dynamic_cells": int(len(keep)),
                    "rejected_cells": int(len(current[0]) - len(keep)),
                    "missing_track_evidence": missing,
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
            "truth_blind_inference_fields": True,
            "exploratory_only": True,
            "selection_post_outcome": True,
            "sequence": sequence,
            "rule": "KEEP_ONLY_IF_UNIQUE_CIWT_TRACK_CENTER_AND_BOTTOM_MOTION_PASS_C22_MOVING_AGREEMENT_CONFIDENCE",
            "missing_or_ambiguous_evidence_policy": "REJECT_X7_CANDIDATE_FAIL_CLOSED",
            "fixed_visual_constants": {
                "image_width": c22.IMAGE_WIDTH,
                "image_height": c22.IMAGE_HEIGHT,
                "height_anchors_m": list(c22.HEIGHT_ANCHORS_M),
                "moving_agreement_sigma_px": c22.AGREEMENT_SIGMA_PX,
                "motion_strength_sigma_px": c22.MOTION_SIGMA_PX,
                "cell_quantile": c22.COMPONENT_QUANTILE,
                "decision_confidence": c22.DECISION_CONFIDENCE,
                "box_references": ["CENTER", "BOTTOM_CENTER"],
                "reference_combination": "MINIMUM_REQUIRING_BOTH_MOVING_CLOSER",
                "horizontal_motion": "SEAM_SAFE_CIRCULAR_X",
            },
            "frozen_downstream": {
                "cell_velocity": "UNCHANGED_X7",
                "route_entry_geometry": "UNCHANGED_R7",
                "event_scorer": "UNCHANGED_X6_FRAME_LOCAL_FALSIFIER",
                "alert_lifecycle": "NOT_USED",
            },
            "source": {
                "x7_ledger": str(source_path),
                "x7_ledger_sha256": sha256_file(source_path),
                "x7_manifest": str(source_manifest_path),
                "x7_manifest_sha256": sha256_file(source_manifest_path),
                "ciwt_tracker": str(tracker_path),
                "ciwt_tracker_sha256": sha256_file(tracker_path),
                "ciwt_audit": tracker_audit,
                "ciwt_provenance": "INCOMPLETE_LOCAL_GENERATION_AND_TRAINING_PROVENANCE",
                "bag": str(bag_path),
                "bag_sha256": sha256_file(bag_path),
                "bag_pose_authority": pose_authority,
                "calibration": calibration,
            },
            "diagnostics": {
                "frames": diagnostics,
                "input_cells": sum(row["input_cells"] for row in diagnostics),
                "authorized_dynamic_cells": sum(row["authorized_dynamic_cells"] for row in diagnostics),
                "rejected_cells": sum(row["rejected_cells"] for row in diagnostics),
            },
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
        }
        write_json(manifest_path, manifest)
        manifests.append(manifest)

    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "truth_blind_inference_fields": True,
        "exploratory_only": True,
        "selection_post_outcome": True,
        "sequences": len(manifests),
        "frames": sum(len(row["diagnostics"]["frames"]) for row in manifests),
        "input_cells": sum(row["diagnostics"]["input_cells"] for row in manifests),
        "authorized_dynamic_cells": sum(row["diagnostics"]["authorized_dynamic_cells"] for row in manifests),
        "rejected_cells": sum(row["diagnostics"]["rejected_cells"] for row in manifests),
        "source_compute_p95_s": float(np.quantile(np.asarray(frame_seconds), 0.95, method="higher")),
        "runtime_boundary": "CIWT parse excluded; stitched projection, association, track-motion confidence, and fail-closed filtering included",
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
    ledgers = _load_ledgers(args, [str(row["sequence"]) for row in units] + [x1.SEQUENCE])
    positive = x5._score_positive(args, ledgers[x1.SEQUENCE])
    rows = []
    for unit in units:
        risk_cells = x2._risk_cells(ledgers[str(unit["sequence"])], int(unit["frame"]))
        rows.append(
            {
                **unit,
                "ciwt_dynamic_authority_route_risk_cells": risk_cells,
                "suppressed": risk_cells == 0,
            }
        )
    source_rows = [row for row in rows if str(row["primary_cause"]) in x2.SOURCE_FAILURES]
    require(len(source_rows) == 34, "x12_source_error_count")
    suppressed = sum(bool(row["suppressed"]) for row in source_rows)
    materialization = json.loads(
        (args.root.resolve(strict=True) / "materialization.json").read_text(encoding="utf-8")
    )
    x8 = json.loads(args.x8_result.resolve(strict=True).read_text(encoding="utf-8"))
    x8_suppressed = int(x8["error_slice"]["suppressed_source_error_units"])
    gate = {
        "positive_correct_frames_at_least_two": positive["correct_frames"] >= 2,
        "positive_correct_route_frames_at_least_two": positive["correct_route_entry_frames"] >= 2,
        "source_error_suppression_at_least_24_of_34": suppressed >= MINIMUM_SOURCE_ERROR_SUPPRESSION,
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_X12_CIWT_TRACK_MOTION_AGREEMENT_FALSIFIER_GATE_MET"
            if met
            else "DTR_X12_CIWT_TRACK_MOTION_AGREEMENT_FALSIFIER_GATE_NOT_MET"
        ),
        "question": "Can sensor-disjoint CIWT track motion act as fail-closed positive dynamic authority for X7 cells?",
        "positive": positive,
        "error_slice": {
            "all_units": len(rows),
            "source_error_units": len(source_rows),
            "suppressed_source_error_units": suppressed,
            "retained_source_error_units": len(source_rows) - suppressed,
            "suppression_rate": suppressed / len(source_rows),
            "required_suppression_units": MINIMUM_SOURCE_ERROR_SUPPRESSION,
        },
        "units": rows,
        "gate": gate,
        "comparison_to_x8": {
            "x8_rule": "FAIL_OPEN_STATIC_EVIDENCE_VETO",
            "x12_rule": "FAIL_CLOSED_POSITIVE_DYNAMIC_AUTHORITY",
            "x8_suppressed_source_error_units": x8_suppressed,
            "x12_suppressed_source_error_units": suppressed,
            "suppression_delta_units": suppressed - x8_suppressed,
            "x8_positive_correct_frames": int(x8["positive"]["correct_frames"]),
            "x12_positive_correct_frames": int(positive["correct_frames"]),
            "x8_positive_correct_route_frames": int(x8["positive"]["correct_route_entry_frames"]),
            "x12_positive_correct_route_frames": int(positive["correct_route_entry_frames"]),
        },
        "runtime": {
            "source_compute_p95_s": float(materialization["source_compute_p95_s"]),
            "boundary": materialization["runtime_boundary"],
        },
        "decision": {
            "exploratory_mechanism_headroom": met,
            "next": (
                "REPLACE_UNBOUND_CIWT_WITH_HASH_BOUND_TRUTH_BLIND_TRACKER_MATERIALIZATION_BEFORE_PROMOTION"
                if met
                else "CLOSE_CIWT_TRACK_MOTION_AUTHORITY_WITHOUT_PARAMETER_SWEEP"
            ),
        },
        "evaluator_firewall": {
            "materialization": "X7 cells, CIWT 2D prediction fields, public calibration, and ego pose only",
            "truth": "native labels opened only by unchanged X6 scorer after all six ledgers were sealed",
        },
        "claim_limits": [
            "Post-outcome exploratory diagnostic on the opened X6 60-frame roster; not confirmation.",
            "Bundled CIWT generation and training provenance is incomplete, so no authority promotion is authorized.",
            "CIWT 3-D placeholder fields, native OBB, known-height tracks, and evaluator truth are excluded from inference.",
            "Missing, ambiguous, malformed, or no-previous-frame track evidence rejects the X7 candidate.",
        ],
        "sources": {
            "x7_result_sha256": sha256_file(args.x7_result.resolve(strict=True)),
            "x0_result_sha256": sha256_file(x0_path),
            "x8_result_sha256": sha256_file(args.x8_result.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
        },
    }
    write_json(args.root.resolve(strict=True) / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    toolkit = (
        REPO
        / "artifacts.local"
        / "datasets"
        / "ustrf-canonical-observation-source-authority-data-pack-r0"
        / "jrdb_toolkit"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "artifacts.local" / "evidence" / "dtr-x12" / "ciwt-track-motion-agreement-falsifier-20260829",
    )
    parser.add_argument("--x7-root", type=Path, default=x7_root)
    parser.add_argument("--x7-result", type=Path, default=x7_root / "result.json")
    parser.add_argument(
        "--x0-result",
        type=Path,
        default=REPO / "artifacts.local" / "evidence" / "dtr-x0" / "motion-source-attribution" / "result.json",
    )
    parser.add_argument(
        "--x8-result",
        type=Path,
        default=REPO / "artifacts.local" / "evidence" / "dtr-x8" / "rgb-static-veto-falsifier-20260829" / "result.json",
    )
    parser.add_argument(
        "--ciwt-root",
        type=Path,
        default=toolkit / "tracking_eval" / "TrackEval" / "data" / "trackers" / "jrdb" / "jrdb_2d_box_train" / "CIWT" / "data",
    )
    parser.add_argument(
        "--bag-root",
        type=Path,
        default=REPO / "artifacts.local" / "datasets" / "dtr-c31-jrdb-fresh-confirmation",
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--calibration-dir", type=Path, default=toolkit / "calibration")
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
