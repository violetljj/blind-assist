"""Run DTR-M3-D realized-future truth-contract decomposition.

This scorer-side diagnostic consumes the sealed M2-D cohort and changes no
prediction, threshold, lifecycle, event, or gate.  At every M2 diagnostic
origin it compares four future-label geometry contracts for the physical
native component that supplied M1 motion evidence:

* EVAL-CIRCLE: realized future center plus evaluator circularized radius;
* REALIZED-OBB: realized future center, yaw, length, and width;
* REALIZED-CENTER + CURRENT-OBB: realized center with current shape held fixed;
* CV-CENTER + REALIZED-SHAPE: M1 constant-velocity center with realized shape.

A discrete CV-CENTER + CURRENT-OBB control is also reported beside the exact
continuous M2-D result so sample-time effects cannot masquerade as dynamics.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from coda_static_ceiling import point_to_box_clearance
from dtr_m1_point_velocity_oracle import NativeBox, load_native_boxes
from dtr_r5_dropout_canary import SegmentCase, cases_from_tracks
from dtr_r7_occupancy_flow_canary import _causal_pose
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    FIRST_FRAME,
    HORIZON_S,
    LAST_FRAME,
    ROUTE_HALF_WIDTH_M,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import load_truth_and_associate, read_jsonl, write_json


SCHEMA = "blindassist-dtr-m3-realized-future-contract-decomposition-v1"
STATUS = "DTR_M3_D_REALIZED_FUTURE_CONTRACT_DECOMPOSITION_COMPLETE"
CLAIM_CEILING = "PRIVILEGED_REALIZED_FUTURE_LABEL_DIAGNOSTIC_ON_CONSUMED_M1_M2_COHORT"


def _box_clearance(
    center_forward_m: float,
    center_left_m: float,
    yaw_ego_rad: float,
    length_m: float,
    width_m: float,
) -> float:
    return point_to_box_clearance(
        0.0,
        0.0,
        center_forward_m,
        center_left_m,
        yaw_ego_rad,
        length_m,
        width_m,
    ) - ROUTE_HALF_WIDTH_M


def _arm_summary(rows: Sequence[dict[str, Any]], clearance_key: str) -> dict[str, Any]:
    values = [(float(row[clearance_key]), row) for row in rows]
    if not values:
        return {
            "future_samples": 0,
            "hit": False,
            "first_hit_frame": None,
            "first_hit_delta_s": None,
            "minimum_clearance_m": None,
            "minimum_clearance_frame": None,
        }
    minimum, minimum_row = min(values, key=lambda item: item[0])
    first_hit = next((row for row in rows if float(row[clearance_key]) <= 1e-9), None)
    return {
        "future_samples": len(rows),
        "hit": first_hit is not None,
        "first_hit_frame": None if first_hit is None else int(first_hit["frame"]),
        "first_hit_delta_s": None if first_hit is None else float(first_hit["delta_s"]),
        "minimum_clearance_m": minimum,
        "minimum_clearance_frame": int(minimum_row["frame"]),
    }


def classify_positive_contract(
    *,
    eval_circle_hit: bool,
    realized_obb_hit: bool,
    realized_center_current_obb_hit: bool,
    cv_center_realized_shape_hit: bool,
    cv_center_current_obb_hit: bool,
) -> str:
    if eval_circle_hit and not realized_obb_hit:
        return "EVAL_CIRCLE_HIT_REALIZED_OBB_MISS"
    if not realized_obb_hit:
        return "NO_REALIZED_OBB_CONTACT"
    if cv_center_current_obb_hit:
        return "CV_CURRENT_OBB_ALREADY_HITS"
    if realized_center_current_obb_hit and cv_center_realized_shape_hit:
        return "CENTER_AND_SHAPE_ARMS_EACH_SUFFICIENT"
    if realized_center_current_obb_hit:
        return "REALIZED_CENTER_DYNAMICS_SUFFICIENT"
    if cv_center_realized_shape_hit:
        return "REALIZED_SHAPE_YAW_DYNAMICS_SUFFICIENT"
    return "COUPLED_REALIZED_CENTER_AND_SHAPE_REQUIRED"


def classify_false_contract(
    *,
    source_is_target: bool,
    eval_circle_hit: bool,
    realized_obb_hit: bool,
) -> str:
    if not source_is_target:
        return "ATTRIBUTED_OTHER_NATIVE_COMPONENT"
    if not eval_circle_hit and realized_obb_hit:
        return "EVALUATOR_CIRCLE_NEGATIVE_REALIZED_OBB_HIT"
    if not realized_obb_hit:
        return "REALIZED_OBB_MISS_FORECAST_FALSE_POSITIVE"
    return "EVALUATOR_AND_REALIZED_OBB_HIT_SEGMENT_MATCHING_MISMATCH"


def _find_case(cases_by_label: dict[str, list[SegmentCase]], label_id: str, frame: int) -> SegmentCase:
    matches = [
        case
        for case in cases_by_label.get(label_id, ())
        if any(sample.frame_index == frame for sample in case.samples)
    ]
    require(len(matches) == 1, f"component_case_not_unique:{label_id}:{frame}:{len(matches)}")
    return matches[0]


def _future_contract(
    *,
    case: SegmentCase,
    origin_frame: int,
    current_box: NativeBox,
    velocity_forward_mps: float,
    velocity_left_mps: float,
    boxes_by_frame: dict[int, list[NativeBox]],
) -> dict[str, Any]:
    origin_index = next(
        index for index, sample in enumerate(case.samples) if sample.frame_index == origin_frame
    )
    origin = case.samples[origin_index]
    future_samples = [
        sample
        for sample in case.samples[origin_index:]
        if sample.time_s - origin.time_s <= HORIZON_S + 1e-9
    ]
    future_rows = []
    for sample in future_samples:
        future_box = next(
            (
                box
                for box in boxes_by_frame.get(sample.frame_index, ())
                if box.label_id == case.label_id
            ),
            None,
        )
        require(future_box is not None, f"future_native_box_missing:{case.label_id}:{sample.frame_index}")
        delta_s = sample.time_s - origin.time_s
        eval_circle_clearance = sample.distance_m - (
            ROUTE_HALF_WIDTH_M + sample.truth_radius_m
        )
        realized_obb_clearance = _box_clearance(
            future_box.center_forward_m,
            future_box.center_left_m,
            future_box.yaw_ego_rad,
            future_box.length_m,
            future_box.width_m,
        )
        realized_center_current_obb_clearance = _box_clearance(
            future_box.center_forward_m,
            future_box.center_left_m,
            current_box.yaw_ego_rad,
            current_box.length_m,
            current_box.width_m,
        )
        cv_forward = current_box.center_forward_m + velocity_forward_mps * delta_s
        cv_left = current_box.center_left_m + velocity_left_mps * delta_s
        cv_center_realized_shape_clearance = _box_clearance(
            cv_forward,
            cv_left,
            future_box.yaw_ego_rad,
            future_box.length_m,
            future_box.width_m,
        )
        cv_center_current_obb_clearance = _box_clearance(
            cv_forward,
            cv_left,
            current_box.yaw_ego_rad,
            current_box.length_m,
            current_box.width_m,
        )
        future_rows.append(
            {
                "frame": sample.frame_index,
                "delta_s": delta_s,
                "evaluator_truth": case.truth[origin_index],
                "realized_center_forward_m": future_box.center_forward_m,
                "realized_center_left_m": future_box.center_left_m,
                "realized_radius_m": sample.truth_radius_m,
                "realized_yaw_ego_rad": future_box.yaw_ego_rad,
                "realized_length_m": future_box.length_m,
                "realized_width_m": future_box.width_m,
                "cv_center_forward_m": cv_forward,
                "cv_center_left_m": cv_left,
                "center_residual_m": math.hypot(
                    future_box.center_forward_m - cv_forward,
                    future_box.center_left_m - cv_left,
                ),
                "eval_circle_clearance_m": eval_circle_clearance,
                "realized_obb_clearance_m": realized_obb_clearance,
                "realized_center_current_obb_clearance_m": (
                    realized_center_current_obb_clearance
                ),
                "cv_center_realized_shape_clearance_m": (
                    cv_center_realized_shape_clearance
                ),
                "cv_center_current_obb_clearance_m": cv_center_current_obb_clearance,
            }
        )

    arms = {
        "eval_circle": _arm_summary(future_rows, "eval_circle_clearance_m"),
        "realized_obb": _arm_summary(future_rows, "realized_obb_clearance_m"),
        "realized_center_current_obb": _arm_summary(
            future_rows, "realized_center_current_obb_clearance_m"
        ),
        "cv_center_realized_shape": _arm_summary(
            future_rows, "cv_center_realized_shape_clearance_m"
        ),
        "cv_center_current_obb_discrete_control": _arm_summary(
            future_rows, "cv_center_current_obb_clearance_m"
        ),
    }
    require(
        arms["eval_circle"]["hit"] == (case.truth[origin_index] is True),
        f"eval_circle_truth_replay_drift:{case.label_id}:{origin_frame}",
    )
    return {
        "origin_frame": origin_frame,
        "origin_time_s": origin.time_s,
        "label_id": case.label_id,
        "segment_index": case.segment_index,
        "evaluator_truth": case.truth[origin_index],
        "evaluator_known": case.known[origin_index],
        "current_box": {
            "center_forward_m": current_box.center_forward_m,
            "center_left_m": current_box.center_left_m,
            "yaw_ego_rad": current_box.yaw_ego_rad,
            "length_m": current_box.length_m,
            "width_m": current_box.width_m,
        },
        "m1_velocity_forward_mps": velocity_forward_mps,
        "m1_velocity_left_mps": velocity_left_mps,
        "arms": arms,
        "maximum_center_residual_m": max(
            (float(row["center_residual_m"]) for row in future_rows), default=None
        ),
        "future_samples": future_rows,
    }


def _aggregate_arm(contracts: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows = [contract["arms"][arm] for contract in contracts]
    minimum_rows = [row for row in rows if row["minimum_clearance_m"] is not None]
    first_rows = [row for row in rows if row["first_hit_delta_s"] is not None]
    return {
        "origins": len(rows),
        "hit": any(bool(row["hit"]) for row in rows),
        "minimum_clearance_m": min(
            (float(row["minimum_clearance_m"]) for row in minimum_rows), default=None
        ),
        "earliest_hit_delta_s": min(
            (float(row["first_hit_delta_s"]) for row in first_rows), default=None
        ),
    }


def _decompose_m2_row(
    *,
    row: dict[str, Any],
    cases_by_label: dict[str, list[SegmentCase]],
    boxes_by_frame: dict[int, list[NativeBox]],
) -> dict[str, Any]:
    contracts = []
    for frame in row["frames"]:
        for component in frame["components"]:
            label_id = str(component["label_id"])
            case = _find_case(cases_by_label, label_id, int(frame["frame"]))
            current_box = next(
                box
                for box in boxes_by_frame[int(frame["frame"])]
                if box.label_id == label_id
            )
            contract = _future_contract(
                case=case,
                origin_frame=int(frame["frame"]),
                current_box=current_box,
                velocity_forward_mps=float(component["velocity_forward_mps"]),
                velocity_left_mps=float(component["velocity_left_mps"]),
                boxes_by_frame=boxes_by_frame,
            )
            contract["is_target_component"] = bool(component["is_target_component"])
            contract["m2_continuous_cv_current_obb_hit"] = bool(
                component["footprint"]["hit"]
            )
            contract["m2_continuous_cv_current_obb_minimum_clearance_m"] = float(
                component["footprint"]["minimum_clearance_m"]
            )
            contracts.append(contract)

    arms = {
        arm: _aggregate_arm(contracts, arm)
        for arm in (
            "eval_circle",
            "realized_obb",
            "realized_center_current_obb",
            "cv_center_realized_shape",
            "cv_center_current_obb_discrete_control",
        )
    }
    source_labels = sorted({contract["label_id"] for contract in contracts})
    source_is_target = bool(contracts) and all(
        bool(contract["is_target_component"]) for contract in contracts
    )
    m2_continuous_hit = any(
        bool(contract["m2_continuous_cv_current_obb_hit"]) for contract in contracts
    )
    if row["truth_positive"]:
        classification = classify_positive_contract(
            eval_circle_hit=bool(arms["eval_circle"]["hit"]),
            realized_obb_hit=bool(arms["realized_obb"]["hit"]),
            realized_center_current_obb_hit=bool(
                arms["realized_center_current_obb"]["hit"]
            ),
            cv_center_realized_shape_hit=bool(
                arms["cv_center_realized_shape"]["hit"]
            ),
            cv_center_current_obb_hit=m2_continuous_hit,
        )
    else:
        classification = classify_false_contract(
            source_is_target=source_is_target,
            eval_circle_hit=bool(arms["eval_circle"]["hit"]),
            realized_obb_hit=bool(arms["realized_obb"]["hit"]),
        )
    return {
        "kind": row["kind"],
        "id": row["id"],
        "target_label_id": row["label_id"],
        "truth_positive": bool(row["truth_positive"]),
        "source_component_labels": source_labels,
        "source_is_target": source_is_target,
        "classification": classification,
        "arms": arms,
        "m2_continuous_cv_current_obb": {
            "hit": m2_continuous_hit,
            "minimum_clearance_m": min(
                (
                    float(contract["m2_continuous_cv_current_obb_minimum_clearance_m"])
                    for contract in contracts
                ),
                default=None,
            ),
        },
        "maximum_center_residual_m": max(
            (
                float(contract["maximum_center_residual_m"])
                for contract in contracts
                if contract["maximum_center_residual_m"] is not None
            ),
            default=None,
        ),
        "origins": contracts,
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8", newline="")
    os.replace(partial, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "kind",
        "id",
        "target_label_id",
        "source_component_labels",
        "source_is_target",
        "truth_positive",
        "classification",
        "eval_circle_hit",
        "realized_obb_hit",
        "realized_center_current_obb_hit",
        "cv_center_realized_shape_hit",
        "m2_continuous_cv_current_obb_hit",
        "eval_circle_minimum_clearance_m",
        "realized_obb_minimum_clearance_m",
        "realized_center_current_obb_minimum_clearance_m",
        "cv_center_realized_shape_minimum_clearance_m",
        "maximum_center_residual_m",
    ]
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                **row,
                "source_component_labels": ",".join(row["source_component_labels"]),
                "eval_circle_hit": row["arms"]["eval_circle"]["hit"],
                "realized_obb_hit": row["arms"]["realized_obb"]["hit"],
                "realized_center_current_obb_hit": row["arms"][
                    "realized_center_current_obb"
                ]["hit"],
                "cv_center_realized_shape_hit": row["arms"][
                    "cv_center_realized_shape"
                ]["hit"],
                "m2_continuous_cv_current_obb_hit": row[
                    "m2_continuous_cv_current_obb"
                ]["hit"],
                "eval_circle_minimum_clearance_m": row["arms"]["eval_circle"][
                    "minimum_clearance_m"
                ],
                "realized_obb_minimum_clearance_m": row["arms"]["realized_obb"][
                    "minimum_clearance_m"
                ],
                "realized_center_current_obb_minimum_clearance_m": row["arms"][
                    "realized_center_current_obb"
                ]["minimum_clearance_m"],
                "cv_center_realized_shape_minimum_clearance_m": row["arms"][
                    "cv_center_realized_shape"
                ]["minimum_clearance_m"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    os.replace(partial, path)


def _matrix_svg(rows: Sequence[dict[str, Any]]) -> str:
    arms = [
        ("eval_circle", "EVAL-CIRCLE"),
        ("realized_obb", "REALIZED-OBB"),
        ("realized_center_current_obb", "REALIZED-CENTER + CURRENT-OBB"),
        ("cv_center_realized_shape", "CV-CENTER + REALIZED-SHAPE"),
    ]
    width = 1180
    row_height = 58
    height = 130 + row_height * len(rows)
    left = 330
    cell_width = 190
    values = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="sans-serif" font-size="21" font-weight="700">DTR-M3-D realized-future contract decomposition</text>',
        '<text x="24" y="59" font-family="sans-serif" font-size="13" fill="#4b5563">Green = hit · Gray = miss · labels are privileged read-only future geometry</text>',
    ]
    for index, (_key, label) in enumerate(arms):
        values.append(
            f'<text x="{left + index * cell_width + cell_width / 2:.1f}" y="93" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700">{html.escape(label)}</text>'
        )
    for row_index, row in enumerate(rows):
        y = 108 + row_index * row_height
        values.append(
            f'<text x="24" y="{y + 21}" font-family="sans-serif" font-size="13" font-weight="700">{html.escape(row["id"])}</text>'
        )
        values.append(
            f'<text x="24" y="{y + 42}" font-family="sans-serif" font-size="11" fill="#4b5563">{html.escape(row["classification"])}</text>'
        )
        for arm_index, (key, _label) in enumerate(arms):
            hit = bool(row["arms"][key]["hit"])
            color = "#16a34a" if hit else "#d1d5db"
            text_color = "#ffffff" if hit else "#374151"
            x = left + arm_index * cell_width + 12
            values.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 24}" height="42" rx="7" fill="{color}"/>'
            )
            clearance = row["arms"][key]["minimum_clearance_m"]
            label = "HIT" if hit else "MISS"
            if clearance is not None:
                label += f"  {float(clearance):+.3f} m"
            values.append(
                f'<text x="{x + (cell_width - 24) / 2:.1f}" y="{y + 26}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="{text_color}">{html.escape(label)}</text>'
            )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(values)
        + "</svg>"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    m2_path = args.m2_result.resolve(strict=True)
    output_path = args.output.resolve()
    m2 = json.loads(m2_path.read_text(encoding="utf-8"))
    require(
        m2.get("status") == "DTR_M2_D_READ_ONLY_EXTENT_GAP_AUDIT_COMPLETE",
        "m2_status_drift",
    )
    require(
        m2.get("decision") == "DTR_M2_D_EXTENT_GAP_NOT_SUPPORTED_NO_FRESH_M2_O",
        "m2_terminal_drift",
    )
    m1_path = Path(m2["source"]["m1_result"]).resolve(strict=True)
    require(sha256_file(m1_path) == m2["source"]["m1_result_sha256"], "m2_m1_hash_drift")
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    source = m1["source"]
    known_tracks_path = Path(source["known_height_tracks"]).resolve(strict=True)
    labels_path = Path(source["labels"]).resolve(strict=True)
    timestamps_path = Path(source["timestamps"]).resolve(strict=True)
    bag_path = Path(source["bag"]).resolve(strict=True)
    require(sha256_file(known_tracks_path) == source["known_height_tracks_sha256"], "known_tracks_hash_drift")
    require(sha256_file(labels_path) == source["labels_sha256"], "labels_hash_drift")
    require(sha256_file(timestamps_path) == source["timestamps_sha256"], "timestamps_hash_drift")
    require(sha256_file(bag_path) == source["bag_sha256"], "bag_hash_drift")

    timestamps = load_image_timestamps(timestamps_path)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    causal_frame_poses = {
        frame: _causal_pose(poses, round(timestamps[frame] * 1e9))
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    boxes_by_frame = load_native_boxes(labels_path, timestamps, causal_frame_poses)
    context = {
        frame: {
            "image_time_s": timestamps[frame],
            "pose": interpolate_pose(poses, round(timestamps[frame] * 1e9)),
        }
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    tracks, geometry_quality = load_truth_and_associate(
        labels_path, read_jsonl(known_tracks_path), context
    )
    cases = cases_from_tracks(tracks)
    cases_by_label: dict[str, list[SegmentCase]] = {}
    for case in cases:
        cases_by_label.setdefault(case.label_id, []).append(case)

    dropout_rows = [
        _decompose_m2_row(
            row=row,
            cases_by_label=cases_by_label,
            boxes_by_frame=boxes_by_frame,
        )
        for row in m2["dropout_miss_trials"]
    ]
    false_rows = [
        _decompose_m2_row(
            row=row,
            cases_by_label=cases_by_label,
            boxes_by_frame=boxes_by_frame,
        )
        for row in m2["m1_new_or_modified_false_segments"]
    ]
    rows = dropout_rows + false_rows
    dropout_counts = Counter(row["classification"] for row in dropout_rows)
    false_counts = Counter(row["classification"] for row in false_rows)
    unique_dropout_labels = sorted({row["target_label_id"] for row in dropout_rows})
    truth_geometry_mismatch = (
        len(dropout_rows) == 3
        and all(
            row["classification"] == "EVAL_CIRCLE_HIT_REALIZED_OBB_MISS"
            for row in dropout_rows
        )
    )
    dynamics_headroom = any(
        row["arms"]["realized_obb"]["hit"]
        and not row["m2_continuous_cv_current_obb"]["hit"]
        for row in dropout_rows
    )
    fresh_m3_o_eligible = dynamics_headroom and not truth_geometry_mismatch
    if truth_geometry_mismatch:
        decision = "DTR_M3_D_EVALUATOR_CIRCLE_OBB_SEMANTICS_MISMATCH_NO_FRESH_M3_O"
    elif fresh_m3_o_eligible:
        decision = "DTR_M3_D_REALIZED_DYNAMICS_HEADROOM_FREEZE_FRESH_M3_O_PROTOCOL"
    else:
        decision = "DTR_M3_D_NO_REALIZED_DYNAMICS_HEADROOM_STOP_FUTURE_OCCUPANCY_ROUTE"

    csv_path = output_path.with_name(output_path.stem + ".contract-decomposition.csv")
    svg_path = output_path.with_name(output_path.stem + ".contract-decomposition.svg")
    _write_csv(csv_path, rows)
    _atomic_text(svg_path, _matrix_svg(rows))
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_ceiling": CLAIM_CEILING,
        "question": (
            "Does the M2 residual come from evaluator circularization, realized center dynamics, "
            "realized shape/yaw dynamics, or a genuine M1 future false positive?"
        ),
        "frozen": {
            "predictions_thresholds_lifecycle_events_and_gate": "READ_ONLY_UNCHANGED",
            "diagnostic_origins": "exact three M2 dropout-miss trials plus five M2 false-segment rows",
            "route_body_radius_m": ROUTE_HALF_WIDTH_M,
            "horizon_s": HORIZON_S,
            "future_sampling": "realized discrete native samples exactly matching evaluator temporal support",
        },
        "contracts": {
            "eval_circle": "realized future center plus max(0.15, 0.5*max(width,length)) evaluator radius",
            "realized_obb": "realized future center/yaw/length/width versus 0.65 m route disk",
            "realized_center_current_obb": "realized future center with origin-frame yaw/length/width fixed",
            "cv_center_realized_shape": "M1 constant-velocity center with realized future yaw/length/width",
            "cv_center_current_obb_discrete_control": "M1 constant-velocity center with current OBB at evaluator sample times",
            "m2_continuous_control": "exact M2 current-OBB continuous sweep result",
        },
        "source": {
            "dataset": "JRDB public train split",
            "sequence": m1["source"]["sequence"],
            "window": m1["source"]["window"],
            "m2_result": str(m2_path),
            "m2_result_sha256": sha256_file(m2_path),
            "m1_result": str(m1_path),
            "m1_result_sha256": sha256_file(m1_path),
            "known_height_tracks": str(known_tracks_path),
            "known_height_tracks_sha256": sha256_file(known_tracks_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "bag_authority": bag_authority,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "evaluator_firewall": {
            "future_native_labels": "diagnostic-only and never enter a prediction",
            "consumed_cohort": True,
            "geometry_quality": geometry_quality,
        },
        "dropout_miss_trials": dropout_rows,
        "m1_new_or_modified_false_segments": false_rows,
        "summary": {
            "dropout_miss_trials": len(dropout_rows),
            "dropout_miss_unique_targets": unique_dropout_labels,
            "dropout_classification_counts": dict(sorted(dropout_counts.items())),
            "false_segments": len(false_rows),
            "false_classification_counts": dict(sorted(false_counts.items())),
            "truth_geometry_mismatch": truth_geometry_mismatch,
            "realized_dynamics_headroom": dynamics_headroom,
            "fresh_m3_o_eligible": fresh_m3_o_eligible,
            "forecasting_opened": False,
            "r8_closed": True,
            "scene_flow_estimator_competition_closed": True,
        },
        "decision": decision,
        "artifacts": {
            "rows_csv": str(csv_path),
            "rows_csv_sha256": sha256_file(csv_path),
            "matrix_svg": str(svg_path),
            "matrix_svg_sha256": sha256_file(svg_path),
        },
        "limitations": [
            "This is post-outcome scorer-side diagnosis on the consumed M1/M2 Development cohort.",
            "The three dropout rows are three durations of one pedestrian:35 event.",
            "Future boxes, centers, dimensions, yaw, and identity are privileged labels, not causal inputs.",
            "The four arms use evaluator-discrete future samples; M2 continuous CV-current-OBB is retained as a separate control.",
            "A circle/OBB mismatch is a semantics mismatch only: whether it is an evaluator bug depends on the intended event definition.",
            "A dynamics signal can authorize freezing a fresh oracle protocol but cannot itself open learned forecasting or R8.",
            "No source-disjoint, Android, product, user-benefit, or safety claim follows.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    print(json.dumps({"decision": result["decision"], **result["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
