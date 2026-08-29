"""X0 scorer-side attribution of the frozen C31 fresh-confirmation errors.

X0 does not change or rescore any prediction.  It opens native JRDB identity,
trajectory, and OBB geometry only after the PDC/C31 predictions are sealed and
uses them to diagnose two missed CONTACT events, all 25 PDC false segments,
and the ten C31 false segments that do not overlap a PDC false segment.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import dtr_c31_temporal_component_authority as c31
from coda_static_ceiling import point_to_box_clearance
from dtr_c1_global_obb_cohort_admission import _load_timestamps
from dtr_m1_point_velocity_oracle import _box_history, load_native_boxes
from dtr_r7_occupancy_flow_canary import (
    FROZEN_FLOW_CONFIG,
    HORIZON_S,
    ROUTE_HALF_WIDTH_M,
    _causal_pose,
    _entry_s,
    _rotate_world_velocity_to_ego,
)
from jrdb_rgb_bridge import read_bag_pose_and_rgb, require, sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]
SCHEMA = "blindassist-dtr-x0-motion-source-attribution-v1"
STATUS = "DTR_X0_MOTION_SOURCE_ATTRIBUTION_COMPLETE"

NO_MOTION_SUPPORT = "NO_MOTION_SUPPORT"
BAD_FLOW = "BAD_FLOW"
STATIC_PSEUDO_MOTION = "STATIC_PSEUDO_MOTION"
REAL_MOVER_NONCRITICAL = "REAL_MOVER_NONCRITICAL"
FRAGMENTATION = "FRAGMENTATION"
WRONG_COMPONENT_BINDING = "WRONG_COMPONENT_BINDING"
ROUTE_GEOMETRY_MISS = "ROUTE_GEOMETRY_MISS"
LATE_SUPPORT = "LATE_SUPPORT"
NOT_EVALUABLE = "NOT_EVALUABLE"

SOURCE_FAILURES = {NO_MOTION_SUPPORT, BAD_FLOW, STATIC_PSEUDO_MOTION}
GEOMETRY_FAILURES = {ROUTE_GEOMETRY_MISS}
STRUCTURAL_FAILURES = {
    REAL_MOVER_NONCRITICAL,
    FRAGMENTATION,
    WRONG_COMPONENT_BINDING,
    LATE_SUPPORT,
}
FLOW_ERROR_LIMIT_MPS = FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps
ASSOCIATION_MARGIN_M = (
    FROZEN_FLOW_CONFIG.association_margin_cells * FROZEN_FLOW_CONFIG.voxel_size_m
)
EARLY_LEAD_S = HORIZON_S * 0.50
MINIMUM_CORRECT_SUPPORT_FRAMES = 2


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    return left == right


def _frame_range(row: Mapping[str, Any]) -> range:
    return range(int(row["first_frame"]), int(row["last_frame"]) + 1)


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return int(left["first_frame"]) <= int(right["last_frame"]) and int(
        right["first_frame"]
    ) <= int(left["last_frame"])


def c31_incremental_ranges(
    pdc_ranges: Sequence[Mapping[str, Any]],
    c31_ranges: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return C31 false ranges with no overlap with a PDC false range."""

    return [
        row for row in c31_ranges if not any(_overlap(row, base) for base in pdc_ranges)
    ]


def choose_primary_cause(
    cell_counts: Mapping[str, int],
    *,
    wrong_binding: bool = False,
    fragmentation: bool = False,
) -> str:
    if wrong_binding:
        return WRONG_COMPONENT_BINDING
    if fragmentation:
        return FRAGMENTATION
    if not cell_counts:
        return NOT_EVALUABLE
    priority = {
        BAD_FLOW: 0,
        STATIC_PSEUDO_MOTION: 1,
        REAL_MOVER_NONCRITICAL: 2,
        ROUTE_GEOMETRY_MISS: 3,
        NOT_EVALUABLE: 4,
    }
    return min(
        cell_counts,
        key=lambda name: (-int(cell_counts[name]), priority.get(name, 99), name),
    )


def choose_route(
    miss_counts: Mapping[str, int], false_counts: Mapping[str, int]
) -> tuple[str, str]:
    false_total = sum(int(value) for value in false_counts.values())
    false_source = sum(int(false_counts.get(name, 0)) for name in SOURCE_FAILURES)
    if any(int(miss_counts.get(name, 0)) > 0 for name in SOURCE_FAILURES):
        return (
            "STRONGER_SCENE_FLOW_SOURCE",
            "At least one missed CONTACT lacks correct raw motion or is dominated by bad flow.",
        )
    if false_total and false_source / false_total >= 0.50:
        return (
            "STRONGER_SCENE_FLOW_SOURCE",
            "At least half of the diagnosed false segments are source-motion failures.",
        )
    if any(int(miss_counts.get(name, 0)) > 0 for name in GEOMETRY_FAILURES):
        return (
            "CONTINUOUS_COLLISION_GEOMETRY",
            "Correct early motion exists, while the frozen discrete route geometry misses CONTACT.",
        )
    return (
        "LEARNED_MOTION_AUTHORITY",
        "The raw source is usable; selection, binding, relevance, or timing dominates.",
    )


def _load_ledger(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    path = path.resolve(strict=True)
    require(sha256_file(path) == expected_sha256, f"ledger_hash_drift:{path}")
    with np.load(path) as source:
        required = {
            "frames",
            "offsets",
            "forward_m",
            "left_m",
            "velocity_forward_mps",
            "velocity_left_mps",
            "component_id",
        }
        require(required <= set(source.files), f"ledger_schema:{path}")
        return {name: source[name].copy() for name in source.files}


def _cells(ledger: Mapping[str, np.ndarray], frame: int) -> list[dict[str, Any]]:
    positions = np.nonzero(ledger["frames"] == frame)[0]
    if len(positions) != 1:
        return []
    index = int(positions[0])
    start, stop = int(ledger["offsets"][index]), int(ledger["offsets"][index + 1])
    return [
        {
            "forward_m": float(ledger["forward_m"][cell]),
            "left_m": float(ledger["left_m"][cell]),
            "velocity_forward_mps": float(ledger["velocity_forward_mps"][cell]),
            "velocity_left_mps": float(ledger["velocity_left_mps"][cell]),
            "component_id": int(ledger["component_id"][cell]),
        }
        for cell in range(start, stop)
    ]


def _cell_clearance(cell: Mapping[str, Any], box: Any) -> float:
    return point_to_box_clearance(
        float(cell["forward_m"]),
        float(cell["left_m"]),
        float(box.center_forward_m),
        float(box.center_left_m),
        float(box.yaw_ego_rad),
        float(box.length_m),
        float(box.width_m),
    )


def _target_velocity(
    box: Any, history: Mapping[tuple[int, str], Any], pose: Mapping[str, Any]
) -> tuple[float, float] | None:
    previous = history.get((int(box.frame), str(box.label_id)))
    if previous is None:
        return None
    delta_s = float(box.time_s) - float(previous.time_s)
    if delta_s <= 0.0:
        return None
    current_xy = np.asarray(box.center_world_xy, dtype=np.float64)
    previous_xy = np.asarray(previous.center_world_xy, dtype=np.float64)
    world_velocity = ((current_xy - previous_xy) / delta_s).reshape(1, 2)
    local = _rotate_world_velocity_to_ego(world_velocity, dict(pose))[0]
    return float(local[0]), float(local[1])


def _realized_entry(
    *,
    label_id: str,
    origin_frame: int,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    boxes_by_frame: Mapping[int, Sequence[Any]],
) -> float | None:
    index = frames.index(origin_frame)
    origin_time = float(timestamps[origin_frame])
    times = [float(timestamps[frame]) for frame in frames]
    final = bisect.bisect_right(times, origin_time + HORIZON_S + 1e-9) - 1
    for future_index in range(index, final + 1):
        frame = frames[future_index]
        for box in boxes_by_frame.get(frame, ()):
            if str(box.label_id) != label_id:
                continue
            clearance = (
                point_to_box_clearance(
                    0.0,
                    0.0,
                    float(box.center_forward_m),
                    float(box.center_left_m),
                    float(box.yaw_ego_rad),
                    float(box.length_m),
                    float(box.width_m),
                )
                - ROUTE_HALF_WIDTH_M
            )
            if clearance <= 1e-9:
                return float(timestamps[frame]) - origin_time
    return None


def _diagnose_cell(
    cell: Mapping[str, Any],
    *,
    frame: int,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    boxes_by_frame: Mapping[int, Sequence[Any]],
    history: Mapping[tuple[int, str], Any],
    poses: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    matches = [
        box
        for box in boxes_by_frame.get(frame, ())
        if _cell_clearance(cell, box) <= ASSOCIATION_MARGIN_M + 1e-9
    ]
    output = {
        **cell,
        "entry_s": _entry_s(
            float(cell["forward_m"]),
            float(cell["left_m"]),
            float(cell["velocity_forward_mps"]),
            float(cell["velocity_left_mps"]),
        ),
        "matched_labels": sorted({str(box.label_id) for box in matches}),
    }
    if not matches:
        output.update(
            primary_cause=STATIC_PSEUDO_MOTION,
            evidence="no current native mover OBB supports the route-risk cell",
        )
        return output
    if len(matches) != 1:
        output.update(
            primary_cause=WRONG_COMPONENT_BINDING,
            evidence="route-risk cell is compatible with multiple current native OBBs",
        )
        return output
    box = matches[0]
    target_velocity = _target_velocity(box, history, poses[frame])
    if target_velocity is None:
        output.update(
            primary_cause=NOT_EVALUABLE, evidence="no admissible causal OBB history"
        )
        return output
    flow = np.asarray(
        (cell["velocity_forward_mps"], cell["velocity_left_mps"]), dtype=np.float64
    )
    target = np.asarray(target_velocity, dtype=np.float64)
    target_speed = float(np.linalg.norm(target))
    error = float(np.linalg.norm(flow - target))
    output.update(
        target_velocity_forward_mps=float(target[0]),
        target_velocity_left_mps=float(target[1]),
        target_speed_mps=target_speed,
        flow_error_mps=error,
    )
    if target_speed < FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps - 1e-12:
        output.update(
            primary_cause=STATIC_PSEUDO_MOTION,
            evidence="matched native object is below the frozen dynamic-speed floor",
        )
    elif error > FLOW_ERROR_LIMIT_MPS + 1e-12:
        output.update(
            primary_cause=BAD_FLOW,
            evidence="flow differs from same-history native velocity beyond the frozen motion floor",
        )
    elif (
        _realized_entry(
            label_id=str(box.label_id),
            origin_frame=frame,
            frames=frames,
            timestamps=timestamps,
            boxes_by_frame=boxes_by_frame,
        )
        is None
    ):
        output.update(
            primary_cause=REAL_MOVER_NONCRITICAL,
            evidence="motion agrees but the native mover does not enter the route in 3 s",
        )
    else:
        output.update(
            primary_cause=ROUTE_GEOMETRY_MISS,
            evidence="motion and realized route entry agree; discrete support geometry remains suspect",
        )
    return output


def _structure_flags(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, list[dict[str, Any]]]:
    if not cells:
        return False, False, []
    rows = [
        {
            "position": (row["forward_m"], row["left_m"]),
            "velocity": (row["velocity_forward_mps"], row["velocity_left_mps"]),
            "q": 1.0,
        }
        for row in cells
    ]
    components = c31.TemporalComponentAuthority()._components(
        rows, list(range(len(rows)))
    )
    groups = []
    label_groups: dict[str, set[int]] = {}
    wrong_binding = False
    for group_index, component in enumerate(components):
        labels = sorted(
            {
                label
                for index in component.members
                for label in cells[index].get("matched_labels", ())
            }
        )
        unmatched = sum(
            not cells[index].get("matched_labels") for index in component.members
        )
        if len(labels) > 1 or (labels and unmatched):
            wrong_binding = True
        for label in labels:
            label_groups.setdefault(label, set()).add(group_index)
        groups.append(
            {
                "group": group_index,
                "members": len(component.members),
                "labels": labels,
                "unmatched_members": unmatched,
            }
        )
    fragmentation = any(len(values) > 1 for values in label_groups.values())
    return wrong_binding, fragmentation, groups


def _diagnose_false_segment(
    *,
    segment_id: str,
    segment: Mapping[str, Any],
    ledger: Mapping[str, np.ndarray],
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    boxes_by_frame: Mapping[int, Sequence[Any]],
    history: Mapping[tuple[int, str], Any],
    poses: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    diagnostic_frame = None
    diagnostics: list[dict[str, Any]] = []
    for frame in _frame_range(segment):
        risky = [
            cell
            for cell in _cells(ledger, frame)
            if _entry_s(
                cell["forward_m"],
                cell["left_m"],
                cell["velocity_forward_mps"],
                cell["velocity_left_mps"],
            )
            is not None
        ]
        if risky:
            diagnostic_frame = frame
            diagnostics = [
                _diagnose_cell(
                    cell,
                    frame=frame,
                    frames=frames,
                    timestamps=timestamps,
                    boxes_by_frame=boxes_by_frame,
                    history=history,
                    poses=poses,
                )
                for cell in risky
            ]
            break
    counts = Counter(str(row["primary_cause"]) for row in diagnostics)
    wrong, fragmented, groups = _structure_flags(diagnostics)
    primary = choose_primary_cause(
        counts, wrong_binding=wrong, fragmentation=(not wrong and fragmented)
    )
    return {
        "segment_id": segment_id,
        "first_frame": int(segment["first_frame"]),
        "last_frame": int(segment["last_frame"]),
        "diagnostic_frame": diagnostic_frame,
        "primary_cause": primary,
        "cell_cause_counts": dict(sorted(counts.items())),
        "wrong_component_binding": wrong,
        "fragmentation": fragmented,
        "diagnostic_groups": groups,
        "diagnostic_cells": diagnostics,
    }


def _near_cell(
    target: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> bool:
    target_position = np.asarray(
        (target["forward_m"], target["left_m"]), dtype=np.float64
    )
    target_velocity = np.asarray(
        (target["velocity_forward_mps"], target["velocity_left_mps"]), dtype=np.float64
    )
    for row in candidates:
        position = np.asarray((row["forward_m"], row["left_m"]), dtype=np.float64)
        velocity = np.asarray(
            (row["velocity_forward_mps"], row["velocity_left_mps"]), dtype=np.float64
        )
        if (
            float(np.linalg.norm(position - target_position))
            <= ASSOCIATION_MARGIN_M + 1e-9
            and float(np.linalg.norm(velocity - target_velocity))
            <= FLOW_ERROR_LIMIT_MPS + 1e-9
        ):
            return True
    return False


def _diagnose_miss(
    *,
    event: Mapping[str, Any],
    raw_ledger: Mapping[str, np.ndarray],
    c31_ledger: Mapping[str, np.ndarray],
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    boxes_by_frame: Mapping[int, Sequence[Any]],
    history: Mapping[tuple[int, str], Any],
    poses: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    label_id = str(event["responsible_components"][0])
    contact_time = float(event["first_time_s"]) + float(
        event["onset_first_hit_delta_s"]
    )
    window_frames = [
        frame
        for frame in frames
        if contact_time - HORIZON_S - 1e-9
        <= float(timestamps[frame])
        <= contact_time + 1e-9
    ]
    associated: list[dict[str, Any]] = []
    correct: list[dict[str, Any]] = []
    correct_route: list[dict[str, Any]] = []
    accepted_frames: set[int] = set()
    for frame in window_frames:
        boxes = [
            box
            for box in boxes_by_frame.get(frame, ())
            if str(box.label_id) == label_id
        ]
        if len(boxes) != 1:
            continue
        box = boxes[0]
        target_velocity = _target_velocity(box, history, poses[frame])
        if target_velocity is None:
            continue
        target = np.asarray(target_velocity, dtype=np.float64)
        target_speed = float(np.linalg.norm(target))
        c31_cells = _cells(c31_ledger, frame)
        for cell in _cells(raw_ledger, frame):
            if _cell_clearance(cell, box) > ASSOCIATION_MARGIN_M + 1e-9:
                continue
            flow = np.asarray(
                (cell["velocity_forward_mps"], cell["velocity_left_mps"]),
                dtype=np.float64,
            )
            error = float(np.linalg.norm(flow - target))
            lead = contact_time - float(timestamps[frame])
            entry = _entry_s(
                cell["forward_m"],
                cell["left_m"],
                cell["velocity_forward_mps"],
                cell["velocity_left_mps"],
            )
            row = {
                "frame": frame,
                "lead_to_contact_s": lead,
                "forward_m": cell["forward_m"],
                "left_m": cell["left_m"],
                "velocity_forward_mps": cell["velocity_forward_mps"],
                "velocity_left_mps": cell["velocity_left_mps"],
                "target_speed_mps": target_speed,
                "flow_error_mps": error,
                "entry_s": entry,
            }
            associated.append(row)
            if error <= FLOW_ERROR_LIMIT_MPS + 1e-12:
                correct.append(row)
                if entry is not None:
                    correct_route.append(row)
                if _near_cell(row, c31_cells):
                    accepted_frames.add(frame)

    correct_frames = sorted({int(row["frame"]) for row in correct})
    route_frames = sorted({int(row["frame"]) for row in correct_route})
    early_frames = sorted(
        {
            int(row["frame"])
            for row in correct
            if float(row["lead_to_contact_s"]) >= EARLY_LEAD_S - 1e-9
        }
    )
    if not associated:
        primary = NO_MOTION_SUPPORT
    elif not correct:
        primary = BAD_FLOW
    elif len(correct_frames) < MINIMUM_CORRECT_SUPPORT_FRAMES or not early_frames:
        primary = LATE_SUPPORT
    elif not route_frames:
        primary = ROUTE_GEOMETRY_MISS
    elif len(accepted_frames) < MINIMUM_CORRECT_SUPPORT_FRAMES:
        primary = FRAGMENTATION
    else:
        primary = WRONG_COMPONENT_BINDING
    return {
        "event_id": str(event["event_id"]),
        "responsible_component": label_id,
        "contact_time_s": contact_time,
        "window_first_frame": min(window_frames) if window_frames else None,
        "window_last_frame": max(window_frames) if window_frames else None,
        "primary_cause": primary,
        "associated_raw_cells": len(associated),
        "associated_raw_frames": len({row["frame"] for row in associated}),
        "correct_raw_cells": len(correct),
        "correct_raw_frames": len(correct_frames),
        "early_correct_raw_frames": len(early_frames),
        "correct_route_entry_frames": len(route_frames),
        "c31_accepted_correct_frames": len(accepted_frames),
        "sufficient_early_correct_raw_motion": (
            len(correct_frames) >= MINIMUM_CORRECT_SUPPORT_FRAMES and bool(early_frames)
        ),
        "correct_raw_motion_can_enter_frozen_route": bool(route_frames),
        "first_correct_lead_s": max(
            (float(row["lead_to_contact_s"]) for row in correct), default=None
        ),
        "minimum_correct_flow_error_mps": min(
            (float(row["flow_error_mps"]) for row in correct), default=None
        ),
        "minimum_associated_flow_error_mps": min(
            (float(row["flow_error_mps"]) for row in associated), default=None
        ),
        "maximum_associated_target_speed_mps": max(
            (float(row["target_speed_mps"]) for row in associated), default=None
        ),
    }


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "unit_type",
        "unit_id",
        "sequence",
        "first_frame",
        "last_frame",
        "primary_cause",
        "diagnostic_frame",
        "detail",
    )
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_path = args.c31_result.resolve(strict=True)
    predictions_path = args.c31_predictions.resolve(strict=True)
    baseline_path = args.baseline_predictions.resolve(strict=True)
    roster_path = args.roster.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    require(
        result.get("terminal_status")
        == "DTR_C31_SOURCE_DISJOINT_CONFIRMATION_GATE_NOT_MET",
        "c31_terminal_drift",
    )
    require(predictions.get("truth_blind") is True, "c31_predictions_not_sealed")
    require(baseline.get("truth_blind") is True, "baseline_predictions_not_sealed")
    require(
        roster["source_authority"]["labels_sha256"] == sha256_file(labels_path),
        "labels_hash",
    )
    require(
        roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path),
        "timestamps_hash",
    )
    require(
        result["source"]["sealed_predictions_sha256"] == sha256_file(predictions_path),
        "c31_prediction_hash",
    )
    require(
        result["source"]["baseline_predictions_sha256"] == sha256_file(baseline_path),
        "baseline_prediction_hash",
    )

    score_rows = {str(row["sequence"]): row for row in result["per_sequence"]}
    prediction_rows = {str(row["sequence"]): row for row in predictions["sequences"]}
    baseline_rows = {str(row["sequence"]): row for row in baseline["sequences"]}
    roster_rows = {str(row["sequence"]): row for row in roster["selected_sequences"]}
    require(
        set(score_rows)
        == set(prediction_rows)
        == set(baseline_rows)
        == set(roster_rows),
        "sequence_coverage",
    )

    misses = []
    pdc_false = []
    c31_incremental = []
    with zipfile.ZipFile(timestamps_path) as timestamp_bundle:
        for sequence in sorted(score_rows):
            timestamps = _load_timestamps(timestamp_bundle, sequence)
            frames = sorted(timestamps)
            base_row = baseline_rows[sequence]
            c31_row = prediction_rows[sequence]
            bag_path = Path(base_row["sources"]["bag"]).resolve(strict=True)
            require(
                sha256_file(bag_path) == base_row["sources"]["bag_sha256"],
                f"bag_hash:{sequence}",
            )
            pose_samples, _rgb, _authority = read_bag_pose_and_rgb(bag_path)
            poses = {
                frame: _causal_pose(pose_samples, round(float(timestamps[frame]) * 1e9))
                for frame in frames
            }
            boxes = load_native_boxes(labels_path, timestamps, poses, sequence=sequence)
            history = _box_history(boxes)
            pdc_source = base_row["sources"]["ledgers"]["M1_PDC_GLOBAL"]
            raw_source = base_row["sources"]["ledgers"]["M1_PD_GLOBAL"]
            c31_source = c31_row["dropout_ledgers"]["C31_SIGNED_TRANSPORT"]
            pdc_ledger = _load_ledger(
                Path(pdc_source["ledger"]), pdc_source["ledger_sha256"]
            )
            raw_ledger = _load_ledger(
                Path(raw_source["ledger"]), raw_source["ledger_sha256"]
            )
            c31_ledger = _load_ledger(
                Path(c31_source["ledger"]), c31_source["ledger_sha256"]
            )
            scores = score_rows[sequence]["scores"]
            pdc_ranges = list(scores["M1_PDC_GLOBAL"]["false_alert_ranges"])
            c31_ranges = list(scores["C31_SIGNED_TRANSPORT"]["false_alert_ranges"])
            for index, segment in enumerate(pdc_ranges, start=1):
                pdc_false.append(
                    {
                        "sequence": sequence,
                        **_diagnose_false_segment(
                            segment_id=f"{sequence}:pdc-false:{index:03d}",
                            segment=segment,
                            ledger=pdc_ledger,
                            frames=frames,
                            timestamps=timestamps,
                            boxes_by_frame=boxes,
                            history=history,
                            poses=poses,
                        ),
                    }
                )
            incremental = c31_incremental_ranges(pdc_ranges, c31_ranges)
            for index, segment in enumerate(incremental, start=1):
                c31_incremental.append(
                    {
                        "sequence": sequence,
                        **_diagnose_false_segment(
                            segment_id=f"{sequence}:c31-incremental-false:{index:03d}",
                            segment=segment,
                            ledger=c31_ledger,
                            frames=frames,
                            timestamps=timestamps,
                            boxes_by_frame=boxes,
                            history=history,
                            poses=poses,
                        ),
                    }
                )
            missed_ids = {
                str(row["event_id"])
                for row in scores["C31_SIGNED_TRANSPORT"]["event_rows"]
                if not bool(row["recalled"])
            }
            for event in roster_rows[sequence]["bounded_contact_event_details"]:
                if str(event["event_id"]) in missed_ids:
                    misses.append(
                        {
                            "sequence": sequence,
                            **_diagnose_miss(
                                event=event,
                                raw_ledger=raw_ledger,
                                c31_ledger=c31_ledger,
                                frames=frames,
                                timestamps=timestamps,
                                boxes_by_frame=boxes,
                                history=history,
                                poses=poses,
                            ),
                        }
                    )

    require(len(misses) == 2, "miss_count_drift")
    require(len(pdc_false) == 25, "pdc_false_count_drift")
    require(len(c31_incremental) == 10, "c31_incremental_count_drift")
    miss_counts = dict(sorted(Counter(row["primary_cause"] for row in misses).items()))
    pdc_counts = dict(
        sorted(Counter(row["primary_cause"] for row in pdc_false).items())
    )
    incremental_counts = dict(
        sorted(Counter(row["primary_cause"] for row in c31_incremental).items())
    )
    false_counts = dict(
        sorted((Counter(pdc_counts) + Counter(incremental_counts)).items())
    )
    route, reason = choose_route(miss_counts, false_counts)

    table_rows = []
    for row in misses:
        table_rows.append(
            {
                "unit_type": "MISS",
                "unit_id": row["event_id"],
                "sequence": row["sequence"],
                "first_frame": row["window_first_frame"],
                "last_frame": row["window_last_frame"],
                "primary_cause": row["primary_cause"],
                "diagnostic_frame": "",
                "detail": json.dumps(
                    {
                        "associated_raw_frames": row["associated_raw_frames"],
                        "correct_raw_frames": row["correct_raw_frames"],
                        "early_correct_raw_frames": row["early_correct_raw_frames"],
                        "correct_route_entry_frames": row["correct_route_entry_frames"],
                    },
                    sort_keys=True,
                ),
            }
        )
    for unit_type, rows in (
        ("PDC_FALSE", pdc_false),
        ("C31_INCREMENTAL_FALSE", c31_incremental),
    ):
        for row in rows:
            table_rows.append(
                {
                    "unit_type": unit_type,
                    "unit_id": row["segment_id"],
                    "sequence": row["sequence"],
                    "first_frame": row["first_frame"],
                    "last_frame": row["last_frame"],
                    "primary_cause": row["primary_cause"],
                    "diagnostic_frame": row["diagnostic_frame"],
                    "detail": json.dumps(row["cell_cause_counts"], sort_keys=True),
                }
            )
    _write_table(args.table.resolve(), table_rows)
    output = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": "Does correct raw motion exist before the two missed CONTACTs, and are false segments dominated by irrelevant movers, authority structure, or bad source flow?",
        "frozen_contract": {
            "association_margin_m": ASSOCIATION_MARGIN_M,
            "flow_correct_error_limit_mps": FLOW_ERROR_LIMIT_MPS,
            "dynamic_speed_floor_mps": FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps,
            "window_s": HORIZON_S,
            "early_lead_s": EARLY_LEAD_S,
            "minimum_correct_support_frames": MINIMUM_CORRECT_SUPPORT_FRAMES,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "risk_scorer_lifecycle_or_prediction_changes": False,
        },
        "summary": {
            "misses": len(misses),
            "pdc_false_segments": len(pdc_false),
            "c31_incremental_false_segments": len(c31_incremental),
            "miss_primary_causes": miss_counts,
            "pdc_false_primary_causes": pdc_counts,
            "c31_incremental_false_primary_causes": incremental_counts,
            "combined_false_primary_causes": false_counts,
        },
        "decision": {
            "next_branch": route,
            "reason": reason,
            "do_not_start_model_from_x0": True,
            "c31_tuning_closed": True,
            "c32_probabilistic_tube_closed": True,
        },
        "misses": misses,
        "pdc_false_segments": pdc_false,
        "c31_incremental_false_segments": c31_incremental,
        "definitions": {
            NO_MOTION_SUPPORT: "No raw M1-PD cell is associated with the responsible OBB in the -3..0 s window.",
            BAD_FLOW: "Raw motion exists, but no associated cell agrees with same-history native velocity within 0.25 m/s.",
            STATIC_PSEUDO_MOTION: "A risk cell has no current native mover support or is attached to an object below the frozen 0.25 m/s dynamic floor.",
            REAL_MOVER_NONCRITICAL: "Flow agrees with a moving native OBB whose realized 3 s future does not enter the body route.",
            FRAGMENTATION: "The same native identity appears in multiple C31-compatible local groups, or correct raw miss support is not retained for two C31 frames.",
            WRONG_COMPONENT_BINDING: "A C31-compatible group mixes native identities/unmatched support, or accepted correct miss support still fails to form the right risk binding.",
            ROUTE_GEOMETRY_MISS: "Correct early motion exists but the frozen point/route geometry produces no route entry.",
            LATE_SUPPORT: "Correct motion appears in fewer than two frames or only after the frozen 1.5 s urgent boundary.",
            NOT_EVALUABLE: "The causal native history or a raw risk cell needed for attribution is absent.",
        },
        "source": {
            "c31_result": str(result_path),
            "c31_result_sha256": sha256_file(result_path),
            "c31_predictions": str(predictions_path),
            "c31_predictions_sha256": sha256_file(predictions_path),
            "baseline_predictions": str(baseline_path),
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "outputs": {"table_csv": str(args.table.resolve())},
        "claim_limits": [
            "Post-outcome scorer-side attribution on already opened source-disjoint truth; not new performance evidence.",
            "Native OBB identity and trajectory are privileged diagnostic labels, not deployable inputs.",
            "No-OBB support is consistent with static pseudo-motion but may include unlabeled movers.",
            "Frame-local component grouping can establish a native-identity split/mix in that frame; it does not prove long-term track identity.",
            "UNKNOWN and NOT_EVALUABLE are not negative results.",
        ],
    }
    write_json(args.output.resolve(), output)
    return output


def parse_args() -> argparse.Namespace:
    evidence = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    source = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    output = (
        REPO / "artifacts.local" / "evidence" / "dtr-x0" / "motion-source-attribution"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c31-result", type=Path, default=evidence / "result.json")
    parser.add_argument(
        "--c31-predictions", type=Path, default=evidence / "predictions.json"
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=evidence / "baseline-predictions.json",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=REPO
        / "research"
        / "active"
        / "dtr-r0"
        / "dtr_c31_fresh_confirmation_roster.json",
    )
    parser.add_argument("--labels", type=Path, default=source / "train_labels.zip")
    parser.add_argument(
        "--timestamps", type=Path, default=source / "train_timestamps.zip"
    )
    parser.add_argument("--output", type=Path, default=output / "result.json")
    parser.add_argument("--table", type=Path, default=output / "attribution.csv")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "decision": result["decision"],
                "misses": result["misses"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
