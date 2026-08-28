"""Read-only DTR-M0 attribution of sealed R7-P false alert segments.

This diagnostic never changes the R7 flow ledger, matcher, lifecycle, gate, or
verdict.  It replays the sealed ledger against the already opened scorer-side
JRDB target trajectories, separates inherited R2 alerts from flow-caused alert
changes, and emits one segment table plus one timeline overlay.

Component identifiers in the R7 ledger are local to a frame.  Consequently,
this diagnostic may flag temporal component discontinuity, but it must not
claim that a particular split or merge has been proven.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dtr_r0 import CausalFrame, DTRConfig, Prediction, Vec2
from dtr_r1 import _first_tube_entry_s
from dtr_r2 import DTRR2Arm
from dtr_r5_dropout_canary import (
    ACTIVE_SIGNALS,
    SegmentCase,
    cases_from_tracks,
    metrics_for_run,
    sample_pose,
    sensor_observation,
)
from dtr_r7_occupancy_flow_canary import (
    FROZEN_FLOW_CONFIG,
    HORIZON_S,
    MINIMUM_CLOSING_SPEED_MPS,
    ROUTE_HALF_WIDTH_M,
    FlowLedger,
    _entry_s,
    load_flow_ledger,
    run_flow_arm,
)
from jrdb_native_ceiling import AlertSegment, ArmAccumulator, alert_segments
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    FIRST_FRAME,
    LAST_FRAME,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import (
    SensorSample,
    load_truth_and_associate,
    read_jsonl,
    write_json,
)


SCHEMA = "blindassist-dtr-m0-r7-error-attribution-v1"
TABLE_SCHEMA = "blindassist-dtr-m0-r7-false-segment-table-v1"
CLAIM_CEILING = "READ_ONLY_ATTRIBUTION_ON_CONSUMED_R7_DEVELOPMENT_COHORT"

PROVENANCE_INHERITED = "INHERITED_R2"
PROVENANCE_NEW = "FLOW_NEW"
PROVENANCE_EXTENDED = "FLOW_EXTENDED"
PROVENANCE_MERGED = "FLOW_MERGED_OR_SPLIT"

CAUSE_INHERITED = "INHERITED_R2_NOT_MOTION_SOURCE"
CAUSE_STATIC = "STATIC_PSEUDO_MOTION"
CAUSE_NONCRITICAL = "REAL_MOVER_BUT_NONCRITICAL"
CAUSE_EXTRAPOLATION = "BAD_ROUTE_EXTRAPOLATION"
CAUSE_ATTRIBUTION = "ATTRIBUTION_OR_FRAGMENTATION"
CAUSE_UNKNOWN = "NOT_EVALUABLE"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8", newline="\n")
    os.replace(partial, path)


def _base_predictions(case: SegmentCase) -> tuple[Prediction, ...]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    runner = DTRR2Arm(config)
    origin = case.samples[0].time_s
    output = []
    for sample in case.samples:
        observation = sensor_observation(sample)
        output.append(
            runner.step(
                CausalFrame(
                    time_s=sample.time_s - origin,
                    ego_pose=sample_pose(sample),
                    observations=() if observation is None else (observation,),
                    person_detection_count=int(observation is not None),
                )
            )
        )
    return tuple(output)


def _scored_segments(case: SegmentCase, predictions: Sequence[Prediction]) -> list[AlertSegment]:
    return [
        segment
        for segment in alert_segments(predictions)
        if any(case.known[index] for index in range(segment.start_index, segment.end_index + 1))
    ]


def _false_segments(case: SegmentCase, predictions: Sequence[Prediction]) -> list[AlertSegment]:
    return [
        segment
        for segment in _scored_segments(case, predictions)
        if not any(
            case.known[index] and case.truth[index] is True
            for index in range(segment.start_index, segment.end_index + 1)
        )
    ]


def classify_provenance(
    segment: AlertSegment,
    baseline_false_segments: Sequence[AlertSegment],
    flow_only_indices: Sequence[int],
) -> str:
    if not flow_only_indices:
        return PROVENANCE_INHERITED
    overlaps = [
        baseline
        for baseline in baseline_false_segments
        if baseline.start_index <= segment.end_index and baseline.end_index >= segment.start_index
    ]
    if not overlaps:
        return PROVENANCE_NEW
    if len(overlaps) > 1:
        return PROVENANCE_MERGED
    return PROVENANCE_EXTENDED


def classify_primary_cause(
    provenance: str,
    truth_value: bool | None,
    target_speed_mps: float | None,
    target_linear_entry_s: float | None,
    flow_target_velocity_error_mps: float | None,
) -> str:
    if provenance == PROVENANCE_INHERITED:
        return CAUSE_INHERITED
    if truth_value is not False:
        return CAUSE_UNKNOWN
    if target_speed_mps is None:
        return CAUSE_UNKNOWN
    if target_speed_mps + 1e-12 < FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps:
        return CAUSE_STATIC
    if target_linear_entry_s is not None:
        return CAUSE_EXTRAPOLATION
    if flow_target_velocity_error_mps is None:
        return CAUSE_UNKNOWN
    if (
        flow_target_velocity_error_mps
        <= FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps + 1e-12
    ):
        return CAUSE_NONCRITICAL
    return CAUSE_ATTRIBUTION


def _target_world_xy(sample: SensorSample) -> tuple[float, float]:
    cosine = math.cos(sample.ego_yaw_rad)
    sine = math.sin(sample.ego_yaw_rad)
    return (
        sample.ego_x_m + sample.forward_m * cosine - sample.left_m * sine,
        sample.ego_y_m + sample.forward_m * sine + sample.left_m * cosine,
    )


def _target_velocity(case: SegmentCase, index: int) -> tuple[float, float] | None:
    current = case.samples[index]
    candidates = [
        prior
        for prior in range(index)
        if FROZEN_FLOW_CONFIG.history_min_s
        <= current.time_s - case.samples[prior].time_s
        <= FROZEN_FLOW_CONFIG.history_max_s
    ]
    if not candidates:
        return None
    history = min(
        candidates,
        key=lambda prior: abs(
            (current.time_s - case.samples[prior].time_s)
            - FROZEN_FLOW_CONFIG.history_target_s
        ),
    )
    previous = case.samples[history]
    span_s = current.time_s - previous.time_s
    previous_world = _target_world_xy(previous)
    current_world = _target_world_xy(current)
    world_vx = (current_world[0] - previous_world[0]) / span_s
    world_vy = (current_world[1] - previous_world[1]) / span_s
    cosine = math.cos(current.ego_yaw_rad)
    sine = math.sin(current.ego_yaw_rad)
    return (
        cosine * world_vx + sine * world_vy,
        -sine * world_vx + cosine * world_vy,
    )


def _target_linear_entry(sample: SensorSample, velocity: tuple[float, float]) -> float | None:
    return _first_tube_entry_s(
        Vec2(sample.forward_m, sample.left_m),
        Vec2(*velocity),
        ROUTE_HALF_WIDTH_M + sample.truth_radius_m,
        HORIZON_S,
        MINIMUM_CLOSING_SPEED_MPS,
    )


def _flow_detail(ledger: FlowLedger, sample: SensorSample) -> dict[str, Any]:
    import numpy as np

    forward, left, velocity_forward, velocity_left, component = ledger.frame_cells(
        sample.frame_index
    )
    if len(forward) == 0:
        return {
            "risk": False,
            "risk_cells": 0,
            "component_ids": [],
            "component_count": 0,
            "minimum_entry_s": None,
            "mean_velocity_forward_mps": None,
            "mean_velocity_left_mps": None,
        }
    margin = FROZEN_FLOW_CONFIG.association_margin_cells * FROZEN_FLOW_CONFIG.voxel_size_m
    within = np.hypot(forward - sample.forward_m, left - sample.left_m) <= (
        sample.truth_radius_m + margin
    )
    risky_indices = []
    entries = []
    for index in np.nonzero(within)[0]:
        entry = _entry_s(
            float(forward[index]),
            float(left[index]),
            float(velocity_forward[index]),
            float(velocity_left[index]),
        )
        if entry is not None:
            risky_indices.append(int(index))
            entries.append(float(entry))
    if not risky_indices:
        return {
            "risk": False,
            "risk_cells": 0,
            "component_ids": [],
            "component_count": 0,
            "minimum_entry_s": None,
            "mean_velocity_forward_mps": None,
            "mean_velocity_left_mps": None,
        }
    component_ids = sorted({int(component[index]) for index in risky_indices})
    return {
        "risk": True,
        "risk_cells": len(risky_indices),
        "component_ids": component_ids,
        "component_count": len(component_ids),
        "minimum_entry_s": min(entries),
        "mean_velocity_forward_mps": float(
            np.mean([velocity_forward[index] for index in risky_indices])
        ),
        "mean_velocity_left_mps": float(
            np.mean([velocity_left[index] for index in risky_indices])
        ),
    }


def _case_key(case: SegmentCase) -> str:
    return f"{case.label_id}#{case.segment_index}"


def _segment_row(
    case: SegmentCase,
    segment: AlertSegment,
    baseline_false: Sequence[AlertSegment],
    baseline_predictions: Sequence[Prediction],
    flow_predictions: Sequence[Prediction],
    details: dict[int, dict[str, Any]],
    component_targets: dict[tuple[int, int], set[str]],
) -> dict[str, Any]:
    indices = list(range(segment.start_index, segment.end_index + 1))
    flow_risk_indices = [index for index in indices if details[index]["risk"]]
    flow_only_indices = [
        index
        for index in flow_risk_indices
        if baseline_predictions[index].raw_alert is not True
    ]
    provenance = classify_provenance(segment, baseline_false, flow_only_indices)
    diagnostic_index = (
        flow_only_indices[0]
        if flow_only_indices
        else (flow_risk_indices[0] if flow_risk_indices else None)
    )
    diagnostic = None if diagnostic_index is None else details[diagnostic_index]
    target_velocity = None if diagnostic_index is None else _target_velocity(case, diagnostic_index)
    target_speed = None if target_velocity is None else math.hypot(*target_velocity)
    target_entry = (
        None
        if target_velocity is None or diagnostic_index is None
        else _target_linear_entry(case.samples[diagnostic_index], target_velocity)
    )
    flow_speed = None
    velocity_error = None
    if diagnostic is not None and diagnostic["mean_velocity_forward_mps"] is not None:
        flow_speed = math.hypot(
            diagnostic["mean_velocity_forward_mps"],
            diagnostic["mean_velocity_left_mps"],
        )
        if target_velocity is not None:
            velocity_error = math.hypot(
                diagnostic["mean_velocity_forward_mps"] - target_velocity[0],
                diagnostic["mean_velocity_left_mps"] - target_velocity[1],
            )
    cause = classify_primary_cause(
        provenance,
        None if diagnostic_index is None else case.truth[diagnostic_index],
        target_speed,
        target_entry,
        velocity_error,
    )

    flags = set()
    if any(details[index]["component_count"] > 1 for index in flow_risk_indices):
        flags.add("MULTIPLE_RESPONSIBLE_COMPONENTS")
    if any(
        len(component_targets[(case.samples[index].frame_index, component_id)]) > 1
        for index in flow_risk_indices
        for component_id in details[index]["component_ids"]
    ):
        flags.add("ATTRIBUTION_AMBIGUOUS_ACROSS_TARGETS")
    for left_index, right_index in zip(flow_risk_indices, flow_risk_indices[1:]):
        left_detail = details[left_index]
        right_detail = details[right_index]
        if right_index != left_index + 1 or (
            left_detail["component_count"] != right_detail["component_count"]
        ):
            flags.add("TEMPORAL_COMPONENT_DISCONTINUITY_SUSPECTED")
        left_velocity = (
            left_detail["mean_velocity_forward_mps"],
            left_detail["mean_velocity_left_mps"],
        )
        right_velocity = (
            right_detail["mean_velocity_forward_mps"],
            right_detail["mean_velocity_left_mps"],
        )
        if None not in left_velocity and None not in right_velocity:
            jump = math.hypot(
                float(right_velocity[0]) - float(left_velocity[0]),
                float(right_velocity[1]) - float(left_velocity[1]),
            )
            if jump + 1e-12 >= FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps:
                flags.add("VELOCITY_DISCONTINUITY_SUSPECTED")
    if diagnostic is not None and target_entry is None:
        flags.add("FLOW_ENTRY_NOT_SUPPORTED_BY_TARGET_LINEAR_VELOCITY")
    if cause == CAUSE_STATIC:
        flags.add("TARGET_SPEED_BELOW_FROZEN_FLOW_MINIMUM")

    start_sample = case.samples[segment.start_index]
    end_sample = case.samples[segment.end_index]
    overlapping_baseline = [
        baseline
        for baseline in baseline_false
        if baseline.start_index <= segment.end_index and baseline.end_index >= segment.start_index
    ]
    return {
        "segment_id": (
            f"{case.label_id}#{case.segment_index}:"
            f"{start_sample.frame_index:06d}-{end_sample.frame_index:06d}"
        ),
        "label_id": case.label_id,
        "target_segment_index": case.segment_index,
        "first_frame": start_sample.frame_index,
        "last_frame": end_sample.frame_index,
        "duration_s": end_sample.time_s - start_sample.time_s,
        "provenance": provenance,
        "primary_cause": cause,
        "flags": sorted(flags),
        "diagnostic_frame": (
            None if diagnostic_index is None else case.samples[diagnostic_index].frame_index
        ),
        "flow_risk_frames": len(flow_risk_indices),
        "flow_only_frames": len(flow_only_indices),
        "overlapping_r2_false_segments": len(overlapping_baseline),
        "responsible_component_count": (
            None if diagnostic is None else diagnostic["component_count"]
        ),
        "responsible_risk_cells": None if diagnostic is None else diagnostic["risk_cells"],
        "flow_entry_s": None if diagnostic is None else diagnostic["minimum_entry_s"],
        "flow_speed_mps": flow_speed,
        "target_speed_mps": target_speed,
        "target_linear_entry_s": target_entry,
        "flow_target_velocity_error_mps": velocity_error,
        "truth_at_diagnostic_frame": (
            None if diagnostic_index is None else case.truth[diagnostic_index]
        ),
        "component_identity_ceiling": "frame_local_only",
    }


def _write_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "segment_id",
        "label_id",
        "target_segment_index",
        "first_frame",
        "last_frame",
        "duration_s",
        "provenance",
        "primary_cause",
        "flags",
        "diagnostic_frame",
        "flow_risk_frames",
        "flow_only_frames",
        "overlapping_r2_false_segments",
        "responsible_component_count",
        "responsible_risk_cells",
        "flow_entry_s",
        "flow_speed_mps",
        "target_speed_mps",
        "target_linear_entry_s",
        "flow_target_velocity_error_mps",
        "truth_at_diagnostic_frame",
        "component_identity_ceiling",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["flags"] = ";".join(row["flags"])
            writer.writerow(value)
    os.replace(partial, path)


def _polyline(values: Sequence[int], x_at: Any, y_at: Any) -> str:
    return " ".join(
        f"{x_at(frame):.2f},{y_at(value):.2f}"
        for frame, value in zip(range(FIRST_FRAME, LAST_FRAME + 1), values)
    )


def render_timeline_svg(
    path: Path,
    *,
    global_risky_cells: Sequence[int],
    truth_counts: Sequence[int],
    baseline_counts: Sequence[int],
    flow_counts: Sequence[int],
    rows: Sequence[dict[str, Any]],
) -> None:
    width = 1800
    margin_left = 410
    margin_right = 40
    plot_width = width - margin_left - margin_right
    top = 78
    panel_height = 145
    second_top = top + panel_height + 55
    rows_top = second_top + panel_height + 75
    row_height = 20
    height = rows_top + row_height * len(rows) + 90
    frame_span = max(1, LAST_FRAME - FIRST_FRAME)

    def x_at(frame: int) -> float:
        return margin_left + (frame - FIRST_FRAME) * plot_width / frame_span

    max_global = max(1, max(global_risky_cells, default=0))
    max_targets = max(1, *(truth_counts + baseline_counts + flow_counts))

    def global_y(value: int) -> float:
        return top + panel_height - value * panel_height / max_global

    def target_y(value: int) -> float:
        return second_top + panel_height - value * panel_height / max_targets

    cause_colors = {
        CAUSE_INHERITED: "#64748b",
        CAUSE_STATIC: "#ef4444",
        CAUSE_NONCRITICAL: "#f59e0b",
        CAUSE_EXTRAPOLATION: "#8b5cf6",
        CAUSE_ATTRIBUTION: "#db2777",
        CAUSE_UNKNOWN: "#111827",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#172033}.title{font-size:24px;font-weight:700}.sub{font-size:13px}.axis{font-size:11px;fill:#526070}.label{font-size:11px}.lane{font-size:10px}</style>',
        '<text x="24" y="34" class="title">DTR-M0 R7-P false-segment attribution timeline</text>',
        '<text x="24" y="57" class="sub">Consumed 143-frame Development cohort; diagnostics only; R7 matcher, lifecycle, gate, and verdict unchanged.</text>',
    ]
    for panel_top, label in ((top, "Global risky cells"), (second_top, "Target-track counts")):
        parts.append(
            f'<rect x="{margin_left}" y="{panel_top}" width="{plot_width}" height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1"/>'
        )
        parts.append(f'<text x="24" y="{panel_top + 18}" class="sub">{label}</text>')
    for tick in range(FIRST_FRAME, LAST_FRAME + 1, 20):
        x = x_at(tick)
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - 55}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x}" y="{height - 35}" text-anchor="middle" class="axis">{tick}</text>')
    parts.append(
        f'<polyline fill="none" stroke="#0f766e" stroke-width="2" points="{_polyline(global_risky_cells, x_at, global_y)}"/>'
    )
    series = (
        (truth_counts, "#16a34a", "Truth-positive targets"),
        (baseline_counts, "#2563eb", "R2 active targets"),
        (flow_counts, "#dc2626", "R7 active targets"),
    )
    legend_x = margin_left
    for values, color, label in series:
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{_polyline(values, x_at, target_y)}"/>'
        )
        parts.append(f'<line x1="{legend_x}" y1="{second_top - 18}" x2="{legend_x + 26}" y2="{second_top - 18}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 32}" y="{second_top - 14}" class="axis">{html.escape(label)}</text>')
        legend_x += 220
    parts.append(f'<text x="24" y="{rows_top - 22}" class="sub">Twenty target-aware false segments</text>')
    for row_index, row in enumerate(rows):
        y = rows_top + row_index * row_height
        color = cause_colors[row["primary_cause"]]
        x1 = x_at(row["first_frame"])
        x2 = max(x1 + 3.0, x_at(row["last_frame"]))
        label = f'{row_index + 1:02d} {row["provenance"]} / {row["primary_cause"]}'
        parts.append(f'<text x="24" y="{y + 12}" class="lane">{html.escape(label)}</text>')
        parts.append(f'<rect x="{x1}" y="{y + 2}" width="{x2 - x1}" height="12" rx="2" fill="{color}" opacity="0.82"/>')
        if row["diagnostic_frame"] is not None:
            marker_x = x_at(row["diagnostic_frame"])
            parts.append(f'<circle cx="{marker_x}" cy="{y + 8}" r="3" fill="#111827"/>')
    legend_y = rows_top + row_height * len(rows) + 28
    for legend_index, (cause, color) in enumerate(cause_colors.items()):
        legend_x = margin_left + (legend_index % 3) * 450
        current_y = legend_y + (legend_index // 3) * 22
        parts.append(
            f'<rect x="{legend_x}" y="{current_y}" width="12" height="12" fill="{color}"/>'
        )
        parts.append(f'<text x="{legend_x + 17}" y="{current_y + 11}" class="axis">{html.escape(cause)}</text>')
    parts.append('</svg>')
    _atomic_text(path, "\n".join(parts) + "\n")


def _count_by_frame(
    cases: Sequence[SegmentCase],
    predictions: dict[str, tuple[Sequence[Prediction], Sequence[Prediction]]],
) -> tuple[list[int], list[int], list[int]]:
    truth_counts = Counter()
    baseline_counts = Counter()
    flow_counts = Counter()
    for case in cases:
        baseline, flow = predictions[_case_key(case)]
        for index, sample in enumerate(case.samples):
            truth_counts[sample.frame_index] += int(
                case.known[index] and case.truth[index] is True
            )
            baseline_counts[sample.frame_index] += int(
                baseline[index].signal in ACTIVE_SIGNALS
            )
            flow_counts[sample.frame_index] += int(flow[index].signal in ACTIVE_SIGNALS)
    frames = range(FIRST_FRAME, LAST_FRAME + 1)
    return (
        [truth_counts[frame] for frame in frames],
        [baseline_counts[frame] for frame in frames],
        [flow_counts[frame] for frame in frames],
    )


def run(
    *,
    r7_result_path: Path,
    ledger_path: Path,
    ledger_manifest_path: Path,
    output_path: Path,
    table_path: Path,
    timeline_path: Path,
) -> dict[str, Any]:
    r7_result_path = r7_result_path.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger_manifest_path = ledger_manifest_path.resolve(strict=True)
    r7 = json.loads(r7_result_path.read_text(encoding="utf-8"))
    require(
        r7.get("gate", {}).get("verdict")
        == "R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_NOT_MET_NO_R8",
        "r7_terminal_drift",
    )
    ledger = load_flow_ledger(ledger_path, ledger_manifest_path)
    require(
        ledger.manifest["ledger_sha256"] == r7["flow_source"]["ledger_sha256"],
        "r7_ledger_authority_mismatch",
    )

    source = r7["source"]
    known_tracks_path = Path(source["known_height_tracks"]).resolve(strict=True)
    labels_path = Path(source["labels"]).resolve(strict=True)
    timestamps_path = Path(source["timestamps"]).resolve(strict=True)
    bag_path = Path(source["bag"]).resolve(strict=True)
    require(sha256_file(known_tracks_path) == source["known_height_tracks_sha256"], "known_tracks_hash_drift")
    require(sha256_file(labels_path) == source["labels_sha256"], "labels_hash_drift")
    require(sha256_file(timestamps_path) == source["timestamps_sha256"], "timestamps_hash_drift")
    require(sha256_file(bag_path) == source["bag_sha256"], "bag_hash_drift")

    sensor_rows = read_jsonl(known_tracks_path)
    poses, _rgb_times, _bag_authority = read_bag_pose_and_rgb(bag_path)
    timestamps = load_image_timestamps(timestamps_path)
    context = {
        frame: {
            "image_time_s": timestamps[frame],
            "pose": interpolate_pose(poses, round(timestamps[frame] * 1e9)),
        }
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    tracks, geometry_quality = load_truth_and_associate(labels_path, sensor_rows, context)
    cases = cases_from_tracks(tracks)

    predictions: dict[str, tuple[Sequence[Prediction], Sequence[Prediction]]] = {}
    details_by_case: dict[str, dict[int, dict[str, Any]]] = {}
    component_targets: dict[tuple[int, int], set[str]] = defaultdict(set)
    baseline_metrics = ArmAccumulator()
    flow_metrics = ArmAccumulator()
    for case in cases:
        key = _case_key(case)
        baseline = _base_predictions(case)
        flow_run = run_flow_arm(case, set(), ledger)
        predictions[key] = (baseline, flow_run.predictions)
        baseline_metrics.merge(
            metrics_for_run(case, SimpleNamespace(predictions=baseline))
        )
        flow_metrics.merge(metrics_for_run(case, flow_run))
        case_details = {}
        for index, sample in enumerate(case.samples):
            detail = _flow_detail(ledger, sample)
            case_details[index] = detail
            for component_id in detail["component_ids"]:
                component_targets[(sample.frame_index, component_id)].add(key)
        details_by_case[key] = case_details

    baseline_summary = baseline_metrics.to_dict(include_escalation=True)
    flow_summary = flow_metrics.to_dict(include_escalation=True)
    for field in (
        "alert_segments",
        "event_detection_evaluable_alert_segments",
        "event_detection_true_positives",
        "false_alert_segments",
    ):
        require(
            baseline_summary[field] == r7["original_cohort"]["r2"][field],
            f"r2_replay_drift:{field}",
        )
        require(
            flow_summary[field] == r7["original_cohort"]["r7_p_occupancy_flow"][field],
            f"r7_replay_drift:{field}",
        )

    rows = []
    for case in cases:
        key = _case_key(case)
        baseline, flow = predictions[key]
        baseline_false = _false_segments(case, baseline)
        for segment in _false_segments(case, flow):
            rows.append(
                _segment_row(
                    case,
                    segment,
                    baseline_false,
                    baseline,
                    flow,
                    details_by_case[key],
                    component_targets,
                )
            )
    rows.sort(key=lambda row: (row["first_frame"], row["label_id"], row["last_frame"]))
    require(len(rows) == r7["original_cohort"]["r7_p_occupancy_flow"]["false_alert_segments"], "r7_false_segment_row_count")

    provenance_counts = dict(sorted(Counter(row["provenance"] for row in rows).items()))
    cause_counts = dict(sorted(Counter(row["primary_cause"] for row in rows).items()))
    flow_caused = [row for row in rows if row["provenance"] != PROVENANCE_INHERITED]
    flow_cause_counts = dict(
        sorted(Counter(row["primary_cause"] for row in flow_caused).items())
    )
    flag_counts = dict(sorted(Counter(flag for row in rows for flag in row["flags"]).items()))

    truth_counts, baseline_counts, flow_counts = _count_by_frame(cases, predictions)
    global_cells = [
        int(r7["global_flow_nuisance"]["risky_cells_by_frame"][f"{frame:06d}"])
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    ]
    _write_table(table_path.resolve(), rows)
    render_timeline_svg(
        timeline_path.resolve(),
        global_risky_cells=global_cells,
        truth_counts=truth_counts,
        baseline_counts=baseline_counts,
        flow_counts=flow_counts,
        rows=rows,
    )
    result = {
        "schema_version": SCHEMA,
        "table_schema_version": TABLE_SCHEMA,
        "status": "DTR_M0_R7_READ_ONLY_ERROR_ATTRIBUTION_COMPLETE",
        "claim_ceiling": CLAIM_CEILING,
        "question": (
            "Which R7-P false alert segments are inherited from R2, and which flow-caused "
            "segments are most consistent with static pseudo-motion, a real but noncritical "
            "mover, or constant-velocity route extrapolation error?"
        ),
        "frozen": {
            "r7_verdict_unchanged": r7["gate"]["verdict"],
            "r7_ledger_sha256": ledger.manifest["ledger_sha256"],
            "motion_threshold_reused_mps": FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps,
            "history_window_reused_s": {
                "target": FROZEN_FLOW_CONFIG.history_target_s,
                "minimum": FROZEN_FLOW_CONFIG.history_min_s,
                "maximum": FROZEN_FLOW_CONFIG.history_max_s,
            },
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "route_horizon_s": HORIZON_S,
            "lifecycle_or_matcher_changes": False,
        },
        "source": {
            "r7_result": str(r7_result_path),
            "r7_result_sha256": sha256_file(r7_result_path),
            "flow_ledger": str(ledger_path),
            "flow_ledger_sha256": sha256_file(ledger_path),
            "flow_manifest": str(ledger_manifest_path),
            "flow_manifest_sha256": sha256_file(ledger_manifest_path),
            "known_height_tracks": str(known_tracks_path),
            "labels": str(labels_path),
            "timestamps": str(timestamps_path),
            "bag": str(bag_path),
            "evaluable_target_segments": len(cases),
            "geometry_quality": geometry_quality,
        },
        "replay_check": {
            "r2": {
                key: baseline_summary[key]
                for key in (
                    "alert_segments",
                    "event_detection_evaluable_alert_segments",
                    "event_detection_true_positives",
                    "false_alert_segments",
                )
            },
            "r7_p": {
                key: flow_summary[key]
                for key in (
                    "alert_segments",
                    "event_detection_evaluable_alert_segments",
                    "event_detection_true_positives",
                    "false_alert_segments",
                )
            },
        },
        "summary": {
            "false_segments": len(rows),
            "flow_caused_or_modified_false_segments": len(flow_caused),
            "provenance_counts": provenance_counts,
            "primary_cause_counts_all": cause_counts,
            "primary_cause_counts_flow_caused": flow_cause_counts,
            "flag_counts": flag_counts,
        },
        "definitions": {
            PROVENANCE_INHERITED: "No flow-only risk frame occurs inside the R7 false segment.",
            PROVENANCE_NEW: "The R7 false segment has flow-only risk and overlaps no R2 false segment.",
            PROVENANCE_EXTENDED: "The R7 false segment has flow-only risk and overlaps one R2 false segment.",
            PROVENANCE_MERGED: "The R7 false segment has flow-only risk and overlaps multiple R2 false segments.",
            CAUSE_STATIC: "At the first flow-only diagnostic frame, scorer-side target speed is below the already frozen R7 minimum dynamic speed.",
            CAUSE_NONCRITICAL: "The scorer-side target is moving, its same-history constant-velocity future does not enter the frozen route tube, and R7 flow velocity agrees within the already frozen minimum motion speed.",
            CAUSE_EXTRAPOLATION: "The scorer-side target is moving and its same-history constant-velocity future enters the route tube, while native future truth remains negative for the false segment.",
            CAUSE_ATTRIBUTION: "The target is moving, but R7 flow velocity differs from scorer-side target velocity by more than the already frozen R7 minimum motion speed; this supports a motion-attribution mismatch, not a proven component split or merge.",
            CAUSE_UNKNOWN: "The sealed evidence lacks a same-history scorer-side target velocity at the diagnostic frame.",
        },
        "false_segments": rows,
        "outputs": {
            "table_csv": str(table_path.resolve()),
            "timeline_svg": str(timeline_path.resolve()),
        },
        "limitations": [
            "This is post-outcome scorer-side attribution on the consumed R7 Development cohort, not new performance evidence.",
            "R7 component IDs are frame-local; temporal split/merge is only a suspicion flag, never a proven identity claim.",
            "Target constant-velocity counterfactuals diagnose information routing; they are not an alternative deployed matcher or a parameter sweep.",
            "A target-speed classification cannot prove whether the responsible LiDAR cells came from the target surface or nearby static clutter.",
            "UNKNOWN and NOT_EVALUABLE are not negative results.",
        ],
    }
    write_json(output_path.resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r7-result", type=Path, required=True)
    parser.add_argument("--flow-ledger", type=Path, required=True)
    parser.add_argument("--flow-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    require(args.table.suffix.lower() == ".csv", "table_must_be_csv")
    require(args.timeline.suffix.lower() == ".svg", "timeline_must_be_svg")
    result = run(
        r7_result_path=args.r7_result,
        ledger_path=args.flow_ledger,
        ledger_manifest_path=args.flow_manifest,
        output_path=args.output,
        table_path=args.table,
        timeline_path=args.timeline,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
