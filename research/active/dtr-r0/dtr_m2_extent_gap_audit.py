"""Run the read-only DTR-M2-D extent-gap audit on the consumed M1 cohort.

This diagnostic never changes a prediction, threshold, lifecycle, or evaluator
gate.  It reuses the sealed M1 native-box point-velocity ledger and compares:

1. the continuous zero-radius trajectory of the M1-attributed dynamic cells;
2. the continuous swept native oriented footprint of the same components.

The route body is the frozen 0.65 m disk and the horizon is the frozen 3.0 s.
The audit covers only the three M1 dropout-window misses and the five M1
point-velocity-induced/modified false segments.  Its output can support freezing
a separate fresh M2-O protocol; it cannot itself reopen, tune, or evaluate one.
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

import numpy as np

from coda_static_ceiling import segment_to_box_entry_fraction
from dtr_m0_r7_error_attribution import (
    PROVENANCE_INHERITED,
    _base_predictions,
    _false_segments,
)
from dtr_m1_point_velocity_oracle import (
    NativeBox,
    ledger_paths,
    load_native_boxes,
    load_oracle_ledger,
)
from dtr_r5_dropout_canary import cases_from_tracks
from dtr_r7_occupancy_flow_canary import _causal_pose, _entry_s, run_flow_arm
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


SCHEMA = "blindassist-dtr-m2-read-only-extent-gap-audit-v1"
STATUS = "DTR_M2_D_READ_ONLY_EXTENT_GAP_AUDIT_COMPLETE"
CLAIM_CEILING = "GEOMETRIC_DIAGNOSIS_ON_CONSUMED_M1_DEVELOPMENT_COHORT"


def _point_geometry(
    forward_m: float,
    left_m: float,
    velocity_forward_mps: float,
    velocity_left_mps: float,
    *,
    horizon_s: float = HORIZON_S,
    route_radius_m: float = ROUTE_HALF_WIDTH_M,
) -> dict[str, float | bool | None]:
    position = np.asarray([forward_m, left_m], dtype=np.float64)
    velocity = np.asarray([velocity_forward_mps, velocity_left_mps], dtype=np.float64)
    speed_squared = float(velocity @ velocity)
    closest_s = 0.0
    if speed_squared > 1e-15:
        closest_s = min(horizon_s, max(0.0, -float(position @ velocity) / speed_squared))
    minimum_distance_m = float(np.linalg.norm(position + velocity * closest_s))

    a = speed_squared
    b = 2.0 * float(position @ velocity)
    c = float(position @ position) - route_radius_m * route_radius_m
    entry_s: float | None = None
    if c <= 1e-12:
        entry_s = 0.0
    elif a > 1e-15:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= -1e-12:
            root = (-b - math.sqrt(max(0.0, discriminant))) / (2.0 * a)
            if -1e-12 <= root <= horizon_s + 1e-12:
                entry_s = min(horizon_s, max(0.0, root))
    return {
        "minimum_center_distance_m": minimum_distance_m,
        "minimum_clearance_m": minimum_distance_m - route_radius_m,
        "closest_time_s": closest_s,
        "entry_s": entry_s,
        "hit": entry_s is not None,
        "frozen_m1_entry_s": _entry_s(
            forward_m,
            left_m,
            velocity_forward_mps,
            velocity_left_mps,
        ),
    }


def _minimum_segment_box_distance(
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    box_x: float,
    box_y: float,
    box_yaw: float,
    length_m: float,
    width_m: float,
) -> tuple[float, float]:
    """Return exact minimum distance and segment fraction for a fixed OBB."""

    cosine = math.cos(box_yaw)
    sine = math.sin(box_yaw)

    def local(point_x: float, point_y: float) -> tuple[float, float]:
        dx = point_x - box_x
        dy = point_y - box_y
        return dx * cosine + dy * sine, -dx * sine + dy * cosine

    start_forward, start_left = local(start_x, start_y)
    end_forward, end_left = local(end_x, end_y)
    velocity_forward = end_forward - start_forward
    velocity_left = end_left - start_left
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    breakpoints = [0.0, 1.0]
    for start, velocity, half_extent in (
        (start_forward, velocity_forward, half_length),
        (start_left, velocity_left, half_width),
    ):
        if abs(velocity) <= 1e-15:
            continue
        for boundary in (-half_extent, half_extent):
            fraction = (boundary - start) / velocity
            if 0.0 < fraction < 1.0:
                breakpoints.append(fraction)
    breakpoints = sorted(set(breakpoints))

    best_squared = math.inf
    best_fraction = 0.0
    for interval_start, interval_end in zip(breakpoints, breakpoints[1:]):
        midpoint = (interval_start + interval_end) / 2.0

        def coefficients(start: float, velocity: float, half_extent: float) -> tuple[float, float]:
            value = start + velocity * midpoint
            if value > half_extent:
                return start - half_extent, velocity
            if value < -half_extent:
                return start + half_extent, velocity
            return 0.0, 0.0

        forward_offset, forward_slope = coefficients(
            start_forward, velocity_forward, half_length
        )
        left_offset, left_slope = coefficients(start_left, velocity_left, half_width)
        quadratic = forward_slope**2 + left_slope**2
        linear = 2.0 * (
            forward_offset * forward_slope + left_offset * left_slope
        )
        candidates = [interval_start, interval_end]
        if quadratic > 1e-15:
            candidates.append(min(interval_end, max(interval_start, -linear / (2.0 * quadratic))))
        for fraction in candidates:
            forward = forward_offset + forward_slope * fraction
            left = left_offset + left_slope * fraction
            squared = forward * forward + left * left
            if squared < best_squared:
                best_squared = squared
                best_fraction = fraction
    return math.sqrt(max(0.0, best_squared)), best_fraction


def _footprint_geometry(
    box: NativeBox,
    velocity_forward_mps: float,
    velocity_left_mps: float,
    *,
    horizon_s: float = HORIZON_S,
    route_radius_m: float = ROUTE_HALF_WIDTH_M,
) -> dict[str, float | bool | None]:
    # Moving-box versus stationary route disk is equivalent to moving the route
    # center by -velocity against a fixed current box.
    end_x = -velocity_forward_mps * horizon_s
    end_y = -velocity_left_mps * horizon_s
    fraction = segment_to_box_entry_fraction(
        0.0,
        0.0,
        end_x,
        end_y,
        box.center_forward_m,
        box.center_left_m,
        box.yaw_ego_rad,
        box.length_m,
        box.width_m,
        route_radius_m,
    )
    minimum_distance_m, closest_fraction = _minimum_segment_box_distance(
        start_x=0.0,
        start_y=0.0,
        end_x=end_x,
        end_y=end_y,
        box_x=box.center_forward_m,
        box_y=box.center_left_m,
        box_yaw=box.yaw_ego_rad,
        length_m=box.length_m,
        width_m=box.width_m,
    )
    return {
        "minimum_footprint_distance_m": minimum_distance_m,
        "minimum_clearance_m": minimum_distance_m - route_radius_m,
        "closest_time_s": closest_fraction * horizon_s,
        "entry_s": None if fraction is None else fraction * horizon_s,
        "hit": fraction is not None,
    }


def classify_geometry(*, point_hit: bool, footprint_hit: bool, truth_positive: bool) -> str:
    if truth_positive:
        if point_hit and footprint_hit:
            return "POINT_HIT_FOOTPRINT_HIT"
        if point_hit:
            return "POINT_HIT_FOOTPRINT_MISS"
        if footprint_hit:
            return "POINT_MISS_FOOTPRINT_HIT"
        return "POINT_MISS_FOOTPRINT_MISS"
    if footprint_hit:
        return "FOOTPRINT_HIT_TRUTH_NEGATIVE"
    if point_hit:
        return "POINT_HIT_FOOTPRINT_MISS_TRUTH_NEGATIVE"
    return "POINT_MISS_FOOTPRINT_MISS_TRUTH_NEGATIVE"


def _box_corners(box: NativeBox, dx: float = 0.0, dy: float = 0.0) -> list[tuple[float, float]]:
    cosine = math.cos(box.yaw_ego_rad)
    sine = math.sin(box.yaw_ego_rad)
    corners = []
    for forward, left in (
        (-box.length_m / 2.0, -box.width_m / 2.0),
        (box.length_m / 2.0, -box.width_m / 2.0),
        (box.length_m / 2.0, box.width_m / 2.0),
        (-box.length_m / 2.0, box.width_m / 2.0),
    ):
        corners.append(
            (
                box.center_forward_m + dx + cosine * forward - sine * left,
                box.center_left_m + dy + sine * forward + cosine * left,
            )
        )
    return corners


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    values = sorted(set(points))
    if len(values) <= 1:
        return values

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _frame_geometry(
    *,
    sample: Any,
    target_label_id: str,
    ledger: Any,
    boxes_by_frame: dict[int, list[NativeBox]],
    index_to_label: dict[int, str],
) -> dict[str, Any]:
    forward, left, velocity_forward, velocity_left, component = ledger.frame_cells(
        sample.frame_index
    )
    margin = (
        float(ledger.manifest["frozen_downstream"]["r7_flow_config"]["association_margin_cells"])
        * float(ledger.manifest["frozen_downstream"]["r7_flow_config"]["voxel_size_m"])
    )
    within = np.hypot(forward - sample.forward_m, left - sample.left_m) <= (
        sample.truth_radius_m + margin
    )
    attributed_indices = np.nonzero(within)[0]
    boxes = {box.label_id: box for box in boxes_by_frame.get(sample.frame_index, ())}
    component_rows = []
    point_rows = []
    figure_components = []
    for component_id_value in sorted(set(int(component[index]) for index in attributed_indices)):
        label_id = index_to_label[component_id_value]
        attributed = attributed_indices[component[attributed_indices] == component_id_value]
        all_component = np.nonzero(component == component_id_value)[0]
        robust_vf = float(np.median(velocity_forward[all_component]))
        robust_vl = float(np.median(velocity_left[all_component]))
        component_point_rows = []
        for index in attributed:
            geometry = _point_geometry(
                float(forward[index]),
                float(left[index]),
                float(velocity_forward[index]),
                float(velocity_left[index]),
            )
            row = {
                **geometry,
                "forward_m": float(forward[index]),
                "left_m": float(left[index]),
                "velocity_forward_mps": float(velocity_forward[index]),
                "velocity_left_mps": float(velocity_left[index]),
            }
            component_point_rows.append(row)
            point_rows.append(row)
        box = boxes.get(label_id)
        footprint = None if box is None else _footprint_geometry(box, robust_vf, robust_vl)
        component_rows.append(
            {
                "component_id": component_id_value,
                "label_id": label_id,
                "is_target_component": label_id == target_label_id,
                "attributed_cells": len(attributed),
                "component_cells": len(all_component),
                "velocity_forward_mps": robust_vf,
                "velocity_left_mps": robust_vl,
                "speed_mps": math.hypot(robust_vf, robust_vl),
                "native_box_available": box is not None,
                "native_box": None
                if box is None
                else {
                    "center_forward_m": box.center_forward_m,
                    "center_left_m": box.center_left_m,
                    "yaw_ego_rad": box.yaw_ego_rad,
                    "length_m": box.length_m,
                    "width_m": box.width_m,
                },
                "point_hit": any(bool(row["hit"]) for row in component_point_rows),
                "frozen_m1_point_risk": any(
                    row["frozen_m1_entry_s"] is not None for row in component_point_rows
                ),
                "footprint": footprint,
            }
        )
        if box is not None:
            figure_components.append(
                {
                    "label_id": label_id,
                    "is_target_component": label_id == target_label_id,
                    "box": box,
                    "velocity_forward_mps": robust_vf,
                    "velocity_left_mps": robust_vl,
                    "points": component_point_rows,
                }
            )

    point_entry_values = [row["entry_s"] for row in point_rows if row["entry_s"] is not None]
    frozen_entry_values = [
        row["frozen_m1_entry_s"]
        for row in point_rows
        if row["frozen_m1_entry_s"] is not None
    ]
    footprint_rows = [
        row["footprint"] for row in component_rows if row["footprint"] is not None
    ]
    footprint_entry_values = [
        row["entry_s"] for row in footprint_rows if row["entry_s"] is not None
    ]
    return {
        "frame": sample.frame_index,
        "time_s": sample.time_s,
        "target_label_id": target_label_id,
        "attributed_cells": len(attributed_indices),
        "attributed_components": len(component_rows),
        "target_component_present": any(row["is_target_component"] for row in component_rows),
        "point_hit": bool(point_entry_values),
        "point_entry_s": min(point_entry_values) if point_entry_values else None,
        "point_minimum_clearance_m": min(
            (float(row["minimum_clearance_m"]) for row in point_rows), default=None
        ),
        "frozen_m1_point_risk": bool(frozen_entry_values),
        "frozen_m1_entry_s": min(frozen_entry_values) if frozen_entry_values else None,
        "footprint_hit": bool(footprint_entry_values),
        "footprint_entry_s": min(footprint_entry_values) if footprint_entry_values else None,
        "footprint_minimum_clearance_m": min(
            (float(row["minimum_clearance_m"]) for row in footprint_rows), default=None
        ),
        "components": component_rows,
        "_figure_components": figure_components,
    }


def _aggregate_geometry(
    *,
    kind: str,
    identifier: str,
    truth_positive: bool,
    frames: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    point_hit = any(bool(row["point_hit"]) for row in frames)
    footprint_hit = any(bool(row["footprint_hit"]) for row in frames)
    frozen_m1_point_risk = any(bool(row["frozen_m1_point_risk"]) for row in frames)
    clean_frames = [
        {key: value for key, value in row.items() if key != "_figure_components"}
        for row in frames
    ]
    return {
        "kind": kind,
        "id": identifier,
        **metadata,
        "truth_positive": truth_positive,
        "classification": classify_geometry(
            point_hit=point_hit,
            footprint_hit=footprint_hit,
            truth_positive=truth_positive,
        ),
        "point_hit": point_hit,
        "footprint_hit": footprint_hit,
        "frozen_m1_point_risk": frozen_m1_point_risk,
        "point_minimum_clearance_m": min(
            (
                float(row["point_minimum_clearance_m"])
                for row in frames
                if row["point_minimum_clearance_m"] is not None
            ),
            default=None,
        ),
        "footprint_minimum_clearance_m": min(
            (
                float(row["footprint_minimum_clearance_m"])
                for row in frames
                if row["footprint_minimum_clearance_m"] is not None
            ),
            default=None,
        ),
        "point_entry_s": min(
            (float(row["point_entry_s"]) for row in frames if row["point_entry_s"] is not None),
            default=None,
        ),
        "footprint_entry_s": min(
            (
                float(row["footprint_entry_s"])
                for row in frames
                if row["footprint_entry_s"] is not None
            ),
            default=None,
        ),
        "frames": clean_frames,
    }


def _figure_svg(frame: dict[str, Any], title: str) -> str:
    components = frame["_figure_components"]
    points = [(0.0, 0.0)]
    for component in components:
        box = component["box"]
        vf = component["velocity_forward_mps"]
        vl = component["velocity_left_mps"]
        points.extend(_box_corners(box))
        points.extend(_box_corners(box, vf * HORIZON_S, vl * HORIZON_S))
        for row in component["points"]:
            points.append((row["forward_m"], row["left_m"]))
            points.append(
                (
                    row["forward_m"] + row["velocity_forward_mps"] * HORIZON_S,
                    row["left_m"] + row["velocity_left_mps"] * HORIZON_S,
                )
            )
    min_x = min(point[0] for point in points) - 1.0
    max_x = max(point[0] for point in points) + 1.0
    min_y = min(point[1] for point in points) - 1.0
    max_y = max(point[1] for point in points) + 1.0
    scale = min(620.0 / max(1.0, max_x - min_x), 620.0 / max(1.0, max_y - min_y))

    def screen(point: tuple[float, float]) -> tuple[float, float]:
        # Forward is up; left is screen-left.
        return 380.0 - point[1] * scale, 390.0 - point[0] * scale

    def polygon(values: Sequence[tuple[float, float]]) -> str:
        return " ".join(f"{screen(value)[0]:.2f},{screen(value)[1]:.2f}" for value in values)

    elements = [
        '<rect width="760" height="760" fill="#ffffff"/>',
        f'<text x="30" y="35" font-family="sans-serif" font-size="20" font-weight="700">{html.escape(title)}</text>',
        '<text x="30" y="60" font-family="sans-serif" font-size="13" fill="#4b5563">Red: zero-radius M1 cell paths · Orange: native swept footprint · Blue: 0.65 m route body</text>',
    ]
    origin_x, origin_y = screen((0.0, 0.0))
    elements.append(
        f'<circle cx="{origin_x:.2f}" cy="{origin_y:.2f}" r="{ROUTE_HALF_WIDTH_M * scale:.2f}" fill="#60a5fa" fill-opacity="0.26" stroke="#2563eb" stroke-width="2"/>'
    )
    elements.append(
        f'<circle cx="{origin_x:.2f}" cy="{origin_y:.2f}" r="4" fill="#1d4ed8"/>'
    )
    for component in components:
        box = component["box"]
        vf = component["velocity_forward_mps"]
        vl = component["velocity_left_mps"]
        start = _box_corners(box)
        end = _box_corners(box, vf * HORIZON_S, vl * HORIZON_S)
        swept = _convex_hull(start + end)
        elements.append(
            f'<polygon points="{polygon(swept)}" fill="#f59e0b" fill-opacity="0.18" stroke="#d97706" stroke-width="1.5"/>'
        )
        elements.append(
            f'<polygon points="{polygon(start)}" fill="#f59e0b" fill-opacity="0.46" stroke="#b45309" stroke-width="2"/>'
        )
        elements.append(
            f'<polygon points="{polygon(end)}" fill="none" stroke="#d97706" stroke-width="2" stroke-dasharray="6 4"/>'
        )
        for row in component["points"]:
            start_point = (row["forward_m"], row["left_m"])
            end_point = (
                row["forward_m"] + row["velocity_forward_mps"] * HORIZON_S,
                row["left_m"] + row["velocity_left_mps"] * HORIZON_S,
            )
            sx, sy = screen(start_point)
            ex, ey = screen(end_point)
            elements.append(
                f'<line x1="{sx:.2f}" y1="{sy:.2f}" x2="{ex:.2f}" y2="{ey:.2f}" stroke="#dc2626" stroke-width="1.5" opacity="0.75"/>'
            )
            elements.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3" fill="#dc2626"/>')
    elements.extend(
        [
            '<text x="30" y="720" font-family="sans-serif" font-size="13" fill="#374151">Continuous 0–3 s translation; current OBB orientation is held fixed.</text>',
            '<text x="30" y="742" font-family="sans-serif" font-size="12" fill="#6b7280">Read-only diagnostic: no prediction, threshold, lifecycle, or gate changed.</text>',
        ]
    )
    return '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="760" viewBox="0 0 760 760">' + "".join(elements) + "</svg>"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8", newline="")
    os.replace(partial, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    fields = [
        "kind",
        "id",
        "label_id",
        "classification",
        "truth_positive",
        "point_hit",
        "footprint_hit",
        "frozen_m1_point_risk",
        "point_minimum_clearance_m",
        "footprint_minimum_clearance_m",
        "point_entry_s",
        "footprint_entry_s",
    ]
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    m1_path = args.m1_result.resolve(strict=True)
    output_path = args.output.resolve()
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    require(
        m1.get("status") == "DTR_M1_O_CAUSAL_POINT_VELOCITY_ORACLE_COMPLETE",
        "m1_status_drift",
    )
    require(
        m1["gate"]["verdict"]
        == "DTR_M1_O_POINT_VELOCITY_ORACLE_CEILING_NOT_MET_CLOSE_SCENE_FLOW_ROUTE",
        "m1_terminal_drift",
    )
    source = m1["source"]
    known_tracks_path = Path(source["known_height_tracks"]).resolve(strict=True)
    labels_path = Path(source["labels"]).resolve(strict=True)
    timestamps_path = Path(source["timestamps"]).resolve(strict=True)
    bag_path = Path(source["bag"]).resolve(strict=True)
    require(sha256_file(known_tracks_path) == source["known_height_tracks_sha256"], "known_tracks_hash_drift")
    require(sha256_file(labels_path) == source["labels_sha256"], "labels_hash_drift")
    require(sha256_file(timestamps_path) == source["timestamps_sha256"], "timestamps_hash_drift")
    require(sha256_file(bag_path) == source["bag_sha256"], "bag_hash_drift")

    oracle_path, oracle_manifest_path = ledger_paths(m1_path)
    ledger = load_oracle_ledger(oracle_path, oracle_manifest_path)
    require(ledger.manifest["ledger_sha256"] == m1["oracle_ledger"]["ledger_sha256"], "m1_ledger_identity_drift")
    timestamps = load_image_timestamps(timestamps_path)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    causal_frame_poses = {
        frame: _causal_pose(poses, round(timestamps[frame] * 1e9))
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    boxes_by_frame = load_native_boxes(labels_path, timestamps, causal_frame_poses)
    labels = sorted({box.label_id for boxes in boxes_by_frame.values() for box in boxes})
    index_to_label = dict(enumerate(labels))
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
    case_by_key = {(case.label_id, case.segment_index): case for case in cases}

    dropout_rows = []
    figure_candidates: list[tuple[float, dict[str, Any], str]] = []
    for duration, duration_result in m1["stress_by_duration_s"].items():
        for trial in duration_result["by_trial"]:
            if trial["dropout_window_alerted"]:
                continue
            case = next(
                case
                for case in cases
                if case.label_id == trial["label_id"]
                and any(sample.frame_index == trial["contact_frame"] for sample in case.samples)
            )
            samples_by_frame = {sample.frame_index: sample for sample in case.samples}
            frame_rows = [
                _frame_geometry(
                    sample=samples_by_frame[int(frame)],
                    target_label_id=case.label_id,
                    ledger=ledger,
                    boxes_by_frame=boxes_by_frame,
                    index_to_label=index_to_label,
                )
                for frame in trial["dropout_frames"]
            ]
            row = _aggregate_geometry(
                kind="DROPOUT_MISS_TRIAL",
                identifier=f"{trial['label_id']}@{duration}s",
                truth_positive=True,
                frames=frame_rows,
                metadata={
                    "label_id": trial["label_id"],
                    "duration_s": float(duration),
                    "category": trial["category"],
                    "event_start_frame": trial["event_start_frame"],
                    "contact_frame": trial["contact_frame"],
                    "dropout_frames": trial["dropout_frames"],
                },
            )
            dropout_rows.append(row)
            for frame_row in frame_rows:
                clearance = frame_row["footprint_minimum_clearance_m"]
                if clearance is not None:
                    outcome = (
                        "footprint hits route"
                        if frame_row["footprint_hit"]
                        else "footprint also misses route"
                    )
                    figure_candidates.append(
                        (
                            float(clearance),
                            frame_row,
                            f"M2-D · {trial['label_id']} · frame {frame_row['frame']} · {outcome}",
                        )
                    )

    false_rows = []
    for source_row in m1["motion_source_false_delta"]["rows"]:
        if source_row["provenance"] == PROVENANCE_INHERITED:
            continue
        case = case_by_key[(source_row["label_id"], int(source_row["target_segment_index"]))]
        baseline = _base_predictions(case)
        oracle = run_flow_arm(case, set(), ledger).predictions
        matching_segment = next(
            segment
            for segment in _false_segments(case, oracle)
            if case.samples[segment.start_index].frame_index == source_row["first_frame"]
            and case.samples[segment.end_index].frame_index == source_row["last_frame"]
        )
        point_only_indices = [
            index
            for index in range(matching_segment.start_index, matching_segment.end_index + 1)
            if oracle[index].raw_alert is True and baseline[index].raw_alert is not True
        ]
        require(len(point_only_indices) == source_row["point_velocity_only_frames"], "false_segment_replay_drift")
        frame_rows = [
            _frame_geometry(
                sample=case.samples[index],
                target_label_id=case.label_id,
                ledger=ledger,
                boxes_by_frame=boxes_by_frame,
                index_to_label=index_to_label,
            )
            for index in point_only_indices
        ]
        false_rows.append(
            _aggregate_geometry(
                kind="M1_NEW_OR_MODIFIED_FALSE_SEGMENT",
                identifier=(
                    f"{source_row['label_id']}:{source_row['target_segment_index']}"
                    f"@{source_row['first_frame']}-{source_row['last_frame']}"
                ),
                truth_positive=False,
                frames=frame_rows,
                metadata={
                    "label_id": source_row["label_id"],
                    "target_segment_index": source_row["target_segment_index"],
                    "first_frame": source_row["first_frame"],
                    "last_frame": source_row["last_frame"],
                    "provenance": source_row["provenance"],
                    "point_velocity_only_frames": source_row["point_velocity_only_frames"],
                },
            )
        )

    all_rows = dropout_rows + false_rows
    dropout_counts = Counter(row["classification"] for row in dropout_rows)
    false_counts = Counter(row["classification"] for row in false_rows)
    all_dropout_misses_are_extent_gap = (
        len(dropout_rows) == 3
        and all(row["classification"] == "POINT_MISS_FOOTPRINT_HIT" for row in dropout_rows)
    )
    fresh_m2_o_eligible = all_dropout_misses_are_extent_gap
    csv_path = output_path.with_name(output_path.stem + ".extent-gap.csv")
    svg_path = output_path.with_name(output_path.stem + ".extent-gap.svg")
    _write_csv(csv_path, all_rows)
    if figure_candidates:
        _clearance, frame, title = min(figure_candidates, key=lambda item: item[0])
        _atomic_text(svg_path, _figure_svg(frame, title))
        figure_status = "WRITTEN"
    else:
        figure_status = "NOT_EVALUABLE_NO_ATTRIBUTED_NATIVE_FOOTPRINT"

    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_ceiling": CLAIM_CEILING,
        "question": (
            "Are the three M1 dropout-window misses and five M1-induced/modified false "
            "segments explained by zero-radius point trajectories versus continuous native footprints?"
        ),
        "frozen": {
            "motion_source": "exact sealed M1 native-box point-velocity ledger",
            "route_radius_m": ROUTE_HALF_WIDTH_M,
            "horizon_s": HORIZON_S,
            "native_footprint_orientation": "current-frame yaw held fixed during translation",
            "predictions_thresholds_lifecycle_evaluator_gate": "READ_ONLY_UNCHANGED",
        },
        "geometry": {
            "point": "continuous p + v*t minimum distance and circle entry over [0,3] s",
            "footprint": (
                "native current OBB translated by robust median component velocity; analytic "
                "continuous Minkowski entry against the 0.65 m route disk"
            ),
            "association": "exact frozen M1 target-radius cell attribution, then grouped by M1 native component_id",
            "closing_gate": (
                "classification uses pure geometric contact; frozen M1 closing-gated entry is reported separately"
            ),
        },
        "source": {
            "dataset": "JRDB public train split",
            "sequence": m1["source"]["sequence"],
            "window": m1["source"]["window"],
            "m1_result": str(m1_path),
            "m1_result_sha256": sha256_file(m1_path),
            "oracle_ledger": str(oracle_path),
            "oracle_ledger_sha256": sha256_file(oracle_path),
            "oracle_manifest": str(oracle_manifest_path),
            "oracle_manifest_sha256": sha256_file(oracle_manifest_path),
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
            "native_truth_use": "current native OBB supplies privileged footprint only",
            "future_truth_use": "event truth labels classification only; no future box enters geometry",
            "consumed_cohort": True,
            "geometry_quality": geometry_quality,
        },
        "dropout_miss_trials": dropout_rows,
        "m1_new_or_modified_false_segments": false_rows,
        "summary": {
            "dropout_miss_trials": len(dropout_rows),
            "dropout_miss_unique_events": len({row["label_id"] for row in dropout_rows}),
            "dropout_classification_counts": dict(sorted(dropout_counts.items())),
            "false_segments": len(false_rows),
            "false_classification_counts": dict(sorted(false_counts.items())),
            "all_three_dropout_misses_are_point_miss_footprint_hit": all_dropout_misses_are_extent_gap,
            "representation_gap_supported": all_dropout_misses_are_extent_gap,
            "fresh_m2_o_eligible": fresh_m2_o_eligible,
            "scene_flow_estimator_competition_closed": True,
            "r8_closed": True,
        },
        "decision": (
            "FREEZE_SEPARATE_FRESH_M2_O_SWEPT_FOOTPRINT_ORACLE"
            if fresh_m2_o_eligible
            else "DTR_M2_D_EXTENT_GAP_NOT_SUPPORTED_NO_FRESH_M2_O"
        ),
        "artifacts": {
            "rows_csv": str(csv_path),
            "rows_csv_sha256": sha256_file(csv_path),
            "figure_svg": str(svg_path) if figure_status == "WRITTEN" else None,
            "figure_svg_sha256": sha256_file(svg_path) if figure_status == "WRITTEN" else None,
            "figure_status": figure_status,
        },
        "limitations": [
            "This is read-only post hoc diagnosis on the already consumed M1 Development cohort.",
            "The three dropout trials are three durations of one pedestrian:35 event, not independent events.",
            "Native label-derived OBBs and component identity are privileged oracle information.",
            "The OBB translates at robust median M1 component velocity with current yaw fixed; no rotation forecast is added.",
            "A supported gap only authorizes freezing a source-disjoint fresh M2-O protocol; it is not M2-O evidence.",
            "No TeFlow, DeltaFlow, R8, route forecasting, Android, product, user-benefit, or safety claim follows.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    print(json.dumps({"decision": result["decision"], **result["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
