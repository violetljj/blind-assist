"""Run CODa's privileged static and vertical route-risk ceiling.

This companion to the dynamic DTR ceiling consumes only source-native 3-D
boxes, persistent IDs, timestamps, dense poses, and the OS1-to-base
calibration.  It asks whether route conditioning and a bounded body-clearance
band can suppress nearby but non-actionable static geometry.

Future boxes remain evaluator-only truth.  The base frame is a calibrated
vertical proxy, not an independently measured terrain plane, so this does not
establish drop-off or thin-branch perception.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from coda_native_ceiling import (
    GroupAccumulator,
    PoseRow,
    read_poses,
    score_groups,
    sha256_file,
)
from dtr_r0 import Arm, CausalFrame, DTRConfig, EgoPose, Observation, Prediction, Signal, run_arm
from dtr_r1 import RiskEventLifecycle
from jrdb_native_ceiling import ArmAccumulator, ratio, score_arm


SCHEMA = "dtr-static-coda-native-box-ceiling-v2"
CLAIM_CEILING = "PUBLIC_REAL_PRIVILEGED_STATIC_NATIVE_BOX_POSE_CALIBRATION_CEILING_ONLY"
HORIZON_S = 3.0
ROUTE_HALF_WIDTH_M = 0.65
MINIMUM_HISTORY_S = 0.20
TIMESTAMP_TOLERANCE_S = 0.05
MAXIMUM_CONTIGUOUS_GAP_S = 0.15
LOWER_BODY_MAXIMUM_M = 1.35
HEAD_MINIMUM_M = 1.35
HEAD_MAXIMUM_M = 2.10
GROUND_TOLERANCE_M = 0.12

PROXIMITY_ARM = "P0_proximity_3m"
ROUTE_XY_ARM = "S1_route_xy"
ROUTE_VERTICAL_ARM = "S2_route_vertical"
CURVED_ROUTE_ARM = "S3_curved_route_vertical"
CONTINUOUS_CURVED_ROUTE_ARM = "S4_continuous_curved_route_vertical"
ROUTE_MOTION_HISTORY_S = 0.50
ROUTE_SAMPLE_STEP_S = 0.10

STATIC_CLASS_GROUP = {
    "Tree": "vegetation",
    "Canopy": "vegetation",
    "Freestanding Plant": "vegetation",
    "Pole": "fixed_structure",
    "Traffic Sign": "fixed_structure",
    "Traffic Light": "fixed_structure",
    "Informational Sign": "fixed_structure",
    "Wall Sign": "fixed_structure",
    "Bike Rack": "barrier_boundary",
    "Bollard": "barrier_boundary",
    "Construction Barrier": "barrier_boundary",
    "Door": "barrier_boundary",
    "Fence": "barrier_boundary",
    "Railing": "barrier_boundary",
    "Traffic Arm": "barrier_boundary",
    "Cone": "temporary_obstacle",
    "Chair": "temporary_obstacle",
    "Bench": "temporary_obstacle",
    "Table": "temporary_obstacle",
    "Trash Can": "temporary_obstacle",
    "Floor Sign": "temporary_obstacle",
    "Stanchion": "temporary_obstacle",
    "Dumpster": "temporary_obstacle",
    "Water Fountain": "temporary_obstacle",
}


@dataclass(frozen=True)
class StaticSample:
    frame_index: int
    time_s: float
    forward_m: float
    left_m: float
    relative_world_x_m: float
    relative_world_y_m: float
    radius_m: float
    length_m: float
    width_m: float
    yaw_rad: float
    ego_x_m: float
    ego_y_m: float
    ego_yaw_rad: float
    bottom_height_m: float
    top_height_m: float
    class_name: str
    object_group: str

    @property
    def distance_m(self) -> float:
        return math.hypot(self.forward_m, self.left_m)

    @property
    def tube_threshold_m(self) -> float:
        return ROUTE_HALF_WIDTH_M + self.radius_m

    @property
    def vertical_kind(self) -> str | None:
        lower = self.top_height_m > GROUND_TOLERANCE_M and self.bottom_height_m < LOWER_BODY_MAXIMUM_M
        head = self.top_height_m >= HEAD_MINIMUM_M and self.bottom_height_m <= HEAD_MAXIMUM_M
        if lower and head:
            return "full_body"
        if lower:
            return "lower_body"
        if head:
            return "head_clearance"
        return None

    def footprint_clearance_m(self) -> float:
        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        delta_x = -self.forward_m
        delta_y = -self.left_m
        box_x = delta_x * cosine + delta_y * sine
        box_y = -delta_x * sine + delta_y * cosine
        outside_x = max(0.0, abs(box_x) - self.length_m / 2.0)
        outside_y = max(0.0, abs(box_y) - self.width_m / 2.0)
        return math.hypot(outside_x, outside_y)


@dataclass(frozen=True)
class StaticTruthEvent:
    start_index: int
    end_index: int
    contact_index: int
    category: str
    object_group: str
    class_name: str


def read_extrinsic(path: Path) -> list[list[float]]:
    match = re.search(r"data:\s*\[(.*?)\]", path.read_text(encoding="utf-8"), re.DOTALL)
    if match is None:
        raise ValueError(f"missing extrinsic matrix in {path}")
    values = [float(item) for item in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(1))]
    if len(values) != 16:
        raise ValueError(f"expected 4x4 extrinsic matrix in {path}")
    return [values[index : index + 4] for index in range(0, 16, 4)]


def rotation_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def calibrated_height_interval(box: dict[str, Any], extrinsic: list[list[float]]) -> tuple[float, float]:
    rotation = rotation_matrix(float(box["r"]), float(box["p"]), float(box["y"]))
    center = (float(box["cX"]), float(box["cY"]), float(box["cZ"]))
    half = (float(box["l"]) / 2.0, float(box["w"]) / 2.0, float(box["h"]) / 2.0)
    vertical_sign = 1.0 if extrinsic[2][2] >= 0.0 else -1.0
    heights = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = (sx * half[0], sy * half[1], sz * half[2])
                point = tuple(
                    center[row] + sum(rotation[row][column] * local[column] for column in range(3))
                    for row in range(3)
                )
                base_z = sum(extrinsic[2][column] * point[column] for column in range(3)) + extrinsic[2][3]
                heights.append(vertical_sign * base_z)
    return min(heights), max(heights)


def read_sequence(
    dataset_root: Path,
    sequence: str,
) -> tuple[dict[str, list[StaticSample]], dict[str, Any], dict[str, str]]:
    bbox_dir = dataset_root / "3d_bbox" / "os1" / sequence
    timestamp_path = dataset_root / "timestamps" / f"{sequence}.txt"
    global_pose_path = dataset_root / "poses" / "dense_global" / f"{sequence}.txt"
    local_pose_path = dataset_root / "poses" / "dense" / f"{sequence}.txt"
    metadata_path = dataset_root / "metadata" / f"{sequence}.json"
    calibration_path = dataset_root / "calibrations" / sequence / "calib_os1_to_base.yaml"
    pose_path = global_pose_path if global_pose_path.exists() else local_pose_path
    required = (bbox_dir, timestamp_path, pose_path, metadata_path, calibration_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing CODa static inputs: {missing}")
    timestamps = [
        float(line)
        for line in timestamp_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    poses: list[PoseRow] = read_poses(pose_path)
    if len(poses) != len(timestamps):
        raise ValueError("CODa dense pose and timestamp counts differ")
    extrinsic = read_extrinsic(calibration_path)

    tracks: dict[str, list[StaticSample]] = {}
    class_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    vertical_counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    label_frames = 0
    objects_seen = 0
    objects_used = 0
    pose_timestamp_mismatches = 0
    for path in sorted(
        bbox_dir.glob("*.json"),
        key=lambda item: int(item.stem.rsplit("_", 1)[-1]),
    ):
        frame_index = int(path.stem.rsplit("_", 1)[-1])
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        payload = json.loads(raw)
        label_frames += 1
        time_s = timestamps[frame_index]
        pose = poses[frame_index]
        if abs(pose.time_s - time_s) > TIMESTAMP_TOLERANCE_S:
            pose_timestamp_mismatches += 1
        cosine = math.cos(pose.yaw_rad)
        sine = math.sin(pose.yaw_rad)
        for box in payload.get("3dbbox", []):
            objects_seen += 1
            class_name = str(box["classId"])
            object_group = STATIC_CLASS_GROUP.get(class_name)
            if object_group is None:
                continue
            class_counts[class_name] += 1
            forward_m = float(box["cX"])
            left_m = float(box["cY"])
            length_m = float(box["l"])
            width_m = float(box["w"])
            yaw_rad = float(box["y"])
            bottom_height_m, top_height_m = calibrated_height_interval(box, extrinsic)
            if not all(
                math.isfinite(value)
                for value in (
                    forward_m,
                    left_m,
                    length_m,
                    width_m,
                    yaw_rad,
                    bottom_height_m,
                    top_height_m,
                )
            ) or min(length_m, width_m) <= 0.0:
                continue
            sample = StaticSample(
                frame_index=frame_index,
                time_s=time_s,
                forward_m=forward_m,
                left_m=left_m,
                relative_world_x_m=forward_m * cosine - left_m * sine,
                relative_world_y_m=forward_m * sine + left_m * cosine,
                radius_m=max(0.15, 0.5 * math.hypot(length_m, width_m)),
                length_m=length_m,
                width_m=width_m,
                yaw_rad=yaw_rad,
                ego_x_m=pose.x_m,
                ego_y_m=pose.y_m,
                ego_yaw_rad=pose.yaw_rad,
                bottom_height_m=bottom_height_m,
                top_height_m=top_height_m,
                class_name=class_name,
                object_group=object_group,
            )
            tracks.setdefault(str(box["instanceId"]), []).append(sample)
            objects_used += 1
            group_counts[object_group] += 1
            vertical_counts[sample.vertical_kind or "nonactionable_vertical"] += 1
    for samples in tracks.values():
        samples.sort(key=lambda item: (item.frame_index, item.time_s))
    return (
        tracks,
        {
            "label_frames": label_frames,
            "objects_seen": objects_seen,
            "static_objects_used": objects_used,
            "native_static_identities": len(tracks),
            "class_box_counts": dict(sorted(class_counts.items())),
            "static_group_box_counts": dict(sorted(group_counts.items())),
            "vertical_box_counts": dict(sorted(vertical_counts.items())),
            "pose_timestamp_mismatch_frames": pose_timestamp_mismatches,
            "pose_authority": "dense_global" if pose_path == global_pose_path else "dense_local",
        },
        {
            "bbox_canonical_stream_sha256": digest.hexdigest(),
            "timestamps_sha256": sha256_file(timestamp_path),
            "poses_sha256": sha256_file(pose_path),
            "metadata_sha256": sha256_file(metadata_path),
            "calibration_sha256": sha256_file(calibration_path),
        },
    )


def contiguous_segments(samples: Sequence[StaticSample]) -> Iterable[list[StaticSample]]:
    current: list[StaticSample] = []
    for sample in samples:
        if current and (
            sample.frame_index != current[-1].frame_index + 1
            or sample.time_s <= current[-1].time_s
            or sample.time_s - current[-1].time_s > MAXIMUM_CONTIGUOUS_GAP_S
        ):
            yield current
            current = []
        current.append(sample)
    if current:
        yield current


def causal_frames(track_id: str, samples: Sequence[StaticSample]) -> list[CausalFrame]:
    origin = samples[0].time_s
    pose = EgoPose(0.0, 0.0, 0.0, 0.0)
    return [
        CausalFrame(
            time_s=sample.time_s - origin,
            ego_pose=pose,
            observations=(
                Observation(
                    track_id=track_id,
                    forward_m=sample.relative_world_x_m,
                    left_m=sample.relative_world_y_m,
                    radius_m=sample.radius_m,
                ),
            ),
            person_detection_count=1,
        )
        for sample in samples
    ]


def future_hits(
    samples: Sequence[StaticSample],
) -> tuple[list[bool | None], list[int | None]]:
    truth: list[bool | None] = []
    contacts: list[int | None] = []
    final_time = samples[-1].time_s
    for index, sample in enumerate(samples):
        hit: int | None = None
        future_index = index
        while (
            future_index < len(samples)
            and samples[future_index].time_s - sample.time_s <= HORIZON_S + 1e-9
        ):
            future = samples[future_index]
            if future.vertical_kind is not None and future.footprint_clearance_m() <= ROUTE_HALF_WIDTH_M:
                hit = future_index
                break
            future_index += 1
        has_full_future = final_time - sample.time_s >= HORIZON_S - TIMESTAMP_TOLERANCE_S
        truth.append(True if hit is not None else False if has_full_future else None)
        contacts.append(hit)
    return truth, contacts


def truth_events(
    samples: Sequence[StaticSample],
    truth: Sequence[bool | None],
    contacts: Sequence[int | None],
) -> tuple[list[StaticTruthEvent], list[bool]]:
    known = [
        value is not None and sample.time_s - samples[0].time_s + 1e-9 >= MINIMUM_HISTORY_S
        for sample, value in zip(samples, truth)
    ]
    events: list[StaticTruthEvent] = []
    index = 0
    while index < len(samples):
        if not known[index] or truth[index] is not True:
            index += 1
            continue
        if index == 0 or not known[index - 1]:
            while index < len(samples) and known[index] and truth[index] is True:
                index += 1
            continue
        if truth[index - 1] is True:
            index += 1
            continue
        start = index
        while index + 1 < len(samples) and known[index + 1] and truth[index + 1] is True:
            index += 1
        end = index
        contact = contacts[start]
        if contact is not None:
            hit = samples[contact]
            events.append(
                StaticTruthEvent(
                    start_index=start,
                    end_index=end,
                    contact_index=contact,
                    category=hit.vertical_kind or "unknown_vertical",
                    object_group=hit.object_group,
                    class_name=hit.class_name,
                )
            )
        index += 1
    return events, known


def height_gate(
    samples: Sequence[StaticSample],
    route_predictions: Sequence[Prediction],
    config: DTRConfig,
) -> list[Prediction]:
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    output = []
    for sample, route in zip(samples, route_predictions):
        raw_alert = (
            None
            if route.raw_alert is None
            else bool(route.raw_alert and sample.vertical_kind is not None)
        )
        future_s = route.diagnostic.get("future_s")
        urgent = bool(
            raw_alert
            and isinstance(future_s, (int, float))
            and float(future_s) <= HORIZON_S / 2.0 + 1e-9
        )
        output.append(
            Prediction(
                time_s=route.time_s,
                signal=lifecycle.update(route.time_s, raw_alert, urgent=urgent),
                raw_alert=raw_alert,
                reason=(
                    "route_and_vertical_clearance_intersection"
                    if raw_alert
                    else "route_or_vertical_clearance_nonintersection"
                    if raw_alert is not None
                    else "insufficient_causal_track"
                ),
                track_id=route.track_id,
                diagnostic={
                    **route.diagnostic,
                    "vertical_kind": sample.vertical_kind or "nonactionable_vertical",
                    "bottom_height_m": sample.bottom_height_m,
                    "top_height_m": sample.top_height_m,
                },
            )
        )
    return output


def fitted_slope(times: Sequence[float], values: Sequence[float]) -> float | None:
    if len(times) < 2 or times[-1] - times[0] < MINIMUM_HISTORY_S:
        return None
    mean_time = sum(times) / len(times)
    denominator = sum((time_s - mean_time) ** 2 for time_s in times)
    if denominator <= 1e-12:
        return None
    mean_value = sum(values) / len(values)
    return sum(
        (time_s - mean_time) * (value - mean_value)
        for time_s, value in zip(times, values)
    ) / denominator


def unwrapped_yaws(samples: Sequence[StaticSample]) -> list[float]:
    output = [samples[0].ego_yaw_rad]
    for sample in samples[1:]:
        delta = (sample.ego_yaw_rad - output[-1] + math.pi) % (2.0 * math.pi) - math.pi
        output.append(output[-1] + delta)
    return output


def point_to_box_clearance(
    point_x: float,
    point_y: float,
    box_x: float,
    box_y: float,
    box_yaw: float,
    length_m: float,
    width_m: float,
) -> float:
    cosine = math.cos(box_yaw)
    sine = math.sin(box_yaw)
    delta_x = point_x - box_x
    delta_y = point_y - box_y
    box_forward = delta_x * cosine + delta_y * sine
    box_lateral = -delta_x * sine + delta_y * cosine
    outside_forward = max(0.0, abs(box_forward) - length_m / 2.0)
    outside_lateral = max(0.0, abs(box_lateral) - width_m / 2.0)
    return math.hypot(outside_forward, outside_lateral)


def segment_to_box_entry_fraction(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    box_x: float,
    box_y: float,
    box_yaw: float,
    length_m: float,
    width_m: float,
    clearance_m: float,
) -> float | None:
    """Earliest continuous segment entry into an oriented box swept by a disk."""

    cosine = math.cos(box_yaw)
    sine = math.sin(box_yaw)

    def box_local(point_x: float, point_y: float) -> tuple[float, float]:
        delta_x = point_x - box_x
        delta_y = point_y - box_y
        return (
            delta_x * cosine + delta_y * sine,
            -delta_x * sine + delta_y * cosine,
        )

    start_forward, start_lateral = box_local(start_x, start_y)
    end_forward, end_lateral = box_local(end_x, end_y)
    velocity_forward = end_forward - start_forward
    velocity_lateral = end_lateral - start_lateral
    half_length = length_m / 2.0
    half_width = width_m / 2.0

    breakpoints = [0.0, 1.0]
    for start, velocity, half_extent in (
        (start_forward, velocity_forward, half_length),
        (start_lateral, velocity_lateral, half_width),
    ):
        if abs(velocity) <= 1e-15:
            continue
        for boundary in (-half_extent, half_extent):
            fraction = (boundary - start) / velocity
            if 0.0 < fraction < 1.0:
                breakpoints.append(fraction)
    breakpoints = sorted(set(breakpoints))

    def outside_coefficients(
        start: float,
        velocity: float,
        half_extent: float,
        midpoint: float,
    ) -> tuple[float, float]:
        value = start + velocity * midpoint
        if value > half_extent:
            return start - half_extent, velocity
        if value < -half_extent:
            return start + half_extent, velocity
        return 0.0, 0.0

    radius_squared = clearance_m * clearance_m
    tolerance = 1e-12
    for interval_start, interval_end in zip(breakpoints, breakpoints[1:]):
        midpoint = (interval_start + interval_end) / 2.0
        forward_offset, forward_slope = outside_coefficients(
            start_forward,
            velocity_forward,
            half_length,
            midpoint,
        )
        lateral_offset, lateral_slope = outside_coefficients(
            start_lateral,
            velocity_lateral,
            half_width,
            midpoint,
        )
        quadratic = forward_slope**2 + lateral_slope**2
        linear = 2.0 * (
            forward_offset * forward_slope
            + lateral_offset * lateral_slope
        )
        constant = (
            forward_offset**2 + lateral_offset**2 - radius_squared
        )

        def value_at(fraction: float) -> float:
            return quadratic * fraction * fraction + linear * fraction + constant

        if value_at(interval_start) <= tolerance:
            return interval_start
        if quadratic <= tolerance:
            continue
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant < -tolerance:
            continue
        root = (-linear - math.sqrt(max(0.0, discriminant))) / (2.0 * quadratic)
        candidate = max(interval_start, root)
        if candidate <= interval_end + tolerance and value_at(candidate) <= tolerance:
            return min(1.0, max(0.0, candidate))
    return None


def ctrv_route_points(
    current: StaticSample,
    forward_speed: float,
    yaw_rate: float,
) -> list[tuple[float, float, float]]:
    points = []
    step_count = round(HORIZON_S / ROUTE_SAMPLE_STEP_S)
    for step in range(step_count + 1):
        future_s = step * ROUTE_SAMPLE_STEP_S
        if abs(yaw_rate) <= 1e-6:
            ego_x = current.ego_x_m + forward_speed * future_s * math.cos(
                current.ego_yaw_rad
            )
            ego_y = current.ego_y_m + forward_speed * future_s * math.sin(
                current.ego_yaw_rad
            )
        else:
            future_yaw = current.ego_yaw_rad + yaw_rate * future_s
            ego_x = current.ego_x_m + forward_speed / yaw_rate * (
                math.sin(future_yaw) - math.sin(current.ego_yaw_rad)
            )
            ego_y = current.ego_y_m - forward_speed / yaw_rate * (
                math.cos(future_yaw) - math.cos(current.ego_yaw_rad)
            )
        points.append((future_s, ego_x, ego_y))
    return points


def curved_route_predictions(
    samples: Sequence[StaticSample],
    config: DTRConfig,
    *,
    continuous_collision: bool = False,
) -> list[Prediction]:
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    output: list[Prediction] = []
    history_start = 0
    for index, current in enumerate(samples):
        cutoff = current.time_s - ROUTE_MOTION_HISTORY_S
        while history_start < index and samples[history_start].time_s < cutoff:
            history_start += 1
        history = samples[history_start : index + 1]
        times = [sample.time_s for sample in history]
        velocity_x = fitted_slope(times, [sample.ego_x_m for sample in history])
        velocity_y = fitted_slope(times, [sample.ego_y_m for sample in history])
        yaw_rate = fitted_slope(times, unwrapped_yaws(history)) if history else None
        if velocity_x is None or velocity_y is None or yaw_rate is None:
            raw_alert: bool | None = None
            first_entry_s: float | None = None
        else:
            forward_speed = max(
                0.0,
                velocity_x * math.cos(current.ego_yaw_rad)
                + velocity_y * math.sin(current.ego_yaw_rad),
            )
            box_x = (
                current.ego_x_m
                + current.forward_m * math.cos(current.ego_yaw_rad)
                - current.left_m * math.sin(current.ego_yaw_rad)
            )
            box_y = (
                current.ego_y_m
                + current.forward_m * math.sin(current.ego_yaw_rad)
                + current.left_m * math.cos(current.ego_yaw_rad)
            )
            box_yaw = current.ego_yaw_rad + current.yaw_rad
            first_entry_s = None
            route_points = ctrv_route_points(current, forward_speed, yaw_rate)
            if continuous_collision:
                for left, right in zip(route_points, route_points[1:]):
                    entry_fraction = segment_to_box_entry_fraction(
                        left[1],
                        left[2],
                        right[1],
                        right[2],
                        box_x,
                        box_y,
                        box_yaw,
                        current.length_m,
                        current.width_m,
                        ROUTE_HALF_WIDTH_M,
                    )
                    if entry_fraction is not None:
                        first_entry_s = left[0] + entry_fraction * (
                            right[0] - left[0]
                        )
                        break
            else:
                for future_s, ego_x, ego_y in route_points:
                    if point_to_box_clearance(
                        ego_x,
                        ego_y,
                        box_x,
                        box_y,
                        box_yaw,
                        current.length_m,
                        current.width_m,
                    ) <= ROUTE_HALF_WIDTH_M:
                        first_entry_s = future_s
                        break
            raw_alert = bool(first_entry_s is not None and current.vertical_kind is not None)
        urgent = bool(first_entry_s is not None and first_entry_s <= HORIZON_S / 2.0 + 1e-9 and raw_alert)
        output.append(
            Prediction(
                time_s=current.time_s - samples[0].time_s,
                signal=lifecycle.update(
                    current.time_s - samples[0].time_s,
                    raw_alert,
                    urgent=urgent,
                ),
                raw_alert=raw_alert,
                reason=(
                    "causal_constant_turn_route_and_vertical_intersection"
                    if raw_alert
                    else "causal_constant_turn_route_nonintersection"
                    if raw_alert is not None
                    else "insufficient_causal_ego_motion"
                ),
                track_id=None,
                diagnostic={
                    "route_model": "causal_constant_turn_rate_and_velocity",
                    "collision_geometry": (
                        "continuous_piecewise_linear_time_of_impact"
                        if continuous_collision
                        else "discrete_route_sample_points"
                    ),
                    "route_history_s": ROUTE_MOTION_HISTORY_S,
                    "first_entry_s": first_entry_s if first_entry_s is not None else "none",
                    "vertical_kind": current.vertical_kind or "nonactionable_vertical",
                },
            )
        )
    return output


def evaluate_segment(
    track_id: str,
    samples: Sequence[StaticSample],
    config: DTRConfig,
) -> tuple[dict[str, ArmAccumulator], dict[str, GroupAccumulator], int, float]:
    if len(samples) < 2 or samples[-1].time_s - samples[0].time_s < MINIMUM_HISTORY_S + HORIZON_S:
        return {}, {}, 0, 0.0
    frames = causal_frames(track_id, samples)
    proximity = run_arm(frames, Arm.B1_DISTANCE, config)
    route = run_arm(frames, Arm.C_ROUTE_INTERSECTION, config)
    predictions = {
        PROXIMITY_ARM: proximity,
        ROUTE_XY_ARM: route,
        ROUTE_VERTICAL_ARM: height_gate(samples, route, config),
        CURVED_ROUTE_ARM: curved_route_predictions(samples, config),
        CONTINUOUS_CURVED_ROUTE_ARM: curved_route_predictions(
            samples,
            config,
            continuous_collision=True,
        ),
    }
    truth, contacts = future_hits(samples)
    events, known = truth_events(samples, truth, contacts)
    if not any(known):
        return {}, {}, 0, 0.0
    scored = {
        name: score_arm(
            samples,
            values,
            events,
            known,
            truth,
            clear_grace_s=config.clear_grace_s,
        )
        for name, values in predictions.items()
    }
    grouped = {
        name: score_groups(samples, values, events, known, truth)
        for name, values in predictions.items()
    }
    known_indices = [index for index, value in enumerate(known) if value]
    exposure_s = (
        samples[known_indices[-1]].time_s - samples[known_indices[0]].time_s
        if len(known_indices) > 1
        else 0.0
    )
    return scored, grouped, len(events), exposure_s


def compare(
    pooled: dict[str, dict[str, Any]],
    challenger: str,
    comparator: str,
) -> dict[str, Any]:
    left = pooled[challenger]
    right = pooled[comparator]
    recall_delta = left["critical_event_recall"] - right["critical_event_recall"]
    false_delta = right["false_alert_segments"] - left["false_alert_segments"]
    return {
        "status": "DOMINATES" if recall_delta >= -1e-12 and false_delta > 0 else "DOES_NOT_DOMINATE",
        "challenger": challenger,
        "comparator": comparator,
        "critical_recall_delta": recall_delta,
        "false_alert_segment_delta": false_delta,
        "false_alert_segment_reduction_fraction": ratio(false_delta, right["false_alert_segments"]),
        "clear_rate_delta": (
            None
            if left["clear_rate"] is None or right["clear_rate"] is None
            else left["clear_rate"] - right["clear_rate"]
        ),
    }


def parse_sequence_root(value: str) -> tuple[str, Path]:
    sequence, separator, root = value.partition("=")
    if not separator or not sequence or not root:
        raise argparse.ArgumentTypeError("expected SEQUENCE=DATASET_ROOT")
    return sequence, Path(root)


def evaluate(sequence_roots: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    config = DTRConfig(
        distance_gate_m=3.0,
        route_horizon_s=HORIZON_S,
        route_half_width_m=ROUTE_HALF_WIDTH_M,
        nominal_wearer_speed_mps=0.0,
    )
    arm_names = (
        PROXIMITY_ARM,
        ROUTE_XY_ARM,
        ROUTE_VERTICAL_ARM,
        CURVED_ROUTE_ARM,
        CONTINUOUS_CURVED_ROUTE_ARM,
    )
    pooled = {name: ArmAccumulator() for name in arm_names}
    pooled_groups = {name: GroupAccumulator() for name in arm_names}
    totals = Counter()
    sequence_results = []
    for sequence, dataset_root in sequence_roots:
        tracks, source, hashes = read_sequence(dataset_root, sequence)
        sequence_arms = {name: ArmAccumulator() for name in arm_names}
        sequence_groups = {name: GroupAccumulator() for name in arm_names}
        track_segments = 0
        evaluable_segments = 0
        critical_events = 0
        exposure_s = 0.0
        for track_id, samples in tracks.items():
            for segment_index, segment in enumerate(contiguous_segments(samples)):
                track_segments += 1
                scored, grouped, events, segment_exposure_s = evaluate_segment(
                    f"{sequence}/{track_id}/{segment_index}", segment, config
                )
                if not scored:
                    continue
                evaluable_segments += 1
                critical_events += events
                exposure_s += segment_exposure_s
                for name in arm_names:
                    sequence_arms[name].merge(scored[name])
                    sequence_groups[name].merge(grouped[name])
                    pooled[name].merge(scored[name])
                    pooled_groups[name].merge(grouped[name])
        sequence_results.append(
            {
                "sequence": sequence,
                "dataset_root": str(dataset_root.resolve()),
                "source": source,
                "input_hashes": hashes,
                "contiguous_track_segments": track_segments,
                "evaluable_track_segments": evaluable_segments,
                "critical_events": critical_events,
                "track_segment_exposure_s": exposure_s,
                "arms": {
                    name: {
                        **sequence_arms[name].to_dict(include_escalation=True),
                        "by_object_group": sequence_groups[name].to_dict(),
                    }
                    for name in arm_names
                },
            }
        )
        totals.update(
            sequences=1,
            label_frames=source["label_frames"],
            static_objects_used=source["static_objects_used"],
            native_static_identities=source["native_static_identities"],
            contiguous_track_segments=track_segments,
            evaluable_track_segments=evaluable_segments,
            critical_events=critical_events,
            track_segment_exposure_s=exposure_s,
        )
        print(
            f"CODa static {sequence}: events={critical_events}, "
            f"evaluable_tracks={evaluable_segments}, frames={source['label_frames']}"
        )
    pooled_dict = {
        name: {
            **pooled[name].to_dict(include_escalation=True),
            "by_object_group": pooled_groups[name].to_dict(),
        }
        for name in arm_names
    }
    return {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "source": {
            "dataset": "UT Campus Object Dataset (CODa)",
            "official_page": "https://amrl.cs.utexas.edu/coda/",
            "dataset_license": "CC BY-NC-SA 4.0 plus dataset terms",
            "acquisition": "HTTP Range extraction of native boxes/poses/timestamps/calibration only",
        },
        "protocol": {
            "truth": "future calibrated-height oriented box intersects actual future ego path capsule",
            "route_horizon_s": HORIZON_S,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "proximity_gate_m": config.distance_gate_m,
            "vertical_bands_m": {
                "ground_tolerance": GROUND_TOLERANCE_M,
                "lower_body_maximum": LOWER_BODY_MAXIMUM_M,
                "head_minimum": HEAD_MINIMUM_M,
                "head_maximum": HEAD_MAXIMUM_M,
            },
            "static_classes": dict(sorted(STATIC_CLASS_GROUP.items())),
            "base_height_authority": "source-native OS1-to-base calibration; not terrain ground truth",
            "curved_route_model": {
                "type": "causal_constant_turn_rate_and_velocity",
                "history_s": ROUTE_MOTION_HISTORY_S,
                "sample_step_s": ROUTE_SAMPLE_STEP_S,
                "sampled_arm_collision_geometry": "discrete route points",
                "continuous_arm_collision_geometry": (
                    "analytic line-segment time-of-impact against oriented box "
                    "Minkowski-expanded by route half width"
                ),
                "inputs": "current-and-past ego pose only",
            },
        },
        "totals": dict(totals),
        "pooled_arms": pooled_dict,
        "route_vertical_vs_proximity": compare(pooled_dict, ROUTE_VERTICAL_ARM, PROXIMITY_ARM),
        "vertical_vs_route_xy": compare(pooled_dict, ROUTE_VERTICAL_ARM, ROUTE_XY_ARM),
        "curved_route_vs_proximity": compare(pooled_dict, CURVED_ROUTE_ARM, PROXIMITY_ARM),
        "curved_route_vs_straight_route": compare(pooled_dict, CURVED_ROUTE_ARM, ROUTE_VERTICAL_ARM),
        "continuous_curved_route_vs_sampled_curved_route": compare(
            pooled_dict,
            CONTINUOUS_CURVED_ROUTE_ARM,
            CURVED_ROUTE_ARM,
        ),
        "sequences": sequence_results,
        "limitations": [
            "Privileged source-native boxes, identities, poses, and calibration are an algorithm ceiling.",
            "Robot base height is only a calibrated vertical proxy; terrain ground plane is not independently measured.",
            "Box topology cannot resolve thin branches, leaves, wall openings, or point-level free space.",
            "No positive head-clearance event occurs in the selected sequences; the bounded height path is implementation and negative-exposure evidence only.",
            "No drop-off truth, RGB/LiDAR detector, Android runtime, natural-distribution, or safety performance is established.",
            "CODa truth is sampled at source frames, so between-frame collision recovery is established only by the controlled geometry canary.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-root",
        action="append",
        required=True,
        type=parse_sequence_root,
        metavar="SEQUENCE=ROOT",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.sequence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["curved_route_vs_proximity"], indent=2))
    print(
        json.dumps(
            result["continuous_curved_route_vs_sampled_curved_route"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
