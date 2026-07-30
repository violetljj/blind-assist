from __future__ import annotations

import bisect
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from bindings import FrozenBundle
from contract import (
    BBOX_CLOSURE_ATOL_PER_S,
    PRIMARY_ARM,
    REFERENCE_ARM,
    SOURCE_CLOSURE_ATOL_MPS,
    ContractError,
    finite_values,
    median,
    raw_mad,
    type7_quantile,
)


VICON_TOPICS = {
    "sensor": "/vicon/event_lidar/event_lidar",
    "track-000": "/vicon/helmet_green/helmet_green",
    "track-001": "/vicon/helmet_yellow/helmet_yellow",
}
NULL_SOURCE_FIELDS = (
    "median_person_approach_component_mps",
    "median_sensor_approach_component_mps",
    "median_abs_person_approach_component_mps",
    "median_abs_sensor_approach_component_mps",
    "sensor_absolute_share",
    "median_camera_translation_speed_mps",
    "p90_camera_translation_speed_mps",
    "median_camera_angular_speed_radps",
    "p90_camera_angular_speed_radps",
)


class ProducerError(ContractError):
    pass


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _deadband_state(value: float) -> str:
    if value >= 0.1:
        return "approaching"
    if value <= -0.1:
        return "receding"
    return "quasi_static"


def _valid_pose(position: np.ndarray, quaternion: np.ndarray) -> bool:
    return bool(
        np.isfinite(position).all()
        and np.isfinite(quaternion).all()
        and np.linalg.norm(position) > 1e-9
        and np.linalg.norm(quaternion) > 1e-6
    )


def _rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not math.isfinite(float(norm)) or norm <= 1e-6:
        raise ProducerError("invalid quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _nearest_index(timestamps: np.ndarray, query_ns: int) -> int:
    right = int(np.searchsorted(timestamps, query_ns, side="left"))
    left = min(max(right - 1, 0), len(timestamps) - 1)
    right = min(max(right, 0), len(timestamps) - 1)
    if abs(query_ns - int(timestamps[left])) <= abs(int(timestamps[right]) - query_ns):
        return left
    return right


def _interval_valid(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    first: int,
    second: int,
) -> bool:
    if not _valid_pose(positions[first], quaternions[first]) or not _valid_pose(
        positions[second], quaternions[second]
    ):
        return False
    dt = (int(timestamps[second]) - int(timestamps[first])) / 1e9
    if not 0.005 <= dt <= 0.05:
        return False
    return bool(np.linalg.norm(positions[second] - positions[first]) / dt <= 5.0)


def source_pair_metrics(
    image_timestamp_ns: int,
    person: dict[str, np.ndarray],
    sensor: dict[str, np.ndarray],
    camera_from_marker: np.ndarray,
    frozen_signed_approach_mps: float,
) -> tuple[dict[str, float] | None, str | None]:
    person_t = person["timestamps_ns"]
    right = bisect.bisect_left(person_t, image_timestamp_ns)
    if right <= 0 or right >= len(person_t):
        return None, "PERSON_BRACKET_BOUNDARY"
    p0_index, p1_index = right - 1, right
    p0_t, p1_t = int(person_t[p0_index]), int(person_t[p1_index])
    if not _interval_valid(
        person_t,
        person["positions"],
        person["quaternions"],
        p0_index,
        p1_index,
    ):
        return None, "PERSON_CONTINUITY_REJECTED"

    sensor_t = sensor["timestamps_ns"]
    s0_index = _nearest_index(sensor_t, p0_t)
    s1_index = _nearest_index(sensor_t, p1_t)
    if (
        abs(int(sensor_t[s0_index]) - p0_t) > 20_000_000
        or abs(int(sensor_t[s1_index]) - p1_t) > 20_000_000
    ):
        return None, "PERSON_SENSOR_SYNC_REJECTED"
    index_delta = s1_index - s0_index
    if index_delta not in (0, 1):
        return None, "SENSOR_INDEX_DELTA_REJECTED"
    if index_delta == 1 and not _interval_valid(
        sensor_t,
        sensor["positions"],
        sensor["quaternions"],
        s0_index,
        s1_index,
    ):
        return None, "SENSOR_CONTINUITY_REJECTED"
    if not _valid_pose(
        sensor["positions"][s0_index], sensor["quaternions"][s0_index]
    ) or not _valid_pose(
        sensor["positions"][s1_index], sensor["quaternions"][s1_index]
    ):
        return None, "SENSOR_POSE_INVALID"

    dt = (p1_t - p0_t) / 1e9
    p0, p1 = person["positions"][p0_index], person["positions"][p1_index]
    s0, s1 = sensor["positions"][s0_index], sensor["positions"][s1_index]
    r0, r1 = p0 - s0, p1 - s1
    denominator = float(np.linalg.norm(r0) + np.linalg.norm(r1))
    if denominator <= 1e-12:
        return None, "RANGE_GEOMETRY_DEGENERATE"
    gradient = (r0 + r1) / denominator
    person_rr = float(np.dot(gradient, (p1 - p0) / dt))
    sensor_rr = float(-np.dot(gradient, (s1 - s0) / dt))
    person_approach = -person_rr
    sensor_approach = -sensor_rr
    signed_approach = -(person_rr + sensor_rr)
    if abs(signed_approach - float(frozen_signed_approach_mps)) > SOURCE_CLOSURE_ATOL_MPS:
        raise ProducerError("source signed-approach closure mismatch")
    if _deadband_state(signed_approach) != _deadband_state(
        float(frozen_signed_approach_mps)
    ):
        raise ProducerError("source deadband closure mismatch")

    share_denominator = abs(sensor_approach) + abs(person_approach)
    share = (
        abs(sensor_approach) / share_denominator
        if share_denominator >= 1e-6
        else None
    )
    transform = np.asarray(camera_from_marker, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ProducerError("invalid T_v_c calibration")
    r_s0, r_s1 = _rotation_matrix(sensor["quaternions"][s0_index]), _rotation_matrix(
        sensor["quaternions"][s1_index]
    )
    camera_center0 = s0 + r_s0 @ transform[:3, 3]
    camera_center1 = s1 + r_s1 @ transform[:3, 3]
    camera_translation = float(np.linalg.norm(camera_center1 - camera_center0) / dt)
    q0 = sensor["quaternions"][s0_index].astype(np.float64, copy=True)
    q1 = sensor["quaternions"][s1_index].astype(np.float64, copy=True)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    half_angle_cosine = min(1.0, max(-1.0, float(np.dot(q0, q1))))
    camera_angular = 2.0 * math.acos(half_angle_cosine) / dt
    return {
        "person_approach_component_mps": person_approach,
        "sensor_approach_component_mps": sensor_approach,
        "signed_approach_mps": signed_approach,
        "sensor_absolute_share": share,
        "camera_translation_speed_mps": camera_translation,
        "camera_angular_speed_radps": camera_angular,
    }, None


def read_vicon_tracks(bag_path: Path) -> dict[str, dict[str, np.ndarray]]:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    output: dict[str, dict[str, np.ndarray]] = {}
    with Reader(bag_path) as reader:
        for name, topic in VICON_TOPICS.items():
            info = reader.topics.get(topic)
            if info is None or len(info.connections) != 1:
                raise ProducerError(f"expected one Vicon connection for {topic}")
            connection = info.connections[0]
            timestamps: list[int] = []
            positions: list[tuple[float, float, float]] = []
            quaternions: list[tuple[float, float, float, float]] = []
            for _, timestamp, rawdata in reader.messages(connections=[connection]):
                message = typestore.deserialize_ros1(rawdata, connection.msgtype)
                transform = message.transform
                timestamps.append(timestamp)
                positions.append(
                    (
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    )
                )
                quaternions.append(
                    (
                        transform.rotation.x,
                        transform.rotation.y,
                        transform.rotation.z,
                        transform.rotation.w,
                    )
                )
            output[name] = {
                "timestamps_ns": np.asarray(timestamps, dtype=np.int64),
                "positions": np.asarray(positions, dtype=np.float64),
                "quaternions": np.asarray(quaternions, dtype=np.float64),
            }
    return output


def read_camera_from_marker(calibration_path: Path) -> np.ndarray:
    import yaml

    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = yaml.safe_load(handle)
    matrix = np.asarray(calibration["T_v_c"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ProducerError("invalid calibration T_v_c")
    return matrix


def _partition(event: dict[str, Any]) -> str:
    if event.get("correct") is True:
        return "CORRECT"
    if event.get("wrong_signed") is True:
        return "WRONG_SIGNED"
    return "OTHER_INCORRECT"


def _component_summary(
    rows: list[dict[str, Any]], field: str, denominator: int
) -> tuple[list[float], dict[str, Any]]:
    values = finite_values(
        row.get("quality", {}).get("components", {}).get(field) for row in rows
    )
    count = len(values)
    coverage = count / denominator
    return values, {
        "finite_count": count,
        "finite_coverage": coverage,
        "missing_count": denominator - count,
        "missing_reason_counts": (
            {"NONFINITE_OR_MISSING_COMPONENT": denominator - count}
            if count < denominator
            else {}
        ),
    }


def _flow_sign_flip_fraction(rows: list[dict[str, Any]]) -> float | None:
    transitions = 0
    flips = 0
    previous_index: int | None = None
    previous_sign: int | None = None
    for row in rows:
        rate = row["flow"].get("signed_approach_rate_per_s")
        if row["flow"].get("abstention_reason") is not None or not _finite(rate):
            previous_index = previous_sign = None
            continue
        sign = 1 if rate >= 0.02 else -1 if rate <= -0.02 else 0
        index = row["source_frame_index"]
        if previous_index is not None and index == previous_index + 1:
            transitions += 1
            flips += sign != previous_sign
        previous_index, previous_sign = index, sign
    return flips / transitions if transitions else None


def _roi_pair(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    bbox: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if previous is None:
        return None, "NO_PREVIOUS_REPLAY_ROW"
    dt = (current["captured_at_ns"] - previous["captured_at_ns"]) / 1e9
    if (
        previous.get("track_epoch") != current.get("track_epoch")
        or current.get("history_reset") is not False
        or not 0 < dt <= 0.1
        or bbox.get("abstention_reason") is not None
        or not _finite(bbox.get("signed_approach_rate_per_s"))
    ):
        return None, "ROI_PAIR_CONTRACT_REJECTED"
    previous_roi, current_roi = previous["roi_xywh_normalized"], current["roi_xywh_normalized"]
    if len(previous_roi) != 4 or len(current_roi) != 4:
        return None, "ROI_SHAPE_CHANGE"
    previous_area = previous_roi[2] * previous_roi[3]
    current_area = current_roi[2] * current_roi[3]
    if previous_area <= 0 or current_area <= 0:
        return None, "ROI_AREA_NONPOSITIVE"
    rate = math.log(current_area / previous_area) / dt
    if abs(rate - float(bbox["signed_approach_rate_per_s"])) > BBOX_CLOSURE_ATOL_PER_S:
        raise ProducerError("BBOX log-area closure mismatch")
    previous_center = (
        previous_roi[0] + previous_roi[2] / 2,
        previous_roi[1] + previous_roi[3] / 2,
    )
    current_center = (
        current_roi[0] + current_roi[2] / 2,
        current_roi[1] + current_roi[3] / 2,
    )
    velocity = (
        (current_center[0] - previous_center[0]) / dt,
        (current_center[1] - previous_center[1]) / dt,
    )
    return {"log_area_rate": rate, "center_velocity": velocity}, None


def _aggregate_event(
    natural: dict[str, Any],
    binding: dict[str, Any],
    primary_eval: dict[str, Any],
    reference_eval: dict[str, Any],
    member_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    denominator = natural["eligible_frame_count"]
    source_pairs = [row["source_pair"] for row in member_rows if row["source_pair"]]
    source_reasons = Counter(
        row["source_missing_reason"]
        for row in member_rows
        if row["source_missing_reason"] is not None
    )
    finite_source_count = len(source_pairs)
    source_coverage = finite_source_count / denominator
    source_supported = finite_source_count >= 3 and source_coverage >= 0.5
    missing: dict[str, str] = {}
    source_values: dict[str, Any] = {field: None for field in NULL_SOURCE_FIELDS}
    if source_supported:
        source_values.update(
            {
                "median_person_approach_component_mps": median(
                    pair["person_approach_component_mps"] for pair in source_pairs
                ),
                "median_sensor_approach_component_mps": median(
                    pair["sensor_approach_component_mps"] for pair in source_pairs
                ),
                "median_abs_person_approach_component_mps": median(
                    abs(pair["person_approach_component_mps"]) for pair in source_pairs
                ),
                "median_abs_sensor_approach_component_mps": median(
                    abs(pair["sensor_approach_component_mps"]) for pair in source_pairs
                ),
                "median_camera_translation_speed_mps": median(
                    pair["camera_translation_speed_mps"] for pair in source_pairs
                ),
                "p90_camera_translation_speed_mps": type7_quantile(
                    (pair["camera_translation_speed_mps"] for pair in source_pairs), 0.9
                ),
                "median_camera_angular_speed_radps": median(
                    pair["camera_angular_speed_radps"] for pair in source_pairs
                ),
                "p90_camera_angular_speed_radps": type7_quantile(
                    (pair["camera_angular_speed_radps"] for pair in source_pairs), 0.9
                ),
            }
        )
        shares = finite_values(pair.get("sensor_absolute_share") for pair in source_pairs)
        if len(shares) >= 3 and len(shares) / denominator >= 0.5:
            source_values["sensor_absolute_share"] = median(shares)
        else:
            missing["sensor_absolute_share"] = "INSUFFICIENT_FINITE_SHARE_SUPPORT"
    else:
        for field in NULL_SOURCE_FIELDS:
            missing[field] = "INSUFFICIENT_FINITE_SOURCE_PAIR_SUPPORT"

    roi_pairs = [row["roi_pair"] for row in member_rows if row["roi_pair"]]
    log_rates = [pair["log_area_rate"] for pair in roi_pairs]
    velocities = [pair["center_velocity"] for pair in roi_pairs]
    velocity_center = (
        median(value[0] for value in velocities),
        median(value[1] for value in velocities),
    )
    center_mad = (
        median(
            math.hypot(value[0] - velocity_center[0], value[1] - velocity_center[1])
            for value in velocities
        )
        if velocities
        else None
    )
    flow_rows = [row for row in member_rows if row["flow"].get("abstention_reason") is None and _finite(row["flow"].get("signed_approach_rate_per_s"))]
    finite_flow_count = len(flow_rows)
    quality_fields = {
        "score_mad_per_s": "median_flow_score_mad_per_s",
        "detected_features": "median_detected_features",
        "surviving_tracks": "median_surviving_tracks",
        "occupied_quadrants": "median_occupied_quadrants",
        "median_fb_error_px": "median_forward_backward_error_px",
    }
    quality_support: dict[str, Any] = {}
    quality_values: dict[str, list[float]] = {}
    for component, output_field in quality_fields.items():
        values, support = _component_summary(
            [row["flow"] for row in member_rows], component, denominator
        )
        quality_values[output_field] = values
        quality_support[output_field] = support
    temporal_fields = ("median_flow_score_mad_per_s", "median_surviving_tracks")
    temporal_supported = all(
        quality_support[field]["finite_count"] >= 3
        and quality_support[field]["finite_coverage"] >= 0.5
        for field in temporal_fields
    )
    temporal_output: dict[str, Any] = {}
    for field in temporal_fields:
        if temporal_supported:
            temporal_output[field] = median(quality_values[field])
        else:
            temporal_output[field] = None
            missing[field] = "INSUFFICIENT_COUPLED_TEMPORAL_SUPPORT"

    duration = natural["duration_s"]
    row = {
        "event_id": natural["event_id"],
        "capture_id": natural["capture_id"],
        "target_id": natural["target_id"],
        "anchor_region": natural["anchor_region"],
        "truth_state": natural["truth_state"],
        "start_timestamp_ns": natural["start_timestamp_ns"],
        "end_timestamp_ns": natural["end_timestamp_ns"],
        "eligible_frame_count": denominator,
        "overlap_component_id": binding["overlap_component_id"],
        "time_block_id_60s": binding["time_block_id_60s"],
        "primary_error_partition": _partition(primary_eval),
        "reference_error_partition": _partition(reference_eval),
        "source_pair_denominator": denominator,
        "finite_source_pair_count": finite_source_count,
        "finite_source_pair_coverage": source_coverage,
        "source_missing_reason_counts": dict(sorted(source_reasons.items())),
        **source_values,
        "median_abs_log_area_rate_per_s": median(abs(value) for value in log_rates),
        "log_area_rate_mad_per_s": raw_mad(log_rates),
        "median_center_speed_normalized_per_s": median(
            math.hypot(*value) for value in velocities
        ),
        "center_velocity_mad_normalized_per_s": center_mad,
        "duration_s": duration,
        "negative_log_duration_s": -math.log(duration),
        "finite_flow_coverage": finite_flow_count / denominator,
        "flow_sign_flip_fraction": _flow_sign_flip_fraction(member_rows),
        **temporal_output,
        "p90_flow_score_mad_per_s": type7_quantile(
            quality_values["median_flow_score_mad_per_s"], 0.9
        ),
        "median_detected_features": median(
            quality_values["median_detected_features"]
        ),
        "minimum_surviving_tracks": (
            min(quality_values["median_surviving_tracks"])
            if quality_values["median_surviving_tracks"]
            else None
        ),
        "median_occupied_quadrants": median(
            quality_values["median_occupied_quadrants"]
        ),
        "median_forward_backward_error_px": median(
            quality_values["median_forward_backward_error_px"]
        ),
        "abstained_pair_count": denominator - finite_flow_count,
        "quality_component_support": quality_support,
        "reference_arm_behavior": reference_eval.get("event_score_per_s"),
    }
    for field, reason in (
        ("median_abs_log_area_rate_per_s", "NO_VALID_ROI_PAIRS"),
        ("log_area_rate_mad_per_s", "NO_VALID_ROI_PAIRS"),
        ("median_center_speed_normalized_per_s", "NO_VALID_ROI_PAIRS"),
        ("center_velocity_mad_normalized_per_s", "NO_VALID_ROI_PAIRS"),
        ("flow_sign_flip_fraction", "NO_ELIGIBLE_ADJACENT_FLOW_TRANSITIONS"),
    ):
        if row[field] is None:
            missing[field] = reason
    for field in (
        "p90_flow_score_mad_per_s",
        "median_detected_features",
        "minimum_surviving_tracks",
        "median_occupied_quadrants",
        "median_forward_backward_error_px",
    ):
        if row[field] is None:
            missing[field] = "NO_FINITE_QUALITY_COMPONENT_ROWS"
    nullable_summary_fields = (
        *NULL_SOURCE_FIELDS,
        "median_abs_log_area_rate_per_s",
        "log_area_rate_mad_per_s",
        "median_center_speed_normalized_per_s",
        "center_velocity_mad_normalized_per_s",
        "flow_sign_flip_fraction",
        "median_flow_score_mad_per_s",
        "p90_flow_score_mad_per_s",
        "median_detected_features",
        "median_surviving_tracks",
        "minimum_surviving_tracks",
        "median_occupied_quadrants",
        "median_forward_backward_error_px",
        "reference_arm_behavior",
    )
    for field in nullable_summary_fields:
        reason_field = f"{field}_missing_reason"
        if row[field] is None:
            row[reason_field] = missing.get(field, "NONFINITE_OR_NULL")
        else:
            row[reason_field] = None
    return row


def build_event_table(
    bundle: FrozenBundle,
    tracks: dict[str, dict[str, np.ndarray]],
    camera_from_marker: np.ndarray,
) -> list[dict[str, Any]]:
    replay_index = {
        (row["capture_id"], row["target_id"], row["source_frame_id"]): row
        for row in bundle.replay_rows
    }
    truth_index = {
        (row["capture_id"], row["target_id"], row["source_frame_id"]): row
        for row in bundle.truth_rows
    }
    r2_index = {
        (row["arm_id"], row["capture_id"], row["target_id"], row["source_frame_id"]): row
        for row in bundle.r2_rows
    }
    natural = {
        row["event_id"]: row
        for row in bundle.natural_rows
        if row.get("primary_event_eligible") is True
    }
    evaluation: dict[tuple[str, str], dict[str, Any]] = {}
    for arm_id, summary in bundle.evaluation["arm_summaries"].items():
        for event in summary["events"]:
            evaluation[(arm_id, event["event_id"])] = event

    previous_by_target: dict[str, dict[str, Any]] = {}
    replay_previous: dict[tuple[str, str, str], dict[str, Any] | None] = {}
    last_index: dict[str, int] = {}
    for replay in bundle.replay_rows:
        target = replay["target_id"]
        source_index = truth_index[
            (replay["capture_id"], target, replay["source_frame_id"])
        ]["source_frame_index"]
        if target in last_index and source_index <= last_index[target]:
            raise ProducerError("replay source_frame_index is not strictly increasing")
        key = (replay["capture_id"], target, replay["source_frame_id"])
        replay_previous[key] = previous_by_target.get(target)
        previous_by_target[target] = replay
        last_index[target] = source_index

    output: list[dict[str, Any]] = []
    for binding in bundle.dependency["event_bindings"]:
        event_id = binding["event_id"]
        event = natural[event_id]
        members = sorted(
            (
                row
                for row in bundle.truth_rows
                if row.get("event_id") == event_id
            ),
            key=lambda row: row["source_frame_index"],
        )
        member_rows: list[dict[str, Any]] = []
        for truth in members:
            key = (truth["capture_id"], truth["target_id"], truth["source_frame_id"])
            replay = replay_index[key]
            bbox = r2_index[(REFERENCE_ARM, *key)]
            flow = r2_index[(PRIMARY_ARM, *key)]
            source_pair, source_reason = source_pair_metrics(
                truth["bag_image_timestamp_ns"],
                tracks[truth["target_id"]],
                tracks["sensor"],
                camera_from_marker,
                truth["truth_signed_approach_mps"],
            )
            roi_pair, _ = _roi_pair(replay_previous[key], replay, bbox)
            member_rows.append(
                {
                    "source_frame_index": truth["source_frame_index"],
                    "source_pair": source_pair,
                    "source_missing_reason": source_reason,
                    "roi_pair": roi_pair,
                    "flow": flow,
                }
            )
        output.append(
            _aggregate_event(
                event,
                binding,
                evaluation[(PRIMARY_ARM, event_id)],
                evaluation[(REFERENCE_ARM, event_id)],
                member_rows,
            )
        )
    if [row["event_id"] for row in output] != sorted(natural):
        raise ProducerError("event-table order drift")
    return output
