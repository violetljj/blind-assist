#!/usr/bin/env python3
"""Profile independent dual-PCD support and bias for JRDB annotation trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    apply_transform,
    canonical_bytes,
    sha256_file,
    transform_from_q,
    write_canonical,
)

STAGE = "JRDB_PERSON_3D_TRAJECTORY_SENSOR_SUPPORT_AND_BIAS_CANARY_R0"
CONFIG_SCHEMA = "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0_config"
LEDGER_SCHEMA = "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0_ledger"
RECEIPT_SCHEMA = "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0_receipt"
CLASSES = ("sensor-supported", "annotation-only", "abstained", "invalid")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def vector_sub(left: Iterable[float], right: Iterable[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def vector_scale(value: Iterable[float], scale: float) -> list[float]:
    return [float(item) * scale for item in value]


def vector_norm(value: Iterable[float]) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in value))


def centroid(points: list[tuple[float, float, float]]) -> list[float] | None:
    if not points:
        return None
    return [sum(point[axis] for point in points) / len(points) for axis in range(3)]


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}

    def pick(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p05": pick(0.05),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def class_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["classification"] for row in rows)
    return {name: counts[name] for name in CLASSES}


def denominator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = class_counts(rows)
    require(sum(counts.values()) == len(rows), "classification conservation")
    return {"expected": len(rows), **counts, "conserved": True}


def lzf_decompress(payload: bytes, expected_size: int) -> bytes:
    output = bytearray()
    cursor = 0
    while cursor < len(payload):
        control = payload[cursor]
        cursor += 1
        if control < 32:
            length = control + 1
            require(cursor + length <= len(payload), "truncated LZF literal")
            output.extend(payload[cursor : cursor + length])
            cursor += length
            continue
        length = control >> 5
        reference = len(output) - ((control & 0x1F) << 8) - 1
        if length == 7:
            require(cursor < len(payload), "truncated LZF length")
            length += payload[cursor]
            cursor += 1
        require(cursor < len(payload), "truncated LZF reference")
        reference -= payload[cursor]
        cursor += 1
        length += 2
        require(reference >= 0, "invalid LZF reference")
        for _ in range(length):
            require(reference < len(output), "invalid LZF overlap")
            output.append(output[reference])
            reference += 1
    require(len(output) == expected_size, f"LZF size {len(output)} != {expected_size}")
    return bytes(output)


def read_pcd_xyz(path: Path) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    with path.open("rb") as handle:
        values: dict[str, str] = {}
        while True:
            line = handle.readline()
            require(line, f"PCD header EOF: {path}")
            text = line.decode("ascii").strip()
            if text and not text.startswith("#"):
                key, _, value = text.partition(" ")
                values[key.upper()] = value.strip()
            if text.upper().startswith("DATA "):
                break
        body = handle.read()
    fields = values["FIELDS"].split()
    sizes = [int(value) for value in values["SIZE"].split()]
    types = values["TYPE"].split()
    counts = [int(value) for value in values.get("COUNT", " ".join("1" for _ in fields)).split()]
    points = int(values["POINTS"])
    require(len(fields) == len(sizes) == len(types) == len(counts), "PCD field contract")
    require({"x", "y", "z"}.issubset(fields), "PCD xyz absent")
    require(values["DATA"] == "binary_compressed", "PCD must be binary_compressed")
    require(len(body) >= 8, "PCD compressed prefix")
    compressed_size, uncompressed_size = struct.unpack_from("<II", body, 0)
    require(compressed_size == len(body) - 8, "PCD compressed size")
    expected = sum(size * count for size, count in zip(sizes, counts)) * points
    require(uncompressed_size == expected, "PCD uncompressed size")
    raw = lzf_decompress(body[8:], uncompressed_size)
    offsets: dict[str, int] = {}
    offset = 0
    for field, size, count in zip(fields, sizes, counts):
        offsets[field] = offset
        offset += size * count * points
    for axis in ("x", "y", "z"):
        index = fields.index(axis)
        require(sizes[index] == 4 and types[index] == "F" and counts[index] == 1, f"PCD {axis} type")
    xyz: list[tuple[float, float, float]] = []
    nonfinite = 0
    for index in range(points):
        point = tuple(struct.unpack_from("<f", raw, offsets[axis] + 4 * index)[0] for axis in ("x", "y", "z"))
        if all(math.isfinite(value) for value in point):
            xyz.append(point)
        else:
            nonfinite += 1
    return xyz, {
        "declared_points": points,
        "finite_points": len(xyz),
        "nonfinite_points": nonfinite,
        "compressed_bytes": compressed_size,
        "uncompressed_bytes": uncompressed_size,
    }


def oriented_box_points(
    points: list[tuple[float, float, float]], box: dict[str, float]
) -> list[tuple[float, float, float]]:
    cx, cy, cz = (box[key] for key in ("cx", "cy", "cz"))
    half_l, half_w, half_h = box["l"] / 2, box["w"] / 2, box["h"] / 2
    cosine, sine = math.cos(box["rot_z"]), math.sin(box["rot_z"])
    selected: list[tuple[float, float, float]] = []
    for point in points:
        dx, dy, dz = point[0] - cx, point[1] - cy, point[2] - cz
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        if abs(local_x) <= half_l and abs(local_y) <= half_w and abs(dz) <= half_h:
            selected.append(point)
    return selected


def support_pattern(upper: int, lower: int) -> str:
    if upper and lower:
        return "both"
    if upper:
        return "upper-only"
    if lower:
        return "lower-only"
    return "neither"


def range_band(distance: float) -> str:
    if distance < 10:
        return "0-10"
    if distance < 20:
        return "10-20"
    if distance < 40:
        return "20-40"
    return "40-plus"


def point_band(count: int) -> str:
    if count == 0:
        return "zero"
    if count <= 2:
        return "sparse-1-2"
    if count <= 9:
        return "supported-3-9"
    return "supported-10-plus"


def load_bound_inputs(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "config identity")
    loaded: dict[str, Any] = {}
    for role, binding in config["parent"].items():
        path = repo / binding["path"]
        require(path.is_file(), f"missing parent {role}")
        require(sha256_file(path) == binding["sha256"], f"parent drift {role}")
        if path.suffix == ".json":
            loaded[role] = json.loads(path.read_text(encoding="utf-8"))
    require(loaded["receipt"]["terminal_state"] == config["parent"]["receipt"]["required_terminal"], "parent terminal")
    require(loaded["validation"]["status"] == config["parent"]["validation"]["required_status"], "parent validation")
    require(loaded["observation_packet"]["status"] == config["parent"]["observation_packet"]["required_status"], "packet status")
    return config, loaded["observation_packet"], loaded["eligibility_ledger"]


def raw_file_index(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["member"]: row for row in packet["raw_payload"]["files"]}


def label_documents(repo: Path, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    files = raw_file_index(packet)
    label_rows = [row for row in files.values() if row["role"] == "labels"]
    labels_2d = next(row for row in label_rows if "/labels_2d_stitched/" in row["member"])
    labels_3d = next(row for row in label_rows if "/labels_3d/" in row["member"])
    for row in (labels_2d, labels_3d):
        require(sha256_file(repo / row["path"]) == row["sha256"], f"label drift {row['member']}")
    return (
        json.loads((repo / labels_2d["path"]).read_text(encoding="utf-8")),
        json.loads((repo / labels_3d["path"]).read_text(encoding="utf-8")),
    )


def transform_points(points: list[tuple[float, float, float]], transform: dict[str, Any]) -> list[tuple[float, float, float]]:
    return [tuple(apply_transform(transform, point)) for point in points]


def object_row(
    label_id: str,
    frame: dict[str, Any],
    item_3d: dict[str, Any] | None,
    item_2d: dict[str, Any] | None,
    upper_points: list[tuple[float, float, float]],
    lower_points: list[tuple[float, float, float]],
    minimum_points: int,
) -> dict[str, Any]:
    base = {
        "frame_index": frame["frame_index"],
        "frame_stem": frame["frame_stem"],
        "label_id": label_id,
        "cross_modal_presence": "3d-and-2d" if item_3d and item_2d else ("3d-only" if item_3d else "2d-only"),
        "occlusion": (item_2d or {}).get("attributes", {}).get("occlusion") or "Unknown",
    }
    if item_3d is None:
        return {**base, "classification": "abstained", "reason": "missing_3d_annotation_box"}
    box_source = item_3d.get("box", {})
    keys = ("cx", "cy", "cz", "w", "l", "h", "rot_z")
    try:
        box = {key: float(box_source[key]) for key in keys}
    except (KeyError, TypeError, ValueError):
        return {**base, "classification": "invalid", "reason": "invalid_3d_box"}
    if not all(math.isfinite(value) for value in box.values()) or min(box["w"], box["l"], box["h"]) <= 0:
        return {**base, "classification": "invalid", "reason": "invalid_3d_box"}
    upper_selected = oriented_box_points(upper_points, box)
    lower_selected = oriented_box_points(lower_points, box)
    fused = upper_selected + lower_selected
    count = len(fused)
    if count >= minimum_points:
        classification, reason = "sensor-supported", "minimum_fused_support_met"
    elif count == 0:
        classification, reason = "annotation-only", "no_in_box_lidar_return"
    else:
        classification, reason = "abstained", "positive_support_below_centroid_minimum"
    annotation_center = [box[key] for key in ("cx", "cy", "cz")]
    sensor_centroid = centroid(fused) if classification == "sensor-supported" else None
    residual = vector_sub(sensor_centroid, annotation_center) if sensor_centroid else None
    distance = vector_norm(annotation_center)
    return {
        **base,
        "classification": classification,
        "reason": reason,
        "source_interpolated": item_3d.get("attributes", {}).get("interpolated") is True,
        "source_num_points_attribute": item_3d.get("attributes", {}).get("num_points"),
        "box_3d": box,
        "annotation_center_logical_rgb360_m": annotation_center,
        "range_m": distance,
        "range_band": range_band(distance),
        "upper_in_box_points": len(upper_selected),
        "lower_in_box_points": len(lower_selected),
        "fused_in_box_points": count,
        "point_support_band": point_band(count),
        "sensor_pattern": support_pattern(len(upper_selected), len(lower_selected)),
        "upper_centroid_logical_rgb360_m": centroid(upper_selected),
        "lower_centroid_logical_rgb360_m": centroid(lower_selected),
        "sensor_centroid_logical_rgb360_m": sensor_centroid,
        "centroid_residual_xyz_m": residual,
        "centroid_residual_horizontal_m": math.hypot(residual[0], residual[1]) if residual else None,
        "centroid_residual_3d_m": vector_norm(residual) if residual else None,
    }


def build_object_rows(
    repo: Path,
    config: dict[str, Any],
    packet: dict[str, Any],
    labels_2d: dict[str, Any],
    labels_3d: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    minimum = int(config["support_contract"]["minimum_fused_in_box_points"])
    transforms = {
        "upper": packet["calibration"]["upper_to_logical_rgb360"],
        "lower": packet["calibration"]["lower_to_logical_rgb360"],
    }
    object_rows: list[dict[str, Any]] = []
    frame_audits: list[dict[str, Any]] = []
    for frame in packet["frames"]:
        stem = frame["frame_stem"]
        sensor_points: dict[str, list[tuple[float, float, float]]] = {}
        audit: dict[str, Any] = {"frame_index": frame["frame_index"], "frame_stem": stem}
        for sensor in ("upper", "lower"):
            source = frame["source"][f"{sensor}_pointcloud"]
            path = repo / source["path"]
            require(path.is_file() and sha256_file(path) == source["sha256"], f"PCD drift {sensor}:{stem}")
            points, metadata = read_pcd_xyz(path)
            require(metadata["declared_points"] == source["points"], f"PCD points {sensor}:{stem}")
            sensor_points[sensor] = transform_points(points, transforms[sensor])
            audit[sensor] = {**metadata, "sha256": source["sha256"], "timestamp_ns": frame["time"][f"{sensor}_pointcloud_timestamp_ns"]}
        audit["sensor_timestamp_skew_seconds"] = abs(audit["upper"]["timestamp_ns"] - audit["lower"]["timestamp_ns"]) / 1e9
        frame_audits.append(audit)
        objects_2d = labels_2d["labels"][f"{stem}.jpg"]
        objects_3d = labels_3d["labels"][f"{stem}.pcd"]
        index_2d = {item["label_id"]: item for item in objects_2d}
        index_3d = {item["label_id"]: item for item in objects_3d}
        require(len(index_2d) == len(objects_2d) and len(index_3d) == len(objects_3d), f"duplicate label {stem}")
        for label_id in sorted(set(index_2d) | set(index_3d)):
            object_rows.append(
                object_row(
                    label_id,
                    frame,
                    index_3d.get(label_id),
                    index_2d.get(label_id),
                    sensor_points["upper"],
                    sensor_points["lower"],
                    minimum,
                )
            )
    return object_rows, frame_audits


def pose_for_frame(packet: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        frame["frame_index"]: transform_from_q(frame["pose"]["translation"], frame["pose"]["quaternion_xyzw"])
        for frame in packet["frames"]
    }


def classify_pair(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, str]:
    classifications = {left["classification"], right["classification"]}
    if "invalid" in classifications:
        return "invalid", "invalid_endpoint"
    if "abstained" in classifications:
        return "abstained", "abstained_endpoint"
    if "annotation-only" in classifications:
        return "annotation-only", "endpoint_without_lidar_return"
    return "sensor-supported", "both_endpoints_sensor_supported"


def build_pair_rows(
    config: dict[str, Any],
    packet: dict[str, Any],
    parent_ledger: dict[str, Any],
    objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    object_index = {(row["frame_index"], row["label_id"]): row for row in objects if row["cross_modal_presence"] != "2d-only"}
    observations = {(row["frame_index"], row["label_id"]): row for row in parent_ledger["observations"]}
    poses = pose_for_frame(packet)
    base_from_rgb = packet["calibration"]["base_link_from_logical_rgb360"]
    contract = config["motion_contract"]
    pairs: list[dict[str, Any]] = []
    for source_pair in parent_ledger["motion_pairs"]:
        key_left = (source_pair["left_frame"], source_pair["label_id"])
        key_right = (source_pair["right_frame"], source_pair["label_id"])
        left, right = object_index[key_left], object_index[key_right]
        classification, reason = classify_pair(left, right)
        row: dict[str, Any] = {
            "label_id": source_pair["label_id"],
            "left_frame": source_pair["left_frame"],
            "right_frame": source_pair["right_frame"],
            "gap_seconds": source_pair["gap_seconds"],
            "classification": classification,
            "reason": reason,
            "annotation_odom_velocity_mps": source_pair["source_annotation_odom_velocity_mps"],
            "annotation_odom_speed_mps": source_pair["source_annotation_odom_speed_mps"],
            "annotation_speed_flag": source_pair["source_annotation_odom_speed_mps"] > float(contract["speed_flag_mps"]),
        }
        left_obs, right_obs = observations[key_left], observations[key_right]
        annotation_delta = vector_sub(right_obs["center_odom_m"], left_obs["center_odom_m"])
        row["annotation_odom_displacement_m"] = annotation_delta
        row["annotation_jump_flag"] = vector_norm(annotation_delta) > float(contract["jump_displacement_flag_meters"])
        if classification == "sensor-supported":
            left_base = apply_transform(base_from_rgb, left["sensor_centroid_logical_rgb360_m"])
            right_base = apply_transform(base_from_rgb, right["sensor_centroid_logical_rgb360_m"])
            sensor_left_odom = apply_transform(poses[source_pair["left_frame"]], left_base)
            sensor_right_odom = apply_transform(poses[source_pair["right_frame"]], right_base)
            sensor_delta = vector_sub(sensor_right_odom, sensor_left_odom)
            frozen_right = apply_transform(poses[source_pair["left_frame"]], right_base)
            frozen_left = apply_transform(poses[source_pair["left_frame"]], left_base)
            frozen_delta = vector_sub(frozen_right, frozen_left)
            annotation_frozen_right = apply_transform(
                poses[source_pair["left_frame"]], right_obs["center_base_link_m"]
            )
            annotation_frozen_left = apply_transform(
                poses[source_pair["left_frame"]], left_obs["center_base_link_m"]
            )
            annotation_frozen_delta = vector_sub(
                annotation_frozen_right, annotation_frozen_left
            )
            gap = float(source_pair["gap_seconds"])
            sensor_velocity = vector_scale(sensor_delta, 1 / gap)
            row.update(
                {
                    "sensor_centroid_odom_displacement_m": sensor_delta,
                    "sensor_centroid_odom_velocity_mps": sensor_velocity,
                    "sensor_centroid_odom_speed_mps": vector_norm(sensor_velocity),
                    "sensor_motion_minus_annotation_xyz_m": vector_sub(sensor_delta, annotation_delta),
                    "sensor_motion_minus_annotation_3d_m": vector_norm(vector_sub(sensor_delta, annotation_delta)),
                    "pose_sensitivity_annotation_m": vector_norm(vector_sub(annotation_delta, annotation_frozen_delta)),
                    "pose_sensitivity_sensor_m": vector_norm(vector_sub(sensor_delta, frozen_delta)),
                    "sensor_jump_flag": vector_norm(sensor_delta) > float(contract["jump_displacement_flag_meters"]),
                    "sensor_speed_flag": vector_norm(sensor_velocity) > float(contract["speed_flag_mps"]),
                }
            )
        pairs.append(row)
    return pairs


def build_acceleration_rows(config: dict[str, Any], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[row["label_id"]].append(row)
    output: list[dict[str, Any]] = []
    threshold = float(config["motion_contract"]["acceleration_flag_mps2"])
    for label_id, rows in grouped.items():
        rows.sort(key=lambda row: row["left_frame"])
        for left, right in zip(rows, rows[1:]):
            if left["right_frame"] != right["left_frame"]:
                continue
            classification, reason = classify_pair(left, right)
            row: dict[str, Any] = {
                "label_id": label_id,
                "left_frame": left["left_frame"],
                "middle_frame": left["right_frame"],
                "right_frame": right["right_frame"],
                "classification": classification,
                "reason": reason,
            }
            interval = (float(left["gap_seconds"]) + float(right["gap_seconds"])) / 2
            annotation_acceleration = vector_scale(
                vector_sub(right["annotation_odom_velocity_mps"], left["annotation_odom_velocity_mps"]),
                1 / interval,
            )
            row["annotation_acceleration_mps2"] = annotation_acceleration
            row["annotation_acceleration_norm_mps2"] = vector_norm(annotation_acceleration)
            row["annotation_acceleration_flag"] = vector_norm(annotation_acceleration) > threshold
            if classification == "sensor-supported":
                sensor_acceleration = vector_scale(
                    vector_sub(right["sensor_centroid_odom_velocity_mps"], left["sensor_centroid_odom_velocity_mps"]),
                    1 / interval,
                )
                row["sensor_acceleration_mps2"] = sensor_acceleration
                row["sensor_acceleration_norm_mps2"] = vector_norm(sensor_acceleration)
                row["sensor_acceleration_flag"] = vector_norm(sensor_acceleration) > threshold
            output.append(row)
    return output


def group_profiles(objects: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dimensions = ("cross_modal_presence", "occlusion", "point_support_band", "range_band", "sensor_pattern", "label_id")
    profiles: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in objects:
            groups[str(row.get(dimension, "not-applicable"))].append(row)
        profiles[dimension] = []
        for value, rows in sorted(groups.items()):
            supported = [row for row in rows if row["classification"] == "sensor-supported"]
            residuals = [row["centroid_residual_3d_m"] for row in supported]
            profiles[dimension].append(
                {
                    "value": value,
                    "denominator": denominator(rows),
                    "sensor_supported_fraction": len(supported) / len(rows),
                    "centroid_residual_3d_m": quantiles(residuals),
                    "signed_residual_x_m": quantiles(
                        row["centroid_residual_xyz_m"][0] for row in supported
                    ),
                    "signed_residual_y_m": quantiles(
                        row["centroid_residual_xyz_m"][1] for row in supported
                    ),
                    "signed_residual_z_m": quantiles(
                        row["centroid_residual_xyz_m"][2] for row in supported
                    ),
                }
            )
    return profiles


def build_ledger(repo: Path, config_path: Path) -> dict[str, Any]:
    config, packet, parent_ledger = load_bound_inputs(repo, config_path)
    labels_2d, labels_3d = label_documents(repo, packet)
    objects, frame_audits = build_object_rows(repo, config, packet, labels_2d, labels_3d)
    pairs = build_pair_rows(config, packet, parent_ledger, objects)
    accelerations = build_acceleration_rows(config, pairs)
    three_d_objects = [row for row in objects if row["cross_modal_presence"] != "2d-only"]
    supported_objects = [row for row in three_d_objects if row["classification"] == "sensor-supported"]
    supported_pairs = [row for row in pairs if row["classification"] == "sensor-supported"]
    require(len(three_d_objects) == parent_ledger["denominators"]["robot_relative_3d_geometry"]["expected"], "3D denominator drift")
    require(len(pairs) == parent_ledger["denominators"]["source_annotation_derived_3d_motion"]["expected"], "pair denominator drift")
    return {
        "schema": LEDGER_SCHEMA,
        "stage": STAGE,
        "status": "COMPLETE",
        "sequence": packet["sequence"],
        "window": packet["window"],
        "parent_packet_sha256": config["parent"]["observation_packet"]["sha256"],
        "support_contract": config["support_contract"],
        "motion_contract": config["motion_contract"],
        "frame_audits": frame_audits,
        "object_frames": objects,
        "motion_pairs": pairs,
        "acceleration_triples": accelerations,
        "denominators": {
            "union_object_frames": denominator(objects),
            "computable_3d_object_frames": denominator(three_d_objects),
            "motion_pairs": denominator(pairs),
            "acceleration_triples": denominator(accelerations),
        },
        "summary": {
            "computable_3d_object_sensor_supported_fraction": len(supported_objects) / len(three_d_objects),
            "computable_motion_pair_sensor_supported_fraction": len(supported_pairs) / len(pairs),
            "sensor_pattern_counts": dict(sorted(Counter(row.get("sensor_pattern", "not-applicable") for row in three_d_objects).items())),
            "centroid_residual_3d_m": quantiles(row["centroid_residual_3d_m"] for row in supported_objects),
            "centroid_residual_horizontal_m": quantiles(row["centroid_residual_horizontal_m"] for row in supported_objects),
            "sensor_motion_minus_annotation_3d_m": quantiles(row["sensor_motion_minus_annotation_3d_m"] for row in supported_pairs),
            "annotation_speed_mps": quantiles(row["annotation_odom_speed_mps"] for row in pairs),
            "sensor_speed_mps": quantiles(row["sensor_centroid_odom_speed_mps"] for row in supported_pairs),
            "pose_sensitivity_annotation_m": quantiles(row["pose_sensitivity_annotation_m"] for row in supported_pairs),
            "pose_sensitivity_sensor_m": quantiles(row["pose_sensitivity_sensor_m"] for row in supported_pairs),
            "annotation_jump_flags": sum(row["annotation_jump_flag"] for row in pairs),
            "sensor_jump_flags": sum(row.get("sensor_jump_flag", False) for row in supported_pairs),
            "annotation_speed_flags": sum(row["annotation_speed_flag"] for row in pairs),
            "sensor_speed_flags": sum(row.get("sensor_speed_flag", False) for row in supported_pairs),
            "annotation_acceleration_flags": sum(row["annotation_acceleration_flag"] for row in accelerations),
            "sensor_acceleration_flags": sum(row.get("sensor_acceleration_flag", False) for row in accelerations),
            "sensor_timestamp_skew_seconds": quantiles(row["sensor_timestamp_skew_seconds"] for row in frame_audits),
        },
        "bias_profiles": group_profiles(objects),
        "authority": config["authority"],
        "limitations": [
            "point-in-box support is conditioned on the annotation box and is not an annotation-free person detector",
            "LiDAR centroid measures visible returns, not an independent true human center",
            "upper and lower scans have distinct timestamps and are reported separately before an un-deskewed fused centroid",
            "all source 3D annotations are interpolated and this is one seen-development sequence",
        ],
    }


def build_receipt(config: dict[str, Any], ledger: dict[str, Any], ledger_sha: str) -> dict[str, Any]:
    object_denominator = ledger["denominators"]["computable_3d_object_frames"]
    pair_denominator = ledger["denominators"]["motion_pairs"]
    if object_denominator["sensor-supported"] == 0:
        terminal = "NOT_EVALUABLE_POINTCLOUD_SUPPORT"
    elif any(
        denominator_row[name] > 0
        for denominator_row in (object_denominator, pair_denominator)
        for name in ("annotation-only", "abstained", "invalid")
    ):
        terminal = "SENSOR_SUPPORT_AND_BIAS_PROFILE_AVAILABLE_WITH_ABSTENTION"
    else:
        terminal = "SENSOR_SUPPORT_AND_BIAS_PROFILE_AVAILABLE_COMPLETE"
    return {
        "schema": RECEIPT_SCHEMA,
        "stage": STAGE,
        "status": "COMPLETE",
        "terminal_state": terminal,
        "validity": "PENDING_INDEPENDENT_VALIDATION",
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "ledger_sha256": ledger_sha,
        "denominators": ledger["denominators"],
        "summary": ledger["summary"],
        "authority": config["authority"],
        "claim": {
            "question_answered": "fraction of computable annotation-derived trajectory units with annotation-conditioned LiDAR support, plus descriptive residual and bias profile",
            "direct_person_center_truth": False,
            "trajectory_accuracy_validated": False,
            "selection_or_safety_authority": False,
        },
    }


def run(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = build_ledger(repo, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    ledger_sha = sha256_bytes(canonical_bytes(ledger))
    return ledger, build_receipt(config, ledger, ledger_sha)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    ledger, receipt = run(repo, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger_path = repo / config["outputs"]["ledger"]
    receipt_path = repo / config["outputs"]["receipt"]
    write_canonical(ledger_path, ledger)
    write_canonical(receipt_path, receipt)
    print(json.dumps({"terminal_state": receipt["terminal_state"], "ledger": str(ledger_path), "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
