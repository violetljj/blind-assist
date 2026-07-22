from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from contract import read_json, sha256, write_json
from idsia_msmpt_calibration import calibration_arrays, register_depth_to_color
from prescreen_third_source_gt import (
    compose_transform_chain,
    timestamp_ns,
    transform_entry,
    validate_front_color_optical,
)


RECEIPT_SCHEMA = "blindassist_ustrf_idsia_msmpt_rgbd_preparation_v1"


def camera_info_dict(message: Any) -> dict[str, Any]:
    return {
        "frame_id": message.header.frame_id,
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": message.distortion_model,
        "D": [float(value) for value in message.d],
        "K": [float(value) for value in message.k],
        "R": [float(value) for value in message.r],
        "P": [float(value) for value in message.p],
        "binning_x": int(message.binning_x),
        "binning_y": int(message.binning_y),
    }


def associate_monotonic_ns(first: list[int], second: list[int], maximum_delta_ns: int) -> list[tuple[int, int]]:
    if maximum_delta_ns <= 0 or len(first) < 2 or len(second) < 2:
        raise ValueError("invalid IDSIA RGB-depth association input")
    first = sorted(first)
    second = sorted(second)
    if any(a >= b for a, b in zip(first, first[1:])) or any(a >= b for a, b in zip(second, second[1:])):
        raise ValueError("duplicate IDSIA RGB or depth header timestamp")
    result: list[tuple[int, int]] = []
    depth_index = 0
    for color_stamp in first:
        while depth_index + 1 < len(second) and abs(second[depth_index + 1] - color_stamp) < abs(second[depth_index] - color_stamp):
            depth_index += 1
        depth_stamp = second[depth_index]
        if abs(depth_stamp - color_stamp) <= maximum_delta_ns:
            result.append((color_stamp, depth_stamp))
            depth_index += 1
            if depth_index >= len(second):
                break
    if not result or any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(result, result[1:])):
        raise ValueError("empty or non-monotonic IDSIA RGB-depth association")
    return result


def matrix_json(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def update_hash_chain(digest: Any, relative: str, file_hash: str) -> None:
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(file_hash.encode("ascii"))
    digest.update(b"\n")


def prepare(
    repo: Path,
    discovery_path: Path,
    sequence_input_path: Path,
    prescreen_path: Path,
    bag_root: Path,
    output: Path,
) -> dict[str, Any]:
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise ValueError("rosbags runtime is required for IDSIA RGB-D extraction") from error

    repo, bag_root, output = repo.resolve(), bag_root.resolve(), output.resolve()
    work = output.with_name(output.name + ".incomplete")
    if output.exists() or work.exists():
        raise ValueError(f"refusing to overwrite prepared IDSIA package: {output}")
    discovery = read_json(discovery_path)
    sequence_input = read_json(sequence_input_path)
    prescreen = read_json(prescreen_path)
    discovery_sha = sha256(discovery_path)
    if sequence_input.get("discovery_manifest_sha256") != discovery_sha:
        raise ValueError("IDSIA sequence input is not bound to discovery manifest")
    sequence_id = sequence_input["sequence_id"]
    if (
        prescreen.get("config_sha256") != discovery_sha
        or prescreen.get("sequence_id") != sequence_id
        or prescreen.get("gt_route_prescreen_passed") is not True
        or prescreen.get("source_count_credit") != 0
        or prescreen.get("evaluator_ran") is not False
    ):
        raise ValueError("IDSIA full preparation requires a passing reject-only GT receipt")
    bag_member = sequence_input["remote_bag_member"]
    bag_path, metadata_path = bag_root / "out_0.db3", bag_root / "metadata.yaml"
    if bag_path.stat().st_size != int(bag_member["file_size"]) or sha256(bag_path) != bag_member["raw_db3_sha256"]:
        raise ValueError("IDSIA bag hash or size mismatch")
    if sha256(metadata_path) != bag_member["metadata_sha256"]:
        raise ValueError("IDSIA metadata hash mismatch")

    prereg = read_json(repo / discovery["frozen_r3_prereg"]["path"])
    maximum_rgb_depth_delta_ms = float(prereg["synchronization"]["maximum_rgb_depth_delta_ms"])
    minimum_aligned_fraction = float(prereg["synchronization"]["minimum_source_aligned_fraction"])
    rgb_topic = "/camera_1/color/image_raw"
    depth_topic = "/camera_1/depth/image_rect_raw"
    color_info_topic = "/camera_1/color/camera_info"
    depth_info_topic = "/camera_1/depth/camera_info"
    required_topics = {rgb_topic, depth_topic, color_info_topic, depth_info_topic, "/tf", "/tf_static"}
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    color_stamps: list[int] = []
    depth_stamps: list[int] = []
    gt_rows: list[tuple[int, list[float]]] = []
    static_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    color_calibration: dict[str, Any] | None = None
    depth_calibration: dict[str, Any] | None = None
    topic_counts = {topic: 0 for topic in required_topics}

    with Reader(bag_root) as reader:
        missing = required_topics - {connection.topic for connection in reader.connections}
        if missing:
            raise ValueError(f"missing IDSIA RGB-D topics: {sorted(missing)}")
        connections = [connection for connection in reader.connections if connection.topic in required_topics]
        for connection, _, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_cdr(rawdata, connection.msgtype)
            topic = connection.topic
            topic_counts[topic] += 1
            if topic == rgb_topic:
                color_stamps.append(timestamp_ns(message.header.stamp))
            elif topic == depth_topic:
                depth_stamps.append(timestamp_ns(message.header.stamp))
            elif topic == color_info_topic:
                row = camera_info_dict(message)
                if color_calibration is not None and row != color_calibration:
                    raise ValueError("IDSIA color CameraInfo changed within sequence")
                color_calibration = row
            elif topic == depth_info_topic:
                row = camera_info_dict(message)
                if depth_calibration is not None and row != depth_calibration:
                    raise ValueError("IDSIA depth CameraInfo changed within sequence")
                depth_calibration = row
            elif topic == "/tf":
                for transform in message.transforms:
                    if transform.header.frame_id == "world" and transform.child_frame_id == "chair":
                        value = transform.transform
                        gt_rows.append((timestamp_ns(transform.header.stamp), [
                            float(value.translation.x), float(value.translation.y), float(value.translation.z),
                            float(value.rotation.x), float(value.rotation.y), float(value.rotation.z), float(value.rotation.w),
                        ]))
            else:
                for transform in message.transforms:
                    entry = transform_entry(transform)
                    pair = (entry["parent"], entry["child"])
                    if pair in static_by_pair and static_by_pair[pair] != entry:
                        raise ValueError(f"conflicting IDSIA static transform: {pair}")
                    static_by_pair[pair] = entry

    if color_calibration is None or depth_calibration is None:
        raise ValueError("missing IDSIA camera calibration")
    calibration_arrays(color_calibration, "color")
    calibration_arrays(depth_calibration, "depth")
    body_to_color_chain = ["chair", "camera_1_link", "camera_1_color_frame", "camera_1_color_optical_frame"]
    body_to_depth_chain = ["chair", "camera_1_link", "camera_1_depth_frame", "camera_1_depth_optical_frame"]
    static_entries = list(static_by_pair.values())
    body_from_color = compose_transform_chain(static_entries, body_to_color_chain)
    body_from_depth = compose_transform_chain(static_entries, body_to_depth_chain)
    forward_axis = validate_front_color_optical(body_from_color)
    color_from_depth = np.linalg.inv(body_from_color) @ body_from_depth

    associations = associate_monotonic_ns(color_stamps, depth_stamps, int(round(maximum_rgb_depth_delta_ms * 1e6)))
    associated_fraction = len(associations) / max(len(color_stamps), len(depth_stamps))
    if associated_fraction < minimum_aligned_fraction:
        raise ValueError(f"IDSIA RGB-depth association fraction {associated_fraction:.6f} < {minimum_aligned_fraction}")
    color_index = {stamp: index for index, (stamp, _) in enumerate(associations)}
    depth_index = {stamp: index for index, (_, stamp) in enumerate(associations)}
    (work / "color/images").mkdir(parents=True)
    (work / "aligned_depth/images").mkdir(parents=True)
    color_written: set[int] = set()
    depth_written: set[int] = set()
    color_chain, aligned_chain = hashlib.sha256(), hashlib.sha256()
    valid_fractions: list[float] = []
    saturated_pixel_count = 0
    out_of_range_pixel_count = 0
    with Reader(bag_root) as reader:
        connections = [connection for connection in reader.connections if connection.topic in {rgb_topic, depth_topic}]
        for connection, _, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_cdr(rawdata, connection.msgtype)
            stamp = timestamp_ns(message.header.stamp)
            if connection.topic == rgb_topic and stamp in color_index:
                index = color_index[stamp]
                if message.encoding != "rgb8" or int(message.is_bigendian) != 0 or int(message.step) < int(message.width) * 3:
                    raise ValueError("unsupported IDSIA RGB image encoding")
                raw = np.frombuffer(message.data, dtype=np.uint8).reshape(int(message.height), int(message.step))
                rgb = raw[:, : int(message.width) * 3].reshape(int(message.height), int(message.width), 3)
                relative = f"color/images/{index:06d}.png"
                Image.fromarray(rgb, mode="RGB").save(work / relative, compress_level=3)
                update_hash_chain(color_chain, relative, sha256(work / relative))
                color_written.add(index)
            elif connection.topic == depth_topic and stamp in depth_index:
                index = depth_index[stamp]
                if message.encoding != "16UC1" or int(message.is_bigendian) != 0 or int(message.step) < int(message.width) * 2:
                    raise ValueError("unsupported IDSIA depth image encoding")
                raw_bytes = np.frombuffer(message.data, dtype=np.uint8).reshape(int(message.height), int(message.step))
                depth_raw = raw_bytes[:, : int(message.width) * 2].copy().view("<u2").reshape(int(message.height), int(message.width))
                saturated = depth_raw == np.iinfo(np.uint16).max
                saturated_pixel_count += int(np.count_nonzero(saturated))
                if np.any(saturated):
                    depth_raw = depth_raw.copy()
                    depth_raw[saturated] = 0
                aligned_m = register_depth_to_color(depth_raw, 1000.0, depth_calibration, color_calibration, color_from_depth)
                aligned_units = np.rint(aligned_m * 1000.0)
                invalid_aligned = ~np.isfinite(aligned_units) | (aligned_units < 0)
                out_of_range = aligned_units > 65535
                out_of_range_pixel_count += int(np.count_nonzero(out_of_range))
                if np.any(invalid_aligned):
                    aligned_units = aligned_units.copy()
                    aligned_units[invalid_aligned] = 0
                if np.any(out_of_range):
                    aligned_units = aligned_units.copy()
                    aligned_units[out_of_range] = 0
                aligned_u16 = aligned_units.astype(np.uint16)
                relative = f"aligned_depth/images/{index:06d}.png"
                Image.fromarray(aligned_u16).save(work / relative, compress_level=3)
                update_hash_chain(aligned_chain, relative, sha256(work / relative))
                valid_fractions.append(float(np.count_nonzero(aligned_u16)) / float(aligned_u16.size))
                depth_written.add(index)
                if len(depth_written) % 100 == 0:
                    print(json.dumps({"prepared_frames": len(depth_written), "total_frames": len(associations)}), flush=True)

    expected_indices = set(range(len(associations)))
    if color_written != expected_indices or depth_written != expected_indices:
        raise ValueError("IDSIA extraction did not materialize every associated frame")
    color_rows = [f"{rgb / 1e9:.9f} color/images/{index:06d}.png\n" for index, (rgb, _) in enumerate(associations)]
    depth_rows = [f"{depth / 1e9:.9f} aligned_depth/images/{index:06d}.png\n" for index, (_, depth) in enumerate(associations)]
    (work / "color.txt").write_text("".join(color_rows), encoding="utf-8")
    (work / "aligned_depth.txt").write_text("".join(depth_rows), encoding="utf-8")
    gt_rows.sort(key=lambda row: row[0])
    if len(gt_rows) < 2 or any(a[0] >= b[0] for a, b in zip(gt_rows, gt_rows[1:])):
        raise ValueError("invalid IDSIA OptiTrack trajectory")
    (work / "groundtruth.txt").write_text("".join(
        f"{stamp / 1e9:.9f} " + " ".join(f"{value:.12g}" for value in pose) + "\n" for stamp, pose in gt_rows
    ), encoding="utf-8")
    calibration = {
        "color": color_calibration,
        "depth": depth_calibration,
        "body_from_color_optical": matrix_json(body_from_color),
        "body_from_depth_optical": matrix_json(body_from_depth),
        "color_from_depth_optical": matrix_json(color_from_depth),
        "body_to_color_chain": body_to_color_chain,
        "body_to_depth_chain": body_to_depth_chain,
        "static_transforms": static_entries,
    }
    write_json(work / "calibration.json", calibration)
    deltas_ms = [abs(rgb - depth) / 1e6 for rgb, depth in associations]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "dataset_id": "idsia_msmpt_ground_robot",
        "sequence_id": sequence_id,
        "selected_camera": "camera_1",
        "discovery_manifest_sha256": discovery_sha,
        "sequence_input_sha256": sha256(sequence_input_path),
        "gt_prescreen_sha256": sha256(prescreen_path),
        "bag_sha256": sha256(bag_path),
        "metadata_sha256": sha256(metadata_path),
        "frame_count": len(associations),
        "original_color_count": len(color_stamps),
        "original_depth_count": len(depth_stamps),
        "associated_fraction": associated_fraction,
        "maximum_rgb_depth_delta_ms": maximum_rgb_depth_delta_ms,
        "rgb_depth_delta_ms_p95": float(np.quantile(deltas_ms, 0.95)),
        "rgb_depth_delta_ms_max": max(deltas_ms),
        "minimum_aligned_valid_depth_fraction": min(valid_fractions),
        "median_aligned_valid_depth_fraction": float(np.median(valid_fractions)),
        "depth_encoding": "uint16_png_z_meters",
        "depth_scale_units_per_meter": 1000.0,
        "depth_scale_basis": "RealSense ROS 16UC1 millimeter convention",
        "raw_depth_invalid_values": [0, 65535],
        "raw_depth_saturated_pixel_count": saturated_pixel_count,
        "aligned_depth_unrepresentable_pixel_count": out_of_range_pixel_count,
        "depth_registered_to_color": True,
        "registration_hole_fill": False,
        "registration_collision_policy": "nearest_z",
        "transform_convention": "parent_T_child",
        "body_to_color_chain": body_to_color_chain,
        "body_to_depth_chain": body_to_depth_chain,
        "base_optical_forward_axis": forward_axis.tolist(),
        "calibration_sha256": sha256(work / "calibration.json"),
        "ground_truth_sha256": sha256(work / "groundtruth.txt"),
        "color_hash_chain_sha256": color_chain.hexdigest(),
        "aligned_depth_hash_chain_sha256": aligned_chain.hexdigest(),
        "candidate_alerts_used": False,
        "source_count_credit": 0,
        "evaluator_ran": False,
        "production_authority": False,
    }
    write_json(work / "preparation_receipt.json", receipt)
    work.rename(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--sequence-input", type=Path, required=True)
    parser.add_argument("--gt-prescreen", type=Path, required=True)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = prepare(args.repo, args.discovery.resolve(), args.sequence_input.resolve(), args.gt_prescreen.resolve(), args.bag_root, args.output)
        print(json.dumps(receipt))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
