"""Acquire one deterministic AV2 validation log shard and emit DTR source rows.

Native 3-D cuboids remain evaluator truth.  Current raw LiDAR and ego pose are
written under ``causal_input``; no future frame or native box is copied there.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import certifi
from pyarrow import feather

BUCKET_LIST_URL = "https://s3.amazonaws.com/argoverse/"
OBJECT_ROOT = "https://s3.amazonaws.com/argoverse/"
SENSOR_VAL_PREFIX = "datasets/av2/sensor/val/"
USER_AGENT = "BlindAssist-AV2-source-canary/1.0"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def fetch_bytes(url: str, timeout_s: float = 90.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s, context=ssl_context()) as response:
        return response.read()


def list_objects(prefix: str, delimiter: str | None = None) -> dict[str, Any]:
    query: dict[str, str] = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    if delimiter:
        query["delimiter"] = delimiter
    payload = fetch_bytes(BUCKET_LIST_URL + "?" + urllib.parse.urlencode(query))
    root = ET.fromstring(payload)
    objects = []
    for node in root.findall("s3:Contents", S3_NS):
        key = node.findtext("s3:Key", default="", namespaces=S3_NS)
        size = int(node.findtext("s3:Size", default="0", namespaces=S3_NS))
        etag = node.findtext("s3:ETag", default="", namespaces=S3_NS).strip('"')
        objects.append({"key": key, "size": size, "etag": etag})
    prefixes = [
        node.text or "" for node in root.findall("s3:CommonPrefixes/s3:Prefix", S3_NS)
    ]
    return {
        "objects": objects,
        "prefixes": prefixes,
        "is_truncated": root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS)
        == "true",
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_object(spec: dict[str, Any], root: Path, prefix: str) -> dict[str, Any]:
    key = str(spec["key"])
    relative = Path(key.removeprefix(prefix))
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != int(spec["size"]):
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(fetch_bytes(OBJECT_ROOT + urllib.parse.quote(key, safe="/"), 180.0))
        if temporary.stat().st_size != int(spec["size"]):
            raise ValueError(f"size mismatch for {key}")
        temporary.replace(target)
    etag = str(spec.get("etag", ""))
    if len(etag) == 32 and "-" not in etag and md5_file(target) != etag:
        raise ValueError(f"ETag/MD5 mismatch for {key}")
    return {
        "key": key,
        "relative_path": str(relative).replace("\\", "/"),
        "size_bytes": target.stat().st_size,
        "etag": etag,
        "sha256": sha256_file(target),
        "path": str(target.resolve()),
    }


def closest_pose(poses: list[dict[str, Any]], timestamp_ns: int) -> dict[str, Any]:
    return min(poses, key=lambda row: abs(int(row["timestamp_ns"]) - timestamp_ns))


def run(
    data_dir: Path,
    output_path: Path,
    adapter_path: Path,
    log_id: str | None,
    max_lidar_sweeps: int,
) -> dict[str, Any]:
    log_listing = list_objects(SENSOR_VAL_PREFIX, delimiter="/")
    if log_listing["is_truncated"] or not log_listing["prefixes"]:
        raise ValueError("could not enumerate AV2 validation logs")
    available_logs = sorted(prefix.rstrip("/").split("/")[-1] for prefix in log_listing["prefixes"])
    selected_log = log_id or available_logs[0]
    if selected_log not in available_logs:
        raise ValueError(f"unknown validation log: {selected_log}")

    log_prefix = f"{SENSOR_VAL_PREFIX}{selected_log}/"
    metadata_names = {
        "annotations.feather",
        "city_SE3_egovehicle.feather",
        "calibration/egovehicle_SE3_sensor.feather",
        "calibration/intrinsics.feather",
    }
    full_listing = list_objects(log_prefix)
    metadata_specs = [
        item for item in full_listing["objects"] if item["key"].removeprefix(log_prefix) in metadata_names
    ]
    if len(metadata_specs) != len(metadata_names):
        raise ValueError("AV2 log is missing required metadata files")

    lidar_prefix = log_prefix + "sensors/lidar/"
    lidar_listing = list_objects(lidar_prefix)
    if lidar_listing["is_truncated"]:
        raise ValueError("LiDAR listing unexpectedly truncated")
    lidar_specs = sorted(lidar_listing["objects"], key=lambda item: item["key"])
    selected_lidar_specs = lidar_specs[:max_lidar_sweeps]
    if len(selected_lidar_specs) < max_lidar_sweeps:
        raise ValueError("not enough LiDAR sweeps")

    selected_root = data_dir / selected_log
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(
            pool.map(
                lambda spec: download_object(spec, selected_root, log_prefix),
                metadata_specs + selected_lidar_specs,
            )
        )
    receipt_by_relative = {item["relative_path"]: item for item in receipts}

    annotations = feather.read_table(selected_root / "annotations.feather").to_pylist()
    poses = feather.read_table(selected_root / "city_SE3_egovehicle.feather").to_pylist()
    sweep_timestamps = [int(Path(item["key"]).stem) for item in selected_lidar_specs]
    sweep_set = set(sweep_timestamps)
    boxes_by_timestamp: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    track_timestamps: defaultdict[str, list[int]] = defaultdict(list)
    route_tube_box_count = 0
    for box in annotations:
        timestamp = int(box["timestamp_ns"])
        if timestamp not in sweep_set:
            continue
        category = str(box["category"])
        category_counts[category] += 1
        track_uuid = str(box["track_uuid"])
        track_timestamps[track_uuid].append(timestamp)
        forward = float(box["tx_m"])
        left = float(box["ty_m"])
        radius = 0.5 * math.hypot(float(box["length_m"]), float(box["width_m"]))
        if 0.0 <= forward <= 12.0 and abs(left) <= 0.65 + radius:
            route_tube_box_count += 1
        boxes_by_timestamp[timestamp].append(
            {
                "track_id": track_uuid,
                "category": category,
                "forward_m": forward,
                "left_m": left,
                "height_m": float(box["height_m"]),
                "radius_m": radius,
                "num_interior_pts": int(box["num_interior_pts"]),
            }
        )

    adapter_rows = []
    first_timestamp = sweep_timestamps[0]
    total_points = 0
    route_prism_points = 0
    pose_max_delta_ns = 0
    for timestamp, spec in zip(sweep_timestamps, selected_lidar_specs):
        relative = spec["key"].removeprefix(log_prefix)
        lidar_path = selected_root / relative
        table = feather.read_table(lidar_path, columns=["x", "y", "z"])
        x_values = table.column("x").to_pylist()
        y_values = table.column("y").to_pylist()
        z_values = table.column("z").to_pylist()
        point_count = len(x_values)
        prism_count = sum(
            0.0 <= float(x) <= 12.0 and abs(float(y)) <= 0.65 and -1.2 <= float(z) <= 2.5
            for x, y, z in zip(x_values, y_values, z_values)
        )
        total_points += point_count
        route_prism_points += prism_count
        pose = closest_pose(poses, timestamp)
        pose_delta = abs(int(pose["timestamp_ns"]) - timestamp)
        pose_max_delta_ns = max(pose_max_delta_ns, pose_delta)
        adapter_rows.append(
            {
                "schema_version": "blindassist-dtr-av2-causal-frame-source-v1",
                "source_log_id": selected_log,
                "time_s": (timestamp - first_timestamp) / 1_000_000_000.0,
                "causal_input": {
                    "ego_pose": {
                        "source_timestamp_ns": int(pose["timestamp_ns"]),
                        "translation_city_m": [
                            float(pose["tx_m"]),
                            float(pose["ty_m"]),
                            float(pose["tz_m"]),
                        ],
                        "quaternion_wxyz": [
                            float(pose["qw"]),
                            float(pose["qx"]),
                            float(pose["qy"]),
                            float(pose["qz"]),
                        ],
                    },
                    "raw_lidar": {
                        "timestamp_ns": timestamp,
                        "relative_path": relative,
                        "sha256": receipt_by_relative[relative]["sha256"],
                        "point_count": point_count,
                        "route_prism_candidate_point_count": prism_count,
                    },
                    "observation_available": True,
                    "time_since_seen_s": 0.0,
                },
                "evaluator_truth": {
                    "native_boxes": boxes_by_timestamp.get(timestamp, []),
                    "future_frames_used_as_input": False,
                },
            }
        )

    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    with adapter_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in adapter_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    tracks_with_multiple_observations = sum(len(values) >= 2 for values in track_timestamps.values())
    result = {
        "schema_version": "blindassist-dtr-av2-raw-log-source-canary-v1",
        "source": {
            "dataset": "Argoverse 2 Sensor validation",
            "bucket_prefix": log_prefix,
            "selected_log_id": selected_log,
            "selection_policy": "First lexicographic public validation log; no outcome access.",
            "available_validation_logs": len(available_logs),
            "license": "CC-BY-NC-SA-4.0",
        },
        "download": {
            "downloaded_object_count": len(receipts),
            "downloaded_bytes": sum(item["size_bytes"] for item in receipts),
            "receipts": receipts,
        },
        "metadata": {
            "annotation_rows_in_selected_window": sum(category_counts.values()),
            "category_counts": dict(category_counts.most_common()),
            "unique_tracks_in_selected_window": len(track_timestamps),
            "tracks_with_multiple_observations": tracks_with_multiple_observations,
            "route_tube_candidate_box_count": route_tube_box_count,
        },
        "raw_lidar": {
            "available_sweeps_in_log": len(lidar_specs),
            "selected_consecutive_sweeps": len(selected_lidar_specs),
            "selected_duration_s": (sweep_timestamps[-1] - sweep_timestamps[0]) / 1_000_000_000.0,
            "total_points": total_points,
            "route_prism_candidate_point_count": route_prism_points,
            "max_pose_alignment_delta_ns": pose_max_delta_ns,
        },
        "adapter": {
            "path": str(adapter_path.resolve()),
            "row_count": len(adapter_rows),
            "sha256": sha256_file(adapter_path),
            "native_boxes_are_evaluator_only": True,
        },
        "verdict": (
            "AV2_RAW_LOG_SOURCE_ADMITTED"
            if adapter_rows and total_points and sum(category_counts.values())
            else "AV2_RAW_LOG_SOURCE_NOT_ADMITTED"
        ),
        "decision": (
            "AV2 supports a detector-independent current-LiDAR source at exact-log "
            "granularity. Freeze a multi-log Development roster before measuring DTR gain."
        ),
        "claim_ceiling": (
            "One automotive validation-log source canary. No DTR improvement, wearer, "
            "natural-distribution, product, or safety claim."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--log-id")
    parser.add_argument("--max-lidar-sweeps", type=int, default=32)
    args = parser.parse_args()
    if args.max_lidar_sweeps < 2:
        parser.error("--max-lidar-sweeps must be at least 2")
    result = run(
        args.data_dir,
        args.output,
        args.adapter_output,
        args.log_id,
        args.max_lidar_sweeps,
    )
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
