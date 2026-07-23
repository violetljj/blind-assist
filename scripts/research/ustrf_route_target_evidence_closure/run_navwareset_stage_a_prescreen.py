#!/usr/bin/env python3
"""Run the frozen one-second NavWareSet packaging/binding micro-canary."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from contract import load_json, sha256_file
from stream_remote_zip_entry import download_range_part
from validate_navwareset_source_prescreen import validate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BlindAssist-USTRF-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        if getattr(response, "status", None) != 200:
            raise RuntimeError(f"JSON endpoint returned HTTP {getattr(response, 'status', None)}")
        payload = response.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024:
        raise RuntimeError("JSON endpoint exceeded 32 MiB cap")
    return json.loads(payload)


def download_bound_file(row: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / row["path"]
    if output.is_file():
        if output.stat().st_size != row["bytes"] or sha256_file(output) != row["content_sha256"]:
            raise RuntimeError(f"existing NavWareSet file differs from frozen binding: {output}")
        return {
            "path": output.as_posix(),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "reused": True,
        }
    result = download_range_part(
        url=row["url"],
        start=0,
        end=int(row["bytes"]) - 1,
        output=output,
        progress=lambda _delta: None,
        request_timeout_seconds=45,
    )
    if result["sha256"] != row["content_sha256"]:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"download SHA differs from frozen content SHA: {row['path']}")
    return {
        "path": output.as_posix(),
        "bytes": result["bytes"],
        "sha256": result["sha256"],
        "reused": result["reused"],
    }


def bag_topic_inventory(path: Path) -> dict[str, Any]:
    with Reader(path) as reader:
        topics = {
            connection.topic: {
                "msgtype": connection.msgtype,
                "msgcount": connection.msgcount,
            }
            for connection in reader.connections
        }
        return {
            "start_timestamp_ns": reader.start_time,
            "end_timestamp_ns": reader.end_time,
            "message_count": reader.message_count,
            "topics": topics,
        }


def decode_first_message(path: Path, topic: str) -> dict[str, Any]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(path) as reader:
        matches = [connection for connection in reader.connections if connection.topic == topic]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one connection for {topic}, got {len(matches)}")
        connection = matches[0]
        if connection.msgtype not in typestore.fielddefs:
            typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for _connection, timestamp, raw in reader.messages(connections=matches):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            return {
                "timestamp_ns": timestamp,
                "msgtype": connection.msgtype,
                "width": int(getattr(message, "width", 0)),
                "height": int(getattr(message, "height", 0)),
                "encoding": str(getattr(message, "encoding", "")),
                "frame_id": str(getattr(getattr(message, "header", None), "frame_id", "")),
            }
    raise RuntimeError(f"topic has no messages: {topic}")


def nearest_topic_age_ms(path: Path, topics: set[str], timestamp_ns: int) -> float:
    nearest: int | None = None
    with Reader(path) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topics]
        if not connections:
            raise RuntimeError(f"missing required pose/TF topics: {sorted(topics)}")
        for _connection, timestamp, _raw in reader.messages(connections=connections):
            age = abs(timestamp - timestamp_ns)
            nearest = age if nearest is None else min(nearest, age)
    if nearest is None:
        raise RuntimeError("pose/TF topics have no messages")
    return nearest / 1e6


def csv_interval(path: Path) -> dict[str, Any]:
    timestamps: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        for row in reader:
            if not row:
                continue
            try:
                timestamps.append(int(row[0]))
            except ValueError:
                continue
    if len(timestamps) < 2:
        raise RuntimeError("pose CSV has fewer than two timestamp rows")
    return {
        "row_count": len(timestamps),
        "start_timestamp_ns": min(timestamps),
        "end_timestamp_ns": max(timestamps),
    }


def annotation_timestamp(path: str) -> int:
    name = Path(path).name.removesuffix(".pcd.json")
    seconds, fraction = name.split(".", 1)
    return int(seconds) * 1_000_000_000 + int(fraction.ljust(9, "0")[:9])


def object_keys(payload: dict[str, Any]) -> set[str]:
    return {
        str(row["key"])
        for row in payload.get("objects", [])
        if isinstance(row, dict) and row.get("key") is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts.local/camera-source-prescreen-r1/navwareset/stage-a"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "artifacts.local/camera-source-prescreen-r1/navwareset/evidence/"
            "stage-a-micro-canary-receipt-r1.json"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    freeze_path = args.freeze_receipt.resolve()
    validation = validate(repo, config_path, freeze_path)
    config = load_json(config_path)
    stage_a = config["stage_a_micro_canary"]
    output_root = (repo / args.output_root).resolve()
    receipt_path = (repo / args.receipt).resolve()
    if receipt_path.exists():
        raise RuntimeError("refusing to overwrite NavWareSet stage A receipt")

    downloads = [download_bound_file(row, output_root) for row in stage_a["files"]]
    by_path = {row["path"]: output_root / row["path"] for row in stage_a["files"]}
    robot_bag = by_path["13_robot.bag"]
    grs_bag = by_path["13_grs.bag"]
    robot_inventory = bag_topic_inventory(robot_bag)
    grs_inventory = bag_topic_inventory(grs_bag)
    robot_rgb_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    robot_info_topic = "/hsrb/head_rgbd_sensor/rgb/camera_info"
    grs_rgb_topic = "/camera/color/image_raw"
    grs_info_topic = "/camera/color/camera_info"
    for topic in (robot_rgb_topic, robot_info_topic, "/tf", "/tf_static"):
        if topic not in robot_inventory["topics"]:
            raise RuntimeError(f"stage A robot bag missing topic: {topic}")
    for topic in (grs_rgb_topic, grs_info_topic, "/rslidar_points"):
        if topic not in grs_inventory["topics"]:
            raise RuntimeError(f"stage A GRS bag missing topic: {topic}")
    robot_rgb = decode_first_message(robot_bag, robot_rgb_topic)
    grs_rgb = decode_first_message(grs_bag, grs_rgb_topic)
    if robot_rgb["width"] <= 0 or robot_rgb["height"] <= 0 or not robot_rgb["encoding"]:
        raise RuntimeError("stage A robot RGB did not decode")
    if grs_rgb["width"] <= 0 or grs_rgb["height"] <= 0 or not grs_rgb["encoding"]:
        raise RuntimeError("stage A GRS RGB did not decode")
    robot_tf_age_ms = nearest_topic_age_ms(robot_bag, {"/tf", "/tf_static"}, robot_rgb["timestamp_ns"])
    overlap_start = max(robot_inventory["start_timestamp_ns"], grs_inventory["start_timestamp_ns"])
    overlap_end = min(robot_inventory["end_timestamp_ns"], grs_inventory["end_timestamp_ns"])
    if overlap_start >= overlap_end:
        raise RuntimeError("stage A robot and GRS bag intervals do not overlap")
    pose_interval = csv_interval(by_path["13_poses/13_robot_and_participants.csv"])
    if max(overlap_start, pose_interval["start_timestamp_ns"]) >= min(
        overlap_end, pose_interval["end_timestamp_ns"]
    ):
        raise RuntimeError("stage A pose CSV does not overlap bag interval")
    load_json(by_path["13_annotated/key_id_map.json"])
    load_json(by_path["13_annotated/meta.json"])
    load_json(by_path["13_annotated/13_grs_to_bot_offset.json"])

    tree = fetch_json(stage_a["official_tree_api_url"])
    annotation_rows = sorted(
        (
            (annotation_timestamp(row["path"]), row)
            for row in tree["tree"]
            if row.get("type") == "blob"
            and row["path"].startswith("13_annotated/scene_13/ann/")
            and row["path"].endswith(".pcd.json")
        ),
        key=lambda item: item[0],
    )
    selected = [row for timestamp, row in annotation_rows if overlap_start <= timestamp <= overlap_end]
    if len(selected) > stage_a["maximum_matching_annotation_json_files"]:
        raise RuntimeError("stage A matching annotation count exceeds frozen cap")
    if len(selected) < 2:
        raise RuntimeError("stage A has fewer than two overlapping annotation frames")
    annotation_receipts = []
    key_sets = []
    for row in selected:
        url = (
            "https://raw.githubusercontent.com/anr-navware/NavWareSet-Tutorials/"
            f"{config['official_tutorial_commit']}/{row['path']}"
        )
        payload = fetch_json(url)
        keys = object_keys(payload)
        key_sets.append(keys)
        annotation_receipts.append(
            {
                "path": row["path"],
                "git_blob_sha1": row["sha"],
                "bytes": row["size"],
                "object_keys": sorted(keys),
            }
        )
    stable_pair_count = sum(
        bool(left) and left == right for left, right in zip(key_sets, key_sets[1:])
    )
    if stable_pair_count < 1:
        raise RuntimeError("stage A has no consecutive annotation frames with a stable nonempty UUID set")

    receipt = {
        "schema": "blindassist_ustrf_route_target_navwareset_stage_a_micro_canary_receipt_r1",
        "authority": "packaging_and_cross_modal_binding_feasibility_only_not_terminal_clear_or_source_admission",
        "config_sha256": sha256_file(config_path),
        "freeze_receipt_sha256": validation["freeze_receipt_sha256"],
        "downloads": downloads,
        "robot_bag": {
            **robot_inventory,
            "first_rgb": robot_rgb,
            "nearest_tf_age_ms": robot_tf_age_ms,
        },
        "grs_bag": {
            **grs_inventory,
            "first_rgb": grs_rgb,
        },
        "robot_grs_overlap": {
            "start_timestamp_ns": overlap_start,
            "end_timestamp_ns": overlap_end,
            "duration_ms": (overlap_end - overlap_start) / 1e6,
        },
        "pose_csv": pose_interval,
        "matching_annotations": annotation_receipts,
        "stable_consecutive_uuid_pair_count": stable_pair_count,
        "candidate_outputs_executed": False,
        "app_detector_or_event_outputs_exposed": False,
        "decision": "PASS_AUTHORIZE_NAVWARESET_STAGE_B_ONLY",
        "source_admission": False,
        "android_shadow": "closed",
        "h2_depth_ttc_route_risk_flip": "closed",
    }
    write_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["decision"], "receipt_sha256": sha256_file(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
