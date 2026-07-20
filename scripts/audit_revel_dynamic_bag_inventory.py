#!/usr/bin/env python3
"""Create a deterministic metadata inventory for the public REveL ROS1 Dynamic bag.

The inventory intentionally reports only source-provided topic names, message
types, counts and time span.  A later trajectory audit must explicitly select
and validate Vicon/IMU/LiDAR topics; topic presence alone is not a safety or
metric-motion admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _topic_record(topic: str, connection: Any) -> dict[str, Any]:
    return {
        "topic": topic,
        "msgtype": connection.msgtype,
        "message_count": connection.msgcount,
    }


def audit(dataset_root: Path) -> dict[str, Any]:
    from rosbags.rosbag1 import Reader

    bag = dataset_root / "dynamic.bag"
    if not bag.is_file():
        raise FileNotFoundError(f"missing completed bag: {bag}")
    with Reader(bag) as reader:
        topics = [_topic_record(topic, connection) for topic, connection in reader.topics.items()]
        report = {
            "format": "blindassist_revel_dynamic_bag_inventory_v1",
            "bag": {"name": bag.name, "bytes": bag.stat().st_size},
            "recording": {
                "start_time_ns": reader.start_time,
                "end_time_ns": reader.end_time,
                "duration_s": reader.duration / 1_000_000_000.0,
                "message_count": reader.message_count,
            },
            "topics": sorted(topics, key=lambda item: item["topic"]),
            "admission": {
                "source_topic_inventory_only": True,
                "metric_trajectory_input_admitted": False,
                "reason": "topic and count metadata do not validate message payload, Vicon quality, sensor-body frame, or associations",
            },
            "production_authority": False,
        }
    qa = dataset_root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "revel_dynamic_bag_inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset_root)
    print(json.dumps({"topics": len(report["topics"]), "messages": report["recording"]["message_count"], "duration_s": report["recording"]["duration_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
