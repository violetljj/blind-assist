#!/usr/bin/env python3
"""List AV2 Sensor S3 metadata without downloading dataset payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ENDPOINT = "https://argoverse.s3.amazonaws.com/"
DATASET_PREFIX = "datasets/av2/sensor/"
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
SPLITS = ("train", "val", "test")


def _request(params: dict[str, str]) -> ET.Element:
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"S3 list failed: {response.status}")
        return ET.fromstring(response.read())


def list_prefixes(prefix: str) -> list[str]:
    root = _request(
        {
            "list-type": "2",
            "delimiter": "/",
            "prefix": prefix,
            "max-keys": "1000",
        }
    )
    if root.findtext("s3:IsTruncated", namespaces=XML_NS) != "false":
        raise AssertionError(f"unexpected truncated prefix inventory: {prefix}")
    return [
        node.text or ""
        for node in root.findall("s3:CommonPrefixes/s3:Prefix", XML_NS)
    ]


def list_objects(prefix: str) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    continuation: str | None = None
    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if continuation:
            params["continuation-token"] = continuation
        root = _request(params)
        for node in root.findall("s3:Contents", XML_NS):
            objects.append(
                {
                    "key": node.findtext("s3:Key", namespaces=XML_NS),
                    "bytes": int(node.findtext("s3:Size", namespaces=XML_NS) or "0"),
                    "etag": (node.findtext("s3:ETag", namespaces=XML_NS) or "").strip('"'),
                    "last_modified": node.findtext(
                        "s3:LastModified", namespaces=XML_NS
                    ),
                }
            )
        if root.findtext("s3:IsTruncated", namespaces=XML_NS) == "false":
            return objects
        continuation = root.findtext(
            "s3:NextContinuationToken", namespaces=XML_NS
        )
        if not continuation:
            raise AssertionError(f"missing continuation token: {prefix}")


def _member_class(relative: str) -> str:
    if relative.startswith("sensors/cameras/"):
        parts = relative.split("/")
        return f"camera:{parts[2]}"
    if relative.startswith("sensors/lidar/"):
        return "lidar"
    if relative.startswith("calibration/"):
        return "calibration"
    if relative.startswith("map/"):
        return "map"
    return relative


def summarize_log(split: str, log_id: str) -> dict[str, object]:
    prefix = f"{DATASET_PREFIX}{split}/{log_id}/"
    objects = list_objects(prefix)
    classes = Counter(
        _member_class(str(item["key"])[len(prefix) :]) for item in objects
    )
    keys = {str(item["key"])[len(prefix) :] for item in objects}
    return {
        "split": split,
        "log_id": log_id,
        "capture_cluster_id": log_id,
        "object_count": len(objects),
        "total_bytes": sum(int(item["bytes"]) for item in objects),
        "member_class_counts": dict(sorted(classes.items())),
        "required_member_presence": {
            "annotations.feather": "annotations.feather" in keys,
            "city_SE3_egovehicle.feather": "city_SE3_egovehicle.feather" in keys,
            "calibration/egovehicle_SE3_sensor.feather": (
                "calibration/egovehicle_SE3_sensor.feather" in keys
            ),
            "calibration/intrinsics.feather": (
                "calibration/intrinsics.feather" in keys
            ),
        },
        "metadata_identity_sha256": hashlib.sha256(
            "\n".join(
                f"{item['key']}\t{item['bytes']}\t{item['etag']}"
                for item in sorted(objects, key=lambda row: str(row["key"]))
            ).encode("utf-8")
        ).hexdigest(),
        "metadata_only": True,
    }


def build_receipt() -> dict[str, object]:
    split_logs: dict[str, list[str]] = {}
    for split in SPLITS:
        prefix = f"{DATASET_PREFIX}{split}/"
        split_logs[split] = [
            item.removeprefix(prefix).removesuffix("/")
            for item in list_prefixes(prefix)
        ]

    all_identity_lines = [
        f"{split}\t{log_id}"
        for split in SPLITS
        for log_id in split_logs[split]
    ]
    return {
        "schema_version": "av2_sensor_official_s3_inventory_r0",
        "source_id": "ARGOVERSE_2_SENSOR",
        "endpoint": ENDPOINT,
        "dataset_prefix": DATASET_PREFIX,
        "transport": "S3_LIST_OBJECTS_V2_NO_SIGN_REQUEST",
        "payload_downloaded": False,
        "split_counts": {split: len(split_logs[split]) for split in SPLITS},
        "log_count": sum(len(logs) for logs in split_logs.values()),
        "log_identity_sha256": hashlib.sha256(
            "\n".join(all_identity_lines).encode("utf-8")
        ).hexdigest(),
        "log_ids": split_logs,
        "deterministic_first_log_metadata": [
            summarize_log(split, split_logs[split][0]) for split in SPLITS
        ],
        "admission": "HOLD_R0_ADMISSION",
        "blocking_gaps": [
            "counterfactual cells are not source-native metadata fields",
            "role split is not frozen",
            "10 Hz object truth to 20 Hz camera join policy is not frozen",
            "vehicle-mounted domain cannot substitute for head-mounted diversity",
        ],
        "authority": {
            "may_build_cell_preview_bundle": True,
            "may_decode_signal": False,
            "may_freeze_role_split": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "log_count": receipt["log_count"],
                "split_counts": receipt["split_counts"],
                "payload_downloaded": receipt["payload_downloaded"],
                "admission": receipt["admission"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
