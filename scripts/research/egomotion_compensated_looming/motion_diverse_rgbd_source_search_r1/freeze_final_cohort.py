"""Freeze the first complete 2-positive + 2-below development cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--eth3d-batch", type=Path, required=True)
    parser.add_argument("--eth3d-result", type=Path, required=True)
    parser.add_argument("--tum-receipt", type=Path, required=True)
    parser.add_argument("--tartanair-result", type=Path, required=True)
    parser.add_argument("--tartanair-extract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "contract": args.contract.resolve(),
        "eth3d_batch": args.eth3d_batch.resolve(),
        "eth3d_result": args.eth3d_result.resolve(),
        "tum_receipt": args.tum_receipt.resolve(),
        "tartanair_result": args.tartanair_result.resolve(),
        "tartanair_extract": args.tartanair_extract.resolve(),
    }
    contract = load(paths["contract"])
    eth3d_batch = load(paths["eth3d_batch"])
    eth3d_result = load(paths["eth3d_result"])
    tum = load(paths["tum_receipt"])
    tartanair = load(paths["tartanair_result"])
    tartanair_extract = load(paths["tartanair_extract"])
    eth3d_id = "desk_changing_1@4065.364250422"
    tartanair_id = "japanesealley/Hard/P002@000260"
    eth3d_summary = next(
        row for row in eth3d_result["window_summaries"] if row["window_id"] == eth3d_id
    )
    eth3d_window = next(row for row in eth3d_batch["windows"] if row["window_id"] == eth3d_id)
    if eth3d_summary["role"] != "POSITIVE_APPROACH_WINDOW":
        raise ValueError("ETH3D_POSITIVE_ROLE")
    if tartanair["terminal"] != "TARTANAIR_SYNTHETIC_POSITIVE_ANCHOR_ADMITTED":
        raise ValueError("TARTANAIR_POSITIVE_TERMINAL")
    if tartanair["window_summary"]["window_id"] != tartanair_id:
        raise ValueError("TARTANAIR_POSITIVE_ID")
    if tartanair_extract["window"]["window_id"] != tartanair_id:
        raise ValueError("TARTANAIR_EXTRACT_ID")
    if tartanair["rgb_bytes_accessed"] != 0 or tartanair_extract["rgb_bytes_accessed"] != 0:
        raise ValueError("EARLY_TARTANAIR_RGB_ACCESS")
    tum_rows = tum["window_summaries"]
    if [row["window_id"] for row in tum_rows] != ["TUM_RGBD_FR2_RPY@2", "TUM_RGBD_FR2_RPY@7"]:
        raise ValueError("TUM_BELOW_IDS")
    if any(row["role"] != "BELOW_TRIGGER_REFERENCE_WINDOW" for row in tum_rows):
        raise ValueError("TUM_BELOW_ROLES")
    windows = [
        {
            "window_id": eth3d_id,
            "source_id": "ETH3D_SLAM_DESK_CHANGING_1",
            "source_kind": "REAL_DEVELOPMENT_SOURCE",
            "role": "POSITIVE_APPROACH_WINDOW",
            "geometry_summary": eth3d_summary,
            "depth_members": eth3d_window["depth_members"],
            "rgb_acquisition": {
                "transport": "REMOTE_RANGE_ZIP_MINIMUM_MEMBERS",
                "archive": "desk_changing_1_mono.zip",
                "member_rule": "replace /depth/ with /rgb/ for frozen source timestamps",
            },
        },
        {
            "window_id": tartanair_id,
            "source_id": "TARTANAIR_JAPANESEALLEY_HARD",
            "source_kind": "SYNTHETIC_DEVELOPMENT_ANCHOR",
            "role": "POSITIVE_APPROACH_WINDOW",
            "geometry_summary": tartanair["window_summary"],
            "frame_ids": tartanair_extract["window"]["frame_ids"],
            "rgb_acquisition": {
                "transport": "LOCAL_TAR_MINIMUM_MEMBERS",
                "member_rule": "<trajectory>/<frame_id>_rgb.png",
            },
        },
        {
            "window_id": "TUM_RGBD_FR2_RPY@2",
            "source_id": "TUM_RGBD_FR2_RPY",
            "source_kind": "BURNED_REAL_DEVELOPMENT_ANCHOR",
            "role": "BELOW_TRIGGER_REFERENCE_WINDOW",
            "window_index": 2,
            "start_timestamp_s": "1311867739",
            "end_timestamp_s": "1311867749",
            "geometry_summary": tum_rows[0],
            "rgb_acquisition": {"transport": "PREEXISTING_LOCAL_SOURCE_NATIVE_RGB"},
        },
        {
            "window_id": "TUM_RGBD_FR2_RPY@7",
            "source_id": "TUM_RGBD_FR2_RPY",
            "source_kind": "BURNED_REAL_DEVELOPMENT_ANCHOR",
            "role": "BELOW_TRIGGER_REFERENCE_WINDOW",
            "window_index": 7,
            "start_timestamp_s": "1311867789",
            "end_timestamp_s": "1311867799",
            "geometry_summary": tum_rows[1],
            "rgb_acquisition": {"transport": "PREEXISTING_LOCAL_SOURCE_NATIVE_RGB"},
        },
    ]
    if sum(row["role"] == "POSITIVE_APPROACH_WINDOW" for row in windows) != 2:
        raise ValueError("FINAL_POSITIVE_COUNT")
    if sum(row["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW" for row in windows) != 2:
        raise ValueError("FINAL_BELOW_COUNT")
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.final_cohort_freeze.v1",
        "protocol_id": contract["protocol_id"],
        "status": "FOUR_WINDOW_IDENTITIES_FROZEN_BEFORE_NEW_RGB_ACCESS",
        "evidence": {
            key: {"path": path.relative_to(Path.cwd()).as_posix(), "sha256": sha(path)}
            for key, path in paths.items()
        },
        "windows": windows,
        "positive_window_count": 2,
        "below_reference_window_count": 2,
        "next_action": "ACQUIRE_MINIMUM_FROZEN_WINDOW_RGB_ONLY",
        "algorithm_change_allowed": False,
        "post_geometry_window_substitution_allowed": False,
        "authority": {
            "development_cohort_usable": True,
            "all_real_cross_source_holdout": False,
            "product_authority": False,
            "android_authority": False,
        },
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(os.fspath(args.output.resolve()), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": result["status"],
                "window_ids": [row["window_id"] for row in windows],
                "cohort_sha256": hashlib.sha256(payload).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
