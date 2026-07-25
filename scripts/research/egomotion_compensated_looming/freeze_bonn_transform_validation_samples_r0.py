#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


WINDOW_SECONDS = 10.0
SAMPLE_OFFSETS_SECONDS = (0.0, 5.0, 9.9)
MAX_DEPTH_JOIN_DELTA_SECONDS = 0.040
DISCOVERY_SEQUENCE_IDS = (
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_balloon",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(data: bytes, columns: int) -> list[list[str]]:
    result = [
        line.split()
        for line in data.decode("utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if not result or any(len(row) != columns for row in result):
        raise ValueError("invalid source index")
    times = [float(row[0]) for row in result]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")
    return result


def nearest(rows_: list[list[str]], target: float) -> tuple[int, list[str]]:
    index, row = min(
        enumerate(rows_),
        key=lambda item: abs(float(item[1][0]) - target),
    )
    return index, row


def freeze_sequence(
    archive_path: Path, sequence_id: str, expected_sha256: str
) -> dict[str, Any]:
    if sha256(archive_path) != expected_sha256:
        raise ValueError("discovery archive SHA-256 mismatch")
    prefix = f"{sequence_id}/"
    with zipfile.ZipFile(archive_path) as archive:
        rgb = rows(archive.read(f"{prefix}rgb.txt"), 2)
        depth = rows(archive.read(f"{prefix}depth.txt"), 2)
    start = float(rgb[0][0])
    samples: list[dict[str, Any]] = []
    for offset in SAMPLE_OFFSETS_SECONDS:
        rgb_index, rgb_row = nearest(rgb, start + offset)
        depth_index, depth_row = nearest(depth, float(rgb_row[0]))
        depth_delta = abs(float(depth_row[0]) - float(rgb_row[0]))
        if depth_delta > MAX_DEPTH_JOIN_DELTA_SECONDS:
            raise ValueError("depth join exceeds frozen hard cap")
        samples.append(
            {
                "sample_offset_seconds": offset,
                "rgb_index": rgb_index,
                "rgb_timestamp": float(rgb_row[0]),
                "rgb_member": f"{prefix}{rgb_row[1]}",
                "depth_index": depth_index,
                "depth_timestamp": float(depth_row[0]),
                "depth_member": f"{prefix}{depth_row[1]}",
                "depth_join_delta_seconds": depth_delta,
            }
        )
    return {
        "sequence_id": sequence_id,
        "archive_filename": archive_path.name,
        "archive_sha256": expected_sha256,
        "window_start_timestamp": start,
        "window_end_timestamp_exclusive": start + WINDOW_SECONDS,
        "samples": samples,
    }


def build(acquisition: dict[str, Any], archive_dir: Path) -> dict[str, Any]:
    by_sequence = {
        item["sequence_id"]: item for item in acquisition["archives"]
    }
    if tuple(by_sequence) != DISCOVERY_SEQUENCE_IDS:
        raise ValueError("unexpected discovery sequence order or identity")
    sequences = [
        freeze_sequence(
            archive_dir / by_sequence[sequence_id]["archive_filename"],
            sequence_id,
            by_sequence[sequence_id]["archive_sha256"],
        )
        for sequence_id in DISCOVERY_SEQUENCE_IDS
    ]
    return {
        "schema_version": "bonn_transform_validation_sample_freeze_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "selection_contract": {
            "stage": "METADATA_ONLY_BEFORE_DEPTH_MEMBER_READ_OR_DECODE",
            "discovery_only": True,
            "window_seconds": WINDOW_SECONDS,
            "sample_offsets_seconds": list(SAMPLE_OFFSETS_SECONDS),
            "nearest_rgb_then_nearest_depth": True,
            "maximum_depth_join_delta_seconds": MAX_DEPTH_JOIN_DELTA_SECONDS,
            "selected_without_rgb_depth_map_pose_or_signal_values": True,
        },
        "sequences": sequences,
        "counts": {
            "sequence_count": len(sequences),
            "sample_count": sum(len(item["samples"]) for item in sequences),
            "depth_member_read_or_decode_count_at_freeze": 0,
            "rgb_member_read_or_decode_count_at_freeze": 0,
            "candidate_signal_computed_at_freeze": False,
        },
        "allowed_next_action": (
            "DECODE_ONLY_THE_SIX_FROZEN_DEPTH_MEMBERS_FOR_OFFICIAL_"
            "TRANSFORM_GEOMETRY_VALIDATION"
        ),
        "terminal": "BONN_TRANSFORM_VALIDATION_SAMPLES_FROZEN",
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    acquisition = json.loads(args.acquisition.read_text(encoding="utf-8"))
    receipt = build(acquisition, args.archive_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                **receipt["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
