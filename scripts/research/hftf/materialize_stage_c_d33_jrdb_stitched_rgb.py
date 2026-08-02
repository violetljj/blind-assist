#!/usr/bin/env python3
"""Restore the exact 480 JRDB stitched RGB frames recorded by D32 packets."""

from __future__ import annotations

import argparse
import binascii
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    DEFAULT_PACKETS,
    REPO_ROOT,
    sha256,
)


KERNEL_ROOT = (
    REPO_ROOT
    / "scripts/research/ustrf_route_target_evidence_closure"
)
sys.path.insert(0, str(KERNEL_ROOT))
from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (  # noqa: E402
    fetch_member,
    get_range,
    parse_central,
)


SCHEMA = "blindassist_hftf_stage_c_d33_jrdb_stitched_rgb_receipt_v0"
DEFAULT_ARCHIVE_CONFIG = (
    REPO_ROOT
    / "artifacts.local/datasets/"
    "jrdb-person-3d-trajectory-sensor-support-and-bias-"
    "cross-sequence-replication-r0/"
    "clark-center-2019-02-28_0/materialization-config.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts.local/datasets/"
    "hftf-stage-c-d33-jrdb-stitched-rgb-v0"
)
DEFAULT_RECEIPT = DEFAULT_OUTPUT_ROOT / "receipt.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def expected_images(
    packet_paths: tuple[Path, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sequences: set[str] = set()
    for packet_path in packet_paths:
        packet = load_json(packet_path)
        sequence = str(packet["sequence"])
        if sequence in sequences:
            raise ValueError(f"D33 duplicate packet sequence: {sequence}")
        sequences.add(sequence)
        images = [
            row
            for row in packet["raw_payload"]["files"]
            if row["role"] == "images"
        ]
        if len(images) != 120:
            raise ValueError(
                f"D33 expected 120 image records: {sequence}/{len(images)}"
            )
        for row in images:
            member = str(row["member"])
            records.append(
                {
                    "sequence": sequence,
                    "frame_stem": Path(member).stem,
                    "member": member,
                    "bytes": int(row["bytes"]),
                    "crc32": int(row["crc32"]),
                    "sha256": str(row["sha256"]),
                    "packet_path": str(packet_path.resolve()),
                    "packet_sha256": sha256(packet_path),
                }
            )
    return sorted(
        records,
        key=lambda row: (row["sequence"], row["frame_stem"]),
    )


def materialize(
    packet_paths: tuple[Path, ...],
    archive_config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    expected = expected_images(packet_paths)
    config = load_json(archive_config_path)
    archive = config["remote_archives"]["images"]
    start = int(archive["central_directory_offset"])
    size = int(archive["central_directory_size"])
    central = get_range(
        str(archive["url"]),
        start,
        start + size - 1,
        str(archive["etag"]),
    )
    member_index = {
        str(row["name"]): row for row in parse_central(central)
    }
    missing = [
        row["member"]
        for row in expected
        if row["member"] not in member_index
    ]
    if missing:
        raise ValueError(f"D33 archive members missing: {missing[:3]}")

    def restore(row: dict[str, Any]) -> tuple[dict[str, Any], int, bool]:
        output = output_root / row["sequence"] / (
            row["frame_stem"] + ".jpg"
        )
        reused = False
        network_bytes = 0
        if output.exists() and sha256(output) == row["sha256"]:
            reused = True
        else:
            raw, network_bytes = fetch_member(
                archive,
                member_index[row["member"]],
            )
            if len(raw) != row["bytes"]:
                raise ValueError(f"D33 image size drift: {row['member']}")
            if binascii.crc32(raw) & 0xFFFFFFFF != row["crc32"]:
                raise ValueError(f"D33 image CRC drift: {row['member']}")
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"D33 image SHA drift: {row['member']}")
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_name(output.name + ".partial")
            partial.write_bytes(raw)
            os.replace(partial, output)
        return (
            {
                **row,
                "path": str(output.resolve()),
            },
            network_bytes,
            reused,
        )

    results: list[tuple[dict[str, Any], int, bool]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(restore, row) for row in expected]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            results.append(future.result())
            if completed % 40 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(futures),
                        }
                    ),
                    flush=True,
                )
    records = sorted(
        (row for row, _, _ in results),
        key=lambda row: (row["sequence"], row["frame_stem"]),
    )
    return {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "archive": {
            "url": archive["url"],
            "etag": archive["etag"],
            "central_directory_bytes": len(central),
        },
        "frame_count": len(records),
        "sequence_count": len({row["sequence"] for row in records}),
        "payload_bytes": sum(row["bytes"] for row in records),
        "network_bytes": len(central)
        + sum(value for _, value, _ in results),
        "reused_frames": sum(
            1 for _, _, reused in results if reused
        ),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        action="append",
        type=Path,
        dest="packets",
    )
    parser.add_argument(
        "--archive-config",
        type=Path,
        default=DEFAULT_ARCHIVE_CONFIG,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_RECEIPT,
    )
    args = parser.parse_args()
    packets = tuple(args.packets) if args.packets else DEFAULT_PACKETS
    receipt = materialize(
        packets,
        args.archive_config,
        args.output_root,
    )
    atomic_json(args.receipt, receipt)
    digest = sha256(args.receipt)
    args.receipt.with_suffix(args.receipt.suffix + ".sha256").write_text(
        f"{digest}  {args.receipt.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "frame_count": receipt["frame_count"],
                "network_bytes": receipt["network_bytes"],
                "receipt_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
