#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


WINDOW_SECONDS = 10.0


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


def build(
    archive_audit: dict[str, Any],
    signal_contract: dict[str, Any],
    old_firewall: dict[str, Any],
    archive_dir: Path,
) -> dict[str, Any]:
    if old_firewall["producer_visibility"]["old_frames"]:
        raise ValueError("old frames visible to producer")
    if "BONN_RGBD_DYNAMIC" in old_firewall["deny_source_families"]:
        raise ValueError("Bonn unexpectedly denied by old-window firewall")
    pair_contract = signal_contract["input_pair_contract"]
    minimum = pair_contract["minimum_delta_seconds"]
    maximum = pair_contract["maximum_delta_seconds"]
    sequences: list[dict[str, Any]] = []
    for item in archive_audit["archives"]:
        archive_path = archive_dir / item["archive_filename"]
        if sha256(archive_path) != item["archive_sha256"]:
            raise ValueError("discovery archive SHA-256 mismatch")
        prefix = f"{item['sequence_id']}/"
        with zipfile.ZipFile(archive_path) as archive:
            rgb = rows(archive.read(f"{prefix}rgb.txt"), 2)
        start = float(rgb[0][0])
        window = [
            row for row in rgb if start <= float(row[0]) < start + WINDOW_SECONDS
        ]
        pairs: list[dict[str, Any]] = []
        for index, (previous, current) in enumerate(zip(window, window[1:])):
            previous_timestamp = float(previous[0])
            current_timestamp = float(current[0])
            delta = current_timestamp - previous_timestamp
            eligible = minimum <= delta <= maximum
            pairs.append(
                {
                    "pair_index": index,
                    "unit_id": (
                        f"{item['sequence_id']}:rgb-pair:"
                        f"{previous[0]}->{current[0]}"
                    ),
                    "previous_timestamp": previous_timestamp,
                    "current_timestamp": current_timestamp,
                    "delta_seconds": delta,
                    "previous_rgb_member": f"{prefix}{previous[1]}",
                    "current_rgb_member": f"{prefix}{current[1]}",
                    "eligible": eligible,
                    "abstained": not eligible,
                    "abstention_reason": (
                        None
                        if eligible
                        else "PAIR_DELTA_OUTSIDE_FROZEN_20_TO_50MS_RANGE"
                    ),
                }
            )
        sequences.append(
            {
                "source_family": "BONN_RGBD_DYNAMIC",
                "capture_cluster_id": "BONN_SHARED_CAPTURE_VOLUME_R0",
                "session_id": item["sequence_id"],
                "archive_filename": item["archive_filename"],
                "archive_sha256": item["archive_sha256"],
                "window_start_timestamp": start,
                "window_end_timestamp_exclusive": start + WINDOW_SECONDS,
                "pairs": pairs,
            }
        )
    all_pairs = [
        pair for sequence in sequences for pair in sequence["pairs"]
    ]
    return {
        "schema_version": "bonn_r1a_rgb_pair_manifest_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "frozen_input_receipts": {
            "archive_audit_sha256": None,
            "signal_contract_sha256": None,
            "old_window_firewall_sha256": None,
        },
        "selection_contract": {
            "metadata_only_before_rgb_member_read_or_decode": True,
            "first_non_overlapping_ten_second_window": True,
            "consecutive_rgb_pairs": True,
            "minimum_delta_seconds": minimum,
            "maximum_delta_seconds": maximum,
            "selected_without_pose_depth_truth_cell_or_signal": True,
        },
        "sequences": sequences,
        "counts": {
            "sequence_count": len(sequences),
            "pair_count": len(all_pairs),
            "eligible_pair_count": sum(
                bool(pair["eligible"]) for pair in all_pairs
            ),
            "abstained_pair_count": sum(
                bool(pair["abstained"]) for pair in all_pairs
            ),
            "rgb_member_read_or_decode_count_at_freeze": 0,
            "candidate_signal_computed_at_freeze": False,
        },
        "old_window_firewall": {
            "deny_source_families": old_firewall["deny_source_families"],
            "candidate_source_family_is_disjoint": True,
            "old_frames_visible_to_producer": False,
            "old_outcomes_visible_to_producer": False,
            "known_gap": (
                "OLD_MANIFESTS_HAVE_NO_DECODED_PIXEL_OR_PERCEPTUAL_HASHES"
            ),
            "gap_disposition": (
                "SOURCE_FAMILY_AND_SEQUENCE_ID_DISJOINT_PRODUCER_SEES_ONLY_"
                "DENY_RECEIPT"
            ),
        },
        "allowed_next_action": (
            "BASE_FLOW_PRODUCER_MAY_DECODE_ONLY_MANIFESTED_DISCOVERY_RGB_MEMBERS"
        ),
        "terminal": "BONN_R1A_RGB_PAIR_MANIFEST_FROZEN",
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-audit", required=True, type=Path)
    parser.add_argument("--signal-contract", required=True, type=Path)
    parser.add_argument("--old-firewall", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_audit = json.loads(
        args.archive_audit.read_text(encoding="utf-8")
    )
    signal_contract = json.loads(
        args.signal_contract.read_text(encoding="utf-8")
    )
    old_firewall = json.loads(
        args.old_firewall.read_text(encoding="utf-8")
    )
    receipt = build(
        archive_audit, signal_contract, old_firewall, args.archive_dir
    )
    receipt["frozen_input_receipts"].update(
        {
            "archive_audit_sha256": sha256(args.archive_audit),
            "signal_contract_sha256": sha256(args.signal_contract),
            "old_window_firewall_sha256": sha256(args.old_firewall),
        }
    )
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
