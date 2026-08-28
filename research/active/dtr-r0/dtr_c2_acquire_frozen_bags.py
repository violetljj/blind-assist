"""Acquire the raw JRDB rosbags frozen by the DTR-C1 roster.

The public train archive is about 42 GB, so C2 range-extracts only the seven
members named by the committed C1 roster.  Completed members are hash checked
and skipped on resume.  A partial member may be redownloaded after failure;
the maximum lost work is therefore one compressed rosbag member.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dtr_c1_global_obb_cohort_admission import ROSTER_SCHEMA, require, sha256_file
from jrdb_range_acquire import acquire, get_range, parse_central


SCHEMA = "blindassist-dtr-c2-frozen-jrdb-bag-acquisition-v1"
STATUS = "DTR_C2_FROZEN_RAW_BAGS_ACQUIRED"
ARCHIVE_URL = "https://jrdb.erc.monash.edu/static/downloads/train_rosbags.zip"
ARCHIVE_CONTENT_LENGTH = 41_876_187_771
CENTRAL_DIRECTORY_OFFSET = 41_876_181_387
CENTRAL_DIRECTORY_SIZE = 6_286


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def load_members() -> dict[str, dict[str, Any]]:
    payload = get_range(
        ARCHIVE_URL,
        CENTRAL_DIRECTORY_OFFSET,
        CENTRAL_DIRECTORY_OFFSET + CENTRAL_DIRECTORY_SIZE - 1,
    )
    members = parse_central(payload)
    return {str(member["name"]): member for member in members}


def member_config(sequence: str, member: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "blindassist-dtr-c2-jrdb-member-config-v1",
        "stage": "DTR_C2_FROZEN_RAW_SENSOR_ACQUISITION",
        "sequence": sequence,
        "remote_zip": {
            "url": ARCHIVE_URL,
            "content_length": ARCHIVE_CONTENT_LENGTH,
            "central_directory_offset": CENTRAL_DIRECTORY_OFFSET,
            "central_directory_size": CENTRAL_DIRECTORY_SIZE,
            "member": {
                "name": member["name"],
                "flags": int(member["flags"]),
                "method": int(member["method"]),
                "crc32": int(member["crc32"]),
                "compressed_size": int(member["compressed"]),
                "uncompressed_size": int(member["uncompressed"]),
                "local_header_offset": int(member["offset"]),
            },
        },
        "resource_gate": {
            "maximum_network_bytes": int(member["compressed"]),
            "maximum_local_bag_bytes": int(member["uncompressed"]),
            "full_archive_download_authorized": False,
        },
    }


def valid_completed(bag: Path, receipt: Path) -> dict[str, Any] | None:
    if not bag.exists() or not receipt.exists():
        return None
    value = json.loads(receipt.read_text(encoding="utf-8"))
    recorded = value.get("bag", {})
    require(value.get("status") == "ACQUIRED", f"receipt_status:{receipt}")
    require(bag.stat().st_size == int(recorded["bytes"]), f"bag_size_drift:{bag}")
    require(sha256_file(bag) == recorded["sha256"], f"bag_hash_drift:{bag}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema_drift")
    require(
        roster.get("status") == "DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY",
        "c1_not_preferred_admitted",
    )
    sequences = [str(row["sequence"]) for row in roster["selected_sequences"]]
    require(len(sequences) == len(set(sequences)), "duplicate_sequence")
    if args.only_sequence is not None:
        require(args.only_sequence in sequences, f"sequence_not_in_roster:{args.only_sequence}")
        sequences = [args.only_sequence]
    members = load_members()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sequence in enumerate(sequences, start=1):
        name = f"rosbags/{sequence}.bag"
        require(name in members, f"archive_member_missing:{name}")
        member = members[name]
        config_path = output_dir / f"{sequence}.config.json"
        bag_path = output_dir / f"{sequence}.bag"
        receipt_path = output_dir / f"{sequence}.receipt.json"
        config = member_config(sequence, member)
        if config_path.exists():
            require(
                json.loads(config_path.read_text(encoding="utf-8")) == config,
                f"config_drift:{sequence}",
            )
        else:
            write_json(config_path, config)
        completed = valid_completed(bag_path, receipt_path)
        if completed is None:
            partial = bag_path.with_suffix(bag_path.suffix + ".partial")
            if partial.exists():
                partial.unlink()
            print(
                json.dumps(
                    {
                        "sequence": sequence,
                        "member": index,
                        "total": len(sequences),
                        "compressed_bytes": int(member["compressed"]),
                        "status": "ACQUIRING",
                    }
                ),
                flush=True,
            )
            completed = acquire(config_path, bag_path, args.maximum_reconnects)
            write_json(receipt_path, completed)
        rows.append(
            {
                "sequence": sequence,
                "bag": str(bag_path),
                "bytes": int(completed["bag"]["bytes"]),
                "sha256": completed["bag"]["sha256"],
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
            }
        )
        print(
            json.dumps(
                {
                    "sequence": sequence,
                    "member": index,
                    "total": len(sequences),
                    "status": "ACQUIRED",
                }
            ),
            flush=True,
        )
    result = {
        "schema": SCHEMA,
        "status": (
            STATUS
            if args.only_sequence is None
            else "DTR_C2_FROZEN_RAW_BAG_MEMBER_ACQUIRED"
        ),
        "roster": str(roster_path),
        "roster_sha256": sha256_file(roster_path),
        "archive": {
            "url": ARCHIVE_URL,
            "content_length": ARCHIVE_CONTENT_LENGTH,
            "central_directory_offset": CENTRAL_DIRECTORY_OFFSET,
            "central_directory_size": CENTRAL_DIRECTORY_SIZE,
        },
        "recovery": {
            "completed_members_are_hash_checked_and_skipped": True,
            "partial_member_resume": False,
            "maximum_lost_work": "one compressed rosbag member",
        },
        "bags": rows,
        "totals": {
            "sequences": len(rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
        },
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c1_fresh_global_obb_roster.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "artifacts.local" / "datasets" / "dtr-c2-jrdb-fresh-global-obb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo
        / "artifacts.local"
        / "evidence"
        / "dtr-c2"
        / "fresh-global-obb-replay"
        / "acquisition.json",
    )
    parser.add_argument("--maximum-reconnects", type=int, default=12)
    parser.add_argument("--only-sequence")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "totals": result["totals"]}))


if __name__ == "__main__":
    main()
