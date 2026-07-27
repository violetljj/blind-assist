"""Verify the frozen RGB payloads and emit the algorithm input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import zlib


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--eth3d-rgb-root", type=Path, required=True)
    parser.add_argument("--eth3d-fetch-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort_path = args.cohort.resolve()
    preparation_path = args.preparation.resolve()
    cohort = load(cohort_path)
    preparation = load(preparation_path)
    if preparation["cohort_sha256"] != sha(cohort_path):
        raise ValueError("RGB_PREPARATION_COHORT_IDENTITY")
    fetch = load(args.eth3d_fetch_inventory.resolve())
    expected = {row["path"]: row for row in preparation["eth3d"]["expected_members"]}
    fetched = {row["path"]: row for row in fetch["members"]}
    if set(expected) != set(fetched):
        raise ValueError("ETH3D_RGB_FETCH_SET")
    eth_root = args.eth3d_rgb_root.resolve()
    eth_records = []
    for member, expected_row in expected.items():
        path = eth_root.joinpath(*Path(member).parts)
        raw = path.read_bytes()
        crc = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
        if len(raw) != int(expected_row["bytes"]) or crc != expected_row["crc32"]:
            raise ValueError(f"ETH3D_RGB_IDENTITY:{member}")
        eth_records.append(
            {
                "path": member,
                "bytes": len(raw),
                "crc32": crc,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.rgb_input_manifest.v1",
        "protocol_id": cohort["protocol_id"],
        "cohort_sha256": sha(cohort_path),
        "preparation_sha256": sha(preparation_path),
        "eth3d_fetch_inventory_sha256": sha(args.eth3d_fetch_inventory.resolve()),
        "windows": [
            {
                "window_id": "desk_changing_1@4065.364250422",
                "role": "POSITIVE_APPROACH_WINDOW",
                "source_kind": "REAL_DEVELOPMENT_SOURCE",
                "rgb_root": eth_root.as_posix(),
                "members": eth_records,
            },
            {
                "window_id": "japanesealley/Hard/P002@000260",
                "role": "POSITIVE_APPROACH_WINDOW",
                "source_kind": "SYNTHETIC_DEVELOPMENT_ANCHOR",
                "rgb_root": preparation["tartanair"]["rgb_root"],
                "members": preparation["tartanair"]["members"],
            },
            *[
                {
                    "window_id": row["window_id"],
                    "role": "BELOW_TRIGGER_REFERENCE_WINDOW",
                    "source_kind": "BURNED_REAL_DEVELOPMENT_ANCHOR",
                    "rgb_root": preparation["tum"]["rgb_root"],
                    "members": row["members"],
                }
                for row in preparation["tum"]["windows"]
            ],
        ],
        "window_count": 4,
        "window_substitution": False,
        "rgb_visual_inspection": False,
        "algorithm_outcome_accessed": False,
        "authority": "DEVELOPMENT_COHORT_INPUT_ONLY",
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
                "window_count": 4,
                "rgb_member_counts": [len(row["members"]) for row in result["windows"]],
                "rgb_input_manifest_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
