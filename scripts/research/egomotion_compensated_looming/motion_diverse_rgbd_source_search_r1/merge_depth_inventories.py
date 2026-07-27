"""Merge exact remote-ZIP inventories for a frozen cross-sequence batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = load(args.batch.resolve())
    expected = {member for window in batch["windows"] for member in window["depth_members"]}
    members = []
    sources = []
    seen = set()
    for path in args.inventory:
        inventory = load(path.resolve())
        sources.append(inventory["source"])
        for member in inventory["members"]:
            if member["path"] in seen:
                raise ValueError(f"DUPLICATE_MEMBER:{member['path']}")
            seen.add(member["path"])
            members.append(member)
    if seen != expected:
        raise ValueError(
            f"INVENTORY_SET_MISMATCH:missing={len(expected-seen)}:extra={len(seen-expected)}"
        )
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.merged_depth_inventory.v1",
        "batch_id": batch["batch_id"],
        "sources": sources,
        "members": sorted(members, key=lambda row: row["path"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"batch_id": batch["batch_id"], "member_count": len(members)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
