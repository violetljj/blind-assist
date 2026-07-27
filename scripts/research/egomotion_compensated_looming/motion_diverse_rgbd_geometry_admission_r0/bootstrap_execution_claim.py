"""Create the exclusive geometry execution claim after the implementation lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    lock_path = args.implementation_lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN_BEFORE_GEOMETRY_EXECUTION":
        raise ValueError("IMPLEMENTATION_LOCK_STATUS")
    for entry in lock["files"]:
        if sha(repo / entry["path"]) != entry["sha256"]:
            raise ValueError(f"IMPLEMENTATION_LOCK_DRIFT:{entry['path']}")
    value = {
        "schema": "rcle.motion_diverse_rgbd.geometry_execution_claim.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "ETH3D_SLAM_DESK_3",
        "implementation_lock_sha256": sha(lock_path),
        "workers": 8,
        "rgb_payload_allowed": False,
        "candidate_replacement_allowed": False,
    }
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path = args.claim.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
