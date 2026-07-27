"""Advance the consumed rank-one claim to selective depth transport only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def write_exclusive(path: Path, value: object) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parent-claim", type=Path, required=True)
    parser.add_argument("--window-freeze", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    parent_path = args.parent_claim.resolve()
    freeze_path = args.window_freeze.resolve()
    amendment = load_object(args.amendment.resolve())
    parent = load_object(parent_path)
    freeze = load_object(freeze_path)
    if amendment.get("status") != "FROZEN_BEFORE_COALESCED_DEPTH_GET":
        raise ValueError("AMENDMENT_STATUS")
    if amendment.get("parent_claim_sha256") != sha256_file(parent_path):
        raise ValueError("PARENT_CLAIM_IDENTITY")
    if amendment.get("window_freeze_sha256") != sha256_file(freeze_path):
        raise ValueError("WINDOW_FREEZE_IDENTITY")
    adapter = repo / amendment["transport_only_change"]["stable_root_adapter"]
    if (
        sha256_file(adapter)
        != amendment["transport_only_change"]["stable_root_adapter_sha256"]
    ):
        raise ValueError("COALESCED_ADAPTER_IDENTITY")
    if parent.get("candidate_id") != "ETH3D_SLAM_DESK_3":
        raise ValueError("PARENT_CANDIDATE")
    eligible = [
        int(window["window_index"])
        for window in freeze["windows"]
        if window["identity_eligible"] is True
    ]
    if eligible != amendment["unchanged"]["identity_eligible_window_indices"]:
        raise ValueError("ELIGIBLE_WINDOW_IDENTITY")
    claim = {
        "schema": "rcle.motion_diverse_rgbd.selective_depth_claim.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "ETH3D_SLAM_DESK_3",
        "parent_claim_sha256": sha256_file(parent_path),
        "window_freeze_sha256": sha256_file(freeze_path),
        "transport_amendment_sha256": sha256_file(args.amendment.resolve()),
        "identity_eligible_window_indices": eligible,
        "stage": "SELECTIVE_DEPTH_MEMBERS_ONLY",
        "candidate_replacement_allowed": False,
        "whole_archive_download_allowed": False,
        "rgb_payload_allowed": False,
    }
    write_exclusive(args.claim.resolve(), claim)
    print(json.dumps(claim, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
