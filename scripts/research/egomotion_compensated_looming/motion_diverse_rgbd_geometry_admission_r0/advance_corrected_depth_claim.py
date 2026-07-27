"""Authorize the pre-geometry dt correction's missing window-4 depth only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-claim", type=Path, required=True)
    parser.add_argument("--gate-completion", type=Path, required=True)
    parser.add_argument("--corrected-freeze", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    args = parser.parse_args()
    parent_path = args.parent_claim.resolve()
    gate_path = args.gate_completion.resolve()
    freeze_path = args.corrected_freeze.resolve()
    parent = load(parent_path)
    gate = load(gate_path)
    freeze = load(freeze_path)
    if gate.get("status") != "FROZEN_BEFORE_ANY_GEOMETRY_METRIC":
        raise ValueError("GATE_COMPLETION_STATUS")
    if gate["pre_geometry_evidence"]["corrected_window_freeze_sha256"] != sha(
        freeze_path
    ):
        raise ValueError("CORRECTED_FREEZE_IDENTITY")
    eligible = [
        int(window["window_index"])
        for window in freeze["windows"]
        if window["identity_eligible"] is True
    ]
    if eligible != list(range(7)):
        raise ValueError("CORRECTED_ELIGIBLE_WINDOWS")
    value = {
        "schema": "rcle.motion_diverse_rgbd.corrected_depth_claim.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "ETH3D_SLAM_DESK_3",
        "parent_selective_depth_claim_sha256": sha(parent_path),
        "gate_completion_sha256": sha(gate_path),
        "corrected_window_freeze_sha256": sha(freeze_path),
        "supplemental_window_indices": [4],
        "geometry_metrics_seen": False,
        "candidate_replacement_allowed": False,
        "whole_archive_download_allowed": False,
        "rgb_payload_allowed": False,
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
