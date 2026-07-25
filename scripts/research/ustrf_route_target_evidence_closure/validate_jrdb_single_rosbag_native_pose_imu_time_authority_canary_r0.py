#!/usr/bin/env python3
"""Independently recompute the frozen JRDB single-rosbag authority receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0 import audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.receipt.read_text(encoding="utf-8"))
    observed = audit(args.repo.resolve(), args.config.resolve(), args.bag.resolve(), args.acquisition.resolve())
    terminal = expected["terminal_state"]
    checks = {
        "deterministic_recomputation": observed == expected,
        "terminal_legal": terminal in {
            "NATIVE_POSE_AUTHORITY_ABSENT",
            "NATIVE_IMU_TIME_AUTHORITY_ABSENT",
            "NATIVE_CLOCK_FRAME_CHAIN_NOT_CLOSED",
            "NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT",
        },
        "bag_hash_bound": observed["bag"]["sha256"] == json.loads(args.acquisition.read_text(encoding="utf-8"))["bag"]["sha256"],
        "single_bag_only": json.loads(args.acquisition.read_text(encoding="utf-8"))["network"]["second_bag_downloaded"] is False,
        "no_full_archive": json.loads(args.acquisition.read_text(encoding="utf-8"))["network"]["full_archive_downloaded"] is False,
        "p2_not_executed": observed["authority"]["p2_executed"] is False,
        "relative_motion_not_computed": observed["authority"]["relative_person_motion_computed"] is False,
        "route_event_safety_closed": observed["authority"]["route_event_safety_authority"] is False,
        "android_human_production_closed": observed["authority"]["android_human_production_authority"] is False,
    }
    validation = {
        "schema": "blindassist_ustrf_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0_validation",
        "status": "VALID" if all(checks.values()) else "INVALID",
        "terminal_state": terminal,
        "checks": checks,
        "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation))
    return 0 if validation["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
