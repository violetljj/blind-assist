#!/usr/bin/env python3
"""Independently recompute and validate the JRDB native multisensor R0 receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_jrdb_native_pose_and_3d_person_motion_authority_r0 import audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.receipt.read_text(encoding="utf-8"))
    observed = audit(args.repo.resolve(), args.config.resolve())
    checks = {
        "deterministic_recomputation": observed == expected,
        "terminal_is_fail_closed_p1b": observed["terminal_state"]
        == "NATIVE_MULTISENSOR_CANARY_ELIGIBLE_POSE_IMU_TOPIC_AUDIT_REQUIRED",
        "bounded_network": observed["network"]["bytes_read"] <= observed["network"]["budget"],
        "no_full_archive": observed["network"]["full_archive_downloaded"] is False,
        "first_120_all_modalities": all(observed["coverage"]["first_120_complete"].values()),
        "pose_payload_unclaimed": observed["authority"]["native_pose_topic_payload_audited"] is False,
        "imu_payload_unclaimed": observed["authority"]["imu_topic_payload_audited"] is False,
        "p2_closed": observed["authority"]["p2_authorized"] is False,
        "route_event_safety_closed": observed["authority"]["route_event_safety_authority"] is False,
    }
    validation = {
        "schema": "blindassist_ustrf_jrdb_native_pose_and_3d_person_motion_authority_audit_r0_validation",
        "status": "VALID" if all(checks.values()) else "INVALID",
        "checks": checks,
        "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation))
    return 0 if validation["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
