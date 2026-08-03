#!/usr/bin/env python3
"""Record explicit user review of the pinned ARKitScenes license."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_protocol import DEFAULT_PROTOCOL, REPO_ROOT, sha256, validate

CONFIRMATION = "I_ACCEPT_ARKITSCENES_LICENSE_FOR_SPATIAL_CALIBRATION_HEAD_R1_MEDIA_DOWNLOAD"
DEFAULT_LOCK = REPO_ROOT / "docs/research/hftf/SPATIAL_CALIBRATION_HEAD_R1_COHORT_ROSTER_LOCK_2026-08-04.json"
DEFAULT_PREFLIGHT = REPO_ROOT / "artifacts.local/evidence/hftf/spatial-calibration-head-r1-asset-head-preflight-20260804-r2/result.json"


def build_receipt(
    protocol: dict[str, Any],
    protocol_sha256: str,
    roster_lock_sha256: str,
    asset_preflight_sha256: str,
    confirmation: str,
    confirmed_by: str,
) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise ValueError("exact explicit license confirmation is required")
    if confirmed_by != "user":
        raise ValueError("license confirmation must come from the user")
    return {
        "schema": "blindassist_spatial_calibration_head_r1_license_receipt",
        "protocol_sha256": protocol_sha256,
        "roster_lock_sha256": roster_lock_sha256,
        "asset_preflight_sha256": asset_preflight_sha256,
        "license_path": "artifacts.local/downloads/ARKitScenes-7283761/LICENSE",
        "license_sha256": protocol["data"]["license_sha256"],
        "confirmed_by": confirmed_by,
        "confirmation": confirmation,
        "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_scope": {
            "development": "20 locked videos; lowres_wide, lowres_depth, confidence, lowres_wide_intrinsics, lowres_wide.traj",
            "sealed_identity_only": "four locked videos; lowres_wide RGB only for capture-identity audit",
            "sealed_metric_assets": False,
            "redistribution": False,
        },
        "media_download_authorized": True,
        "terminal": "ARKITSCENES_LICENSE_REVIEW_RECORDED_R1_SCOPED_MEDIA_DOWNLOAD_AUTHORIZED",
    }


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--roster-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--asset-preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--confirmed-by", choices=("user",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    errors = validate(protocol)
    if errors:
        raise ValueError(f"protocol invalid: {errors}")
    lock = json.loads(args.roster_lock.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != sha256(args.protocol):
        raise ValueError("roster lock protocol mismatch")
    preflight = json.loads(args.asset_preflight.read_text(encoding="utf-8"))
    if preflight.get("terminal") != "SPATIAL_CALIBRATION_HEAD_R1_ASSET_HEADERS_AVAILABLE":
        raise ValueError("asset preflight did not pass")
    if preflight.get("protocol_sha256") != sha256(args.protocol) or preflight.get("roster_lock_sha256") != sha256(args.roster_lock):
        raise ValueError("asset preflight authority mismatch")
    receipt = build_receipt(
        protocol, sha256(args.protocol), sha256(args.roster_lock), sha256(args.asset_preflight),
        args.confirmation, args.confirmed_by,
    )
    write_json_new(args.output, receipt)
    print(json.dumps({**receipt, "output": str(args.output.resolve()), "output_sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
