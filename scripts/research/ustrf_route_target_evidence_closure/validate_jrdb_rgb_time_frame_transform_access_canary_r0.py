#!/usr/bin/env python3
"""Independent validation for the JRDB RGB/time/frame-transform access canary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from audit_jrdb_rgb_time_frame_transform_access_canary_r0 import (
    SCHEMA,
    STAGE,
    atomic_write,
    audit,
    load_json,
    sha256_file,
)


def without_process(value: dict[str, Any]) -> dict[str, Any]:
    copy = dict(value)
    copy.pop("process_id", None)
    return copy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    config = load_json(config_path)
    receipt_path = repo / config["outputs"]["receipt"]
    receipt = load_json(receipt_path)
    recomputed = audit(repo, config_path)
    checks = {
        "schema": receipt.get("schema") == SCHEMA,
        "stage": receipt.get("stage") == STAGE,
        "process_isolation": receipt.get("process_id") != os.getpid(),
        "deterministic_recomputation": without_process(receipt) == without_process(recomputed),
        "terminal": receipt.get("terminal_state") == "ACCESS_BLOCKED_LOGIN_REQUIRED",
        "no_real_payload_in_sample_structure": not receipt["sample_structure"]["contains_real_files"],
        "capture_time_not_inferred_from_label_key": not receipt["authority"]["label_key_is_capture_time_authority"],
        "higher_authority_closed": not any(
            receipt["claim_boundary"][key]
            for key in (
                "g1_authorized",
                "signal_authorized",
                "route_truth_authorized",
                "android_authorized",
                "human_authorized",
                "production_authorized",
            )
        ),
    }
    result = {
        "schema": "blindassist_ustrf_jrdb_rgb_time_frame_transform_access_canary_validation_r0",
        "stage": STAGE,
        "status": "VALID" if all(checks.values()) else "INVALID",
        "terminal_state": receipt.get("terminal_state"),
        "process_id": os.getpid(),
        "producer_process_id": receipt.get("process_id"),
        "config_sha256": sha256_file(config_path),
        "receipt_sha256": sha256_file(receipt_path),
        "checks": checks,
    }
    atomic_write(repo / config["outputs"]["validation"], result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
