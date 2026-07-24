#!/usr/bin/env python3
"""Independent recomputation for the JRDB source-authority canary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from audit_jrdb_source_authority_canary_r0 import (
    SCHEMA,
    STAGE,
    atomic_write,
    audit,
    load_json,
    sha256_file,
)


VALIDATION_SCHEMA = "blindassist_ustrf_jrdb_source_authority_canary_validation_r0"


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
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_json(config_path)
    receipt_path = repo / config["outputs"]["receipt"]
    receipt = load_json(receipt_path)
    recomputed = audit(repo, config_path)

    checks = {
        "receipt_schema": receipt.get("schema") == SCHEMA,
        "stage": receipt.get("stage") == STAGE,
        "process_isolation": receipt.get("process_id") != os.getpid(),
        "deterministic_recomputation": without_process(receipt) == without_process(recomputed),
        "archive_safe": receipt["archive"]["unsafe_member_count"] == 0,
        "truncation_nonconstant": receipt["stitched_label_canary"]["source_native_truncation_is_nonconstant"],
        "authority_terminal": receipt.get("terminal_state") == "AUTHORITY_CANARY_PRESENT_ROUTE_ROLE_PENDING",
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
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "status": "VALID" if all(checks.values()) else "INVALID",
        "terminal_state": receipt.get("terminal_state"),
        "process_id": os.getpid(),
        "producer_process_id": receipt.get("process_id"),
        "config_sha256": sha256_file(config_path),
        "receipt_sha256": sha256_file(receipt_path),
        "checks": checks,
    }
    output = repo / config["outputs"]["validation"]
    atomic_write(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
