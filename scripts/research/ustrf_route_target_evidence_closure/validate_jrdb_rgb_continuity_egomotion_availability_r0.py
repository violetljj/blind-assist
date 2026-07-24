#!/usr/bin/env python3
"""Independent recomputation for JRDB short-window ego-motion availability."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from audit_jrdb_rgb_continuity_egomotion_availability_r0 import (
    SCHEMA,
    STAGE,
    atomic_write,
    audit,
    canonical_bytes,
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
    recomputed = audit(repo, config_path, persist_frames=False)
    frame_dir = repo / config["outputs"]["frame_directory"]
    persisted_hashes = {
        row["frame"]: sha256_file(frame_dir / row["frame"]) == row["sha256"]
        for row in receipt["frames"]
    }
    checks = {
        "schema": receipt.get("schema") == SCHEMA,
        "stage": receipt.get("stage") == STAGE,
        "process_isolation": receipt.get("process_id") != os.getpid(),
        "deterministic_recomputation": without_process(receipt) == without_process(recomputed),
        "all_frames_persisted_and_hash_bound": all(persisted_hashes.values()),
        "bounded_network": receipt["network"]["bytes_read"] <= receipt["network"]["budget_bytes"],
        "no_full_archive": not receipt["network"]["full_archive_downloaded"],
        "terminal_recomputed": receipt["terminal_state"] == recomputed["terminal_state"],
        "higher_authority_closed": not any(
            receipt["claim_boundary"][key]
            for key in (
                "g3_authorized",
                "g4_authorized",
                "signal_authorized",
                "route_truth_authorized",
                "android_authorized",
                "human_authorized",
                "production_authorized",
            )
        ),
    }
    result = {
        "schema": f"{SCHEMA}_validation",
        "stage": STAGE,
        "status": "VALID" if all(checks.values()) else "INVALID",
        "terminal_state": receipt.get("terminal_state"),
        "process_id": os.getpid(),
        "producer_process_id": receipt.get("process_id"),
        "config_sha256": sha256_file(config_path),
        "receipt_sha256": sha256_file(receipt_path),
        "validator_network_bytes": recomputed["network"]["bytes_read"],
        "checks": checks,
    }
    atomic_write(repo / config["outputs"]["validation"], canonical_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
