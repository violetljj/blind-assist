#!/usr/bin/env python3
"""Independent recomputation for the JRDB single-frame canary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from materialize_jrdb_single_frame_rgb_time_transform_canary_r1 import (
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
    recomputed = audit(repo, config_path, persist_image=False)
    image_path = repo / config["outputs"]["image"]
    checks = {
        "schema": receipt.get("schema") == SCHEMA,
        "stage": receipt.get("stage") == STAGE,
        "process_isolation": receipt.get("process_id") != os.getpid(),
        "deterministic_recomputation": without_process(receipt) == without_process(recomputed),
        "terminal": receipt.get("terminal_state") == "RGB_TIME_TRANSFORM_CANARY_PRESENT",
        "image_persisted": image_path.is_file(),
        "image_hash": sha256_file(image_path) == receipt["canary"]["jpeg"]["sha256"],
        "bounded_network": receipt["network"]["bytes_read"] <= receipt["network"]["budget_bytes"],
        "no_full_archive": not receipt["network"]["full_archive_downloaded"],
        "same_frame": receipt["canary"]["timestamp"]["source_url"].endswith(
            f"/{receipt['canary']['sequence']}/stitched_image0/{receipt['canary']['frame']}"
        ),
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
        "schema": f"{SCHEMA}_validation",
        "stage": STAGE,
        "status": "VALID" if all(checks.values()) else "INVALID",
        "terminal_state": receipt.get("terminal_state"),
        "process_id": os.getpid(),
        "producer_process_id": receipt.get("process_id"),
        "config_sha256": sha256_file(config_path),
        "receipt_sha256": sha256_file(receipt_path),
        "checks": checks,
        "validator_network_bytes": recomputed["network"]["bytes_read"],
    }
    atomic_write(repo / config["outputs"]["validation"], canonical_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
