#!/usr/bin/env python3
"""Acquire the frozen THÖR truth canary and bind the official payload bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CONFIG_SCHEMA = (
    "blindassist_ustrf_independent_person_trajectory_truth_"
    "source_authority_and_admission_r0_config"
)
ACQUISITION_SCHEMA = (
    "blindassist_ustrf_independent_person_trajectory_truth_"
    "source_authority_and_admission_r0_acquisition"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies the source-published checksum only
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(repo: Path, config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("config_identity")
    if config.get("status") != "frozen_before_candidate_output_read":
        raise RuntimeError("config_not_prefrozen")
    canary = config["canary"]
    destination = repo / config["outputs"]["dataset"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    if not destination.exists():
        request = urllib.request.Request(
            canary["url"], headers={"User-Agent": "BlindAssist truth-authority audit/1.0"}
        )
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(destination)
    actual_bytes = destination.stat().st_size
    actual_md5 = md5_file(destination)
    if actual_bytes != int(canary["declared_bytes"]):
        raise RuntimeError(f"payload_size_drift:{actual_bytes}")
    if actual_md5 != canary["declared_md5"]:
        raise RuntimeError(f"payload_md5_drift:{actual_md5}")
    return {
        "schema": ACQUISITION_SCHEMA,
        "stage": config["stage"],
        "status": "ACQUIRED_AND_HASH_BOUND",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "record": config["source_inventory"][2]["official_record"],
            "url": canary["url"],
            "member": canary["member"],
            "license": config["source_inventory"][2]["license"],
        },
        "selection": {
            "rule": canary["selection_rule"],
            "payload_values_read_before_freeze": False,
            "candidate_outputs_visible": False,
        },
        "payload": {
            "path": config["outputs"]["dataset"],
            "bytes": actual_bytes,
            "md5": actual_md5,
            "sha256": sha256_file(destination),
        },
        "config_sha256": sha256_file(config_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    receipt = acquire(repo, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = repo / config["outputs"]["acquisition"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
