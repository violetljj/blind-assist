#!/usr/bin/env python3
"""Independent validator for USTRF canonical observation G0."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import canonical_observation_denominator_availability_r0 as core


VALIDATION_SCHEMA = "blindassist_ustrf_canonical_observation_authority_validation_r0"


def validate(repo: Path, config_path: Path) -> dict[str, object]:
    config = core.load_json(config_path)
    inventory_path = repo / config["authority_inventory"]["path"]
    inventory_sha = str(config["authority_inventory"]["sha256"])
    inventory, first_read = core.verify_inventory_first(
        repo, inventory_path, inventory_sha
    )
    config = core.load_and_verify_config(
        repo,
        config_path,
        inventory_path=inventory_path,
        inventory_sha256=inventory_sha,
    )
    root = repo / config["outputs"]["root"]
    availability_path = root / config["outputs"]["availability"]
    terminal_path = root / config["outputs"]["terminal"]
    audit_path = root / config["outputs"]["audit"]
    for path, label in (
        (availability_path, "availability"),
        (terminal_path, "terminal"),
        (audit_path, "audit"),
    ):
        core.require(path.is_file(), f"{label}_missing")
    availability = core.load_json(availability_path)
    terminal = core.load_json(terminal_path)
    audit = core.load_json(audit_path)
    core.require(
        availability.get("schema") == core.AVAILABILITY_SCHEMA,
        "availability_schema_drift",
    )
    core.require(
        terminal.get("schema") == core.TERMINAL_SCHEMA,
        "terminal_schema_drift",
    )
    core.require(audit.get("schema") == core.AUDIT_SCHEMA, "audit_schema_drift")
    sequences, frames, states, ledgers_complete = core._verify_inventory_ledgers(
        repo, inventory
    )
    source_absent = states.get("severe_truncation", {}).get("absent", 0) == frames
    scope_complete = sequences == 41 and frames == 62229
    expected_terminal = core._terminal(
        audit_complete=ledgers_complete and scope_complete,
        required_source_authority_absent=source_absent,
        availability_complete=bool(availability["availability_complete"]),
    )
    core.require(
        terminal["terminal_state"] == expected_terminal == audit["terminal_state"],
        "terminal_recompute_drift",
    )
    core.require(
        availability["read_manifest"][0] == first_read,
        "inventory_not_first_b_read",
    )
    core.require(
        int(inventory["process_id"]) != int(availability["process_id"])
        and int(inventory["process_id"]) != os.getpid()
        and int(availability["process_id"]) != os.getpid(),
        "three_process_isolation_failed",
    )
    core.require(
        all(value == 0 for value in inventory["decoded_counters"].values())
        and all(value == 0 for value in availability["decoded_counters"].values()),
        "forbidden_decode_counter_nonzero",
    )
    core.require(
        terminal["claim_boundary"]
        == {
            "g1_repair_authorized": False,
            "signal_authorized": False,
            "android_authorized": False,
            "human_authorized": False,
            "production_authorized": False,
        },
        "claim_boundary_drift",
    )
    validation = {
        "schema": VALIDATION_SCHEMA,
        "stage": core.STAGE,
        "status": "VALID",
        "process_id": os.getpid(),
        "inventory_process_id": inventory["process_id"],
        "availability_process_id": availability["process_id"],
        "config_sha256": core.sha256_file(config_path),
        "inventory_sha256": inventory_sha,
        "availability_sha256": core.sha256_file(availability_path),
        "terminal_sha256": core.sha256_file(terminal_path),
        "audit_sha256": core.sha256_file(audit_path),
        "terminal_state": expected_terminal,
        "checks": {
            "inventory_verified_before_b_config": True,
            "a_b_validator_process_isolation": True,
            "sequence_count": sequences,
            "frame_count": frames,
            "source_authority_absence_recomputed": source_absent,
            "denominator_projection_is_aggregate_only": True,
            "forbidden_decode_counters_zero": True,
            "no_signal_slope_frontier_or_schema_repair": True,
        },
    }
    output = root / config["outputs"]["validation"]
    core.atomic_write_json(output, validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.repo.resolve(), args.config.resolve()),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
