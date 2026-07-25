#!/usr/bin/env python3
"""Independent local reconstruction validator for the JRDB P2 geometry canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    MATERIALIZATION_SCHEMA,
    PACKET_SCHEMA,
    RECEIPT_SCHEMA,
    STAGE,
    audit_packet,
    build_packet,
    canonical_bytes,
    sha256_file,
    write_canonical,
)

VALIDATION_SCHEMA = (
    "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0_validation"
)


def validate(repo: Path, config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    materialization_path = repo / config["outputs"]["materialization"]
    packet_path = repo / config["outputs"]["observation_packet"]
    receipt_path = repo / config["outputs"]["receipt"]
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "materialization_identity": materialization["schema"] == MATERIALIZATION_SCHEMA,
        "receipt_identity": receipt["schema"] == RECEIPT_SCHEMA and receipt["stage"] == STAGE,
        "terminal_legal": receipt["terminal_state"] in config["terminal_states"],
        "all_out_of_scope_authority_closed": not any(config["authority"].values()),
    }
    if materialization["status"] == "MATERIALIZED":
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        rebuilt_packet = build_packet(repo, config_path, materialization)
        checks.update(
            {
                "packet_identity": packet["schema"] == PACKET_SCHEMA and packet["stage"] == STAGE,
                "packet_exact_reconstruction": canonical_bytes(rebuilt_packet) == packet_path.read_bytes(),
                "raw_member_count": packet["raw_payload"]["member_count"] == 362,
                "frame_count": len(packet["frames"]) == int(config["canary"]["frame_count"]),
            }
        )
        rebuilt_receipt = audit_packet(config, rebuilt_packet, sha256_file(packet_path))
        rebuilt_receipt["materialization_sha256"] = sha256_file(materialization_path)
        checks["receipt_exact_reconstruction"] = canonical_bytes(rebuilt_receipt) == receipt_path.read_bytes()
    else:
        checks.update(
            {
                "packet_identity": not packet_path.exists(),
                "packet_exact_reconstruction": True,
                "raw_member_count": True,
                "frame_count": True,
                "receipt_exact_reconstruction": receipt["terminal_state"] == materialization["terminal_state"],
            }
        )
    checks.update(
        {
            "label_fail_prevents_motion": (
                receipt["terminal_state"] != "FAIL_CLOSED_LABEL_JOIN"
                or (
                    receipt.get("motion_pairs") == []
                    and receipt.get("claims", {}).get("source_native_person_motion_available") is False
                    and receipt.get("claims", {}).get("robot_relative_geometry_available") is False
                )
            ),
            "route_event_alert_closed": (
                not receipt["authority"]["route_risk"]
                and not receipt["authority"]["event_lifecycle"]
                and not receipt["authority"]["alert_logic"]
            ),
            "android_human_production_closed": (
                not receipt["authority"]["android"]
                and not receipt["authority"]["human_safety"]
                and not receipt["authority"]["production"]
            ),
        }
    )
    return {
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "status": "VALID" if all(checks.values()) else "INVALID",
        "terminal_state": receipt["terminal_state"],
        "config_sha256": sha256_file(config_path),
        "materialization_sha256": sha256_file(materialization_path),
        "observation_packet_sha256": sha256_file(packet_path) if packet_path.exists() else None,
        "receipt_sha256": sha256_file(receipt_path),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = validate(repo, config_path)
    output = repo / config["outputs"]["validation"]
    write_canonical(output, result)
    print(json.dumps({"output": output.as_posix(), "status": result["status"]}, ensure_ascii=False))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
