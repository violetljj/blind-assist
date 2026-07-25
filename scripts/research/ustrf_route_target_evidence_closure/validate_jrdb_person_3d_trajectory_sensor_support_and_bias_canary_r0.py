#!/usr/bin/env python3
"""Independently rebuild and validate the JRDB sensor-support/bias canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0 import (
    CLASSES,
    CONFIG_SCHEMA,
    LEDGER_SCHEMA,
    RECEIPT_SCHEMA,
    STAGE,
    run,
)
from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    canonical_bytes,
    sha256_file,
    write_canonical,
)

VALIDATION_SCHEMA = "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0_validation"


def check_denominator(value: dict[str, Any]) -> bool:
    return (
        value["expected"] == sum(value[name] for name in CLASSES)
        and value["conserved"] is True
    )


def validate(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger_path = repo / config["outputs"]["ledger"]
    receipt_path = repo / config["outputs"]["receipt"]
    actual_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    actual_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rebuilt_ledger, rebuilt_receipt = run(repo, config_path)
    checks = {
        "config_identity": config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE,
        "ledger_identity": actual_ledger["schema"] == LEDGER_SCHEMA and actual_ledger["stage"] == STAGE,
        "receipt_identity": actual_receipt["schema"] == RECEIPT_SCHEMA and actual_receipt["stage"] == STAGE,
        "ledger_exact_rebuild": canonical_bytes(actual_ledger) == canonical_bytes(rebuilt_ledger),
        "receipt_exact_rebuild": canonical_bytes(actual_receipt) == canonical_bytes(rebuilt_receipt),
        "all_denominators_conserved": all(check_denominator(value) for value in actual_ledger["denominators"].values()),
        "parent_object_denominator_preserved": actual_ledger["denominators"]["computable_3d_object_frames"]["expected"] == 1350,
        "parent_pair_denominator_preserved": actual_ledger["denominators"]["motion_pairs"]["expected"] == 1336,
        "triple_denominator_recomputed": actual_ledger["denominators"]["acceleration_triples"]["expected"] == 1322,
        "all_120_frames_redecoded": len(actual_ledger["frame_audits"]) == 120,
        "dual_sensor_audit_present": all("upper" in row and "lower" in row for row in actual_ledger["frame_audits"]),
        "no_point_loss_in_decode": all(
            row[sensor]["declared_points"] == row[sensor]["finite_points"] + row[sensor]["nonfinite_points"]
            for row in actual_ledger["frame_audits"]
            for sensor in ("upper", "lower")
        ),
        "local_degradation_preserved": actual_ledger["denominators"]["computable_3d_object_frames"]["invalid"] == 0,
        "annotation_conditioning_disclosed": actual_ledger["support_contract"]["box_query_is_annotation_conditioned"] is True,
        "diagnostic_authority_only": actual_receipt["authority"]["ceiling"] == "DIAGNOSTIC"
        and not any(
            actual_receipt["authority"][key]
            for key in ("candidate_selection", "route_risk", "event_lifecycle", "alert_logic", "android", "human_safety", "production")
        ),
        "terminal_legal": actual_receipt["terminal_state"] in config["terminal_states"],
    }
    return {
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "status": "VALID" if all(checks.values()) else "INVALID",
        "checks": checks,
        "config_sha256": sha256_file(config_path),
        "ledger_sha256": sha256_file(ledger_path),
        "receipt_sha256": sha256_file(receipt_path),
        "recomputed_terminal_state": rebuilt_receipt["terminal_state"],
        "authority": config["authority"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = validate(repo, config_path)
    write_canonical(repo / config["outputs"]["validation"], result)
    print(json.dumps({"status": result["status"], "checks": sum(result["checks"].values()), "total": len(result["checks"])}))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
