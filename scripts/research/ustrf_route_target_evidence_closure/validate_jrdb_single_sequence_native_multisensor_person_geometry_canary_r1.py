#!/usr/bin/env python3
"""Independent deterministic validator for the corrected JRDB geometry R1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    canonical_bytes,
    sha256_file,
    write_canonical,
)
from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1 import (
    LEDGER_SCHEMA,
    RECEIPT_SCHEMA,
    STAGE,
    run,
)

VALIDATION_SCHEMA = (
    "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1_validation"
)


def validate(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger_path = repo / config["outputs"]["eligibility_ledger"]
    receipt_path = repo / config["outputs"]["receipt"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rebuilt_ledger, rebuilt_receipt = run(repo, config_path)
    checks = {
        "ledger_identity": ledger["schema"] == LEDGER_SCHEMA and ledger["stage"] == STAGE,
        "receipt_identity": receipt["schema"] == RECEIPT_SCHEMA and receipt["stage"] == STAGE,
        "ledger_exact_reconstruction": canonical_bytes(rebuilt_ledger) == ledger_path.read_bytes(),
        "receipt_exact_reconstruction": canonical_bytes(rebuilt_receipt) == receipt_path.read_bytes(),
        "parent_terminal_preserved": (
            receipt["parent_r0"]["terminal_preserved"] is True
            and receipt["parent_r0"]["terminal_state"] == "FAIL_CLOSED_LABEL_JOIN"
        ),
        "artifact_integrity_valid": receipt["artifact_integrity"] == "VALID",
        "denominator_conservation": all(
            row["expected"] == row["eligible"] + row["abstained"] + row["invalid"]
            for row in receipt["denominators"].values()
        ),
        "cross_modal_missingness_does_not_remove_3d": (
            receipt["denominators"]["robot_relative_3d_geometry"]["eligible"]
            > receipt["denominators"]["cross_modal_from_3d"]["eligible"]
        ),
        "interpolated_not_direct": (
            receipt["support"]["direct_3d_observations"] == 0
            and receipt["support"]["direct_motion_pairs"] == 0
        ),
        "diagnostic_ceiling": receipt["authority_ceiling"] == "DIAGNOSTIC",
        "route_event_product_closed": (
            not receipt["authority"]["route_risk"]
            and not receipt["authority"]["event_lifecycle"]
            and not receipt["authority"]["alert_logic"]
            and not receipt["authority"]["android"]
            and not receipt["authority"]["human_safety"]
            and not receipt["authority"]["production"]
        ),
        "terminal_legal": receipt["terminal_state"] in config["terminal_states"],
    }
    return {
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "status": "VALID" if all(checks.values()) else "INVALID",
        "terminal_state": receipt["terminal_state"],
        "config_sha256": sha256_file(config_path),
        "eligibility_ledger_sha256": sha256_file(ledger_path),
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
