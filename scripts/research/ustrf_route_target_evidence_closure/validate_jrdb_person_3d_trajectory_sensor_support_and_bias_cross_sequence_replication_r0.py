#!/usr/bin/env python3
"""Independently rebuild the JRDB cross-sequence replication artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0 import (
    LEDGER_SCHEMA,
    RECEIPT_SCHEMA,
    build_aggregate,
    build_input_manifest,
    load_config,
    load_freeze,
)
from freeze_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0 import (
    STAGE,
)
from run_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0 import CLASSES
from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    canonical_bytes,
    sha256_file,
    write_canonical,
)

VALIDATION_SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0_validation"
)


def conserved(row: dict[str, Any]) -> bool:
    return row["expected"] == sum(row[name] for name in CLASSES) and row["conserved"] is True


def validate(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(repo, config_path)
    freeze = load_freeze(repo, config, config_path)
    manifest_path = repo / config["outputs"]["input_manifest"]
    ledger_path = repo / config["outputs"]["ledger"]
    receipt_path = repo / config["outputs"]["receipt"]
    actual_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    actual_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rebuilt_ledger, rebuilt_receipt = build_aggregate(repo, config, config_path)
    pooled = actual_ledger["pooled_primitives"]
    per_sequences = actual_ledger["per_sequences"]
    checks = {
        "freeze_exact_rebuild": len(freeze["selected"]) == 3,
        "input_manifest_exact_rebuild": canonical_bytes(build_input_manifest(repo, config, config_path)) == manifest_path.read_bytes(),
        "ledger_identity": actual_ledger["schema"] == LEDGER_SCHEMA and actual_ledger["stage"] == STAGE,
        "receipt_identity": actual_receipt["schema"] == RECEIPT_SCHEMA and actual_receipt["stage"] == STAGE,
        "ledger_exact_rebuild": canonical_bytes(rebuilt_ledger) == ledger_path.read_bytes(),
        "receipt_exact_rebuild": canonical_bytes(rebuilt_receipt) == receipt_path.read_bytes(),
        "three_new_sequences": len(per_sequences) == 3
        and len({row["sequence"] for row in per_sequences}) == 3
        and config["baseline"]["sequence"] not in {row["sequence"] for row in per_sequences},
        "all_windows_120": all(len(row["ledger"]["frame_audits"]) == 120 for row in per_sequences),
        "per_sequence_denominators_conserved": all(
            conserved(value)
            for row in per_sequences
            for value in row["ledger"]["denominators"].values()
        ),
        "pooled_denominators_conserved": all(conserved(value) for value in actual_ledger["pooled_denominators"].values()),
        "pooled_primitives_are_concatenated": (
            len(pooled["object_frames"]) == sum(len(row["ledger"]["object_frames"]) for row in per_sequences)
            and len(pooled["motion_pairs"]) == sum(len(row["ledger"]["motion_pairs"]) for row in per_sequences)
            and len(pooled["acceleration_triples"]) == sum(len(row["ledger"]["acceleration_triples"]) for row in per_sequences)
        ),
        "parent_denominators_not_meyer_hardcoded": all(
            row["ledger"]["denominators"]["computable_3d_object_frames"]["expected"]
            == row["ledger"]["denominators"]["computable_3d_object_frames"]["sensor-supported"]
            + row["ledger"]["denominators"]["computable_3d_object_frames"]["annotation-only"]
            + row["ledger"]["denominators"]["computable_3d_object_frames"]["abstained"]
            + row["ledger"]["denominators"]["computable_3d_object_frames"]["invalid"]
            for row in per_sequences
        ),
        "worst_sequence_traceable": all(
            value.get("status") == "NOT_EVALUABLE"
            or value.get("sequence") in {row["sequence"] for row in per_sequences}
            for value in actual_ledger["worst_sequence"].values()
        ),
        "direction_status_legal": all(
            value["status"] in ("DIRECTION_REPLICATED", "MIXED_OR_CONTRADICTED", "NOT_EVALUABLE")
            for value in actual_ledger["directional_replication"].values()
        ),
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
        "sequence_freeze_sha256": sha256_file(repo / config["outputs"]["sequence_freeze"]),
        "input_manifest_sha256": sha256_file(manifest_path),
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
    output = repo / config["outputs"]["validation"]
    if output.exists():
        raise RuntimeError("validation already exists; never overwrite")
    write_canonical(output, result)
    print(json.dumps({"status": result["status"], "checks": sum(result["checks"].values()), "total": len(result["checks"])}))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
