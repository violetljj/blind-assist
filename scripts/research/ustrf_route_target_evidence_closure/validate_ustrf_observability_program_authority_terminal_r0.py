#!/usr/bin/env python3
"""Independent validation of the USTRF program authority terminal."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from audit_ustrf_observability_program_authority_terminal_r0 import (
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
    recomputed = audit(repo, config_path)
    checks = {
        "schema": receipt.get("schema") == SCHEMA,
        "stage": receipt.get("stage") == STAGE,
        "process_isolation": receipt.get("process_id") != os.getpid(),
        "deterministic_recomputation": without_process(receipt) == without_process(recomputed),
        "all_authority_checks_passed": all(receipt["checks"].values()),
        "terminal": receipt.get("terminal_state") == "EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY",
        "not_algorithm_rejection": not receipt["non_claims"]["core_hypothesis_rejected"],
        "not_unobservability_claim": not receipt["non_claims"]["task_unobservable_with_authoritative_inputs"],
        "production_closed": not receipt["non_claims"]["production_ready"],
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
    }
    atomic_write(repo / config["outputs"]["validation"], canonical_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
