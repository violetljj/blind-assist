#!/usr/bin/env python3
"""Validate the frozen Attempt-04 calibrated factor execution lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT04_CALIBRATED_FACTOR_EXECUTION_LOCK_2026-08-11.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate(lock_path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    bindings = lock["bindings"]
    paths = {key: REPO_ROOT / row["path"] for key, row in bindings.items() if isinstance(row, dict) and "path" in row and "sha256" in row}
    exact = {key: path.is_file() and sha256_file(path) == bindings[key]["sha256"] for key, path in paths.items()}
    runner = paths["runner"].read_text(encoding="utf-8")
    factor_diagnostic = json.loads(paths["factor_selection_diagnostic"].read_text(encoding="utf-8"))
    depth_diagnostic = json.loads(paths["depth_uncertainty_selection"].read_text(encoding="utf-8"))
    factor_variant = next(row for row in factor_diagnostic["variants"] if row["depth_epistemic_weight"] == 0.0 and row["depth_sigma_scale"] == 1.0)
    gates = {
        "A4_E01_BINDINGS_EXACT": all(exact.values()),
        "A4_E02_TEN_PRIMARY_SELECTION_PASS": factor_variant["gate"]["all_primary_metrics_passed"] is True,
        "A4_E03_SUPPORT_AND_BOUNDARY_UNCERTAINTY_SELECTION_PASS": factor_variant["gate"]["uncertainty"]["support"]["passed"] is True and factor_variant["gate"]["uncertainty"]["boundary"]["passed"] is True,
        "A4_E04_DEPTH_UNCERTAINTY_SELECTION_PASS": depth_diagnostic["selected"] is not None and depth_diagnostic["selected"]["eligible"] is True,
        "A4_E05_NO_OPTIMIZER": "torch.optim" not in runner and "optimizer.step" not in runner,
        "A4_E06_CANARY_DELAYED": "if selection_passed" in runner and "canary_after_selection_pass" in runner,
        "A4_E07_FACTOR_SERIALIZATION_RECEIPT_BOUND": "camera_geometry_receipt_sha256" in runner and "BOUND_IN_SOURCE_LABEL_RECEIPT" not in runner,
        "A4_E08_TASK_FIREWALL": "reducer_or_task_outcome_read\": False" in runner,
    }
    return {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt04_execution_lock_validation_v1",
        "passed": all(gates.values()),
        "lock": str(lock_path.resolve()),
        "lock_sha256": sha256_file(lock_path),
        "binding_checks": exact,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    result = validate(args.lock.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
