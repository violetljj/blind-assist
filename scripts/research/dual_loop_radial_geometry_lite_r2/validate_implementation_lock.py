#!/usr/bin/env python3
"""Validate the R2 execution-envelope implementation identity."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
EXPECTED_MODULE_FILES = {
    "README.md",
    "evaluate_replay.py",
    "radial_geometry.py",
    "run_replay.py",
    "test_r2_identity.py",
    "validate_implementation_lock.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "schema",
        lock.get("schema")
        == "blindassist_dual_loop_radial_geometry_implementation_lock_v1",
        lock.get("schema"),
    )
    check(
        "not_run",
        lock.get("execution_state") == "NOT_RUN"
        and lock.get("formal_execution_authorized") is False
        and lock.get("truth_join_authorized") is False,
        {
            "execution_state": lock.get("execution_state"),
            "formal": lock.get("formal_execution_authorized"),
            "truth": lock.get("truth_join_authorized"),
        },
    )
    for name, binding in sorted(lock.get("bindings", {}).items()):
        path = REPO_ROOT / str(binding.get("path", ""))
        actual = sha256_file(path) if path.is_file() else None
        check(
            f"binding_{name}",
            actual == binding.get("sha256"),
            {"expected": binding.get("sha256"), "actual": actual},
        )
    module_hashes = lock.get("module_file_hashes", {})
    check(
        "module_file_set",
        set(module_hashes) == EXPECTED_MODULE_FILES,
        sorted(module_hashes),
    )
    for name, expected in sorted(module_hashes.items()):
        path = MODULE_DIR / name
        actual = sha256_file(path) if path.is_file() else None
        check(
            f"module_{name}",
            actual == expected,
            {"expected": expected, "actual": actual},
        )
    for name, binding in sorted(lock.get("dependency_hashes", {}).items()):
        if binding.get("path"):
            path = REPO_ROOT / str(binding["path"])
        else:
            path = (
                REPO_ROOT
                / "scripts"
                / "research"
                / str(binding.get("module_id", ""))
                / str(binding.get("file", ""))
            )
        actual = sha256_file(path) if path.is_file() else None
        check(
            f"dependency_{name}",
            actual == binding.get("sha256"),
            {"expected": binding.get("sha256"), "actual": actual},
        )
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    geometry = importlib.import_module("radial_geometry")
    evaluator = importlib.import_module("evaluate_replay")
    check(
        "identity",
        geometry.PROTOCOL_ID == lock.get("protocol_id")
        and geometry.IMPLEMENTATION_ID == lock.get("implementation_id")
        and geometry.PARAMETER_SHA256 == lock.get("parameter_sha256"),
        {
            "protocol": geometry.PROTOCOL_ID,
            "implementation": geometry.IMPLEMENTATION_ID,
            "parameter": geometry.PARAMETER_SHA256,
        },
    )
    check(
        "scientific_inheritance",
        geometry.PARAMETER_SHA256 == geometry._R1.PARAMETER_SHA256
        and evaluator.SCIENTIFIC_GATE_CONTRACT_SHA256
        == lock.get("scientific_gate_contract_sha256"),
        {
            "parameter": geometry.PARAMETER_SHA256,
            "gates": evaluator.SCIENTIFIC_GATE_CONTRACT_SHA256,
        },
    )
    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "status": "VALID" if not failures else "INVALID",
        "lock_sha256": sha256_file(lock_path),
        "checks": checks,
        "failures": failures,
        "truth_or_event_opened": False,
        "formal_execution_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.implementation_lock)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
