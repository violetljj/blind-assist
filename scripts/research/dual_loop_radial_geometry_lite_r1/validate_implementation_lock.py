#!/usr/bin/env python3
"""Validate the R1 implementation lock without opening outcome ledgers."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
EXPECTED_FILES = {
    "README.md",
    "audit_source_shapes.py",
    "evaluate_replay.py",
    "radial_geometry.py",
    "run_replay.py",
    "test_audit_source_shapes.py",
    "test_evaluate_replay.py",
    "test_radial_geometry.py",
    "test_run_replay.py",
    "validate_implementation_lock.py",
}
EXPECTED_SCIENTIFIC_GATE_CONTRACT = {
    "candidate_deadband_per_s": 0.02,
    "event_evaluable": {
        "minimum_finite_rows": 3,
        "minimum_coverage": 0.50,
    },
    "readiness_floor": {
        "minimum_correct_fraction": 0.60,
        "maximum_wrong_signed_fraction": 0.20,
        "minimum_evaluable_fraction": 0.80,
        "minimum_each_truth_state_correct_fraction": 0.50,
    },
    "flow_over_bbox": {
        "minimum_correct_event_gain": 2,
        "wrong_signed_events_must_not_increase": True,
        "maximum_evaluable_event_loss": 23,
        "positive_correct_gain_required_for_targets": [
            "track-000",
            "track-001",
        ],
        "minimum_regions_with_positive_correct_gain": 2,
        "minimum_distinct_events_accounting_for_correct_gain": 2,
    },
    "fixed_primary_event_denominator": 469,
    "non_evaluable_events_count_as_incorrect": True,
    "wrong_sign_pairs": [
        ["approaching", "receding"],
        ["receding", "approaching"],
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def validate(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    _record(
        checks,
        "schema",
        lock.get("schema")
        == "blindassist_dual_loop_radial_geometry_implementation_lock_v1",
        lock.get("schema"),
    )
    _record(
        checks,
        "formal_not_run",
        lock.get("execution_state") == "NOT_RUN"
        and lock.get("formal_execution_authorized") is False,
        {
            "execution_state": lock.get("execution_state"),
            "formal_execution_authorized": lock.get(
                "formal_execution_authorized"
            ),
        },
    )
    bindings = lock.get("bindings", {})
    for label in ("design_lock", "source_shape_audit", "replay_input"):
        binding = bindings.get(label, {})
        path = REPO_ROOT / str(binding.get("path", ""))
        actual = sha256_file(path) if path.is_file() else None
        _record(
            checks,
            f"binding_{label}",
            actual == binding.get("sha256"),
            {"expected": binding.get("sha256"), "actual": actual},
        )

    file_hashes = lock.get("module_file_hashes", {})
    _record(
        checks,
        "module_file_set",
        set(file_hashes) == EXPECTED_FILES,
        {
            "expected": sorted(EXPECTED_FILES),
            "actual": sorted(file_hashes),
        },
    )
    for name, expected in sorted(file_hashes.items()):
        path = MODULE_DIR / name
        actual = sha256_file(path) if path.is_file() else None
        _record(
            checks,
            f"module_hash_{name}",
            actual == expected,
            {"expected": expected, "actual": actual},
        )

    adapter = lock.get("stable_adapter", {})
    adapter_path = REPO_ROOT / str(adapter.get("path", ""))
    adapter_actual = (
        sha256_file(adapter_path) if adapter_path.is_file() else None
    )
    _record(
        checks,
        "stable_adapter_hash",
        adapter_actual == adapter.get("sha256"),
        {"expected": adapter.get("sha256"), "actual": adapter_actual},
    )
    predecessor_dependencies = lock.get("predecessor_dependencies", {})
    _record(
        checks,
        "predecessor_dependency_set",
        set(predecessor_dependencies) == {
            "radial_geometry.py",
            "evaluate_replay.py",
        },
        sorted(predecessor_dependencies),
    )
    for name, expected in sorted(predecessor_dependencies.items()):
        predecessor_path = (
            MODULE_DIR.parents[0]
            / "dual_loop_radial_geometry_lite_r0"
            / name
        )
        predecessor_actual = (
            sha256_file(predecessor_path)
            if predecessor_path.is_file()
            else None
        )
        _record(
            checks,
            f"predecessor_hash_{name}",
            predecessor_actual == expected,
            {"expected": expected, "actual": predecessor_actual},
        )

    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    geometry = importlib.import_module("radial_geometry")
    evaluator = importlib.import_module("evaluate_replay")
    _record(
        checks,
        "implementation_identity",
        geometry.PROTOCOL_ID == lock.get("protocol_id")
        and geometry.IMPLEMENTATION_ID == lock.get("implementation_id")
        and geometry.PARAMETER_SHA256 == lock.get("parameter_sha256"),
        {
            "protocol_id": geometry.PROTOCOL_ID,
            "implementation_id": geometry.IMPLEMENTATION_ID,
            "parameter_sha256": geometry.PARAMETER_SHA256,
        },
    )
    _record(
        checks,
        "shape_guard_contract",
        geometry.PARAMETERS.get("common_shape_guard", {}).get(
            "abstention_reason"
        )
        == "FRAME_SHAPE_CHANGE"
        and geometry.TTL_NS == 100_000_000
        and tuple(geometry.ARMS)
        == ("BBOX_LOG_AREA_GROWTH", "ROI_SPARSE_RADIAL_FLOW"),
        geometry.PARAMETERS.get("common_shape_guard"),
    )
    _record(
        checks,
        "scientific_gate_contract",
        evaluator.SCIENTIFIC_GATE_CONTRACT
        == EXPECTED_SCIENTIFIC_GATE_CONTRACT
        and evaluator.SCIENTIFIC_GATE_CONTRACT_SHA256
        == lock.get("scientific_gate_contract_sha256"),
        {
            "expected": EXPECTED_SCIENTIFIC_GATE_CONTRACT,
            "actual": evaluator.SCIENTIFIC_GATE_CONTRACT,
            "sha256": evaluator.SCIENTIFIC_GATE_CONTRACT_SHA256,
        },
    )

    producer_tree = ast.parse(
        (MODULE_DIR / "run_replay.py").read_text(encoding="utf-8")
    )
    imported = {
        alias.name
        for node in ast.walk(producer_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        str(node.module)
        for node in ast.walk(producer_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden_imports = sorted(
        name for name in imported
        if any(token in name.lower() for token in ("evaluat", "truth", "event"))
    )
    _record(
        checks,
        "producer_import_firewall",
        not forbidden_imports,
        forbidden_imports,
    )

    failures = [check["name"] for check in checks if not check["passed"]]
    return {
        "status": "VALID" if not failures else "INVALID",
        "lock_path": lock_path.as_posix(),
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
