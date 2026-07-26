from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    PROTOCOL_SHA256,
    load_protocol,
    sha256_file,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.receipt import (
    canonical_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0.evaluation import (
    IMPLEMENTATION_REVISION,
    summarize_and_decide,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0.receipt import (
    DESIGN_LOCK_SHA256,
    environment_manifest,
    source_manifest,
    validate_candidate_lock,
)


CANDIDATE_ID = IMPLEMENTATION_REVISION
CANDIDATE_LOCK_SHA256 = (
    "a1dc1388ea6b6cb8ff7e7541da407cb827b6384678cdf46a097426a2111a5497"
)
DEVELOPMENT_RECEIPT_SHA256 = (
    "93b4c9244e9ef3bd11e8ab3557bfda0ad6dd6cd324116dbd05898f71b5214e3c"
)
RECEIPT_SCHEMA_VERSION = (
    "rcle.observable_support_recovery.sealed_validation_receipt.v1"
)
VALIDATION_SCHEMA_VERSION = (
    "rcle.observable_support_recovery.sealed_validation_recompute.v1"
)
TERMINAL_PASS = (
    "INDEPENDENT_SYNTHETIC_PASS_ONLY_PHASE_B_REMAINS_CLOSED_PENDING_SEPARATE_DECISION"
)
TERMINAL_FAIL = (
    "CLOSE_OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0_WITH_NO_PATCH_OR_RERUN"
)


def _git(args: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _manifest(root: Path, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded or set()
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() not in excluded
    }


def validation_control_manifest(repo_root: Path) -> dict[str, str]:
    module = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
    )
    files = [
        module / "run_observable_support_recovery_sealed_validation_r0.py",
        module / "rcle_observable_support_validation_r0" / "__init__.py",
        module / "rcle_observable_support_validation_r0" / "receipt.py",
        module
        / "schemas"
        / "rcle_observable_support_recovery_sealed_validation_receipt_r0.schema.json",
        module
        / "tests_observable_support_validation_r0"
        / "__init__.py",
        module
        / "tests_observable_support_validation_r0"
        / "test_sealed_validation_boundary_r0.py",
    ]
    return {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in files
        if path.exists()
    }


def _paths(repo_root: Path) -> tuple[Path, Path, Path]:
    module = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
    )
    candidate_lock = (
        module
        / "rcle_observable_support_r0"
        / "RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json"
    )
    development_receipt = (
        repo_root
        / "artifacts.local"
        / "evidence"
        / "rcle_observable_support_recovery_r0"
        / "development_gate_r0"
        / "receipt.json"
    )
    validation_lock = (
        module
        / "rcle_observable_support_validation_r0"
        / "RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_VALIDATION_LOCK.json"
    )
    return candidate_lock, development_receipt, validation_lock


def validate_validation_lock(repo_root: Path) -> dict[str, Any]:
    candidate_lock_path, development_receipt_path, validation_lock_path = (
        _paths(repo_root)
    )
    validate_candidate_lock(repo_root)
    lock = json.loads(validation_lock_path.read_text(encoding="utf-8"))
    output = str(
        repo_root
        / "artifacts.local"
        / "evidence"
        / "rcle_observable_support_recovery_r0"
        / "sealed_validation_gate_r0"
    )
    checks = {
        "candidate_id": lock.get("candidate_id") == CANDIDATE_ID,
        "design_lock": lock.get("design_lock_sha256") == DESIGN_LOCK_SHA256,
        "candidate_lock": (
            lock.get("candidate_lock_sha256") == CANDIDATE_LOCK_SHA256
            and sha256_file(candidate_lock_path) == CANDIDATE_LOCK_SHA256
        ),
        "development_receipt": (
            lock.get("development_receipt_sha256")
            == DEVELOPMENT_RECEIPT_SHA256
            and sha256_file(development_receipt_path)
            == DEVELOPMENT_RECEIPT_SHA256
        ),
        "protocol": lock.get("protocol_sha256") == PROTOCOL_SHA256,
        "seeds": lock.get("sealed_validation_seeds")
        == list(range(3000, 3020)),
        "trials": lock.get("planned_trials") == 2520,
        "output": lock.get("validation_output_location") == output,
        "schema": lock.get("receipt_schema_version")
        == RECEIPT_SCHEMA_VERSION,
        "environment": lock.get("environment") == environment_manifest(),
        "candidate_sources": lock.get("candidate_source_manifest")
        == source_manifest(repo_root),
        "validation_controls": lock.get("validation_control_manifest")
        == validation_control_manifest(repo_root),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"VALIDATION_LOCK_MISMATCH:{','.join(failed)}")
    return lock


def build_receipt(
    repo_root: Path,
    output_root: Path,
    command: Sequence[str],
    summary: dict[str, Any],
    started_at: str,
    finished_at: str,
    worker_count: int,
) -> dict[str, Any]:
    candidate_lock_path, development_receipt_path, validation_lock_path = (
        _paths(repo_root)
    )
    lock = validate_validation_lock(repo_root)
    dirty = _git(["status", "--short"], repo_root)
    scientific_pass = bool(summary["scientific_gate_pass"])
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "candidate_lock_sha256": sha256_file(candidate_lock_path),
        "development_receipt_sha256": sha256_file(development_receipt_path),
        "protocol_sha256": PROTOCOL_SHA256,
        "validation_lock_sha256": sha256_file(validation_lock_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "command": list(command),
        "worker_count": worker_count,
        "repo": {
            "head": _git(["rev-parse", "HEAD"], repo_root),
            "branch": _git(["branch", "--show-current"], repo_root),
            "dirty": bool(dirty),
            "status_short": dirty.splitlines(),
        },
        "environment": environment_manifest(),
        "candidate_source_manifest": lock["candidate_source_manifest"],
        "output_manifest": _manifest(
            output_root,
            excluded={"receipt.json", "receipt_validation.json"},
        ),
        "scientific_summary_sha256": hashlib.sha256(
            canonical_json(summary).encode("utf-8")
        ).hexdigest(),
        "scientific_gate_pass": scientific_pass,
        "terminal_state": TERMINAL_PASS if scientific_pass else TERMINAL_FAIL,
        "authority": "INDEPENDENT_SYNTHETIC_VALIDATION_EVIDENCE_ONLY",
        "recompute_command": [
            sys.executable,
            (
                "scripts/research/egomotion_compensated_looming/"
                "run_observable_support_recovery_sealed_validation_r0.py"
            ),
            "--validate-existing",
        ],
    }


def validate_existing(repo_root: Path, output_root: Path) -> dict[str, Any]:
    _, _, validation_lock_path = _paths(repo_root)
    validate_validation_lock(repo_root)
    receipt_path = output_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError("VALIDATION_RECEIPT_SCHEMA_MISMATCH")
    if receipt["validation_lock_sha256"] != sha256_file(validation_lock_path):
        raise ValueError("VALIDATION_LOCK_HASH_MISMATCH")
    rows = read_jsonl(output_root / "trial_metrics.jsonl")
    if len(rows) != 2520:
        raise ValueError("SEALED_VALIDATION_TRIAL_COUNT_MISMATCH")
    if {row["seed"] for row in rows} != set(range(3000, 3020)):
        raise ValueError("SEALED_VALIDATION_SEED_ROLE_MISMATCH")
    if any(
        row.get("implementation_revision") != CANDIDATE_ID for row in rows
    ):
        raise ValueError("SEALED_VALIDATION_CANDIDATE_MISMATCH")
    protocol = load_protocol()
    protocol["trials"]["seeds"] = list(range(3000, 3020))
    recomputed = summarize_and_decide(rows, protocol)
    stored = json.loads(
        (output_root / "scientific_summary.json").read_text(encoding="utf-8")
    )
    if canonical_json(recomputed) != canonical_json(stored):
        raise ValueError("SEALED_VALIDATION_SUMMARY_RECOMPUTE_MISMATCH")
    scientific_hash = hashlib.sha256(
        canonical_json(recomputed).encode("utf-8")
    ).hexdigest()
    if scientific_hash != receipt["scientific_summary_sha256"]:
        raise ValueError("SEALED_VALIDATION_SUMMARY_HASH_MISMATCH")
    if source_manifest(repo_root) != receipt["candidate_source_manifest"]:
        raise ValueError("SEALED_VALIDATION_CANDIDATE_SOURCE_DRIFT")
    current_output = _manifest(
        output_root,
        excluded={"receipt.json", "receipt_validation.json"},
    )
    if current_output != receipt["output_manifest"]:
        raise ValueError("SEALED_VALIDATION_OUTPUT_MANIFEST_MISMATCH")
    terminal = TERMINAL_PASS if recomputed["scientific_gate_pass"] else TERMINAL_FAIL
    if receipt["terminal_state"] != terminal:
        raise ValueError("SEALED_VALIDATION_TERMINAL_STATE_MISMATCH")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "VALID",
        "candidate_id": CANDIDATE_ID,
        "receipt_sha256": sha256_file(receipt_path),
        "validation_lock_sha256": sha256_file(validation_lock_path),
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
        "development_receipt_sha256": DEVELOPMENT_RECEIPT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "trial_count": len(rows),
        "scientific_gate_pass": bool(recomputed["scientific_gate_pass"]),
        "terminal_state": terminal,
    }


__all__ = [
    "CANDIDATE_ID",
    "RECEIPT_SCHEMA_VERSION",
    "TERMINAL_FAIL",
    "TERMINAL_PASS",
    "build_receipt",
    "validate_existing",
    "validate_validation_lock",
    "validation_control_manifest",
    "write_json",
    "write_jsonl",
]
