from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import cv2
import matplotlib
import numpy as np
from PIL import __version__ as pillow_version

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    PROTOCOL_SHA256,
    load_protocol,
    sha256_file,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.receipt import (
    canonical_json,
    read_jsonl,
    runtime_summary,
    write_json,
    write_jsonl,
)


CANDIDATE_ID = "OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0"
DESIGN_LOCK_SHA256 = (
    "3fcc21e28ba84e18d10b1c236a9a0df167d2a6464ea5ebefcb52ce4395152bac"
)
RECEIPT_SCHEMA_VERSION = (
    "rcle.observable_support_recovery.development_receipt.v1"
)
VALIDATION_SCHEMA_VERSION = (
    "rcle.observable_support_recovery.development_validation.v1"
)
TERMINAL_PASS = "DEVELOPMENT_GATE_PASS_VALIDATION_REMAINS_UNAUTHORIZED"
TERMINAL_FAIL = "CLOSE_OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0"


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
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            result[relative] = sha256_file(path)
    return result


def source_manifest(repo_root: Path) -> dict[str, str]:
    module = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
    )
    roots = [
        module / "rcle_observable_support_r0",
        module / "tests_observable_support_r0",
    ]
    files: list[Path] = []
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name
            != "RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json"
            and path.suffix in {".py", ".md", ".txt", ".json"}
        )
    files.extend(
        [
            module / "run_observable_support_recovery_development_r0.py",
            module
            / "schemas"
            / "rcle_observable_support_recovery_development_receipt_r0.schema.json",
            module / "rcle_minimal" / "evaluation.py",
            module / "rcle_minimal" / "protocol.py",
            module / "rcle_minimal" / "receipt.py",
            module / "rcle_minimal" / "rotation_compensation.py",
            module / "rcle_minimal" / "synthetic_generator.py",
            module / "rcle_minimal_r1" / "local_expansion.py",
            module / "rcle_minimal_r1" / "sparse_flow.py",
            module / "configs" / "phase_a_synthetic_signal_audit_r0.json",
            module / "configs" / "phase_a_synthetic_signal_audit_r0.lock.json",
            repo_root
            / "docs"
            / "research"
            / "rcle"
            / "RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_LOCK_2026-07-26.json",
        ]
    )
    return {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(set(files))
        if path.exists()
    }


def environment_manifest() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": pillow_version,
        "matplotlib": matplotlib.__version__,
        "opencv_threads_in_worker": "1",
    }


def validate_candidate_lock(repo_root: Path) -> dict[str, Any]:
    module = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "rcle_observable_support_r0"
    )
    lock_path = (
        module / "RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    design_path = (
        repo_root
        / "docs"
        / "research"
        / "rcle"
        / "RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_LOCK_2026-07-26.json"
    )
    manifest = source_manifest(repo_root)
    manifest_sha256 = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    expected_output = str(
        repo_root
        / "artifacts.local"
        / "evidence"
        / "rcle_observable_support_recovery_r0"
        / "development_gate_r0"
    )
    schema_path = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "schemas"
        / "rcle_observable_support_recovery_development_receipt_r0.schema.json"
    )
    checks = {
        "candidate_id": lock.get("candidate_id") == CANDIDATE_ID,
        "design_lock_sha256": (
            lock.get("design_lock_sha256") == DESIGN_LOCK_SHA256
            and sha256_file(design_path) == DESIGN_LOCK_SHA256
        ),
        "protocol_sha256": lock.get("protocol_sha256") == PROTOCOL_SHA256,
        "development_seeds": lock.get("development_seeds")
        == list(range(2000, 2020)),
        "planned_trials": lock.get("planned_trials") == 2520,
        "receipt_schema_version": (
            lock.get("receipt_schema_version") == RECEIPT_SCHEMA_VERSION
        ),
        "output_location": (
            lock.get("development_output_location") == expected_output
        ),
        "receipt_schema_sha256": (
            lock.get("receipt_schema_sha256") == sha256_file(schema_path)
        ),
        "tests_locked_count": lock.get("tests_locked_count") == 16,
        "source_manifest": lock.get("source_manifest") == manifest,
        "source_manifest_sha256": (
            lock.get("source_manifest_sha256") == manifest_sha256
        ),
        "environment": lock.get("environment") == environment_manifest(),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"CANDIDATE_LOCK_MISMATCH:{','.join(failed)}")
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
    lock_path = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "rcle_observable_support_r0"
        / "RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json"
    )
    lock = validate_candidate_lock(repo_root)
    dirty = _git(["status", "--short"], repo_root)
    scientific_pass = bool(summary["scientific_gate_pass"])
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "candidate_lock_sha256": sha256_file(lock_path),
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
        "source_manifest": lock["source_manifest"],
        "output_manifest": _manifest(
            output_root,
            excluded={"receipt.json", "receipt_validation.json"},
        ),
        "scientific_summary_sha256": hashlib.sha256(
            canonical_json(summary).encode("utf-8")
        ).hexdigest(),
        "scientific_gate_pass": scientific_pass,
        "terminal_state": TERMINAL_PASS if scientific_pass else TERMINAL_FAIL,
        "authority": "SYNTHETIC_DEVELOPMENT_EVIDENCE_ONLY",
        "recompute_command": [
            sys.executable,
            (
                "scripts/research/egomotion_compensated_looming/"
                "run_observable_support_recovery_development_r0.py"
            ),
            "--validate-existing",
        ],
    }


def validate_existing(repo_root: Path, output_root: Path) -> dict[str, Any]:
    from .evaluation import IMPLEMENTATION_REVISION, summarize_and_decide

    lock_path = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "rcle_observable_support_r0"
        / "RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json"
    )
    validate_candidate_lock(repo_root)
    receipt_path = output_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError("RECEIPT_SCHEMA_VERSION_MISMATCH")
    if receipt["candidate_id"] != CANDIDATE_ID:
        raise ValueError("RECEIPT_CANDIDATE_MISMATCH")
    if receipt["design_lock_sha256"] != DESIGN_LOCK_SHA256:
        raise ValueError("RECEIPT_DESIGN_LOCK_MISMATCH")
    if receipt["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("RECEIPT_PROTOCOL_HASH_MISMATCH")
    if receipt["candidate_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("RECEIPT_CANDIDATE_LOCK_MISMATCH")
    rows = read_jsonl(output_root / "trial_metrics.jsonl")
    if len(rows) != 2520:
        raise ValueError("DEVELOPMENT_TRIAL_COUNT_MISMATCH")
    if {row["seed"] for row in rows} != set(range(2000, 2020)):
        raise ValueError("DEVELOPMENT_SEED_ROLE_MISMATCH")
    if any(
        row.get("implementation_revision") != IMPLEMENTATION_REVISION
        for row in rows
    ):
        raise ValueError("TRIAL_IMPLEMENTATION_REVISION_MISMATCH")
    protocol = load_protocol()
    protocol["trials"]["seeds"] = list(range(2000, 2020))
    recomputed = summarize_and_decide(rows, protocol)
    stored = json.loads(
        (output_root / "scientific_summary.json").read_text(encoding="utf-8")
    )
    if canonical_json(recomputed) != canonical_json(stored):
        raise ValueError("SCIENTIFIC_SUMMARY_RECOMPUTE_MISMATCH")
    scientific_hash = hashlib.sha256(
        canonical_json(recomputed).encode("utf-8")
    ).hexdigest()
    if scientific_hash != receipt["scientific_summary_sha256"]:
        raise ValueError("SCIENTIFIC_SUMMARY_HASH_MISMATCH")
    if source_manifest(repo_root) != receipt["source_manifest"]:
        raise ValueError("SOURCE_MANIFEST_MISMATCH")
    current_output = _manifest(
        output_root,
        excluded={"receipt.json", "receipt_validation.json"},
    )
    if current_output != receipt["output_manifest"]:
        raise ValueError("OUTPUT_MANIFEST_MISMATCH")
    terminal = TERMINAL_PASS if recomputed["scientific_gate_pass"] else TERMINAL_FAIL
    if receipt["terminal_state"] != terminal:
        raise ValueError("TERMINAL_STATE_MISMATCH")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "VALID",
        "candidate_id": CANDIDATE_ID,
        "receipt_sha256": sha256_file(receipt_path),
        "candidate_lock_sha256": sha256_file(lock_path),
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "trial_count": len(rows),
        "scientific_gate_pass": bool(recomputed["scientific_gate_pass"]),
        "terminal_state": terminal,
    }


__all__ = [
    "CANDIDATE_ID",
    "DESIGN_LOCK_SHA256",
    "RECEIPT_SCHEMA_VERSION",
    "TERMINAL_FAIL",
    "TERMINAL_PASS",
    "build_receipt",
    "environment_manifest",
    "runtime_summary",
    "source_manifest",
    "validate_candidate_lock",
    "validate_existing",
    "write_json",
    "write_jsonl",
]
