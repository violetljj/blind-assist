#!/usr/bin/env python3
"""Create the hash-bound D0-R3 implementation lock without granting authority."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from validate_implementation_lock import (
    LOCK_SCHEMA,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    _expected_implementation_paths,
    _normalized_input_binding,
    canonical_sha256,
    sha256_file,
    write_exclusive_fsync_json,
)


def _git_identity(repo_root: Path) -> dict[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    if status:
        raise ValueError("implementation lock requires a clean worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if head != origin:
        raise ValueError("implementation lock requires HEAD == origin/master")
    return {"head": head, "origin_master": origin}


def build_lock(repo_root: Path) -> dict[str, object]:
    protocol_path = repo_root / PROTOCOL_PATH
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 drift")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen_inputs = {
        name: _normalized_input_binding(specification)
        for name, specification in protocol["frozen_inputs"].items()
    }
    predecessor = protocol["predecessor_gate"]
    implementation_paths = _expected_implementation_paths(repo_root, protocol)
    return {
        "schema_version": LOCK_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "implementation_status": "FROZEN_FOR_INDEPENDENT_REVIEW",
        "execution_state": "NOT_RUN",
        "authority": {
            "activation_authorized": False,
            "formal_execution_authorized": False,
            "scientific_exit_authorized": False,
        },
        "protocol": {
            "path": PROTOCOL_PATH,
            "sha256": PROTOCOL_SHA256,
        },
        "predecessor_bindings": {
            "result": dict(predecessor["result"]),
            "independent_validation": dict(predecessor["independent_validation"]),
            "seal": dict(predecessor["seal"]),
        },
        "scientific_contract_binding": dict(
            protocol["scientific_contract_binding"]
        ),
        "r1_failure_gate": dict(protocol["r1_failure_gate"]),
        "r2_failure_gate": dict(protocol["r2_failure_gate"]),
        "runtime_environment": dict(protocol["runtime_environment"]),
        "frozen_inputs": frozen_inputs,
        "frozen_inputs_sha256": canonical_sha256(frozen_inputs),
        "canonical_serialization_sha256": canonical_sha256(
            protocol["planned_implementation"]["canonical_serialization"]
        ),
        "implementation_file_hashes": {
            relative: sha256_file(repo_root / relative)
            for relative in sorted(implementation_paths)
        },
        "repository": _git_identity(repo_root),
        "claim_ceiling": protocol["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = build_lock(root)
    write_exclusive_fsync_json(output, lock)
    print(json.dumps(lock, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
