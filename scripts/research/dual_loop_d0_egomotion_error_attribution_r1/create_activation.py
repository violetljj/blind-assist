#!/usr/bin/env python3
"""Create D0-R1 activation only after a hash-bound independent review passes."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from contract import PROTOCOL_ID, PROTOCOL_SHA256, sha256_file
from validate_implementation_lock import (
    ACTIVATION_SCHEMA,
    canonical_json_bytes,
    validate,
    write_exclusive_fsync_json,
)


FORMAL_OUTPUT_ROOT = (
    "artifacts.local/evidence/dual-loop/"
    "d0-egomotion-error-attribution-r1/run-r1"
)
IMPLEMENTATION_REVIEW_SCHEMA = (
    "blindassist.d0_egomotion_error_attribution.implementation_review.v1"
)


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


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
        raise ValueError("activation requires a clean worktree")
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
        raise ValueError("activation requires HEAD == origin/master")
    return {"head": head, "origin_master": origin}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--implementation-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    lock_path = args.implementation_lock.resolve()
    review_path = args.implementation_review.resolve()
    output = args.output.resolve()
    if validate(lock_path, root)["status"] != "VALID":
        raise ValueError("implementation lock is not valid")
    repository = _git_identity(root)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes.decode("utf-8"))
    if review_bytes != canonical_json_bytes(review) + b"\n":
        raise ValueError("implementation review receipt is not canonical JSON")
    expected_lock = {
        "path": _relative(root, lock_path),
        "sha256": sha256_file(lock_path),
    }
    expected_protocol = {
        "protocol_id": PROTOCOL_ID,
        "sha256": PROTOCOL_SHA256,
    }
    checks = review.get("checks")
    if (
        review.get("schema_version") != IMPLEMENTATION_REVIEW_SCHEMA
        or review.get("status") != "PASS"
        or review.get("reviewer_role") != "INDEPENDENT_READ_ONLY_REVIEW"
        or review.get("protocol") != expected_protocol
        or review.get("implementation_lock") != expected_lock
        or review.get("repository") != repository
        or lock.get("repository") != repository
        or review.get("formal_execution_authorized") is not False
        or not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(check, dict) or check.get("passed") is not True
            for check in checks
        )
    ):
        raise ValueError("independent implementation review receipt is invalid")
    formal_namespace = (root / FORMAL_OUTPUT_ROOT).resolve()
    if formal_namespace.exists():
        raise ValueError("formal output namespace already exists")
    activation = {
        "schema_version": ACTIVATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_state": "NOT_RUN",
        "formal_output_root": FORMAL_OUTPUT_ROOT,
        "implementation_lock": expected_lock,
        "implementation_review": {
            "path": _relative(root, review_path),
            "sha256": sha256_file(review_path),
        },
        "repository": repository,
        "formal_execution_authorized": True,
        "authority": {
            "formal_execution_authorized": True,
            "successor_execution_authorized": False,
            "confirmation_authorized": False,
            "product_or_safety_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive_fsync_json(output, activation)
    print(json.dumps(activation, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
