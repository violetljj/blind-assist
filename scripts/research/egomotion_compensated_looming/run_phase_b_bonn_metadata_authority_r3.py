#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import sys


def _lexical_repo_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )


def _claim_first(repo_root: str, started_at: str) -> None:
    output = os.path.normpath(
        os.path.join(
            repo_root,
            "artifacts.local",
            "evidence",
            "rcle_phase_b_bonn_entry_r3",
            "authority_gate_r3",
        )
    )
    claim_path = os.path.join(output, "run_claim.json")
    claim = {
        "schema_version": (
            "rcle.phase_b.bonn_metadata_authority_r3.run_claim.v1"
        ),
        "candidate_id": "BONN_MINIMAL_BOOTSTRAP_PRECLAIM_AUTHORITY_R3",
        "claimed_at": started_at,
        "canonical_output": output,
        "maximum_materialization_claims": 1,
        "first_application_file_operation": True,
        "survives_failure_interrupt_and_success": True,
    }
    descriptor = os.open(
        claim_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        os.write(
            descriptor,
            (json.dumps(claim, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    validate_only = sys.argv[1:] == ["--validate-existing"]
    if sys.argv[1:] not in ([], ["--validate-existing"]):
        raise SystemExit("usage: runner [--validate-existing]")
    repo_root_text = _lexical_repo_root()
    started_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    if not validate_only:
        _claim_first(repo_root_text, started_at)

    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    from pathlib import Path

    from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_authority_r3.authority import (
        build_receipt,
        canonical_json,
        canonical_paths,
        sha256_file,
        validate_existing,
        write_json,
    )

    repo_root = Path(repo_root_text)
    paths = canonical_paths(repo_root)
    if validate_only:
        result = validate_existing(repo_root)
        write_json(paths["validation"], result)
        print(canonical_json(result))
        return 0

    receipt = build_receipt(
        repo_root=repo_root,
        started_at=started_at,
        command=[sys.executable, *sys.argv],
    )
    write_json(paths["receipt"], receipt)
    print(
        canonical_json(
            {
                "gate_pass": True,
                "terminal_state": receipt["terminal_state"],
                "receipt_sha256": sha256_file(paths["receipt"]),
                "run_claim_sha256": sha256_file(paths["run_claim"]),
                "cohort_identity_sha256": receipt[
                    "cohort_identity_sha256"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
