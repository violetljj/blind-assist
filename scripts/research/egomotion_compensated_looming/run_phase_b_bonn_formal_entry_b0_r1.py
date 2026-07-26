#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys


def _repo_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )


def main() -> int:
    if sys.argv[1:] not in ([], ["--validate-existing"]):
        raise SystemExit("usage: runner [--validate-existing]")
    repo_root_text = _repo_root()
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    from pathlib import Path

    from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_b0_r1.protocol import (
        canonical_paths,
        create_run_claim,
        preflight_not_started,
        run_b0,
        sha256_file,
        validate_implementation_lock,
        write_json,
    )

    repo_root = Path(repo_root_text)
    paths = canonical_paths(repo_root)
    lock = validate_implementation_lock(repo_root, paths)
    if sys.argv[1:] == ["--validate-existing"]:
        from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_b0_r1.validator import (
            validate,
        )

        validation = validate(repo_root)
        write_json(paths["validation"], validation)
        print(json.dumps(validation, sort_keys=True))
        return 0

    if lock.get("canonical_execution_authorized") is not True:
        raise SystemExit(
            "canonical execution is not authorized by implementation lock"
        )
    preflight_not_started(paths)
    claim = create_run_claim(repo_root, [sys.executable, *sys.argv])
    receipt = run_b0(repo_root, claim)
    print(
        json.dumps(
            {
                "gate_pass": receipt["gate_pass"],
                "terminal_state": receipt["terminal_state"],
                "receipt_sha256": sha256_file(paths["receipt"]),
                "run_claim_sha256": sha256_file(paths["run_claim"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
