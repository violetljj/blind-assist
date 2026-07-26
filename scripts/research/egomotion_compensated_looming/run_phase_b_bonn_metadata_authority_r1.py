#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_authority_r1.authority import (  # noqa: E402
    CANDIDATE_ID,
    build_receipt,
    canonical_json,
    canonical_paths,
    create_exclusive_claim,
    sha256_file,
    validate_existing,
    validate_implementation_lock,
)


HKT = timezone(timedelta(hours=8))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, paths = validate_implementation_lock(REPO_ROOT)
    if args.validate_existing:
        result = validate_existing(REPO_ROOT)
        write_json(paths["validation"], result)
        print(canonical_json(result))
        return 0

    started_at = datetime.now(HKT).isoformat()
    claim = {
        "schema_version": "rcle.phase_b.bonn_metadata_authority_r1.run_claim.v1",
        "candidate_id": CANDIDATE_ID,
        "claimed_at": started_at,
        "implementation_lock_sha256": sha256_file(
            paths["implementation_lock"]
        ),
        "canonical_output": str(paths["output"]),
        "maximum_materialization_claims": 1,
        "survives_failure_or_interrupt": True,
    }
    create_exclusive_claim(paths["run_claim"], claim)
    receipt = build_receipt(
        repo_root=REPO_ROOT,
        started_at=started_at,
        finished_at=datetime.now(HKT).isoformat(),
        command=[sys.executable, *sys.argv],
    )
    write_json(paths["receipt"], receipt)
    print(
        canonical_json(
            {
                "gate_pass": True,
                "terminal_state": receipt["terminal_state"],
                "cohort_identity_sha256": receipt[
                    "cohort_identity_sha256"
                ],
                "receipt_sha256": sha256_file(paths["receipt"]),
                "run_claim_sha256": sha256_file(paths["run_claim"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
