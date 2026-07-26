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

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_authority_r2.authority import (  # noqa: E402
    CANDIDATE_ID,
    build_receipt,
    canonical_json,
    canonical_paths,
    create_preclaim_first,
    sha256_file,
    validate_existing,
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
    paths = canonical_paths(REPO_ROOT)
    if args.validate_existing:
        result = validate_existing(REPO_ROOT)
        write_json(paths["validation"], result)
        print(canonical_json(result))
        return 0

    started_at = datetime.now(HKT).isoformat()
    create_preclaim_first(
        paths["run_claim"],
        {
            "schema_version": (
                "rcle.phase_b.bonn_metadata_authority_r2.run_claim.v1"
            ),
            "candidate_id": CANDIDATE_ID,
            "claimed_at": started_at,
            "canonical_output": str(paths["output"]),
            "maximum_materialization_claims": 1,
            "first_application_file_operation": True,
            "survives_failure_interrupt_and_success": True,
        },
    )
    receipt = build_receipt(
        repo_root=REPO_ROOT,
        paths=paths,
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
