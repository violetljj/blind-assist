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

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_entry_r0.gate import (  # noqa: E402
    build_receipt,
    canonical_json,
    sha256_file,
    validate_existing,
    validate_receipt_shape,
)


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_OFFICIAL_PAGE = (
    REPO_ROOT
    / "artifacts.local"
    / "datasets"
    / "egomotion_compensated_looming_r1"
    / "bonn_metadata_r0"
    / "official_page.html"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts.local"
    / "evidence"
    / "rcle_phase_b_bonn_entry_r0"
    / "metadata_gate_r0"
)
DEFAULT_LOCK = (
    MODULE_ROOT
    / "rcle_phase_b_bonn_entry_r0"
    / "RCLE_PHASE_B_BONN_METADATA_GATE_R0_IMPLEMENTATION_LOCK.json"
)
HONG_KONG_TIMEZONE = timezone(timedelta(hours=8))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-page", type=Path, default=DEFAULT_OFFICIAL_PAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt_path = args.output_root / "receipt.json"
    validation_path = args.output_root / "receipt_validation.json"
    if args.validate_existing:
        result = validate_existing(
            repo_root=REPO_ROOT,
            official_page_path=args.official_page,
            lock_path=args.lock,
            receipt_path=receipt_path,
        )
        write_json(validation_path, result)
        print(canonical_json(result))
        return 0

    if receipt_path.exists() or validation_path.exists():
        raise RuntimeError(
            "METADATA_GATE_ALREADY_MATERIALIZED_USE_VALIDATE_EXISTING"
        )
    started_at = datetime.now(HONG_KONG_TIMEZONE).isoformat()
    command = [sys.executable, *sys.argv]
    receipt = build_receipt(
        repo_root=REPO_ROOT,
        official_page_path=args.official_page,
        lock_path=args.lock,
        command=command,
        started_at=started_at,
        finished_at=datetime.now(HONG_KONG_TIMEZONE).isoformat(),
    )
    validate_receipt_shape(receipt)
    write_json(receipt_path, receipt)
    print(
        canonical_json(
            {
                "gate_pass": receipt["gate_pass"],
                "terminal_state": receipt["terminal_state"],
                "official_universe_count": receipt[
                    "official_universe_count"
                ],
                "selected_sequence_ids": receipt["selected_sequence_ids"],
                "cohort_identity_sha256": receipt[
                    "cohort_identity_sha256"
                ],
                "receipt_sha256": sha256_file(receipt_path),
            }
        )
    )
    return 0 if receipt["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
