"""Create the post-review, still-unclassified burned event ledger.

This step binds the output-blind frozen windows to the sealed primitive-review
bundle without inventing scenario categories or physical conditions.  Pilot
coverage remains explicitly unclassified; formal category coverage is a later
cohort gate and is never filled from source-mask discovery metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_file, sha256_json
from .judge_audit import EVENT_LEDGER_SCHEMA
from .prepare_judge_burned_pilot import FREEZE_SCHEMA
from .seal_judge_review_bundle import SEAL_SCHEMA


class EventLedgerError(ValueError):
    """Raised when the burned event ledger cannot be finalized safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EventLedgerError(message)


def finalize(*, freeze: dict[str, Any], seal: dict[str, Any], output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite event ledger: {output}")
    _require(freeze.get("schema_version") == FREEZE_SCHEMA, "pilot freeze schema mismatch")
    _require(freeze.get("protocol_id") == PROTOCOL_ID, "pilot freeze protocol mismatch")
    _require(seal.get("schema_version") == SEAL_SCHEMA, "review seal schema mismatch")
    _require(seal.get("protocol_id") == PROTOCOL_ID, "review seal protocol mismatch")
    _require(seal.get("status") == "PRIMITIVE_REVIEWS_SEALED_BEFORE_PAIR_SELECTION", "review bundle is not sealed")
    _require(seal.get("pilot_freeze_sha256") == sha256_json(freeze), "review seal/freeze binding mismatch")
    items: list[dict[str, Any]] = []
    for row in freeze.get("items", []):
        items.append({
            "event_id": row["pilot_event_id"],
            "source_session_id": row["source_session_id"],
            "discovery_arm": row["discovery_arm"],
            "frame_indices": row["frame_indices"],
            "frame_timestamps_ms": row["frame_timestamps_ms"],
            "coverage": [],
            "coverage_status": "UNCLASSIFIED_PILOT_PENDING",
            "physical_condition": "UNKNOWN",
            "evidence_sufficiency": "UNKNOWN",
            "label_provenance": row["label_provenance"],
        })
    result = {
        "schema_version": EVENT_LEDGER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "cohort_role": "CALIBRATION_BURNED",
        "status": "PRIMITIVE_REVIEW_SEALED_COVERAGE_UNCLASSIFIED",
        "pilot_freeze_sha256": sha256_json(freeze),
        "review_bundle_sha256": seal["review_bundle_sha256"],
        "coverage_is_formal_truth": False,
        "items": items,
        "next_gate": "Run only the burned calibration audit with a CALIBRATION_BURNED contract; formal category coverage remains closed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-freeze", type=Path, required=True)
    parser.add_argument("--review-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(freeze=read_json(args.pilot_freeze), seal=read_json(args.review_seal), output=args.output)
    print(f"status={result['status']} event_count={len(result['items'])} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
