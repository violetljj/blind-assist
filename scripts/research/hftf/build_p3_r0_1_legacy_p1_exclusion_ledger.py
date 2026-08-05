#!/usr/bin/env python3
"""Build the identity-only P3 exclusion ledger from the consumed legacy P1 roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from p3_r0_1_asset_common import (
    commit_outputs,
    exact_fields,
    load_json,
    output_receipt,
    pretty_bytes,
    reject_outcome_keys,
    request_sha256,
    require,
    validate_protocol,
    verify_bound_file,
    verify_producer_sha,
)


REQUEST_SCHEMA = "blindassist_p3_r0_1_legacy_p1_exclusion_request"
LEDGER_SCHEMA = "blindassist_p3_r0_1_legacy_p1_ancestry_exclusion_ledger"
RECEIPT_SCHEMA = "blindassist_p3_r0_1_legacy_p1_exclusion_materialization_receipt"


def build(repo_root: Path, request: dict[str, Any], source_path: Path) -> None:
    exact_fields(
        request,
        {"schema", "protocol", "legacy_p1_roster", "producer_sha256", "outputs"},
        "request",
    )
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    producer_sha = verify_producer_sha(request["producer_sha256"], source_path)
    _, protocol_sha = validate_protocol(repo_root, request["protocol"])
    roster_path = verify_bound_file(repo_root, request["legacy_p1_roster"], "legacy P1 roster")
    roster = load_json(roster_path)
    reject_outcome_keys(roster, label="legacy P1 roster")
    exact_fields(
        roster,
        {"schema", "data_role", "source_manifest_sha256", "sequence_counts", "rows"},
        "legacy P1 roster",
    )
    require(roster["schema"] == "blindassist_dav2_model_variant_gate_r0_roster", "legacy roster schema drift")
    require(roster["data_role"] == "consumed_development_engineering_regression_only", "legacy roster is not consumed")
    sequence_counts = roster["sequence_counts"]
    require(isinstance(sequence_counts, dict) and sequence_counts, "legacy parent counts missing")
    observed: dict[str, int] = {}
    for row in roster["rows"]:
        require(isinstance(row, dict), "legacy roster row must be an object")
        exact_fields(
            row,
            {"depth_path", "depth_sha256", "frame_id", "index", "intrinsics_fx_fy_cx_cy", "rgb_path", "rgb_sha256", "sequence_id", "sequence_root", "timestamp"},
            "legacy P1 roster row",
        )
        sequence_id = str(row.get("sequence_id", ""))
        require(sequence_id in sequence_counts, "legacy row parent absent from counts")
        observed[sequence_id] = observed.get(sequence_id, 0) + 1
    require(observed == {str(key): int(value) for key, value in sequence_counts.items()}, "legacy sequence counts mismatch")
    ledger = {
        "schema": LEDGER_SCHEMA,
        "status": "FROZEN_CONSUMED_EXCLUSION_ONLY",
        "protocol_sha256": protocol_sha,
        "source_p1_roster_sha256": request["legacy_p1_roster"]["sha256"].upper(),
        "source_manifest_sha256": str(roster["source_manifest_sha256"]).upper(),
        "parent_ids": sorted(observed),
        "frame_count": sum(observed.values()),
        "parent_frame_counts": dict(sorted(observed.items())),
        "outcomes_read": False,
    }
    exact_fields(request["outputs"], {"ledger", "receipt"}, "outputs")
    ledger_bytes = pretty_bytes(ledger)
    outputs = {"ledger": (str(request["outputs"]["ledger"]), ledger_bytes)}
    receipt = output_receipt(
        schema=RECEIPT_SCHEMA,
        producer_sha256=producer_sha,
        request_sha256=request_sha256(request),
        input_sha256={"protocol": protocol_sha, "legacy_p1_roster": request["legacy_p1_roster"]["sha256"].upper()},
        outputs=outputs,
    )
    commit_outputs(
        repo_root,
        outputs=outputs,
        receipt_relative=str(request["outputs"]["receipt"]),
        receipt=receipt,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    build(args.repo_root.resolve(), request, Path(__file__).resolve())


if __name__ == "__main__":
    main()
