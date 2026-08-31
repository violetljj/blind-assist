#!/usr/bin/env python3
"""Post-hoc consumed-Development successor for grouped brand-sign consensus.

This wrapper preserves the V1 runner and changes one observed brittle surface:
two spatially distinct initial-mark rows are scored as a group rather than each
being required to clear the same standalone confidence threshold.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import l10_panolab_track_lexical_ledger as base


PROTOCOL_SCHEMA = "blindassist-l10-panolab-track-lexical-consensus-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-track-lexical-consensus-result-v1"


def match_target_name(
    rows: list[dict[str, Any]],
    entity_name: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    target = base.significant_name_tokens(entity_name, contract)
    exact = [
        row for row in rows
        if row["score"] >= float(contract["minimum_exact_name_row_score"])
        and base.contains_contiguous(row["ascii_tokens"], target)
    ]
    if exact:
        return {
            "matched": True,
            "tier": "EXACT_SIGNIFICANT_NAME_TOKENS_IN_ONE_ROW",
            "target_tokens": target,
            "witnesses": exact,
        }

    grouped = contract["two_initial_brand_signature"]
    if len(target) != 2 or not all(len(token) == 1 and token.isalpha() for token in target):
        return {"matched": False, "tier": "NONE", "target_tokens": target, "witnesses": []}
    candidates = []
    for row in rows:
        joined = "".join(row["ascii_tokens"])
        if (
            row["score"] >= float(grouped["minimum_member_row_score"])
            and int(grouped["minimum_canonical_characters"]) <= len(joined) <= int(grouped["maximum_canonical_characters"])
            and joined.startswith(target[0])
            and joined.endswith(target[1])
        ):
            candidates.append(row)
    distinct = []
    for candidate in sorted(candidates, key=lambda row: (-row["score"], tuple(row["box_xyxy"]))):
        if all(
            base.iou(candidate["box_xyxy"], kept["box_xyxy"])
            <= float(grouped["maximum_pair_iou"])
            for kept in distinct
        ):
            distinct.append(candidate)
    required = int(grouped["minimum_distinct_rows"])
    selected = distinct[:required]
    geometric_mean = (
        math.prod(float(row["score"]) for row in selected) ** (1.0 / required)
        if len(selected) == required
        else 0.0
    )
    matched = (
        len(selected) == required
        and geometric_mean >= float(grouped["minimum_group_geometric_mean_score"])
    )
    return {
        "matched": matched,
        "tier": "GROUP_CONFIDENT_REPEATED_TWO_INITIAL_BRAND_SIGNATURE" if matched else "NONE",
        "target_tokens": target,
        "witnesses": selected if matched else [],
        "candidate_count_before_spatial_deduplication": len(candidates),
        "distinct_candidate_count": len(distinct),
        "selected_group_geometric_mean_score": round(geometric_mean, 8),
    }


def main() -> None:
    base.require("--protocol" in sys.argv, "PROTOCOL_ARGUMENT_REQUIRED")
    protocol_path = Path(sys.argv[sys.argv.index("--protocol") + 1]).resolve()
    protocol = base.load(protocol_path)
    base.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    base_path = base.verify_file(protocol["frozen_inputs"]["base_ledger_evaluator"])
    base.require(base_path.resolve() == Path(base.__file__).resolve(), "LOADED_BASE_EVALUATOR_MISMATCH")
    base.verify_file(protocol["frozen_inputs"]["base_ledger_result"])
    original_protocol_schema = base.PROTOCOL_SCHEMA
    original_result_schema = base.RESULT_SCHEMA
    original_file = base.__file__
    original_match = base.match_target_name
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.RESULT_SCHEMA = RESULT_SCHEMA
    base.__file__ = str(Path(__file__).resolve())
    base.match_target_name = match_target_name
    try:
        base.main()
    finally:
        base.PROTOCOL_SCHEMA = original_protocol_schema
        base.RESULT_SCHEMA = original_result_schema
        base.__file__ = original_file
        base.match_target_name = original_match


if __name__ == "__main__":
    main()
