#!/usr/bin/env python3
"""Validate the fail-closed terminal implied by ADT geometry proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_CELLS = (
    "PURE_EGO_ROTATION_NO_CLOSING",
    "EGO_APPROACH_STATIC_SURFACE",
    "STATIONARY_EGO_ACTIVE_TARGET_APPROACH",
    "LATERAL_PASS_NO_SUSTAINED_CLOSING",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.proposals.read_text(encoding="utf-8"))
    assert (
        receipt["terminal"]
        == "ADT_GEOMETRY_CELL_PROPOSALS_READY_FOR_INDEPENDENT_REVIEW"
    )
    assert receipt["cohort_role"] == "SOURCE_PRESCREEN_ONLY"
    assert receipt["sequence_count"] == 16
    assert receipt["component_count"] == 16
    assert receipt["candidate_signal_read_count"] == 0
    assert receipt["old_window_or_outcome_read_count"] == 0
    assert receipt["rgb_or_vrs_read_count"] == 0
    assert receipt["role_split_frozen"] is False
    assert receipt["cell_review_complete"] is False
    assert (
        receipt["visibility_contract"]["bbox_coordinate_field_access_count"] == 0
    )

    recomputed = {cell: 0 for cell in REQUIRED_CELLS}
    for sequence in receipt["sequence_results"]:
        seen = set()
        for proposal in sequence["proposals"]:
            cell = proposal["cell"]
            assert cell in REQUIRED_CELLS
            assert cell not in seen
            seen.add(cell)
            recomputed[cell] += 1
    reported = {
        cell: receipt["accepted_eligible_object_proposal_counts"].get(cell, 0)
        for cell in REQUIRED_CELLS
    }
    assert recomputed == reported
    assert receipt["skeleton_diagnostic_proposal_coverage"] == "NOT_IMPLEMENTED"
    insufficient = [cell for cell, count in recomputed.items() if count < 2]
    assert insufficient
    terminal = "ADT_CELL_PRESCREEN_INSUFFICIENT"
    print(
        json.dumps(
            {
                "status": "VALID",
                "terminal": terminal,
                "proposal_sha256": sha256(args.proposals),
                "accepted_eligible_object_proposal_counts": recomputed,
                "insufficient_cells": insufficient,
                "skeleton_diagnostic_proposal_coverage": "NOT_IMPLEMENTED",
                "source_admission": "HOLD_R0_ADMISSION",
                "may_decode_rgb_or_run_signal": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
