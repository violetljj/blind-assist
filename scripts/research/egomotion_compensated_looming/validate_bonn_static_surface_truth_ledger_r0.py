#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SEQUENCES = {
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_balloon",
}
EXPECTED_TERMINAL = "BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(receipt: dict[str, Any], points_path: Path) -> dict[str, Any]:
    require(receipt["status"] == "VALID", "receipt status is not VALID")
    require(receipt["terminal"] == EXPECTED_TERMINAL, "unexpected terminal")
    require(
        receipt["schema_version"] == "bonn_static_surface_truth_ledger_r0",
        "unexpected schema",
    )
    require(
        receipt["input_identity"]["static_map_sample_sha256"]
        == sha256(points_path),
        "static-map sample identity mismatch",
    )
    counts = receipt["counts"]
    require(counts["sequence_count"] == 2, "unexpected sequence count")
    require(counts["anchor_count"] == 40, "unexpected anchor count")
    require(counts["cell_trajectory_count"] == 18, "unexpected cell count")
    require(
        counts["depth_member_read_or_decode_count"] == 6,
        "depth read count changed",
    )
    require(
        counts["rgb_member_read_or_decode_count"] == 0,
        "RGB firewall violated",
    )
    require(
        counts["validation_or_holdout_read_count"] == 0,
        "sealed-role firewall violated",
    )
    require(
        counts["candidate_signal_computed"] is False,
        "candidate signal was computed",
    )
    require(
        counts["eligible_cell_trajectory_count"] == 0,
        "failed transform canary left an eligible cell",
    )
    require(
        counts["c2_static_surface_closing_mechanics_candidate_count"] == 0,
        "failed transform canary left a C2 candidate",
    )

    canary = receipt["transform_canary"]
    require(canary["official_formula_passed"], "official formula check failed")
    require(canary["projection_canary_passed"], "projection canary failed")
    require(canary["usable_depth_frame_count"] == 3, "usable frame count changed")
    require(canary["depth_canary_passed"] is False, "depth quorum unexpectedly passed")
    require(canary["passed"] is False, "transform canary unexpectedly passed")
    require(
        canary["median_of_frame_median_absolute_depth_error_meters"] <= 0.10,
        "usable-frame absolute depth agreement regressed",
    )
    require(
        canary["median_of_frame_median_absolute_relative_depth_error"] <= 0.05,
        "usable-frame relative depth agreement regressed",
    )

    sequences = receipt["sequences"]
    require(
        {sequence["sequence_id"] for sequence in sequences} == EXPECTED_SEQUENCES,
        "sequence identity changed",
    )
    cells = [cell for sequence in sequences for cell in sequence["cells"]]
    require(
        all(
            cell["abstention_reason"] == "TRANSFORM_GEOMETRY_CANARY_FAILED"
            and cell["eligible"] is False
            and cell["c2_static_surface_closing_mechanics_candidate"] is False
            for cell in cells
        ),
        "cell-level fail-closed state is incomplete",
    )
    require(
        any(cell["supported_anchor_count"] >= 18 for cell in cells),
        "diagnostic support was not retained",
    )
    boundary = receipt["claim_boundary"]
    require(
        boundary["static_visible_surface_distance_only"]
        and not boundary["obstacle_semantics_proven"]
        and not boundary["dynamic_target_truth_proven"]
        and not boundary["route_or_event_truth_proven"]
        and not boundary["alert_or_safety_authority"],
        "claim boundary changed",
    )
    return {
        "status": "VALID",
        "terminal": EXPECTED_TERMINAL,
        "sequence_count": counts["sequence_count"],
        "cell_trajectory_count": counts["cell_trajectory_count"],
        "usable_depth_frame_count": canary["usable_depth_frame_count"],
        "candidate_signal_computed": counts["candidate_signal_computed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    print(json.dumps(validate(receipt, args.points), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
