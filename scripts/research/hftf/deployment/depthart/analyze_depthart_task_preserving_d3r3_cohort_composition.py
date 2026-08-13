#!/usr/bin/env python3
"""Test whether D3R3 support is cohort-compositional rather than per-parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


MINIMUM_KNOWN_CELLS_PER_PARENT = 1800
MINIMUM_VALID_CLEARANCES_PER_PARENT = 450
MINIMUM_CLASS_CELLS_PER_ROLE = {"clear": 270, "occupied": 900}
MINIMUM_GRID_CLASS_CELLS_PER_ROLE = 30


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "exclusive write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def geometry_observable(row: dict[str, Any]) -> bool:
    counts = row["truth_support"]
    return (
        row["coverage_evaluable"] is True
        and int(counts["known_cells"]) >= MINIMUM_KNOWN_CELLS_PER_PARENT
        and int(counts["valid_band_clearances"]) >= MINIMUM_VALID_CLEARANCES_PER_PARENT
    )


def role_summary(
    rows: list[dict[str, Any]], selected_indices: list[int], strata: list[tuple[str, str]]
) -> dict[str, Any]:
    selected = [rows[index] for index in selected_indices]
    by_stratum = {
        f"{kind}:{grid}": sum(
            int(row["truth_support"][f"{kind}_by_grid"][grid]) for row in selected
        )
        for kind, grid in strata
    }
    contributor_count = {
        f"{kind}:{grid}": sum(
            int(row["truth_support"][f"{kind}_by_grid"][grid]) > 0 for row in selected
        )
        for kind, grid in strata
    }
    return {
        "identity_count": len(selected),
        "phase_a_selection_orders": [int(row["selection_order"]) for row in selected],
        "identities": [
            {
                "phase_a_selection_order": int(row["selection_order"]),
                "pool_order": int(row["pool_order"]),
                "visit_id": str(row["visit_id"]),
                "video_id": str(row["video_id"]),
                "source_unavailable_frame_count": int(row["source_unavailable_frame_count"]),
            }
            for row in selected
        ],
        "aggregate_clear_cells": sum(int(row["truth_support"]["clear_cells"]) for row in selected),
        "aggregate_occupied_cells": sum(int(row["truth_support"]["occupied_cells"]) for row in selected),
        "by_stratum": by_stratum,
        "minimum_stratum_count": min(by_stratum.values()),
        "contributing_parent_count_by_stratum": contributor_count,
        "minimum_contributing_parent_count": min(contributor_count.values()),
    }


def solve_role_split(
    rows: list[dict[str, Any]], role_size: int = 8
) -> tuple[dict[str, Any], dict[str, Any], int]:
    require(rows, "empty source-truth rows")
    grids = list(rows[0]["truth_support"]["clear_by_grid"])
    strata = [(kind, grid) for kind in ("clear", "occupied") for grid in grids]
    eligible_indices = [index for index, row in enumerate(rows) if geometry_observable(row)]
    require(len(eligible_indices) >= 2 * role_size, "insufficient geometry-observable parents")
    eligible = [rows[index] for index in eligible_indices]
    counts = np.asarray(
        [
            [int(row["truth_support"][f"{kind}_by_grid"][grid]) for row in eligible]
            for kind, grid in strata
        ],
        dtype=float,
    )
    total_class = {
        kind: np.asarray([int(row["truth_support"][f"{kind}_cells"]) for row in eligible], dtype=float)
        for kind in ("clear", "occupied")
    }
    n = len(eligible)
    # Variables are TRAIN[0:n], DEVELOPMENT[n:2n], and shared max-min t.
    objective = np.r_[np.zeros(2 * n), -1.0]
    rows_a: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for role in range(2):
        vector = np.zeros(2 * n + 1)
        vector[role * n : (role + 1) * n] = 1
        rows_a.append(vector)
        lower.append(role_size)
        upper.append(role_size)
    for index in range(n):
        vector = np.zeros(2 * n + 1)
        vector[index] = 1
        vector[n + index] = 1
        rows_a.append(vector)
        lower.append(0)
        upper.append(1)
    for role in range(2):
        for stratum in range(len(strata)):
            vector = np.zeros(2 * n + 1)
            vector[role * n : (role + 1) * n] = counts[stratum]
            vector[-1] = -1
            rows_a.append(vector)
            lower.append(0)
            upper.append(np.inf)
            threshold = np.zeros(2 * n + 1)
            threshold[role * n : (role + 1) * n] = counts[stratum]
            rows_a.append(threshold)
            lower.append(MINIMUM_GRID_CLASS_CELLS_PER_ROLE)
            upper.append(np.inf)
        for kind in ("clear", "occupied"):
            vector = np.zeros(2 * n + 1)
            vector[role * n : (role + 1) * n] = total_class[kind]
            rows_a.append(vector)
            lower.append(MINIMUM_CLASS_CELLS_PER_ROLE[kind])
            upper.append(np.inf)

    result = milp(
        objective,
        integrality=np.r_[np.ones(2 * n), 0],
        bounds=Bounds(np.r_[np.zeros(2 * n), 0], np.r_[np.ones(2 * n), np.inf]),
        constraints=LinearConstraint(np.asarray(rows_a), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30},
    )
    require(result.success and result.x is not None, f"role-composition MILP failed: {result.message}")
    optimum = int(round(float(result.x[-1])))
    train_local = list(np.flatnonzero(result.x[:n] > 0.5))
    development_local = list(np.flatnonzero(result.x[n : 2 * n] > 0.5))
    train_indices = [eligible_indices[index] for index in train_local]
    development_indices = [eligible_indices[index] for index in development_local]
    require(set(train_indices).isdisjoint(development_indices), "role overlap")
    return (
        role_summary(rows, train_indices, strata),
        role_summary(rows, development_indices, strata),
        optimum,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-truth-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_json(args.source_truth_result)
    require(source.get("status") == "D3R3_PHASE_B_SOURCE_TRUTH_SUPPORT_NOT_EVALUABLE", "source-truth status drift")
    rows = source.get("processed")
    require(isinstance(rows, list) and len(rows) == 32, "source-truth identity count drift")
    train, development, optimum = solve_role_split(rows)
    all_strata_above_floor = all(
        role["minimum_stratum_count"] >= MINIMUM_GRID_CLASS_CELLS_PER_ROLE
        for role in (train, development)
    )
    contributor_diversity = min(
        train["minimum_contributing_parent_count"],
        development["minimum_contributing_parent_count"],
    )
    result = {
        "schema": "blindassist_depthart_task_preserving_d3r3_cohort_compositional_support_canary_v1",
        "status": "D3R3_COHORT_COMPOSITION_MECHANISM_SUPPORTED_PARENT_DIVERSITY_INSUFFICIENT",
        "problem": "The old gate requires every parent to contain both classes in every grid, but a routed model learns from role cohorts.",
        "hypothesis": "Moving class-balance requirements to parent-disjoint TRAIN and DEVELOPMENT cohorts while retaining per-parent geometry observability reveals usable complementary support.",
        "baseline_per_identity_qualified_count": int(source["primary_qualified_identity_count"]),
        "geometry_observable_parent_count": sum(geometry_observable(row) for row in rows),
        "per_parent_requirements_retained": {
            "minimum_known_cells": MINIMUM_KNOWN_CELLS_PER_PARENT,
            "minimum_valid_band_clearances": MINIMUM_VALID_CLEARANCES_PER_PARENT,
            "minimum_source_available_frames": source["primary_policy"]["minimum_source_available_frames"],
        },
        "role_aggregate_requirements": {
            "identities_per_role": 8,
            "minimum_clear_cells": MINIMUM_CLASS_CELLS_PER_ROLE["clear"],
            "minimum_occupied_cells": MINIMUM_CLASS_CELLS_PER_ROLE["occupied"],
            "minimum_each_clear_and_occupied_grid": MINIMUM_GRID_CLASS_CELLS_PER_ROLE,
        },
        "max_min_stratum_count": optimum,
        "all_role_strata_above_30": all_strata_above_floor,
        "TRAIN": train,
        "DEVELOPMENT": development,
        "minimum_parent_contributors_per_role_stratum": contributor_diversity,
        "selection_lock": False,
        "reason_not_locked": "At least one rare CLEAR stratum in each role depends on a single parent; expand the already continuity-eligible pool before role lock.",
        "old_zero_of_32_result_preserved": True,
        "model_or_rgb_read": False,
        "r2_access": "NONE",
        "next_action": "EXPAND_DEPTH_CONFIDENCE_SOURCE_TRUTH_TO_REMAINING_21_PHASE_A_CONTINUITY_ELIGIBLE_IDENTITIES",
    }
    write_json_exclusive(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "geometry_observable": result["geometry_observable_parent_count"],
        "max_min": optimum,
        "train_min": train["minimum_stratum_count"],
        "development_min": development["minimum_stratum_count"],
        "minimum_parent_contributors": contributor_diversity,
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
