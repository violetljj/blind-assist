#!/usr/bin/env python3
"""Canary for D3R4 risk-asymmetric selective-horizon certificates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from scripts.research.hftf.deployment.depthart.analyze_depthart_task_preserving_d3r3_cohort_composition import (
    geometry_observable,
)


MINIMUM_STRATUM_COUNT = 30
MINIMUM_PARENT_CONTRIBUTORS = 2


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


def active_strata(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    grids = list(rows[0]["truth_support"]["clear_by_grid"])
    return (
        [("clear", grid) for grid in grids if grid.endswith("@1.0m")]
        + [("occupied", grid) for grid in grids]
    )


def summarize_role(
    rows: list[dict[str, Any]], selected: list[int], strata: list[tuple[str, str]]
) -> dict[str, Any]:
    chosen = [rows[index] for index in selected]
    counts = {
        f"{kind}:{grid}": sum(
            int(row["truth_support"][f"{kind}_by_grid"][grid]) for row in chosen
        )
        for kind, grid in strata
    }
    parents = {
        f"{kind}:{grid}": sum(
            int(row["truth_support"][f"{kind}_by_grid"][grid]) > 0 for row in chosen
        )
        for kind, grid in strata
    }
    return {
        "identity_count": len(chosen),
        "identities": [
            {
                "role_order": order,
                "phase_a_eligible_order": int(row["selection_order"]),
                "pool_order": int(row["pool_order"]),
                "visit_id": str(row["visit_id"]),
                "video_id": str(row["video_id"]),
                "source_unavailable_frame_count": int(row["source_unavailable_frame_count"]),
            }
            for order, row in enumerate(chosen, start=1)
        ],
        "counts_by_active_stratum": counts,
        "parent_contributors_by_active_stratum": parents,
        "minimum_active_stratum_count": min(counts.values()),
        "minimum_parent_contributors": min(parents.values()),
    }


def solve_selective_roles(
    rows: list[dict[str, Any]], role_size: int = 8
) -> tuple[dict[str, Any], dict[str, Any], int]:
    strata = active_strata(rows)
    eligible_indices = [index for index, row in enumerate(rows) if geometry_observable(row)]
    require(len(eligible_indices) >= 2 * role_size, "insufficient geometry-observable parents")
    eligible = [rows[index] for index in eligible_indices]
    counts = np.asarray(
        [[int(row["truth_support"][f"{kind}_by_grid"][grid]) for row in eligible] for kind, grid in strata],
        dtype=float,
    )
    contributors = (counts > 0).astype(float)
    n = len(eligible)
    objective = np.r_[np.zeros(2 * n), -1.0]
    matrix: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for role in range(2):
        vector = np.zeros(2 * n + 1)
        vector[role * n : (role + 1) * n] = 1
        matrix.append(vector)
        lower.append(role_size)
        upper.append(role_size)
    for index in range(n):
        vector = np.zeros(2 * n + 1)
        vector[index] = vector[n + index] = 1
        matrix.append(vector)
        lower.append(0)
        upper.append(1)
    for role in range(2):
        for stratum in range(len(strata)):
            vector = np.zeros(2 * n + 1)
            vector[role * n : (role + 1) * n] = counts[stratum]
            vector[-1] = -1
            matrix.append(vector)
            lower.append(0)
            upper.append(np.inf)
            count_floor = np.zeros(2 * n + 1)
            count_floor[role * n : (role + 1) * n] = counts[stratum]
            matrix.append(count_floor)
            lower.append(MINIMUM_STRATUM_COUNT)
            upper.append(np.inf)
            parent_floor = np.zeros(2 * n + 1)
            parent_floor[role * n : (role + 1) * n] = contributors[stratum]
            matrix.append(parent_floor)
            lower.append(MINIMUM_PARENT_CONTRIBUTORS)
            upper.append(np.inf)
    result = milp(
        objective,
        integrality=np.r_[np.ones(2 * n), 0],
        bounds=Bounds(np.r_[np.zeros(2 * n), 0], np.r_[np.ones(2 * n), np.inf]),
        constraints=LinearConstraint(np.asarray(matrix), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30},
    )
    require(result.success and result.x is not None, f"selective-horizon MILP failed: {result.message}")
    train = [eligible_indices[index] for index in np.flatnonzero(result.x[:n] > 0.5)]
    development = [eligible_indices[index] for index in np.flatnonzero(result.x[n : 2 * n] > 0.5)]
    require(set(train).isdisjoint(development), "TRAIN/DEVELOPMENT overlap")
    return (
        summarize_role(rows, train, strata),
        summarize_role(rows, development, strata),
        int(round(float(result.x[-1]))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source-truth-result", type=Path, required=True)
    parser.add_argument("--extension-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load_json(args.base_source_truth_result)
    extension = load_json(args.extension_result)
    rows = list(base["processed"]) + list(extension["processed_extension"])
    require(len(rows) == 53, "combined identity count drift")
    train, development, optimum = solve_selective_roles(rows)
    result = {
        "schema": "blindassist_depthart_d3r4_selective_horizon_certificate_canary_v1",
        "status": "D3R4_SELECTIVE_HORIZON_SOURCE_SUPPORT_PASS",
        "problem": "Far-horizon CLEAR is too rare and parent-concentrated for honest bidirectional learning, while near CLEAR and all-horizon OCCUPIED are abundant.",
        "hypothesis": "A risk-asymmetric router can learn CLEAR release only at 1.0m and OCCUPIED veto at 1.0/1.5/2.0m; unsupported far CLEAR must abstain to baseline/UNKNOWN.",
        "inherited": "D3 bidirectional release/veto idea, D3R3 fixed source truth, geometry observability, and UNKNOWN semantics.",
        "new_contribution": "Horizon-conditional action space derived from parent-diverse observability, not a backbone change or post-hoc negative relabeling.",
        "combined_identity_count": 53,
        "geometry_observable_identity_count": sum(geometry_observable(row) for row in rows),
        "release_enabled_horizons_m": [1.0],
        "release_abstain_horizons_m": [1.5, 2.0],
        "veto_enabled_horizons_m": [1.0, 1.5, 2.0],
        "unsupported_far_clear_is_negative": False,
        "unsupported_far_clear_action": "ABSTAIN_TO_BASELINE_OR_UNKNOWN",
        "role_requirements": {
            "identities_per_role": 8,
            "minimum_active_stratum_count": MINIMUM_STRATUM_COUNT,
            "minimum_parent_contributors_per_active_stratum": MINIMUM_PARENT_CONTRIBUTORS,
        },
        "max_min_active_stratum_count": optimum,
        "candidate_role_split": {"TRAIN": train, "DEVELOPMENT": development},
        "candidate_role_lock": True,
        "old_d3r3_zero_of_32_terminal_preserved": True,
        "rgb_or_model_read": False,
        "r2_access": "NONE",
        "next_action": "ACQUIRE_RGB_FOR_EXACT_D3R4_8_TRAIN_8_DEVELOPMENT_CANDIDATE_ROSTER",
    }
    write_json_exclusive(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "geometry_observable": result["geometry_observable_identity_count"],
        "max_min": optimum,
        "train_min": train["minimum_active_stratum_count"],
        "train_min_parents": train["minimum_parent_contributors"],
        "development_min": development["minimum_active_stratum_count"],
        "development_min_parents": development["minimum_parent_contributors"],
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
