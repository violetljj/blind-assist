"""Freeze and predict X32 observation-conditioned representative cores.

X32 keeps X31's finite current-shift transport states, detector inputs,
association radius, motion authority, route geometry, and decision thresholds.
It removes two representation artifacts exposed by the C16 Development replay:

* histories that reach the same current shift share the strongest authorized
  motion state instead of unioning mutually exclusive historical geometry;
* measured support uses the convex hull of occupied lattice-cell squares,
  while HOLD support retains X31's conservative axis-aligned envelope.

The predictor never opens evaluator truth.  Results on a source used to design
X32 remain same-source synthetic Development, not confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x29_temporal_occupancy_lineage_predictor as x29  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X32_OBSERVATION_CONDITIONED_REPRESENTATIVE_CORE"
FREEZE_SCHEMA = "blindassist-dtr-carla-x32-observation-conditioned-core-freeze-v1"
PREDICTION_SCHEMA = (
    "blindassist-dtr-carla-x32-observation-conditioned-core-predictions-v1"
)
ARM_X32 = "X32_ISSUED_PLAN_OBSERVATION_CONDITIONED_REPRESENTATIVE_CORE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x31.fixed_constants(),
        "representation": "OBSERVATION_CONDITIONED_REPRESENTATIVE_SURFACE_CORE",
        "current_shift_state_rule": (
            "STRONGEST_AUTHORIZED_MOTION_STATE_PER_DISTINCT_CURRENT_SHIFT"
        ),
        "current_shift_geometry_rule": "STRONGEST_EVIDENCE_REPRESENTATIVE_LINEAGE",
        "measured_support_footprint": "CONVEX_HULL_OF_OCCUPIED_LATTICE_CELL_SQUARES",
        "hold_support_footprint": "INHERITED_AXIS_ALIGNED_LINEAGE_ENVELOPE",
        "detector_threshold_change": False,
        "association_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def paths(run_root: Path) -> dict[str, Path]:
    return {
        "x24_freeze": run_root / "freeze-x24.json",
        "x24_predictions": run_root / "predictions-x24.json",
        "freeze": run_root / "freeze-x32.json",
        "predictions": run_root / "predictions-x32.json",
    }


def convex_cell_hull(cells: frozenset[x27.Cell]) -> np.ndarray:
    """Convex envelope of complete lattice cells, not only their centres."""

    x24.require(len(cells) >= x27.MINIMUM_ALIGNMENT_CELLS, "x32_lineage_support")
    cell = x27.LATTICE_CELL_SIZE_M
    half = cell * 0.5
    points = sorted(
        {
            (x * cell + dx, y * cell + dy)
            for x, y in cells
            for dx in (-half, half)
            for dy in (-half, half)
        }
    )

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    x24.require(len(hull) >= 4, "x32_degenerate_cell_hull")
    return np.asarray(hull, dtype=np.float64)


def representative_shift_cores(
    branches: Sequence[x31.TransportBranch], now_s: float
) -> list[x31.TransportBranch]:
    """Collapse ancestry permutations without union-inflating their geometry."""

    groups: dict[x27.Cell, list[x31.TransportBranch]] = {}
    for branch in branches:
        groups.setdefault(branch.last_shift_cells, []).append(branch)
    output: list[x31.TransportBranch] = []
    for shift, group in groups.items():
        authorized = [
            branch
            for branch in group
            if x31.resolve_branch_authority(branch, now_s)[0] == x27.RIGID_DYNAMIC
        ]
        representative = max(
            authorized or group,
            key=lambda branch: x31._branch_priority(branch, now_s),
        )
        output.append(
            x31.TransportBranch(
                evidence=list(representative.evidence),
                anchor_times_s=sorted(
                    {
                        value
                        for branch in group
                        for value in x31._trimmed_anchor_times(
                            branch.anchor_times_s, now_s
                        )
                    }
                ),
                world_lineage=list(representative.world_lineage),
                last_shift_cells=shift,
                last_state=representative.last_state,
            )
        )
    return output


class ObservationConditionedCoreTracker(x31.AmbiguityPreservingSurfaceTracker):
    def _row(self, **kwargs: Any) -> dict[str, Any]:
        row = super()._row(**kwargs)
        parent_track_id = str(kwargs["parent_track_id"])
        measured_ids = kwargs["measured_ids"]
        if parent_track_id not in measured_ids:
            return row
        lineage = kwargs["lineage"]
        support_cells, maximum_support = x29.repeated_support(lineage)
        if len(support_cells) < x27.MINIMUM_ALIGNMENT_CELLS:
            return row
        footprint = convex_cell_hull(support_cells)
        center = np.mean(footprint, axis=0)
        row.update(
            {
                "position_forward_m": float(center[0]),
                "position_right_m": float(center[1]),
                "footprint_xy": [
                    [float(value) for value in point] for point in footprint
                ],
                "footprint_area_m2": x25.polygon_area(footprint),
                "lineage_cells": len(support_cells),
                "lineage_observations": len(lineage),
                "maximum_cell_observations": maximum_support,
                "support_footprint_mode": "MEASURED_CONVEX_CELL_HULL",
            }
        )
        return row


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    original_tracker = x31.AmbiguityPreservingSurfaceTracker
    original_coalescer = x31._coalesce_current_shift_envelopes
    x31.AmbiguityPreservingSurfaceTracker = ObservationConditionedCoreTracker
    x31._coalesce_current_shift_envelopes = representative_shift_cores
    try:
        value = x31.predict_episode(episode, candidate_values, calibration)
    finally:
        x31.AmbiguityPreservingSurfaceTracker = original_tracker
        x31._coalesce_current_shift_envelopes = original_coalescer
    value["arms"][ARM_X32] = value["arms"].pop(x31.ARM_X31)
    value["diagnostics"]["x32_route_mode_counts"] = value["diagnostics"].pop(
        "x31_route_mode_counts"
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X32] = frame["arms"].pop(x31.ARM_X31)
        for track in frame["tracks"]:
            track.setdefault(
                "support_footprint_mode", "HOLD_AXIS_ALIGNED_LINEAGE_ENVELOPE"
            )
    return value


def _algorithm_files() -> dict[str, Path]:
    return {
        **x31._algorithm_files(),
        "x31_base_predictor": Path(x31.__file__).resolve(),
        "x32_predictor": Path(__file__).resolve(),
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["freeze"].exists(), f"x32_freeze_exists:{output['freeze']}")
    frozen_x24, contract, _candidates = x24.require_freeze(run_root)
    x24.require(output["x24_predictions"].is_file(), "x32_x24_baseline_missing")
    value = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN_TRUTH_BLIND_PENDING_PREDICTION",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "source": {
            "x24_freeze_sha256": x24.sha256_file(output["x24_freeze"]),
            "x24_predictions_sha256": x24.sha256_file(output["x24_predictions"]),
            "model_manifest_sha256": contract.manifest_sha256,
            "candidate_aggregate_sha256": frozen_x24["candidates"][
                "aggregate_sha256"
            ],
        },
        "algorithm_files": {
            name: {"path": str(path), "sha256": x24.sha256_file(path)}
            for name, path in _algorithm_files().items()
        },
        "episodes": len(contract.episodes),
        "frames": len(x24.flatten_observations(contract)),
        "fixed_constants": fixed_constants(),
        "arm": ARM_X32,
        "claim_boundary": {
            "same_source_use_requires_development_only_label": True,
            "source_disjoint_confirmation_claimed": False,
        },
    }
    x24.write_json_exclusive(output["freeze"], value)
    return {**value, "freeze_sha256": x24.sha256_file(output["freeze"])}


def require_freeze(
    run_root: Path,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    output = paths(run_root)
    frozen = x24.read_json(output["freeze"].resolve(strict=True), "x32_freeze")
    x24.require(frozen.get("schema") == FREEZE_SCHEMA, "x32_freeze_schema")
    x24.require(frozen.get("fixed_constants") == fixed_constants(), "x32_constants_drift")
    for name, path in _algorithm_files().items():
        x24.require(
            frozen["algorithm_files"][name]["sha256"] == x24.sha256_file(path),
            f"x32_algorithm_drift:{name}",
        )
    frozen_x24, contract, candidates = x24.require_freeze(run_root)
    x24.require(
        frozen["source"]["x24_freeze_sha256"]
        == x24.sha256_file(output["x24_freeze"]),
        "x32_x24_freeze_drift",
    )
    x24.require(
        frozen["source"]["x24_predictions_sha256"]
        == x24.sha256_file(output["x24_predictions"]),
        "x32_x24_prediction_drift",
    )
    x24.require(
        frozen["source"]["candidate_aggregate_sha256"]
        == frozen_x24["candidates"]["aggregate_sha256"],
        "x32_candidate_drift",
    )
    return frozen, contract, candidates


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(
        not output["predictions"].exists(),
        f"x32_predictions_exist:{output['predictions']}",
    )
    frozen, contract, candidate_values = require_freeze(run_root)
    cursor = 0
    episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        episodes[episode.episode_id] = predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        cursor += count
    value = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED_TRUTH_BLIND_PENDING_DEVELOPMENT_SCORE",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "arms": [ARM_X32],
        "episodes": episodes,
        "fixed_constants": fixed_constants(),
        "source": {
            "freeze_sha256": x24.sha256_file(output["freeze"]),
            **frozen["source"],
        },
        "claim_boundary": {
            "synthetic_development": True,
            "evaluator_opened_by_predictor": False,
            "current_actor_oracle_used": False,
            "source_disjoint_confirmation_claimed": False,
        },
    }
    x24.write_json_exclusive(output["predictions"], value)
    return {
        **value,
        "predictions_sha256": x24.sha256_file(output["predictions"]),
    }


def self_check() -> dict[str, Any]:
    inherited = x31.self_check()
    cells = frozenset({(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 1)})
    hull = convex_cell_hull(cells)
    aabb = x29.lineage_footprint(cells)
    x24.require(
        x25.polygon_area(hull) <= x25.polygon_area(aabb) + x31.EPSILON,
        "x32_convex_hull_not_tighter_than_aabb",
    )
    candidate = x31._synthetic_candidate((-1, 0), anchored=True, center=True)
    first = x31._new_branch(
        candidate,
        now_s=0.1,
        delta_s=0.1,
        previous_cells=cells,
        current_cells=x27.shifted(cells, (-1, 0)),
    )
    second = x31._new_branch(
        candidate,
        now_s=0.2,
        delta_s=0.1,
        previous_cells=x27.shifted(cells, (0, 2)),
        current_cells=x27.shifted(cells, (-1, 2)),
    )
    cores = representative_shift_cores([first, second], now_s=0.2)
    x24.require(len(cores) == 1, "x32_shift_core_not_finite")
    return {
        "status": "X32_OBSERVATION_CONDITIONED_CORE_STRUCTURAL_FALSIFIER_MET",
        "x31_structural_status": inherited["status"],
        "convex_cell_hull_preserves_tighter_supported_geometry": True,
        "one_representative_core_per_current_shift": True,
        "measured_and_hold_geometry_modes_distinct": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "predict"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, required=True)
    subparsers.add_parser("self-check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        value = freeze(args)
    elif args.command == "predict":
        value = predict(args)
    else:
        value = self_check()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
